"""Transient integrator for the HL Fokker-Planck equation.

Time discretisation is IMEX (implicit-explicit), chosen to be both *stable* and
*exactly mass-conserving*:

* **Implicit** part: diffusion D*L2 and advection A.  Diffusion is stiff (an
  explicit step would need dt ~ h^2 / D); advection treated implicitly is
  unconditionally stable.  Both have zero column sums.
* **Explicit** part: the yielding loss and the reinjection source, evaluated at
  the *same* (old) time level.  Individually they move mass, but together their
  column sums cancel, so the explicit half conserves mass exactly.

Both halves conserve mass, so total probability is preserved to machine
precision every step.  The only nonlinearity, D = alpha * Gamma(P), is lagged;
an optional Picard sub-iteration tightens it near criticality.  ``theta`` selects
backward-Euler (1, robust) or Crank-Nicolson (1/2, second order).

The implicit operator I - theta*dt*(D*L2 + A) is tridiagonal and is assembled
and solved in banded form with fully vectorised numpy -- no per-step sparse
assembly or Python loops, so each step is a fast O(n) banded solve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.linalg import solve_banded

from .grid import Grid


@dataclass
class Trajectory:
    """Recorded observables along a transient run."""
    t: np.ndarray
    stress: np.ndarray
    Gamma: np.ndarray
    D: np.ndarray
    P_final: np.ndarray = field(repr=False)
    grid: Grid = field(repr=False)


class TransientStepper:
    """Banded IMEX propagator for fixed alpha and grid.

    Precomputes the constant reflecting-Laplacian bands and the yielding mask so
    that repeated stepping (e.g. the LAOS period map) is cheap.
    """

    def __init__(self, alpha: float, grid: Grid, parts: dict | None = None,
                 theta: float = 1.0, picard_iters: int = 0):
        self.alpha = float(alpha)
        self.grid = grid
        self.theta = float(theta)
        self.picard_iters = int(picard_iters)

        n, h = grid.n, grid.h
        self.n, self.h = n, h
        self.i_zero = grid.i_zero
        self.mask = grid.yield_mask.astype(float)

        # Reflecting-Laplacian bands (row i: lower*P[i-1] + main*P[i] + upper*P[i+1]).
        inv = 1.0 / (h * h)
        self.l2_main = np.full(n, -2.0 * inv)
        self.l2_main[0] = -inv
        self.l2_main[-1] = -inv
        self.l2_upper = np.full(n, inv)   # coeff of P[i+1] in row i (i=0..n-2 used)
        self.l2_lower = np.full(n, inv)   # coeff of P[i-1] in row i (i=1..n-1 used)

    # -- helpers -------------------------------------------------------------
    def _D_of(self, P: np.ndarray) -> float:
        # D = alpha * Gamma is a (nonnegative) rate.  Clamp at zero so that
        # intermediate unphysical iterates explored by Newton-Krylov -- which can
        # make the raw yield integral negative -- cannot turn the implicit
        # diffusion into anti-diffusion and render the banded solve singular.
        # At the true fixed point P >= 0, so the clamp is inactive.
        return self.alpha * self.h * max(0.0, float(P[self.grid.yield_mask].sum()))

    def _advection_bands(self, v: float):
        """Vectorised conservative-upwind advection bands for rate v."""
        n, h = self.n, self.h
        a_main = np.zeros(n)
        a_upper = np.zeros(n)
        a_lower = np.zeros(n)
        if v > 0.0:
            a_main[:-1] += -v / h        # outflow through right face (no term at n-1)
            a_lower[1:] += v / h         # inflow from left face (no term at 0)
        elif v < 0.0:
            a_main[1:] += v / h          # outflow through left face (no term at 0)
            a_upper[:-1] += -v / h       # inflow from right face (no term at n-1)
        return a_main, a_upper, a_lower

    def _react(self, P: np.ndarray) -> np.ndarray:
        """(Yield + Source) @ P, vectorised."""
        r = -self.mask * P
        r[self.i_zero] += self.h * float(P[self.grid.yield_mask].sum()) / self.h
        return r

    def step(self, P: np.ndarray, gammadot_next: float, dt: float,
             D_old: float | None = None) -> tuple[np.ndarray, float]:
        """Advance P by dt with strain rate ``gammadot_next`` at the new level."""
        if D_old is None:
            D_old = self._D_of(P)
        th, n = self.theta, self.n
        a_main, a_upper, a_lower = self._advection_bands(gammadot_next)

        # explicit half (same time level) -- conserves mass exactly
        rhs_expl = P + dt * self._react(P)

        def implicit_solve(D_val):
            lin_main = D_val * self.l2_main + a_main
            lin_upper = D_val * self.l2_upper + a_upper
            lin_lower = D_val * self.l2_lower + a_lower
            rhs = rhs_expl.copy()
            if th != 1.0:
                linP = lin_main * P
                linP[1:] += lin_lower[1:] * P[:-1]
                linP[:-1] += lin_upper[:-1] * P[1:]
                rhs += (1.0 - th) * dt * linP
            ab = np.zeros((3, n))
            ab[0, 1:] = -th * dt * lin_upper[:-1]    # super-diagonal
            ab[1, :] = 1.0 - th * dt * lin_main      # main diagonal
            ab[2, :-1] = -th * dt * lin_lower[1:]    # sub-diagonal
            return solve_banded((1, 1), ab, rhs)

        P_new = implicit_solve(D_old)
        D_used = D_old
        for _ in range(self.picard_iters):
            D_new = self._D_of(P_new)
            P_new = implicit_solve(D_new)
            converged = abs(D_new - D_used) <= 1e-12 * (1.0 + abs(D_used))
            D_used = D_new
            if converged:
                break
        return P_new, D_used

    # -- drivers -------------------------------------------------------------
    def run(self, P0: np.ndarray, gammadot_func: Callable[[float], float],
            t_end: float, dt: float, t0: float = 0.0,
            record_every: int = 1) -> Trajectory:
        n_steps = int(round((t_end - t0) / dt))
        P = np.array(P0, dtype=float)
        ts, ss, gs, ds = [], [], [], []
        D = self._D_of(P)
        sigma = self.grid.sigma
        for k in range(n_steps + 1):
            t = t0 + k * dt
            if k % record_every == 0:
                ts.append(t)
                ss.append(self.h * float(sigma @ P))
                gs.append(self.h * float(P[self.grid.yield_mask].sum()))
                ds.append(D)
            if k == n_steps:
                break
            P, D = self.step(P, float(gammadot_func(t + dt)), dt, D_old=D)
        return Trajectory(np.array(ts), np.array(ss), np.array(gs), np.array(ds),
                          P, self.grid)

    def propagate(self, P0: np.ndarray, gammadot_func: Callable[[float], float],
                  t_end: float, dt: float, t0: float = 0.0) -> np.ndarray:
        n_steps = int(round((t_end - t0) / dt))
        P = np.array(P0, dtype=float)
        D = self._D_of(P)
        for k in range(n_steps):
            t = t0 + k * dt
            P, D = self.step(P, float(gammadot_func(t + dt)), dt, D_old=D)
        return P


def initial_delta(grid: Grid) -> np.ndarray:
    """All probability at sigma = 0 (a natural relaxed/quenched start)."""
    P = np.zeros(grid.n)
    P[grid.i_zero] = 1.0 / grid.h
    return P
