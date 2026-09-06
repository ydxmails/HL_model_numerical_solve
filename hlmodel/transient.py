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

Advection scheme
----------------
``advection="tvd"`` (default) advects with a conservative **van Leer
flux-limited** (second-order TVD) scheme.  ``advection="upwind"`` restores the
original first-order upwind operator, kept for comparison.

The limited flux is nonlinear in P, so it cannot go into the banded implicit
operator directly.  Rather than operator-split around it, we **defect correct**:
the monotone upwind operator stays implicit and the difference between the two
flux divergences rides along in the explicit half.  The point is the fixed
point -- the upwind contributions cancel there, leaving exactly

    D L2 P + A_tvd(P) + Yield P + Source P = 0,

which is precisely the stationary equation ``steady.solve_steady`` solves.  So
the integrator relaxes to the direct solver's answer to round-off *at any dt*,
preserving the two-independent-methods cross-check that validates the package.
Strang splitting would agree only to O(dt).  The defect is a difference of two
advection discretisations and is treated explicitly, so it carries a CFL-type
limit |gammadot| dt / h = O(1); the drive is slow in the regimes where the
limiter matters, so this is not binding there.

The strain rate is sampled at the step **midpoint**, gammadot(t + dt/2).  The
right-endpoint value would make the accumulated strain a right-rectangle sum,
gamma_num(t) = gamma(t + dt/2) + O(dt^2) -- a half-step phase lead that shows up
as a spurious loss modulus G'' = |G*| sin(pi/n_steps), independent of amplitude
and frequency.  At the common steps_per_period=400 that is 7.9e-3, which swamps
the true G'' at small oscillation amplitude in the jammed phase (where the exact
answer is zero).  The midpoint rule removes it: measured 139-557x smaller.

The distinction matters because first-order upwind carries a *numerical*
diffusivity

    D_num ~ |gammadot| h / 2 ,

which is indistinguishable from the model's own self-generated D = alpha*Gamma.
Under steady shear this is harmless (the physical D ~ gammadot/2 exceeds it by
~1/h), but in small-amplitude LAOS the physical activity vanishes much faster
than the numerical floor, so the upwind scheme manufactures spurious yielding:
at alpha=0.4, omega=0.2, n_per_unit=200 the cycle-mean D_bar was ~90-96%
numerical for gamma0 <~ 0.05, and G'' ~40-55% numerical for gamma0 <~ 0.12.
That also destroys HL's absorbing state (D = 0 is an exact fixed point of the
continuum dynamics but not of the upwind discretisation).

The TVD scheme is second order in smooth regions and reduces to upwind only near
extrema, so it is positivity preserving (P is a probability density) and
conserves mass exactly (flux form, zero flux at both walls).

Only the *defect* between the upwind and limited fluxes is explicit, not the full
advective flux, so the practical stability limit is much weaker than the naive
Courant condition |gammadot| dt / h <= 1.  Measured at alpha = 0.8, gamma0 = 3,
omega = 0.3: G1' converges monotonically at clean first order (successive
differences 7.6e-4, 3.8e-4, 1.9e-4, 1.0e-4) across Courant numbers 4.7 down to
0.29, with no instability and no break in the sequence.  Enforcing a Courant cap
of 0.9 would demand ~2100 steps per period instead of 400 at h = 0.01 -- a 5x
cost regression for no accuracy gain -- so no subcycling is performed.
``cfl_max`` is retained as an advisory scale only; ``PeriodMap.peak_cfl`` reports
the achieved value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.linalg import solve_banded

from .grid import Grid
from .operators import advection_tvd_rhs


@dataclass
class Trajectory:
    """Recorded observables along a transient run."""
    t: np.ndarray
    stress: np.ndarray
    Gamma: np.ndarray
    D: np.ndarray
    P_final: np.ndarray = field(repr=False)
    grid: Grid = field(repr=False)
    # Full distributions at the recorded times, shape (n_recorded, n); only
    # populated when run(..., record_states=True) (e.g. for cycle-averaged or
    # phase-resolved yield-stress distributions).
    P_states: np.ndarray | None = field(default=None, repr=False)


class TransientStepper:
    """Banded IMEX propagator for fixed alpha and grid.

    Precomputes the constant reflecting-Laplacian bands and the yielding mask so
    that repeated stepping (e.g. the LAOS period map) is cheap.
    """

    def __init__(self, alpha: float, grid: Grid, parts: dict | None = None,
                 theta: float = 1.0, picard_iters: int = 0,
                 advection: str = "tvd", cfl_max: float = 0.9):
        self.alpha = float(alpha)
        self.grid = grid
        self.theta = float(theta)
        self.picard_iters = int(picard_iters)
        if advection not in ("tvd", "upwind"):
            raise ValueError("advection must be 'tvd' or 'upwind'")
        self.advection = advection
        self.cfl_max = float(cfl_max)

        n, h = grid.n, grid.h
        self.n, self.h = n, h
        self.i_zero = grid.i_zero
        self.mask = grid.yield_weight          # half weight exactly at |sigma| = 1

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
        return self.alpha * self.h * max(0.0, float(P @ self.mask))

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
        # Reinjection uses the *same* weights as the loss, so the two cancel
        # term by term and the explicit half conserves mass exactly.
        r[self.i_zero] += float(P @ self.mask)
        return r

    # -- flux-limited advection ----------------------------------------------
    def _tvd_defect(self, P: np.ndarray, v: float) -> np.ndarray:
        """A_upwind @ P  -  A_tvd(P):  what upwind adds over the limited flux.

        Both terms are conservative flux divergences, so this sums to zero and
        adding it to the explicit half cannot break mass conservation.
        """
        if v == 0.0:
            return np.zeros(self.n)
        a_main, a_upper, a_lower = self._advection_bands(v)
        up = a_main * P
        up[1:] += a_lower[1:] * P[:-1]
        up[:-1] += a_upper[:-1] * P[1:]
        return up - advection_tvd_rhs(P, v, self.grid)

    def step(self, P: np.ndarray, gammadot_next: float, dt: float,
             D_old: float | None = None) -> tuple[np.ndarray, float]:
        """Advance P by dt with strain rate ``gammadot_next`` at the new level."""
        if D_old is None:
            D_old = self._D_of(P)

        th, n = self.theta, self.n
        a_main, a_upper, a_lower = self._advection_bands(gammadot_next)

        # explicit half (same time level) -- conserves mass exactly
        rhs_expl = P + dt * self._react(P)
        if self.advection == "tvd":
            # Defect correction rather than operator splitting: keep the monotone
            # upwind operator implicit (it is what the banded solve can invert)
            # and carry the flux-limiter correction explicitly.  At a fixed point
            # the upwind terms cancel and what remains is exactly
            #   D L2 P + A_tvd(P) + Yield P + Source P = 0,
            # i.e. the *same* stationary equation solve_steady solves -- so the
            # integrator relaxes to the direct solver's answer to round-off, for
            # any dt.  Strang splitting would only match it to O(dt).
            rhs_expl = rhs_expl - dt * self._tvd_defect(P, float(gammadot_next))

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
            record_every: int = 1, record_states: bool = False) -> Trajectory:
        n_steps = int(round((t_end - t0) / dt))
        P = np.array(P0, dtype=float)
        ts, ss, gs, ds = [], [], [], []
        Ps = [] if record_states else None
        D = self._D_of(P)
        sigma = self.grid.sigma
        for k in range(n_steps + 1):
            t = t0 + k * dt
            if k % record_every == 0:
                ts.append(t)
                ss.append(self.h * float(sigma @ P))
                gs.append(self.h * float(P @ self.mask))
                ds.append(D)
                if record_states:
                    Ps.append(P.copy())
            if k == n_steps:
                break
            P, D = self.step(P, float(gammadot_func(t + 0.5 * dt)), dt, D_old=D)
        P_states = np.array(Ps) if record_states else None
        return Trajectory(np.array(ts), np.array(ss), np.array(gs), np.array(ds),
                          P, self.grid, P_states)

    def propagate(self, P0: np.ndarray, gammadot_func: Callable[[float], float],
                  t_end: float, dt: float, t0: float = 0.0) -> np.ndarray:
        n_steps = int(round((t_end - t0) / dt))
        P = np.array(P0, dtype=float)
        D = self._D_of(P)
        for k in range(n_steps):
            t = t0 + k * dt
            P, D = self.step(P, float(gammadot_func(t + 0.5 * dt)), dt, D_old=D)
        return P


def initial_delta(grid: Grid) -> np.ndarray:
    """All probability at sigma = 0 (a natural relaxed/quenched start)."""
    P = np.zeros(grid.n)
    P[grid.i_zero] = 1.0 / grid.h
    return P
