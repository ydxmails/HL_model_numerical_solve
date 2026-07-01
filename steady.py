"""Direct stationary-state solver for steady shear.

Strategy (the "oracle" that validates the time-stepping engine):

For a *fixed* D, the stationary equation G(D, gammadot) P = 0 is a homogeneous
linear system.  Because the reinjection source is built conservatively, every
column sum of G is zero, so the constant vector is an exact left null vector and
G has rank n-1: a one-parameter family of solutions P = R * p.  We pin the scale
by replacing one redundant row with the normalisation integral P dsigma = 1,
giving a non-singular system with a unique normalised solution.

We then close the model with D = alpha * Gamma(P(D)) by a one-dimensional
root-find on D.  Under any nonzero shear rate the paper guarantees a single
stationary solution, so this root is unique.

This construction has no relaxation-time problem, so it is fast and accurate
exactly where the transient integrator crawls (low rate, alpha -> alpha_c).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq

from .grid import Grid
from .operators import advection_matrix, build_static_parts
from .selfconsistency import ALPHA_C, D_quiescent

# Probability allowed within the outer 10% of the domain before we judge the box
# too small (the distribution is being truncated and the stress underestimated).
BOUNDARY_FRACTION_TOL = 1e-3


@dataclass
class SteadyResult:
    alpha: float
    gammadot: float
    D: float
    Gamma: float
    stress: float
    P: np.ndarray = field(repr=False)
    grid: Grid = field(repr=False)
    converged: bool = True          # False if D hit the conditioning floor (rate too low)
    boundary_fraction: float = 0.0  # probability mass in the outer 10% of the domain


def _boundary_fraction(P: np.ndarray, grid: Grid) -> float:
    """Fraction of probability sitting in the outer 10% of the stress domain.

    A large value means the distribution is being squashed against the no-flux
    walls at +/-sigma_max, so the domain is too small and the stress is
    underestimated -- increase sigma_max.
    """
    outer = np.abs(grid.sigma) > 0.9 * grid.sigma_max
    return float(grid.integrate(P[outer]))


def stationary_distribution(D: float, gammadot: float, grid: Grid,
                            parts: dict | None = None) -> np.ndarray:
    """Normalised stationary P for fixed (D, gammadot).

    Solves G P = 0 with one row swapped for the normalisation constraint.
    """
    if parts is None:
        parts = build_static_parts(grid)
    A = advection_matrix(gammadot, grid)
    G = (D * parts["L2"] + A + parts["Yield"] + parts["Source"]).tolil()

    n, h = grid.n, grid.h
    # Replace the last (redundant by conservation) row with normalisation.
    G[n - 1, :] = h
    b = np.zeros(n)
    b[n - 1] = 1.0

    P = spla.spsolve(G.tocsr(), b)
    return P


def _closure_residual(D: float, alpha: float, gammadot: float, grid: Grid,
                      parts: dict) -> float:
    """r(D) = alpha * Gamma(P(D)) - D; a root is a self-consistent state."""
    P = stationary_distribution(D, gammadot, grid, parts)
    Gamma = grid.yield_fraction(P)
    return alpha * Gamma - D


def solve_steady(alpha: float, gammadot: float, grid: Grid | None = None,
                 parts: dict | None = None,
                 D_hi: float = 5.0, n_scan: int = 80,
                 warn: bool = True) -> SteadyResult:
    """Solve the steady HL state at coupling ``alpha`` and shear rate ``gammadot``.

    Returns a :class:`SteadyResult` with the noise amplitude D, yielding rate
    Gamma, macroscopic stress, and the full distribution P.

    If ``warn`` is True (default), emits a :class:`RuntimeWarning` when the result
    is resolution-limited: either the shear rate is too low for the grid (the
    self-consistent D fell to the conditioning floor ~4h^2 -> refine
    ``n_per_unit``) or the distribution is being truncated by the domain (too much
    probability near +/-sigma_max -> increase ``sigma_max``).  Sweep drivers pass
    ``warn=False`` and aggregate the diagnostics instead.
    """
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)

    gd = abs(float(gammadot))      # solve at |gammadot|; stress sign restored below
    sgn = 1.0 if gammadot >= 0 else -1.0

    # The stationary solve is well conditioned only while diffusion dominates the
    # O(1) yielding term, i.e. D / h^2 >> 1.  Floor the search accordingly; very
    # near alpha_c (D -> 0) this is the resolution limit -- refine n_per_unit.
    D_lo = max(1e-6, 4.0 * grid.h * grid.h)

    # ---- Quiescent base state (gammadot = 0): D is known in closed form -------
    if gd < 1e-14:
        D0 = D_quiescent(alpha)
        if D0 <= 0.0:                      # jammed, frozen, degenerate manifold
            P = np.zeros(grid.n)
            P[grid.i_zero] = 1.0 / grid.h  # representative delta at sigma = 0
            return SteadyResult(alpha, 0.0, 0.0, 0.0, 0.0, P, grid)
        P = stationary_distribution(D0, 0.0, grid, parts)
        return SteadyResult(alpha, 0.0, float(D0),
                            float(grid.yield_fraction(P)),
                            float(grid.first_moment(P)), P, grid,
                            boundary_fraction=_boundary_fraction(P, grid))

    # ---- Sheared state: root-find the self-consistent D -----------------------
    Ds = np.logspace(np.log10(D_lo), np.log10(D_hi), n_scan)
    rs = np.array([_closure_residual(D, alpha, gd, grid, parts) for D in Ds])

    root = None
    for k in range(len(Ds) - 1):
        if rs[k] == 0.0:
            root = Ds[k]
            break
        if np.isfinite(rs[k]) and np.isfinite(rs[k + 1]) and rs[k] * rs[k + 1] < 0.0:
            root = brentq(_closure_residual, Ds[k], Ds[k + 1],
                          args=(alpha, gd, grid, parts), xtol=1e-13, rtol=1e-13)
            break

    converged = root is not None
    if root is None:
        # No sign change: residual stayed negative -> self-consistent D below the
        # conditioning floor (near-jammed).  Report the floor and flag it.
        root = D_lo

    D = float(root)
    P = stationary_distribution(D, gd, grid, parts)
    Gamma = float(grid.yield_fraction(P))
    stress = sgn * float(grid.first_moment(P))
    bfrac = _boundary_fraction(P, grid)

    if warn:
        if not converged:
            warnings.warn(
                f"gammadot={gammadot:g} is below the resolution floor for this grid "
                f"(self-consistent D fell to ~4h^2={D_lo:.1e}); the steady result is "
                f"unreliable. Increase n_per_unit (finer grid) to reach lower rates.",
                RuntimeWarning, stacklevel=2)
        if bfrac > BOUNDARY_FRACTION_TOL:
            warnings.warn(
                f"gammadot={gammadot:g}: {bfrac:.1%} of the probability lies in the "
                f"outer 10% of the domain -- the distribution is being truncated at "
                f"sigma_max={grid.sigma_max:g} and the stress is underestimated. "
                f"Increase sigma_max (the high-rate branch needs sigma_max >> gammadot).",
                RuntimeWarning, stacklevel=2)

    return SteadyResult(alpha, float(gammadot), D, Gamma, stress, P, grid,
                        converged, bfrac)
