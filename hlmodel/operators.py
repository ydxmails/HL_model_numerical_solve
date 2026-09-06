"""Spatial finite-difference operators for the HL Fokker-Planck equation.

The evolution equation (reduced units tau = sigma_c = G0 = 1) is

    dP/dt = -d_sigma J(P)  -  H(|sigma|-1) P  +  Gamma delta(sigma),
    J(P)  =  gammadot * P  -  D * d_sigma P            (probability current)

with the self-consistent closures Gamma = integral_{|sigma|>1} P dsigma and
D = alpha * Gamma.

**Conservative (flux-form) discretisation.**  We place P on nodes and fluxes on
the half-way faces, upwinding the advective part and using a no-flux condition
J = 0 at the two outermost faces.  The advection + diffusion update then
telescopes, so the column sums of the assembled generator vanish *exactly*: the
constant vector is an exact left null vector and total probability is conserved
to machine precision.  This has two payoffs:

* the stationary problem becomes a well-posed null-vector computation
  (``steady.py``): G is exactly rank n-1, and one normalisation row closes it;
* the transient integrator conserves mass identically (``transient.py``).

First-order upwinding (rather than centered) keeps the scheme monotone in the
advection-dominated jammed phase (D -> 0), where centered differences would ring
near the sharp fronts at +/-1.  It is, however, only *first* order: it carries a
numerical diffusivity

    D_num ~ |gammadot| h / 2 ,

which in this model is not a benign truncation error but a contamination of the
state variable itself, since HL's own noise D = alpha*Gamma is a diffusivity of
the same kind.  :func:`advection_tvd_rhs` therefore provides a second-order
flux-limited alternative; it is nonlinear in P (the limiter is solution
dependent) so it cannot be assembled as a matrix, and the stationary solver
embeds it by defect correction on top of the upwind matrix below.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .grid import Grid


def diffusion_matrix(grid: Grid) -> sp.csr_matrix:
    """Reflecting (no-flux) Laplacian L2 so that D*L2 ~ D d2_sigma.

    Row sums vanish, hence column sums vanish, hence mass is conserved.
    """
    n, h = grid.n, grid.h
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    L2 = sp.diags([off, main, off], offsets=[-1, 0, 1], format="lil")
    # No-flux ends: the missing outward face contributes nothing.
    L2[0, 0] = -1.0
    L2[-1, -1] = -1.0
    return (L2.tocsr()) / (h * h)


def advection_matrix(gammadot: float, grid: Grid) -> sp.csr_matrix:
    """Conservative first-order upwind discretisation of -d_sigma(gammadot P).

    Sign-aware (stable for sign-changing oscillatory drives) and built with
    no-flux ends so that it conserves mass exactly.  Assembled directly from its
    diagonals (no Python loop over nodes).
    """
    n, h = grid.n, grid.h
    v = float(gammadot)
    if v == 0.0:
        return sp.csr_matrix((n, n))
    if v > 0.0:
        # J_{i+1/2} = v P_i (upstream = left): outflow -v/h on the diagonal
        # (except the last node, no-flux), inflow +v/h on the sub-diagonal.
        main = np.full(n, -v / h)
        main[-1] = 0.0
        sub = np.full(n - 1, v / h)              # offset -1: A[i, i-1]
        A = sp.diags([sub, main], offsets=[-1, 0], format="csr")
    else:
        # J_{i+1/2} = v P_{i+1} (upstream = right): outflow +v/h on the diagonal
        # (except the first node, no-flux), inflow -v/h on the super-diagonal.
        main = np.full(n, v / h)
        main[0] = 0.0
        sup = np.full(n - 1, -v / h)             # offset +1: A[i, i+1]
        A = sp.diags([main, sup], offsets=[0, 1], format="csr")
    return A


def advection_tvd_rhs(P: np.ndarray, gammadot: float, grid: Grid) -> np.ndarray:
    """-d_sigma(gammadot P) with a van Leer flux-limited (second-order TVD) flux.

    This is the zero-CFL (stationary) limit of the flux used by
    :class:`~hlmodel.transient.TransientStepper`; the transient version carries
    an extra (1 -/+ C) Lax-Wendroff factor that vanishes as dt -> 0, so the two
    schemes are the same spatial discretisation.

    Nonlinear in ``P``: the limiter ratio r depends on the local solution, which
    is exactly what lets the scheme be second order in smooth regions while
    reverting to monotone upwind at the kinks in +/-1.  Returned in flux form
    with zero flux at the two outer faces, so the result sums to zero to machine
    precision (mass conserving) -- the property the stationary solver relies on
    to keep its dropped row redundant.
    """
    n, h = grid.n, grid.h
    v = float(gammadot)
    out = np.zeros(n)
    if v == 0.0:
        return out

    d = np.diff(P)                        # P[i+1]-P[i], one per interior face
    safe = np.abs(d) > 1e-300
    r = np.zeros(n - 1)                   # r = 0 where unsafe -> phi = 0 -> upwind
    num = np.empty(n - 1)
    if v > 0.0:
        # face i+1/2 upwinds from cell i: r_i = (P_i - P_{i-1}) / (P_{i+1} - P_i)
        num[0] = 0.0                      # no P_{-1}: fall back to first order
        num[1:] = d[:-1]
        np.divide(num, d, out=r, where=safe)
        phi = (r + np.abs(r)) / (1.0 + np.abs(r))      # van Leer limiter
        F = v * (P[:-1] + 0.5 * phi * d)
    else:
        # face i+1/2 upwinds from cell i+1: r = (P_{i+2}-P_{i+1})/(P_{i+1}-P_i)
        num[-1] = 0.0                     # no P_{n}: fall back to first order
        num[:-1] = d[1:]
        np.divide(num, d, out=r, where=safe)
        phi = (r + np.abs(r)) / (1.0 + np.abs(r))
        F = v * (P[1:] - 0.5 * phi * d)

    out[:-1] -= F / h                     # -F_{i+1/2} for i = 0..n-2
    out[1:] += F / h                      # +F_{i-1/2} for i = 1..n-1
    return out


def advection_matrix_frozen(gammadot: float, grid: Grid,
                            P_ref: np.ndarray) -> sp.csr_matrix:
    """Linear van Leer advection with the limiter frozen at ``P_ref``.

    :func:`advection_tvd_rhs` is nonlinear in P, which is fine for a stationary
    solve or a time march but breaks any construction that needs an actual
    *linear operator* -- the cohort/first-passage machinery in ``residence.py``
    and ``elastic_ratio.py``, whose identities (e.g. P = Gamma * integral Q da)
    hold only for a linear generator, since integral A(Q) da != A(integral Q da).

    The van Leer ratio r = (P_i - P_{i-1}) / (P_{i+1} - P_i) is invariant under
    P -> cP, so evaluating the limiter once at ``P_ref`` and holding it fixed
    gives a linear (tridiagonal, mass-conserving) matrix A with

        A @ P_ref == advection_tvd_rhs(P_ref, gammadot, grid)

    exactly.  The stationary state is therefore unchanged, while cohorts built
    on A obey the linear identities.
    """
    n, h = grid.n, grid.h
    v = float(gammadot)
    if v == 0.0:
        return sp.csr_matrix((n, n))

    P = np.asarray(P_ref, dtype=float)
    d = np.diff(P)
    safe = np.abs(d) > 1e-300
    r = np.zeros(n - 1)
    num = np.empty(n - 1)
    if v > 0.0:
        num[0] = 0.0
        num[1:] = d[:-1]
    else:
        num[-1] = 0.0
        num[:-1] = d[1:]
    np.divide(num, d, out=r, where=safe)
    phi = (r + np.abs(r)) / (1.0 + np.abs(r))

    # Face flux F_i = v (a_i P_i + b_i P_{i+1}) on the n-1 interior faces.
    if v > 0.0:
        a, b = 1.0 - 0.5 * phi, 0.5 * phi
    else:
        a, b = 0.5 * phi, 1.0 - 0.5 * phi

    main = np.zeros(n)
    upper = np.zeros(n - 1)      # A[i, i+1]
    lower = np.zeros(n - 1)      # A[i+1, i]
    main[:-1] -= v * a / h       # -F_i out of cell i (right face)
    upper[:] = -v * b / h
    main[1:] += v * b / h        # +F_{i-1} into cell i (left face)
    lower[:] = v * a / h
    return sp.diags([lower, main, upper], offsets=[-1, 0, 1], format="csr")


def yielding_matrix(grid: Grid) -> sp.csr_matrix:
    """Diagonal plastic-loss operator -H(|sigma| - 1) (rate 1/tau = 1).

    Uses ``grid.yield_weight`` (half weight exactly at |sigma| = 1) rather than
    the strict mask; :func:`source_matrix` must use the same weights for the
    discrete mass balance to stay exact.
    """
    return sp.diags(-grid.yield_weight, format="csr")


def source_matrix(grid: Grid) -> sp.csr_matrix:
    """Reinjection source Gamma*delta(sigma) as a rank-one operator.

    Row ``i_zero`` receives the discrete yielding rate sum_{|sigma|>1} P_j
    (the h from the integration weight and the 1/h from the discrete delta
    cancel).  Together with the yielding loss this exactly balances mass.
    """
    n = grid.n
    w = grid.yield_weight
    cols = np.where(w > 0.0)[0]
    rows = np.full(cols.size, grid.i_zero)
    data = w[cols]                       # same weights as the loss term
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def generator(D: float, gammadot: float, grid: Grid,
              precomputed: dict | None = None) -> sp.csr_matrix:
    """Full HL generator G(D, gammadot) = D*L2 + A + Yield + Source."""
    if precomputed is None:
        L2 = diffusion_matrix(grid)
        Y = yielding_matrix(grid)
        S = source_matrix(grid)
    else:
        L2, Y, S = precomputed["L2"], precomputed["Yield"], precomputed["Source"]
    A = advection_matrix(gammadot, grid)
    return (D * L2 + A + Y + S).tocsr()


def build_static_parts(grid: Grid) -> dict:
    """Pre-assemble the D- and gammadot-independent operator blocks."""
    return {
        "L2": diffusion_matrix(grid),
        "Yield": yielding_matrix(grid),
        "Source": source_matrix(grid),
    }
