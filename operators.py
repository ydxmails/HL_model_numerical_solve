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
near the sharp fronts at +/-1.  Its numerical diffusion is controlled by
``n_per_unit`` and cross-checked against the analytic steady/SAOS oracles.
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
    no-flux ends so that it conserves mass exactly.
    """
    n, h = grid.n, grid.h
    v = float(gammadot)
    A = sp.lil_matrix((n, n))
    if v == 0.0:
        return A.tocsr()
    if v > 0.0:
        # face value = upstream (left) node; J_{i+1/2} = v P_i
        for i in range(n):
            if i > 0:
                A[i, i - 1] += v / h          # inflow from left face
            if i < n - 1:
                A[i, i] += -v / h             # outflow through right face
            # i == n-1: right face is no-flux (no outflow term)
    else:
        # face value = upstream (right) node; J_{i+1/2} = v P_{i+1}
        for i in range(n):
            if i < n - 1:
                A[i, i + 1] += -v / h         # inflow from right face
            if i > 0:
                A[i, i] += v / h              # outflow through left face
            # i == 0: left face is no-flux (no outflow term)
    return A.tocsr()


def yielding_matrix(grid: Grid) -> sp.csr_matrix:
    """Diagonal plastic-loss operator -H(|sigma| - 1) (rate 1/tau = 1)."""
    d = np.where(grid.yield_mask, -1.0, 0.0)
    return sp.diags(d, format="csr")


def source_matrix(grid: Grid) -> sp.csr_matrix:
    """Reinjection source Gamma*delta(sigma) as a rank-one operator.

    Row ``i_zero`` receives the discrete yielding rate sum_{|sigma|>1} P_j
    (the h from the integration weight and the 1/h from the discrete delta
    cancel).  Together with the yielding loss this exactly balances mass.
    """
    n = grid.n
    cols = np.where(grid.yield_mask)[0]
    rows = np.full(cols.size, grid.i_zero)
    data = np.ones(cols.size)
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
