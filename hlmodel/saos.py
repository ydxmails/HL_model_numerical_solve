"""Small-amplitude oscillatory shear (SAOS): linear viscoelastic moduli.

In the liquid phase (alpha > alpha_c) the quiescent state is unique, so the
response to an infinitesimal oscillatory strain can be obtained by *linearising*
about it -- no time stepping, one complex solve per frequency.

Write P = P0 + Re[P1 e^{i omega t}] with the package strain convention
gamma(t) = gamma0 sin(omega t).  To first order the amplitude p1 = P1 / gamma0
solves the linear integro-differential boundary-value problem

    (i omega I - G_lin) p1 = -omega d_sigma P0 ,

    G_lin = D0 L2 + Yield + Source + alpha (L2 P0)(h * mask)^T ,

where the last (nonlocal) term is the linearised noise feedback D1 = alpha
Gamma1.  Crucially Source = e0 mask^T and the feedback term *share the same row*
(proportional to mask^T), so together they form a single rank-one update to the
tridiagonal operator T = i omega I - (D0 L2 + Yield).  We invert with the
Sherman-Morrison formula: two tridiagonal complex solves per frequency.

The complex stress amplitude is S* = integral sigma p1 dsigma, and with the sin
convention the moduli are

    G'(omega)  = -Im S* ,     G''(omega) = Re S* .

So defined, they coincide with the LAOS first-harmonic moduli as gamma0 -> 0
(see ``periodic.py`` and the alignment test).  At low frequency they must
recover the Maxwell limit G' ~ omega^2, G'' ~ omega (> 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded

from .grid import Grid
from .operators import (build_static_parts, advection_matrix_frozen,
                        advection_tvd_rhs)
from .selfconsistency import ALPHA_C, D_quiescent
from .steady import stationary_distribution, solve_steady, SteadyResult


@dataclass
class SAOSResult:
    alpha: float
    omega: np.ndarray
    Gp: np.ndarray            # storage modulus G'(omega)
    Gpp: np.ndarray           # loss modulus G''(omega)
    D0: float                 # base-state noise amplitude


def liquid_base_state(alpha: float, grid: Grid, parts: dict | None = None,
                      tol: float = 1e-13, max_iter: int = 200):
    """Discretely self-consistent quiescent base state (alpha > alpha_c).

    Polishes the analytic D = 1/2 + sqrt(D) + D guess to a fixed point of the
    discrete closure D = alpha * Gamma(P), so the base exactly satisfies the
    discrete stationary equation that SAOS is linearised about.
    """
    if alpha <= ALPHA_C:
        raise ValueError("SAOS base state requires the liquid phase alpha > 1/2")
    if parts is None:
        parts = build_static_parts(grid)
    D0 = D_quiescent(alpha)
    P0 = stationary_distribution(D0, 0.0, grid, parts)
    for _ in range(max_iter):
        D_new = alpha * float(grid.yield_fraction(P0))
        if abs(D_new - D0) <= tol * (1.0 + abs(D0)):
            D0 = D_new
            break
        D0 = D_new
        P0 = stationary_distribution(D0, 0.0, grid, parts)
    return P0, float(D0)


def _laplacian_bands(grid: Grid):
    n, h = grid.n, grid.h
    inv = 1.0 / (h * h)
    main = np.full(n, -2.0 * inv)
    main[0] = -inv
    main[-1] = -inv
    upper = np.full(n, inv)
    lower = np.full(n, inv)
    return main, upper, lower


def _apply_tridiag(main, upper, lower, x):
    y = main * x
    y[1:] += lower[1:] * x[:-1]
    y[:-1] += upper[:-1] * x[1:]
    return y


def saos_modulus(alpha: float, omega: float, grid: Grid,
                 parts: dict | None = None,
                 base: tuple[np.ndarray, float] | None = None
                 ) -> tuple[float, float]:
    """Complex modulus at a single frequency.  Returns (G', G'')."""
    if parts is None:
        parts = build_static_parts(grid)
    if base is None:
        P0, D0 = liquid_base_state(alpha, grid, parts)
    else:
        P0, D0 = base

    n, h = grid.n, grid.h
    mask = grid.yield_weight               # half weight exactly at |sigma| = 1
    l2_main, l2_upper, l2_lower = _laplacian_bands(grid)

    # T = D0 L2 + Yield  (tridiagonal);  M_T = i omega I - T
    t_main = D0 * l2_main - mask
    t_upper = D0 * l2_upper
    t_lower = D0 * l2_lower
    ab = np.zeros((3, n), dtype=complex)
    ab[0, 1:] = -t_upper[:-1]
    ab[1, :] = 1j * omega - t_main
    ab[2, :-1] = -t_lower[1:]

    # rank-one row update u mask^T  with  u = e_{i0} + alpha h (L2 P0)
    L2P0 = _apply_tridiag(l2_main, l2_upper, l2_lower, P0)
    u = alpha * h * L2P0.astype(complex)
    u[grid.i_zero] += 1.0

    # forcing f = -omega d_sigma P0
    g0 = np.gradient(P0, h)
    g0[0] = 0.0
    g0[-1] = 0.0
    f = (-omega * g0).astype(complex)

    x = solve_banded((1, 1), ab, f)
    y = solve_banded((1, 1), ab, u)
    denom = 1.0 - mask @ y
    p1 = x + y * (mask @ x) / denom

    S = h * (grid.sigma @ p1)          # complex stress amplitude
    Gp = -float(S.imag)
    Gpp = float(S.real)
    return Gp, Gpp


def saos_spectrum(alpha: float, omegas, grid: Grid | None = None,
                  parts: dict | None = None) -> SAOSResult:
    """Linear moduli over a list/array of frequencies."""
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)
    base = liquid_base_state(alpha, grid, parts)
    omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
    Gp = np.empty(omegas.size)
    Gpp = np.empty(omegas.size)
    for k, w in enumerate(omegas):
        Gp[k], Gpp[k] = saos_modulus(alpha, float(w), grid, parts, base=base)
    return SAOSResult(alpha, omegas, Gp, Gpp, base[1])


# ---------------------------------------------------------------------------
# Parallel superposition: linear response about the *sheared* steady state
# ---------------------------------------------------------------------------
"""
Everything above linearises about the quiescent base state, which is the right
thing for SAOS.  Parallel superposition instead superposes a small oscillation
*collinear* with a steady shear,

    gammadot(t) = gammadot0 + gammaA omega cos(omega t) ,

and asks for the linear response about the resulting sheared steady state
P_s(sigma; gammadot0) with its self-consistent D_s = alpha Gamma_s.  Writing
P = P_s + Re[P1 e^{i omega t}] and keeping first order, the amplitude
p1 = P1 / gammaA solves

    (i omega I - G_lin) p1 = -omega d_sigma P_s ,

    G_lin = D_s L2 + A(gammadot0) + Yield + Source
            + alpha (L2 P_s)(h mask)^T .

Two differences from the SAOS operator, and nothing else:

* the tridiagonal block T = D_s L2 + A + Yield now carries advection bands;
* the base state is sheared, so d_sigma P_s is not the quiescent gradient.

The advection block uses ``advection_matrix_frozen`` -- the van Leer limiter
frozen at P_s.  That is exactly what the frozen operator was built for: it is
linear and tridiagonal, and satisfies A @ P_s == advection_tvd_rhs(P_s, ...)
exactly, so the linearisation is taken about the *same* discrete stationary
equation ``solve_steady`` actually solved, not a first-order proxy for it.

The forcing reuses the same flux.  A(v) is linear in v at fixed sign, so the
discrete -omega d_sigma P_s is (omega / gammadot0) * advection_tvd_rhs(P_s,
gammadot0, grid) -- second order, and consistent with the operator above.  (SAOS
uses ``np.gradient`` instead, which is the correct choice there precisely because
the quiescent base carries no advection operator to be consistent with.)

Source and the noise-feedback term still share the same mask^T row, so the whole
correction is still rank one and Sherman-Morrison still applies: two complex
tridiagonal solves per frequency, the same cost as SAOS.

Two exact properties are used as checks:

* **Mass neutrality.**  1^T G_lin = 0 (every block has zero column sums, including
  the rank-one feedback since L2 P_s sums to zero), and the forcing is a flux
  divergence so 1^T f = 0.  Hence i omega 1^T p1 = 0, i.e. the perturbation
  carries no mass.  Measured ~1e-17.
* **The gammadot0 -> 0 limit is SAOS.**  With alpha > alpha_c the sheared base
  state tends to the quiescent one and these moduli must converge to
  ``saos_modulus``.
"""


@dataclass
class PSRResult:
    """Parallel-superposition linear moduli at fixed (alpha, gammadot0)."""
    alpha: float
    gammadot0: float
    omega: np.ndarray
    Gp: np.ndarray            # G'_parallel(omega; gammadot0)
    Gpp: np.ndarray           # G''_parallel(omega; gammadot0)
    D0: float                 # base-state noise amplitude D_s
    Gamma0: float             # base-state yielding rate
    stress0: float            # base-state steady stress (= the DC term of a LAOS run)


def psr_modulus(alpha: float, gammadot0: float, omega: float, grid: Grid,
                parts: dict | None = None,
                base: "SteadyResult | None" = None,
                return_p1: bool = False):
    """Parallel-superposition complex modulus at one frequency.  Returns (G', G'').

    ``base`` is a :class:`~hlmodel.steady.SteadyResult` at ``(alpha, gammadot0)``;
    pass it in to share one base state across a frequency sweep.  With
    ``return_p1`` the complex perturbation amplitude is returned as a third value
    (its integral should vanish -- see the module notes).

    The moduli are even in ``gammadot0`` (the model is invariant under
    sigma -> -sigma, gammadot -> -gammadot), so the sign is ignored.
    """
    if parts is None:
        parts = build_static_parts(grid)
    gd = abs(float(gammadot0))
    if gd == 0.0:
        raise ValueError(
            "gammadot0 = 0 is pure SAOS, not parallel superposition; use "
            "saos_modulus, which linearises about the quiescent base state with "
            "a centered gradient (the correct choice when there is no advection "
            "operator to stay consistent with).")
    if base is None:
        base = solve_steady(alpha, gd, grid, parts, warn=False)
    P_s, D_s = base.P, float(base.D)

    n, h = grid.n, grid.h
    mask = grid.yield_weight
    l2_main, l2_upper, l2_lower = _laplacian_bands(grid)

    # advection bands, van Leer limiter frozen at the base state
    A = advection_matrix_frozen(gd, grid, P_s)
    a_main = np.asarray(A.diagonal(0), dtype=float)
    a_upper = np.zeros(n); a_upper[:-1] = A.diagonal(1)
    a_lower = np.zeros(n); a_lower[1:] = A.diagonal(-1)

    # T = D_s L2 + A + Yield  (tridiagonal);  M_T = i omega I - T
    t_main = D_s * l2_main + a_main - mask
    t_upper = D_s * l2_upper + a_upper
    t_lower = D_s * l2_lower + a_lower
    ab = np.zeros((3, n), dtype=complex)
    ab[0, 1:] = -t_upper[:-1]
    ab[1, :] = 1j * omega - t_main
    ab[2, :-1] = -t_lower[1:]

    # rank-one row update u mask^T (Source + linearised noise feedback)
    L2Ps = _apply_tridiag(l2_main, l2_upper, l2_lower, P_s)
    u = alpha * h * L2Ps.astype(complex)
    u[grid.i_zero] += 1.0

    # forcing f = -omega d_sigma P_s, same TVD flux as the operator
    f = ((omega / gd) * advection_tvd_rhs(P_s, gd, grid)).astype(complex)

    x = solve_banded((1, 1), ab, f)
    y = solve_banded((1, 1), ab, u)
    denom = 1.0 - mask @ y
    p1 = x + y * (mask @ x) / denom

    S = h * (grid.sigma @ p1)          # complex stress amplitude
    Gp, Gpp = -float(S.imag), float(S.real)
    return (Gp, Gpp, p1) if return_p1 else (Gp, Gpp)


def psr_spectrum(alpha: float, gammadot0: float, omegas, grid: Grid | None = None,
                 parts: dict | None = None) -> PSRResult:
    """Parallel-superposition moduli over many frequencies (shared base state)."""
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)
    base = solve_steady(alpha, abs(float(gammadot0)), grid, parts, warn=False)
    ws = np.atleast_1d(np.asarray(omegas, dtype=float))
    Gp = np.empty(ws.size); Gpp = np.empty(ws.size)
    for k, w in enumerate(ws):
        Gp[k], Gpp[k] = psr_modulus(alpha, gammadot0, float(w), grid, parts, base=base)
    return PSRResult(alpha, float(gammadot0), ws, Gp, Gpp,
                     float(base.D), float(base.Gamma), float(base.stress))
