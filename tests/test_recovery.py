"""Recovery-rheology tests: the recoverable / unrecoverable strain decomposition."""

import numpy as np
import pytest

from hlmodel import Grid, solve_steady
from hlmodel.operators import build_static_parts
from hlmodel.recovery import (affine_shift, plastic_stress, recover_from_state,
                              steady_recovery, laos_recovery)


# --- the affine strain step ------------------------------------------------
@pytest.mark.parametrize("dgamma", [-0.338649, 0.12345, -1.7, 0.00333333, 2.5])
def test_affine_shift_conserves_mass_and_shifts_the_moment(dgamma):
    # A step strain is affine: it must move Sigma by exactly dgamma while
    # conserving probability.  The donor-cell remap is exact in both.
    grid = Grid(sigma_max=10, n_per_unit=200)
    P = solve_steady(0.4, 0.1, grid, warn=False).P
    Q = affine_shift(P, dgamma, grid)
    assert abs(grid.integrate(Q) - grid.integrate(P)) < 1e-12
    assert abs(grid.first_moment(Q) - (grid.first_moment(P) + dgamma)) < 1e-11


def test_affine_shift_by_whole_cells_is_an_exact_translation():
    grid = Grid(sigma_max=6, n_per_unit=100)
    P = solve_steady(0.4, 0.2, grid, warn=False).P
    k = 7
    Q = affine_shift(P, k * grid.h, grid)
    expected = np.zeros_like(P)
    expected[k:] = P[:-k]
    assert np.allclose(Q, expected, atol=1e-14)


def test_affine_shift_of_zero_is_identity():
    grid = Grid(sigma_max=6, n_per_unit=100)
    P = solve_steady(0.4, 0.2, grid, warn=False).P
    assert np.array_equal(affine_shift(P, 0.0, grid), P)


# --- the first-moment identity ---------------------------------------------
@pytest.mark.parametrize("alpha", [0.3, 0.4, 0.8])
@pytest.mark.parametrize("gd", [0.01, 0.1])
def test_plastic_stress_balances_the_rate_in_steady_state(alpha, gd):
    # dSigma/dt = gammadot - Sigma_pl, so a steady state has Sigma_pl = gammadot.
    grid = Grid(sigma_max=10, n_per_unit=300)
    res = solve_steady(alpha, gd, grid, warn=False)
    assert np.isclose(plastic_stress(res.P, grid), gd, rtol=1e-4)


# --- steady-shear recovery --------------------------------------------------
def test_steady_recovery_liquid_reaches_rest_and_balances():
    grid = Grid(sigma_max=8, n_per_unit=150)
    r = steady_recovery(0.8, 0.1, grid, warn=False, dt=1e-2, t_max=500.0)
    assert r.converged
    assert abs(r.mass_lost) < 1e-12
    # the dSigma/dt identity is independent of the solver internals, so a small
    # residual certifies both the stepping and the held-stress constraint
    assert r.balance_error / abs(r.stress) < 1e-6
    # the constraint really was held: the released state carries no stress
    assert abs(grid.first_moment(r.P_final)) < 1e-8
    assert 0.0 < r.gamma_rec < 2.0 * r.stress


def test_steady_recovery_has_no_strain_origin():
    # gamma(t) = gammadot * t is unbounded under steady shear, so gamma_unrec
    # is undefined -- correctly, since all steady flow is unrecoverable.
    grid = Grid(sigma_max=8, n_per_unit=150)
    r = steady_recovery(0.8, 0.1, grid, warn=False, dt=1e-2, t_max=500.0)
    assert np.isnan(r.gamma_at_release)
    assert np.isnan(r.gamma_unrec)
    assert np.isfinite(r.gamma_rec)


def test_delayed_creep_changes_sign_with_shear_rate():
    # After the affine recoil, whether the surviving over-threshold population
    # carries net positive or negative stress depends on the width of the
    # distribution against the size of the shift.  At low rate the distribution
    # is narrow, the recoil pushes the downstream tail below +1, and the
    # material keeps recoiling backward (over-recovery, affine_ratio > 1).
    grid = Grid(sigma_max=10, n_per_unit=200)
    parts = build_static_parts(grid)
    lo = steady_recovery(0.4, 0.01, grid, parts, warn=False, dt=1e-2, t_max=60.0)
    hi = steady_recovery(0.4, 0.1, grid, parts, warn=False, dt=1e-2, t_max=60.0)
    assert lo.delayed_creep < 0.0 < hi.delayed_creep
    assert lo.affine_ratio > 1.0 > hi.affine_ratio


def test_quiescent_jammed_state_has_nothing_to_recover():
    # Frozen, symmetric, zero stress: no recoil, no creep.
    grid = Grid(sigma_max=6, n_per_unit=100)
    res = solve_steady(0.4, 0.0, grid, warn=False)
    r = recover_from_state(res.P, 0.4, grid, warn=False, dt=1e-2, t_max=50.0)
    assert abs(r.stress) < 1e-12
    assert abs(r.gamma_rec) < 1e-10
    assert np.isnan(r.affine_ratio)          # 0/0, not silently zero


def test_base_state_resolution_is_reachable():
    # RecoveryResult.converged refers to the recovery integration ALONE.  A
    # recovery can converge perfectly from a base state that was itself
    # resolution-limited, which would silently produce a well-converged wrong
    # answer -- so the SteadyResult is kept and folded into base_ok.
    grid = Grid(sigma_max=5, n_per_unit=200)      # steady floor ~ 8h^2 = 2e-4
    parts = build_static_parts(grid)
    opts = dict(warn=False, tol=1e-6, dt=1e-2, t_max=400.0)
    ok = steady_recovery(0.4, 3e-4, grid, parts, **opts)
    bad = steady_recovery(0.4, 5e-5, grid, parts, **opts)   # below the floor
    assert ok.base_ok and ok.base.converged
    assert not bad.base_ok
    assert bad.converged and not bad.base.converged   # the trap: recovery ok, base not


def test_low_rate_ratio_is_the_recoverable_compliance():
    # In the liquid phase the gammadot -> 0 recovery is classical linear
    # viscoelasticity: gamma_rec = J_e0 * Sigma, so affine_ratio -> J_e0 * G0.
    # SAOS gives J_e0 independently and exactly in time, since G' -> eta0^2 J_e0
    # omega^2 and G'' -> eta0 omega, hence J_e0 = lim G'/(G'')^2.  Two unrelated
    # solvers, one number -- this ties recovery.py to saos.py.
    #
    # J_e0 * G_inf >= 1 by Cauchy-Schwarz, with equality only for a single
    # relaxation time, so the ratio exceeding 1 ("over-recovery") is generic and
    # grows as the spectrum broadens toward alpha_c.
    from hlmodel.saos import saos_spectrum
    grid = Grid(sigma_max=12, n_per_unit=300)
    parts = build_static_parts(grid)
    for alpha in (0.8, 0.6):
        sp = saos_spectrum(alpha, [1e-4], grid, parts)
        Je0 = float(sp.Gp[0] / sp.Gpp[0] ** 2)
        assert Je0 >= 1.0                       # Cauchy-Schwarz
        r = steady_recovery(alpha, 1e-3, grid, parts, warn=False, tol=1e-10,
                            dt=5e-3, t_max=600.0)
        assert r.converged and r.base_ok
        assert abs(r.affine_ratio - Je0) < 1e-3 * Je0


# --- LAOS, phase-resolved ---------------------------------------------------
def test_laos_subyield_jammed_recovers_elastically():
    # Below the yield strain the jammed material never yields, so Sigma_pl = 0,
    # there is no delayed creep, and the recovery is the pure affine recoil:
    # gamma_rec == Sigma exactly.
    #
    # The cycle itself is NOT unique there -- the period map is the identity, so
    # solve_laos returns whatever seeded it, and that state can carry a frozen-in
    # stress offset (Sigma(t) = gamma(t) + const).  The decomposition then reports
    # a *constant* gamma_unrec.  So the robust assertion is that nothing is
    # accumulated within the cycle, not that gamma_unrec vanishes: the latter
    # holds only for whichever absorbing state the seed happened to reach.
    grid = Grid(sigma_max=6, n_per_unit=100)
    cr = laos_recovery(0.4, 0.03, 0.3, grid, n_phases=4, steps_per_period=200,
                       warn=False, dt=1e-2, t_max=50.0)
    # The absorbing state is approached asymptotically, not reached exactly, so
    # these are relative to the drive amplitude rather than at machine precision.
    g0 = 0.03
    assert np.allclose(cr.gamma_rec, cr.stress, atol=1e-5 * g0)   # pure affine recoil
    assert all(abs(r.delayed_creep) < 1e-5 * g0 for r in cr.results)  # no plastic creep
    assert np.std(cr.gamma_unrec) < 1e-3 * g0                     # nothing accumulated


def test_laos_decomposition_is_consistent():
    grid = Grid(sigma_max=6, n_per_unit=100)
    cr = laos_recovery(0.4, 0.5, 0.3, grid, n_phases=4, steps_per_period=200,
                       warn=False, dt=2e-2, t_max=60.0)
    # gamma = gamma_rec + gamma_unrec holds by construction; this guards the signs
    assert np.allclose(cr.gamma_rec + cr.gamma_unrec, cr.gamma, atol=1e-12)
    # above the yield strain some strain really is unrecoverable
    assert np.max(np.abs(cr.gamma_unrec)) > 1e-3
    assert cr.phase.shape == cr.gamma_rec.shape == (4,)


def test_laos_refuses_a_superposed_cycle():
    # laos_recovery rebuilds gamma(t) = gamma0 sin(omega t) and uses the sine's
    # strain origin, so it must refuse a drive with a steady drift rather than
    # silently return a plausible but wrong decomposition.
    from hlmodel.superposed import solve_superposed
    grid = Grid(sigma_max=6, n_per_unit=100)
    parts = build_static_parts(grid)
    sup = solve_superposed(0.4, 0.1, 0.2, 0.5, grid, parts, with_linear=False,
                           steps_per_period=200, warn=False)
    with pytest.raises(NotImplementedError):
        laos_recovery(0.4, 0.2, 0.5, grid, parts, n_phases=2,
                      laos_result=sup.cycle, warn=False, dt=2e-2, t_max=20.0)
