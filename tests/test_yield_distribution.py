"""Tests for the yield-stress distribution rho_ac(sigma) (steady + oscillatory)."""

import numpy as np

from hlmodel import (Grid, solve_steady, yield_stress_distribution,
                     D_quiescent)
from hlmodel.periodic import solve_laos


def test_quiescent_is_exponential():
    # Quiescent liquid: rho_ac is a symmetric two-sided exponential with mean
    # overshoot sqrt(D); numerics must match the analytic form.
    grid = Grid(sigma_max=10, n_per_unit=300)
    alpha = 0.8
    res = solve_steady(alpha, 0.0, grid)
    D = res.D
    yd = res.yield_stress_distribution()
    assert yd.defined
    # unit integral
    assert np.isclose(grid.h * yd.rho.sum(), 1.0, atol=1e-10)
    # mean penetration past threshold == sqrt(D)
    assert np.isclose(yd.mean_overshoot, np.sqrt(D), rtol=0.02)
    # symmetric (no shear direction)
    assert yd.asymmetry < 1e-3
    # pointwise match to (1/2 sqrt D) exp(-(|s|-1)/sqrt D)
    s, rho = yd.support
    rho_exact = (1.0 / (2.0 * np.sqrt(D))) * np.exp(-(np.abs(s) - 1.0) / np.sqrt(D))
    assert np.max(np.abs(rho - rho_exact)) < 3e-2


def test_steady_shear_is_asymmetric_with_correct_decay():
    # Under shear the tails stay exponential but with direction-dependent decay
    # lengths ell_pm = 2D / (sqrt(gd^2 + 4D) -+ gd); downstream (+) is fatter.
    grid = Grid(sigma_max=12, n_per_unit=300)
    alpha, gd = 0.8, 1.0
    res = solve_steady(alpha, gd, grid, warn=False)
    D = res.D
    yd = res.yield_stress_distribution()
    assert np.isclose(grid.h * yd.rho.sum(), 1.0, atol=1e-10)
    assert yd.asymmetry > 0.1                       # clearly asymmetric

    s, rho = yd.support
    up = s > 1
    dn = s < -1
    lp = 2 * D / (np.sqrt(gd**2 + 4 * D) - gd)      # analytic downstream length
    lm = 2 * D / (np.sqrt(gd**2 + 4 * D) + gd)      # analytic upstream length
    assert lp > lm                                   # downstream tail longer

    def measured_len(x, y):
        m = (x > 0.15) & (x < 1.2) & (y > 1e-4)
        return -1.0 / np.polyfit(x[m], np.log(y[m]), 1)[0]

    lp_meas = measured_len(s[up] - 1.0, rho[up])
    lm_meas = measured_len(-s[dn] - 1.0, rho[dn])
    assert np.isclose(lp_meas, lp, rtol=0.05)
    assert np.isclose(lm_meas, lm, rtol=0.05)


def test_convenience_matches_function():
    grid = Grid(sigma_max=8, n_per_unit=150)
    res = solve_steady(0.6, 0.3, grid)
    a = res.yield_stress_distribution()
    b = yield_stress_distribution(res.P, grid)
    assert np.allclose(a.rho, b.rho)
    assert np.isclose(a.Gamma, b.Gamma)


def test_jammed_frozen_is_undefined():
    # Quiescent jammed state (alpha < alpha_c) has D = Gamma = 0: nothing yields.
    grid = Grid(sigma_max=8, n_per_unit=150)
    assert D_quiescent(0.3) == 0.0
    res = solve_steady(0.3, 0.0, grid)
    yd = res.yield_stress_distribution()
    assert not yd.defined
    assert np.all(yd.rho == 0.0)
    assert yd.asymmetry == 0.0


def test_laos_cycle_distribution_symmetry_and_phases():
    # Period-averaged rho_ac is symmetric (half-period symmetry); an individual
    # phase is asymmetric.  yield_phases=4 attaches both to the result.
    grid = Grid(sigma_max=8, n_per_unit=150)
    res = solve_laos(0.8, 1.0, 0.3, grid, method="picard", steps_per_period=600,
                     tol=1e-9, yield_phases=4)
    cyd = res.yield_distribution
    assert cyd is not None
    avg = cyd.averaged
    assert avg.defined
    assert np.isclose(grid.h * avg.rho.sum(), 1.0, atol=1e-9)
    assert avg.asymmetry < 5e-3                      # symmetric to grid/dt floor
    # phase-resolved slices returned, and at least one is strongly asymmetric
    assert cyd.resolved is not None and len(cyd.resolved) == 4
    assert max(r.asymmetry for r in cyd.resolved) > 0.1


def test_laos_yield_distribution_off_by_default():
    grid = Grid(sigma_max=8, n_per_unit=150)
    res = solve_laos(0.8, 1.0, 0.3, grid, method="picard", steps_per_period=400,
                     tol=1e-9)
    assert res.yield_distribution is None
