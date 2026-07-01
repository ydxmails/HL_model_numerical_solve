"""Steady-state and self-consistency tests."""

import numpy as np
import pytest

from hlmodel import (Grid, solve_steady, D_quiescent, alpha_of_D_quiescent,
                     ALPHA_C)
from hlmodel.operators import generator


def test_alpha_c():
    assert ALPHA_C == 0.5
    assert D_quiescent(0.5) == 0.0
    assert D_quiescent(0.4) == 0.0          # jammed -> no noise


def test_quiescent_identity_roundtrip():
    # alpha = 1/2 + sqrt(D) + D  and its inverse must round-trip exactly.
    for D in [0.0, 1e-3, 1e-2, 0.1, 0.5, 1.0]:
        a = alpha_of_D_quiescent(D)
        assert np.isclose(D_quiescent(a), D, atol=1e-12, rtol=1e-9)


def test_generator_is_conservative():
    grid = Grid(sigma_max=6, n_per_unit=100)
    G = generator(0.05, 0.3, grid).toarray()
    # exact discrete mass conservation <=> zero column sums
    assert np.max(np.abs(G.sum(axis=0))) < 1e-9


def test_base_state_matches_analytic():
    grid = Grid(sigma_max=10, n_per_unit=300)
    for alpha in [0.55, 0.7, 0.9]:
        res = solve_steady(alpha, 0.0, grid)
        assert np.isclose(res.D, D_quiescent(alpha), rtol=1e-3)
        assert np.isclose(grid.integrate(res.P), 1.0, atol=1e-7)


@pytest.mark.parametrize("alpha", [0.3, 0.45, 0.6])
@pytest.mark.parametrize("gd", [0.05, 0.5])
def test_sheared_self_consistency_and_mass(alpha, gd):
    grid = Grid(sigma_max=8, n_per_unit=150)
    res = solve_steady(alpha, gd, grid)
    # closure D = alpha * Gamma must hold to solver tolerance
    assert abs(res.D - alpha * res.Gamma) < 1e-8
    # probability is normalised
    assert np.isclose(grid.integrate(res.P), 1.0, atol=1e-7)
    assert res.converged


def test_stress_sign_follows_rate():
    grid = Grid(sigma_max=8, n_per_unit=150)
    s_plus = solve_steady(0.3, 0.5, grid).stress
    s_minus = solve_steady(0.3, -0.5, grid).stress
    assert s_plus > 0 and s_minus < 0
    assert np.isclose(s_plus, -s_minus, rtol=1e-6)


def test_low_rate_floor_warns_and_flags():
    import warnings
    grid = Grid(sigma_max=10, n_per_unit=400)   # floor 4h^2 = 2.5e-5
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = solve_steady(0.4, 1e-6, grid)      # well below the floor
    assert res.converged is False
    assert any("resolution floor" in str(x.message) for x in w)


def test_high_rate_truncation_warns_and_flags():
    import warnings
    grid = Grid(sigma_max=10, n_per_unit=60)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = solve_steady(0.4, 100.0, grid, n_scan=40)  # needs sigma_max >> 100
    assert res.boundary_fraction > 1e-3
    assert any("truncated" in str(x.message) for x in w)


def test_well_resolved_run_is_silent():
    import warnings
    grid = Grid(sigma_max=20, n_per_unit=200)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = solve_steady(0.4, 1.0, grid)       # comfortably inside both limits
    assert res.converged and res.boundary_fraction < 1e-3
    assert len(w) == 0
