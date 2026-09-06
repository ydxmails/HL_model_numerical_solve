"""Tests for the residence-time (age-at-yield) distribution."""

import numpy as np
import pytest

from hlmodel import (Grid, solve_steady, residence_time_distribution,
                     cycle_residence_time_distribution)


def _integral(y, x):
    return float(np.sum(0.5 * (y[:-1] + y[1:])) * (x[1] - x[0]))


@pytest.mark.parametrize("alpha,gd", [(0.8, 0.0), (0.8, 1.0), (0.4, 0.5)])
def test_steady_mean_is_inverse_gamma(alpha, gd):
    # Exact: <a> = 1/Gamma (Little's law).  Also psi is a normalised density.
    grid = Grid(sigma_max=10, n_per_unit=150)
    r = residence_time_distribution(alpha, gd, grid, da=0.04, n_mean=15)
    assert r.defined
    assert np.isclose(r.mean, 1.0 / r.Gamma, rtol=0.02)
    assert np.isclose(r.mean, r.mean_exact, rtol=0.02)
    assert r.captured_fraction > 0.99
    assert np.isclose(_integral(r.psi, r.a), 1.0, atol=0.02)


def test_steady_state_is_age_integral_of_cohort():
    # P(sigma) = Gamma * integral_0^inf Q(sigma,a) da
    grid = Grid(sigma_max=10, n_per_unit=150)
    r = residence_time_distribution(0.8, 0.5, grid, da=0.03, n_mean=16)
    P = solve_steady(0.8, 0.5, grid).P
    assert np.max(np.abs(r.Gamma * r.occupation - P)) < 1e-3


def test_high_rate_tail_is_the_yielding_rate():
    # At high shear the residence tail decays at the bare yielding rate 1/tau = 1.
    grid = Grid(sigma_max=12, n_per_unit=150)
    r = residence_time_distribution(0.8, 3.0, grid, da=0.03, n_mean=14,
                                    compute_gap=True)
    assert r.spectral_gap is not None
    assert np.isclose(r.spectral_gap, 1.0, rtol=0.05)


def test_jammed_quiescent_residence_is_infinite():
    # Jammed phase at rest: D = Gamma = 0, blocks never yield -> mean = inf,
    # and the function warns rather than returning a meaningless curve.
    import warnings
    grid = Grid(sigma_max=8, n_per_unit=120)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        r = residence_time_distribution(0.4, 0.0, grid)
    assert not r.defined
    assert r.mean_exact == float("inf")
    assert any("does not yield" in str(w.message) for w in ws)


def test_oscillatory_sub_yield_is_flagged():
    # Jammed alpha with amplitude below the yield strain (~sigma_y~0.15): the
    # material barely yields, so the residence distribution is ill-defined and the
    # function flags it (defined=False) with a warning instead of a capped curve.
    import warnings
    grid = Grid(sigma_max=5.0, n_per_unit=150)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        cyd = cycle_residence_time_distribution(
            0.4, 0.05, 0.1, grid, n_birth_phases=4, da=0.1, a_max=50.0,
            steps_per_period=300)
    assert not cyd.defined
    assert cyd.captured_fraction < 0.01           # essentially nothing yields
    assert any("barely yields" in str(w.message) for w in ws)


def test_oscillatory_mean_and_birth_phase_dependence():
    # Phase-averaged mean = 1/<Gamma>; residence depends on birth phase.
    grid = Grid(sigma_max=8, n_per_unit=120)
    cyd = cycle_residence_time_distribution(
        0.8, 2.0, 0.5, grid, n_birth_phases=6, da=0.06, n_mean=7,
        steps_per_period=300)
    assert np.isclose(cyd.mean, 1.0 / cyd.Gamma_mean, rtol=0.03)
    # born at maximum strain rate (t0/T ~ 0) yields sooner than the cycle average
    i0 = int(np.argmin(np.abs(cyd.birth_phase - 0.0)))
    assert cyd.mean_by_phase[i0] < cyd.mean
    # genuine spread across birth phases
    assert cyd.mean_by_phase.max() > 1.1 * cyd.mean_by_phase.min()


# ---------------------------------------------------------------------------
# Non-affine stress trajectory
# ---------------------------------------------------------------------------
from hlmodel import (nonaffine_stress_trajectory,          # noqa: E402
                     cycle_nonaffine_stress_trajectory)


@pytest.mark.parametrize("gd", [0.3, 1.0])
def test_nonaffine_early_msd_slope_is_2D(gd):
    # Before absorption the element random-walks freely: MSD_na(a) ~ 2 D a.
    grid = Grid(sigma_max=6, n_per_unit=200)
    nat = nonaffine_stress_trajectory(0.7, gd, grid, da=0.01, n_mean=20)
    assert nat.defined
    k = int(np.argmin(np.abs(nat.a - 0.06)))
    slope = nat.nonaffine_msd[k] / nat.a[k]
    assert np.isclose(slope, 2.0 * nat.D, rtol=0.05)
    # starts exactly at zero
    assert abs(nat.nonaffine_mean[0]) < 1e-12 and abs(nat.nonaffine_msd[0]) < 1e-12


def test_nonaffine_mean_stress_matches_macroscopic_stress():
    # Since P = Gamma * int Q da, the macroscopic stress is Sigma = Gamma * int m1 da,
    # with m1(a) = <sigma>_surv(a) * S(a).  This ties the non-affine trajectory to
    # the steady stress (validation chain).
    grid = Grid(sigma_max=8, n_per_unit=150)
    nat = nonaffine_stress_trajectory(0.7, 0.5, grid, da=0.02, n_mean=25)
    Sigma_direct = solve_steady(0.7, 0.5, grid).stress
    Sigma_cohort = nat.Gamma * _integral(nat.survivor_mean * nat.survival, nat.a)
    assert np.isclose(Sigma_cohort, Sigma_direct, rtol=1e-2)


def test_nonaffine_mean_bends_below_affine():
    # The survivor mean tracks the affine line at first, then bends below it
    # (yielding preferentially removes the high-stress tail), so the non-affine
    # mean displacement is <= 0 and clearly negative by the end of the window.
    grid = Grid(sigma_max=6, n_per_unit=200)
    nat = nonaffine_stress_trajectory(0.7, 0.8, grid, da=0.02, n_mean=15)
    assert nat.nonaffine_mean.max() < 1e-6            # never rises above affine
    assert nat.nonaffine_mean[-1] < -0.1              # bent well below by late age


def test_nonaffine_jammed_frozen_undefined():
    # Jammed phase at rest: no yielding, no absorption -> trivial non-affine motion.
    import warnings
    grid = Grid(sigma_max=8, n_per_unit=120)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        nat = nonaffine_stress_trajectory(0.4, 0.0, grid)
    assert not nat.defined
    assert any("does not yield" in str(w.message) for w in ws)


def test_cycle_nonaffine_halfperiod_symmetry():
    # Sine drive symmetry (sigma,t)->(-sigma,t+T/2): cohorts born at t0 and t0+T/2
    # are mirror images, so their non-affine MSDs coincide and their non-affine
    # means are opposite.  The affine part is removed by the co-deforming transform.
    grid = Grid(sigma_max=6, n_per_unit=150)
    c = cycle_nonaffine_stress_trajectory(0.7, 1.0, 0.5, grid, n_birth_phases=8,
                                          da=0.02, n_mean=6)
    half = len(c.birth_phase) // 2
    assert np.max(np.abs(c.nonaffine_msd[:half] - c.nonaffine_msd[half:])) < 1e-6
    assert np.max(np.abs(c.nonaffine_mean[:half] + c.nonaffine_mean[half:])) < 1e-6
    # every cohort starts at sigma = 0 with zero non-affine displacement
    assert np.max(np.abs(c.nonaffine_mean[:, 0])) < 1e-12
    assert np.max(np.abs(c.nonaffine_msd[:, 0])) < 1e-12


# ---------------------------------------------------------------------------
# Yield-phase distribution
# ---------------------------------------------------------------------------
from hlmodel import cycle_yield_phase_distribution              # noqa: E402

_YPD_CACHE = {}


def _ypd():
    """Compute one yield-phase distribution, cached across the tests below."""
    if "r" not in _YPD_CACHE:
        grid = Grid(sigma_max=5, n_per_unit=120)
        _YPD_CACHE["r"] = cycle_yield_phase_distribution(
            0.4, 0.3, 0.2, grid, n_birth_phases=16, n_phase=120, da=0.1,
            n_mean=6, a_max_cap=2000, steps_per_period=200, warn=False)
    return _YPD_CACHE["r"]


def test_yield_phase_aggregate_matches_instantaneous_rate():
    # Summed over birth phase (weighted by the rebirth rate), blocks must yield at
    # the instantaneous yield rate Gamma(phi): the folded aggregate == yield_rate.
    c = _ypd()
    assert c.defined
    assert np.max(np.abs(c.rho - c.yield_rate)) / c.yield_rate.max() < 0.06


def test_yield_phase_densities_normalised():
    # Every density integrates to 1 over phase in [0,1).
    c = _ypd(); n = c.phase.size
    assert np.isclose(c.rho.sum() / n, 1.0, atol=1e-6)
    assert np.isclose(c.yield_rate.sum() / n, 1.0, atol=1e-6)
    assert np.allclose(c.rho_by_phase.sum(axis=1) / n, 1.0, atol=1e-6)


def test_yield_phase_halfperiod_symmetry():
    # Sine drive => Gamma has period T/2 => yield_rate invariant under phi->phi+T/2.
    c = _ypd(); n = c.phase.size
    shifted = np.roll(c.yield_rate, n // 2)
    assert np.max(np.abs(c.yield_rate - shifted)) / c.yield_rate.max() < 0.03
