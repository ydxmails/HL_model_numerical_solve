"""Parallel superposition: oscillation collinear with steady shear.

The validation chain mirrors the package's existing one -- an exact identity, a
limit that must reproduce already-validated code, and two independent solvers
that must agree:

* the linear perturbation carries no mass (exact: 1^T G_lin = 0 and the forcing
  is a flux divergence, so i omega 1^T p1 = 0);
* psr_modulus -> saos_modulus as gammadot0 -> 0, quadratically (the moduli are
  even in gammadot0 by the sigma -> -sigma symmetry);
* the nonlinear periodic-orbit solver -> psr_modulus as gammaA -> 0 and dt -> 0;
* the cycle-mean stress -> the steady stress at gammadot0.
"""

import numpy as np
import pytest

from hlmodel import Grid, solve_steady, SuperposedShear
from hlmodel.saos import saos_modulus, psr_modulus, psr_spectrum
from hlmodel.periodic import solve_laos, PeriodMap


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------
def test_superposed_protocol_algebra():
    p = SuperposedShear(gammadot0=0.1, gammaA=0.2, omega=0.5)
    assert np.isclose(p.period, 2.0 * np.pi / 0.5)
    assert np.isclose(p.gammadot(0.0), 0.1 + 0.2 * 0.5)
    assert np.isclose(p.mean_rate, 0.1)
    # Lambda = 1 grazes zero rate without crossing it
    assert np.isclose(p.Lambda, 1.0) and not p.reverses
    assert SuperposedShear(0.1, 1.0, 0.5).reverses
    # cycle-mean rate really is gammadot0
    t = np.linspace(0.0, p.period, 4096, endpoint=False)
    assert np.isclose(p.gammadot(t).mean(), 0.1, atol=1e-12)


def test_record_stride_divides_step_count():
    """Regression: a stride that does not divide n_steps aliases the projection."""
    g = Grid(sigma_max=8, n_per_unit=50)
    for spp in (256, 400, 800, 1600, 3200, 6400):
        pm = PeriodMap(0.8, 1.0, 0.3, g, steps_per_period=spp)
        for n_sample in (64, 128, 256, 400):
            rec = pm._record_stride(n_sample)
            assert rec >= 1 and pm.n_steps % rec == 0


# --------------------------------------------------------------------------
# linear oracle
# --------------------------------------------------------------------------
@pytest.mark.parametrize("alpha,gd,w", [(0.4, 0.1, 0.5), (0.8, 0.5, 1.0), (0.3, 0.01, 0.05)])
def test_perturbation_is_mass_neutral(alpha, gd, w):
    g = Grid(sigma_max=8, n_per_unit=150)
    _, _, p1 = psr_modulus(alpha, gd, w, g, return_p1=True)
    assert abs(complex(g.integrate(p1))) < 1e-11


def test_psr_requires_nonzero_shear_rate():
    g = Grid(sigma_max=8, n_per_unit=50)
    with pytest.raises(ValueError):
        psr_modulus(0.8, 0.0, 0.5, g)


def test_psr_moduli_are_even_in_shear_rate():
    g = Grid(sigma_max=8, n_per_unit=150)
    a = psr_modulus(0.8, 0.3, 0.7, g)
    b = psr_modulus(0.8, -0.3, 0.7, g)
    assert np.allclose(a, b, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("w", [0.1, 0.5, 2.0])
def test_psr_reduces_to_saos_at_zero_rate(w):
    """gammadot0 -> 0 must recover SAOS, and quadratically (moduli are even)."""
    g = Grid(sigma_max=8, n_per_unit=200)
    alpha = 0.8
    Gp_s, Gpp_s = saos_modulus(alpha, w, g)
    errs = []
    for gd in (3e-3, 1e-3):
        Gp, Gpp = psr_modulus(alpha, gd, w, g)
        errs.append(abs(Gp - Gp_s) + abs(Gpp - Gpp_s))
    assert errs[-1] < 2e-4                      # converged
    assert errs[0] / errs[1] > 5.0              # faster than linear => quadratic


def test_negative_storage_modulus_in_the_thinning_regime():
    """G'_parallel goes negative at low frequency deep in the jammed phase.

    This is the classic parallel-superposition pathology.  Here it comes out of an
    exact linear-response solve about a strongly shear-thinning steady state, so
    it is a property of the model, not of the measurement.
    """
    g = Grid(sigma_max=8, n_per_unit=200)
    sp = psr_spectrum(0.3, 0.01, np.logspace(-2.2, 0.5, 12), g)
    assert (sp.Gp < 0).any()
    assert sp.Gp[-1] > 0                        # positive at high frequency
    assert np.all(sp.Gpp > 0)                   # loss modulus stays dissipative


# --------------------------------------------------------------------------
# nonlinear solver vs the oracle
# --------------------------------------------------------------------------
# Kept on a coarse grid deliberately: every assertion below is a *scaling*, and
# the scalings are clean at n_per_unit = 100.  Note that superposition is far
# better conditioned than pure small-amplitude LAOS -- the drift keeps the
# material fluidised, so the dominant Floquet multiplier is ~4e-3 here and the
# fixed point converges in ~20 period solves, versus the critical slowing down
# that afflicts a pure sine drive at the same amplitude in the jammed phase.
ALPHA, GDOT0, OMEGA = 0.4, 0.1, 0.5


def _superposed(gammaA, spp, grid, floquet=False):
    return solve_laos(ALPHA, gammaA, OMEGA, grid,
                      protocol=SuperposedShear(GDOT0, gammaA, OMEGA),
                      steps_per_period=spp, method="newton_krylov",
                      tol=1e-10, n_harmonics=5, floquet=floquet, warn=False)


def test_cycle_mean_stress_recovers_the_steady_stress():
    """The DC term is the steady stress plus an O(gammaA^2) thinning correction.

    The offset is essentially dt-independent (it moves by <2% across a 4x change
    in dt), so the fluidisation map is cheap even where the moduli are not.
    """
    g = Grid(sigma_max=8, n_per_unit=100)
    Sigma_s = solve_steady(ALPHA, GDOT0, g, warn=False).stress
    d = []
    for gammaA in (0.01, 0.005):
        r = _superposed(gammaA, 512, g)
        assert r.converged
        assert r.response.dc < Sigma_s          # oscillation thins, never thickens
        d.append(abs(r.response.dc - Sigma_s))
    assert d[0] < 1e-3
    assert 3.4 < d[0] / d[1] < 4.6              # quadratic in gammaA


def test_laos_converges_to_the_linear_oracle():
    """First-harmonic moduli -> psr_modulus, at first order in dt."""
    g = Grid(sigma_max=8, n_per_unit=100)
    Gp0, _ = psr_modulus(ALPHA, GDOT0, OMEGA, g)
    err = [abs(_superposed(0.01, spp, g).G1_prime - Gp0)
           for spp in (256, 512, 1024)]
    assert err[-1] < 4e-3
    for lo, hi in zip(err[1:], err[:-1]):
        assert 1.8 < hi / lo < 2.3              # halves with dt


def test_cycle_is_stable_and_well_conditioned():
    """|mu_max| < 1 certifies the limit cycle; here it is very small."""
    g = Grid(sigma_max=8, n_per_unit=100)
    r = _superposed(0.01, 512, g, floquet=True)
    assert r.floquet is not None
    assert 0.0 < r.floquet < 1.0


def test_drift_makes_even_harmonics_physical():
    """Half-period symmetry is broken by gammadot0, so even harmonics are signal.

    Normalised by gammaA they are O(gammaA) -- i.e. O(gammaA^2) in stress -- which
    is what separates them from a dt-independent numerical artifact.  A pure sine
    drive at comparable settings stays at the 1e-8 noise floor.
    """
    g = Grid(sigma_max=8, n_per_unit=100)
    I2 = [_superposed(gammaA, 512, g).response.intensity[1]
          for gammaA in (0.02, 0.005)]
    assert min(I2) > 1e-4                       # far above the noise floor
    assert 3.4 < I2[0] / I2[1] < 4.6            # linear in gammaA (4x amplitude)

    pure = solve_laos(0.8, 0.1, OMEGA, g, method="picard", steps_per_period=512,
                      tol=1e-10, n_harmonics=5, warn=False)
    assert pure.response.even_harmonic_error < 1e-6
    assert abs(pure.response.dc) < 1e-6         # no drift => no DC stress


# --------------------------------------------------------------------------
# containment: cycle-analysis routines must refuse a drive they cannot handle
# --------------------------------------------------------------------------
from hlmodel.cyclic_noise import cycle_mean_noise
from hlmodel.recoverable import laos_recoverable_fraction
from hlmodel.residence import (cycle_residence_time_distribution,
                               cycle_nonaffine_stress_trajectory,
                               cycle_yield_phase_distribution)
from hlmodel.protocols import Oscillatory, require_sine_drive


def test_cycle_records_the_protocol_it_was_made_with():
    g = Grid(sigma_max=8, n_per_unit=50)
    pure = solve_laos(0.8, 0.5, 1.0, g, steps_per_period=256, method="picard",
                      warn=False)
    sup = solve_laos(0.8, 0.5, 1.0, g, protocol=SuperposedShear(0.2, 0.5, 1.0),
                     steps_per_period=256, method="picard", warn=False)
    assert isinstance(pure.protocol, Oscillatory)
    assert isinstance(sup.protocol, SuperposedShear)


def test_require_sine_drive_admits_only_a_sine():
    require_sine_drive(None, "t")                    # legacy result, no field
    require_sine_drive(Oscillatory(1.0, 0.5), "t")
    with pytest.raises(NotImplementedError):
        require_sine_drive(SuperposedShear(0.1, 1.0, 0.5), "t")


def test_analysis_routines_refuse_a_superposed_cycle():
    """These rebuild the drive from (gamma0, omega); silence here would be a
    plausible-looking ~2% error rather than an obvious failure."""
    g = Grid(sigma_max=8, n_per_unit=50)
    sup = solve_laos(0.8, 0.5, 1.0, g, protocol=SuperposedShear(0.2, 0.5, 1.0),
                     steps_per_period=256, method="picard", warn=False)
    for fn, kw in [(cycle_mean_noise, dict(laos_result=sup)),
                   (laos_recoverable_fraction, dict(laos_result=sup)),
                   (cycle_residence_time_distribution, dict(cycle=sup)),
                   (cycle_nonaffine_stress_trajectory, dict(cycle=sup)),
                   (cycle_yield_phase_distribution, dict(cycle=sup))]:
        with pytest.raises(NotImplementedError):
            fn(0.8, 0.5, 1.0, g, **kw)


def test_pure_sine_analysis_still_works():
    """The guard must not disturb the path it was added to protect."""
    g = Grid(sigma_max=8, n_per_unit=50)
    pure = solve_laos(0.8, 0.5, 1.0, g, steps_per_period=256, method="picard",
                      warn=False)
    cn = cycle_mean_noise(0.8, 0.5, 1.0, g, laos_result=pure, steps_per_period=256)
    assert cn.D_bar > 0.0


# --------------------------------------------------------------------------
# high-level driver and sweeps
# --------------------------------------------------------------------------
from hlmodel import solve_superposed
from hlmodel.sweep import psr_spectrum_sweep, superposed_map


def test_solve_superposed_matches_the_raw_path():
    g = Grid(sigma_max=8, n_per_unit=100)
    r = solve_superposed(ALPHA, GDOT0, 0.01, OMEGA, g, steps_per_period=512,
                         method="newton_krylov", tol=1e-10)
    raw = _superposed(0.01, 512, g)
    assert np.isclose(r.G1_prime, raw.G1_prime, rtol=1e-12)
    assert np.isclose(r.mean_stress, raw.response.dc, rtol=1e-12)
    assert np.isclose(r.steady_stress,
                      solve_steady(ALPHA, GDOT0, g, warn=False).stress, rtol=1e-12)
    assert r.thinning < 0.0                      # oscillation fluidises
    assert np.isclose(r.Lambda, 0.01 * OMEGA / GDOT0)
    assert not r.reverses
    assert np.isfinite(r.Gp_linear)


def test_superposed_result_regime_flags():
    g = Grid(sigma_max=8, n_per_unit=50)
    r = solve_superposed(0.8, 0.1, 1.0, 0.5, g, steps_per_period=256,
                         method="picard", tol=1e-9, with_linear=False)
    assert np.isclose(r.Lambda, 5.0)
    assert r.reverses
    assert np.isnan(r.Gp_linear)


def test_psr_spectrum_sweep_matches_serial():
    g = Grid(sigma_max=8, n_per_unit=150)
    ws = np.array([0.05, 0.3, 1.5])
    sp = psr_spectrum_sweep(0.3, 0.05, ws, g, n_workers=1)
    for k, w in enumerate(ws):
        Gp, Gpp = psr_modulus(0.3, 0.05, float(w), g)
        assert np.isclose(sp.Gp[k], Gp, rtol=1e-12)
        assert np.isclose(sp.Gpp[k], Gpp, rtol=1e-12)
    assert sp.negative_Gp.dtype == bool


def test_superposed_map_shapes_and_thinning():
    g = Grid(sigma_max=8, n_per_unit=50)
    gAs, ws = np.array([0.05, 0.5]), np.array([0.3, 0.6])
    m = superposed_map(0.8, 0.1, gAs, ws, g, n_workers=1, steps_per_period=256,
                       method="picard", tol=1e-9)
    assert m.mean_stress.shape == (2, 2)
    assert np.allclose(m.Lambda, np.abs(np.outer(gAs, ws) / 0.1))
    assert np.all(m.thinning <= 0.0)             # never thickens
    # thinning grows with amplitude at fixed frequency
    assert np.all(m.thinning[1, :] < m.thinning[0, :])
