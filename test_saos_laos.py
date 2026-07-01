"""SAOS and LAOS tests, including the SAOS<->LAOS alignment check."""

import numpy as np
import pytest

from hlmodel import Grid
from hlmodel.saos import saos_spectrum, saos_modulus
from hlmodel.periodic import solve_laos


def test_saos_maxwell_low_frequency():
    grid = Grid(sigma_max=10, n_per_unit=250)
    res = saos_spectrum(0.8, [1e-3, 3e-3, 1e-2], grid)
    # G' ~ omega^2 and G'' ~ omega: the reduced ratios are ~constant
    r_storage = res.Gp / res.omega**2
    r_loss = res.Gpp / res.omega
    assert np.all(res.Gpp > 0)                       # dissipative
    assert np.std(r_storage) / np.mean(r_storage) < 0.05
    assert np.std(r_loss) / np.mean(r_loss) < 0.05


def test_saos_high_frequency_plateau():
    grid = Grid(sigma_max=10, n_per_unit=250)
    Gp, Gpp = saos_modulus(0.8, 50.0, grid)
    assert np.isclose(Gp, 1.0, atol=0.02)            # plateau modulus G0 = 1
    assert Gpp < Gp                                   # mostly elastic at high w


def test_saos_requires_liquid_phase():
    grid = Grid(sigma_max=8, n_per_unit=100)
    with pytest.raises(ValueError):
        saos_modulus(0.4, 1.0, grid)                  # jammed: no unique base


def test_laos_saos_alignment():
    # First-harmonic LAOS moduli must approach SAOS moduli as gamma0 -> 0.
    grid = Grid(sigma_max=8, n_per_unit=150)
    alpha, omega = 0.8, 0.3
    Gp_s, Gpp_s = saos_modulus(alpha, omega, grid)
    # Use a fine timestep to suppress the first-order splitting error.
    res = solve_laos(alpha, 0.01, omega, grid, steps_per_period=1200,
                     method="picard", tol=1e-10)
    assert np.isclose(res.G1_prime, Gp_s, rtol=0.02)
    assert np.isclose(res.G1_doubleprime, Gpp_s, rtol=0.02)
    assert res.residual < 1e-6


def test_laos_odd_harmonics_only():
    grid = Grid(sigma_max=8, n_per_unit=150)
    res = solve_laos(0.8, 1.0, 0.3, grid, steps_per_period=400,
                     method="picard", tol=1e-9, n_harmonics=6)
    # half-period symmetry forces even harmonics to vanish
    assert res.response.even_harmonic_error < 1e-5
    # genuine third-harmonic generation at large amplitude
    assert res.response.intensity[2] > 1e-3
