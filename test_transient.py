"""Transient-integrator tests: conservation and agreement with the oracle."""

import numpy as np
import pytest

from hlmodel import Grid, solve_steady
from hlmodel.transient import TransientStepper, initial_delta
from hlmodel.protocols import SteadyShear, Oscillatory


@pytest.mark.parametrize("alpha,gd", [(0.3, 0.5), (0.45, 0.2), (0.6, 0.5)])
def test_transient_relaxes_to_oracle(alpha, gd):
    grid = Grid(sigma_max=8, n_per_unit=150)
    oracle = solve_steady(alpha, gd, grid)
    step = TransientStepper(alpha, grid, theta=0.5, picard_iters=1)
    traj = step.run(initial_delta(grid), SteadyShear(gd).gammadot,
                    t_end=50.0, dt=0.02, record_every=500)
    assert np.isclose(traj.stress[-1], oracle.stress, rtol=1e-4)
    assert np.isclose(traj.D[-1], oracle.D, rtol=1e-4)


def test_mass_conserved_steady():
    grid = Grid(sigma_max=8, n_per_unit=150)
    step = TransientStepper(0.4, grid, theta=1.0)
    traj = step.run(initial_delta(grid), SteadyShear(0.7).gammadot,
                    t_end=20.0, dt=0.01, record_every=200)
    masses = [grid.integrate(traj.P_final)]
    assert np.allclose(masses, 1.0, atol=1e-7)


def test_mass_conserved_oscillatory():
    grid = Grid(sigma_max=8, n_per_unit=150)
    step = TransientStepper(0.6, grid, theta=0.5)
    prot = Oscillatory(gamma0=1.0, omega=1.0)
    P = initial_delta(grid)
    dt = prot.period / 200
    n = int(round(5 * prot.period / dt))
    for k in range(n):
        P, _ = step.step(P, prot.gammadot((k + 1) * dt), dt)
    assert np.isclose(grid.integrate(P), 1.0, atol=1e-7)
