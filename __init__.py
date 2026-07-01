"""hlmodel -- numerical solution of the Hebraud-Lequeux (1998) model.

Reduced units throughout: tau = sigma_c = G0 = 1.  Critical coupling alpha_c =
1/2.  See ``theory.md`` for the governing equations and derivations, and
``README.md`` for a quickstart.

Solver map
----------
* steady shear   : ``solve_steady`` (direct stationary oracle) and the
                   ``TransientStepper`` IMEX engine -- they agree.
* SAOS           : ``saos_spectrum`` / ``saos_modulus`` -- one linear solve per
                   frequency (linearised about the liquid base state).
* LAOS           : ``solve_laos`` -- periodic-orbit fixed point of the period
                   map (Newton-Krylov or Picard).
* parallel sweeps: ``sweep.flow_curve``, ``sweep.saos_spectrum``,
                   ``sweep.laos_map``.
"""

from __future__ import annotations

from .grid import Grid
from .selfconsistency import ALPHA_C, D_quiescent, alpha_of_D_quiescent
from .operators import generator, build_static_parts
from .steady import SteadyResult, solve_steady, stationary_distribution
from .protocols import SteadyShear, Oscillatory
from .transient import TransientStepper, Trajectory, initial_delta
from .observables import (
    OscillatoryResponse,
    decompose_oscillatory,
    macroscopic_stress,
    yielding_rate,
)
from .saos import SAOSResult, saos_modulus, saos_spectrum, liquid_base_state
from .periodic import LAOSResult, PeriodMap, solve_laos

__all__ = [
    "Grid",
    "ALPHA_C",
    "D_quiescent",
    "alpha_of_D_quiescent",
    "generator",
    "build_static_parts",
    "SteadyResult",
    "solve_steady",
    "stationary_distribution",
    "SteadyShear",
    "Oscillatory",
    "TransientStepper",
    "Trajectory",
    "initial_delta",
    "OscillatoryResponse",
    "decompose_oscillatory",
    "macroscopic_stress",
    "yielding_rate",
    "SAOSResult",
    "saos_modulus",
    "saos_spectrum",
    "liquid_base_state",
    "LAOSResult",
    "PeriodMap",
    "solve_laos",
]

__version__ = "0.1.0"
