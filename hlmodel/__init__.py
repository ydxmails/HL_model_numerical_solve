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
from .protocols import (SteadyShear, Oscillatory, SuperposedShear,
                        require_sine_drive)
from .superposed import SuperposedResult, solve_superposed
from .transient import TransientStepper, Trajectory, initial_delta
from .observables import (
    OscillatoryResponse,
    decompose_oscillatory,
    macroscopic_stress,
    yielding_rate,
    YieldStressDistribution,
    yield_stress_distribution,
)
from .saos import (SAOSResult, saos_modulus, saos_spectrum, liquid_base_state,
                   PSRResult, psr_modulus, psr_spectrum)
from .periodic import (
    LAOSResult,
    PeriodMap,
    solve_laos,
    CycleYieldDistribution,
    cycle_yield_stress_distribution,
)
from .residence import (
    ResidenceTimeDistribution,
    residence_time_distribution,
    CycleResidenceDistribution,
    cycle_residence_time_distribution,
    NonAffineStressTrajectory,
    nonaffine_stress_trajectory,
    CycleNonAffineStressTrajectory,
    cycle_nonaffine_stress_trajectory,
    YieldPhaseDistribution,
    cycle_yield_phase_distribution,
)

from .recovery import (
    RecoveryResult,
    CycleRecovery,
    affine_shift,
    plastic_stress,
    recover_from_state,
    steady_recovery,
    laos_recovery,
)

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
    "SuperposedShear",
    "require_sine_drive",
    "SuperposedResult",
    "solve_superposed",
    "TransientStepper",
    "Trajectory",
    "initial_delta",
    "OscillatoryResponse",
    "decompose_oscillatory",
    "macroscopic_stress",
    "yielding_rate",
    "YieldStressDistribution",
    "yield_stress_distribution",
    "SAOSResult",
    "saos_modulus",
    "saos_spectrum",
    "liquid_base_state",
    "PSRResult",
    "psr_modulus",
    "psr_spectrum",
    "LAOSResult",
    "PeriodMap",
    "solve_laos",
    "CycleYieldDistribution",
    "cycle_yield_stress_distribution",
    "ResidenceTimeDistribution",
    "residence_time_distribution",
    "CycleResidenceDistribution",
    "cycle_residence_time_distribution",
    "NonAffineStressTrajectory",
    "nonaffine_stress_trajectory",
    "CycleNonAffineStressTrajectory",
    "cycle_nonaffine_stress_trajectory",
    "YieldPhaseDistribution",
    "cycle_yield_phase_distribution",
    "RecoveryResult",
    "CycleRecovery",
    "affine_shift",
    "plastic_stress",
    "recover_from_state",
    "steady_recovery",
    "laos_recovery",
]

__version__ = "0.1.0"
