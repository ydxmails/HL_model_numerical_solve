"""Parallel superposition: oscillation collinear with steady shear.

    gamma(t)    = gammadot0 * t  +  gammaA * sin(omega t)
    gammadot(t) = gammadot0      +  gammaA * omega * cos(omega t)

This is the collinear protocol, the one a scalar model can express.  Orthogonal
superposition cannot be represented here: an orthogonal probe perturbs |gammadot|
only in quadrature, i.e. at O(gammaA^2), whereas a collinear one perturbs it at
O(gammaA).  Since the HL closure routes everything through the single scalar
Gamma, the oscillation modulates the noise D = alpha Gamma at first order.  There
is therefore no passive probe here: what is measured is always a *fluidised*
state, never the steady state at gammadot0.  That is the physics, not an artifact
-- it is the model's rejuvenation mechanism -- but it decides how the protocol
should be read.

Regimes are organised by

    Lambda = |gammaA omega / gammadot0| ,

the peak oscillatory rate over the steady rate, *not* by gammaA alone.
Lambda < 1 pulses the flow without reversing it; Lambda > 1 reverses it within
each cycle.  In reduced units gammadot0, omega and gammaA are already
dimensionless (the yield strain is 1), so all three are independent knobs, but
Lambda is what separates the regimes.

Two observables, with very different numerical character:

* **The cycle-mean stress** ``mean_stress``.  Its offset from the unperturbed
  steady stress is the oscillation-induced thinning, O(gammaA^2) and negative.
  It is essentially dt-independent -- measured to move by <2% across a 4x change
  in dt -- because the mass-conserving IMEX scheme reproduces cycle averages far
  better than instantaneous waveforms.  The fluidisation map is therefore cheap.
* **The first-harmonic moduli**.  These carry the integrator's O(dt) error and
  need genuine resolution.  Where the linear (gammaA -> 0) answer is wanted,
  ``saos.psr_modulus`` gives it exactly for two banded solves, with no time
  stepping at all.

Because the drift breaks the half-period symmetry (gammadot(t + T/2) = -gammadot(t)
requires gammadot0 = 0), even harmonics are *physical* here.  The package's usual
free error check -- even-harmonic content at the 1e-8 noise floor -- is therefore
unavailable, and ``even_harmonic_error`` should be read as signal.  Use
Richardson extrapolation in dt and the Floquet multiplier instead.

One practical bonus: superposition is much better conditioned than pure LAOS.
The drift keeps the material fluidised, so the dominant Floquet multiplier is
small (~4e-3 at Lambda = 5, alpha = 0.4) and the fixed point converges in ~20
period solves, rather than suffering the critical slowing down that afflicts
small-amplitude LAOS in the jammed phase.

>>> r = solve_superposed(alpha=0.4, gammadot0=0.1, gammaA=0.2, omega=0.5)  # doctest: +SKIP
>>> r.Lambda, r.reverses, r.thinning                                       # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import Grid
from .observables import OscillatoryResponse
from .operators import build_static_parts
from .periodic import LAOSResult, solve_laos
from .protocols import SuperposedShear
from .saos import psr_modulus
from .steady import solve_steady


@dataclass
class SuperposedResult:
    """One converged superposed limit cycle."""

    alpha: float
    gammadot0: float
    gammaA: float
    omega: float

    response: OscillatoryResponse = field(repr=False)
    P0: np.ndarray = field(repr=False)

    steady_stress: float = 0.0      # unperturbed Sigma(gammadot0), no oscillation
    Gp_linear: float = float("nan")  # psr_modulus: the exact gammaA -> 0 moduli
    Gpp_linear: float = float("nan")

    residual: float = 0.0
    converged: bool = True
    floquet: float | None = None
    peak_cfl: float = 0.0
    n_period_solves: int = 0
    cycle: LAOSResult | None = field(default=None, repr=False)

    # -- regime ------------------------------------------------------------
    @property
    def Lambda(self) -> float:
        """Peak oscillatory rate / steady rate.  > 1 means the flow reverses."""
        if self.gammadot0 == 0.0:
            return float("inf")
        return abs(self.gammaA * self.omega / self.gammadot0)

    @property
    def reverses(self) -> bool:
        return self.Lambda > 1.0

    # -- stress ------------------------------------------------------------
    @property
    def mean_stress(self) -> float:
        """Cycle-mean shear stress (the n = 0 Fourier coefficient)."""
        return float(self.response.dc)

    @property
    def thinning(self) -> float:
        """``mean_stress - steady_stress``.

        Negative: the oscillation fluidises.  O(gammaA^2) at small amplitude, and
        robust to dt -- see the module notes.
        """
        return self.mean_stress - self.steady_stress

    # -- moduli ------------------------------------------------------------
    @property
    def G1_prime(self) -> float:
        return float(self.response.Gp[0])

    @property
    def G1_doubleprime(self) -> float:
        return float(self.response.Gpp[0])

    @property
    def linear_offset(self) -> tuple[float, float]:
        """First-harmonic moduli minus the exact linear (gammaA -> 0) answer.

        Two contributions, with different scalings: the integrator's O(dt) error
        and the model's own O(gammaA^2) nonlinearity.  Halving dt halves the
        first; halving gammaA quarters the second.  A residual that refuses to
        shrink under *either* refinement means something else is wrong.
        """
        return (self.G1_prime - self.Gp_linear,
                self.G1_doubleprime - self.Gpp_linear)

    @property
    def harmonic_intensity(self) -> np.ndarray:
        """|G_n*| / |G_1*|.  Even entries are physical here, not error."""
        return self.response.intensity


def solve_superposed(alpha: float, gammadot0: float, gammaA: float, omega: float,
                     grid: Grid | None = None, parts: dict | None = None,
                     with_linear: bool = True, **solver_opts) -> SuperposedResult:
    """Converge the superposed limit cycle and package the observables.

    Thin wrapper over :func:`~hlmodel.periodic.solve_laos` with a
    :class:`~hlmodel.protocols.SuperposedShear` drive.  The seeding is handled
    upstream: because ``SuperposedShear.mean_rate`` is nonzero, ``_default_seed``
    starts from the sheared steady state rather than the quiescent one, which is
    what makes this converge quickly in both phases.

    ``with_linear`` also evaluates ``psr_modulus`` at the same point (two banded
    solves, negligible next to the cycle solve) so every result carries its own
    linear reference.  See :attr:`SuperposedResult.linear_offset`.
    """
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)

    protocol = SuperposedShear(float(gammadot0), float(gammaA), float(omega))
    cycle = solve_laos(alpha, float(gammaA), float(omega), grid, parts,
                       protocol=protocol, **solver_opts)

    steady = solve_steady(alpha, abs(float(gammadot0)), grid, parts, warn=False)
    if with_linear and gammadot0 != 0.0:
        Gp_lin, Gpp_lin = psr_modulus(alpha, gammadot0, omega, grid, parts,
                                      base=steady)
    else:
        Gp_lin = Gpp_lin = float("nan")

    return SuperposedResult(
        alpha=float(alpha), gammadot0=float(gammadot0), gammaA=float(gammaA),
        omega=float(omega), response=cycle.response, P0=cycle.P0,
        steady_stress=float(steady.stress), Gp_linear=Gp_lin, Gpp_linear=Gpp_lin,
        residual=float(cycle.residual), converged=bool(cycle.converged),
        floquet=cycle.floquet, peak_cfl=float(cycle.peak_cfl),
        n_period_solves=int(cycle.n_period_solves), cycle=cycle,
    )
