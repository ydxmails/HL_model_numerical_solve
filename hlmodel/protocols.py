"""Deformation protocols.

A protocol supplies the strain ``gamma(t)`` and strain rate ``gammadot(t)`` that
drive the Fokker-Planck equation.  The oscillatory protocol uses the package-wide
convention gamma(t) = gamma0 sin(omega t) with t = 0 pinned at gamma = 0, so that
SAOS and LAOS moduli are defined identically (see ``observables.py``).

Only ``gammadot(t)`` enters the HL equation -- accumulated strain appears nowhere
in the dynamics -- so a protocol is, as far as the solvers are concerned, a scalar
function of time.  ``gamma(t)`` is carried for post-processing (affine-loading
subtraction in ``residence.py``, recoverable strain in ``recoverable.py``).

Every protocol also exposes two hints used by the periodic-orbit solver to pick a
starting guess:

* ``mean_rate``  -- the cycle-mean strain rate (nonzero => there is a steady
  drift, so the quiescent base state is a poor seed);
* ``seed_rate``  -- the peak |gammadot| over a cycle, the rate whose steady state
  best resembles the t = 0 state of the cycle (at t = 0 the strain is zero but
  the rate is maximal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SteadyShear:
    """Constant shear rate; strain accumulates linearly."""

    gammadot_value: float

    @property
    def mean_rate(self) -> float:
        return float(self.gammadot_value)

    @property
    def seed_rate(self) -> float:
        return abs(float(self.gammadot_value))

    def gamma(self, t):
        return self.gammadot_value * t

    def gammadot(self, t):
        return self.gammadot_value + 0.0 * np.asarray(t)


@dataclass(frozen=True)
class Oscillatory:
    """Sinusoidal strain gamma(t) = gamma0 sin(omega t)."""

    gamma0: float
    omega: float

    @property
    def period(self) -> float:
        return 2.0 * np.pi / self.omega

    @property
    def mean_rate(self) -> float:
        return 0.0

    @property
    def seed_rate(self) -> float:
        return abs(float(self.gamma0) * float(self.omega))

    def gamma(self, t):
        return self.gamma0 * np.sin(self.omega * t)

    def gammadot(self, t):
        return self.gamma0 * self.omega * np.cos(self.omega * t)


@dataclass(frozen=True)
class SuperposedShear:
    """Oscillation superposed *parallel* to steady shear (parallel superposition).

        gamma(t)    = gammadot0 * t  +  gammaA * sin(omega t)
        gammadot(t) = gammadot0      +  gammaA * omega * cos(omega t)

    Because the HL equation contains only ``gammadot`` -- never the accumulated
    strain -- the distribution P(sigma, t) still settles onto a genuine T-periodic
    limit cycle even though gamma(t) grows without bound.  The periodic-orbit
    machinery in ``periodic.py`` therefore applies unchanged.

    Note that the sine phase is retained from :class:`Oscillatory`, so the
    oscillatory part uses the same convention as SAOS/LAOS and the first-harmonic
    moduli remain comparable across all three protocols.

    Regimes are organised by

        Lambda = |gammaA * omega / gammadot0| ,

    the ratio of peak oscillatory rate to steady rate.  ``Lambda < 1`` merely
    pulses the flow (``gammadot`` keeps one sign); ``Lambda > 1`` reverses it
    within each cycle.  Measures that assume an unloading branch (e.g.
    ``recoverable.py``) are only meaningful for ``Lambda > 1``.

    The sine drive's half-period symmetry (sigma, t) -> (-sigma, t + T/2) requires
    gammadot(t + T/2) = -gammadot(t), which fails for any gammadot0 != 0.  Even
    harmonics of the stress are therefore *physical* here, not a numerical error
    diagnostic as they are in pure LAOS.
    """

    gammadot0: float
    gammaA: float
    omega: float

    @property
    def period(self) -> float:
        return 2.0 * np.pi / self.omega

    @property
    def Lambda(self) -> float:
        """Peak oscillatory rate / steady rate; > 1 means the flow reverses."""
        if self.gammadot0 == 0.0:
            return float("inf")
        return abs(self.gammaA * self.omega / self.gammadot0)

    @property
    def reverses(self) -> bool:
        """True if gammadot changes sign within a cycle."""
        return self.Lambda > 1.0

    @property
    def mean_rate(self) -> float:
        return float(self.gammadot0)

    @property
    def seed_rate(self) -> float:
        return abs(float(self.gammadot0)) + abs(float(self.gammaA) * float(self.omega))

    def gamma(self, t):
        t = np.asarray(t, dtype=float)
        return self.gammadot0 * t + self.gammaA * np.sin(self.omega * t)

    def gammadot(self, t):
        return self.gammadot0 + self.gammaA * self.omega * np.cos(self.omega * t)


def require_sine_drive(protocol, caller: str) -> None:
    """Raise unless ``protocol`` is a pure sinusoidal drive.

    Several cycle-analysis routines accept a precomputed ``LAOSResult`` and then
    rebuild the drive themselves from ``(gamma0, omega)`` -- constructing a fresh
    ``PeriodMap`` with no protocol, reconstructing ``gamma(t) = gamma0 sin(omega
    t)``, or assuming the T/2 half-period symmetry that only a pure sine has.
    That was safe while ``Oscillatory`` was the only drive a cycle could come
    from.  It stopped being safe the moment ``PeriodMap`` accepted ``protocol=``.

    Handing such a routine a superposed cycle makes it re-propagate a converged
    P0 under the *wrong* drive, silently: measured 2.1% on ``D_bar`` at
    Lambda = 5, with no warning and no exception.  Errors of that size read as
    discretisation artifacts but are large enough to move a conclusion, so these
    routines refuse rather than guess.

    ``None`` passes: a cycle with no recorded protocol predates the field, and
    could only have come from the default sine drive.
    """
    if protocol is None or isinstance(protocol, Oscillatory):
        return
    name = type(protocol).__name__
    raise NotImplementedError(
        f"{caller}() rebuilds the drive from (gamma0, omega) and assumes a pure "
        f"sine, but this cycle was produced with {name}. It would re-propagate "
        f"the converged state under the wrong strain rate and return a plausible "
        f"but wrong answer. Generalising it needs the protocol threaded through "
        f"(and, for recoverable strain, turning points at cos(omega t) = -1/Lambda "
        f"rather than T/4 and 3T/4)."
    )
