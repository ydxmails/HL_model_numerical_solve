"""Deformation protocols.

A protocol supplies the strain ``gamma(t)`` and strain rate ``gammadot(t)`` that
drive the Fokker-Planck equation.  The oscillatory protocol uses the package-wide
convention gamma(t) = gamma0 sin(omega t) with t = 0 pinned at gamma = 0, so that
SAOS and LAOS moduli are defined identically (see ``observables.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SteadyShear:
    """Constant shear rate; strain accumulates linearly."""

    gammadot_value: float

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

    def gamma(self, t):
        return self.gamma0 * np.sin(self.omega * t)

    def gammadot(self, t):
        return self.gamma0 * self.omega * np.cos(self.omega * t)
