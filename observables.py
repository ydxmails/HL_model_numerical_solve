"""Macroscopic observables and oscillatory-response analysis.

The strain convention is fixed package-wide as

    gamma(t)     = gamma0 * sin(omega t)
    gammadot(t)  = gamma0 * omega * cos(omega t)

with t = 0 pinned at gamma = 0.  Under this convention the periodic stress is

    sigma(t) = gamma0 * sum_n [ G_n' sin(n omega t) + G_n'' cos(n omega t) ]

so the first-harmonic moduli G_1', G_1'' are defined *identically* to the SAOS
moduli G', G''.  That is what makes the SAOS <-> LAOS limit
G_1'(gamma0 -> 0) -> G'(omega) a clean numerical check rather than a convention
juggling exercise.  The half-period symmetry of the model forces even harmonics
to vanish, so their residual amplitude is a free error diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid


def macroscopic_stress(P: np.ndarray, grid: Grid) -> float:
    return float(grid.first_moment(P))


def yielding_rate(P: np.ndarray, grid: Grid) -> float:
    return float(grid.yield_fraction(P))


@dataclass
class OscillatoryResponse:
    omega: float
    gamma0: float
    n_harmonics: int
    harmonics: np.ndarray          # the n values analysed (1, 2, 3, ...)
    Gp: np.ndarray                 # G_n' (storage), aligned to sin(n omega t)
    Gpp: np.ndarray                # G_n'' (loss), aligned to cos(n omega t)
    intensity: np.ndarray          # |G_n*| / |G_1*|, harmonic content
    even_harmonic_error: float     # max |G_n*| for even n, /|G_1*| (should ~ 0)

    @property
    def G1_prime(self) -> float:
        return float(self.Gp[0])

    @property
    def G1_doubleprime(self) -> float:
        return float(self.Gpp[0])


def decompose_oscillatory(t: np.ndarray, sigma_t: np.ndarray, omega: float,
                          gamma0: float, n_harmonics: int = 9
                          ) -> OscillatoryResponse:
    """Project a one-period stress series onto harmonics of the sin drive.

    ``t`` must span exactly one period [0, T) (uniform samples, endpoint
    excluded) and ``sigma_t`` the corresponding macroscopic stress.
    """
    T = 2.0 * np.pi / omega
    dt = t[1] - t[0]
    ns = np.arange(1, n_harmonics + 1)

    Gp = np.zeros(ns.size)
    Gpp = np.zeros(ns.size)
    for k, n in enumerate(ns):
        # Trapezoidal projection over a full period (periodic -> rectangle ok).
        Gp[k] = (2.0 / (gamma0 * T)) * np.sum(sigma_t * np.sin(n * omega * t)) * dt
        Gpp[k] = (2.0 / (gamma0 * T)) * np.sum(sigma_t * np.cos(n * omega * t)) * dt

    mag = np.hypot(Gp, Gpp)
    intensity = mag / mag[0] if mag[0] > 0 else mag
    even = ns % 2 == 0
    even_err = float(intensity[even].max()) if even.any() else 0.0

    return OscillatoryResponse(omega, gamma0, n_harmonics, ns, Gp, Gpp,
                               intensity, even_err)
