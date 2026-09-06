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

from dataclasses import dataclass, field

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
    # n = 0 Fourier coefficient: the cycle-mean stress <Sigma>_T.  A *stress*, not
    # a modulus -- it is not divided by gamma0, since there is no strain amplitude
    # to normalise a DC offset by.  Zero to numerical noise under a pure sine drive
    # (half-period symmetry); nonzero whenever the protocol carries a steady drift
    # (parallel superposition), where it is the mean shear stress.
    dc: float = 0.0

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

    # n = 0 coefficient.  The stride fix in PeriodMap guarantees the samples tile
    # [0, T) uniformly, so the plain mean is exactly the periodic quadrature.
    dc = float(np.mean(sigma_t))

    mag = np.hypot(Gp, Gpp)
    intensity = mag / mag[0] if mag[0] > 0 else mag
    even = ns % 2 == 0
    even_err = float(intensity[even].max()) if even.any() else 0.0

    return OscillatoryResponse(omega, gamma0, n_harmonics, ns, Gp, Gpp,
                               intensity, even_err, dc)


# ---------------------------------------------------------------------------
# Distribution of the local stress at yield (sigma_ac)
# ---------------------------------------------------------------------------
@dataclass
class YieldStressDistribution:
    """Distribution rho_ac(sigma) of the local stress carried by a block at the
    instant it yields.

    HL yields at a *flat* rate 1/tau over the whole over-threshold region, so the
    rate of yield events occurring at stress sigma is (1/tau) H(|sigma|-1) P(sigma);
    normalising by the total rate Gamma = integral_{|sigma|>1} P gives

        rho_ac(sigma) = H(|sigma|-1) P(sigma) / Gamma ,

    supported on |sigma| > 1 with unit integral.  (Because the yield rate is
    uniform this coincides with the normalised over-threshold *population*; a
    sigma-dependent rate would reweight by that rate.)
    """
    sigma: np.ndarray                 # full grid stress axis
    rho: np.ndarray                   # density on the full grid (0 for |sigma| <= 1)
    Gamma: float                      # normalisation used (a yield rate, or its cycle mean)
    grid: Grid = field(repr=False)
    defined: bool = True              # False if Gamma == 0 (jammed & frozen: no yield)

    @property
    def support(self) -> tuple[np.ndarray, np.ndarray]:
        """(sigma, rho) restricted to the yielding region |sigma| > 1."""
        m = self.grid.yield_mask
        return self.sigma[m], self.rho[m]

    @property
    def mean_overshoot(self) -> float:
        """Mean penetration past threshold, <|sigma_ac| - 1>."""
        m = self.grid.yield_mask
        if not self.defined:
            return 0.0
        return float(self.grid.h * np.sum((np.abs(self.sigma[m]) - 1.0) * self.rho[m]))

    @property
    def asymmetry(self) -> float:
        """max |rho(sigma) - rho(-sigma)| / peak.

        Zero for a symmetric distribution (quiescent, or period-averaged
        oscillation); nonzero under steady shear or at a single oscillation phase,
        where yielding is biased toward the flow direction.
        """
        m = self.grid.yield_mask
        peak = self.rho[m].max() if self.defined and m.any() else 0.0
        if peak <= 0.0:
            return 0.0
        return float(np.max(np.abs(self.rho - self.rho[::-1])[m]) / peak)


def yield_stress_distribution(P: np.ndarray, grid: Grid) -> YieldStressDistribution:
    """Normalised distribution rho_ac(sigma) of the local stress at yield.

    ``P`` may be a steady distribution (``solve_steady``), an instantaneous
    ``P(sigma, t)`` for a phase-resolved cycle slice, or a time-integral
    ``int P dt`` for the period average -- each self-normalises, since ``Gamma`` is
    taken from the same ``P``.  In the jammed, frozen quiescent state nothing
    yields (Gamma = 0) and the distribution is genuinely undefined; it is then
    returned as all zeros with ``defined=False``.
    """
    P = np.asarray(P, dtype=float)
    # Weight with grid.yield_weight, not the strict mask: Gamma below is the
    # trapezoidally-weighted integral, so rho must carry the same weights or it
    # does not integrate to one (the half-weighted nodes at |sigma| = 1 would be
    # counted in the normalisation but omitted from the density).
    w = grid.yield_weight
    Gamma = float(grid.yield_fraction(P))
    rho = np.zeros(grid.n)
    defined = Gamma > 0.0
    if defined:
        rho = w * P / Gamma
    return YieldStressDistribution(grid.sigma, rho, Gamma, grid, defined)
