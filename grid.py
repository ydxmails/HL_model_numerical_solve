"""Stress-space grid for the Hebraud-Lequeux model.

Reduced units throughout the package: tau = sigma_c = G0 = 1.

The grid is uniform with spacing ``h = 1 / n_per_unit`` so that the three
physically special points sigma = 0 and sigma = +/- sigma_c = +/- 1 always fall
exactly on grid nodes.  This matters because (i) the Dirac reinjection source
acts at sigma = 0 and (ii) the yielding indicator H(|sigma| - 1) switches at
sigma = +/- 1; placing those points on nodes keeps the discrete operators clean
and the discrete mass balance exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    """Uniform stress grid symmetric about sigma = 0.

    Parameters
    ----------
    sigma_max : float
        Half-width of the domain; the grid runs over [-sigma_max, sigma_max].
        Must be a positive integer (in units of sigma_c) so that +/-1 are nodes.
    n_per_unit : int
        Number of intervals per unit stress.  Spacing is h = 1 / n_per_unit.
    """

    sigma_max: float = 8.0
    n_per_unit: int = 200

    # --- derived arrays (filled in __post_init__) -----------------------
    sigma: np.ndarray = None  # type: ignore[assignment]
    h: float = None  # type: ignore[assignment]
    i_zero: int = None  # type: ignore[assignment]
    yield_mask: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if abs(self.sigma_max - round(self.sigma_max)) > 1e-12:
            raise ValueError("sigma_max must be an integer (units of sigma_c)")
        if self.n_per_unit < 1:
            raise ValueError("n_per_unit must be >= 1")

        L = int(round(self.sigma_max))
        m = int(self.n_per_unit)
        h = 1.0 / m
        n = 2 * L * m + 1
        sigma = np.linspace(-L, L, n)
        # Guard against round-off in the node coordinates.
        sigma = np.round(sigma / h) * h

        i_zero = L * m  # index of sigma = 0
        # H(|sigma| - 1) = 1 strictly for |sigma| > 1; the nodes at +/-1 are NOT
        # yielding (H(0) = 0 in the paper's convention).
        yield_mask = np.abs(sigma) > 1.0 + 1e-9

        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "i_zero", i_zero)
        object.__setattr__(self, "yield_mask", yield_mask)

    # --- convenience ----------------------------------------------------
    @property
    def n(self) -> int:
        return self.sigma.size

    def integrate(self, f: np.ndarray) -> float | complex:
        """Rectangle-rule integral over the whole domain.

        Rectangle (not trapezoid) weighting is deliberate: it makes the
        discrete mass balance -- yielding removes exactly what the source
        reinjects -- hold to machine precision.
        """
        return self.h * np.sum(f, axis=-1)

    def yield_fraction(self, P: np.ndarray) -> float | complex:
        """Yielding rate Gamma = (1/tau) * integral_{|sigma|>1} P dsigma."""
        return self.h * np.sum(P[..., self.yield_mask], axis=-1)

    def first_moment(self, P: np.ndarray) -> float | complex:
        """Macroscopic stress Sigma = integral sigma P dsigma."""
        return self.h * np.sum(self.sigma * P, axis=-1)
