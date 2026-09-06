"""Stress-space grid for the Hebraud-Lequeux model.

Reduced units throughout the package: tau = sigma_c = G0 = 1.

The grid is uniform with spacing ``h = 1 / n_per_unit`` so that the three
physically special points sigma = 0 and sigma = +/- sigma_c = +/- 1 always fall
exactly on grid nodes.  This matters because (i) the Dirac reinjection source
acts at sigma = 0 and (ii) the yielding indicator H(|sigma| - 1) switches at
sigma = +/- 1; placing those points on nodes keeps the discrete operators clean
and the discrete mass balance exact.

Weighting at the threshold
--------------------------
Landing +/-1 on nodes is not by itself enough.  With rectangle-rule weights a
*hard* mask (|sigma| > 1, threshold nodes excluded) integrates the yielding
region from 1 + h/2 rather than from 1, so Gamma -- and through the closure
D = alpha*Gamma the whole solution -- carries an O(h) bias.  ``yield_weight``
therefore gives the two nodes exactly at |sigma| = 1 weight 1/2, which is the
trapezoidal edge and is O(h^2).  Used consistently in the loss operator, the
reinjection source and Gamma it leaves the discrete mass balance exact (column
sums still cancel term by term) while restoring second-order accuracy.

``yield_mask`` is kept as the strict boolean region selector for diagnostics
that ask "where is the material yielding"; anything that *integrates* should use
``yield_weight`` (or :meth:`Grid.yield_fraction`, which does).
"""

from __future__ import annotations

import warnings
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
    yield_edge : {"trapezoid", "hard"}
        Weighting of the two nodes exactly at |sigma| = 1 in the yield integral.
        ``"trapezoid"`` (default, weight 1/2) is second order.  ``"hard"``
        (weight 0) reproduces the pre-fix behaviour and is O(h) biased -- it is
        provided only for reproducing older results and warns when used.
    """

    sigma_max: float = 8.0
    n_per_unit: int = 200
    yield_edge: str = "trapezoid"

    # --- derived arrays (filled in __post_init__) -----------------------
    sigma: np.ndarray = None  # type: ignore[assignment]
    h: float = None  # type: ignore[assignment]
    i_zero: int = None  # type: ignore[assignment]
    yield_mask: np.ndarray = None  # type: ignore[assignment]
    yield_weight: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if abs(self.sigma_max - round(self.sigma_max)) > 1e-12:
            raise ValueError("sigma_max must be an integer (units of sigma_c)")
        if self.n_per_unit < 1:
            raise ValueError("n_per_unit must be >= 1")
        if self.yield_edge not in ("trapezoid", "hard"):
            raise ValueError("yield_edge must be 'trapezoid' or 'hard'")

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
        # Trapezoidal edge: half weight exactly at the threshold (see module
        # docstring).  Zero inside, one outside, 1/2 on the two nodes at +/-1.
        yield_weight = yield_mask.astype(float)
        if self.yield_edge == "trapezoid":
            yield_weight[np.abs(np.abs(sigma) - 1.0) < 1e-9] = 0.5
        else:
            warnings.warn(
                "Grid(yield_edge='hard') excludes the nodes at |sigma| = 1 from the "
                "yield integral, which integrates the yielding region from 1 + h/2 "
                "instead of 1.  Gamma -- and through the closure D = alpha*Gamma the "
                "whole solution -- is then biased at O(h), and the scheme drops from "
                "second to first order (steady-stress error ~10x larger at "
                "n_per_unit=200).  This is the pre-fix behaviour, kept only for "
                "reproducing older results; do not use it for new work.",
                RuntimeWarning, stacklevel=3)

        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "i_zero", i_zero)
        object.__setattr__(self, "yield_mask", yield_mask)
        object.__setattr__(self, "yield_weight", yield_weight)

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
        """Yielding rate Gamma = (1/tau) * integral_{|sigma|>1} P dsigma.

        Trapezoidal at the +/-1 endpoints (see module docstring), so second order
        rather than the O(h)-biased strict-mask sum.
        """
        return self.h * np.sum(P * self.yield_weight, axis=-1)

    def first_moment(self, P: np.ndarray) -> float | complex:
        """Macroscopic stress Sigma = integral sigma P dsigma."""
        return self.h * np.sum(self.sigma * P, axis=-1)
