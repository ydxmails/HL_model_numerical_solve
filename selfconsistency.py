"""Self-consistency closures for the noise amplitude D.

The HL closure is D = alpha * Gamma with Gamma the yielding rate.  At zero shear
rate this collapses to a single scalar relation between alpha and D that we can
solve in closed form; it is the package's primary correctness anchor.

Quiescent identity (gammadot = 0), derived analytically from the stationary
Fokker-Planck equation:

    alpha = 1/2 + sqrt(D) + D .

This reproduces both facts quoted in Hebraud & Lequeux (1998): the critical
coupling alpha_c = 1/2 (set D = 0) and the scaling D ~ (alpha - alpha_c)^2 near
threshold (small-D expansion gives sqrt(D) ~ alpha - 1/2).
"""

from __future__ import annotations

import numpy as np

ALPHA_C = 0.5  # critical coupling


def alpha_of_D_quiescent(D: float) -> float:
    """Reference relation alpha(D) at gammadot = 0."""
    return 0.5 + np.sqrt(D) + D


def D_quiescent(alpha: float) -> float:
    """Stationary noise amplitude D at gammadot = 0 for given alpha.

    Returns 0 in the jammed phase (alpha <= alpha_c); otherwise inverts
    alpha = 1/2 + sqrt(D) + D by solving the quadratic in x = sqrt(D):
    x^2 + x + (1/2 - alpha) = 0, taking the positive root.
    """
    if alpha <= ALPHA_C:
        return 0.0
    disc = 4.0 * alpha - 1.0  # = 1 + 4*(alpha - 1/2)
    x = 0.5 * (-1.0 + np.sqrt(disc))
    return x * x
