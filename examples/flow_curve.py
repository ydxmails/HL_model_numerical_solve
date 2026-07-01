"""Steady flow curves for a jammed and a liquid sample.

Run:  python examples/flow_curve.py
Saves the data to flow_curve.npz; plotting (matplotlib) is optional.
"""

import numpy as np

from hlmodel import Grid
from hlmodel.sweep import flow_curve

grid = Grid(sigma_max=12, n_per_unit=300)
gammadots = np.logspace(-3, 0.5, 30)

print(f"{'gammadot':>12} {'sigma(a=0.3)':>14} {'sigma(a=0.5)':>14} {'sigma(a=0.8)':>14}")
curves = {}
for alpha in (0.3, 0.5, 0.8):
    fc = flow_curve(alpha, gammadots, grid)
    curves[alpha] = fc.stress

for i, gd in enumerate(gammadots):
    print(f"{gd:12.4e} {curves[0.3][i]:14.5f} {curves[0.5][i]:14.5f} "
          f"{curves[0.8][i]:14.5f}")

# Jammed (0.3): finite yield stress as gammadot -> 0 (Herschel-Bulkley).
# Critical (0.5): sigma ~ gammadot^{1/5}.
# Liquid (0.8): Newtonian sigma ~ gammadot at low rate (zero yield stress).
np.savez("flow_curve.npz", gammadot=gammadots,
         **{f"sigma_alpha_{a}": curves[a] for a in curves})
print("\nsaved flow_curve.npz")

try:
    import matplotlib.pyplot as plt
    for a in curves:
        plt.loglog(gammadots, curves[a], "o-", label=f"alpha={a}")
    plt.xlabel(r"$\dot\gamma$"); plt.ylabel(r"$\Sigma$")
    plt.legend(); plt.tight_layout(); plt.savefig("flow_curve.png", dpi=140)
    print("saved flow_curve.png")
except Exception:
    pass
