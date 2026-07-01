"""Linear viscoelastic spectrum (SAOS) in the liquid phase.

Run:  python examples/saos_spectrum.py
"""

import numpy as np

from hlmodel import Grid
from hlmodel.sweep import saos_spectrum

grid = Grid(sigma_max=12, n_per_unit=300)
omegas = np.logspace(-3, 1.5, 40)
alpha = 0.8

sp = saos_spectrum(alpha, omegas, grid)

print(f"alpha = {alpha}")
print(f"{'omega':>10} {'G_prime':>14} {'G_doubleprime':>14} {'tan_delta':>10}")
for w, gp, gpp in zip(sp.omega, sp.Gp, sp.Gpp):
    print(f"{w:10.4e} {gp:14.5e} {gpp:14.5e} {gpp/gp:10.4f}")

# Low-frequency Maxwell limit: read off plateau modulus and relaxation time.
lo = omegas < 3e-3
a_loss = np.mean(sp.Gpp[lo] / sp.omega[lo])      # -> G0 * tau
a_stor = np.mean(sp.Gp[lo] / sp.omega[lo]**2)    # -> G0 * tau^2
G0_est = a_loss**2 / a_stor                      # -> G0
tau_est = a_stor / a_loss                        # -> tau
print(f"\nLow-omega Maxwell fit:  G0 ~ {G0_est:.3f} (expect 1),  "
      f"relaxation time tau ~ {tau_est:.3f}")

np.savez("saos_spectrum.npz", omega=sp.omega, Gp=sp.Gp, Gpp=sp.Gpp)
print("saved saos_spectrum.npz")

try:
    import matplotlib.pyplot as plt
    plt.loglog(sp.omega, sp.Gp, "o-", label="G'")
    plt.loglog(sp.omega, sp.Gpp, "s-", label="G''")
    plt.xlabel(r"$\omega$"); plt.ylabel("modulus")
    plt.legend(); plt.tight_layout(); plt.savefig("saos_spectrum.png", dpi=140)
    print("saved saos_spectrum.png")
except Exception:
    pass
