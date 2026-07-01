"""LAOS: amplitude sweep showing (a) convergence to SAOS as gamma0 -> 0 and
(b) growth of odd harmonics at large amplitude.

Run:  python examples/laos_harmonics.py
"""

import numpy as np

from hlmodel import Grid
from hlmodel.saos import saos_modulus
from hlmodel.periodic import solve_laos

grid = Grid(sigma_max=8, n_per_unit=150)
alpha, omega = 0.8, 0.3

# SAOS reference at this frequency
Gp_s, Gpp_s = saos_modulus(alpha, omega, grid)
print(f"SAOS reference (alpha={alpha}, omega={omega}):  "
      f"G' = {Gp_s:.5f}   G'' = {Gpp_s:.5f}\n")

print("Amplitude sweep (Picard from base state, fine timestep):")
print(f"{'gamma0':>8} {'G1_prime':>10} {'G1_dprime':>10} "
      f"{'I3/I1':>9} {'I5/I1':>9} {'even_err':>9}")
for gamma0 in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
    res = solve_laos(alpha, gamma0, omega, grid, method="picard",
                     steps_per_period=600, tol=1e-9, n_harmonics=7)
    I = res.response.intensity
    print(f"{gamma0:8.2f} {res.G1_prime:10.5f} {res.G1_doubleprime:10.5f} "
          f"{I[2]:9.2e} {I[4]:9.2e} {res.response.even_harmonic_error:9.1e}")

print("\nAt small gamma0 the first-harmonic moduli approach the SAOS values;")
print("at large gamma0 the third/fifth harmonics grow while even harmonics")
print("stay ~0 (the half-period symmetry of the model).")
