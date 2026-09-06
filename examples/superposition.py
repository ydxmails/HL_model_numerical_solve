"""Parallel superposition: a small oscillation collinear with steady shear.

Shows (a) the exact linear moduli from the sheared-state oracle, including the
low-frequency regime where G' goes negative, (b) convergence of the nonlinear
limit cycle onto that oracle as gammaA -> 0, and (c) the fluidisation map: how
the cycle-mean stress falls below the unperturbed steady stress as the
oscillation grows.

Run:  python examples/superposition.py
"""

import numpy as np

from hlmodel import Grid, solve_steady, solve_superposed
from hlmodel.saos import psr_modulus
from hlmodel.sweep import psr_spectrum_sweep

grid = Grid(sigma_max=8, n_per_unit=150)

# --------------------------------------------------------------------------
# (a) Linear moduli about the sheared steady state -- no time stepping at all.
#     Two banded solves per frequency, the same cost as SAOS.
# --------------------------------------------------------------------------
alpha, gammadot0 = 0.3, 0.01          # jammed, low rate => strongly thinning
sp = psr_spectrum_sweep(alpha, gammadot0, np.logspace(-2.2, 0.7, 14), grid)

print(f"Parallel-superposition moduli (alpha={alpha}, gammadot0={gammadot0})")
print(f"unperturbed steady stress: {sp.steady_stress:.5f}\n")
print(f"{'omega':>9} {'G_par_prime':>12} {'G_par_dprime':>13}")
for w, gp, gpp in zip(sp.omega, sp.Gp, sp.Gpp):
    flag = "   <- negative" if gp < 0 else ""
    print(f"{w:9.4f} {gp:12.5f} {gpp:13.5f}{flag}")

n_neg = int(sp.negative_Gp.sum())
print(f"\nG' < 0 at {n_neg} of {sp.omega.size} frequencies "
      f"(omega <= {sp.omega[sp.negative_Gp].max():.3g}).")
print("This is the classic parallel-superposition pathology. Here it comes out")
print("of an exact linear-response solve about a shear-thinning steady state,")
print("so it is a property of the model, not of the measurement.\n")

# --------------------------------------------------------------------------
# (b) The nonlinear limit cycle must converge onto the oracle as gammaA -> 0.
#     The residual is A*dt + B*gammaA^2: halving dt halves the first term,
#     halving gammaA quarters the second.
# --------------------------------------------------------------------------
alpha, gammadot0, omega = 0.4, 0.1, 0.5
Gp_lin, Gpp_lin = psr_modulus(alpha, gammadot0, omega, grid)
Sigma_s = solve_steady(alpha, gammadot0, grid, warn=False).stress
print(f"Linear oracle (alpha={alpha}, gammadot0={gammadot0}, omega={omega}):  "
      f"G' = {Gp_lin:.5f}   G'' = {Gpp_lin:.5f}")
print(f"Unperturbed steady stress: {Sigma_s:.5f}\n")

print("Convergence of the nonlinear solver onto the oracle:")
print(f"{'gammaA':>8} {'steps':>7} {'dG1_prime':>11} {'dG1_dprime':>11} "
      f"{'thinning':>11}")
for gammaA in (0.02, 0.01):
    for spp in (512, 1024):
        r = solve_superposed(alpha, gammadot0, gammaA, omega, grid,
                             steps_per_period=spp, method="newton_krylov",
                             tol=1e-10, n_harmonics=5)
        dGp, dGpp = r.linear_offset
        print(f"{gammaA:8.3f} {spp:7d} {dGp:11.2e} {dGpp:11.2e} "
              f"{r.thinning:11.2e}")

print("\nThe modulus offsets halve with dt; the thinning barely moves, because")
print("the mass-conserving scheme reproduces cycle averages far better than")
print("instantaneous waveforms. The fluidisation map is cheap; the moduli are not.\n")

# --------------------------------------------------------------------------
# (c) Fluidisation: mean stress vs the regime parameter Lambda.
# --------------------------------------------------------------------------
print(f"Fluidisation sweep (alpha={alpha}, gammadot0={gammadot0}, omega={omega}):")
print(f"{'gammaA':>8} {'Lambda':>8} {'reverses':>9} {'mean_stress':>12} "
      f"{'thinning':>11} {'I2':>10} {'floquet':>9}")
for gammaA in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
    r = solve_superposed(alpha, gammadot0, gammaA, omega, grid,
                         steps_per_period=768, method="newton_krylov",
                         tol=1e-10, n_harmonics=7, floquet=True)
    print(f"{gammaA:8.2f} {r.Lambda:8.2f} {str(r.reverses):>9} "
          f"{r.mean_stress:12.5f} {r.thinning:11.2e} "
          f"{r.harmonic_intensity[1]:10.2e} {r.floquet:9.4f}")

print(f"\nLambda = gammaA*omega/gammadot0 separates the regimes: below 1 the flow")
print("is pulsed but never reverses, above 1 it reverses within each cycle.")
print("Note I2: the steady drift breaks the half-period symmetry, so even")
print("harmonics are physical here and the usual 1e-8 noise-floor error check")
print("is unavailable -- use dt refinement and the Floquet multiplier instead.")
print("The multipliers stay far inside the unit circle: superposition is much")
print("better conditioned than small-amplitude LAOS in the jammed phase.")
