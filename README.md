# hlmodel

Numerical solution of the **Hébraud–Lequeux (1998)** mode-coupling model for the
rheology of soft glassy materials. The package solves the self-consistent
Fokker–Planck equation for the local-stress distribution and computes:

- **steady shear** — flow curves $\Sigma(\dot\gamma)$, with a direct stationary
  solver and an independent transient integrator that agree to ~1e-11;
- **SAOS** — linear viscoelastic moduli $G'(\omega), G''(\omega)$ by
  linearisation about the liquid base state (one solve per frequency);
- **LAOS** — the nonlinear periodic response as a periodic-orbit fixed point of
  the stroboscopic map, with first-harmonic moduli and higher-harmonic content;
- **parallel parameter sweeps** over rate, frequency, and amplitude.

All quantities are in reduced units $\tau = \sigma_c = G_0 = 1$; the critical
coupling is $\alpha_c = 1/2$. See [`theory.md`](theory.md) for the equations,
the analytic validation identities, and the numerical methods.

## Install

```bash
cd hlmodel-pkg
pip install -e .            # core (numpy, scipy)
pip install -e ".[test]"    # + pytest, threadpoolctl  (for the test suite)
```

Multi-core sweeps use `threadpoolctl` (in the `parallel`/`test` extras) to pin
BLAS to one thread per worker.

## Quickstart

### Steady flow curve

```python
import numpy as np
from hlmodel import Grid
from hlmodel.sweep import flow_curve

grid = Grid(sigma_max=10, n_per_unit=300)
gammadots = np.logspace(-3, 0, 24)

fc = flow_curve(alpha=0.3, gammadots=gammadots, grid=grid)  # jammed: yield stress
print(fc.stress)        # macroscopic stress at each rate
print(fc.D)             # self-consistent noise amplitude
```

A single point, with the distribution itself:

```python
from hlmodel import solve_steady
res = solve_steady(alpha=0.6, gammadot=0.1, grid=grid)
print(res.stress, res.D, res.Gamma)   # checks: D == alpha*Gamma, integral(P) == 1
```

### SAOS spectrum (linear moduli)

```python
import numpy as np
from hlmodel import Grid
from hlmodel.sweep import saos_spectrum

grid = Grid(sigma_max=10, n_per_unit=300)
omegas = np.logspace(-3, 1, 40)
sp = saos_spectrum(alpha=0.8, omegas=omegas, grid=grid)   # liquid phase only
# sp.Gp ~ omega^2, sp.Gpp ~ omega at low frequency (Maxwell); Gp -> 1 at high omega
```

### LAOS (nonlinear oscillatory response)

```python
from hlmodel import Grid
from hlmodel.periodic import solve_laos

grid = Grid(sigma_max=8, n_per_unit=200)
res = solve_laos(alpha=0.8, gamma0=1.0, omega=0.3, grid=grid,
                 method="newton_krylov", steps_per_period=400,
                 n_harmonics=9, floquet=True)

print(res.G1_prime, res.G1_doubleprime)        # first-harmonic moduli
print(res.response.intensity)                  # |G_n*|/|G_1*|, odd harmonics
print(res.response.even_harmonic_error)        # ~ 0 by symmetry (error check)
print(res.floquet)                             # dominant Floquet multiplier ~ omega_c
```

As $\gamma_0 \to 0$, `res.G1_prime` and `res.G1_doubleprime` converge to the SAOS
$G'(\omega), G''(\omega)$ — the two solvers share one strain convention by
design, so this limit is a built-in consistency check.

A LAOS map over amplitude and frequency, in parallel:

```python
import numpy as np
from hlmodel.sweep import laos_map

m = laos_map(alpha=0.8,
             gamma0s=np.array([0.1, 0.3, 1.0, 3.0]),
             omegas=np.array([0.1, 0.3, 1.0]),
             grid=grid, method="newton_krylov", steps_per_period=400)
print(m.G1p.shape)     # (len(gamma0s), len(omegas))
```

## Choosing parameters

- **`Grid(sigma_max, n_per_unit)`** — `sigma_max` (integer) must comfortably
  exceed the stress range so the tails are negligible; `n_per_unit` sets the
  resolution $h = 1/n_\text{per\_unit}$. The two parameters guard the two ends of
  the accessible shear-rate range:
  - *Lowest rate* is limited by `n_per_unit`. The self-consistent $D \to 0$ as
    $\dot\gamma \to 0$, and the stationary solve becomes ill-conditioned once
    $D$ drops to the floor $\sim 4h^2$; in the jammed phase $D \approx
    \tfrac12\dot\gamma$, so the floor is reached near $\dot\gamma \approx 8h^2$.
    To go lower, raise `n_per_unit`.
  - *Highest rate* is limited by `sigma_max`. The high-rate branch is Newtonian,
    $\Sigma \approx \dot\gamma$ (a block is swept a distance $\sim\dot\gamma$ past
    the threshold before it yields), so the distribution extends to $\sigma \sim
    \dot\gamma$. If `sigma_max` is not $\gg \dot\gamma$ the distribution piles
    against the domain wall and the stress saturates near `sigma_max` — a
    truncation artifact, not real shear-thinning. Use `sigma_max` a few times
    larger than the largest rate.

  Because $n = 2\,\sigma_\text{max}\,n_\text{per\_unit}$, resolving very low *and*
  very high rates at once is expensive; it is usually best to split a wide
  rate range into sub-ranges with grids sized to each.

- **Resolution warnings.** `solve_steady` and `flow_curve` warn automatically when
  a result is resolution-limited (rate below the floor, or distribution truncated
  by the domain), and the results carry the diagnostics directly:
  `SteadyResult.converged` / `FlowCurve.converged` is `False` for under-resolved
  low rates, and `.boundary_fraction` reports the probability sitting in the outer
  10% of the domain (large $\Rightarrow$ raise `sigma_max`). To silence the
  warnings, pass `warn=False` to `solve_steady`, or filter `RuntimeWarning`.

- **Steady vs transient** — `solve_steady` is the fast, accurate route for flow
  curves. The `TransientStepper` is for explicit time histories and underlies
  the LAOS period map.
- **LAOS solver** — `newton_krylov` (default) is robust against critical slowing
  down; `picard` is simplest and, started from the base state, fast in the
  liquid phase. The transient is first order in $\Delta t$, so reduce
  `steps_per_period` (i.e. $\Delta t$) for tight quantitative LAOS work, or read
  the linear limit off the exact SAOS solver.

## Tests

```bash
pytest -q
```

The suite encodes the physics: the quiescent identity $\alpha = 1/2 + \sqrt D +
D$, steady self-consistency and mass conservation, transient/oracle agreement,
the Maxwell low-frequency limit, the SAOS↔LAOS alignment, and the critical
exponents $\Sigma \sim \dot\gamma^{1/5}$ and $\sigma_y \sim (\alpha_c -
\alpha)^{1/2}$.

## Reference

P. Hébraud and F. Lequeux, *Mode-Coupling Theory for the Pasty Rheology of Soft
Glassy Materials*, Phys. Rev. Lett. **81**, 2934 (1998).
