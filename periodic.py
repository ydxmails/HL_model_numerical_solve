"""Large-amplitude oscillatory shear (LAOS): periodic-orbit solver.

Under a periodic drive there is no fixed point dP/dt = 0, but the long-time
response settles onto a limit cycle P(sigma, t) = P(sigma, t + T), T = 2 pi /
omega.  That cycle is a *fixed point of the period (stroboscopic) map*

    M : P(.,0) |-> P(.,T)      (integrate exactly one period)

i.e. it solves F(P0) = M(P0) - P0 = 0.  We provide two solvers:

* **Picard** -- iterate P <- M(P).  Simple, but its convergence rate is the
  dominant Floquet multiplier |mu_max|, which -> 1 (critical slowing down) at
  small gamma0 or alpha -> alpha_c, exactly the interesting regime.
* **Newton-Krylov** (default) -- solve F(P0) = 0 with a Jacobian-free Newton
  method.  The action of the monodromy/Jacobian is finite-differenced, so each
  Krylov step costs one extra one-period integration; convergence is quadratic
  and *independent* of |mu_max|, curing the slowing down.

Because M conserves mass, F is orthogonal to the constant vector and the
Jacobian is singular along it (the conserved-mass direction); we simply
renormalise to unit mass, which fixes that one gauge.

Once converged we integrate one more period sampling the macroscopic stress and
project it onto harmonics of the *sin* drive (``observables.decompose_oscillatory``),
so the first-harmonic moduli G1', G1'' are defined identically to the SAOS
moduli -- the gamma0 -> 0 limit G1' -> G' is a clean alignment check.  The
dominant Floquet multiplier (power iteration on the linearised map) doubles as a
convergence certificate and as the amplitude-dependent relaxation rate omega_c.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from .grid import Grid
from .operators import build_static_parts
from .observables import OscillatoryResponse, decompose_oscillatory
from .protocols import Oscillatory
from .selfconsistency import ALPHA_C
from .transient import TransientStepper, initial_delta


@dataclass
class LAOSResult:
    alpha: float
    gamma0: float
    omega: float
    response: OscillatoryResponse
    P0: np.ndarray = field(repr=False)         # distribution at t = 0 (gamma = 0)
    residual: float = 0.0                      # ||M(P0) - P0||
    floquet: float | None = None               # dominant |mu|, ~ omega_c / omega
    method: str = "newton_krylov"
    n_period_solves: int = 0

    @property
    def G1_prime(self) -> float:
        return self.response.G1_prime

    @property
    def G1_doubleprime(self) -> float:
        return self.response.G1_doubleprime


class PeriodMap:
    """One-period propagator P(0) -> P(T) for given (alpha, gamma0, omega)."""

    def __init__(self, alpha: float, gamma0: float, omega: float, grid: Grid,
                 parts: dict | None = None, steps_per_period: int = 400,
                 theta: float = 0.5, picard_iters: int = 1, dt_max: float = 0.5):
        self.grid = grid
        self.parts = parts if parts is not None else build_static_parts(grid)
        self.protocol = Oscillatory(gamma0, omega)
        self.T = self.protocol.period
        # The explicit reaction half (yielding at rate 1/tau = 1) is stable only
        # for dt < 1; cap dt well below that.  At low omega the period is long, so
        # a fixed step count would give a huge, unstable dt -- size the step count
        # from the period instead, taking the finer of the two requirements.
        n_steps = max(int(steps_per_period), int(np.ceil(self.T / dt_max)))
        self.n_steps = n_steps
        self.dt = self.T / n_steps
        self.stepper = TransientStepper(alpha, grid, self.parts, theta=theta,
                                        picard_iters=picard_iters)
        self.n_calls = 0

    def __call__(self, P0: np.ndarray) -> np.ndarray:
        self.n_calls += 1
        return self.stepper.propagate(P0, self.protocol.gammadot, self.T, self.dt,
                                      t0=0.0)

    def sample_stress(self, P0: np.ndarray, n_sample: int = 400):
        """Integrate one period from P0, returning (t, sigma(t)) over [0, T)."""
        rec = max(1, int(round((self.T / self.dt) / n_sample)))
        traj = self.stepper.run(P0, self.protocol.gammadot, self.T, self.dt,
                                t0=0.0, record_every=rec)
        # keep exactly one period, endpoint excluded
        keep = traj.t < self.T - 1e-9
        return traj.t[keep], traj.stress[keep]


def _normalize(P: np.ndarray, grid: Grid) -> np.ndarray:
    return P / grid.integrate(P)


def solve_laos(alpha: float, gamma0: float, omega: float, grid: Grid | None = None,
               parts: dict | None = None, steps_per_period: int = 400,
               method: str = "newton_krylov", tol: float = 1e-8,
               max_iter: int = 200, P_init: np.ndarray | None = None,
               n_harmonics: int = 9, floquet: bool = False,
               theta: float = 0.5, dt_max: float = 0.5) -> LAOSResult:
    """Solve the periodic LAOS response and extract the moduli/harmonics.

    ``dt_max`` caps the time step (default 0.5); at low omega the step count is
    raised above ``steps_per_period`` so dt stays below the explicit reaction's
    stability limit (dt < 1).  Without this, low frequencies would otherwise use
    a huge, unstable dt.
    """
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)

    pmap = PeriodMap(alpha, gamma0, omega, grid, parts,
                     steps_per_period=steps_per_period, theta=theta,
                     dt_max=dt_max)

    if P_init is not None:
        x0 = np.array(P_init, dtype=float)
    elif alpha > ALPHA_C:
        # liquid phase: the cycle is a perturbation of the quiescent base state;
        # starting there sidesteps the long delta-relaxation transient.
        from .saos import liquid_base_state
        x0 = liquid_base_state(alpha, grid, parts)[0]
    else:
        x0 = initial_delta(grid)
    x0 = _normalize(np.asarray(x0, dtype=float), grid)

    used = method
    if method == "newton_krylov":
        def F(x):
            return pmap(x) - x
        try:
            P0 = newton_krylov(F, x0, f_tol=tol, maxiter=max_iter,
                               method="lgmres")
        except NoConvergence as exc:    # fall back to Picard from best iterate
            P0 = np.asarray(exc.args[0], dtype=float)
            used = "newton_krylov->picard"
            P0 = _picard(pmap, _normalize(P0, grid), grid, tol, max_iter)
    elif method == "picard":
        P0 = _picard(pmap, x0, grid, tol, max_iter)
    else:
        raise ValueError("method must be 'newton_krylov' or 'picard'")

    P0 = _normalize(P0, grid)
    residual = float(np.linalg.norm(pmap(P0) - P0))

    t, sig = pmap.sample_stress(P0, n_sample=max(8 * n_harmonics, 256))
    response = decompose_oscillatory(t, sig, omega, gamma0, n_harmonics=n_harmonics)

    mu = _dominant_floquet(pmap, P0, grid) if floquet else None

    return LAOSResult(alpha, gamma0, omega, response, P0, residual, mu, used,
                      pmap.n_calls)


def _picard(pmap: PeriodMap, x0: np.ndarray, grid: Grid, tol: float,
            max_iter: int) -> np.ndarray:
    x = x0
    for _ in range(max_iter):
        x_new = _normalize(pmap(x), grid)
        if np.linalg.norm(x_new - x) <= tol:
            return x_new
        x = x_new
    return x


def _dominant_floquet(pmap: PeriodMap, P0: np.ndarray, grid: Grid,
                      n_iter: int = 30, eps: float = 1e-6) -> float:
    """Power iteration on the linearised period map (finite-difference J v).

    Returns the dominant Floquet multiplier magnitude.  The perturbation is kept
    mass-neutral (sum zero) to project out the trivial mass eigenvalue.
    """
    MP0 = pmap(P0)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(grid.n)
    v -= v.mean()                      # mass-neutral
    v /= np.linalg.norm(v)
    mu = 0.0
    for _ in range(n_iter):
        Jv = (pmap(P0 + eps * v) - MP0) / eps
        Jv -= Jv.mean()
        nrm = np.linalg.norm(Jv)
        if nrm == 0.0:
            return 0.0
        mu = nrm
        v = Jv / nrm
    return float(mu)
