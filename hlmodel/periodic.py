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

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import newton_krylov, NoConvergence

from .grid import Grid
from .operators import build_static_parts
from .observables import (OscillatoryResponse, decompose_oscillatory,
                          YieldStressDistribution, yield_stress_distribution)
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
    converged: bool = True                      # rel. cycle-closure error <= 1e-3
    floquet: float | None = None               # dominant |mu|, ~ omega_c / omega
    method: str = "newton_krylov"
    n_period_solves: int = 0
    peak_cfl: float = 0.0                      # peak|gammadot| dt / h (diagnostic)
    # The drive this cycle was actually produced with.  Recorded so that
    # downstream analysis can tell -- a LAOSResult is otherwise
    # indistinguishable from one made with a plain sine of the same
    # (gamma0, omega), which is exactly the confusion that silently
    # corrupts the cycle-analysis routines.  See protocols.require_sine_drive.
    protocol: object | None = None
    # Yield-stress distribution over the converged cycle; populated only when
    # solve_laos is called with yield_phases not None (see below).
    yield_distribution: "CycleYieldDistribution | None" = field(default=None,
                                                                repr=False)


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
                 theta: float = 0.5, picard_iters: int = 1, dt_max: float = 0.5,
                 protocol=None):
        self.grid = grid
        self.parts = parts if parts is not None else build_static_parts(grid)
        # ``protocol`` overrides the default sine drive.  Any object exposing
        # ``period`` and ``gammadot(t)`` works -- the HL equation sees nothing but
        # gammadot(t), so e.g. SuperposedShear (parallel superposition), square or
        # multi-tone drives all slot in here with no change to the solvers.
        self.protocol = Oscillatory(gamma0, omega) if protocol is None else protocol
        self.T = self.protocol.period
        # The explicit reaction half (yielding at rate 1/tau = 1) is stable only
        # for dt < 1, and a large dt also drives the implicit operator toward the
        # singular reflecting Laplacian; clamp dt below that limit regardless of
        # the requested value.  At low omega the period is long, so a fixed step
        # count would give a huge dt -- size the step count from the period
        # instead, taking the finer of the two requirements.
        dt_cap = min(float(dt_max), 0.9)
        n_steps = max(int(steps_per_period), int(np.ceil(self.T / dt_cap)))
        self.n_steps = n_steps
        self.dt = self.T / n_steps
        self.stepper = TransientStepper(alpha, grid, self.parts, theta=theta,
                                        picard_iters=picard_iters)
        self.n_calls = 0

    # -- diagnostics ---------------------------------------------------------
    @property
    def peak_rate(self) -> float:
        """max |gammadot(t)| over one period (sampled; protocol-agnostic)."""
        ts = np.linspace(0.0, self.T, 512, endpoint=False)
        return float(np.max(np.abs(self.protocol.gammadot(ts))))

    @property
    def peak_cfl(self) -> float:
        """Advective Courant number peak|gammadot| dt / h.

        Reported, not enforced.  The explicit part of the TVD scheme is only the
        *defect* between two advection discretisations, not the full advective
        flux, so its stability limit is far weaker than this number suggests:
        measured convergence stays clean and first order at peak_cfl ~ 5.  Treat
        it as a scale, not a threshold.
        """
        return self.peak_rate * self.dt / self.grid.h

    def _record_stride(self, n_sample: int) -> int:
        """Record every ``rec`` steps, with ``rec`` a divisor of ``n_steps``.

        ``decompose_oscillatory`` assumes its samples tile [0, T) uniformly with
        the endpoint excluded.  That holds only when the stride divides the step
        count; otherwise the last gap differs from the rest and the harmonic
        projection aliases.  Measured: at omega = 0.3 with steps_per_period = 800
        (stride 3, remainder 2) the even-harmonic content jumped from ~1e-11 to
        1.9e-3 and the G'' convergence sequence broke.  Snapping down to the
        nearest divisor costs at most a few extra samples.
        """
        rec = max(1, int(round(self.n_steps / max(1, int(n_sample)))))
        while self.n_steps % rec:
            rec -= 1
        return rec

    def __call__(self, P0: np.ndarray) -> np.ndarray:
        self.n_calls += 1
        return self.stepper.propagate(P0, self.protocol.gammadot, self.T, self.dt,
                                      t0=0.0)

    def sample_stress(self, P0: np.ndarray, n_sample: int = 400):
        """Integrate one period from P0, returning (t, sigma(t)) over [0, T)."""
        rec = self._record_stride(n_sample)
        traj = self.stepper.run(P0, self.protocol.gammadot, self.T, self.dt,
                                t0=0.0, record_every=rec)
        # keep exactly one period, endpoint excluded
        keep = traj.t < self.T - 1e-9
        return traj.t[keep], traj.stress[keep]

    def sample_states(self, P0: np.ndarray, n_sample: int = 400):
        """Integrate one period from P0, returning (t, P(.,t)) over [0, T).

        Like :meth:`sample_stress` but keeps the full distributions, for
        cycle-averaged / phase-resolved post-processing (e.g. the yield-stress
        distribution).  ``P_states`` has shape ``(len(t), grid.n)``.
        """
        rec = self._record_stride(n_sample)
        traj = self.stepper.run(P0, self.protocol.gammadot, self.T, self.dt,
                                t0=0.0, record_every=rec, record_states=True)
        keep = traj.t < self.T - 1e-9
        return traj.t[keep], traj.P_states[keep]


@dataclass
class CycleYieldDistribution:
    """Yield-stress distribution rho_ac over one converged LAOS cycle.

    ``averaged`` is the event-weighted period average -- the distribution of the
    stress at which yield events happen, aggregated over a whole period.  It is
    symmetric in sigma by the sine drive's half-period symmetry
    (sigma, t) -> (-sigma, t + T/2), and is the oscillatory analogue of the
    steady-shear rho_ac.  If phase resolution was requested, ``phase`` holds the
    phases omega*t in [0, 2 pi) and ``resolved`` the instantaneous distributions
    there (each asymmetric, biased toward the instantaneous flow direction).
    """
    omega: float
    gamma0: float
    averaged: YieldStressDistribution
    phase: np.ndarray | None = None
    resolved: list | None = None


def cycle_yield_stress_distribution(pmap: "PeriodMap", P0: np.ndarray,
                                    n_phases: int = 0, n_sample: int = 400
                                    ) -> CycleYieldDistribution:
    """Yield-stress distribution over a converged LAOS cycle.

    Integrates one period from the cycle state ``P0`` and forms the
    period-averaged rho_ac from the time-average of P (so it is normalised by the
    cycle-mean yield rate <Gamma>).  With ``n_phases > 0`` it also returns the
    instantaneous rho_ac at that many evenly spaced phases across the period.
    """
    t, P_states = pmap.sample_states(P0, n_sample=n_sample)
    grid = pmap.grid
    P_bar = P_states.mean(axis=0)                    # (1/T) int_0^T P dt over [0, T)
    averaged = yield_stress_distribution(P_bar, grid)
    phase = None
    resolved = None
    if n_phases and int(n_phases) > 0:
        omega = pmap.protocol.omega
        ph = (omega * t) % (2.0 * np.pi)
        targets = np.linspace(0.0, 2.0 * np.pi, int(n_phases), endpoint=False)
        # nearest sampled phase to each target (shortest circular distance)
        idx = [int(np.argmin(np.abs(((ph - p + np.pi) % (2.0 * np.pi)) - np.pi)))
               for p in targets]
        phase = ph[idx]
        resolved = [yield_stress_distribution(P_states[i], grid) for i in idx]
    return CycleYieldDistribution(pmap.protocol.omega, pmap.protocol.gamma0,
                                  averaged, phase, resolved)


def _normalize(P: np.ndarray, grid: Grid) -> np.ndarray:
    return P / grid.integrate(P)


def _default_seed(alpha: float, gamma0: float, omega: float, grid: Grid,
                  parts: dict | None, protocol=None) -> np.ndarray:
    """Physically-motivated initial guess for the periodic fixed-point solve.

    Liquid (alpha > alpha_c): the quiescent base state, which the small-amplitude
    cycle perturbs.  Jammed (alpha <= alpha_c): at t = 0 the strain is zero but
    the strain *rate* is maximal (gamma0*omega), so the cycle's t = 0 state
    resembles steady shear at that rate -- a far better start than a delta at
    sigma = 0, which has essentially no overlap with the sheared cycle.
    """
    drift = 0.0 if protocol is None else float(getattr(protocol, "mean_rate", 0.0))
    gdot = (abs(float(gamma0) * float(omega)) if protocol is None
            else float(getattr(protocol, "seed_rate", abs(gamma0 * omega))))
    # With a steady drift the quiescent base state has essentially no overlap with
    # the cycle even in the liquid phase, so seed from the sheared steady state
    # regardless of alpha.  With drift == 0 this reduces exactly to the previous
    # behaviour.
    if alpha > ALPHA_C and drift == 0.0:
        from .saos import liquid_base_state
        return liquid_base_state(alpha, grid, parts)[0]
    if gdot <= 0.0:
        return initial_delta(grid)
    try:
        from .steady import solve_steady
        return solve_steady(alpha, gdot, grid, warn=False).P
    except Exception:
        return initial_delta(grid)


def solve_laos(alpha: float, gamma0: float, omega: float, grid: Grid | None = None,
               parts: dict | None = None, steps_per_period: int = 400,
               method: str = "newton_krylov", tol: float = 1e-8,
               max_iter: int = 200, P_init: np.ndarray | None = None,
               n_harmonics: int = 9, floquet: bool = False,
               theta: float = 0.5, dt_max: float = 0.5,
               protocol=None, yield_phases: int | None = None,
               warn: bool = True, verbose: bool = False) -> LAOSResult:
    """Solve the periodic LAOS response and extract the moduli/harmonics.

    ``dt_max`` caps the time step (default 0.5); at low omega the step count is
    raised above ``steps_per_period`` so dt stays below the explicit reaction's
    stability limit (dt < 1).  Without this, low frequencies would otherwise use
    a huge, unstable dt.

    ``yield_phases`` controls the yield-stress distribution rho_ac on the result
    (``LAOSResult.yield_distribution``): ``None`` (default) skips it; ``0`` returns
    the period-averaged distribution only; a positive integer additionally returns
    the instantaneous distribution at that many evenly spaced phases.
    """
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)

    pmap = PeriodMap(alpha, gamma0, omega, grid, parts,
                     steps_per_period=steps_per_period, theta=theta,
                     dt_max=dt_max, protocol=protocol)

    if P_init is not None:
        x0 = np.array(P_init, dtype=float)
    else:
        x0 = _default_seed(alpha, gamma0, omega, grid, parts, protocol)
    x0 = _normalize(np.asarray(x0, dtype=float), grid)

    used = method
    if method == "newton_krylov":
        # The fixed-point residual M(P0)-P0 is invariant to the total mass
        # (J = DM - I is exactly singular along the constant direction, since the
        # map conserves mass), so a bare Newton-Krylov lets the mass drift; the
        # nonlinear map then makes a post-hoc renormalization break the solution
        # -- visibly worse on finer grids.  Pin the mass by adding (integral - 1)
        # along the constant direction, which de-singularizes the Jacobian.
        def F(x):
            return (pmap(x) - x) + (grid.integrate(x) - 1.0)

        cb = _make_progress_cb(gamma0, omega, verbose) if verbose else None
        try:
            P0 = newton_krylov(F, x0, f_tol=tol, maxiter=max_iter,
                               method="lgmres", callback=cb)
        except NoConvergence as exc:    # fall back to Picard from best iterate
            P0 = np.asarray(exc.args[0], dtype=float)
            used = "newton_krylov->picard"
            P0 = _picard(pmap, _normalize(P0, grid), grid, tol, max_iter,
                         verbose=verbose, gamma0=gamma0, omega=omega)
        except (ValueError, np.linalg.LinAlgError):
            # Newton-Krylov can fail outright (singular/zero Jacobian step) when
            # pushed into a stiff/ill-conditioned regime; degrade to Picard from
            # the initial guess rather than crashing the caller's sweep.
            used = "newton_krylov->picard"
            P0 = _picard(pmap, x0, grid, tol, max_iter,
                         verbose=verbose, gamma0=gamma0, omega=omega)
    elif method == "picard":
        P0 = _picard(pmap, x0, grid, tol, max_iter,
                     verbose=verbose, gamma0=gamma0, omega=omega)
    else:
        raise ValueError("method must be 'newton_krylov' or 'picard'")

    P0 = _normalize(P0, grid)
    residual = float(np.linalg.norm(pmap(P0) - P0))
    rel_residual = residual / (float(np.linalg.norm(P0)) + 1e-300)
    converged = rel_residual <= 1e-3

    if warn and not converged:
        warnings.warn(
            f"LAOS may not have converged at gamma0={gamma0:g}, omega={omega:g} "
            f"(relative cycle-closure error {rel_residual:.1e}); the moduli may be "
            f"unreliable. This is common at small amplitude or near alpha_c, where "
            f"the period map is ill-conditioned. Try continuation from a nearby "
            f"solution (sweep.amplitude_sweep / sweep.omega_sweep), more "
            f"steps_per_period, or a coarser grid.",
            RuntimeWarning, stacklevel=2)

    if verbose:
        print(f"  [LAOS gamma0={gamma0:g} omega={omega:g}] done: "
              f"rel_resid={rel_residual:.1e} converged={converged} "
              f"({pmap.n_calls} period solves)", flush=True)

    t, sig = pmap.sample_stress(P0, n_sample=max(8 * n_harmonics, 256))
    response = decompose_oscillatory(t, sig, omega, gamma0, n_harmonics=n_harmonics)

    mu = _dominant_floquet(pmap, P0, grid) if floquet else None

    ydist = None
    if yield_phases is not None:
        ydist = cycle_yield_stress_distribution(
            pmap, P0, n_phases=int(yield_phases),
            n_sample=max(8 * n_harmonics, 256))

    return LAOSResult(alpha, gamma0, omega, response, P0, residual, converged,
                      mu, used, pmap.n_calls, pmap.peak_cfl, pmap.protocol,
                      ydist)


def _make_progress_cb(gamma0: float, omega: float, every: int = 1):
    """Build a Newton-Krylov callback that prints the residual each iteration."""
    state = {"k": 0}

    def cb(x, fx):
        state["k"] += 1
        if state["k"] % every == 0:
            print(f"  [LAOS gamma0={gamma0:g} omega={omega:g}] Newton iter "
                  f"{state['k']:3d}: ||F||={float(np.linalg.norm(fx)):.3e}",
                  flush=True)

    return cb


def _picard(pmap: PeriodMap, x0: np.ndarray, grid: Grid, tol: float,
            max_iter: int, verbose: bool = False,
            gamma0: float = 0.0, omega: float = 0.0) -> np.ndarray:
    x = x0
    for k in range(max_iter):
        x_new = _normalize(pmap(x), grid)
        delta = float(np.linalg.norm(x_new - x))
        if verbose and (k % 10 == 0 or delta <= tol):
            print(f"  [LAOS gamma0={gamma0:g} omega={omega:g}] Picard iter "
                  f"{k + 1:3d}: ||dP||={delta:.3e}", flush=True)
        if delta <= tol:
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
