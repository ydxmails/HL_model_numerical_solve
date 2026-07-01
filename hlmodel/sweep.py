"""Parallel parameter sweeps (flow curves, SAOS spectra, LAOS maps).

The HL solvers are cheap per parameter point but a campaign -- a flow curve over
many shear rates, a spectrum over many frequencies, a LAOS map over a
(gamma0, omega) grid -- embarrasses the parallelism: every point is independent.
We therefore parallelise at the *parameter level* with a process pool, leaving
each individual solve sequential (time stepping is causal and cannot be
parallelised across steps).

Two practical points make this scale well:

* **No oversubscription.**  Each worker pins BLAS/OpenMP to a single thread
  (``threadpoolctl``), so N worker processes use N cores rather than N x (BLAS
  threads).  The inner linear algebra here is small banded solves, for which
  threading would only add overhead.
* **Picklable, top-level workers.**  The work functions live at module scope and
  receive plain tuples (grid specified by its parameters and rebuilt inside the
  worker), so they pickle cleanly across processes.

If only one core is available the pool degrades gracefully to serial execution.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except Exception:  # threadpoolctl is optional
    threadpool_limits = None

from .grid import Grid
from .operators import build_static_parts
from .saos import liquid_base_state, saos_modulus
from .selfconsistency import ALPHA_C
from .steady import solve_steady, BOUNDARY_FRACTION_TOL
from .periodic import solve_laos


# --------------------------------------------------------------------------
# worker-side helpers (must be importable / picklable)
# --------------------------------------------------------------------------
def _limit_threads():
    if threadpool_limits is not None:
        return threadpool_limits(limits=1)
    # no-op context manager
    from contextlib import nullcontext
    return nullcontext()


def _make_grid(grid_spec) -> Grid:
    sigma_max, n_per_unit = grid_spec
    return Grid(sigma_max=sigma_max, n_per_unit=n_per_unit)


def _steady_task(args):
    alpha, gammadot, grid_spec = args
    with _limit_threads():
        grid = _make_grid(grid_spec)
        res = solve_steady(alpha, gammadot, grid, warn=False)
    return (gammadot, res.D, res.Gamma, res.stress, res.converged,
            res.boundary_fraction)


def _saos_task(args):
    alpha, omega, grid_spec, base = args
    with _limit_threads():
        grid = _make_grid(grid_spec)
        parts = build_static_parts(grid)
        Gp, Gpp = saos_modulus(alpha, omega, grid, parts, base=base)
    return (omega, Gp, Gpp)


def _laos_task(args):
    alpha, gamma0, omega, grid_spec, opts = args
    opts = dict(opts)
    opts.setdefault("warn", False)   # aggregate in the parent instead of per worker
    with _limit_threads():
        grid = _make_grid(grid_spec)
        res = solve_laos(alpha, gamma0, omega, grid, **opts)
    return (gamma0, omega, res.G1_prime, res.G1_doubleprime,
            res.response.intensity.copy(), res.residual, res.floquet,
            res.converged)


# --------------------------------------------------------------------------
# generic parallel map
# --------------------------------------------------------------------------
def parallel_map(func, arg_list, n_workers: int | None = None, cost=None):
    """Map ``func`` over ``arg_list`` across processes, preserving input order.

    Falls back to serial execution when only one worker is available.

    If ``cost`` is given (one estimated relative runtime per task), the tasks are
    dispatched longest-first (the classic LPT heuristic): this keeps every worker
    busy until near the end and avoids the imbalance that occurs when the slow
    tasks happen to sit together in the list (e.g. low-frequency LAOS points all
    at the front of a sweep).  Results are returned in the original order
    regardless of dispatch order.
    """
    n = len(arg_list)
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, n))
    if n_workers == 1 or n <= 1:
        return [func(a) for a in arg_list]

    order = list(range(n))
    if cost is not None:
        # dispatch most expensive first; remember how to put results back
        order = sorted(order, key=lambda i: float(cost[i]), reverse=True)
    reordered = [arg_list[i] for i in order]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_in_dispatch_order = list(pool.map(func, reordered))

    out = [None] * n
    for slot, res in zip(order, results_in_dispatch_order):
        out[slot] = res
    return out


# --------------------------------------------------------------------------
# convenience campaigns
# --------------------------------------------------------------------------
@dataclass
class FlowCurve:
    alpha: float
    gammadot: np.ndarray
    D: np.ndarray
    Gamma: np.ndarray
    stress: np.ndarray
    converged: np.ndarray            # False where the rate was below the grid's floor
    boundary_fraction: np.ndarray    # probability in the outer 10% of the domain


def flow_curve(alpha: float, gammadots, grid: Grid | None = None,
               n_workers: int | None = None) -> FlowCurve:
    """Steady flow curve sigma(gammadot) at fixed alpha, computed in parallel.

    Emits a single consolidated :class:`RuntimeWarning` (from this process, not
    the workers) if any rate is resolution-limited: too low for the grid (D hit
    the conditioning floor -> refine ``n_per_unit``) or large enough that the
    distribution is truncated by the domain (-> increase ``sigma_max``).
    """
    if grid is None:
        grid = Grid()
    spec = (grid.sigma_max, grid.n_per_unit)
    gds = np.atleast_1d(np.asarray(gammadots, dtype=float))
    args = [(alpha, float(gd), spec) for gd in gds]
    out = parallel_map(_steady_task, args, n_workers)
    D = np.array([o[1] for o in out])
    Gamma = np.array([o[2] for o in out])
    stress = np.array([o[3] for o in out])
    converged = np.array([o[4] for o in out])
    bfrac = np.array([o[5] for o in out])

    too_low = gds[~converged]
    if too_low.size:
        floor = 4.0 * grid.h * grid.h
        warnings.warn(
            f"{too_low.size} shear rate(s) <= {too_low.max():g} are below the "
            f"resolution floor for this grid (D ~ 4h^2 = {floor:.1e}); those points "
            f"are unreliable. Increase n_per_unit to reach lower rates.",
            RuntimeWarning, stacklevel=2)
    truncated = gds[bfrac > BOUNDARY_FRACTION_TOL]
    if truncated.size:
        warnings.warn(
            f"{truncated.size} shear rate(s) >= {truncated.min():g} truncate against "
            f"sigma_max={grid.sigma_max:g} (probability piling at the domain edge); "
            f"their stress is underestimated. Increase sigma_max -- the high-rate "
            f"branch sigma ~ gammadot needs sigma_max >> gammadot.",
            RuntimeWarning, stacklevel=2)

    return FlowCurve(alpha, gds, D, Gamma, stress, converged, bfrac)


@dataclass
class SAOSSpectrum:
    alpha: float
    omega: np.ndarray
    Gp: np.ndarray
    Gpp: np.ndarray


def saos_spectrum(alpha: float, omegas, grid: Grid | None = None,
                  n_workers: int | None = None) -> SAOSSpectrum:
    """Linear moduli over many frequencies in parallel (shared base state)."""
    if grid is None:
        grid = Grid()
    spec = (grid.sigma_max, grid.n_per_unit)
    parts = build_static_parts(grid)
    base = liquid_base_state(alpha, grid, parts)   # computed once, shared
    ws = np.atleast_1d(np.asarray(omegas, dtype=float))
    args = [(alpha, float(w), spec, base) for w in ws]
    out = parallel_map(_saos_task, args, n_workers)
    Gp = np.array([o[1] for o in out])
    Gpp = np.array([o[2] for o in out])
    return SAOSSpectrum(alpha, ws, Gp, Gpp)


@dataclass
class LAOSMap:
    alpha: float
    gamma0: np.ndarray            # grid of amplitudes
    omega: np.ndarray             # grid of frequencies
    G1p: np.ndarray               # shape (len(gamma0), len(omega))
    G1pp: np.ndarray
    residual: np.ndarray
    floquet: np.ndarray
    converged: np.ndarray         # per-point convergence flag


def laos_map(alpha: float, gamma0s, omegas, grid: Grid | None = None,
             n_workers: int | None = None, **solver_opts) -> LAOSMap:
    """LAOS first-harmonic moduli over a (gamma0, omega) grid, in parallel.

    Points that fail to converge are reported in a single consolidated warning
    (and flagged in the returned ``converged`` array) rather than warning from
    each worker process.
    """
    if grid is None:
        grid = Grid()
    spec = (grid.sigma_max, grid.n_per_unit)
    g0s = np.atleast_1d(np.asarray(gamma0s, dtype=float))
    ws = np.atleast_1d(np.asarray(omegas, dtype=float))
    args = [(alpha, float(g), float(w), spec, solver_opts)
            for g in g0s for w in ws]
    # cost proxy: period solves scale with the step count, which is bounded below
    # by steps_per_period and grows like 1/omega once dt hits its cap (low omega
    # => long period => many steps).  Dispatching these first balances the pool.
    spp = float(solver_opts.get("steps_per_period", 400))
    dt_cap = min(float(solver_opts.get("dt_max", 0.5)), 0.9)
    cost = [max(spp, np.ceil((2.0 * np.pi / float(w)) / dt_cap))
            for _g in g0s for w in ws]
    out = parallel_map(_laos_task, args, n_workers, cost=cost)

    ng, nw = g0s.size, ws.size
    G1p = np.empty((ng, nw)); G1pp = np.empty((ng, nw))
    resid = np.empty((ng, nw)); floq = np.full((ng, nw), np.nan)
    conv = np.empty((ng, nw), dtype=bool)
    idx = 0
    for i in range(ng):
        for j in range(nw):
            _, _, gp, gpp, _inten, r, mu, cflag = out[idx]
            G1p[i, j] = gp; G1pp[i, j] = gpp; resid[i, j] = r
            conv[i, j] = cflag
            if mu is not None:
                floq[i, j] = mu
            idx += 1

    n_bad = int((~conv).sum())
    if n_bad:
        warnings.warn(
            f"{n_bad} of {conv.size} LAOS point(s) did not converge (relative "
            f"cycle-closure error > 1e-3); see the returned `converged` array. "
            f"These are typically at small amplitude or near alpha_c -- consider "
            f"sweep.amplitude_sweep / sweep.omega_sweep (continuation) or more "
            f"steps_per_period.", RuntimeWarning, stacklevel=2)
    return LAOSMap(alpha, g0s, ws, G1p, G1pp, resid, floq, conv)


# --------------------------------------------------------------------------
# continuation sweeps (sequential, warm-started -- robust at small amplitude
# and near alpha_c, where independent solves of the period map struggle)
# --------------------------------------------------------------------------
@dataclass
class LAOSSweepResult:
    alpha: float
    gamma0: np.ndarray        # amplitude(s)
    omega: np.ndarray         # frequency(ies)
    G1p: np.ndarray           # first-harmonic storage modulus, aligned to the swept axis
    G1pp: np.ndarray          # first-harmonic loss modulus
    residual: np.ndarray
    converged: np.ndarray     # per-point convergence flag (rel. error <= 1e-3)
    method: list


def amplitude_sweep(alpha: float, gamma0s, omega: float, grid: Grid | None = None,
                    P_init=None, descending: bool = True,
                    **solver_opts) -> LAOSSweepResult:
    """LAOS amplitude sweep at fixed ``omega`` using continuation between amplitudes.

    Amplitudes are solved sequentially, each warm-started from the previous
    converged cycle.  This is far more robust and faster than independent solves
    (e.g. ``laos_map``) at small amplitude or near alpha_c, where the period map
    is ill-conditioned and a cold-started Newton-Krylov stalls.  By default the
    sweep runs from large to small amplitude (``descending=True``), starting
    where convergence is easiest; the first point uses ``solve_laos``'s default
    physically-motivated seed unless ``P_init`` is supplied.  Results are returned
    aligned to the input ``gamma0s`` order.  Extra keyword arguments are forwarded
    to ``solve_laos`` (e.g. ``steps_per_period``, ``tol``, ``method``, ``verbose``).
    """
    if grid is None:
        grid = Grid()
    g0s = np.atleast_1d(np.asarray(gamma0s, dtype=float))
    order = np.argsort(g0s)[::-1] if descending else np.argsort(g0s)
    P = None if P_init is None else np.asarray(P_init, dtype=float)
    Gp = np.empty(g0s.size); Gpp = np.empty(g0s.size)
    resid = np.empty(g0s.size); conv = np.empty(g0s.size, dtype=bool)
    meth = [None] * g0s.size
    for idx in order:
        r = solve_laos(alpha, float(g0s[idx]), float(omega), grid, P_init=P,
                       **solver_opts)
        P = r.P0
        Gp[idx] = r.G1_prime; Gpp[idx] = r.G1_doubleprime
        resid[idx] = r.residual; conv[idx] = r.converged; meth[idx] = r.method
    return LAOSSweepResult(alpha, g0s, np.atleast_1d(float(omega)),
                           Gp, Gpp, resid, conv, meth)


def omega_sweep(alpha: float, gamma0: float, omegas, grid: Grid | None = None,
                P_init=None, descending: bool = True,
                **solver_opts) -> LAOSSweepResult:
    """LAOS frequency sweep at fixed amplitude ``gamma0`` using continuation.

    Frequencies are solved sequentially, each warm-started from the previous
    converged cycle.  By default the sweep runs from high to low frequency
    (``descending=True``), starting from the easily-converged high-frequency end;
    the first point uses ``solve_laos``'s default seed unless ``P_init`` is given.
    Results are returned aligned to the input ``omegas`` order.  Extra keyword
    arguments are forwarded to ``solve_laos``.
    """
    if grid is None:
        grid = Grid()
    ws = np.atleast_1d(np.asarray(omegas, dtype=float))
    order = np.argsort(ws)[::-1] if descending else np.argsort(ws)
    P = None if P_init is None else np.asarray(P_init, dtype=float)
    Gp = np.empty(ws.size); Gpp = np.empty(ws.size)
    resid = np.empty(ws.size); conv = np.empty(ws.size, dtype=bool)
    meth = [None] * ws.size
    for idx in order:
        r = solve_laos(alpha, float(gamma0), float(ws[idx]), grid, P_init=P,
                       **solver_opts)
        P = r.P0
        Gp[idx] = r.G1_prime; Gpp[idx] = r.G1_doubleprime
        resid[idx] = r.residual; conv[idx] = r.converged; meth[idx] = r.method
    return LAOSSweepResult(alpha, np.atleast_1d(float(gamma0)), ws,
                           Gp, Gpp, resid, conv, meth)
