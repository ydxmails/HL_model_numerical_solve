"""Steady-shear elastic/plastic ratio from a tagged sigma=0 cohort.

Under steady shear the material is in a stationary state with a fixed
self-generated noise ``D = alpha * Gamma``.  We tag the blocks sitting at
``sigma = 0`` at a chosen instant and follow that cohort forward under the full
(mass-conserving) dynamics -- advection + diffusion + yielding + reinjection --
holding ``D`` at its steady value.  Starting from ``sigma = 0``, after the
external strain has advanced by a window ``dgamma`` (i.e. after a time
``dgamma / gammadot``) the cohort's stress displacement is compared with the
purely affine (elastic) prediction ``G0 * dgamma``:

    R = <sigma>_cohort(dgamma) / (G0 * dgamma)          (elastic ratio)

``R = 1`` is the purely elastic response (the cohort followed the affine ramp);
``R < 1`` signals plastic relaxation, and 1 - R is the plastic fraction.  A
companion signal-to-noise number

    P = (G0 * dgamma) / std[ sigma_cohort(dgamma) ]      (affine / non-affine spread)

compares the affine loading with the block-to-block non-affine spread over the
same window (an advection/diffusion Peclet number).  ``P`` from the full cohort
mixes diffusive spread with plastic-reset scatter; the optional survivor variant
(absorbing cohort, no reinjection) isolates the pure diffusive spread and is the
cleaner shear-vs-diffusion measure once ``dgamma`` exceeds the yield strain.

Because R depends on ``dgamma`` relative to the (rate-dependent) yield stress
``Sigma(gammadot)`` rather than on ``dgamma`` absolutely, ``scale_by_stress``
lets the window track the stress: with it on, the passed ``dgamma`` is a
multiplier ``c`` and the window becomes ``c * Sigma(gammadot) / G0`` at each rate.

Reduced units are used throughout, so the elastic modulus is ``G0 = 1``.

Example
-------
>>> from hlmodel.elastic_ratio import elastic_ratio_based_dgamma, hb_yield_stress
>>> sy = hb_yield_stress(0.4)                                    # doctest: +SKIP
>>> er = elastic_ratio_based_dgamma(0.4, 1e-2, sy)               # doctest: +SKIP
>>> er.R, er.P                                                   # doctest: +SKIP
(0.93..., 0.47...)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .grid import Grid
from .operators import advection_matrix, build_static_parts
from .steady import solve_steady

G0 = 1.0  # reduced elastic modulus


@dataclass
class ElasticRatio:
    """Result of a steady-shear elastic-ratio measurement.

    Attributes
    ----------
    alpha, gammadot : float
        Coupling and shear rate.
    dgamma : float
        The strain window actually used (already resolved: if ``scale_by_stress``
        was set this is ``c * Sigma / G0``, otherwise the value passed).
    R : float
        Elastic ratio ``<sigma>(dgamma) / (G0 * dgamma)``.  ``R = 1`` elastic,
        ``R < 1`` plastic; ``1 - R`` is the plastic fraction.
    sigma_mean, sigma_std : float
        Mean and standard deviation of the full cohort's stress at ``dgamma``.
    P : float
        Affine/non-affine spread ratio ``(G0 * dgamma) / sigma_std`` (full cohort;
        ``inf`` if the spread is zero).
    D, stress : float
        The steady self-consistent noise ``D`` and stress ``Sigma(gammadot)``.
    P_surv, survival : float
        Survivor-cohort (pure-diffusion) ``P`` and the surviving (not-yet-yielded)
        fraction at ``dgamma``; ``nan`` unless ``survivor=True`` was requested.
    """

    alpha: float
    gammadot: float
    dgamma: float
    R: float
    sigma_mean: float
    sigma_std: float
    P: float
    D: float
    stress: float
    P_surv: float = float("nan")
    survival: float = float("nan")


def _cohort_moments(G, grid: Grid, n_steps: int, da: float):
    """Crank-Nicolson-evolve a delta at sigma=0 under ``G`` for ``n_steps`` and
    return (mean, std, mass) of the resulting distribution.

    The moments are normalised by the surviving mass, so for a mass-conserving
    generator (mass ~ 1) they are the ordinary moments, while for an absorbing
    generator they are conditioned on the not-yet-yielded sub-population.
    """
    n, h, sig = grid.n, grid.h, grid.sigma
    identity = sp.eye(n, format="csc")
    lu = spla.splu((identity - 0.5 * da * G).tocsc())
    Mp = identity + 0.5 * da * G
    Q = np.zeros(n)
    Q[grid.i_zero] = 1.0 / h
    for _ in range(n_steps):
        Q = lu.solve(Mp.dot(Q))
    m0 = h * Q.sum()
    m1 = h * (sig * Q).sum()
    m2 = h * (sig ** 2 * Q).sum()
    mean = m1 / m0
    var = m2 / m0 - mean ** 2
    return mean, float(np.sqrt(max(var, 0.0))), float(m0)


def elastic_ratio_based_dgamma(alpha: float, gammadot: float, dgamma: float,
                               grid: Grid | None = None, parts: dict | None = None,
                               scale_by_stress: bool = False, survivor: bool = False,
                               da: float | None = None, da_cap: float = 0.05,
                               n_min: int = 200, warn: bool = True) -> ElasticRatio:
    """Elastic ratio ``R`` of the tagged sigma=0 cohort over a strain window.

    Parameters
    ----------
    alpha, gammadot : float
        Coupling and shear rate.
    dgamma : float
        Strain window.  If ``scale_by_stress`` is False this is the absolute
        window; if True it is a multiplier ``c`` and the window used is
        ``c * Sigma(gammadot) / G0``.  A good fixed choice is the yield strain
        ``sigma_y`` (see :func:`hb_yield_stress`).
    grid, parts : optional
        Stress grid and cached static operators; built with defaults if omitted.
        Share ``parts`` across a sweep to avoid rebuilding operators.
    scale_by_stress : bool, default False
        Interpret ``dgamma`` as a multiple of the steady stress ``Sigma`` rather
        than an absolute strain (R depends on ``dgamma / Sigma``, so this holds
        the response at a fixed fraction of the rate-dependent yield strain).
    survivor : bool, default False
        Also evolve the absorbing cohort (no reinjection) and return the
        pure-diffusion ``P_surv`` and the surviving fraction.
    da : float, optional
        Age step.  If None, chosen rate-adaptively as ``min(da_cap, dt / n_min)``
        so the window is resolved by at least ``n_min`` steps while the age step
        stays below ``da_cap`` (which resolves the unit-rate yielding).
    da_cap : float, default 0.05
        Upper bound on the age step.
    n_min : int, default 200
        Minimum number of steps across the window.
    warn : bool, default True
        Forwarded to :func:`solve_steady`.

    Returns
    -------
    ElasticRatio
    """
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)

    st = solve_steady(alpha, gammadot, grid, parts, warn=warn)
    D, Sigma = float(st.D), float(st.stress)

    window = float(dgamma) * Sigma / G0 if scale_by_stress else float(dgamma)
    if window <= 0.0:
        raise ValueError("resolved dgamma window must be positive")

    dt = window / gammadot
    da_use = float(da) if da is not None else min(da_cap, dt / n_min)
    n_steps = max(1, int(round(dt / da_use)))
    da_use = dt / n_steps  # land the last step exactly on dt

    # Full cohort (with reinjection): R and the total-scatter P.
    #
    # Deliberately the first-order *upwind* flux, not the flux-limited one used
    # elsewhere.  R is by definition a ratio to affine loading, so the affine
    # transport has to be exact.  A conservative flux F_i = v Q_i moves the first
    # moment at exactly v: dm1/dt = h sum_i F_i = v * mass.  A *spatially
    # varying* limiter breaks that -- its face weights a_i, b_i attach to
    # different nodes, so sum_j (a_j + b_{j-1}) Q_j != sum_j Q_j -- and the
    # resulting O(h) error appears directly as R > 1 (measured 1.059 at
    # n_per_unit=100, 1.016 at 200, against the exact 1.000 upwind gives).
    # The steady background (D, Sigma) still comes from the TVD solver.
    A = advection_matrix(gammadot, grid)
    Gfull = (D * parts["L2"] + A + parts["Yield"] + parts["Source"]).tocsr()
    mean_f, std_f, _ = _cohort_moments(Gfull, grid, n_steps, da_use)
    R = mean_f / (G0 * window)
    P = (G0 * window) / std_f if std_f > 0.0 else float("inf")

    P_surv, surv = float("nan"), float("nan")
    if survivor:
        Gabs = D * parts["L2"] + A + parts["Yield"]
        _, std_s, S = _cohort_moments(Gabs.tocsc(), grid, n_steps, da_use)
        P_surv = (G0 * window) / std_s if std_s > 0.0 else float("inf")
        surv = S

    return ElasticRatio(alpha=alpha, gammadot=gammadot, dgamma=window,
                        R=R, sigma_mean=mean_f, sigma_std=std_f, P=P,
                        D=D, stress=Sigma, P_surv=P_surv, survival=surv)


def _elastic_ratio_task(args):
    """Worker for a parallel sweep: rebuilds the grid inside the process (so only
    a small spec is pickled) and pins BLAS/OpenMP to one thread."""
    from .sweep import _limit_threads, _make_grid  # lazy: only when sweeping parallel
    alpha, gammadot, dgamma, grid_spec, opts = args
    with _limit_threads():
        grid = _make_grid(grid_spec)
        parts = build_static_parts(grid)
        er = elastic_ratio_based_dgamma(alpha, gammadot, dgamma, grid, parts, **opts)
    return er


def elastic_ratio_sweep(alpha: float, gammadots, dgamma: float,
                        grid: Grid | None = None, parts: dict | None = None,
                        scale_by_stress: bool = False, survivor: bool = False,
                        n_workers: int | None = None, warn: bool = False,
                        **kwargs) -> list[ElasticRatio]:
    """Run :func:`elastic_ratio_based_dgamma` across a list of shear rates.

    ``dgamma`` is held fixed (absolute) unless ``scale_by_stress`` is set, in
    which case each rate uses its own ``c * Sigma(gammadot)``.  Returns a list of
    :class:`ElasticRatio`, one per rate, in the input order.

    Parallelism
    -----------
    The points are independent, so with ``n_workers > 1`` they are farmed out to a
    process pool (via :func:`hlmodel.sweep.parallel_map`), each worker rebuilding
    the grid from its parameters and pinning BLAS to a single thread -- so N
    workers use N cores rather than N x (BLAS threads).  With ``n_workers=1`` the
    sweep runs serially and reuses the passed ``grid``/``parts`` directly;
    ``n_workers=None`` uses ``os.cpu_count()``.  Low shear rates cost more (a
    longer strain window means more time steps), so tasks are dispatched
    longest-first for load balance.

    Notes
    -----
    In the parallel path ``parts`` is rebuilt inside each worker, so the passed
    ``parts`` is ignored there; the grid is taken from ``grid`` (or the default).
    """
    gammadots = np.asarray(gammadots, dtype=float)
    if grid is None:
        grid = Grid()

    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(int(n_workers), len(gammadots)))

    if n_workers == 1:
        if parts is None:
            parts = build_static_parts(grid)
        return [elastic_ratio_based_dgamma(alpha, float(gd), dgamma, grid, parts,
                                           scale_by_stress=scale_by_stress,
                                           survivor=survivor, warn=warn, **kwargs)
                for gd in gammadots]

    from .sweep import parallel_map  # lazy import of the parallel infrastructure
    grid_spec = (grid.sigma_max, grid.n_per_unit)
    opts = dict(scale_by_stress=scale_by_stress, survivor=survivor, warn=warn, **kwargs)
    args = [(alpha, float(gd), dgamma, grid_spec, opts) for gd in gammadots]
    cost = [1.0 / max(float(gd), 1e-12) for gd in gammadots]  # low rate -> more steps
    return parallel_map(_elastic_ratio_task, args, n_workers=n_workers, cost=cost)


def hb_yield_stress(alpha: float, gammadots=None, grid: Grid | None = None,
                    parts: dict | None = None, warn: bool = False) -> float:
    """Herschel-Bulkley yield stress of the flow curve, ``Sigma = sigma_y + K*gdot**n``.

    In reduced units the yield stress equals the yield strain ``sigma_y / G0``,
    which is the recommended fixed ``dgamma`` for :func:`elastic_ratio_based_dgamma`.
    Only defined in the jammed phase; raises for ``alpha >= 1/2`` (no yield stress
    in the liquid phase).
    """
    if alpha >= 0.5:
        raise ValueError("no Herschel-Bulkley yield stress for alpha >= 1/2 "
                         "(liquid phase); pass dgamma explicitly instead")
    from scipy.optimize import curve_fit

    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)
    if gammadots is None:
        gammadots = np.logspace(-4.0, 0.0, 12)
    gammadots = np.asarray(gammadots, dtype=float)

    Sigma = np.array([solve_steady(alpha, gd, grid, parts, warn=warn).stress
                      for gd in gammadots])

    def hb(gd, sy, K, n):
        return sy + K * gd ** n

    p0 = [max(0.8 * float(Sigma.min()), 1e-3), 0.5, 0.5]
    popt, _ = curve_fit(hb, gammadots, Sigma, p0=p0, maxfev=20000)
    return float(popt[0])


if __name__ == "__main__":
    # Sample call: alpha in the jammed phase; fixed dgamma = HB yield strain.
    # Run with:  python -m hlmodel.elastic_ratio
    alpha = 0.4
    grid = Grid(sigma_max=12, n_per_unit=200)
    parts = build_static_parts(grid)

    sy = hb_yield_stress(alpha, grid=grid, parts=parts)
    print(f"alpha={alpha}:  HB yield strain sigma_y = {sy:.4f}\n")

    print(f"R at fixed dgamma = sigma_y, vs shear rate:")
    print(f"{'gdot':>9} {'R':>7} {'P':>7} {'D':>9} {'Sigma':>8}")
    for gd in np.logspace(-3.5, -0.5, 7):
        er = elastic_ratio_based_dgamma(alpha, gd, sy, grid, parts, warn=False)
        print(f"{gd:9.2e} {er.R:7.3f} {er.P:7.3f} {er.D:9.4f} {er.stress:8.4f}")

    print(f"\nrate-scaled window (dgamma = 2*Sigma) with survivor P:")
    er = elastic_ratio_based_dgamma(alpha, 1e-2, 2.0, grid, parts,
                                    scale_by_stress=True, survivor=True, warn=False)
    print(f"  gdot=1e-2: window={er.dgamma:.4f}  R={er.R:.3f}  "
          f"P_full={er.P:.3f}  P_surv={er.P_surv:.3f}  survival={er.survival:.3f}")
