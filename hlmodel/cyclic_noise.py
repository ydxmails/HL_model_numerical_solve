"""Cycle-averaged mechanical noise for oscillatory (LAOS) steady states.

In the Hebraud-Lequeux model the diffusivity is self-generated,
``D(t) = alpha * Gamma(t)`` with ``Gamma(t)`` the fraction of blocks above the
yield threshold ``|sigma| > 1``.  Under oscillatory shear this noise varies over
the cycle; its time-average over one converged period,

    D_bar = < D(t) >_cycle = alpha * < Gamma(t) >_cycle ,

is the natural oscillatory counterpart of the steady-state ``D``.  It is the
quantity that sets the diffusive spread accumulated over a window -- the variance
added over ``[t0, t0 + dt]`` is ``2 * integral(D dt) = 2 * D_bar * dt`` -- and it
feeds the oscillatory advection/diffusion ratio

    r_osc = mean_rate / (2 * D_bar) = gamma0 * omega / (pi * D_bar) ,

where ``mean_rate = (2/pi) * gamma0 * omega`` is the mean strain-rate magnitude
over the ``T/4 -> 3T/4`` half-cycle (the interval used for the elastic-plastic
stress window).  This is the direct analogue of ``gammadot / (2 D)`` in steady
shear; unlike the steady ratio it varies strongly with omega and gamma0, because
the oscillation timescale is external and the self-generated noise does not track
it.

Because ``D(t)`` depends on the strain magnitude it has period ``T/2``, so its
half-cycle and full-cycle averages coincide; the average here is taken over a
full sampled period for robustness.

The flat cycle average ``D_bar`` is dominated by the near-maximum-strain part of
the cycle (large ``D``, small strain rate), whereas the advection/diffusion
balance is set where the shearing happens.  ``cycle_mean_noise`` therefore also
returns a *rate-weighted* noise,

    D_rate = < D(t) |gammadot(t)| > / < |gammadot(t)| > ,

the mean noise experienced per unit strain traversed (weight
``|gammadot| dt = |dgamma|``).  It leans on the high-strain-rate part of the
cycle -- so it tracks the peak-rate noise ``D_peak = D(t=0)`` much more closely
than ``D_bar`` -- while remaining an average, hence robust in sign where the
single-instant ``D(t=0)`` can flip negative near the numerical floor.  Each noise
measure has a matching ratio ``r_osc`` / ``r_osc_rate`` / ``r_osc_peak`` that
shares the same numerator.

Example
-------
>>> from hlmodel.cyclic_noise import cycle_mean_noise
>>> cn = cycle_mean_noise(alpha=0.4, gamma0=0.5, omega=0.1)
>>> cn.D_bar, cn.r_osc                      # doctest: +SKIP
(0.00891..., 1.78...)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import Grid
from .operators import build_static_parts
from .periodic import LAOSResult, PeriodMap, solve_laos
from .protocols import require_sine_drive


@dataclass
class CycleNoise:
    """Cycle-averaged noise of an oscillatory HL steady state.

    Attributes
    ----------
    alpha, gamma0, omega : float
        The parameters the cycle was solved at.
    D_bar : float
        Cycle-mean diffusivity ``< alpha * Gamma(t) >`` over one period.  This is
        the primary output.
    Gamma_bar : float
        Cycle-mean yield fraction ``< Gamma(t) >`` (so ``D_bar = alpha * Gamma_bar``).
    r_osc : float
        Oscillatory advection/diffusion ratio ``gamma0 * omega / (pi * D_bar)``
        (``inf`` if ``D_bar`` is zero).
    D_rate : float
        Rate-weighted ("strain-weighted") noise
        ``< D(t) |gammadot(t)| > / < |gammadot(t)| >`` -- the mean noise a block
        experiences *per unit strain traversed* (weight ``|gammadot| dt = |dgamma|``).
        It emphasises the high-strain-rate part of the cycle, where advection
        dominates and the advection/diffusion balance is actually set, so it tracks
        the peak-rate noise far more closely than the flat cycle average ``D_bar``
        while remaining an average -- hence robust in sign, unlike a single sample.
    r_osc_rate : float
        Advection/diffusion ratio built from ``D_rate`` (``inf`` if ``D_rate`` is
        zero).  Uses the *same* numerator as ``r_osc``, so the three ``r_osc*``
        variants differ only in which noise measure sets the denominator.
    D_peak : float
        Instantaneous noise ``D(t)`` at maximum strain rate (``t = 0``), the most
        advection-dominated instant.  Non-negative by construction (see the clamp
        in :func:`cycle_mean_noise`); this is the ``D_t[0]`` some analyses use,
        made safe against the sub-yield sign flip.
    r_osc_peak : float
        Advection/diffusion ratio built from ``D_peak`` (``inf`` if ``D_peak`` is
        zero); same numerator as ``r_osc``.
    t, D_t, gammadot_t : ndarray
        Phase-resolved samples over one period ``[0, T)``: the sample times, the
        (clamped, non-negative) noise ``D(t) = alpha * Gamma(t)``, and the strain
        rate ``gammadot(t)``.
    period : float
        The oscillation period ``T = 2*pi/omega``.
    converged : bool
        Whether the underlying limit-cycle solve converged.
    """

    alpha: float
    gamma0: float
    omega: float
    D_bar: float
    Gamma_bar: float
    r_osc: float
    t: np.ndarray = field(repr=False)
    D_t: np.ndarray = field(repr=False)
    gammadot_t: np.ndarray = field(repr=False)
    period: float = 0.0
    converged: bool = True
    D_rate: float = 0.0
    r_osc_rate: float = float("inf")
    D_peak: float = 0.0
    r_osc_peak: float = float("inf")


def cycle_mean_noise(alpha: float, gamma0: float, omega: float,
                     grid: Grid | None = None, parts: dict | None = None,
                     n_sample: int = 400, steps_per_period: int = 400,
                     laos_result: LAOSResult | None = None,
                     warn: bool = True, **laos_kwargs) -> CycleNoise:
    """Cycle-averaged mechanical noise ``D_bar`` of the LAOS steady state.

    Solves the periodic (limit-cycle) response at ``(alpha, gamma0, omega)``,
    integrates one period to obtain the phase-resolved distribution ``P(., t)``,
    and returns ``D_bar = < alpha * Gamma(t) >`` together with the derived
    advection/diffusion ratio ``r_osc``.

    Parameters
    ----------
    alpha, gamma0, omega : float
        Coupling, strain amplitude, and angular frequency.
    grid, parts : optional
        Stress grid and cached static operators; built with defaults if omitted.
        Pass a shared ``parts = build_static_parts(grid)`` across many calls to
        avoid rebuilding operators in a sweep.
    n_sample : int, default 400
        Number of phase samples over one period used for the cycle average.
    steps_per_period : int, default 400
        Time steps per period for both the limit-cycle solve and the sampling
        pass (kept equal so the sampled cycle matches the converged one).
    laos_result : LAOSResult, optional
        A precomputed converged cycle at the *same* ``(alpha, gamma0, omega)`` and
        ``steps_per_period``; supply it to skip the re-solve.
    warn : bool, default True
        Forwarded to :func:`hlmodel.periodic.solve_laos`.
    **laos_kwargs
        Extra keyword arguments forwarded to :func:`solve_laos` (e.g. ``tol``,
        ``theta``, ``dt_max``, ``method``).

    Returns
    -------
    CycleNoise

    Notes
    -----
    The samples returned by the period integration are uniform in time over
    ``[0, T)``, so the plain mean of ``Gamma(t)`` is the (rectangle-rule) time
    average.  ``D(t)`` has period ``T/2``, hence averaging over the full period
    equals averaging over any half-period.
    """
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)

    # 1) Converged limit cycle (reuse one if supplied).
    require_sine_drive(getattr(laos_result, "protocol", None), "cycle_mean_noise")
    if laos_result is None:
        laos_result = solve_laos(alpha, gamma0, omega, grid, parts,
                                 steps_per_period=steps_per_period,
                                 warn=warn, **laos_kwargs)

    # 2) Integrate one period from the converged strobe state to get P(., t).
    pmap = PeriodMap(alpha, gamma0, omega, grid, parts,
                     steps_per_period=steps_per_period)
    t, P_states = pmap.sample_states(laos_result.P0, n_sample=n_sample)

    # 3) Gamma(t) and D(t) = alpha * Gamma(t), vectorised over phases.
    #    Gamma is the probability mass above |sigma| > 1, so Gamma(t) >= 0 exactly.
    #    Clamp away tiny negative values: they are dispersive undershoot of the
    #    flux-limited advection in the |sigma| > 1 tail, and become visible only
    #    where the true yield fraction is near the numerical floor (sub-yield
    #    amplitudes, and especially at t = 0 -- maximum strain rate -- where the
    #    yielded mass is at its cycle minimum).  Without this guard the instantaneous
    #    D(t), and in particular D(t=0), can come out spuriously negative; the
    #    magnitude of that artefact tracks the limit-cycle solve tolerance.
    Gamma_t = np.maximum(np.asarray(grid.yield_fraction(P_states), dtype=float), 0.0)
    D_t = alpha * Gamma_t

    # Strain rate over the sampled period (also used as the rate weight below).
    gammadot_t = np.asarray(pmap.protocol.gammadot(t), dtype=float)
    absrate = np.abs(gammadot_t)

    # 4a) Plain time average: uniform-in-time samples => mean is the time average.
    Gamma_bar = float(Gamma_t.mean())
    D_bar = alpha * Gamma_bar

    # 4b) Rate-weighted ("strain-weighted") noise: weight D(t) by |gammadot(t)|,
    #     i.e. by the strain increment |dgamma| = |gammadot| dt.  This is the mean
    #     noise experienced per unit strain traversed; it leans on the high-rate
    #     part of the cycle (where advection sets the balance) yet stays an average,
    #     so it is robust in sign where the single-instant D(t=0) is not.
    wsum = float(absrate.sum())
    D_rate = float(alpha * np.sum(Gamma_t * absrate) / wsum) if wsum > 0.0 else D_bar

    # 4c) Peak-rate noise: D at maximum |gammadot| (t = 0).  Non-negative now.
    D_peak = float(D_t[int(np.argmax(absrate))])

    # 5) Oscillatory advection/diffusion ratios (analogue of gammadot / 2D).
    #    All three share the numerator (the mean strain-rate magnitude), so they
    #    differ only in the noise measure used in the denominator.
    mean_rate = (2.0 / np.pi) * gamma0 * omega          # < |gammadot| > on T/4..3T/4

    def _ratio(D_scale: float) -> float:
        return mean_rate / (2.0 * D_scale) if D_scale > 0.0 else float("inf")

    return CycleNoise(alpha=alpha, gamma0=gamma0, omega=omega,
                      D_bar=D_bar, Gamma_bar=Gamma_bar, r_osc=_ratio(D_bar),
                      t=np.asarray(t, dtype=float), D_t=D_t,
                      gammadot_t=gammadot_t, period=float(pmap.T),
                      converged=bool(laos_result.converged),
                      D_rate=D_rate, r_osc_rate=_ratio(D_rate),
                      D_peak=D_peak, r_osc_peak=_ratio(D_peak))


if __name__ == "__main__":
    # Sample call: fixed frequency, amplitude sweep, alpha in the jammed phase.
    # Run with:  python -m hlmodel.cyclic_noise
    alpha, omega = 0.4, 0.1
    grid = Grid(sigma_max=10, n_per_unit=150)
    parts = build_static_parts(grid)                    # share across the sweep

    print(f"alpha={alpha}, omega={omega}")
    print(f"{'gamma0':>8} {'D_bar':>10} {'D_rate':>10} {'D_peak':>10} "
          f"{'r_osc':>8} {'r_rate':>8} {'r_peak':>8} {'conv':>6}")
    for gamma0 in (0.05, 0.15, 0.25, 0.40, 0.55, 0.75):
        cn = cycle_mean_noise(alpha, gamma0, omega, grid, parts, warn=False)
        print(f"{gamma0:8.2f} {cn.D_bar:10.5f} {cn.D_rate:10.5f} {cn.D_peak:10.5f} "
              f"{cn.r_osc:8.3f} {cn.r_osc_rate:8.3f} {cn.r_osc_peak:8.3f} "
              f"{str(cn.converged):>6}")
