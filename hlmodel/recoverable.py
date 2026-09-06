"""Recoverable-strain fraction under LAOS -- the oscillatory analogue of the
steady reverse-shear recoverable ratio.

Under *steady* shear a natural "how elastic is the state" number is the
recoverable fraction obtained by reverse-shearing to zero stress:

    phi_rec^steady = G0 * gamma_star / Sigma ,

where ``gamma_star`` is the reverse strain at which the stress crosses zero and
``Sigma`` the steady stress (see :func:`steady_recoverable_fraction`, and the
``tst_reverse_shear`` experiment).  ``phi = 1`` is fully elastic (the whole
stress is recoverable, ``gamma_star = Sigma / G0``); plasticity during the
reversal makes ``gamma_star < Sigma`` so ``phi < 1``.

The oscillatory counterpart uses the fact that the LAOS **down-sweep**
``T/4 -> 3T/4`` (strain running ``+gamma0 -> -gamma0``) *is itself* a driven
reverse-strain ramp launched from the cycle's turning-point state -- exactly the
same operation as the steady reverse shear, only starting from the oscillatory
state and with a cosinusoidally varying rate.  So we read the recoverable strain
directly off the converged limit cycle: the reverse strain from the turning point
``+gamma0`` to the point on the down-sweep where the stress first crosses zero,

    gamma_rec^LAOS = gamma0 - gamma(sigma = 0 on the down-sweep) ,
    phi_rec^LAOS   = gamma_rec^LAOS / gamma0 .

Normalising by the (bounded) applied amplitude ``gamma0`` makes ``phi_rec`` a
clean fraction, but it ties the measure to an *externally imposed* quantity: as
``gamma0 -> infinity`` the recoverable strain saturates at the recoverable yield
strain while ``gamma0`` keeps growing, so ``phi_rec -> gamma_rec / gamma0 -> 0``
-- a decay that reflects the drive, not the material.

An *intrinsic* recovery measure instead uses only material responses.  Both the
peak (turning-point) stress ``sigma_max`` and the recoverable strain
``gamma_rec`` saturate at large amplitude (the dynamic yield stress and the
recoverable yield strain), so their ratio converges to a material constant.  We
report

    recovery_modulus_ratio = sigma_max / (G0 * gamma_rec) = G_rec / G0 ,

the secant modulus of the unloading path ``G_rec = sigma_max / gamma_rec``
(slope from the turning point down to the zero-stress crossing) relative to the
affine ``G0``.  It reads as ``1`` for affine recovery (the small-amplitude
elastic limit) and ``> 1`` when plasticity stiffens the recovery -- the same
plasticity-steepening that keeps the steady ``G0*gamma_star/Sigma <= 1``.  Its
*deviation from 1*, not its magnitude, is the non-affine-recovery signal.

Linear-limit identity
---------------------
As ``gamma0 -> 0`` the cycle becomes ``sigma(t) = gamma0 |G*| sin(omega t + delta)``
with ``delta = arctan(G''/G')`` the loss angle, and the down-sweep zero crossing
lands at strain ``gamma0 sin(delta)``, so

    phi_rec^LAOS  -->  1 - sin(delta)  =  1 - G'' / |G*| .

This is *not* the geometric storage fraction ``G'_L / |G*| -> cos(delta)`` -- the
two agree only at the elastic (delta -> 0) and viscous (delta -> pi/2) ends, and
differ substantially in between (e.g. by more than 2x at delta = 45 deg).  That
gap is the point: the recoverable fraction is a genuine recovery measurement, not
a Fourier storage projection.  The linear identity is the built-in validation
check exercised in ``__main__``.

Reduced units are used throughout, so the affine modulus is ``G0 = 1``.

Example
-------
>>> from hlmodel.recoverable import laos_recoverable_fraction
>>> r = laos_recoverable_fraction(alpha=0.4, gamma0=0.3, omega=0.5)  # doctest: +SKIP
>>> r.phi_rec, r.phi_linear_pred                                     # doctest: +SKIP
(0.62..., 0.71...)
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from .grid import Grid
from .operators import build_static_parts
from .periodic import LAOSResult, PeriodMap, solve_laos
from .protocols import require_sine_drive
from .steady import solve_steady
from .transient import TransientStepper

G0 = 1.0  # reduced affine elastic modulus


def _first_sign_change(x: np.ndarray, y: np.ndarray) -> float | None:
    """Return the interpolated ``x`` at the first sign change of ``y``.

    Scans for the first straddle ``y[i-1], y[i]`` of opposite sign (either
    direction) and linearly interpolates the crossing.  Returns ``None`` if
    ``y`` never changes sign.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    for i in range(1, len(y)):
        y0, y1 = y[i - 1], y[i]
        if y0 == 0.0:
            return float(x[i - 1])
        if (y0 > 0.0) != (y1 > 0.0):
            frac = abs(y0) / (abs(y0) + abs(y1))
            return float(x[i - 1] + frac * (x[i] - x[i - 1]))
    return None


@dataclass
class LAOSRecoverable:
    """Recoverable-strain fraction of a LAOS limit cycle.

    Attributes
    ----------
    alpha, gamma0, omega : float
        The parameters solved at.
    gamma_rec : float
        Recoverable strain: reverse strain from the turning point ``+gamma0`` to
        the stress zero-crossing on the down-sweep (``nan`` if the stress does
        not cross zero on the down-sweep).
    phi_rec : float
        Primary output ``gamma_rec / gamma0`` in ``[0, 1]``.
    recovery_modulus_ratio : float
        Intrinsic recovery-modulus ratio ``sigma_max / (G0 * gamma_rec) = G_rec/G0``,
        the secant modulus of the unloading path relative to the affine ``G0``.
        Built only from material responses (both ``sigma_max`` and ``gamma_rec``
        saturate at large amplitude), so it converges to a material constant
        rather than decaying like ``phi_rec``.  It is ``1`` for affine recovery
        and ``> 1`` when plasticity stiffens the recovery; the *deviation from 1*
        is the non-affine signal.  (A modulus ratio, not a bounded fraction --
        the steady counterpart is ``Sigma / (G0 * gamma_star)``.)
    sigma_peak, sigma_min : float
        Stress at the strain extrema ``t = T/4`` (``+gamma0``) and ``t = 3T/4``
        (``-gamma0``).
    GpL : float
        Large-strain secant storage modulus ``(sigma_peak - sigma_min)/(2 gamma0)``.
    Gp, Gpp : float
        First-harmonic moduli ``G1'``, ``G1''`` of the cycle.
    delta : float
        Loss angle ``arctan2(Gpp, Gp)`` (radians).
    phi_linear_pred : float
        Linear-limit prediction ``1 - sin(delta)``; ``phi_rec`` converges to it
        as ``gamma0 -> 0``.
    storage_fraction : float
        Geometric storage fraction ``GpL / |G*|`` (``-> cos(delta)`` linearly);
        reported for contrast, it is *not* the recoverable fraction.
    t, sigma, gamma : ndarray
        Phase-resolved waveform over one period ``[0, T)`` (for inspection/plots).
    period : float
        Oscillation period ``T``.
    converged : bool
        Whether the underlying limit-cycle solve converged.
    """

    alpha: float
    gamma0: float
    omega: float
    gamma_rec: float
    phi_rec: float
    recovery_modulus_ratio: float
    sigma_peak: float
    sigma_min: float
    GpL: float
    Gp: float
    Gpp: float
    delta: float
    phi_linear_pred: float
    storage_fraction: float
    t: np.ndarray = field(repr=False)
    sigma: np.ndarray = field(repr=False)
    gamma: np.ndarray = field(repr=False)
    period: float = 0.0
    converged: bool = True


@dataclass
class SteadyRecoverable:
    """Steady reverse-shear recoverable fraction ``G0 * gamma_star / Sigma``.

    Attributes
    ----------
    alpha, gammadot : float
        Coupling and (forward) shear rate the state was prepared at.
    gamma_star : float
        Reverse strain at the stress zero-crossing (``nan`` if none found).
    phi_rec : float
        Recoverable fraction ``G0 * gamma_star / Sigma``.  Usually ``< 1`` because
        plasticity during the reversal steepens the unloading, but it may exceed 1
        marginally when the reversal is fast enough that the net over-threshold
        flux turns negative (see the note on protocol dependence below).
    stress, D : float
        Steady stress ``Sigma`` and self-consistent noise ``D``.
    converged : bool
        Whether the steady solve converged (resolution diagnostic).
    rate_rev : float
        Magnitude of the reverse rate actually used (the drive applied was
        ``-rate_rev``).  Recorded because ``gamma_star`` and ``phi_rec`` are
        **protocol-dependent**: they are properties of the pair (state, probe),
        not of the state alone, so a reported value is incomplete without it.

    The remaining fields describe the reverse branch *past* the zero crossing and
    are ``nan`` unless ``settle_tol`` was supplied.

    gamma_min, stress_min : float
        Strain and stress at the undershoot extremum.  **Parameter-free**: it is
        the point where ``dSigma/dgamma = 0``, equivalently ``M = G0 * gammadot``
        (the plastic release rate exactly balances the reverse elastic loading),
        equivalently where the instantaneous elastic ratio ``1 - M/(G0*gammadot)``
        passes through zero.  Marks peak Bauschinger softening.
    gamma_reverse_first : float
        Strain at which the stress *first* touches ``-Sigma``.  Recorded for
        comparison only -- it is a coincidence of a single moment of P, not
        arrival at the reverse steady state: at this strain the distribution is
        still far from the mirrored steady state (~5% of its initial distance at
        alpha=0.4), and the stress promptly leaves the level again and
        undershoots.  It underestimates the settling strain by roughly 2x.
    gamma_settle : float
        Strain at which the stress *settles* onto ``-Sigma`` to within
        ``settle_tol``: the last exit from the band ``|Sigma + Sigma_ss| <=
        settle_tol * Sigma_ss``.  Last-exit rather than first-entry, because the
        stress crosses the level on the way down, undershoots, and returns.  This
        is the strain required to erase the forward-shear memory, and it is of
        order the yield strain ``sigma_c/G0 = 1``.
    settled_reverse : bool
        Whether the trajectory settled inside the band before
        ``settle_strain_max``.  If False, ``gamma_settle`` is ``nan``.
    settle_tol : float
        The relative tolerance used.
    """

    alpha: float
    gammadot: float
    gamma_star: float
    phi_rec: float
    stress: float
    D: float
    converged: bool = True
    rate_rev: float = float("nan")
    gamma_min: float = float("nan")
    stress_min: float = float("nan")
    gamma_reverse_first: float = float("nan")
    gamma_settle: float = float("nan")
    settled_reverse: bool = False
    settle_tol: float = float("nan")


def laos_recoverable_fraction(alpha: float, gamma0: float, omega: float,
                              grid: Grid | None = None, parts: dict | None = None,
                              steps_per_period: int = 800, n_dense: int = 4001,
                              laos_result: LAOSResult | None = None,
                              warn: bool = True, **laos_kwargs) -> LAOSRecoverable:
    """Recoverable-strain fraction ``phi_rec = gamma_rec / gamma0`` of the LAOS cycle.

    Solves (or reuses) the limit cycle at ``(alpha, gamma0, omega)``, samples the
    stress waveform over one period, and reads the recoverable strain as the
    reverse strain from the turning point ``+gamma0`` to the first stress
    zero-crossing on the down-sweep ``T/4 -> 3T/4``.

    Parameters
    ----------
    alpha, gamma0, omega : float
        Coupling, strain amplitude, angular frequency.
    grid, parts : optional
        Stress grid and cached operators; built with defaults if omitted.  Share
        ``parts`` across a sweep to avoid rebuilding operators.
    steps_per_period : int, default 800
        Time steps per period, used for *both* the limit-cycle solve and the
        sampling pass (kept equal so the sampled cycle matches the converged one).
    n_dense : int, default 4001
        Number of points on the dense down-sweep grid used to locate the stress
        zero-crossing (decouples the crossing resolution from ``steps_per_period``).
    laos_result : LAOSResult, optional
        A precomputed converged cycle at the *same* parameters and
        ``steps_per_period``; supply it to skip the re-solve (e.g. continuation).
    warn : bool, default True
        Forwarded to :func:`solve_laos`.
    **laos_kwargs
        Extra keyword arguments forwarded to :func:`solve_laos`.

    Returns
    -------
    LAOSRecoverable
    """
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)

    # 1) Converged limit cycle (reuse if supplied).
    require_sine_drive(getattr(laos_result, "protocol", None), "laos_recoverable_fraction")
    if laos_result is None:
        laos_result = solve_laos(alpha, gamma0, omega, grid, parts,
                                 steps_per_period=steps_per_period,
                                 warn=warn, **laos_kwargs)

    # 2) Sample the stress waveform over one period (same step count as the solve).
    pmap = PeriodMap(alpha, gamma0, omega, grid, parts,
                     steps_per_period=steps_per_period)
    T = float(pmap.T)
    t, sigma = pmap.sample_stress(laos_result.P0, n_sample=10 * steps_per_period)
    t = np.asarray(t, float)
    sigma = np.asarray(sigma, float)
    gamma = gamma0 * np.sin(omega * t)

    # 3) Stress at the strain extrema and the large-strain secant modulus.
    sigma_peak = float(np.interp(0.25 * T, t, sigma))   # t = T/4, gamma = +gamma0
    sigma_min = float(np.interp(0.75 * T, t, sigma))    # t = 3T/4, gamma = -gamma0
    GpL = (sigma_peak - sigma_min) / (2.0 * gamma0)

    # 4) Recoverable strain: first stress zero-crossing on the down-sweep T/4->3T/4.
    td = np.linspace(0.25 * T, 0.75 * T, int(n_dense))
    sd = np.interp(td, t, sigma)
    tzc = _first_sign_change(td, sd)
    if tzc is None:
        gamma_rec = float("nan")
    else:
        gamma_zc = gamma0 * np.sin(omega * tzc)
        gamma_rec = gamma0 - gamma_zc
    phi_rec = gamma_rec / gamma0
    # Intrinsic recovery-modulus ratio G_rec/G0 = sigma_max / (G0 * gamma_rec).
    # sigma_max and gamma_rec are both material responses that saturate at large
    # amplitude, so unlike phi_rec (normalised by the external gamma0, decaying as
    # 1/gamma0) this converges to a material constant.  == 1 for affine recovery,
    # > 1 when plasticity stiffens it.
    recovery_modulus_ratio = (sigma_peak / (G0 * gamma_rec)) if gamma_rec > 0.0 else float("nan")

    # 5) Loss angle and the linear-limit prediction / storage-fraction contrast.
    Gp = float(laos_result.G1_prime)
    Gpp = float(laos_result.G1_doubleprime)
    delta = float(np.arctan2(Gpp, Gp))
    phi_linear_pred = 1.0 - np.sin(delta)
    Gmag = np.hypot(Gp, Gpp)
    storage_fraction = (GpL / Gmag) if Gmag > 0.0 else float("nan")

    return LAOSRecoverable(
        alpha=alpha, gamma0=gamma0, omega=omega,
        gamma_rec=float(gamma_rec), phi_rec=float(phi_rec),
        recovery_modulus_ratio=float(recovery_modulus_ratio),
        sigma_peak=sigma_peak, sigma_min=sigma_min, GpL=float(GpL),
        Gp=Gp, Gpp=Gpp, delta=delta, phi_linear_pred=float(phi_linear_pred),
        storage_fraction=float(storage_fraction),
        t=t, sigma=sigma, gamma=gamma, period=T,
        converged=bool(laos_result.converged))


def steady_recoverable_fraction(alpha: float, gammadot: float,
                                grid: Grid | None = None, parts: dict | None = None,
                                theta: float = 0.5, strain_margin: float = 1.5,
                                strain_step: float = 2e-3, dt_max: float = 0.1,
                                rate_rev: float | None = None,
                                settle_tol: float | None = None,
                                settle_strain_max: float = 8.0,
                                warn: bool = True) -> SteadyRecoverable:
    """Steady recoverable fraction ``G0 * gamma_star / Sigma`` via reverse shear.

    Solves the steady state at ``(alpha, gammadot)``, then reverses the drive to
    ``-rate_rev`` and integrates until the stress crosses zero; ``gamma_star`` is
    the reverse strain at that crossing.

    Protocol dependence
    -------------------
    The forward rate prepares the *state*; ``rate_rev`` sets the *probe*.  These are
    independent, and ``gamma_star`` depends on both.  Integrating the model's
    constitutive law ``dSigma/dt = G0 * gammadot - M``, with
    ``M = integral_{|sigma|>1} sigma P dsigma`` the over-threshold stress moment,
    from release to the zero-stress crossing gives the exact statement

        G0 * gamma_star = Sigma - integral_0^{t*} M dt ,

    where ``t* = gamma_star / rate_rev``.  Both terms grow with the forward rate but
    enter with opposite signs, and the truncation time ``t*`` is set by the probe,
    so ``gamma_star`` is a property of the (state, probe) pair rather than of the
    state.  Concretely, at ``alpha = 0.4`` a single state can be made to report
    anything from ``0.25 * Sigma/G0`` to ``Sigma/G0`` by varying ``rate_rev`` alone;
    and at ``rate_rev ~ 0.03`` the map from forward rate to ``gamma_star`` is
    non-monotonic, so distinct forward states share one value.  Sweep ``rate_rev``
    and report the curve rather than a single point.  For a probe-free recoverable
    strain (release the stress instead of driving through it) see
    :func:`hlmodel.recoil.recoil_recovery`.

    Parameters
    ----------
    alpha, gammadot : float
        Coupling and forward shear rate; these prepare the state.
    grid, parts : optional
        Stress grid and cached operators; built with defaults if omitted.
    theta : float, default 0.5
        Crank-Nicolson parameter for the transient reversal.
    strain_margin : float, default 1.5
        Reverse until strain ``strain_margin * Sigma / G0``.  The pure-elastic
        unload reaches zero stress at exactly ``Sigma / G0`` and plasticity during
        the reversal normally shortens that, so a margin > 1 brackets the crossing;
        the margin also covers the fast-reversal case where ``gamma_star`` may
        marginally exceed ``Sigma / G0``.
    strain_step : float, default 2e-3
        Target reverse strain per time step; ``dt = min(dt_max, strain_step/rate_rev)``
        so the crossing is well resolved while ``dt`` stays below the explicit
        reaction's stability limit.
    dt_max : float, default 0.1
        Upper bound on the time step.  Note ``dt = min(dt_max, strain_step/rate_rev)``,
        so ``dt_max`` becomes the binding constraint once
        ``rate_rev < strain_step / dt_max`` (0.02 with the defaults).  The defaults
        are tuned for the coupled convention ``rate_rev = |gammadot|``; when
        sweeping ``rate_rev`` down to slow probes they under-resolve the crossing
        (at ``alpha = 0.4``, ``gammadot = 0.1``: 0.06% error at ``rate_rev = 0.03``,
        0.5% at 0.01, 2.7% at 0.003).  For a ``rate_rev`` sweep pass something like
        ``strain_step=2e-4, dt_max=5e-3`` and check convergence.  The same applies
        to the ``settle_tol`` landmarks: ``gamma_star`` is converged at the
        defaults, but ``gamma_min`` and especially ``gamma_settle`` sit on
        low-amplitude late features and are not.  At ``alpha = 0.4``,
        ``gammadot = 0.06`` the default dt (0.033) gives ``gamma_settle`` 1.1% low
        and ``gamma_min`` 0.3% low relative to ``dt_max=1e-3``; pass
        ``dt_max=5e-3`` or finer when the landmarks matter.
    rate_rev : float, optional
        Magnitude of the reverse shear rate; the drive applied is ``-rate_rev``.
        The sign is ignored (``-0.1`` and ``0.1`` both mean "reverse at 0.1").
        Defaults to ``abs(gammadot)``, i.e. reversing at the same rate the state
        was sheared at -- a convention, not a requirement (see above).
    settle_tol : float, optional
        Relative tolerance for arrival at the reverse steady state ``-Sigma``.
        Supplying it switches on the reverse-branch landmarks (``gamma_min``,
        ``gamma_reverse_first``, ``gamma_settle``) and extends the integration
        window, which costs more time; leave it ``None`` (default) for the
        original zero-crossing-only behaviour.  A tolerance is unavoidable for
        ``gamma_settle`` because the approach to ``-Sigma`` is asymptotic.  The
        dependence is mild and roughly logarithmic, but not negligible: at
        ``alpha = 0.4``, ``gammadot = 0.06`` it runs 1.020 (3e-2), 1.145 (1e-2),
        1.224 (3e-3), 1.595 (1e-3), 1.721 (3e-4), 2.018 (1e-4) -- roughly a
        doubling over three decades.  It also advances in steps rather than
        smoothly, because the stress reaches ``-Sigma`` by a *damped oscillation*
        (crossing the level going up near strain 1.29, overshooting to +5e-4 near
        1.45, then decaying through further sign changes), so each lobe that pokes
        outside the band pushes the last exit out to the next one.  1e-3 is a
        reasonable choice; below ~1e-4 the answer is set by those late lobes and
        ultimately by round-off, so treat it as unresolved there.  ``gamma_min``
        needs no tolerance at all.
    settle_strain_max : float, default 8
        Cap on the extended reverse strain when ``settle_tol`` is given.  Settling
        occurs near the yield strain ``sigma_c/G0 = 1``, so 8 is ample; raise it
        if ``settled_reverse`` comes back False.
    warn : bool, default True
        Forwarded to :func:`solve_steady`, and used to warn if the reverse branch
        did not settle within ``settle_strain_max``.

    Returns
    -------
    SteadyRecoverable
    """
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)

    st = solve_steady(alpha, gammadot, grid, parts, warn=warn)
    Sigma, D, P = float(st.stress), float(st.D), st.P
    # The forward rate prepares the state; the reverse rate is the probe.  They
    # default to the same magnitude, but only by convention.
    rate = abs(float(gammadot)) if rate_rev is None else abs(float(rate_rev))
    if rate == 0.0:
        raise ValueError(
            "rate_rev must be non-zero: it sets both the reverse drive and the "
            "integration window t_end = strain_margin * Sigma / (G0 * rate_rev).")

    max_reverse_strain = strain_margin * Sigma / G0
    t_end = max_reverse_strain / rate
    dt = min(float(dt_max), float(strain_step) / rate)
    n_steps = max(1, int(round(t_end / dt)))
    dt = t_end / n_steps
    record_every = max(1, n_steps // 5000)

    # The reverse-branch landmarks live far past the zero crossing: the reverse
    # steady state is only reached after a strain of order the yield strain
    # sigma_c/G0 = 1, whereas the default window is only strain_margin*Sigma/G0.
    # Extend by ADDING steps, so dt and record_every -- and therefore gamma_star
    # -- stay bit-identical to the default path.
    if settle_tol is not None:
        n_total = max(n_steps, int(np.ceil(float(settle_strain_max) / (rate * dt))))
        t_end = n_total * dt

    stepper = TransientStepper(alpha, grid, parts, theta=theta, picard_iters=1)
    traj = stepper.run(P, gammadot_func=lambda t: -rate, t_end=t_end, dt=dt,
                       t0=0.0, record_every=record_every, record_states=False)

    tzc = _first_sign_change(traj.t, traj.stress)
    if tzc is None:
        gamma_star = float("nan")
    else:
        gamma_star = tzc * rate
    phi_rec = (G0 * gamma_star / Sigma) if Sigma != 0.0 else float("nan")

    g_min = s_min = g_rev1 = g_settle = float("nan")
    settled = False
    if settle_tol is not None:
        tt = np.asarray(traj.t, dtype=float)
        S = np.asarray(traj.stress, dtype=float)

        # (a) undershoot extremum, dSigma/dgamma = 0  <=>  M = G0*gammadot.
        #     Parameter-free; parabolic refinement of the discrete minimum.
        i = int(np.argmin(S))
        if 0 < i < S.size - 1:
            y0, y1, y2 = S[i - 1], S[i], S[i + 1]
            den = y0 - 2.0 * y1 + y2
            d = 0.5 * (y0 - y2) / den if den != 0.0 else 0.0
            g_min = (tt[i] + d * (tt[i + 1] - tt[i])) * rate
            s_min = y1 - 0.25 * (y0 - y2) * d

        # (b) first touch of -Sigma (a stress coincidence, not arrival).
        t1 = _first_sign_change(tt, S + Sigma)
        if t1 is not None:
            g_rev1 = t1 * rate

        # (c) settling = LAST exit from the band around -Sigma.
        band = float(settle_tol) * Sigma
        dev = np.abs(S + Sigma)
        out = np.nonzero(dev > band)[0]
        if out.size == 0:
            g_settle, settled = 0.0, True
        elif out[-1] < S.size - 1:
            j = int(out[-1])
            den = dev[j] - dev[j + 1]
            f = (dev[j] - band) / den if den != 0.0 else 0.0
            g_settle = (tt[j] + f * (tt[j + 1] - tt[j])) * rate
            settled = True
        if warn and not settled:
            warnings.warn(
                f"steady_recoverable_fraction: reverse branch did not settle to "
                f"within settle_tol={settle_tol:g} of -Sigma within "
                f"settle_strain_max={settle_strain_max:g} "
                f"(alpha={alpha:g}, gammadot={gammadot:g}). Increase "
                f"settle_strain_max.", RuntimeWarning, stacklevel=2)

    return SteadyRecoverable(alpha=alpha, gammadot=gammadot,
                             gamma_star=float(gamma_star), phi_rec=float(phi_rec),
                             stress=Sigma, D=D, converged=bool(st.converged),
                             rate_rev=float(rate),
                             gamma_min=float(g_min), stress_min=float(s_min),
                             gamma_reverse_first=float(g_rev1),
                             gamma_settle=float(g_settle),
                             settled_reverse=bool(settled),
                             settle_tol=float(settle_tol) if settle_tol is not None
                             else float("nan"))


# --------------------------------------------------------------------------
# Steady sweep over shear rates (parallel; points are independent)
# --------------------------------------------------------------------------
@dataclass
class SteadyRecoverableSweep:
    """Steady recoverable fraction over a range of shear rates.

    All arrays are aligned to the input ``gammadot`` order.
    """

    alpha: float
    gammadot: np.ndarray
    gamma_star: np.ndarray
    phi_rec: np.ndarray
    stress: np.ndarray
    D: np.ndarray
    converged: np.ndarray


def _steady_recoverable_task(args):
    """Worker for the parallel steady sweep: rebuilds the grid inside the process
    (only a small spec is pickled) and pins BLAS/OpenMP to one thread."""
    from .sweep import _limit_threads, _make_grid  # lazy: only when sweeping parallel
    alpha, gammadot, grid_spec, opts = args
    with _limit_threads():
        grid = _make_grid(grid_spec)
        parts = build_static_parts(grid)
        s = steady_recoverable_fraction(alpha, gammadot, grid, parts, warn=False, **opts)
    return s


def steady_recoverable_sweep(alpha: float, gammadots,
                             grid: Grid | None = None, parts: dict | None = None,
                             n_workers: int | None = None, warn: bool = True,
                             **kwargs) -> SteadyRecoverableSweep:
    """Steady recoverable fraction ``G0*gamma_star/Sigma`` across many shear rates.

    The rates are independent, so with ``n_workers > 1`` they are farmed out to a
    process pool (via :func:`hlmodel.sweep.parallel_map`), each worker rebuilding
    the grid from its spec and pinning BLAS to a single thread.  With
    ``n_workers=1`` the sweep runs serially and reuses the passed ``grid``/``parts``
    directly; ``n_workers=None`` uses ``os.cpu_count()``.  Low rates cost more (a
    longer reverse transient), so tasks are dispatched longest-first for balance.

    Extra keyword arguments are forwarded to :func:`steady_recoverable_fraction`
    (e.g. ``strain_margin``, ``strain_step``, ``dt_max``, ``theta``).  Results are
    returned aligned to the input ``gammadots`` order.
    """
    gds = np.atleast_1d(np.asarray(gammadots, dtype=float))
    if grid is None:
        grid = Grid()
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(int(n_workers), gds.size))

    if n_workers == 1:
        if parts is None:
            parts = build_static_parts(grid)
        results = [steady_recoverable_fraction(alpha, float(gd), grid, parts,
                                               warn=False, **kwargs)
                   for gd in gds]
    else:
        from .sweep import parallel_map  # lazy import of the parallel infrastructure
        spec = (grid.sigma_max, grid.n_per_unit)
        opts = dict(kwargs)
        args = [(alpha, float(gd), spec, opts) for gd in gds]
        cost = [1.0 / max(float(gd), 1e-12) for gd in gds]  # low rate -> longer transient
        results = parallel_map(_steady_recoverable_task, args,
                               n_workers=n_workers, cost=cost)

    gamma_star = np.array([r.gamma_star for r in results])
    phi = np.array([r.phi_rec for r in results])
    stress = np.array([r.stress for r in results])
    D = np.array([r.D for r in results])
    conv = np.array([r.converged for r in results])

    if warn and not conv.all():
        bad = gds[~conv]
        warnings.warn(
            f"{bad.size} steady point(s) <= {bad.max():g} did not converge (the "
            f"self-consistent D hit the grid's resolution floor); refine n_per_unit "
            f"to reach lower rates. See the returned `converged` array.",
            RuntimeWarning, stacklevel=2)

    return SteadyRecoverableSweep(alpha, gds, gamma_star, phi, stress, D, conv)


# --------------------------------------------------------------------------
# LAOS amplitude sweep (sequential, warm-started continuation over gamma0)
# --------------------------------------------------------------------------
@dataclass
class LAOSRecoverableSweep:
    """LAOS recoverable fraction over a range of amplitudes at fixed omega.

    All arrays are aligned to the input ``gamma0`` order.
    """

    alpha: float
    omega: np.ndarray
    gamma0: np.ndarray
    gamma_rec: np.ndarray
    phi_rec: np.ndarray
    recovery_modulus_ratio: np.ndarray
    sigma_peak: np.ndarray
    GpL: np.ndarray
    Gp: np.ndarray
    Gpp: np.ndarray
    delta: np.ndarray
    phi_linear_pred: np.ndarray
    storage_fraction: np.ndarray
    residual: np.ndarray
    converged: np.ndarray
    method: list


def laos_recoverable_sweep(alpha: float, gamma0s, omega: float,
                           grid: Grid | None = None, parts: dict | None = None,
                           P_init=None, descending: bool = True,
                           steps_per_period: int = 800, n_dense: int = 4001,
                           progress: bool = False,
                           warn: bool = True, **solver_opts) -> LAOSRecoverableSweep:
    """LAOS recoverable fraction over an amplitude sweep at fixed ``omega``.

    Amplitudes are solved **sequentially**, each warm-started from the previous
    converged limit cycle (continuation), exactly as
    :func:`hlmodel.sweep.amplitude_sweep`.  By default the sweep runs from large
    to small amplitude (``descending=True``), starting where the period map is
    best conditioned and threading ``P0`` down to the small-amplitude points that
    would otherwise be slow or fail to converge cold.  This is deliberately *not*
    parallelised: the continuation makes each point depend on the previous one.

    Each amplitude is solved once with :func:`solve_laos` (providing both the
    continuation ``P0`` and the moduli); that result is reused by
    :func:`laos_recoverable_fraction` so the cycle is not re-solved.  The first
    point cold-starts from ``solve_laos``'s default seed unless ``P_init`` is
    given.  ``steps_per_period`` is shared between the solve and the sampling pass.

    Extra keyword arguments are forwarded to :func:`solve_laos` (e.g. ``tol``,
    ``method``, ``dt_max``, ``verbose``).  Results are returned aligned to the
    input ``gamma0s`` order.

    Set ``progress=True`` for a concise one-line-per-amplitude tick (index,
    ``gamma0``, ``phi_rec``, convergence, and per-point wall time) printed in the
    order the amplitudes are actually solved -- useful because this sweep is
    sequential and the small-amplitude tail can be slow.  For the detailed
    per-Newton-iteration trace instead, forward ``verbose=True``.
    """
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)
    g0s = np.atleast_1d(np.asarray(gamma0s, dtype=float))
    order = np.argsort(g0s)[::-1] if descending else np.argsort(g0s)
    P = None if P_init is None else np.asarray(P_init, dtype=float)

    n = g0s.size
    gamma_rec = np.empty(n); phi = np.empty(n); rmr = np.empty(n)
    speak = np.empty(n); gpl = np.empty(n); gp = np.empty(n); gpp = np.empty(n)
    delta = np.empty(n); phlin = np.empty(n); store = np.empty(n)
    resid = np.empty(n); conv = np.empty(n, dtype=bool); meth = [None] * n

    if progress:
        arrow = "descending" if descending else "ascending"
        print(f"[laos_recoverable_sweep] alpha={alpha:g} omega={omega:g}: "
              f"{n} amplitude(s), {arrow} gamma0, continuation warm-start",
              flush=True)

    for k, idx in enumerate(order, start=1):
        g0 = float(g0s[idx])
        t0 = time.perf_counter()
        r = solve_laos(alpha, g0, float(omega), grid, parts, P_init=P,
                       steps_per_period=steps_per_period, warn=False, **solver_opts)
        P = r.P0  # warm start for the next (smaller) amplitude
        rec = laos_recoverable_fraction(alpha, g0, float(omega), grid, parts,
                                        steps_per_period=steps_per_period,
                                        n_dense=n_dense, laos_result=r, warn=False)
        gamma_rec[idx] = rec.gamma_rec; phi[idx] = rec.phi_rec
        rmr[idx] = rec.recovery_modulus_ratio; speak[idx] = rec.sigma_peak
        gpl[idx] = rec.GpL; gp[idx] = rec.Gp; gpp[idx] = rec.Gpp
        delta[idx] = rec.delta; phlin[idx] = rec.phi_linear_pred
        store[idx] = rec.storage_fraction
        resid[idx] = r.residual; conv[idx] = r.converged; meth[idx] = r.method

        if progress:
            flag = "ok" if r.converged else "NOT CONVERGED"
            print(f"  [{k:>2}/{n}] gamma0={g0:8.4f}  phi_rec={rec.phi_rec:7.4f}  "
                  f"{flag:>13}  ({time.perf_counter() - t0:5.1f}s, {r.method})",
                  flush=True)

    if warn and not conv.all():
        bad = g0s[~conv]
        warnings.warn(
            f"{bad.size} of {n} LAOS amplitude(s) did not converge (relative "
            f"cycle-closure error > 1e-3); see the returned `converged` array. "
            f"Try more steps_per_period or a finer amplitude spacing for the "
            f"continuation.", RuntimeWarning, stacklevel=2)

    return LAOSRecoverableSweep(alpha, np.atleast_1d(float(omega)), g0s,
                                gamma_rec, phi, rmr, speak, gpl, gp, gpp,
                                delta, phlin, store, resid, conv, meth)


if __name__ == "__main__":
    # Run with:  python -m hlmodel.recoverable
    import numpy as np

    alpha, omega = 0.4, 0.5
    grid = Grid(sigma_max=10, n_per_unit=150)
    parts = build_static_parts(grid)

    # --- 1) Steady sweep over shear rates (PARALLEL) ----------------------------
    print(f"alpha={alpha}")
    print("Steady recoverable fraction  G0*gamma_star/Sigma  (parallel rate sweep):")
    gammadots = np.logspace(-3, -0.5, 6)
    ss = steady_recoverable_sweep(alpha, gammadots, grid, parts)
    print(f"{'gammadot':>10} {'phi_rec':>9} {'gamma*':>9} {'Sigma':>9} {'D':>9} {'conv':>6}")
    for i in range(ss.gammadot.size):
        print(f"{ss.gammadot[i]:10.3e} {ss.phi_rec[i]:9.4f} {ss.gamma_star[i]:9.4f} "
              f"{ss.stress[i]:9.4f} {ss.D[i]:9.4f} {str(ss.converged[i]):>6}")

    # --- 2) LAOS amplitude sweep (SEQUENTIAL, continuation from large gamma0) ----
    print(f"\nLAOS recoverable fraction, omega={omega}  (continuation, descending gamma0):")
    gamma0s = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.0])
    ls = laos_recoverable_sweep(alpha, gamma0s, omega, grid, parts,
                                steps_per_period=600)
    print(f"{'gamma0':>8} {'phi_rec':>9} {'1-sin(d)':>9} {'rel.err':>9} "
          f"{'delta[deg]':>10} {'cos(d)=GpL/|G*|':>16} {'conv':>6}")
    for i in range(ls.gamma0.size):
        rel = abs(ls.phi_rec[i] - ls.phi_linear_pred[i]) / max(abs(ls.phi_linear_pred[i]), 1e-12)
        print(f"{ls.gamma0[i]:8.2f} {ls.phi_rec[i]:9.4f} {ls.phi_linear_pred[i]:9.4f} "
              f"{rel:9.1e} {np.degrees(ls.delta[i]):10.3f} "
              f"{ls.storage_fraction[i]:16.4f} {str(ls.converged[i]):>6}")

    # --- 3) Figure: construction + phi_rec vs amplitude -------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                    # pragma: no cover
        print(f"\n[skip plot: matplotlib unavailable: {exc}]")
        raise SystemExit(0)

    # One detailed waveform for the Lissajous panel (reuse the sweep's omega).
    rr = laos_recoverable_fraction(alpha, 0.4, omega, grid, parts,
                                   steps_per_period=600, warn=False)
    T = rr.period

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax0.plot(rr.gamma / rr.gamma0, rr.sigma, color="0.4", lw=1.5)
    down = (rr.t >= 0.25 * T) & (rr.t <= 0.75 * T)
    ax0.plot(rr.gamma[down] / rr.gamma0, rr.sigma[down], "C0", lw=2.5,
             label="down-sweep  T/4$\\to$3T/4")
    gamma_zc = rr.gamma0 - rr.gamma_rec
    ax0.axhline(0.0, color="k", lw=0.7)
    ax0.plot([1.0], [rr.sigma_peak], "C3o", label="turning point $+\\gamma_0$")
    ax0.plot([gamma_zc / rr.gamma0], [0.0], "C2s", label="$\\sigma=0$ crossing")
    ax0.annotate("", xy=(gamma_zc / rr.gamma0, 0.0), xytext=(1.0, 0.0),
                 arrowprops=dict(arrowstyle="<->", color="C2"))
    ax0.text(0.5 * (1.0 + gamma_zc / rr.gamma0), 0.03 * rr.sigma_peak,
             "$\\gamma_{rec}/\\gamma_0$", color="C2", ha="center", va="bottom")
    ax0.set_xlabel("strain  $\\gamma/\\gamma_0$")
    ax0.set_ylabel("stress  $\\sigma$")
    ax0.set_title(f"Lissajous, $\\gamma_0$={rr.gamma0}, $\\omega$={omega}")
    ax0.legend(fontsize=8, loc="upper left")

    # (b) recoverable fraction vs amplitude, from the LAOS sweep results
    ax1.semilogx(ls.gamma0, ls.phi_rec, "C0o-", label="$\\phi_{rec}=\\gamma_{rec}/\\gamma_0$")
    ax1.semilogx(ls.gamma0, ls.phi_linear_pred, "C1^--", label="$1-\\sin\\delta$ (linear pred.)")
    ax1.semilogx(ls.gamma0, ls.storage_fraction, "C7s:", label="$G'_L/|G^*|$ (storage, $\\to\\cos\\delta$)")
    ax1.set_xlabel("amplitude  $\\gamma_0$")
    ax1.set_ylabel("fraction")
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f"Recoverable vs storage fraction, $\\alpha$={alpha}")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    out = "recoverable_fraction_demo.png"
    fig.savefig(out, dpi=130)
    print(f"\n[saved figure: {out}]")
