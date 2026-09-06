"""Recovery rheology: the recoverable / unrecoverable strain decomposition.

A recovery test asks the material a physical question rather than a
mathematical one: *release it, and see what strain comes back*.  At a chosen
instant the imposed deformation is stopped, the macroscopic stress is held at
zero, and the strain is followed until the material comes to rest.  What
returns is **recoverable**; what does not is **unrecoverable**.  Unlike G'/G'',
this needs no assumed waveform, no basis functions and no linearity, so it
stays meaningful at any amplitude.  See ``recovery_rheology.md`` for the
background and the experimental protocol.

The identity that makes it cheap here
-------------------------------------
Take the first moment of the HL equation.  Advection contributes gammadot,
diffusion integrates away, the reinjection source sits at sigma = 0 and
contributes nothing, and only the yielding loss survives:

    dSigma/dt = gammadot - Sigma_pl ,   Sigma_pl = integral_{|sigma|>1} sigma P dsigma

Read physically: the stress rises with the imposed strain rate and falls at a
rate set by the stress carried by the *over-threshold* blocks -- the ones about
to yield.  In steady state the two balance, gammadot = Sigma_pl, which
``solve_steady`` reproduces to ~1e-6.

A recovery test is the constraint Sigma = 0.  On a rate-driven solver a stress
constraint would normally mean an implicit solve every step; the identity turns
it into the explicit rule gammadot(t) = Sigma_pl(t), one extra moment per step
and no root-finding.

The protocol, in the model
--------------------------
1. **Instantaneous affine recoil.**  A strain step translates the distribution
   bodily, P(sigma) -> P(sigma - dgamma), so Sigma -> Sigma + dgamma.  Setting
   dgamma = -Sigma zeroes the stress at once.  This is the elastic snap-back,
   of size Sigma / G0.
2. **Delayed recovery.**  Integrate with gammadot = Sigma_pl until the material
   stops moving, accumulating a further strain ``delayed_creep``.
3. The recoverable strain is the net backward excursion,

       gamma_rec = Sigma - delayed_creep .

The delayed term carries **either sign**.  After the shift, whether the
surviving over-threshold population carries net positive or negative stress
depends on the width of the distribution against the size of the shift, so the
material may creep forward (recovering less than the affine Sigma/G0) or keep
recoiling backward (recovering *more*).  ``gamma_rec / (Sigma/G0)`` is
therefore **not** bounded by 1 -- there is no principle requiring it to be,
since the stored energy is <sigma^2>/2, not Sigma^2/2, and a distribution can
carry large internal stress with small net stress.

Note this is a different measurement from ``recoverable.py``, which reads the
reverse strain to the zero-stress crossing on the LAOS down-sweep.  That is a
*driven* reversal -- the material is still being sheared at gammadot(t)
throughout -- whereas this is a *free* release.  Both are legitimate; they
answer different questions and do not agree in general.

Unrecoverable strain needs an origin
------------------------------------
gamma_unrec = gamma(t*) - gamma_rec depends on where the strain zero is put.
Under oscillation there is a natural origin (gamma = 0 at t = 0) and
gamma_unrec is well defined.  Under steady shear gamma(t) grows without bound,
so gamma_unrec does too -- which is not a defect but the correct statement that
all steady flow is unrecoverable.  ``steady_recovery`` therefore reports
gamma_at_release = nan and leaves gamma_unrec undefined; the finite, reportable
quantity there is gamma_rec itself.

Example
-------
>>> from hlmodel.recovery import steady_recovery
>>> r = steady_recovery(alpha=0.4, gammadot=0.1)          # doctest: +SKIP
>>> r.gamma_rec, r.affine_ratio                            # doctest: +SKIP
(0.3326..., 0.982...)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .grid import Grid
from .operators import build_static_parts
from .protocols import require_sine_drive
from .steady import solve_steady
from .transient import TransientStepper

G0 = 1.0  # reduced elastic modulus

# Probability allowed to fall off the domain edge during the affine shift
# before the result is untrustworthy (raise sigma_max).
MASS_LOSS_TOL = 1e-9


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def plastic_stress(P: np.ndarray, grid: Grid) -> float:
    """Sigma_pl = integral_{|sigma|>1} sigma P dsigma, the over-threshold stress.

    Uses ``grid.yield_weight`` (half weight exactly at |sigma| = 1), the same
    weighting as the loss operator, so this is consistent with the Sigma
    evolution identity in the module docstring.
    """
    return float(grid.h * np.sum(grid.sigma * grid.yield_weight * P))


def affine_shift(P: np.ndarray, dgamma: float, grid: Grid) -> np.ndarray:
    """Translate P by a strain step: P(sigma) -> P(sigma - dgamma).

    A step strain is affine, so every block's stress moves by the same dgamma.
    The step is generally not a whole number of cells, so we use a conservative
    donor-cell remap rather than interpolating P: the mass h*P_i of cell i is
    split between cells i+n and i+n+1 with weights (1-f) and f, where
    dgamma/h = n + f.

    This is exact in *both* invariants that matter here.  Mass: the two weights
    sum to one.  First moment: the mass lands at mean position
    (1-f)*sigma_{i+n} + f*sigma_{i+n+1} = sigma_i + dgamma, so Sigma shifts by
    exactly dgamma -- which is what lets step 1 of the protocol zero the stress
    to round-off.  (Plain interpolation of P conserves neither exactly.)

    The shape is mildly smoothed, as any non-grid-aligned remap must be; that
    is a genuine O(h^2) diffusion of the *shape*, not of Sigma.
    """
    n = grid.n
    out = np.zeros(n)
    if dgamma == 0.0:
        return np.array(P, dtype=float)

    k = float(dgamma) / grid.h
    n0 = int(np.floor(k))
    f = k - n0                       # 0 <= f < 1

    for shift, wgt in ((n0, 1.0 - f), (n0 + 1, f)):
        if wgt == 0.0:
            continue
        if shift >= 0:
            if shift < n:
                out[shift:] += wgt * P[:n - shift]
        else:
            if -shift < n:
                out[:n + shift] += wgt * P[-shift:]
    return out


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass
class RecoveryResult:
    """One zero-stress recovery test."""

    alpha: float
    stress: float                 # Sigma at the release instant
    gamma_rec: float              # recoverable strain
    gamma_at_release: float = float("nan")   # nan when there is no strain origin
    delayed_creep: float = 0.0    # integral gammadot dt during the held-stress phase
    plastic_integral: float = 0.0  # integral Sigma_pl dt over the same phase
    converged: bool = True        # material reached rest within t_max -- the
                                  # RECOVERY only; see ``base`` for whether the
                                  # state released from was itself well resolved
    t_recover: float = 0.0        # time to rest
    residual_plastic: float = 0.0  # |Sigma_pl| when stopped
    balance_error: float = 0.0    # residual of the dSigma/dt identity (should ~ 0)
    mass_lost: float = 0.0        # probability pushed off the domain by the shift
    P_final: np.ndarray = field(default=None, repr=False)
    grid: Grid = field(default=None, repr=False)
    # The state this was released from, when there is one (``steady_recovery``
    # stores its SteadyResult here; ``recover_from_state`` leaves it None).
    # ``converged`` above refers to the recovery integration alone, so without
    # this there is no way to tell that a perfectly-converged recovery was run
    # on a resolution-limited base state -- check ``base_ok``.
    base: object = field(default=None, repr=False)

    @property
    def base_ok(self) -> bool:
        """True if the released-from state was itself well resolved.

        ``False`` when the base steady solve hit the D ~ 4h^2 conditioning floor
        (rate too low for the grid -> raise ``n_per_unit``) or was truncated by
        the domain (-> raise ``sigma_max``).  Returns True when there is no base
        to check.  A converged recovery from an unresolved base is still wrong.
        """
        b = self.base
        if b is None:
            return True
        from .steady import BOUNDARY_FRACTION_TOL
        return bool(b.converged) and b.boundary_fraction <= BOUNDARY_FRACTION_TOL

    @property
    def gamma_unrec(self) -> float:
        """Unrecoverable strain; ``nan`` when no strain origin is defined."""
        return self.gamma_at_release - self.gamma_rec

    @property
    def affine_ratio(self) -> float:
        """gamma_rec / (Sigma / G0).

        1 means the recovery was exactly the affine elastic snap-back.  Below 1
        the material crept forward after release; **above 1** it kept recoiling
        (over-recovery).  Not bounded by 1 -- see the module docstring.  Returns
        nan when the release stress is zero.
        """
        if self.stress == 0.0:
            return float("nan")
        return self.gamma_rec / (self.stress / G0)

    @property
    def recoverable_fraction(self) -> float:
        """gamma_rec / gamma(t*); nan when no strain origin is defined."""
        g = self.gamma_at_release
        if not np.isfinite(g) or g == 0.0:
            return float("nan")
        return self.gamma_rec / g


@dataclass
class CycleRecovery:
    """Phase-resolved recovery over one converged LAOS cycle.

    The recovery test destroys the state it probes, so a real experiment must
    re-run the whole protocol once per phase (Rogers' "iteratively punctuated"
    protocol).  Here the converged limit cycle can be branched at every phase
    for free, which is what makes the decomposition cheap in the model and
    laborious in the laboratory.
    """

    alpha: float
    gamma0: float
    omega: float
    phase: np.ndarray             # omega*t at each release, in [0, 2 pi)
    t: np.ndarray                 # release times
    gamma: np.ndarray             # imposed strain gamma0 sin(omega t)
    stress: np.ndarray            # Sigma at release
    gamma_rec: np.ndarray
    gamma_unrec: np.ndarray
    converged: np.ndarray = field(repr=False)
    results: list = field(default_factory=list, repr=False)
    cycle: object = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# the recovery integration
# ---------------------------------------------------------------------------
def recover_from_state(P: np.ndarray, alpha: float, grid: Grid,
                       parts: dict | None = None,
                       gamma_at_release: float = float("nan"),
                       dt: float = 5e-3, t_max: float = 2000.0,
                       tol: float = 1e-9, feedback_steps: float = 20.0,
                       theta: float = 0.5, picard_iters: int = 1,
                       warn: bool = True) -> RecoveryResult:
    """Release ``P`` to zero macroscopic stress and follow the recoil to rest.

    This is the primitive behind :func:`steady_recovery` and
    :func:`laos_recovery`; call it directly to release from any state (a
    startup transient, a custom protocol, a hand-built distribution).

    Parameters
    ----------
    P : ndarray
        Distribution at the release instant.  Assumed normalised.
    gamma_at_release : float
        Imposed strain at the release instant, in whatever origin the caller is
        using.  Leave as ``nan`` (default) when no origin is defined -- then
        ``gamma_rec`` is still returned but ``gamma_unrec`` is ``nan``.
    dt, t_max, tol :
        Time step, cap on the recovery duration, and the |Sigma_pl| threshold
        below which the material counts as at rest.  ``tol`` is the one genuine
        judgement call in the method (full recovery is asymptotic), so it is
        exposed rather than buried; the experimental analogue is how long you
        wait before reading the recovered strain.
    feedback_steps : float
        The held-stress constraint is gammadot = Sigma_pl in the continuum, but
        discrete truncation lets Sigma drift, so a weak proportional term
        -Sigma/(feedback_steps*dt) is added to pin it.  It contributes
        negligibly to the accumulated strain (Sigma stays at ~1e-11) and its
        size is reported through ``balance_error``.
    """
    if parts is None:
        parts = build_static_parts(grid)

    P = np.array(P, dtype=float)
    h, s, w = grid.h, grid.sigma, grid.yield_weight

    Sigma0 = float(grid.first_moment(P))
    mass0 = float(grid.integrate(P))

    # -- step 1: instantaneous affine recoil, dgamma = -Sigma -----------------
    P = affine_shift(P, -Sigma0, grid)
    mass_lost = mass0 - float(grid.integrate(P))
    if warn and abs(mass_lost) > MASS_LOSS_TOL:
        warnings.warn(
            f"affine recoil pushed {mass_lost:.2e} of the probability off the "
            f"domain edge at sigma_max={grid.sigma_max:g}; the recovery is "
            f"unreliable. Increase sigma_max.", RuntimeWarning, stacklevel=2)

    # -- step 2: hold Sigma = 0 and follow the delayed creep ------------------
    stepper = TransientStepper(alpha, grid, parts, theta=theta,
                               picard_iters=picard_iters)
    Tc = float(feedback_steps) * dt
    Sigma_start = float(grid.first_moment(P))

    creep = 0.0            # integral gammadot dt   (the strain that actually moved)
    plastic = 0.0          # integral Sigma_pl dt
    D = None
    Spl = plastic_stress(P, grid)
    converged = False
    n_steps = max(1, int(round(t_max / dt)))
    k = 0

    for k in range(n_steps):
        Spl = plastic_stress(P, grid)
        Sig = float(grid.first_moment(P))
        gammadot = Spl - Sig / Tc

        P, D = stepper.step(P, gammadot, dt, D_old=D)
        creep += gammadot * dt
        plastic += Spl * dt

        # At rest: nothing is above threshold carrying stress, so no further
        # strain can be driven out.  Require a few steps first so a momentarily
        # small Sigma_pl at the release instant cannot stop the run.
        if k >= 10 and abs(Spl) <= tol:
            converged = True
            break

    t_recover = (k + 1) * dt
    Sigma_end = float(grid.first_moment(P))

    # Free check on the whole construction: the first-moment identity says
    # integral gammadot dt = Delta Sigma + integral Sigma_pl dt.  This is
    # independent of the solver internals, so a small residual certifies both
    # the stepping and the constraint.
    balance_error = abs(creep - ((Sigma_end - Sigma_start) + plastic))

    if warn and not converged:
        warnings.warn(
            f"recovery did not reach rest within t_max={t_max:g} "
            f"(|Sigma_pl|={abs(Spl):.2e} > tol={tol:g}); gamma_rec is a bound, "
            f"not a converged value. Raise t_max, or relax tol if the remaining "
            f"tail is negligible.", RuntimeWarning, stacklevel=2)

    return RecoveryResult(
        alpha=float(alpha), stress=Sigma0, gamma_rec=Sigma0 - creep,
        gamma_at_release=float(gamma_at_release), delayed_creep=creep,
        plastic_integral=plastic, converged=converged, t_recover=t_recover,
        residual_plastic=abs(Spl), balance_error=balance_error,
        mass_lost=mass_lost, P_final=P, grid=grid,
    )


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def steady_recovery(alpha: float, gammadot: float, grid: Grid | None = None,
                    parts: dict | None = None, warn: bool = True,
                    **recover_opts) -> RecoveryResult:
    """Release a steady-shear state to zero stress.

    Under steady shear all *flow* is unrecoverable, so the rate split is
    trivial (gammadot_unrec = gammadot, gammadot_rec = 0).  The material still
    carries a fixed stored elastic strain, and that constant -- ``gamma_rec`` --
    is the material function of interest.  It is the modern form of the
    steady-state recoverable compliance J_e^0 = gamma_rec / Sigma, and the
    origin of elastic recoil and die swell.

    ``gamma_at_release`` is left as nan: gamma(t) = gammadot * t grows without
    bound under steady shear, so gamma_unrec is unbounded (correctly -- see the
    module docstring).  Read ``gamma_rec`` and ``affine_ratio`` instead.

    The ``SteadyResult`` released from is kept on ``RecoveryResult.base``, and
    ``base_ok`` folds its two resolution flags into one check.  Use it: a
    recovery can converge perfectly while the state it started from was
    resolution-limited, and ``RecoveryResult.converged`` says nothing about that.

    The moduli are even in the sign of ``gammadot`` (the model is invariant
    under sigma -> -sigma, gammadot -> -gammadot), but the sign is *kept* here
    so that the recovery runs in the physically correct direction.
    """
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)

    base = solve_steady(alpha, gammadot, grid, parts, warn=warn)
    res = recover_from_state(base.P, alpha, grid, parts, warn=warn,
                             **recover_opts)
    res.base = base
    return res


def laos_recovery(alpha: float, gamma0: float, omega: float,
                  grid: Grid | None = None, parts: dict | None = None,
                  n_phases: int = 8, laos_result=None, n_sample: int = 400,
                  steps_per_period: int = 400, warn: bool = True,
                  laos_kwargs: dict | None = None,
                  **recover_opts) -> CycleRecovery:
    """Phase-resolved recovery over a converged LAOS cycle.

    Converges the limit cycle, samples the distribution through one period, and
    runs an independent zero-stress recovery from each of ``n_phases`` evenly
    spaced phases -- the model's version of Rogers' iteratively punctuated
    protocol, with the re-runs replaced by branching off the stored cycle.

    Under a sine drive the strain has a natural origin (gamma = 0 at t = 0), so
    both halves of the decomposition are well defined here:

        gamma(t) = gamma_rec(t) + gamma_unrec(t),   gamma(t) = gamma0 sin(omega t)

    Pass a precomputed ``laos_result`` to reuse a cycle you already have; it
    must have been produced with the same ``steps_per_period``.

    Below the yield strain in the jammed phase the material never yields, so
    Sigma_pl = 0, there is no delayed creep, and the recovery is the pure affine
    recoil: gamma_rec == Sigma to round-off.  Note what that implies.  The cycle
    is *not* unique there -- the period map is effectively the identity, so
    ``solve_laos`` returns whatever the seed relaxed into (see
    ``periodic.solve_laos``), and that state can carry a frozen-in stress offset,
    Sigma(t) = gamma(t) + const.  The decomposition then reports a **constant**
    gamma_unrec equal to minus that offset, and the half-period symmetry of
    gamma_rec is broken by twice it.

    That is the measurement being faithful, not a numerical error: a sub-yield
    jammed state really does hold a residual stress it can never relax.  But it
    is history, not accumulated plasticity, so read the *variation* of
    gamma_unrec across the cycle rather than its value at sub-yield amplitudes.
    """
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)

    # Imported here to keep the module importable from periodic.py's own
    # dependents without a cycle.
    from .periodic import PeriodMap, solve_laos

    if laos_result is None:
        laos_result = solve_laos(alpha, gamma0, omega, grid, parts,
                                 steps_per_period=steps_per_period, warn=warn,
                                 **(laos_kwargs or {}))
    # This routine rebuilds the drive from (gamma0, omega) and uses the sine's
    # strain origin, so it must refuse anything else rather than guess.
    require_sine_drive(getattr(laos_result, "protocol", None), "laos_recovery")

    pmap = PeriodMap(alpha, gamma0, omega, grid, parts,
                     steps_per_period=steps_per_period)
    t, P_states = pmap.sample_states(laos_result.P0, n_sample=n_sample)

    ph = (omega * t) % (2.0 * np.pi)
    targets = np.linspace(0.0, 2.0 * np.pi, int(n_phases), endpoint=False)
    idx = [int(np.argmin(np.abs(((ph - p + np.pi) % (2.0 * np.pi)) - np.pi)))
           for p in targets]

    results = []
    for i in idx:
        gamma_i = float(gamma0 * np.sin(omega * t[i]))
        results.append(recover_from_state(P_states[i], alpha, grid, parts,
                                          gamma_at_release=gamma_i,
                                          warn=warn, **recover_opts))

    return CycleRecovery(
        alpha=float(alpha), gamma0=float(gamma0), omega=float(omega),
        phase=ph[idx], t=t[idx],
        gamma=np.array([r.gamma_at_release for r in results]),
        stress=np.array([r.stress for r in results]),
        gamma_rec=np.array([r.gamma_rec for r in results]),
        gamma_unrec=np.array([r.gamma_unrec for r in results]),
        converged=np.array([r.converged for r in results]),
        results=results, cycle=laos_result,
    )
