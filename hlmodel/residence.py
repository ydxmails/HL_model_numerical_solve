"""Residence-time (age-at-yield) distribution for the Hebraud-Lequeux model.

A block is reborn at sigma = 0 and, in the self-consistent steady background,
drifts at the strain rate ``gammadot``, diffuses with noise ``D``, and is
absorbed (yields) at rate 1/tau = 1 whenever |sigma| > 1.  The distribution of
its age ``a`` at the moment it yields is the absorption-time density of a cohort
launched from sigma = 0 into the model's generator *without the reinjection
source* (a tagged block is not re-added when *others* yield):

    d_a Q = -gammadot d_sigma Q + D d2_sigma Q - H(|sigma|-1) Q,   Q(.,0) = delta(sigma),

with survival S(a) = integral Q dsigma and age-at-yield density
psi(a) = -dS/da = integral H(|sigma|-1) Q dsigma.

Two exact relations (both checked in tests/test_residence.py):

* the steady state is the age-integral of the cohort,
      P(sigma) = Gamma * integral_0^inf Q(sigma, a) da ;
* the mean age is exactly
      <a> = 1/Gamma = alpha/D            (steady shear)
      <a> = 1/<Gamma>                    (oscillation, cycle-mean rate),
  by Little's law: one block (integral P = 1), reborn at rate Gamma.

The long-time tail of psi decays at the spectral gap of the no-source generator,
which -> the bare yielding rate 1 at high shear.  At low rate escape is
diffusion-limited and the tail is much slower.

**Validity.**  The distribution is only meaningful when the material actually
yields (Gamma > 0 appreciably).  In the jammed phase (alpha < 1/2) at rest, or at
an oscillation amplitude below the yield strain (gamma0 <~ sigma_y), the material
essentially never yields: Gamma -> 0 and the residence time diverges (that
divergence *is* the glass).  Both entry points detect this -- ``defined`` /
``captured_fraction`` flag it and a warning is emitted -- rather than returning a
meaningless capped curve.

Under oscillation the background D(t) = alpha*Gamma(t) and gammadot(t) are the
limit-cycle values, so the residence time depends on the *birth phase*: we return
the phase-averaged density (weighted by the rebirth rate Gamma(t)) and the
birth-phase-resolved densities.  This is the temporal companion of the
yield-stress distribution rho_ac (observables.py): rho_ac is *where* blocks yield,
psi is *when*.

The same cohort also carries the *non-affine* stress trajectory of an element
before it yields.  Under shear a block loads affinely -- its stress advects at the
strain rate (the -gammadot d_sigma term).  Subtracting that affine loading from the
survivor stress moments of Q leaves the non-affine motion: a mean displacement and
a mean-square displacement that grow diffusively (MSD ~ 2 D a) at short age and are
truncated by absorption at the yield threshold at long age (a stress-space "cage").
Under oscillation the affine part is removed by the co-deforming transform
xi = sigma - gamma(t).  See ``nonaffine_stress_trajectory`` (steady) and
``cycle_nonaffine_stress_trajectory`` (birth-phase-resolved, oscillation).  Folding
the age-at-yield density onto one period gives the *yield-phase* distribution -- when
in the cycle a block breaks (``cycle_yield_phase_distribution``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve_banded

from .grid import Grid
from .operators import (advection_matrix, advection_matrix_frozen,
                        build_static_parts)
from .steady import solve_steady
from .protocols import require_sine_drive

# Surviving fraction below which the survivor-conditioned moments m/S are pure
# round-off (and physically vacuous); the trajectory window is cut there.
SURVIVAL_FLOOR = 1e-10


# ---------------------------------------------------------------------------
# Steady shear
# ---------------------------------------------------------------------------
@dataclass
class ResidenceTimeDistribution:
    """Age-at-yield density psi(a) for a block reborn at sigma = 0 under steady shear."""
    alpha: float
    gammadot: float
    a: np.ndarray                              # age grid
    psi: np.ndarray                            # age-at-yield density (int psi da = 1)
    survival: np.ndarray                       # S(a) = P(not yet yielded by age a)
    Gamma: float                               # steady yield rate; mean = 1/Gamma
    D: float
    mean: float                                # numeric <a> = int S da (equals 1/Gamma)
    occupation: np.ndarray = field(repr=False)  # R(sigma)=int Q da; Gamma*R = P_steady
    grid: Grid = field(repr=False)
    spectral_gap: float | None = None          # tail decay rate (if requested)
    defined: bool = True                       # False if the material does not yield

    @property
    def mean_exact(self) -> float:
        """Exact mean residence time 1/Gamma (= inf in the frozen jammed state)."""
        return 1.0 / self.Gamma if self.Gamma > 0.0 else float("inf")

    @property
    def captured_fraction(self) -> float:
        """1 - S(a_max): fraction of the distribution covered by the age window."""
        return float(1.0 - self.survival[-1]) if self.survival.size else 0.0


def _cohort_steady(D: float, gammadot: float, grid: Grid, parts: dict,
                   da: float, nst: int, P_ref: np.ndarray | None = None):
    """Integrate the no-source cohort Q(sigma, a) from a delta at sigma = 0.

    Crank-Nicolson in age (LU factored once, constant steady background).  Returns
    per-age arrays S(a) = int Q, psi(a) = int_{|sigma|>1} Q and the stress moments
    m1(a) = int sigma Q, m2(a) = int sigma^2 Q, plus the occupation-time integral
    R(sigma) = int Q da.  Shared by ``residence_time_distribution`` (age-at-yield)
    and ``nonaffine_stress_trajectory`` (survivor stress moments).

    ``P_ref`` is the steady distribution this cohort lives in.  When given, the
    advection is the van Leer flux with the limiter frozen at ``P_ref`` -- the
    linearisation that matches ``solve_steady``'s (TVD) stationary state exactly
    while keeping the generator linear, which the identity P = Gamma * int Q da
    requires.  Without it the cohort would be advected by the first-order upwind
    operator while the steady state it is compared against is not, and the
    identity would fail at O(h).
    """
    n, h = grid.n, grid.h
    sig = grid.sigma
    # Same weights as parts["Yield"] uses, so psi matches the loss it models.
    mask = grid.yield_weight
    A = (advection_matrix(gammadot, grid) if P_ref is None
         else advection_matrix_frozen(gammadot, grid, P_ref))
    G0 = (D * parts["L2"] + A + parts["Yield"]).tocsc()
    I = sp.eye(n, format="csc")
    lu = spla.splu((I - 0.5 * da * G0).tocsc())   # factor once
    Mp = (I + 0.5 * da * G0)
    S = np.empty(nst + 1)
    psi = np.empty(nst + 1)
    m1 = np.empty(nst + 1)
    m2 = np.empty(nst + 1)
    R = np.zeros(n)
    Q = np.zeros(n)
    Q[grid.i_zero] = 1.0 / h                       # delta at sigma = 0
    for k in range(nst + 1):
        S[k] = h * Q.sum()
        psi[k] = h * float(Q @ mask)
        m1[k] = h * (sig * Q).sum()
        m2[k] = h * (sig * sig * Q).sum()
        R += (0.5 * da if (k == 0 or k == nst) else da) * Q
        if k < nst:
            Q = lu.solve(Mp.dot(Q))
    return S, psi, m1, m2, R


def residence_time_distribution(alpha: float, gammadot: float,
                                grid: Grid | None = None, parts: dict | None = None,
                                da: float = 0.05, n_mean: float = 14.0,
                                a_max: float | None = None, a_max_cap: float = 4000.0,
                                compute_gap: bool = False, warn: bool = True,
                                steady=None) -> ResidenceTimeDistribution:
    """Distribution of the age at which a block yields after rebirth at sigma = 0.

    Integrates the cohort d_a Q = G0 Q (G0 = D*L2 + advection - yielding, i.e. the
    generator without the reinjection source) from a delta at sigma = 0, using the
    self-consistent steady (D, gammadot).  The mean equals 1/Gamma exactly.

    The age window runs to ``a_max`` (default ``n_mean``/Gamma, safety-capped at
    ``a_max_cap``); ``da`` is the age step (Crank-Nicolson, unconditionally
    stable).  ``compute_gap=True`` also returns the tail decay rate (spectral gap
    of G0).  If ``warn`` and the material barely yields (window not captured), a
    RuntimeWarning is issued and ``defined`` is set False.  Pass a precomputed
    ``steady`` SteadyResult to avoid re-solving.
    """
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)
    if steady is None:
        steady = solve_steady(alpha, gammadot, grid, parts, warn=False)
    D, Gamma = float(steady.D), float(steady.Gamma)
    n, h = grid.n, grid.h

    if Gamma <= 0.0:                            # jammed & frozen: never yields
        if warn:
            warnings.warn(
                f"residence_time_distribution: alpha={alpha:g}, gammadot={gammadot:g} "
                f"does not yield (Gamma=0), so the residence time is infinite -- this "
                f"is the frozen jammed state. Apply shear or use a liquid alpha>1/2.",
                RuntimeWarning, stacklevel=2)
        z = np.zeros(n)
        return ResidenceTimeDistribution(
            float(alpha), float(gammadot), np.array([0.0]), np.array([0.0]),
            np.array([1.0]), 0.0, D, float("inf"), z, grid, None, False)

    if a_max is None:
        a_max_auto = n_mean / Gamma
        a_max_used = min(a_max_auto, a_max_cap)
        capped = a_max_auto > a_max_cap
    else:
        a_max_used = float(a_max); capped = False

    # Integrate the no-source cohort (shared with nonaffine_stress_trajectory).
    nst = int(np.ceil(a_max_used / da))
    a = np.arange(nst + 1) * da
    S, psi, _m1, _m2, R = _cohort_steady(D, gammadot, grid, parts, da, nst,
                                         P_ref=steady.P)
    mean = float(np.sum(0.5 * (S[:-1] + S[1:])) * da)   # <a> = int S da
    captured = float(1.0 - S[-1])
    defined = captured >= 0.5

    if warn and (not defined or capped or captured < 0.99):
        warnings.warn(
            f"residence_time_distribution: only {captured:.1%} of blocks yield "
            f"within the age window a<={a_max_used:g} (Gamma={Gamma:.2e}, "
            f"1/Gamma={1.0/Gamma:.3g}); the tail is truncated and the mean is "
            f"underestimated. Increase a_max/n_mean (or the material barely yields).",
            RuntimeWarning, stacklevel=2)

    gap = None
    if compute_gap:
        G0 = (D * parts["L2"]
              + advection_matrix_frozen(gammadot, grid, steady.P)
              + parts["Yield"]).tocsc()
        try:
            lam = spla.eigs(G0, k=1, which="LR", return_eigenvectors=False,
                            maxiter=5000)
            gap = -float(np.real(lam[0]))
        except Exception:
            gap = None

    return ResidenceTimeDistribution(
        float(alpha), float(gammadot), a, psi, S, Gamma, D, mean, R, grid, gap,
        defined)


# ---------------------------------------------------------------------------
# Non-affine stress trajectory (steady shear)
# ---------------------------------------------------------------------------
@dataclass
class NonAffineStressTrajectory:
    """Non-affine stress evolution of an element between rebirth and yield (steady shear).

    A block reborn at sigma = 0 loads *affinely* at sigma_aff(a) = gammadot * a --
    the elastic advection.  The no-source cohort Q(sigma, a) gives the stress
    distribution of blocks that have survived to age a without yielding; from it we
    read the survivor mean stress and split off the *non-affine* part (actual minus
    affine):

        survivor_mean(a)   = <sigma>_surv(a) = int sigma Q / int Q
        nonaffine_mean(a)  = <sigma>_surv(a) - sigma_aff(a)
        nonaffine_msd(a)   = <(sigma - sigma_aff)^2>_surv(a)

    Before any absorption the block random-walks freely, so ``nonaffine_msd`` grows
    as 2 D a (the mechanical-noise diffusion, D = alpha * Gamma); absorption at the
    yield threshold |sigma| = 1 then truncates it -- the un-yielded population cannot
    spread past threshold, so the MSD saturates at a stress-space "cage".  The
    survivor mean tracks the affine line at first, then bends below it and plateaus
    (a surviving block is one whose non-affine kicks kept it off the affine ramp to
    yield), so ``nonaffine_mean`` tends to -sigma_aff at long age.

    Exact cross-check (tests): since P(sigma) = Gamma * int Q da, the macroscopic
    stress is Sigma = Gamma * int m1(a) da, and the early-age ``nonaffine_msd`` slope
    is 2 D.
    """
    alpha: float
    gammadot: float
    a: np.ndarray                              # age since rebirth
    survivor_mean: np.ndarray                  # <sigma>_surv(a)
    affine: np.ndarray                         # sigma_aff(a) = gammadot * a
    nonaffine_mean: np.ndarray                 # <sigma>_surv - affine
    nonaffine_msd: np.ndarray                  # <(sigma - affine)^2>_surv
    survivor_var: np.ndarray                   # <sigma^2>_surv - <sigma>_surv^2
    survival: np.ndarray                       # S(a)
    Gamma: float
    D: float
    grid: Grid = field(repr=False)
    defined: bool = True                       # False if the material does not yield

    @property
    def free_msd(self) -> np.ndarray:
        """Free-diffusion reference 2 D a (unbounded, no absorption)."""
        return 2.0 * self.D * self.a


def nonaffine_stress_trajectory(alpha: float, gammadot: float,
                                grid: Grid | None = None, parts: dict | None = None,
                                da: float = 0.02, n_mean: float = 8.0,
                                a_max: float | None = None, a_max_cap: float = 4000.0,
                                warn: bool = True,
                                steady=None) -> NonAffineStressTrajectory:
    """Non-affine stress trajectory of an element before it yields, under steady shear.

    Integrates the no-source cohort in the self-consistent steady (D, gammadot)
    background and subtracts the affine loading gammadot * a to isolate the
    non-affine stress motion; see :class:`NonAffineStressTrajectory`.  ``da`` is the
    age step (Crank-Nicolson); the age window runs to ``a_max`` (default
    ``n_mean``/Gamma, capped at ``a_max_cap``).  Pass a precomputed ``steady``
    SteadyResult to avoid re-solving.

    In the frozen jammed state (Gamma = 0) nothing yields: with no absorption the
    element just loads affinely and the non-affine motion is trivial, so the result
    is returned with ``defined=False``.  Any nonzero shear rate makes it well posed.
    """
    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)
    if steady is None:
        steady = solve_steady(alpha, gammadot, grid, parts, warn=False)
    D, Gamma = float(steady.D), float(steady.Gamma)

    if Gamma <= 0.0:                            # jammed & frozen: never yields
        if warn:
            warnings.warn(
                f"nonaffine_stress_trajectory: alpha={alpha:g}, gammadot={gammadot:g} "
                f"does not yield (Gamma=0); with no absorption the element loads "
                f"affinely forever and the non-affine motion is trivial. Apply shear "
                f"or use a liquid alpha>1/2.", RuntimeWarning, stacklevel=2)
        z = np.zeros(1)
        return NonAffineStressTrajectory(
            float(alpha), float(gammadot), np.array([0.0]), z.copy(), z.copy(),
            z.copy(), z.copy(), z.copy(), np.array([1.0]), 0.0, D, grid, False)

    if a_max is None:
        a_max_used = min(n_mean / Gamma, a_max_cap)
        capped = (n_mean / Gamma) > a_max_cap
    else:
        a_max_used = float(a_max); capped = False

    nst = int(np.ceil(a_max_used / da))
    a = np.arange(nst + 1) * da
    # Deliberately the *upwind* cohort (P_ref=None), unlike
    # residence_time_distribution.  The two need different things from the
    # operator.  There, the quantity is the age-integral R, compared against
    # solve_steady's state, so the advection must match it (frozen van Leer).
    # Here, every output is a ratio m/S conditioned on a survival that decays
    # exponentially, which demands a *monotone* operator: the flux limiter's
    # anti-diffusive correction rings at the 1e-10 level, and once S is that
    # small the ringing swamps the ratio.  In the well-conditioned range the two
    # agree to ~1%, so nothing physical is given up.
    S, _psi, m1, m2, _R = _cohort_steady(D, gammadot, grid, parts, da, nst)

    # The survivor-conditioned moments are ratios m/S, so once the surviving
    # fraction is at round-off level the ratio is pure noise -- dividing by
    # S ~ 1e-14 amplifies double-precision error by ~1e16.  It is also
    # physically vacuous: nothing is left to condition on.  Truncate the window
    # where the cohort is exhausted (the discarded tail carries weight <= S_MIN,
    # so integrals over the trajectory are unaffected).
    live = np.nonzero(S >= SURVIVAL_FLOOR)[0]
    last = int(live[-1]) if live.size else 0
    if last < nst:
        a, S, m1, m2 = a[:last + 1], S[:last + 1], m1[:last + 1], m2[:last + 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        s_mean = np.where(S > 0.0, m1 / S, 0.0)
        s2 = np.where(S > 0.0, m2 / S, 0.0)
    affine = gammadot * a
    na_mean = s_mean - affine
    na_msd = s2 - 2.0 * affine * s_mean + affine * affine
    var = s2 - s_mean * s_mean

    if warn and (capped or (1.0 - S[-1]) < 0.99):
        warnings.warn(
            f"nonaffine_stress_trajectory: only {1.0 - S[-1]:.1%} of blocks have "
            f"yielded by the end of the age window a<={a_max_used:g} "
            f"(Gamma={Gamma:.2e}); the late-age trajectory is not fully resolved. "
            f"Increase a_max/n_mean.", RuntimeWarning, stacklevel=2)

    return NonAffineStressTrajectory(
        float(alpha), float(gammadot), a, s_mean, affine, na_mean, na_msd, var,
        S, float(Gamma), float(D), grid, True)


# ---------------------------------------------------------------------------
# Oscillation
# ---------------------------------------------------------------------------
@dataclass
class CycleResidenceDistribution:
    """Age-at-yield density over a converged LAOS cycle.

    ``psi`` is the phase-averaged density (over birth phases, weighted by the
    rebirth rate Gamma(t)); its mean equals 1/<Gamma>.  ``birth_phase`` holds the
    phases t0/T, ``psi_by_phase`` their densities, ``mean_by_phase`` their means
    (period T/2 by the sine drive's half-period symmetry; blocks born at maximum
    strain rate yield sooner than those born at the turning point).

    ``captured_fraction`` is the fraction of cohorts that yield within the age
    window; ``defined`` is False when the material barely yields (sub-yield jammed
    regime, gamma0 below the yield strain -> residence time effectively infinite).
    """
    alpha: float
    gamma0: float
    omega: float
    a: np.ndarray
    psi: np.ndarray
    Gamma_mean: float
    mean: float
    grid: Grid = field(repr=False)
    birth_phase: np.ndarray | None = field(default=None, repr=False)
    psi_by_phase: np.ndarray | None = field(default=None, repr=False)
    mean_by_phase: np.ndarray | None = None
    captured_fraction: float = 1.0
    defined: bool = True

    @property
    def mean_exact(self) -> float:
        return 1.0 / self.Gamma_mean if self.Gamma_mean > 0.0 else float("inf")


def _td_cohort(grid: Grid, D_of, gdot_of, t0: float, a_max: float, da: float):
    """Integrate one cohort under time-dependent (D(t), gammadot(t)); Crank-Nicolson.

    Returns (a, psi, S, mean).  Banded (tridiagonal) solve per step since the
    coefficients change; no reinjection source.
    """
    n, h = grid.n, grid.h
    inv = 1.0 / (h * h)
    l2m = np.full(n, -2.0 * inv); l2m[0] = -inv; l2m[-1] = -inv
    l2u = np.full(n, inv)
    l2l = np.full(n, inv)
    mask = grid.yield_weight               # half weight exactly at |sigma| = 1

    def bands(t):
        D = D_of(t); v = gdot_of(t)
        am = np.zeros(n); au = np.zeros(n); al = np.zeros(n)
        if v > 0.0:
            am[:-1] += -v / h; al[1:] += v / h
        elif v < 0.0:
            am[1:] += v / h; au[:-1] += -v / h
        return D * l2m + am - mask, D * l2u + au, D * l2l + al

    sig = grid.sigma
    nst = int(np.ceil(a_max / da))
    a = np.arange(nst + 1) * da
    psi = np.empty(nst + 1)
    S = np.empty(nst + 1)
    m1 = np.empty(nst + 1)
    m2 = np.empty(nst + 1)
    Q = np.zeros(n)
    Q[grid.i_zero] = 1.0 / h
    for k in range(nst + 1):
        psi[k] = h * (mask * Q).sum()
        S[k] = h * Q.sum()
        m1[k] = h * (sig * Q).sum()
        m2[k] = h * (sig * sig * Q).sum()
        if k < nst:
            t = t0 + k * da
            gm0, gu0, gl0 = bands(t)
            gm1, gu1, gl1 = bands(t + da)
            GQ = gm0 * Q
            GQ[1:] += gl0[1:] * Q[:-1]
            GQ[:-1] += gu0[:-1] * Q[1:]
            rhs = Q + 0.5 * da * GQ
            ab = np.zeros((3, n))
            ab[0, 1:] = -0.5 * da * gu1[:-1]
            ab[1, :] = 1.0 - 0.5 * da * gm1
            ab[2, :-1] = -0.5 * da * gl1[1:]
            Q = solve_banded((1, 1), ab, rhs)
    mean = float(np.sum(0.5 * (S[:-1] + S[1:])) * da)
    return a, psi, S, m1, m2, mean


def cycle_residence_time_distribution(alpha: float, gamma0: float, omega: float,
                                      grid: Grid | None = None,
                                      parts: dict | None = None,
                                      n_birth_phases: int = 16, da: float = 0.05,
                                      n_mean: float = 8.0,
                                      a_max: float | None = None,
                                      a_max_cap: float = 2000.0,
                                      steps_per_period: int = 400,
                                      warn: bool = True,
                                      cycle=None) -> CycleResidenceDistribution:
    """Residence-time distribution over a converged LAOS cycle.

    Solves the limit cycle (unless a converged ``cycle`` LAOSResult is supplied),
    reads its Gamma(t) -> D(t) = alpha*Gamma(t), then integrates cohorts launched
    at ``n_birth_phases`` evenly spaced birth phases under the time-dependent
    background.  Returns the rebirth-weighted phase-average (mean = 1/<Gamma>) and
    the birth-phase-resolved densities.

    The age window runs to ``a_max`` (default ``n_mean``/<Gamma>, safety-capped at
    ``a_max_cap``).  If ``warn`` and the material barely yields (sub-yield jammed
    regime: gamma0 below the yield strain, so <Gamma> -> 0 and the residence time
    diverges), a RuntimeWarning is issued and ``defined`` is set False.
    """
    from .periodic import solve_laos, PeriodMap

    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)
    require_sine_drive(getattr(cycle, "protocol", None), "cycle_residence_time_distribution")
    if cycle is None:
        cycle = solve_laos(alpha, gamma0, omega, grid, parts,
                           steps_per_period=steps_per_period, warn=False)

    pmap = PeriodMap(alpha, gamma0, omega, grid, parts,
                     steps_per_period=steps_per_period)
    T = pmap.T
    # Gamma(t) over one period from the converged cycle
    traj = pmap.stepper.run(cycle.P0, pmap.protocol.gammadot, T, pmap.dt,
                            record_every=1)
    keep = traj.t < T - 1e-9
    tt, Gam = traj.t[keep], traj.Gamma[keep]
    Gamma_mean = float(Gam.mean())

    if a_max is None:
        a_max_auto = n_mean / Gamma_mean if Gamma_mean > 0.0 else np.inf
        a_max_used = float(min(a_max_auto, a_max_cap))
        capped = a_max_auto > a_max_cap
    else:
        a_max_used = float(a_max); capped = False

    D_of = lambda t: alpha * np.interp(np.mod(t, T), tt, Gam, period=T)
    gdot_of = lambda t: gamma0 * omega * np.cos(omega * t)

    phases = np.linspace(0.0, 1.0, int(n_birth_phases), endpoint=False)
    a_ref = None
    psi_by = []
    mean_by = []
    cap_by = []
    for ph in phases:
        a, psi, S, _m1, _m2, m = _td_cohort(grid, D_of, gdot_of, ph * T,
                                            a_max_used, da)
        if a_ref is None:
            a_ref = a
        m_len = a_ref.size
        psi_by.append(psi[:m_len] if psi.size >= m_len
                      else np.pad(psi, (0, m_len - psi.size)))
        mean_by.append(m)
        cap_by.append(1.0 - S[-1])
    psi_by = np.array(psi_by)
    mean_by = np.array(mean_by)

    w = np.array([np.interp(np.mod(ph * T, T), tt, Gam, period=T) for ph in phases])
    wsum = w.sum()
    w = w / wsum if wsum > 0 else np.full_like(w, 1.0 / w.size)
    psi_avg = (w[:, None] * psi_by).sum(axis=0)
    mean_avg = float((w * mean_by).sum())
    captured = float((w * np.array(cap_by)).sum())
    defined = captured >= 0.5

    if warn:
        if not defined:
            warnings.warn(
                f"cycle_residence_time_distribution: the material barely yields "
                f"(<Gamma>={Gamma_mean:.2e}); only {captured:.2%} of cohorts yield "
                f"within a<={a_max_used:g}, so psi is essentially numerical noise and "
                f"the mean is unreliable. This is the sub-yield jammed regime -- "
                f"gamma0={gamma0:g} is below the yield strain (of order sigma_y). "
                f"Increase gamma0 (or use a liquid alpha>1/2).",
                RuntimeWarning, stacklevel=2)
        elif capped or captured < 0.99:
            warnings.warn(
                f"cycle_residence_time_distribution: age window truncated at "
                f"a<={a_max_used:g} ({captured:.1%} of cohorts captured); the tail is "
                f"cut and the mean is underestimated (1/<Gamma>={1.0/Gamma_mean:.3g}). "
                f"Pass a larger a_max or n_mean.", RuntimeWarning, stacklevel=2)

    return CycleResidenceDistribution(
        float(alpha), float(gamma0), float(omega), a_ref, psi_avg, Gamma_mean,
        mean_avg, grid, phases, psi_by, mean_by, captured, defined)


# ---------------------------------------------------------------------------
# Non-affine stress trajectory (oscillation)
# ---------------------------------------------------------------------------
@dataclass
class CycleNonAffineStressTrajectory:
    """Birth-phase-resolved non-affine stress trajectories over a converged LAOS cycle.

    In oscillation the affine loading of a cohort born at phase t0 is

        sigma_aff(a; t0) = gamma0 [ sin(omega (t0 + a)) - sin(omega t0) ],

    i.e. the affine strain accumulated since birth.  Subtracting it is exactly the
    co-deforming transform xi = sigma - gamma(t), in which the advection is removed
    and only diffusion and the (now oscillating) yield threshold remain -- so what is
    left is purely non-affine.  This generalises the stroboscopic subtraction
    (sampling at a = nT, where sigma_aff = 0) to every age.

    Arrays are shaped ``(n_birth_phases, n_age)``.  ``survivor_mean`` is the raw
    <sigma>_surv per birth phase; ``affine`` is sigma_aff(a; t0); ``nonaffine_mean``
    and ``nonaffine_msd`` are their non-affine parts.  ``nonaffine_mean_avg`` and
    ``nonaffine_msd_avg`` are the rebirth-weighted (weight Gamma(t0)) averages over
    birth phase.  By the sine drive's half-period symmetry (sigma, t) -> (-sigma,
    t + T/2), the rows for t0 and t0 + T/2 are mirror images: their non-affine MSDs
    coincide and their non-affine means are opposite.
    """
    alpha: float
    gamma0: float
    omega: float
    a: np.ndarray                              # age since rebirth
    birth_phase: np.ndarray                    # t0 / T in [0, 1)
    survivor_mean: np.ndarray                  # (n_phase, n_age) raw <sigma>_surv
    affine: np.ndarray                         # (n_phase, n_age)
    nonaffine_mean: np.ndarray                 # (n_phase, n_age)
    nonaffine_msd: np.ndarray                  # (n_phase, n_age)
    survival: np.ndarray                       # (n_phase, n_age)
    nonaffine_mean_avg: np.ndarray             # (n_age,) rebirth-weighted
    nonaffine_msd_avg: np.ndarray              # (n_age,)
    Gamma_mean: float
    grid: Grid = field(repr=False)
    defined: bool = True


def cycle_nonaffine_stress_trajectory(alpha: float, gamma0: float, omega: float,
                                      grid: Grid | None = None,
                                      parts: dict | None = None,
                                      n_birth_phases: int = 16, da: float = 0.05,
                                      n_mean: float = 8.0, a_max: float | None = None,
                                      a_max_cap: float = 2000.0,
                                      steps_per_period: int = 400, warn: bool = True,
                                      cycle=None) -> CycleNonAffineStressTrajectory:
    """Non-affine stress trajectories before yield over a converged LAOS cycle.

    Solves the limit cycle (unless a converged ``cycle`` LAOSResult is supplied),
    reads Gamma(t) -> D(t) = alpha*Gamma(t), then integrates no-source cohorts
    launched at ``n_birth_phases`` evenly spaced birth phases under the
    time-dependent background.  For each phase it subtracts the affine loading
    gamma0[sin(omega(t0+a)) - sin(omega t0)] (the co-deforming transform) to isolate
    the non-affine stress excursion, and returns the birth-phase-resolved
    trajectories plus their rebirth-weighted average.  See
    :class:`CycleNonAffineStressTrajectory`.

    ``a_max`` defaults to ``n_mean``/<Gamma> (capped at ``a_max_cap``).  In the
    sub-yield jammed regime (<Gamma> -> 0) most cohorts never yield within the
    window; the caged non-affine diffusion is still valid, but ``defined`` is set
    False (with a warning) because the "before yield" horizon is then the whole run.
    """
    from .periodic import solve_laos, PeriodMap

    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)
    require_sine_drive(getattr(cycle, "protocol", None), "cycle_nonaffine_stress_trajectory")
    if cycle is None:
        cycle = solve_laos(alpha, gamma0, omega, grid, parts,
                           steps_per_period=steps_per_period, warn=False)

    pmap = PeriodMap(alpha, gamma0, omega, grid, parts,
                     steps_per_period=steps_per_period)
    T = pmap.T
    traj = pmap.stepper.run(cycle.P0, pmap.protocol.gammadot, T, pmap.dt,
                            record_every=1)
    keep = traj.t < T - 1e-9
    tt, Gam = traj.t[keep], traj.Gamma[keep]
    Gamma_mean = float(Gam.mean())

    if a_max is None:
        a_max_auto = n_mean / Gamma_mean if Gamma_mean > 0.0 else np.inf
        a_max_used = float(min(a_max_auto, a_max_cap))
    else:
        a_max_used = float(a_max)

    D_of = lambda t: alpha * np.interp(np.mod(t, T), tt, Gam, period=T)
    gdot_of = lambda t: gamma0 * omega * np.cos(omega * t)

    phases = np.linspace(0.0, 1.0, int(n_birth_phases), endpoint=False)
    a_ref = None
    sm_by, aff_by, nam_by, namsd_by, S_by, cap_by = [], [], [], [], [], []
    for ph in phases:
        t0 = ph * T
        a, _psi, S, m1, m2, _m = _td_cohort(grid, D_of, gdot_of, t0, a_max_used, da)
        if a_ref is None:
            a_ref = a
        L = a_ref.size
        S = S[:L]; m1 = m1[:L]; m2 = m2[:L]
        with np.errstate(divide="ignore", invalid="ignore"):
            s_mean = np.where(S > 0.0, m1 / S, 0.0)
            s2 = np.where(S > 0.0, m2 / S, 0.0)
        affine = gamma0 * (np.sin(omega * (t0 + a_ref)) - np.sin(omega * t0))
        sm_by.append(s_mean)
        aff_by.append(affine)
        nam_by.append(s_mean - affine)
        namsd_by.append(s2 - 2.0 * affine * s_mean + affine * affine)
        S_by.append(S)
        cap_by.append(1.0 - S[-1])
    sm_by = np.array(sm_by); aff_by = np.array(aff_by)
    nam_by = np.array(nam_by); namsd_by = np.array(namsd_by); S_by = np.array(S_by)

    w = np.array([np.interp(np.mod(ph * T, T), tt, Gam, period=T) for ph in phases])
    wsum = w.sum()
    w = w / wsum if wsum > 0 else np.full_like(w, 1.0 / w.size)
    nam_avg = (w[:, None] * nam_by).sum(axis=0)
    namsd_avg = (w[:, None] * namsd_by).sum(axis=0)
    captured = float((w * np.array(cap_by)).sum())
    defined = captured >= 0.5

    if warn and not defined:
        warnings.warn(
            f"cycle_nonaffine_stress_trajectory: the material barely yields "
            f"(<Gamma>={Gamma_mean:.2e}); most cohorts do not yield within "
            f"a<={a_max_used:g}. The caged non-affine diffusion is still valid, but "
            f"the 'before yield' horizon is effectively the whole run -- this is the "
            f"sub-yield jammed regime (gamma0 below the yield strain).",
            RuntimeWarning, stacklevel=2)

    return CycleNonAffineStressTrajectory(
        float(alpha), float(gamma0), float(omega), a_ref, phases, sm_by, aff_by,
        nam_by, namsd_by, S_by, nam_avg, namsd_avg, Gamma_mean, grid, defined)


# ---------------------------------------------------------------------------
# Yield-phase distribution (oscillation)
# ---------------------------------------------------------------------------
def _wrap_to_phase(a: np.ndarray, psi_a: np.ndarray, t0: float, T: float,
                   u: np.ndarray) -> np.ndarray:
    """Fold an age-at-yield density psi(a) (born at time t0) onto cycle phase.

    ``u`` holds phase fractions phi/T in [0,1).  A block born at t0 that yields at
    age a lands at phase (t0+a) mod T, so the phase density is the sum over periods
    rho(phi) = sum_j psi(((phi - t0) mod T) + jT).  Returned unnormalised (it
    integrates to the captured fraction of psi).  Vectorised over u and periods.
    """
    a_max = float(a[-1])
    base = (u * T - t0) % T                             # (n_u,) in [0, T)
    K = int(np.floor(a_max / T)) + 2
    A = base[:, None] + np.arange(K)[None, :] * T       # (n_u, K) ages
    vals = np.interp(A.ravel(), a, psi_a, left=0.0, right=0.0).reshape(A.shape)
    vals[A > a_max] = 0.0
    return vals.sum(axis=1)


@dataclass
class YieldPhaseDistribution:
    """Distribution of the cycle phase at which blocks yield, over a LAOS cycle.

    The residence density psi(a; t0) is a *duration* -- the age from rebirth to
    yield.  The yield *phase* is (t0 + a) mod T; since a block usually survives
    several periods, the phase density folds psi onto one period,

        rho(phi | t0) = sum_j psi(((phi - t0) mod T) + jT ; t0).

    ``phase`` is phi/T in [0,1); ``rho_by_phase`` are the per-birth-phase densities
    (each normalised to unit integral over phase), ``rho`` is their rebirth-weighted
    (weight Gamma(t0)) aggregate, and ``yield_rate`` is the instantaneous yield rate
    Gamma(phi)/<Gamma> read straight off the converged cycle.  The two aggregates
    coincide: summed over birth phase, blocks yield at exactly the instantaneous
    rate Gamma(phi) -- a built-in consistency check.  All densities integrate to 1
    over phase in [0,1).  For the symmetric sine drive Gamma has period T/2, so
    ``yield_rate`` (and ``rho``) are invariant under phi -> phi + T/2, and yielding
    clusters at the two strain extrema.
    """
    alpha: float
    gamma0: float
    omega: float
    phase: np.ndarray                  # phi/T in [0,1)
    rho: np.ndarray                    # rebirth-weighted aggregate yield-phase density
    yield_rate: np.ndarray             # Gamma(phi)/<Gamma>, normalised (independent check)
    birth_phase: np.ndarray            # t0/T
    rho_by_phase: np.ndarray           # (n_birth, n_phase)
    Gamma_mean: float
    grid: Grid = field(repr=False)
    defined: bool = True


def cycle_yield_phase_distribution(alpha: float, gamma0: float, omega: float,
                                   grid: Grid | None = None,
                                   parts: dict | None = None,
                                   n_birth_phases: int = 16, n_phase: int = 200,
                                   da: float = 0.05, n_mean: float = 8.0,
                                   a_max: float | None = None,
                                   a_max_cap: float = 2000.0,
                                   steps_per_period: int = 400, warn: bool = True,
                                   cycle=None,
                                   residence: CycleResidenceDistribution | None = None
                                   ) -> YieldPhaseDistribution:
    """Yield-phase distribution over a converged LAOS cycle.

    Folds the birth-phase-resolved age-at-yield density psi(a; t0) onto one period
    to give the distribution of the cycle phase at which a block yields; see
    :class:`YieldPhaseDistribution`.  Solves the limit cycle (unless a converged
    ``cycle`` LAOSResult is supplied) and reuses ``cycle_residence_time_distribution``
    for psi (unless a precomputed ``residence`` result is supplied).  ``n_phase`` is
    the number of phase bins.  Also returns the instantaneous yield rate Gamma(phi)
    as an independent aggregate that the rebirth-weighted fold must reproduce.
    """
    from .periodic import solve_laos, PeriodMap

    if grid is None:
        grid = Grid()
    if parts is None:
        parts = build_static_parts(grid)
    require_sine_drive(getattr(cycle, "protocol", None), "cycle_yield_phase_distribution")
    if cycle is None:
        cycle = solve_laos(alpha, gamma0, omega, grid, parts,
                           steps_per_period=steps_per_period, warn=False)

    # instantaneous yield rate Gamma(t) over one converged period
    pmap = PeriodMap(alpha, gamma0, omega, grid, parts,
                     steps_per_period=steps_per_period)
    T = pmap.T
    traj = pmap.stepper.run(cycle.P0, pmap.protocol.gammadot, T, pmap.dt,
                            record_every=1)
    keep = traj.t < T - 1e-9
    tt, Gam = traj.t[keep], traj.Gamma[keep]
    Gamma_mean = float(Gam.mean())

    # birth-phase-resolved age-at-yield densities psi(a; t0)
    if residence is None:
        residence = cycle_residence_time_distribution(
            alpha, gamma0, omega, grid, parts, n_birth_phases=n_birth_phases,
            da=da, n_mean=n_mean, a_max=a_max, a_max_cap=a_max_cap,
            steps_per_period=steps_per_period, warn=False, cycle=cycle)
    a = residence.a
    bphase = residence.birth_phase
    psi_by = residence.psi_by_phase
    if bphase is None or psi_by is None:
        raise ValueError("residence result carries no birth-phase-resolved "
                         "densities; use n_birth_phases >= 1.")

    u = np.arange(int(n_phase)) / int(n_phase)          # phi/T in [0,1)
    n_b = len(bphase)
    rho_by = np.empty((n_b, u.size))
    for j in range(n_b):
        r = _wrap_to_phase(a, psi_by[j], float(bphase[j]) * T, T, u)
        s = r.sum()
        rho_by[j] = r * (u.size / s) if s > 0 else r    # normalise int_0^1 rho du = 1

    # rebirth weights Gamma(t0_j); aggregate = weighted sum over birth phase
    w = np.array([np.interp((float(ph) * T) % T, tt, Gam, period=T) for ph in bphase])
    w = w / w.sum() if w.sum() > 0 else np.full(n_b, 1.0 / n_b)
    rho_agg = (w[:, None] * rho_by).sum(axis=0)

    # instantaneous yield rate as a phase density (independent aggregate)
    yr = np.interp(u * T, tt, Gam, period=T)
    ssum = yr.sum()
    yr = yr * (u.size / ssum) if ssum > 0 else yr

    defined = bool(residence.defined)
    if warn and not defined:
        warnings.warn(
            "cycle_yield_phase_distribution: the residence distribution is "
            "ill-defined (sub-yield jammed regime, <Gamma> -> 0); the yield-phase "
            "distribution is unreliable.", RuntimeWarning, stacklevel=2)

    return YieldPhaseDistribution(
        float(alpha), float(gamma0), float(omega), u, rho_agg, yr,
        np.asarray(bphase, dtype=float), rho_by, Gamma_mean, grid, defined)
