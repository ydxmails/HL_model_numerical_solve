# Recovery Rheology: separating what is stored from what is lost

### A beginner's guide to the recoverable / unrecoverable strain decomposition

*Written for someone comfortable with calculus and with the basic language of
rheology (stress, strain, moduli), but not assumed to know LAOS analysis. It
explains what the method is, why it was invented, what it measures, and how it
maps onto the Hebraud-Lequeux model implemented in this package.*

---

## 1. The question

Deform a material and it responds with a stress. Some of the work you did is
**stored** — the material will give it back if you let go. Some is **lost** — the
material flowed, rearranged, and will never return to where it started.

Every rheological measurement is, at bottom, an attempt to answer: *how much of
each?* A rubber band is nearly all storage. Honey is nearly all loss. A
mayonnaise, a paste, a gel, a dense suspension — these sit in between, and worse,
they move between the two as you push them harder. That movement is called
**yielding**, and it is the interesting part.

## 2. The classical answer, and why it is not enough

The standard tool is **small-amplitude oscillatory shear** (SAOS). Impose

$$\gamma(t) = \gamma_0\sin\omega t$$

and if the amplitude is small the stress comes back as a pure sinusoid, phase
shifted:

$$\sigma(t) = \gamma_0\big[G'\sin\omega t + G''\cos\omega t\big]$$

The in-phase part is called elastic (storage modulus $G'$), the out-of-phase part
viscous (loss modulus $G''$). This works beautifully — *as long as the response is
actually a sinusoid.*

Push harder and it is not. The stress waveform distorts, picking up higher
harmonics ($3\omega$, $5\omega$, ...). You can still compute $G'$ and $G''$ by
projecting onto $\sin\omega t$ and $\cos\omega t$, and everyone does. But now they
are **coefficients of a fit**, not physical quantities. Three problems follow:

1. **They are convention-dependent.** Project onto a different basis (Chebyshev
   polynomials, say) and you get a different, equally defensible split.
2. **They are cycle-averaged.** A yield-stress material spends part of each cycle
   behaving like a solid and part like a liquid. One number per cycle cannot see
   that alternation — it reports a blend.
3. **They can behave strangely.** The clearest example: as you increase the
   amplitude through the yielding transition, $G''$ frequently rises to a peak
   before falling — the **$G''$ overshoot**. For decades this was a robust,
   widely-observed, and essentially unexplained feature.

The trouble is that "elastic" and "viscous" were *defined* by a mathematical
decomposition rather than by a physical operation. Recovery rheology reverses
that: define them by an experiment, then see what the numbers do.

## 3. The physical definition

> **Recoverable strain** is the strain that comes back when you release the
> material. **Unrecoverable strain** is the strain that does not.

That is the whole idea, and it is entirely operational — no basis functions, no
assumed waveform, no linearity. At any instant the total strain splits as

$$\gamma(t) \;=\; \gamma_{\rm rec}(t) \;+\; \gamma_{\rm unrec}(t)$$

and correspondingly the rates, $\dot\gamma = \dot\gamma_{\rm rec} + \dot\gamma_{\rm unrec}$.

The interpretation is direct. $\gamma_{\rm rec}$ is strain the material is
*holding* — elastic energy in stretched bonds, deformed droplets, compressed
contacts. $\gamma_{\rm unrec}$ is strain that has been *dissipated* into
rearrangement — bonds broken, droplets slipped past one another, contacts
reformed. Stress is carried by the recoverable part; the unrecoverable part is
the record of flow.

## 4. The measurement: a recovery test

The operation is old — it is the "recovery" half of a classical creep-recovery
experiment. What is new is applying it *inside* an arbitrary protocol, and
resolving it in time.

**The steps.**

1. Run whatever protocol you care about — oscillation, steady shear, startup.
2. At a chosen instant $t^\ast$, stop controlling the strain. Switch the
   rheometer to **stress control and command $\sigma = 0$.**
3. The material now moves under its own stored stress. Record $\gamma(t)$.
4. Wait until it stops moving (in practice, until the strain rate falls below a
   set threshold). Call the final strain $\gamma_\infty$.
5. Then

$$\gamma_{\rm rec}(t^\ast) = \gamma(t^\ast) - \gamma_\infty, \qquad
  \gamma_{\rm unrec}(t^\ast) = \gamma_\infty$$

The recoil typically has two visible parts: an almost instantaneous elastic
snap-back, followed by a slower creeping return (the delayed, or *anelastic*,
recovery). Both count as recoverable — the criterion is whether the strain comes
back, not how fast.

**Why "iteratively punctuated".** The test *changes the material*. Once you have
released it, you are no longer in the state you were probing, and you cannot
simply continue the cycle. So to map a whole oscillation you must re-run the
entire experiment for each phase point: establish the steady cycle, run to phase
$\varphi_1$, release, measure; start over, run to $\varphi_2$, release, measure;
and so on. This is why the protocol is laborious in the laboratory — and why a
simulation, which can branch off a converged cycle as often as it likes at no
cost, is such a natural partner for it.

## 5. What you get

Because the decomposition is defined pointwise in time, it yields *time-resolved*
material information that Fourier moduli cannot express:

- **A physically-grounded modulus.** Stress is carried by stored strain, so
  $\sigma / \gamma_{\rm rec}$ is a modulus in the literal sense, valid
  instant-by-instant and at any amplitude.
- **A flow rate.** $\dot\gamma_{\rm unrec}$ is the genuine plastic flow rate — the
  rheological analogue of a plastic strain rate in solid mechanics.
- **A dissipation split.** Work done against recoverable strain is stored and
  returned; work done against unrecoverable strain is lost. The two can be
  separated within the cycle instead of lumped into $G''$.
- **A resolved yielding transition.** Within one period you can see the material
  hand over from storing to flowing, and identify *when* it does so.

## 6. The $G''$ overshoot, explained

This was the headline application (Donley, Singh, Shetty & Rogers, *PNAS* 2020).

Sweep the amplitude upward through yielding and watch $G''$ rise, peak, and fall.
Decomposed, the picture becomes simple:

- **Below yield**, essentially all strain is recoverable. The material is a soft
  solid. Dissipation is small, and what there is comes from *viscoelastic solid*
  losses — internal friction while remaining connected.
- **Well above yield**, essentially all strain is unrecoverable. The material is
  flowing. Dissipation is large in absolute terms, but per unit of imposed strain
  it is that of a simple liquid.
- **In between**, the material does both within each cycle — storing during part
  of the period and flowing during another. The *handover* between the two modes
  is where dissipation per unit strain is maximal.

The $G''$ peak marks that handover. It is not a resonance, not an artifact, and
not a property of the Fourier projection: it is the continuous transition from
recoverable to unrecoverable strain accumulation.

## 7. Which protocols can it be used with?

**Any of them.** The test only requires that you can stop imposing deformation and
command zero stress. Some cases worth naming:

| Protocol | What the decomposition looks like |
|---|---|
| **Steady shear** | All flow is unrecoverable, so the unrecoverable rate equals the imposed rate, while the recoverable strain saturates at a *constant* — the stored elastic strain at that rate. That constant is the material function of interest. |
| **Startup** | The recoverable strain grows, overshoots, then settles — the microscopic content of a stress overshoot. |
| **Creep** | The classical case. Recovery after creep is where the method originates. |
| **LAOS** | Requires iterative punctuation, one re-run per phase. The richest case, and the one in the PNAS paper. |
| **Quiescent jammed solid** | Nothing flows, so nothing is unrecoverable — the decomposition is trivial and all strain is recoverable. |

Steady shear deserves a comment, because it looks degenerate but is not. The
total strain grows without bound and all of the *flow* is unrecoverable; the
split of the rate is simply

$$\dot\gamma_{\rm unrec} = \dot\gamma, \qquad \dot\gamma_{\rm rec} = 0 .$$

But the material still carries a fixed stored elastic strain
$\gamma_{\rm rec}(\dot\gamma)$, constant in time, and *that* is the quantity to
measure. It is the modern descendant of the steady-state recoverable compliance
$J_e^0 = \gamma_{\rm rec}/\sigma$, and the origin of elastic recoil and die swell.

**Real limitations**, worth knowing before designing an experiment:

- The test destroys the probed state (hence the re-runs).
- "Zero stress" means zero *torque*, which in a real instrument includes inertia
  and bearing friction. Instrument compliance and finite response time smear the
  fast part of the recoil.
- Full recovery is asymptotic; you always stop at some finite time, so
  $\gamma_{\rm rec}$ carries a mild dependence on the waiting criterion. Report it.
- Very fast flows leave little stored strain, so the recoil becomes small and the
  signal-to-noise degrades.

## 8. How this maps onto the Hebraud-Lequeux model

In HL a material is an ensemble of mesoscopic blocks with local stresses
$\sigma$, distributed as $P(\sigma,t)$; the macroscopic stress is
$\Sigma = \int\sigma P\,d\sigma$. Reduced units $\tau = \sigma_c = G_0 = 1$.

**The key identity.** Take the first moment of the HL equation. Advection
contributes $\dot\gamma$, diffusion integrates away, the source sits at
$\sigma = 0$ and contributes nothing, and only the yielding loss survives:

$$\frac{d\Sigma}{dt} \;=\; \dot\gamma \;-\; \Sigma_{\rm pl},
\qquad \Sigma_{\rm pl} \equiv \int_{|\sigma|>1}\sigma\,P\,d\sigma$$

Read it physically: the stress rises with the imposed strain rate and falls at a
rate set by the stress currently carried by *over-threshold* blocks — those about
to yield. In steady state the two balance, $\dot\gamma = \Sigma_{\rm pl}$, which
this package reproduces to about $10^{-6}$.

**Why that makes the recovery test cheap.** A recovery test is the constraint
$\Sigma = 0$. Normally imposing a stress constraint on a rate-driven solver means
an implicit solve at every step. Here the identity gives it explicitly: holding
$\Sigma=0$ requires simply

$$\dot\gamma(t) = \Sigma_{\rm pl}(t)$$

one extra moment per time step, no root-finding.

**The protocol in the model.**

1. Take the state $P$ at the release instant, with stress $\Sigma$.
2. **Instantaneous recoil.** A strain step translates the distribution bodily,
   so $\Sigma \to \Sigma + \delta\gamma$. Setting $\delta\gamma = -\Sigma$ zeroes
   the stress at once. This is the elastic snap-back, of size $\Sigma/G_0$.
3. **Delayed recovery.** Integrate forward with $\dot\gamma = \Sigma_{\rm pl}$
   until the material stops moving, accumulating $\int\Sigma_{\rm pl}\,dt$.
4. The recoverable strain is the net backward excursion,
   $\gamma_{\rm rec} = \Sigma - \int_0^\infty \Sigma_{\rm pl}\,dt$.

It converges in both phases: in the jammed phase the noise $D = \alpha\Gamma$ dies
away and the material freezes; in the liquid phase it relaxes to the symmetric
quiescent state, where $\Sigma_{\rm pl} = 0$ by symmetry.

**Important caveat.** $\Sigma_{\rm pl}$ is an *internal* moment of the model's
distribution. It is not measured in any experiment and forms no part of the
recovery algorithm — it is purely the device that makes the zero-stress
constraint explicit here. Experimentally, $\gamma_{\rm rec}$ is read directly off
the recoil, with no model at all. (If you want a bridge: rearranged,
$\Sigma_{\rm pl} = \tau(G_0\dot\gamma - d\Sigma/dt)$, and both terms on the right
are measurable — but that is an inference *assuming* HL, not a measurement.)

**Relation to what this package already computes.** `hlmodel/recoverable.py`
measures a related but distinct quantity: the reverse strain from the strain
turning point to the point on the LAOS down-sweep where the stress crosses zero.
That is a **driven** reversal — the material is still being sheared at
$\dot\gamma(t)$ throughout — whereas a recovery test is a **free** release. Both
are legitimate; they answer different questions and will not agree in general.

## 9. Common misunderstandings

- **"Recoverable = elastic = $G'$."** No. $G'$ is a Fourier coefficient of a
  cycle; $\gamma_{\rm rec}$ is a measured strain at an instant. They agree in the
  linear limit and diverge exactly where the physics gets interesting.
- **"The instantaneous snap-back is the recoverable strain."** No — delayed
  recovery counts too. Stopping the test early systematically undercounts.
- **"Unrecoverable strain means the material is broken."** No. It means it flowed.
  A steadily shearing liquid accumulates unrecoverable strain forever and is in a
  perfectly reproducible state.
- **"You can decompose a single continuous run."** Not experimentally — each
  release ends the run. Time resolution comes from re-running, not from
  post-processing one dataset.
- **Recovery rheology is not SPP.** The same group also developed the *sequence of
  physical processes* framework, which extracts instantaneous moduli from the
  stress-strain trajectory by waveform analysis. SPP is a way of *analysing* data;
  recovery rheology is a way of *taking* it. They are complementary, not the same.

## 10. References

- G. J. Donley, P. K. Singh, A. Shetty, S. A. Rogers, *Elucidating the G'' overshoot
  in soft materials with a yield transition via a time-resolved experimental strain
  decomposition*, **PNAS 117**, 21945 (2020). doi:10.1073/pnas.2003869117
- S. A. Rogers, *A sequence of physical processes determined and quantified in
  LAOS: An instantaneous local 2D/3D approach*, **J. Rheol. 56**, 1129 (2012).
  (The SPP framework — related, but a different method.)
- *A reexamination of the Cox-Merz rule through the lens of recovery rheology*,
  **J. Rheol. 68**, 381 (2024).
- P. Hebraud, F. Lequeux, *Mode-Coupling Theory for the Pasty Rheology of Soft
  Glassy Materials*, **PRL 81**, 2934 (1998). (The model in this package.)
