# Why does a human iBCI mapping work in one session but not another?

Consider two cursor-control sessions recorded months apart from the same
BrainGate participant. We fit an offline mapping to the neural activity in the
earlier session and use it, unchanged, to predict the cursor-to-target
direction in the later session. If its predictions are much worse there, it is
tempting to call the result “decoder drift.” The score alone does not tell us
what changed.

The later recording may contain less direction-related signal that this model
can recover at all. The signal may still be present, but its relationship to
the target-direction proxy may have changed in a way that a small amount of
later-session data can repair. Or changes that are visible without using any
direction labels—such as missing electrodes, invalid samples, or altered
channel statistics—may be sufficient to reproduce the loss. More than one of
these patterns may occur at the same time.

EP16 asks which of those explanations is supported when every comparison uses
the same released feature, trial definition, direction proxy, preprocessing,
and model family. The first decisive comparison comes before assigning a
failure mode. For each participant, we compare mappings transported between
sessions 1--30 days apart with mappings transported at least 180 days apart.
We ask whether the long-lag mapping loses more performance than expected after
accounting for how difficult the source and target sessions are on their own.

A transported mapping can look bad simply because the later session is hard
for every model. We therefore compare its score with both a model trained
inside the later session and a direction-balanced null on exactly the same
held-out trials. If long-lag transport is not worse than the near-lag
reference, there is no normalized excess mismatch for the measurement and
remapping explanations. The separate question of whether later sessions lose
locally recoverable signal is still tested on its own.

If a reproducible deficit is present, the deeper question is why. The matched
diagnostic panel asks whether the later session's own recoverable score also
falls, whether a label-blind emulator of its observed measurement state can
reproduce the deficit, and whether a bounded remapping trained with 32 labelled
later-session trials can repair it. These are competing but not mutually
exclusive explanations. A correlation between recording quality and decoder
performance is not enough to choose among them.

The intended paper must do more than show that an old mapping sometimes
performs poorly. A shared explanation must pass the same diagnostic panel in
all three held-out participants. If more than one explanation is supported,
they are reported together; if the evidence cannot separate them, the answer
remains inconclusive. If participants appear to have different failure
patterns, each person's pattern must recur in two separate sets of that
person's sessions before the difference is treated as reproducible.

Even a clean result supports only an operational statement about these offline
analyses of the released, preprocessed recordings. It cannot identify hardware
failure, neuron loss, or representational drift as a cause; reconstruct the
historical online decoder; show how a different decoder would have changed
closed-loop behavior; or establish clinical benefit. The precise endpoint,
split, budget, and audit rules below set the boundary of that statement.

## At a glance

| Question | EP16 design |
| --- | --- |
| What data are used? | Published version 6 of the BrainGate Dryad release `10.5061/dryad.x0k6djj1h`. |
| What counts as an independent biological case? | One whole participant. All of that participant's arrays and sessions stay together. |
| What is used for development and final evaluation? | Provisionally, six whole participants for development and three for one held-out audit. Their roles freeze after the exposure record and metadata-only support check are complete. |
| What does the model see? | Mean spike-band power (`sbp`) from one fixed onset window. `tx_4_5` is a required sensitivity check, not an alternative selected because it scores better. |
| What must it predict? | One cursor-to-target unit vector for each complete trial, fixed at the start of the neural window. This is a task-derived proxy, not a direct intention label. |
| What is the basic comparison? | Fit one mapping in a source session and score it in a different target session. Compare that score with a model fit inside the target session and with a null, all on the same held-out trials. |
| Which session pairs matter most? | Every eligible directed source-to-target pair is retained. The primary contrast compares near pairs (1--30 days) with long-lag pairs (at least 180 days); 31--179 days is diagnostic only. |
| How are trials held out? | Each session contributes one frozen 128-trial packet, split by whole blocks into two 64/64 folds whose fit and evaluation roles are reversed. |
| How much target-session data can adaptation use? | Fixed nested packets of 16, 32, and 64 whole trials. The main low-budget decision uses exactly 32 labelled trials. |
| How is performance scored? | Direction-balanced held-out negative 2-D mean-squared error; higher is better. |
| How broad is development? | Forty to 96 complete panel evaluations, run on CPUs. One entire panel is selected, not whichever explanation looks most interesting. |
| What is the final test? | After the panel is locked, one evaluator opens all three audit participants together and applies it once. A supported audit result must satisfy the same pre-set decision rule independently in all three participants; it is not a participant-population `p < 0.05` claim. |

The same panel gives each participant five answers:

| Question | What the panel checks | What a clear result can support |
| --- | --- | --- |
| Is there more mapping mismatch after a long gap? | The primary and robust near-versus-long comparisons must both clear their positive margins. | A real extra loss at long lags, making the measurement and remapping explanations eligible for testing. |
| Is less usable signal recoverable in the later sessions? | Both the change in the target session's local score and the transported model's above-null score must clear their negative margins. | Under this fixed analysis, later sessions contain less usable signal rather than only a differently aligned mapping. |
| Can changes visible in the released recording reproduce the deficit? | A label-blind emulator must reproduce the eligible long-lag loss and the required near/long score changes, while passing the matched-random channel control. | Those observed recording changes are sufficient for the chronological session pairs the emulator is allowed to address. |
| Can a small labelled sample repair the mismatch? | The one locked remapping method, using exactly 32 labelled target trials, must clear both recovery margins. | A meaningful part of the mismatch can be repaired with that fixed low label budget. |
| Do participants show different repeatable patterns? | At least two fully resolved failure profiles must differ across participants, and each participant's profile must repeat in two separate session partitions. | The participants have reproducibly different profiles; this does not establish population subtypes. |

## Measurement and off-policy boundary

The public release contains preprocessed 10-ms-binned neural features, cursor
and target positions, trial/block boundaries, and electrode metadata. It
**does not** contain historical online decoder weights or versions, the full
online preprocessing state, raw voltage, target size, or a direct intention
label.

Consequently, `F(s→t)` below means:

> performance of one uniformly specified mapping fitted offline in session
> `s`, with source-fitted state, when evaluated on session `t`.

It does not mean the performance that the historical source decoder would have
achieved if deployed in session `t`. The target neural and cursor
trajectories were generated under the decoder actually used in that session.
A different deployed decoder could have changed correction, adaptation, and
the neural distribution. Every result is therefore an off-policy diagnostic.

## Observed target-schedule strata and the transport matrix

Strata match **observed target schedules**, not verified task identity. Before
matching, target coordinates are centered on the session target-set centroid
and divided by the median nonzero inter-target distance. Translation and
isotropic scale are nuisance transformations; orientation and handedness are
retained. Coordinates are neither rotated nor reflected to force a match.

The deterministic signature contains normalized target centers,
orientation-preserving axes and eccentricities, and the binary allowed-target
transition graph. It excludes transition frequencies and trial durations. A
single relative coordinate tolerance must be frozen from target-only precision
diagnostics before role assignment. Multiple layouts, unstable quantization,
conflicting graphs, inadequate target coverage, a size-dependent schedule with
missing target size, or detected material instruction differences make a
stratum `ambiguous` and exclude it from primary pairing. Missing task names
alone do not exclude a coordinate-
identifiable fixed-center schedule; the claim remains schedule-matched, not
task-matched. Fitts-like or other size-dependent schedules are excluded from
primary pairing when target size is absent.

Role assignment and scientific scorability are deliberately separate:

1. **Structural preassignment.** A trusted extractor may count declared whole
   trial epochs and inspect one target coordinate per trial, but not cursor
   trajectories or neural values. A structurally supported session–stratum cell
   has at least 128 declared trial epochs and a partition of whole blocks into
   two sides with at least 64 declared trials each. Candidate near/long pairs
   are constructed from participant, physical implanted-array set, observed
   schedule, and day.
2. **Post-role scorability.** Only after whole-participant roles and the
   structural pair manifest are hashed may the frozen scorer inspect cursor
   coordinates and neural-field integrity. Development masks are computed from
   development data; audit masks are computed atomically by the one-shot
   evaluator. It may remove unscorable structural candidates but may not add a
   pair, replace a participant, or relax a rule.

A directed pair `(s,t)` is scientifically scorable only when it has:

- the same participant and physical array set;
- the same frozen observed target-schedule stratum;
- different sessions and nonoverlapping trials;
- a frozen 128-trial packet in each session that supports the two block-disjoint
  outer folds below; and
- all required feature, block, trial, target, cursor, and electrode fields.

Primary lag classes are fixed from implant-day metadata:

- **near:** 1--30 days;
- **mid:** 31--179 days, excluded from the primary contrast but retained for
  diagnostics; and
- **long:** at least 180 days.

Structural role assignment requires at least six undirected near and six
undirected long candidate pairs in a common stratum. After role freeze, each
audit participant must still have at least six scorable pairs of each class in
one common observed-schedule graph component; counts cannot be pooled across
strata. Every contributing component must pass the rank gates below. Failure
closes as `closed_task_support_incommensurate`; it never triggers participant
replacement or a second opening. Both directions are retained whenever
scorable:

$$
F_{s\rightarrow t},\qquad F_{t\rightarrow s}.
$$

All admissible directed pairs form the source × target transport matrix. Pairs,
sessions, blocks, trials, channels, and arrays are repeated measurements, not
biological replicates.

## Frozen prediction contract

Let whole trial `q` start at released index `a_q`. For one globally shared,
human-signed onset offset `δ`, set `r_q=a_q+δ` and define a 300-ms window

$$
W_q(\delta)=[r_q,\ r_q+300\ \mathrm{ms}).
$$

The single numeric `δ` must be frozen from task timing and physiological prior,
without real neural-score optimization. It can never vary by participant,
session, pair, feature, or audit result. A trial is unscorable if `r_q` or the
complete neural window is absent or crosses its declared trial/block boundary.

Each trial contributes exactly one predictor and one proxy label:

$$
x_q=\operatorname{mean}_{\tau\in W_q(\delta)}X_\tau,
\qquad
e_q=p^{\mathrm{target}}_{r_q}-p^{\mathrm{cursor}}_{r_q},
\qquad
u_q=\frac{e_q}{\|e_q\|_2}.
$$

Trials are excluded only when `X(W_q)` or target/cursor at `r_q` is nonfinite,
the start-of-window error has zero norm, or its distance is below 5% of the
stratum's median nonzero inter-target distance. No later cursor sample affects
eligibility. Freezing both label and distance gate at the neural-window start
prevents cursor corrections during that window from changing the label or
sample set. It is still a task-derived proxy, not observed intention or an
online counterfactual.

The primary `x_q` is mean `sbp`; `tx_4_5` over the identical window is the
required feature-family falsifier. Searching other released thresholds is
forbidden. Ten-millisecond bins and overlapping windows are never separate
samples, labels, or statistical weights.

For every session–stratum cell, the scorer freezes one hash-selected,
direction-balanced 128-trial packet. Whole blocks are first assigned to one of
two disjoint sides; exactly 64 trials are then sampled within each side. A cell
without two block sets that each contain at least 64 scorable trials is
unscorable. In outer fold `k`, `C_{jk}` is one 64-trial side and `E_{jk}` the
other; the roles swap in the second fold.

Within each `C_{jk}` it freezes

$$
C^{16}_{jk}\subset C^{32}_{jk}\subset C^{64}_{jk}=C_{jk}.
$$

`B_t` and `L_t` train on all 64 target fitting trials; `F(s→t)` trains on
exactly 64 source fitting trials and scores `E_{tk}`; and `A(m,b)` alone may use
`C^b_{tk}`. “Full labels” means `C^64`, never the rest of the session. Ridge
selection is nested inside the fitting pool or globally locked from
development. Folds are aggregated before ratios.

The primary model is capacity-matched multi-output ridge regression. Direction
octants receive equal total weight; within an octant every selected trial has
equal weight. The identical weights are used for ridge fitting, null fitting,
and scoring. Whole trials and blocks stay together. Random splitting of 10-ms
bins is invalid.

The feature axis is keyed by `(implanted array identity, electrode ID)` over the
frozen physical array universe, not by column position in a session file. A
channel absent or invalid in a target session is set to the source-training
center after source-fitted scaling; it is never silently deleted. A local model
uses the same axis with target-fold-fitted scaling. This is the primary
alignment rule. Pairwise survivor electrodes are a separate control, not the
definition of the primary cohort. An entirely unavailable implanted array is
encoded as measurement-state degradation rather than changing the declared
array set.

For predictions on a held-out target fold `E`, the primary score is

$$
P(\widehat u;E)
=-
\frac{\sum_{q\in E}w_q
\left\|u_q-\widehat u_q\right\|_2^2}
{\sum_{q\in E}w_q},
$$

where fixed weights equalize angular octants without giving any trial multiple
observations. Held-out multivariate
R² and angular error are secondary. All folds are aggregated before any
ratio below is formed.

## Core performance quantities

For target session `t`, define:

$$
B_t=\text{performance of the cross-fitted intercept-only target null},
$$

$$
L_t=\text{cross-fitted performance of a target-session-trained model},
$$

$$
F_{s\rightarrow t}=\text{source-session model performance on the same target folds},
$$

and, for adaptation class `m` with exactly `b` labelled target trials,

$$
A_{s\rightarrow t}^{(m,b)}
=\text{adapted performance on those same target evaluation folds}.
$$

`L` means **session-local benchmark**, never oracle. Every quantity is
conditional on the target-direction proxy, feature family, trial eligibility,
preprocessing, finite training data, and bounded model class.

The locally recoverable above-null score is

$$
J_t=L_t-B_t.
$$

The normalized transport deficit is

$$
R_{s,t}=
\frac{L_t-F_{s\rightarrow t}}
{L_t-B_t}.
$$

The corresponding raw transport gap is

$$
G_{s,t}=L_t-F_{s\rightarrow t}.
$$

The low-budget recovery fraction is

$$
Q_{s,t}^{(m,b)}=
\frac{A_{s\rightarrow t}^{(m,b)}-F_{s\rightarrow t}}
{L_t-F_{s\rightarrow t}}.
$$

The normalized gain relative to locally recoverable above-null signal is

$$
K_{s,t}^{(m,b)}=
\frac{A_{s\rightarrow t}^{(m,b)}-F_{s\rightarrow t}}
{L_t-B_t}.
$$

The raw transported above-null score is

$$
V_{s,t}=F_{s\rightarrow t}-B_t.
$$

No ratio is clipped. Negative values and values above one remain visible. `R`
and `K` require the target `J_t` readiness gate; `Q` requires its aggregated
raw-gap gate. No failed target or pair is silently dropped after scores are
known: a required denominator failure makes that normalized signature
`unresolved`. Raw `L-F` and `A-F` remain diagnostics; `F-B` enters a terminal
only through the prespecified signal-reduction route `D_i` below.

For a directed-pair quantity `Z`, define the frozen target-node-balanced mean
within lag class `C` as

$$
\langle Z\rangle_{i,C}=
\frac{1}{|T_{i,C}|}
\sum_{t\in T_{i,C}}
\frac{1}{|S_{i,C}(t)|}
\sum_{s\in S_{i,C}(t)}Z_{s,t}.
$$

This gives every target session equal total influence and every source into the
same target equal conditional influence. It is used wherever a participant
aggregate is stated below. The participant, not a node or pair, remains the
inferential unit.

## Excess long-lag transport deficit

Within participant `i`, the primary lag estimand uses every scorable directed
pair in a constrained two-way session model. Because schedule strata can
create disconnected graphs, each connected session–stratum component `g` is
fit separately:

$$
R_{s,t}=a_{s,g}+b_{t,g}+H_{i,g}\,\mathbf 1\{(s,t)\in\mathcal L_i\}
+\epsilon_{s,t}.
$$

Source and target effects have sum-to-zero constraints **within component**;
mid-lag pairs are omitted and target sessions receive equal total weight. Each
contributing component must contain both lag classes, have full constrained
design rank, and pass human-frozen condition-number, residualized-long-column,
and maximum-node-leverage thresholds. Component estimates are combined with
equal stratum weight and, within stratum, equal component weight:

$$
H_i=\sum_g\omega_{i,g}H_{i,g},
\qquad \sum_g\omega_{i,g}=1.
$$

Failure of any frozen estimability gate is support failure, not a numerical
result. Thus `H_i` is excess long-lag deficit after general source and target
difficulty. Uncertainty uses a frozen session-node bootstrap that rebuilds all
induced directed pairs and then resamples blocks/trials within sampled nodes;
calibration packets remain fixed and pairs are never bootstrapped as
independent observations.

The approved transparent robustness summary is also required:

$$
H_i^{\mathrm{med}}=
\operatorname{median}_{t\in T_{i,\mathcal L}}
\operatorname{median}_{s\in S_{i,\mathcal L}(t)}R_{s,t}
-
\operatorname{median}_{t\in T_{i,\mathcal N}}
\operatorname{median}_{s\in S_{i,\mathcal N}(t)}R_{s,t}.
$$

A harmful-long-lag gate requires the primary and robustness summaries to have
the same positive direction and to clear their frozen practical margins.

## Session-local signal reduction

For each observed-schedule component `g`, retain chronological pairs `s < t`
whose source `J_s` passes readiness. A component contributes only if it contains
both near- and long-lag pairs. Define

$$
S_{i,g}=
\left\langle\frac{J_t-J_s}{J_s}\right\rangle_{i,g,\mathcal L^+}
-
\left\langle\frac{J_t-J_s}{J_s}\right\rangle_{i,g,\mathcal N^+},
$$

$$
D_{i,g}=
\left\langle V_{s,t}\right\rangle_{i,g,\mathcal L^+}
-
\left\langle V_{s,t}\right\rangle_{i,g,\mathcal N^+},
$$

and aggregate with the same equal-stratum, then equal-component construction
used for `H_i`, renormalized over components eligible for this route:

$$
S_i=\sum_g\omega^{\mathrm{local}}_{i,g}S_{i,g},
\qquad
D_i=\sum_g\omega^{\mathrm{local}}_{i,g}D_{i,g}.
$$

Pair means within each lag class retain equal target-node, then equal-source
weighting. Computing the contrast inside a common component prevents a
different near/long schedule mixture from creating the raw bypass. A
sufficiently negative `S_i` means that later target sessions have a larger
fractional loss than the near-lag reference under the frozen contract; it does
not mean that motor cortex has lost movement information. `D_i` separately
requires an above-null transported-score decline.

A session-local signal-reduction signature requires both `S_i` and `D_i` to
clear frozen negative margins. Thus a local-score decline alone is not called a
cross-session failure, while failure of the normalized `R` denominator does
not make the signal-reduction mode logically impossible. This raw route cannot
support measurement-sufficiency or remapping claims.

## Budget × flexibility adaptation panel

Target exposure and model flexibility are separate axes:

| Target exposure | Low complexity | Medium complexity | High complexity |
| --- | --- | --- | --- |
| 0 target samples | frozen source model and source-fitted state | — | — |
| fixed `X` only | causal streaming mean/scale | bounded unlabeled covariance or orthogonal alignment | unrestricted adaptation prohibited |
| 32 labels | gain, bias, and 2-D output rotation | orthogonal or low-rank latent remapping | bounded supervised alignment |
| full labels | session-local ridge benchmark | capacity-matched latent local model | one development-frozen ceiling model |

For each target fold, the fixed-packet zero-label methods receive exactly
`X(C^32)` and never `Y(C^32)`; supervised methods receive the identical
`X(C^32),Y(C^32)`. Budgets 16 and 64 use the nested packets above. Every method
fits once before `E_t` is opened and may not use evaluation features to choose
normalization, covariance, rank, stopping, or a transform. A separately named
rolling-origin branch may use the identical past-only unlabeled stream across
competitors, logs cumulative `X` exposure, and cannot see the current or future
sample. Whole-session transductive normalization is a full-`X` diagnostic
upper bound; it earns no zero-label, low-cost, or deployment credit.

Before candidate scoring, every bounded method needs a human-signed numeric grammar:
maximum latent rank and effective degrees of freedom, ridge/shrinkage grids,
condition-number and singular-value bounds, normalization warm-up/time
constants, transform-norm and temporal-stability thresholds, and channel-QC
cutoffs. All are currently unset scoring prerequisites.

One complete method--hyperparameter recipe `m*` alone defines audit recovery.
Before any candidate-discriminating score is opened, the bounded grammar is
expanded into a finite, hashed set of admissible recipes. Each recipe is held
fixed while it is scored: pair-specific coefficients use only source `C^64`
and the prescribed target `C^32`, and evaluation uses only `E^64`. `E^64`
never fits a recipe's coefficients, rank, regularization, normalization, or
stopping, although development `E^64` scores do enter the one declared
development-only recipe-selection step below.

Fold scores are combined before long-pair aggregation, using equal stratum
weight, then equal target-node weight, then equal sources within target. First,
an exact leave-one-development-participant-out influence check selects a split
winner using only the other support-qualified participants and evaluates that
frozen winner on the held-out participant. Each split uses the same maximin
scalar and tie rule described below. Split-winner disagreement is reported as
an influence diagnostic, not converted into an undefined pass threshold or a
new terminal gate; inability to produce a valid winner in every split is an
integrity failure. Second, every fixed recipe's already-computed scores from
all support-qualified development participants are combined; its scalar is
the minimum participant-level aggregated `A(m,32)`, and the final `m*`
maximizes that scalar. This final step explicitly uses all development
participants but no audit participant.

Within a pre-signed practical tie tolerance, choose the first in the order
output gain/bias/rotation, orthogonal remapping, low-rank remapping, bounded
supervised alignment; remaining ties use the configuration hash. There is no
cross-participant parameter refit: the chosen recipe is frozen, and every
audit pair fits only its source `C^64` and adapts only with its target `C^32`.
No method, hyperparameter, or refit rule can be chosen after audit because it
yields a preferred signature.

Causal streaming normalization uses only samples observed before the current
prediction and has one globally frozen warm-up/time constant.

## Observable-measurement counterfactual panel

For each pair, let `M_s` and `M_t` contain only frozen label-blind session
profiles: array/electrode identities, availability, finite-value rates, and
prespecified marginal channel-quality summaries. Exact-day yield and impedance
may supplement these profiles when present but cannot define the primary
complete-case cohort. In each outer fold, neural marginals use `X(C^64_s)` and
`X(C^64_t)` only, with labels hidden; evaluation features cannot define or tune
the emulator. Whole-session profiles are transductive diagnostics only.

A human-signed partial order specifies when `M_t` is observably no better than
`M_s`. The primary emulator `E(M_s→M_t)` may drop channels, attenuate signal, or
add frozen noise to match a degradation-admissible target; it may never impute
a missing channel, amplify information, or map a worse source into a better
target. The order is evaluated separately from the two fold-specific `C^64`
profiles. A pair enters the primary measurement subset only if **both** folds
classify the chronological transition as degradation-admissible. It is then
counted once for support, while its emulator remains fold-specific. A pair with
one or two improvement-like/incomparable calls, or discordant fold calls, is a
mandatory falsifier and cannot enter the sufficiency average or support count.

At least six consensus-admissible near and six consensus-admissible long
transitions are required for a participant-level measurement bit; otherwise
the participant fails measurement-panel support and the atomic audit closes as
`closed_task_support_incommensurate`. This support failure is not interpreted
as evidence for or against measurement sufficiency.

The frozen emulator applies the source-to-target change to held-out source
folds:

$$
X_s\longrightarrow \widetilde X_s^{\,M_s\rightarrow M_t}.
$$

For each source outer fold `k`, apply the frozen emulator separately to its
fitting and evaluation pools. On transformed held-out source examples compute:

- `L_tilde(s,t)`: fit on `E(M_s→M_t)(C_sk)` and test on
  `E(M_s→M_t)(E_sk)`;
- `F_tilde(s,t)`: fit the original source model on untransformed `C_sk`
  and test it on `E(M_s→M_t)(E_sk)` using source-fitted preprocessing; and
- `B_tilde(s,t)`: the corresponding transformed-data null.

No model used to compute a tilde quantity may train on `E_sk`.

Then

$$
\widetilde R_{s,t}=
\frac{\widetilde L_{s,t}-\widetilde F_{s,t}}
{\widetilde L_{s,t}-\widetilde B_{s,t}}.
$$

Let

$$
\widetilde J_{s,t}=\widetilde L_{s,t}-\widetilde B_{s,t},
$$

and compare observed and emulated **transitions**, not their absolute levels:

$$
\Delta J^{\mathrm{obs}}_{s,t}=\frac{J_t-J_s}{J_s},
\qquad
\Delta J^{\mathrm{emul}}_{s,t}=\frac{\widetilde J_{s,t}-J_s}{J_s},
$$

$$
e^J_{s,t}=\left|\Delta J^{\mathrm{obs}}_{s,t}
-\Delta J^{\mathrm{emul}}_{s,t}\right|,
\qquad
e^R_{s,t}=\left|R_{s,t}-\widetilde R_{s,t}\right|.
$$

`e^J` requires source and target `J` readiness; `e^R` additionally requires
interpretable observed and emulated `R`. For both near and long admissible
chronological transition sets, define

$$
E^J_{i,C}=\langle e^J\rangle_{i,C},
\qquad
E^R_{i,C}=\langle e^R\rangle_{i,C},
\qquad C\in\{\mathcal N^+,\mathcal L^+\}.
$$

For compact adjudication notation, set

$$
E_i^J=\max_C E^J_{i,C},
\qquad
E_i^R=\max_C E^R_{i,C}.
$$

The measurement candidate must also explain an excess deficit on the exact
chronological, degradation-admissible subset to which the emulator can apply.
Within every admissible observed-schedule component containing both lag
classes, fit the same constrained source- and target-session fixed-effect model
used for `H_i`, once to observed `R` and once to emulated `R_tilde`. Apply the
same rank, conditioning, leverage, equal-stratum, and equal-component rules to
obtain

$$
H_i^M=\sum_g\omega^M_{i,g}H^M_{i,g},
\qquad
\widetilde H_i^M=\sum_g\omega^M_{i,g}\widetilde H^M_{i,g},
\qquad
E_i^H=\max_{g:\omega^M_{i,g}>0}
\left|H^M_{i,g}-\widetilde H^M_{i,g}\right|.
$$

`H_i^M` must have a simultaneous lower bound above its own frozen positive
margin, and `E_i^H` must have a simultaneous upper bound inside its frozen
equivalence margin. The maximum component error prevents opposite component
errors from cancelling. Therefore an emulator cannot pass by reproducing easy
transitions while missing the deficit that drives the candidate. Absolute
pair errors likewise prevent cancellation across pairs. All four
class-specific `J/R` errors must also pass simultaneously. Near-lag
reproduction prevents an emulator that predicts a generic deficit at every lag
from passing as an explanation of the long-lag excess. Correlation between
yield and performance is insufficient.

Two mandatory controls accompany the emulator:

1. **Survivor-electrode analysis:** recompute `L`, `F`, `R`, and `Q`
   using only electrode IDs satisfying the frozen availability rule in both
   sessions.
2. **Matched-random degradation:** preserve the number and quality distribution
   of affected channels while randomizing identities, producing a frozen null
   bank. Its prespecified bank-average errors are computed for the same five
   components `c in {J-near, J-long, R-near, R-long, H}`. Let

$$
\Delta_i^{\mathrm{rand}}
=\min_c\left(E^{\mathrm{rand}}_{i,c}-E^{\mathrm{id}}_{i,c}\right).
$$

For `H`, both identity and random errors are the maximum absolute
componentwise excess errors just defined; random-bank averaging occurs before
that maximum. The five `E^{rand}_{i,c}` quantities receive simultaneous bounds
in the same uncertainty family. A random emulator is `adequate` only if the
upper bounds for all five errors lie inside their corresponding equivalence
margins, `ruled_out` if the lower bound for any error exceeds its margin, and
otherwise `unresolved`. Identity-aware adequacy uses the identical rule.

The random bank size, seeds, and aggregation are frozen before scores. First
adjudicate `H_i^M`: if it is explicitly below its positive margin, measurement
sufficiency is ruled out; if it is unresolved, the measurement bit is
unresolved. Conditional on supported `H_i^M`, apply this exact control rule:

- identity-aware and random emulators both adequate: support generic
  quantity/quality sufficiency;
- identity adequate, random explicitly ruled out, and `Delta_i^rand`
  supported: support channel-pattern-specific sufficiency;
- identity adequate but the random state or `Delta_i^rand` has any other
  value: unresolved;
- random adequate but identity not adequate: unresolved emulator
  inconsistency;
- both emulators explicitly ruled out: rule out observable measurement
  sufficiency; and
- every other incomplete combination: unresolved.

The random control therefore can change the main measurement bit and terminal
eligibility. Once that bit is supported, however, its generic versus
channel-pattern-specific subtype only qualifies the wording and does not
create a different candidate terminal name. The same electrode ID across years
does not imply the same recorded neuron.

## Participant-level signature estimands

For the one locked audit method `m*`, define over long-lag pairs

$$
N_i=\left\langle A_{s\rightarrow t}^{(m^*,32)}-F_{s\rightarrow t}
\right\rangle_{i,\mathcal L},
$$

$$
\overline Q_i=\frac{N_i}
{\left\langle L_t-F_{s\rightarrow t}\right\rangle_{i,\mathcal L}},
\qquad
K_i=\frac{N_i}{\left\langle J_t\right\rangle_{i,\mathcal L}}.
$$

The two aggregate denominators must clear separate frozen readiness margins.
These ratios of weighted sums, rather than an average of unstable pair ratios,
define the recovery decision.

Every audit signature is tri-state:

| Signature | `supported` | `ruled_out` | `unresolved` |
| --- | --- | --- | --- |
| Excess long-lag deficit | simultaneous lower bounds for `H_i` and `H_i^med` exceed their margins | an upper bound for either required contrast is below its margin | every other valid case, including denominator or post-fit numerical ambiguity |
| 32-trial recovery | simultaneous lower bounds for `Qbar_i` and `K_i` exceed their margins | an upper bound for either required quantity is below its margin | every other valid case or a failed denominator |
| Session-local reduction | simultaneous upper bounds for `S_i` and `D_i` are below their negative margins | a lower bound for either required quantity is above its margin | every other valid case |
| Observable-measurement sufficiency | `H_i^M` is positive; identity-aware `E_i^H` and all four near/long `J/R` errors are equivalent; and the random-control rule resolves to generic or identity-specific support | `H_i^M` is below its positive margin, or both identity and random emulators are explicitly outside equivalence | every other valid case, including identity/random inconsistency, uncertain separation, or a failed readiness gate |

Equality at a margin is `unresolved`. Failure to prove equivalence, recovery,
or reduction is never treated as evidence of absence.

## Margins and inference

The following are proposed human-facing anchors, not frozen decision values:

| Quantity | Proposed anchor | Current status |
| --- | ---: | --- |
| Harmful excess long-lag deficit | `H_i ≥ 0.10` normalized units | human sign-off required |
| Low-budget recovery | `Qbar_i ≥ 0.50` and `K_i ≥ 0.10` | human sign-off required |
| Session-local reduction | `S_i ≤ -0.15` | human sign-off required |
| Raw transported decline | not yet set for `D_i` | synthetic qualification and human sign-off required |
| Measurement equivalence | absolute normalized errors in both `J` and `R` at most 0.10 | human sign-off required |
| Admissible measurement deficit and reproduction | not yet set for `H_i^M` and `E_i^H` | synthetic qualification and human sign-off required |
| Identity-versus-random separation | not yet set for `Delta_i^rand` | synthetic qualification and human sign-off required |
| `J` and raw-gap readiness | not yet set | synthetic qualification and human sign-off required |

Before any candidate-discriminating neural score is opened, one immutable
margin and bounds record must be signed after: the structural support matrix is
known, the complete score/fold implementation passes synthetic fixtures, and
the estimator's numerical scale is verified. It includes the target-coordinate
tolerance, single onset offset, channel-QC rule, method bounds, rank and
leverage thresholds, denominator gates, effect/equivalence margins, tie
tolerance, and uncertainty rule. Synthetic qualification may reject an
infeasible value; it cannot choose, relax, or widen a scientific margin.
Failure to freeze the record blocks candidate scoring and audit interpretation.

The participant is the inferential unit. Development uses leave-one-participant
out emulation. Within-participant bounds use one frozen simultaneous
session-node/block bootstrap family covering `H`, `H^med`, `Qbar`, `K`, `S`,
`D`, `H^M`, identity and random `E^H`, `Delta^rand`, and the identity and
random near/long `E^J` and `E^R` components for `m*`; the resampling and
familywise calibration rule must be signed before scores. These bounds
describe within-participant stability and
do not turn sessions or pairs into population replicates. Audit is strict
adequacy: all three participants must clear the same frozen tri-state rule.
With three signs, the smallest exact sign-flip probability is `1/8 = 0.125`, so
no ordinary participant-population `p ≤ 0.05` claim is permitted.

## Adaptive branch logic

The adaptive object is one complete diagnostic panel, not whichever decoder
maximizes drift.

1. If `L_t` remains adequate while `F(s→t)` falls, prioritize
   causal normalization, output calibration, and bounded alignment.
2. If `L_t` also falls, prioritize measurement emulation, survivor channels,
   feature-family sensitivity, and trial-definition falsification.
3. If the emulator reproduces `J_t` but not `R(s,t)`, conclude that the
   measured state may explain signal quantity but not mapping mismatch.
4. If output-only calibration is adequate, higher-dimensional alignment becomes
   retirement-eligible only after full coverage and two native post-coverage
   falsifiers; one failed trial can never retire a branch.
5. If only higher-dimensional alignment succeeds, require transform stability,
   low-rank necessity, equal label budgets, and shuffled-target controls.
6. If task strata disagree, test task/instruction confounding before assigning
   participant heterogeneity.

Every proposal records its parent trial, observed trigger, changed operator,
and predicted signature before execution. At least two genuine adaptive
successor cycles and two incumbent/challenger decisions are required. After
coverage, at least 40% of valid trials must be falsifiers, ablations, influence
checks, or reciprocal-pair tests.

The valid identity-aware measurement emulator is never retired because it
fails equivalence: that failure is a scientific result. Emulator variants may
be rejected only for synthetic invalidity, leakage, or non-executability, and
one adjudication emulator remains in every complete panel. The exact 20-row
coverage allocation is frozen in `SEARCH_POLICY.yaml`; completing fewer rows
cannot be called branch coverage.

## Required controls and falsifiers

- intercept-only and block-preserving shifted-label nulls;
- whole-block versus deliberately invalid random-bin splitting regression test;
- `sbp` primary versus fixed `tx_4_5` sensitivity;
- source-quality and target-quality fixed effects plus within-source and
  within-target matched contrasts;
- reciprocal `s→t` versus `t→s` consistency;
- direction-octant and observed-target-schedule balance checks;
- 16/32/64 label-budget curves with identical calibration trials;
- causal streaming versus whole-session transductive normalization;
- output-only versus orthogonal/low-rank versus bounded supervised adaptation;
- shuffled target labels for every supervised adaptation branch;
- identity-aware emulator versus matched-random channel degradation;
- two-fold measurement-admissibility consensus with discordant-fold calls as
  falsifiers;
- degradation-admissible chronological transitions versus reciprocal,
  improvement-like, and incomparable no-information-creation falsifiers;
- survivor-electrode analysis;
- local-model learning curves to distinguish limited training data from lower
  recoverable score;
- omission of incomplete impedance/yield covariates from the primary emulator;
- leave-one-development-participant and leave-one-task-stratum influence; and
- negative controls showing that a performance correlation alone cannot pass
  measurement equivalence.

## Stages and lock

1. **Metadata-only feasibility:** build the task/support matrix, complete the
   literature-exposure ledger, choose 6/3 participant roles, and create
   content-addressed handoffs without opening audit neural outcomes.
2. **Synthetic and integrity qualification:** verify folds, score signs,
   denominator behavior, known mismatch/degradation/reduction fixtures,
   emulator recovery, leakage traps, and compute profile. Freeze human-signed
   margins.
3. **Balanced coverage:** run the exact 20 complete development-panel rows in
   `SEARCH_POLICY.yaml`, spanning every adaptation and counterfactual family.
4. **Adaptive development:** continue to 40--96 valid panels under the lineage,
   falsification, patience, and resource rules in `SEARCH_POLICY.yaml`.
5. **Panel selection:** only after coverage, two adaptive successor cycles, two
   incumbent/challenger decisions, and the post-coverage falsifier quota,
   choose one complete panel by frozen reliability, falsifier survival,
   leave-one-participant stability, and complexity rules; never select the
   interpretation that looks most interesting.
6. **Configuration lock:** hash code, environment, data identities, participant
   roles, structural candidate-pair manifest, scorable-mask algorithm, folds,
   features, proxy, models, budgets, margins, emulator, nulls, decision table,
   and exact audit command. The evaluator-derived audit scorable subset does
   not exist yet; its hash is emitted atomically with the audit result.
7. **One-shot audit:** the trusted evaluator opens all three participants once,
   runs the locked panel, emits the complete signature vector, and provides no
   feedback to search.

## One-shot outcome classes

The evaluator applies this precedence exactly:

1. Integrity, exposure-firewall, or evaluator failure is `technical_failure`.
2. Structural/scorable support or fixed-effect rank failure is
   `closed_task_support_incommensurate`.
3. For each participant adjudicate two entry gates: normalized excess mismatch
   (`H_i` and `H_i^med`) and the local-signal route (`S_i` and `D_i`).
4. If normalized mismatch is `supported` in all three participants,
   independently adjudicate observable-measurement sufficiency, recovery by
   the single locked `m*`, and session-local reduction. Any unresolved bit
   yields `closed_unexplained_or_unresolved`. Otherwise let `P_i` be the set of
   supported bits. Identical one-bit sets produce the matching single-signature
   candidate; identical multi-bit sets produce the composite candidate; and
   three identical empty sets close as unexplained. A single-signature outcome
   in this route requires every omitted bit to be explicitly ruled out.
5. If normalized mismatch is not supported in all three but the local-signal
   route is `supported` in all three, yield
   `candidate_ready_session_local_signal_reduction`. This route does not claim
   that measurement or remapping signatures were ruled out; their normalized
   precondition was absent or unresolved.
6. Otherwise, if both entry gates are `ruled_out` in all three, yield
   `closed_no_long_lag_excess_transport_deficit`. Differing fully resolved gate
   or signature profiles yield heterogeneity only if its frozen replication
   rule passes. Other resolved disagreement yields
   `closed_nonreplicating_long_lag_deficit` or, after a common normalized gate,
   `closed_mixed_nonreproducible_mechanisms`. Any unresolved required state
   yields `closed_unexplained_or_unresolved`.

Before audit, heterogeneity eligibility requires two disjoint,
time-stratified session-node partitions for each participant, each containing
both lag classes and independently passing the rank/profile gates. At least two
distinct fully resolved profiles must occur across participants; each
participant's profile must agree in both partitions and in the full panel; and
the development-frozen classifier must assign it without audit-specific
thresholds. If this structural requirement fails, the heterogeneity terminal
is disabled before neural access rather than weakened afterward.

- `candidate_ready_observable_measurement_sufficiency`
- `candidate_ready_low_budget_remappable_mismatch`
- `candidate_ready_session_local_signal_reduction`
- `candidate_ready_reproducible_composite_signature`
- `candidate_ready_reproducible_failure_heterogeneity`
- `closed_no_long_lag_excess_transport_deficit`
- `closed_nonreplicating_long_lag_deficit`
- `closed_mixed_nonreproducible_mechanisms`
- `closed_task_support_incommensurate`
- `closed_unexplained_or_unresolved`
- `technical_failure`

The matched-random control participates in the measurement bit and can
therefore change terminal eligibility. Once measurement sufficiency is
supported, its generic versus channel-pattern-specific subtype only qualifies
wording and does not create a different terminal name. `tx_4_5`, alternate lag
cutoffs, survivor-only transport, and other declared sensitivities are
non-rescuing diagnostics:
their failure narrows a supported primary claim to the SBP primary contract but
does not overturn it unless a predeclared integrity or leakage gate fails.
Heterogeneity supports reproducible different signatures among these
participants, not stable population subtypes.

## Prior work and novelty boundary

The provider study and release already support analyses of yield, decoding
signal-to-noise ratio, tuning stability, and electrode-count scaling.
MINDFUL studied neural-distribution shift and fixed-decoder closed-loop
performance in T5/T11. PRI-T compared recalibration methods across 73 T5
sessions and showed that long intervals can defeat multiple methods. NoMAD
showed that latent-dynamics alignment can improve stability, including a
narrow human analysis.

Therefore none of the following is novel here: drift exists; frozen mappings
can degrade; calibration can help; yield correlates with decoding; or latent
alignment sometimes works. The episode's possible increment is a common
task-, budget-, model-, and audit-matched diagnostic panel that directly tests
observable-measurement sufficiency, low-cost remappability, and session-local
recoverable signal across multiple participants.

## Claim boundary

A successful episode may support only an operational statement under the
frozen proxy, feature, session, and model contract. In particular, a supported
measurement result means sufficiency for the chronological,
degradation-admissible portion of the replicated excess deficit; it does not
claim to explain every directed pair contributing to `H_i`. It cannot
establish:

- historical online-decoder performance or a true closed-loop counterfactual;
- additive causal percentages;
- hardware degradation, neuron loss, or representational drift as causes;
- direct conscious intention or disappearance of movement information;
- equality of neuron identity across a stable electrode ID;
- a population prevalence estimate from three audit participants;
- clinical readiness or deployment benefit; or
- independent confirmation beyond this publication-exposed dataset.

The audit is a campaign-sealed strict replication of a locked procedure on
whole participants. Independent confirmation would require new compatible
participants or prospectively collected sessions under a separately governed
episode.
