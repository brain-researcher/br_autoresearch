# Can EEG forecast a movement before peripheral sensors detect its onset?

Suppose a decoder raises an alarm 100 ms before the recorded movement onset.
That sounds like a forecast, but the hand or muscles may already have started
to change, and an acausal filter may have carried those future changes backward
in time. A model can therefore score well while recognizing a movement that is
already beginning rather than predicting one that is still in the future.

EP19 moves the decision earlier. Every 50 ms while a participant is still at
risk, a model predicts when movement will begin over the next 900 ms. The main
question concerns the 300–600 ms interval before the earliest onset found by a
fixed panel of non-EEG sensors. All filtering, normalization, and model state
must use only information available at the moment of the prediction.

The first test asks whether EEG adds useful information beyond what is already
known from the cue, task context, and past peripheral signals. The same strong
non-neural model is scored with and without real EEG. Real EEG must also beat
retrained surrogate-EEG models with the same capacity. A high absolute score is
not neural evidence if the non-neural history or a scrambled EEG input can
produce it.

EP19 also asks whether changing the timing of the question changes which model
looks best. For the WAY-EEG-GAL ranking evaluated on series 8 and 9, each
prespecified model family is fit once on the permitted training series to
predict all horizons, then scored both near the event, at 0–300 ms, and at the
stricter 300–600 ms horizon. Their ordering may remain stable, disappear, or
reverse. That comparison is kept separate from a bounded search for one better
forecasting model, so the newly selected model cannot redefine the reference
panel it is meant to challenge.

An apparent forecast can still come from cue timing, a weak peripheral
baseline, future-looking preprocessing, state carried across recording gaps,
extra model capacity, or a few favorable participants. The design therefore
uses source-specific non-neural baselines, strict state resets, hidden
surrogate EEG, participant-level estimates, and a second task with eight
held-out participants.

This is the larger scientific question:

> When a decoder is no longer allowed to recognize a movement that has already
> begun, what prospective information remains in EEG, and which model
> advantages survive the earlier prediction horizon?

The intended paper should show the full transition from detection to
forecasting. It should report how much real EEG adds at each horizon, which
model comparisons genuinely change, whether the selected forecast stays
calibrated, and whether the result repeats in new participants performing a
different self-paced task. A model that succeeds only near onset supports an
event-recognition result, not a strict forecast. A model that forecasts on the
development task but fails in the second participant audit supports a narrower
dataset-specific result.

Even a positive result would not identify conscious intention, prove a causal
motor-preparation mechanism, locate the earliest biological command, or show
that the model is useful in an online BCI. It would establish something more
specific: under the registered sensors and past-only pipeline, EEG improves a
calibrated 300–600 ms movement-onset forecast beyond strong non-neural and
capacity-matched surrogate baselines.

## At a glance

| Question | EP19 design |
| --- | --- |
| What must be predicted? | At each 50 ms decision point, the probability that movement will begin in each part of the next 900 ms. |
| What is the main window? | Movement onset 300–600 ms in the future, before the onset detected from the registered peripheral sensors. |
| What is compared? | A strong cue/context/peripheral-history model versus the same model with real EEG. |
| What checks that EEG itself matters? | Real EEG must beat a zero-EEG copy and eight separately retrained surrogate-EEG models with the same capacity. |
| What model-ranking question is asked? | On WAY-EEG-GAL series 8 and 9, whether pairwise conclusions among a frozen set of model families are retained, lost, newly visible, or reversed when moving from 0–300 ms to 300–600 ms. |
| What is searched separately? | One challenger model that may improve the strict forecast but can never be added back into the frozen ranking panel. |
| What is held out? | WAY-EEG-GAL series 8 and 9 for one joint opening, plus eight whole participants from a second self-paced task. |
| What can the first study conclude? | Whether EEG contains incremental prospective information at the strict horizon and whether model conclusions depend on forecast timing. |
| What would strengthen the result? | The forecast remains calibrated, the EEG gain survives every leakage and surrogate control, and the gain repeats after the fixed 32-event calibration in the second-task participants. |

## Three tracks

### Track A — historical compatibility

Before audit, reproduce a legacy-compatible bridge on series 1–6 to series 7,
using the six released event definitions and mean column-wise AUROC. Report
`HandStart` separately because it is the closest legacy event to the new onset
target. The exact historical series 1–8 to series 9 reproduction is run only
after the joint audit as a non-promoting diagnosis; otherwise it would open the
no-feedback series 8 before lock.

This track is an implementation check and historical bridge. It does not
support a premovement claim. The public multimodal source has only series 1–9;
Kaggle series 10 and its complete multimodal outcomes are not publicly
available. Unless the original entrants' code, features, checkpoints, and
predictions are recovered, EP19 must not claim to have rerun or reordered the
literal historical leaderboard.

The archival anchor is the public winning repository at pinned commit
`36fe555d523c3ca3f201e765b1b1004dc5383dd2`, using its compact `Safe1` stack.
A native historical execution and a modern semantic port are different
artifacts; the detailed compatibility checks live in `DATASETS.md`.

The original solution used causal filtering but carried state across
concatenated series. Track A may reproduce that archival behavior after audit;
all scientific forecasting runs reset filters, normalizers, and recurrent state
at every acquisition gap and series/session boundary. The difference between
state carry and strict reset is itself reported as a benchmark audit, not hidden
inside preprocessing.

### Track B — strict forecasting and model training

Rebuild the target from raw non-EEG sensors, train every reference family on a
common discrete-time survival objective, measure neural information beyond a
frozen non-neural baseline, and conduct the bounded challenger search. This is
the primary scientific track.

WAY fitting is participant-specific: P1–P12 receive separate model weights fit
on their own series 1–6 and, after lock, refit on their own series 1–7. Model
family, preprocessing, hyperparameters, epoch rule, and seed schedule are one
common recipe selected by participant-macro development scores; they are never
selected separately per person. Cross-participant outcome pooling is a
non-promoting diagnostic, not the primary WAY fit. This differs from the
self-paced audit-facing group encoder, which is trained across its 15
development participants.

Two analyses are reported separately:

- **joint multi-horizon fit:** each family is trained once from scratch with
  the common 19-category objective and the same selection budget, then that
  same fitted recipe is scored at 0–300 and 300–600 ms; this is the **primary
  ranking mode**; and
- **representation transfer:** an encoder trained on the near-event task is
  frozen and only an equal-budget forecast head is refit; this is a secondary
  mechanism diagnostic and cannot determine the terminal rank label.

The first asks whether a model family's relative advantage depends on forecast
horizon without confounding the comparison by refitting a different model.
The second asks whether its near-event representation already contains earlier
prospective information. They are not interchangeable explanations.

### Track C — second-task whole-participant audit

The final configuration is evaluated on eight whole held-out participants in a
different, self-paced reaching task. The primary audit regime uses the first 32
complete labeled events only for frozen low-dimensional calibration; the main
encoder cannot be updated. Zero-label transfer is a stricter secondary result,
and a full local refit is a non-promoting upper-bound diagnostic.

“First 32” means the earliest 32 complete eligible reaches independently for
each participant in chronological acquisition order. Their entire stream
prefix, all eligible calibration anchors and censoring targets, and the full
history guard are calibration-only. C0 and C32 are scored on the identical
suffix after that guard. Supervised C32 updates are limited to one temperature
and three free coarse-horizon logit offsets; the encoder, spatial projection,
and 19 unconstrained category biases cannot change. Unlabeled channel affine
state may update only sample by sample. Calibrated parameters persist across
that participant's later sessions, while filter/recurrent state resets at every
session boundary.

Before any audit signal is opened, freeze the complete C32 calibration map as
\(\operatorname{softmax}(z/T+A\gamma)\): \(T>0\), \(T=1\) at initialization,
and \(\gamma=0\) initially, with one offset for each of the 0–300, 300–600,
and 600–900 ms groups and the `>900` group fixed as reference. Calibration
minimizes the natural-prevalence-weighted, censoring-aware 19-category NLL over
the declared packet. The regularization strength, parameter bounds, optimizer,
maximum iterations, convergence tolerance, and deterministic tie/failure rule
must be frozen before C32 sees candidate-discriminating outcomes and remain
identical across models. Until those fields are populated, C32 is not
executable.

The evaluator fits this four-parameter map **separately to an isolated copy of
every scored model**, including standalone \(B_d^\star\), real EEG, zero EEG,
and every surrogate/reference/challenger branch. All copies receive the same
participant packet, algorithm, parameter budget, and failure rule, but their
four fitted values are model-specific. The self-paced C32 versions of
\(\Delta\), \(G\), \(C^{final}\), and zero-twin noninferiority compare these separately
calibrated copies on their identical suffix. Canonical checkpoints and all
encoder, baseline, and residual weights remain unchanged.

The non-promoting Cfull diagnostic uses a different, fully specified support.
For each audit participant with at least four complete valid runs, the earliest
\(\lfloor R/2\rfloor\) runs form a local-refit prefix and all later runs form
the score suffix. Inside the prefix, complete runs are split chronologically as
close as possible to 80/20, with at least one training and one validation run.
That split selects only the epoch count. Every inner fit starts from an
**isolated diagnostic copy of the locked checkpoint** and uses the frozen seed
schedule for minibatch order and stochastic operators. The evaluator then
resets another isolated copy to that same locked checkpoint and fine-tunes it on
the entire prefix for exactly the selected number of epochs. The architecture,
preprocessing graph, loss, optimizer, hyperparameters, twin construction, and
seed schedule cannot change; only otherwise trainable weights in the diagnostic
copies of the source-specific baseline and the real, zero, and surrogate twins
may update. This never mutates canonical
\(B_d^\star\), C0, C32, or their stored predictions. No suffix label,
signal-derived statistic, or early-stopping result enters that refit. Cfull has
its own score support and can neither promote nor rescue the primary C32 claim.

Audit-facing group encoders and the self-paced non-neural baseline are fit only
on the 15 designated self-paced development participants under the locked
recipe. Their EEG outcomes may inform development. No audit-participant signal
may enter pre-lock training, self-supervised pretraining, model selection,
normalization-state initialization, or canonical/C0/C32 encoder or baseline
weight fitting. The only supervised audit-time fit allowed for the primary
regime is the audited four-parameter C32 call above; isolated, non-promoting
Cfull copies follow their separate rule above. The primary audit therefore
tests replication in
new whole participants from a development-exposed second task under a fixed
32-event calibration budget. It is not an unseen-task or direct
WAY-weight-transfer claim. A separate WAY-to-self-paced weight-transfer
analysis is non-promoting.
During search, second-task performance is estimated with one frozen five-fold
whole-participant split of those 15 people; only aggregate fold-complete output
is returned, and it consumes the same logical feedback query as the
corresponding WAY series-7 result. After lock, the final group recipe fits all
15 before the eight audit participants are opened.

Because this dataset has wrist accelerometry but no EMG, it can confirm a claim
about prediction before accelerometer-detectable wrist motion, not the stronger
WAY-specific claim about prediction before any measured muscle activation.
This is a campaign-sealed, second-task participant replication of a relation,
not a globally untouched dataset or a claim of universal zero-shot transfer.

## Competing explanations

1. **Prospective neural information.** EEG improves calibrated 300–600 ms
   forecasts beyond strong past-only cue, context, and peripheral history, and
   the improvement exceeds the full eight-member surrogate-EEG bank frozen
   before locked evaluation.
2. **Near-event recognition.** Performance is strong only at 0–300 ms and
   disappears after the strict safety interval.
3. **No resolved primary EEG increment.** EEG does not clear the practical
   300–600 ms increment margin beyond the frozen non-neural history. This does
   not prove that the baseline is sufficient or that the brain is uninformative.
4. **Benchmark-target dependence.** Strict forecasting retains neural signal
   but changes one or more well-resolved pairwise model conclusions.
5. **Stable inductive advantage.** Well-resolved reference-panel orderings on
   WAY survive the target change, and the selected model's neural increment
   replicates in held-out participants from the second task.
6. **Second-task participant nonreplication.** A development result exists, but
   the frozen self-paced procedure does not generalize to its eight new
   participants.
7. **Underidentification.** Participant-level uncertainty is too large to
   distinguish retention, collapse, and reversal.
8. **Temporal leakage.** Apparent prediction is caused by acausal filtering,
   future normalization, split overlap, onset construction, or movement
   artifact. This is a validation failure, not a biological result.

## Time, onset, and risk set

Let \(t\) be a decision time and let \(\mathcal H_t\) contain only samples and
states available at or before \(t\). Let \(\tau\) be the onset time returned by
a frozen sequential detector that never reads EEG.

Every candidate is queried at 50 ms intervals and emits one categorical
time-to-onset distribution over the next 900 ms. With 18 bins of width 50 ms,

\[
p_{t,k}
=
P(\tau-t \in I_k
\mid \tau>t,\mathcal H_t),
\qquad k=1,\ldots,18.
\]

The intervals are left-closed and right-open:
\(I_k=[50(k-1),50k)\) ms. If administrative follow-up ends at \(c<900\)
ms after an anchor, the evaluator deliberately discards the partially observed
bin by setting \(u=50\lfloor c/50\rfloor\) ms. Its training likelihood is the
sum of masses for intervals whose lower edge is at least \(u\), including the
`>900` mass. Thus a straddling bin contributes no partial-bin information; if
\(u=0\), that anchor contributes the neutral likelihood one.

An additional mass \(p_{t,>900}\) represents no onset in the next 900 ms, and
all 19 masses sum to one. These are interval probabilities, not recursive
conditional hazards. A true hazard implementation is allowed internally only
if the evaluator converts it deterministically to this same distribution.

An anchor with a complete 900-ms future contributes ordinary categorical log
loss. If acquisition ends earlier for a reason unrelated to the candidate, its
likelihood is the sum of all still-compatible interval masses; it is not labeled
as a negative. Outcome-dependent truncation makes the anchor ineligible.

The same distribution yields the three fixed endpoints: 0–300, 300–600, and
600–900 ms. The 300–600 ms interval is primary. A model that improves only the
0–300 ms comparator cannot support the episode's strict prospective-forecasting
claim.
Endpoint scores use the binary probability obtained by summing the six
corresponding 50-ms masses; the 19-way loss remains the common training loss.
For endpoint evaluation, an anchor is included only if its status is observed
through that endpoint's upper edge, regardless of whether an onset occurred
earlier. Administratively censored anchors may contribute the compatible-mass
19-category training likelihood but never enter a selectively observed binary
endpoint score.

The eligible risk set contains only decision times that a streaming,
past-only stillness rule regards as pre-onset. A future sample may confirm that
an onset label was real, but it may not decide that a time was eligible, enter
the model input, set normalization, or select a negative anchor. If anchors
are subsampled, their inclusion probabilities are saved and inverse-probability
weights reconstruct the natural risk distribution.

### Dataset-specific onset contracts

For WAY, the primary detector uses the earliest validated change across the
five EMG channels and wrist-marker P4 velocity. Object motion, load, and grip
force are prespecified sanity checks. For the self-paced dataset, onset is
recomputed from raw three-axis wrist accelerometry. Provider onsets derived
with zero-phase filters are never primary.

WAY detector parameters are fit only from permitted series 1–7, and self-paced
detector parameters only from the 15 development participants. Each detector
uses one-sided filtering, a frozen threshold/dwell rule, and an explicit
ambiguous-onset exclusion. C32 cannot update the threshold, dwell, clock,
ambiguity, or onset rule. At the first provisional threshold crossing the
stream enters `pending` and emits no further scored anchors; later dwell samples
may confirm the original timestamp or prospectively re-arm the detector, but
can never reinstate intervening anchors. Timestamp, group delay, clock
alignment, missing-sensor logic, and uncertainty bound are stored. The lower
edge of the primary horizon must exceed the frozen worst-case synchronization
and filtering uncertainty; otherwise the primary forecast is ineligible.

"Earliest" always means earliest under this finite sensor-and-detector panel.
It does not mean the first biological change anywhere in the body.

## The baseline that EEG must beat

For domain \(d\in\{\mathrm{WAY},\mathrm{SP}\}\), choose and freeze a strong
source-specific non-neural survival model \(B_d^\star\) before any candidate EEG
score from that source is opened. Both instances use one frozen selection
procedure and candidate-family panel, but each is fitted only on its own
development partition because the measured peripheral modalities differ. Their
permitted history includes:

- cue/prompt state and time since cue;
- target and task context only when it is causally disclosed at decision time;
- past EMG, kinematics, force, accelerometry, or EOG when present;
- posture, recent motion state, and duration of stillness; and
- session time and missing-sensor indicators.

WAY may use its past EMG, kinematic, and force streams; self-paced may use past
accelerometry and EOG but never the completed movement's `TgtID` or chosen cup.
The finite baseline candidates are regularized logistic hazard, boosted hazard
trees, and a small causal temporal convolutional model. Once selected,
\(B_{\mathrm{WAY}}^\star\) and \(B_{\mathrm{SP}}^\star\) are immutable. For
each source, first discard any recipe that fails the frozen causal or
calibration gates. Select one common family/configuration by the highest
participant-macro natural-prevalence binary log score at 300–600 ms. Break exact
ties, in order, by lower censor-aware 19-category NLL, lower parameter count,
lower measured latency, and lexicographically smaller configuration hash. WAY
uses only participant-local, complete-series out-of-fold predictions from
series 1–6: eligible anchors are pooled by natural time within participant,
then participants are macro-averaged. It then fits separate weights per
participant and never performs participant-specific recipe selection. The
self-paced baseline applies the same rule to out-of-fold predictions from its
frozen whole-participant development split, macro-averages the 15 held-out
participant scores, and fits one group model on all 15 after recipe lock.

Every primary neural candidate is a residual time-to-onset extension. Given a
numerical constant \(\epsilon\) frozen with the configuration, first sanitize
\(B_d^\star\) once,

\[
\bar p_{B_d^\star,t,k}
=\frac{\max(p_{B_d^\star,t,k},\epsilon)}
       {\sum_j\max(p_{B_d^\star,t,j},\epsilon)},
\]

then define centered baseline logits

\[
z_{B_d^\star,t,k}
=\log\bar p_{B_d^\star,t,k}
-\frac{1}{19}\sum_{j=1}^{19}
 \log\bar p_{B_d^\star,t,j}.
\]

Use this same sanitized distribution both to score the standalone baseline and
to construct the residual offset logits. Then

\[
p_{d,m,t,\cdot}^{\mathrm{real}}
=\operatorname{softmax}\!\left(
 z_{B_d^\star,t,\cdot}
 +r_{m,\theta}(EEG_{\leq t},\bar p_{B_d^\star,t,\cdot})
 \right).
\]

The residual branch receives EEG and the frozen baseline time-to-onset vector;
it cannot reread the full behavior history. Its final layer is initialized to
exactly zero, so epoch 0 reproduces \(B_d^\star\), and epoch 0 is an eligible
early-stopping checkpoint under this common sanitizer. A flexible all-modal
interaction model may be reported as a secondary predictive system, but it
cannot support the primary neural-increment claim.

The submitter supplies one declarative graph. A trusted builder, not candidate
code, creates real-EEG, zero-EEG recalibration, and surrogate-EEG twins with the
same non-neural history, graph, temporal context, folds, initialization seeds,
optimizer credits, and calibration packet. The zero-EEG twin tests whether
training or recalibration damages \(B_d^\star\); it is not called a
capacity-matched neural null and must remain noninferior to \(B_d^\star\).

The evaluator also constructs a hidden bank, frozen with the configuration, of
four session-preserving block shifts and four cross-channel-coherent phase
randomizations. Each surrogate twin is retrained from scratch on its transformed
training stream with paired seeds; evaluation uses the corresponding hidden
transform. Block shifts exceed the input/state history, forecast horizon, and
boundary guard. To keep the adaptive budget executable, each development trial
uses the same one hidden shift and one hidden phase sentinel. Only locked
references and the finalist run the full eight-null bank. The final neural gate
compares real EEG with the prespecified participantwise maximum over the eight
eligible surrogate scores. Development therefore uses \(C^{dev}\), based on
the two shared sentinels, while locked reference/finalist and audit decisions
use \(C^{final}\), based on all eight. Each maximum is part of its prespecified
estimand; there is no second surrogate-bank multiplicity correction. A
submitter cannot choose or inspect a null transformation.

## Two leaderboards, one scientific gate

The benchmark publishes two distinct development tables:

1. **Practical forecasting:** absolute 300–600 ms proper log score of the
   source-specific baseline plus real EEG.
2. **Neural incremental value:** separate \(\Delta\), \(G\), and \(C^{dev}\) gates
   against the source-specific baseline, zero-EEG recalibration twin, and
   hidden surrogate-EEG bank.

The scientific conclusion comes from the second table and its component
contrasts. The locked challenger is selected by absolute forecasting score
only after every neural-increment, calibration, and causality gate passes.
Thus a strong cue/peripheral model may win an absolute row without being
misreported as evidence for useful EEG.

The frozen reference-panel transition matrix is a third, separate output. New
adaptive entrants never enter its denominator.

## Board and audit separation

| Stage | Data visible to training | Feedback returned | Permitted adaptation | Scientific role |
| --- | --- | --- | --- | --- |
| Open train | WAY series 1–6 | inner-fold diagnostics | preprocessing, architecture, optimization | fit and qualification only |
| Feedback board | WAY series 7 plus self-paced development CV | rate-limited source-by-horizon aggregates | bounded successors | adaptive development only |
| No-feedback lock | WAY series 8 | none until joint release | none | one locked benchmark check |
| Additional sealed evaluation | WAY series 9 | none until joint release | none | pooled campaign-sealed benchmark support |
| Second-task audit | 8 whole self-paced participants | none until joint release | only frozen C32 calibration | new-participant replication in an exposed task domain |

Series 8 and 9 are public on the internet, so this is a campaign seal, not a
claim that the data are globally hidden. The evaluator strips participant,
fold, null-bank-member, file-order, and calibration-packet details from
feedback, while retaining source-by-horizon aggregates needed by the declared
failure routes.
One challenger and the complete frozen reference panel are locked before the
last three rows are evaluated together. A runner-up cannot replace a failed
winner.

### Executable submission boundary

EP19 borrows the evaluator abstraction, not the data or metric, from FALCON.
Each submission is a declarative training graph inside a no-network OCI image,
or an equivalently reproducible recipe. The trusted builder controls which
neural transform is visible and calls:

```text
fit(role_filtered_training_stream, neural_transform_handle, seed)
freeze() -> model_hash
reset(random_stream_token, regime, minimal_structural_metadata)
calibrate(participant_specific_packet, permitted_parameter_mask)
step(causal_observation_t) -> time_to_onset_distribution[19]
boundary_reset(acquisition_gap_or_causally_disclosed_run_or_session_boundary)
```

The opaque token cannot encode participant, session, date, or filename.
Scored onsets, labels, targets, masks, and null-bank identities never enter the
container. In C32, calibration targets enter only through the audited
`calibrate` call before the scored suffix; the evaluator enforces the parameter
mask. In Cfull, the same call receives only the declared local-refit prefix and
may fine-tune all trainable parameters of diagnostic copies initialized from
the locked checkpoints under the exact refit rule above. It cannot mutate canonical
\(B_d^\star\), C0, or C32 artifacts. C0 permits sample-by-sample causal
normalization state but no learned parameter updates. C0, C32, and Cfull run in
isolated model copies. Prediction
and scoring run in separate processes under frozen CPU, GPU, memory, scratch,
wall-time, and latency meters.

No FALCON dataset enters EP19. In particular, FALCON H2 overlaps BrainGate T5
used by EP16; importing it would create a participant-level exposure collision.

## Frozen reference panel

Before forecast scores are visible, pin code, versions, preprocessing, channel
rules, parameter tiers, and search credits for these five nonredundant families:

1. causal filter-bank/band-power plus elastic-net time-to-onset model;
2. causal covariance/Riemannian tangent-space model;
3. causalized EEGNet;
4. dilated causal temporal convolutional network; and
5. a small unidirectional GRU.

Every family must be executable at both the near (0–300 ms) and strict
(300–600 ms) forecast endpoints. Centered windows, bidirectional recurrence,
`filtfilt`, future-padded `same` convolutions, and whole-record normalization
are prohibited even for classic methods.

Track A uses the archived implementation's native 32-channel montage. The
scientific Track B panel, challenger, and Track C procedure all use the frozen
29-channel intersection. Historical compatibility and scientific forecasting
therefore never silently share a channel contract.

This panel is the denominator for the ranking question. A method found during
adaptive search is a challenger, not a sixth retrospective reference model.

## Training a new model

The adaptive model family is a **Causal Residual Forecasting Network** with:

- a channel/montage projection and missing-channel mask;
- a strictly causal temporal encoder;
- a multi-horizon residual time-to-onset head;
- the frozen source-specific \(B_d^\star\) distribution as an offset; and
- at most a low-dimensional participant calibration head.

The search starts from the best eligible frozen reference recipe; it does not
rebrand the reference TCN as a distinct challenger. The first challenger that
differs from every frozen reference recipe must be a one-operator successor
justified by a recorded failure. A diagonal state-space layer is eligible only
after a named long-memory failure, and a geometry-aware channel projection only
after a montage/participant failure. External EEG foundation-model weights are
prohibited; development-only self-supervised pretraining is a separate declared
operator.

Hard limits are two million trainable parameters, two seconds of explicit EEG
input buffer, a 50 ms update interval, and a prespecified p95
inference-latency ceiling on frozen hardware. GRU/SSM state may persist within a
continuous series for at most eight seconds under a frozen truncated-BPTT and
detach rule; every acquisition gap or series/session boundary clears it.
Allowed augmentation is limited to causal crop, amplitude perturbation, and
channel dropout.

### Common training protocol

- Optimize censoring-aware time-to-onset negative log likelihood, never AUROC or
  class-balanced accuracy.
- Keep whole continuous series or sessions intact across folds. The boundary
  guard covers the longest filter, resampler, receptive-field, and state-reset
  history.
- For each frozen reference family only, run the same 12-configuration
  outcome-blind Sobol tuner inside series 1–6. These inner subfits are hidden
  from the adaptive controller. For every configuration, compute the
  participant-macro censor-aware 19-category NLL after averaging its three
  frozen-seed scores within participant. Each participant score pools only
  complete-series out-of-fold eligible anchors by natural time; participants
  are then equally weighted. Select one common configuration per family by the
  lowest such NLL; break exact ties, in order, by lower parameter count, lower
  measured latency, and lexicographically smaller configuration hash. Then fit
  separate weights for every WAY
  participant. Per-participant configuration selection is prohibited.
  The selected full twin panel counts as one scientific trial. Each adaptive
  successor is one exact hyperparameter configuration, not another nested
  12-way search. Final panels use the same three frozen seeds.
- Fit preprocessing, early stopping, and hyperparameters only inside the
  training side of the relevant continuous-series split.
- Do not prune a candidate from partial participant or outer-series results.
- After epoch selection, refit according to one frozen rule before evaluation.
- Use identical calibration events and label budgets for every model.

One valid scientific trial is a complete exact model recipe evaluated as
\((B_d^\star, B_{d,m}^{zero}, B_{d,m}^{real},
B_{d,m}^{surrogate,dev1:2})\) across all required WAY and self-paced development
participants, series, horizons, seeds, applicable C0/C32 emulations, and
leakage tests. The full eight-null bank, Cfull, and representation transfer run
only for the locked finalist/reference outputs, not inside every adaptive
trial. A participant, horizon, seed, fold, twin, or scheduler job is not
another trial.

### Adaptive successor rule

Every successor is proposed before execution and records:

- its parent scored trial and visible ledger-prefix hash;
- the concrete failure pattern and competing explanation;
- exactly one primary operator change;
- directional predictions for all three horizons;
- one native falsifier and a retirement condition;
- unchanged operators, parameter count, latency, and compute estimate; and
- code, configuration, and environment hashes.

Changing the onset, horizon, baseline, negative sampling, audit split, or
reference-panel membership is not an admissible model successor. The search
must complete at least two outcome-adaptive successor cycles and two explicit
incumbent/challenger decisions. After coverage, at least 40% of valid work is
reserved for falsification, ablation, influence, or direct replication.

A patience opportunity is one complete adaptive challenger that passes every
non-comparative integrity and eligibility gate and would be able to replace the
incumbent except for the incumbent-score comparison. It resets the counter only
when its WAY series-7 primary log score exceeds the incumbent by the scientist-
signed development improvement margin; otherwise it increments the counter.
Coverage panels, required falsifiers, ablations, influence checks, direct
replications, invalid submissions, and exact retries neither increment nor
reset patience, although they still consume their declared trial, feedback, and
resource budgets. Patience cannot stop the search until the 24-trial minimum,
required branch coverage, mandatory prelock falsifiers, and the 40% post-
coverage falsification fraction are complete.

The frozen admissible-space manifest enumerates every legal operator and
configuration before outcomes are read. Exhausting that manifest, the 40-call
feedback quota, or a compute ceiling before the 24-trial minimum is a technical
execution failure, not a negative neuroscience result. After the minimum, such
exhaustion closes development and reduces the set of otherwise valid distinct
challengers in this exact order:

1. if any challenger passes every promotion-eligibility gate and clears the
   signed improvement margin, lock the best such challenger;
2. otherwise, if any challenger's eligibility or—after eligibility passes—its
   improvement comparison is unresolved, return `closed_unresolved`;
3. otherwise, if at least one challenger passes every eligibility gate but all
   such challengers are resolved below the improvement margin, return
   `closed_no_challenger_improvement`; and
4. otherwise every otherwise valid challenger has a resolved eligibility
   failure, so return `closed_prelock_promotion_eligibility_failure`.

This reduction requires at least one otherwise valid distinct challenger;
candidate-specific integrity failures merely invalidate those submissions,
while shared evaluator, reference, or locked-finalist integrity failures are
`technical_failure`. Reaching the engineering-failure ceiling is always
`technical_failure`.

## Primary score and neural increment

For source \(d\), participant \(p\), horizon \(h\), and model \(m\), let
\(R_{p,d,h}\) be the eligible 50-ms anchors. If anchors are subsampled, let
\(w_t\) be the inverse inclusion probability; otherwise \(w_t=1\). With
binary endpoint log score \(\ell_{t,h}(m)\), define

\[
S_{p,d,h}(m)
=\frac{\sum_{t\in R_{p,d,h}}w_t\ell_{t,h}(m)}
       {\sum_{t\in R_{p,d,h}}w_t}.
\]

This weights eligible time at its natural prevalence within participant, then
weights participants equally. Participant is the only resampling unit: every
session, trial, and anchor stays attached to that participant and never becomes
a pseudo-replicate or equal-weight stratum. The censoring-aware 19-category NLL
is the training loss; the six-bin-sum binary log score is the endpoint score.

For every stochastic real, zero, reference, challenger, and surrogate branch,
compute \(S_{p,d,h}^{(s)}\) separately for each of the three paired frozen seeds,
then take the arithmetic mean of the three **scores** within participant and
branch. All contrasts below use those fixed seed-mean scores; probabilities are
never averaged into a seed ensemble. Form \(\Delta\) and \(G\) from paired branch
seed means. For \(C^{dev}\), first seed-average the two development sentinels
separately, then take their within-participant maximum. For \(C^{final}\), do
the same before taking the maximum over all eight locked surrogates. The final
challenger-versus-reference contrasts follow the same seed rule. Bootstrap resampling
therefore receives participant-level fixed seed means, not individual seeds or
ensembled predictions.

The primary neural increment is

\[
\Delta_{m,h,d}
=
\frac{1}{P}\sum_{p=1}^{P}
\left[S_{p,d,h}(m^{real})-S_{p,d,h}(B_d^\star)\right].
\]

In words, \(\Delta\) is how much calibrated forecasting improves when real EEG
is added to the strongest frozen source-specific non-neural baseline. For bank
size \(J\), define the conservative capacity-control contrast

\[
C^{(J)}_{m,h,d}
=
\frac{1}{P}\sum_{p=1}^{P}
\left[S_{p,d,h}(m^{real})
-\max_{j\leq J}S_{p,d,h}(m^{surrogate,j})\right].
\]

Set \(J=2\) for \(C^{dev}\), which alone enters adaptive development,
prelock eligibility, and development robustness. Set \(J=8\) for
\(C^{final}\), which enters locked/audit neural gates, near-horizon terminal
labels, and scientific claims. The same signed practical margin applies to both
unless a different pair is scientist-signed before outcome access.

The zero-EEG recalibration diagnostic is

\[
G_{m,h,d}
=
\frac{1}{P}\sum_p
\left[S_{p,d,h}(m^{real})-S_{p,d,h}(m^{zero})\right].
\]

Two reliability estimands are also frozen:

\[
Z_{m,d}
=\frac{1}{P}\sum_p
\left[S_{p,d,300:600}(m^{zero})-S_{p,d,300:600}(B_d^\star)\right],
\qquad
L_{m,d}=\Delta_{m,600:900,d}.
\]

The zero twin is noninferior when the simultaneous lower bound for \(Z_{m,d}\)
exceeds minus its frozen noninferiority margin. Longer-horizon non-degradation
passes when the corresponding lower bound for \(L_{m,d}\) exceeds minus its
frozen margin. For either gate, an upper bound below the negative margin is a
resolved failure and overlap is unresolved. One separate simultaneous
participant-bootstrap reliability family covers \(Z\) and \(L\) in both
sources, but routing remains source-ordered: WAY bounds are evaluated at the
WAY reliability stage, and self-paced bounds only at the second-task stage.

Participant robustness has separate development and audit rules. Before lock,
each of \(\Delta\), \(G\), and \(C^{dev}\) must be positive for at least 9 of 12 WAY
series-7 participants and at least 12 of 15 self-paced out-of-fold development
participants; every source-specific leave-one-participant-out macro mean must
also remain positive. At joint audit, every leave-one-participant-out WAY mean
must retain the positive direction; in the self-paced audit, each contrast must
be positive in at least six of eight participants and every leave-one-
participant-out mean must remain positive. These are qualification/stability
gates, not extra inferential units.

The terminal calibration gate is marginal and deliberately simple. Let
\(q^{(s)}_{p,t,d}\) be the locked real branch's 300–600 ms probability for seed
\(s\) and
\(y_{p,t,d}\) its binary outcome on the same fully observed endpoint support
used for scoring. Define participant calibration-in-the-large residual

\[
e^{(s)}_{p,d}
=\frac{\sum_t w_t(y_{p,t,d}-q^{(s)}_{p,t,d})}{\sum_t w_t},
\qquad
K_d=\frac{1}{P}\sum_p\frac{1}{3}\sum_{s=1}^{3}|e^{(s)}_{p,d}|.
\]

Thus calibration is computed per seed and then averaged as a metric; seed
probabilities are never ensembled.

One simultaneous participant-bootstrap family covers \(K_{\mathrm{WAY}}\) on
pooled series 8+9 and \(K_{\mathrm{SP}}\) on the C32 score suffix. For each
source, calibration passes only when its upper bound is below the frozen
tolerance, is a resolved failure when its lower bound exceeds that tolerance,
and is otherwise unresolved; joint success requires both sources to pass.
The shared correction does not change routing: WAY calibration is decided at
the WAY reliability stage and self-paced calibration at the second-task stage.
Near and longer-horizon calibration are reported but nonterminal. The same
metric and tolerance qualify reference recipes on their permitted development
predictions; a frozen reference is never removed after audit. This gate measures
marginal calibration, not full conditional calibration.

A neural-information claim requires \(\Delta_{m,300:600,d}\),
\(G_{m,300:600,d}\), and \(C^{final}_{m,300:600,d}\) to exceed their separately signed
practical margins, while \(m^{zero}\) remains noninferior to \(B_d^\star\).
Here \(G\) is a training/recalibration diagnostic; \(C^{final}\), not \(G\), is the
input-dependent capacity control. Calibration and participant robustness must
also pass. At the joint evaluation, simultaneous 95% participant
bootstrap lower and upper bounds cover \(\Delta\), \(G\), and \(C^{final}\) in both
sources at the primary horizon plus the three WAY near-horizon contrasts used
by the near-only terminal. A gate passes only when its lower bound clears its
signed margin, is a resolved failure only when its upper bound lies below that
margin, and is otherwise unresolved. The shared correction does not alter the
source-ordered cascade: WAY bounds are decided first and self-paced bounds only
after WAY passes. AUROC and AUPRC are secondary outside Track A.

Report raw horizon-specific increments and an entropy-normalized skill curve.
Also report cumulative real-EEG-versus-\(B_d^\star\) log-score gain in bits per
eligible minute at the fixed 50-ms stride, using the same anchor weights and
then averaging participants equally. Because anchors overlap, this is not
mutual information. It is a display scale, not a replacement objective.

The descriptive retention fraction

\[
Q_{h,d}=\Delta_{m,h,d}/\Delta_{m,0:300,d}
\]

is interpreted only when the near-event denominator clears a frozen
qualification margin. It is never clipped, never replaces raw \(\Delta\), and
is not used to select a model.

## Ranking transition

The terminal ranking estimand uses the **same jointly trained multi-horizon
model on WAY only**. It pools the jointly opened series 8 and 9 within each
participant using the natural-time score above. For every prespecified
reference pair \((a,b)\), task \(z\), and primary mode \(J\), define

\[
D^{J,\mathrm{WAY}}_{ab,z}
=
\frac{1}{P}\sum_p
\left[S_{p,\mathrm{WAY},z}(a_J)-S_{p,\mathrm{WAY},z}(b_J)\right].
\]

Here \(a_J\) and \(b_J\) are the source-specific-baseline-plus-real-EEG twins
of reference families \(a\) and \(b\). Zero-input and surrogate twins are gates
and diagnostics, never ranked entries. Compute each participant–endpoint score
separately for the three frozen seeds, then take the arithmetic mean of those
three scores before forming a pairwise edge. Do not average seed probabilities
into an ensemble. Participant bootstrap resampling operates on these fixed
seed-mean scores.

The tasks are a common-score near-event comparator and the primary 300–600 ms
forecast. The official Kaggle AUROC ordering is reported separately and cannot
be mixed numerically with proper-score differences.

The 0–300 ms comparator is a forward-looking time-to-onset task. It is not the
historical Kaggle label, which spans approximately 150 ms before through 150 ms
after an event.

For endpoint \(z\), let \([L_{ab,z},U_{ab,z}]\) be its simultaneous 95%
participant-bootstrap interval and let \(\delta_z>0\) be the frozen practical
margin. Classify an edge as **positive** when \(L_{ab,z}>\delta_z\),
**negative** when \(U_{ab,z}<-\delta_z\), **equivalent** when the entire interval
lies inside \([-\delta_z,\delta_z]\), and **unresolved** otherwise. Point
estimates never determine an edge state.

Each pairwise edge is classified as:

- **retained:** both tasks cross their practical margins in the same direction;
- **reversed:** both cross margins in opposite directions;
- **collapsed:** the near-event edge is identifiable and the forecast edge is
  practically equivalent;
- **emergent:** the near-event edge is equivalent and the forecast edge is
  identifiable;
- **stable-equivalent:** both tasks are practically equivalent; or
- **unresolved:** the relevant interval remains too wide.

A point-estimate sign flip is not a reversal. One participant bootstrap
correction covers all frozen reference pairs and both tasks simultaneously.
Kendall or Spearman correlation is only descriptive. The frozen-encoder
transfer matrix and second-task reference rankings are reported as secondary
mechanistic outputs; neither changes the terminal rank class.

## Incumbent selection

The adaptive challenger is selected lexicographically, not by best development
AUROC:

1. every causality, timing, state-reset, and split-integrity test passes;
2. in each development source, real EEG beats \(B_d^\star\), its zero-EEG
   recalibration twin, and the two-sentinel development surrogate bank
   (\(C^{dev}\)) at the primary horizon by their separate frozen margins;
3. calibration passes its frozen tolerance;
4. the participant-level worst-case rule passes and no meaningful degradation
   appears at 600–900 ms;
5. WAY series-7 strict-horizon absolute log score exceeds the best eligible
   reference model; self-paced development is a qualification gate, not pooled
   with WAY participants into one pseudo-sample; and
6. ties go to lower parameter count, latency, and compute.

Ranking disruption itself is never an optimization reward. The controller is
not allowed to search for a dramatic reversal.

The final joint evaluation does not choose a favorable reference after seeing
the data. It forms five paired participant-level differences between the locked
challenger and each eligible frozen reference on pooled WAY series 8+9. One
simultaneous 95% participant bootstrap family covers all five. The final
improvement gate passes only if every lower bound clears the same signed final
margin, is a resolved failure if any upper bound lies below that margin, and is
otherwise unresolved. This challenger family is separate from the neural-gate
and pairwise reference-ranking families.

## Required falsification

Every eligible reference model and challenger must pass:

- future-input mutation: changing samples after \(t\) leaves the prediction at
  \(t\) bit-identical;
- sample-by-sample, variable-chunk, and batched streaming replay equivalence;
- a future impulse has exactly zero influence on prior outputs;
- filter, resampler, normalization, and recurrent state reset at acquisition
  gaps and causally disclosed run or series/session boundaries;
- explicit rejection of centered filtering, future padding, and whole-session
  normalization sentinels;
- no overlapping window across a split boundary;
- cue-stratified label shuffle and circular/block EEG shift return to null;
- participant/session-ID-only prediction cannot explain the result;
- capacity-matched phase-randomized EEG cannot reproduce the neural increment;
- a pre-frozen onset-sensitivity envelope cannot create the primary conclusion;
- an intentionally leaked future-signal positive control is detected as
  abnormally strong;
- a synthetic causal premovement motif is recovered at prespecified SNRs; and
- the legacy near-event positive control reaches its frozen qualification range.

A shared evaluator/onset timing certificate or positive-control failure, or a
leakage failure in a locked reference or finalist, is `technical_failure`. A
candidate-specific pre-lock leakage failure only invalidates that submission
and consumes its declared budgets; the search must still satisfy its valid-
trial and successor requirements. Neither case is evidence that prospective
EEG information is absent.

## Development and lock sequence

1. **Metadata and provenance.** Freeze source versions, hashes, clocks, channel
   maps, participant roles, and the exposure ledger without opening held-out
   signals.
2. **Causal qualification.** Validate the onset detector, streaming operators,
   synthetic fixtures, and deliberate leak sentinels.
3. **Baseline lock.** Select \(B_{\mathrm{WAY}}^\star\) and
   \(B_{\mathrm{SP}}^\star\) with their non-EEG development folds before any
   corresponding candidate EEG score is read.
4. **Reference coverage.** Run the five frozen families on common folds,
   endpoints, budgets, and controls.
5. **Adaptive challengers.** Execute bounded one-operator successors and
   falsifiers using WAY series 1–6 for fit and aggregate series-7 board
   feedback, plus frozen five-fold whole-participant evaluation on only the 15
   designated self-paced development participants.
6. **Configuration lock.** Freeze one challenger, the reference panel, onset
   code, channel rules, folds, seeds, margins, calibration packets, evaluator,
   and exact audit command in a content hash.
7. **Joint one-shot evaluation.** Final WAY recipes refit on series 1–7; final
   self-paced group recipes fit only on its 15 development participants. The
   trusted evaluator opens WAY series 8, WAY series 9, and all eight self-paced
   audit participants together and returns no partial scores. Audit outcomes
   cannot update the search. Exact historical 1–8-to-9 compatibility may run
   afterward as a non-promoting appendix.

## One-shot outcome classes

- `candidate_ready_strict_forecast_rank_robust`: strict neural increment clears
  all gates in WAY and the second-task participant audit, the challenger beats
  the best eligible reference, and every prespecified identifiable WAY edge is
  retained.
- `candidate_ready_strict_forecast_rank_changed`: strict neural increment
  clears those gates, the challenger improves, and no edge is unresolved, but
  at least one WAY edge is reversed, collapsed, or emergent; mixtures with
  retained edges remain here.
- `candidate_ready_strict_forecast_rank_underidentified`: strict neural
  increment and challenger-improvement gates clear, but at least one
  prespecified WAY edge is unresolved.
- `candidate_ready_strict_forecast_rank_no_reference_signal`: strict neural
  increment and challenger-improvement gates clear, but every near-event
  reference edge is practically equivalent, so there is no initial ordering
  whose retention can be tested.
- `closed_near_event_but_not_primary_strict`: all three WAY near-horizon
  \(\Delta\), \(G\), and \(C^{final}\) gates clear their signed near margins, but the WAY
  primary-horizon \(\Delta\) upper bound is below its signed margin. The
  600–900 ms pattern is reported descriptively and does not change this
  primary-endpoint label.
- `closed_no_resolved_eeg_increment_at_primary_horizon`: the WAY primary-
  horizon \(\Delta\) upper bound is below its signed margin and the named
  near-horizon pattern above does not hold. This is a bounded non-detection of
  incremental value, not proof that the non-neural baseline is sufficient.
- `closed_neural_specificity_gate_failure`: WAY \(\Delta\) clears its margin,
  but real EEG has a resolved failure against its zero-input or hidden surrogate
  twin.
- `closed_candidate_reliability_gate_failure`: the three WAY neural contrasts
  clear, but a predeclared WAY zero-twin noninferiority, calibration,
  participant-robustness, or longer-horizon gate, or a global latency/resource
  requirement, has a resolved failure.
- `closed_second_task_participant_nonreplication`: all WAY and global gates
  pass, but at least one required self-paced neural, zero-twin, calibration,
  participant-robustness, longer-horizon, macro, 6-of-8, or leave-one-participant
  gate has a resolved failure.
- `closed_no_challenger_improvement`: the benchmark and neural gates are valid,
  no otherwise valid challenger's prelock state remains unresolved, and at least
  one such challenger passes every promotion-eligibility gate, but all eligible
  challengers are resolved below the signed development improvement margin after
  a valid bounded search; the same label also applies when the locked challenger
  fails the separate final WAY improvement margin against the best eligible
  frozen reference.
- `closed_prelock_promotion_eligibility_failure`: a valid bounded search
  completes its minimum, coverage, and falsification duties and at least one
  otherwise valid distinct challenger completes, no such challenger's prelock
  state remains unresolved, and every such challenger has a resolved failure of
  at least one frozen promotion-eligibility gate such as calibration, neural
  specificity, reliability, latency, or resource compliance. Candidate-specific
  integrity failures do not create this label, and shared/reference/finalist
  integrity failures remain technical. This is a non-promoting post-adaptive
  development outcome, not evidence that prospective EEG information is absent.
- `closed_unresolved`: the next decision-relevant neural, specificity,
  reliability, replication, or challenger-improvement gate remains interval-
  unresolved at the frozen resource ceiling. Ranking uncertainty after all
  candidate gates pass instead receives the candidate-ready
  `rank_underidentified` label.
- `technical_failure`: data identity, synchronization, onset construction,
  causal replay, positive control, minimum search execution, or evaluator
  integrity fails.
- `policy_violation`: held-out access, post-audit adaptation, baseline
  replacement, or reference-panel mutation violates the contract.

The four `candidate_ready_*` outcomes require the locked audit. A new model
winning development alone is not a terminal scientific result.

The terminal decision is a fixed, source-ordered short-circuit cascade. A policy
violation is assigned first, then a technical failure. Before opening the joint
audit, first lock any challenger that passes eligibility and improvement;
absent one, any unresolved challenger state gives `closed_unresolved`; if all
states are resolved, at least one eligible challenger gives
`closed_no_challenger_improvement`, and otherwise the result is
`closed_prelock_promotion_eligibility_failure`. Only a
successfully locked challenger advances to the audit. At each audit gate, an
unresolved interval gives `closed_unresolved` only if no earlier resolved
failure already determines the outcome. Within every multi-contrast stage,
reduce outcomes as **any resolved failure first, otherwise any unresolved,
otherwise pass**; an unresolved member can never hide a resolved failure in the
same stage. Evaluate WAY first: a resolved WAY
primary \(\Delta\) failure maps to
`closed_near_event_but_not_primary_strict` only when all three WAY near-horizon
gates clear their signed margins, and otherwise to
`closed_no_resolved_eeg_increment_at_primary_horizon`. If WAY \(\Delta\) passes,
a resolved WAY \(G\) or \(C^{final}\) failure gives
`closed_neural_specificity_gate_failure`; after all three pass, a resolved WAY
reliability or global latency/resource failure gives
`closed_candidate_reliability_gate_failure`. Only after every WAY and global
gate passes is the self-paced audit evaluated; any resolved failure of a
required self-paced neural, calibration, or reliability gate gives
`closed_second_task_participant_nonreplication`. Then evaluate the challenger-
improvement gate, whose resolved failure gives
`closed_no_challenger_improvement`. Only after those gates pass is rank mapped:
all practically equivalent near-event edges give the candidate-ready
`candidate_ready_strict_forecast_rank_no_reference_signal`; otherwise any
unresolved edge gives
`candidate_ready_strict_forecast_rank_underidentified`; with no unresolved
edge, any reversed, collapsed, or emergent edge gives
`candidate_ready_strict_forecast_rank_changed`; all remaining edges are
retained or stable-equivalent with at least one retained edge and give
`candidate_ready_strict_forecast_rank_robust`. This cascade is exhaustive and
assigns exactly one terminal label.

## Prior-work and claim boundary

WAY-EEG-GAL and the Kaggle competition establish a widely used event-detection
lineage. BCI Competition IV and later decoding benchmarks establish concurrent
movement reconstruction and cross-session evaluation. FALCON supplies useful
held-out-session and evaluator mechanics. None of these facts by itself shows
that performance survives a detector-defined premovement interval after a
strong peripheral/context baseline.

Premovement EEG prediction itself is not new. A seven-person online study
reported predictions averaging roughly 0.62 s before EMG-detected wrist
movement (<https://doi.org/10.1016/j.clinph.2010.07.010>), and many readiness-
potential studies precede it. EP19 therefore makes no priority claim for
"predicting movement before onset." Its new object is the conjunction of a
continuous natural-risk forecast, a strong past-only peripheral baseline,
source-specific timing certificates, hidden retrained nulls, a frozen ranking
transition, and one-shot whole-participant replication. A 2026 handwriting
study likewise found that access to movement-onset timing materially changes
decoding accuracy, reinforcing why onset knowledge must be part of the audited
benchmark contract (<https://arxiv.org/abs/2605.15698>).

WAY-EEG-GAL itself has also supported pre-onset, lagged-window kinematics
reconstruction with deep models, including within-participant and leave-one-
participant-out analyses (<https://arxiv.org/abs/2205.01050>;
<https://arxiv.org/abs/2209.01932>). EP19 therefore cannot claim the first
premovement use of WAY, the first deep model on those signals, or the first
cross-participant analysis. Those studies predict an onset-aligned hand
trajectory and score correlation. They do not provide the continuous at-risk
onset distribution, source-specific past-peripheral comparator, hidden neural
null bank, frozen near-to-strict rank estimand, or one-shot audit defined here.

A 2026 WAY-EEG-GAL study compared classical, compact convolutional, graph, and
Transformer models for object-property decoding. It found no reliable deep
advantage over the best classical models, a from-scratch Transformer deficit,
and strong EMG dependence. That task used zero-phase preprocessing and
sustained-hold epochs, so its scores cannot enter this leaderboard; its lesson
is to require modality and bandwidth controls and avoid a gratuitous large-model
zoo (<https://doi.org/10.3389/fnins.2026.1874302>). MOABB likewise motivates
dataset–pipeline–evaluation separation but does not supply the continuous onset
evaluator required here (<https://arxiv.org/abs/2404.15319>).

The 2026 EEG/EMG Foundation Challenge now provides sealed evaluation for cued
command decoding and other biosignal shifts. Its BCI track predicts three cued
mental commands across sessions, not spontaneous future movement onset beyond
peripheral history. It is infrastructure precedent, not a substitute task.
See <https://neural-interfaces26.github.io/tracks.html> and the FALCON
infrastructure precedent at <https://github.com/snel-repo/falcon-challenge>.

A positive EP19 result would show only that, under the registered sensors,
detector, horizons, tasks, calibration budget, and model panel, EEG contains
incremental prospective information and that particular model relations do or
do not survive the target change. It would not establish conscious intention,
causal motor preparation, the earliest biological motor command, online BCI
utility, or a universal model ranking.

AJILE12 remains a deferred exploratory ECoG extension. Its public ECoG and
pose-event products do not yet provide the raw filtering and timing provenance
needed for the primary strict-streaming claim.

## Requirements before held-out evaluation

Held-out evaluation remains closed until all of the following are frozen:

- content-addressed WAY and self-paced source manifests plus access terms;
- raw-signal clock, group-delay, resampling, gap, and channel certificates;
- exact sequential onset/stillness code and ambiguous-onset rule;
- the five executable reference recipes, both frozen non-neural baselines, and
  historical-artifact availability;
- complete self-paced development/audit handoffs with evaluator isolation;
- numeric practical, calibration, denominator-qualification, latency, and
  participant-robustness margins signed by the scientist;
- exact parameter/search ranges in a hashed admissible-space manifest,
  calibration events, the complete deterministic C32 calibration algorithm,
  and compute profile;
- append-only adaptive ledger, configuration-lock schema, and evaluator
  procedure.
