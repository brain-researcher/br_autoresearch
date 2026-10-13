# EP19 components — forecasting before sensor-detectable movement

Design selection, 2026-09-30. This note makes the existing onset, input-block,
baseline and calibration comparisons operational. Starting choices remain open
to development-only refinement before scoring; no signal analysis has run.
The 50-ms grid, 300–600-ms primary horizon, source splits, five reference
families, margins and shared trial/resource limits are unchanged.

## 1. Onset and the online risk state

| Source | Selected primary detector | Provider information, not a ready-made EP19 label |
| --- | --- | --- |
| WAY | Earliest independently confirmed crossing among five EMG-envelope paths and backward-derived hand/wrist motion; object/force signals check consistency | Published `HandStart` uses smoothed derivatives; the reported ≤2-ms synchronization figure is not a full detector/acquisition bound |
| Self-paced | Raw wrist-axis departure from a past stationary baseline, vector norm and one-sided change summaries | Published `AccStartIndex` uses zero-phase smoothing and manual review; raw audio/TRIG is needed for causal cue availability |

Sources: [WAY methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC4365902/),
[self-paced acquisition and event methods](https://www.nature.com/articles/s41597-025-06039-9).
The second source supports accelerometer-detectable onset, not an EMG claim.
Provider annotations are agreement checks. Do not copy their thresholds or
move a delayed detection backward by an assumed group delay.

Starting engineering choices are 20-ms trailing sensor summaries, separate
high/low hysteresis thresholds, 40-ms confirmation and 500-ms continuous
stillness for entry. Fix threshold scales and exact coefficients from
development-only stationary sensor noise, never from EEG forecast gains.
Provider agreement diagnoses discrepancies; it does not force these detectors
to reproduce a future-smoothed annotation. These values are starting proposals.

```text
unarmed -> known start cue + past stillness -> at risk
at risk -> first high-threshold crossing -> pending, predictions suspended
pending -> confirmed -> onset labelled at the first crossing; disarmed
pending -> rejected -> requalify through stillness; paused points stay absent
```

One primary reach is armed per start cue. After a confirmed onset, holding the
object still does not rearm a reach: require the end/return phase, a new known
start cue and qualifying past stillness. Return movements are not targets.
Cue information enters only after a raw LED/audio detector makes it available;
an upcoming end cue can censor labels but cannot select the current risk state.
Use one eligibility log on the same 50-ms grid for every comparison.

Require the predeclared primary sensor paths to be valid; missing WAY EMG cannot
silently turn the target into hand-only onset. Gaps, saturation or unresolved
crossings yield unknown/censored status, not invented negatives. Confirmation
changes labels only, not inputs or previously suspended decisions. Keep every
detector path running while predictions are suspended: choose the earliest
accepted crossing timestamp, not the sensor that confirms first. Establish
the existing strictly-below-300-ms worst-case bound from clocks, acquisition,
resampling/filter response, sample discretization and crossing uncertainty.
Record confirmation delay separately. Neither published synchronization nor
the word "raw" verifies the full information-availability chain.

### 1.1 WAY completed-return predicate: parameterized development definition

Specification clarification, 2026-10-05; no detector replay or threshold fit is
selected here. This defines the missing return-state logic independently of
the unresolved acquisition/timing proof. It uses only already available
peripheral samples, never provider event times or EEG results.

The [official methods](https://www.nature.com/articles/sdata201447) identify
position sensors P1 as object, P2 as index finger, P3 as thumb and P4 as wrist.
Their automatic cue route uses digit–object separation of at least 15 cm and
a 1–3-second delay; surface-change trials instead use a researcher button. That
automatic-cue criterion is source context, not proof of return on every trial.
For this development definition, adopt 15 cm as an additional clearance
condition together with the existing 500-ms entry-stillness requirement;
do not claim to reconstruct the provider's cue-generation rule or `HandStop`.

Use XYZ channels `P{axis}{sensor} - position {axis} sensor {sensor}` for sensors
1–4 and each plate's `FX`, `FY`, `FZ` channels, named
`F{AXIS}{plate} - force {axis} plate {plate}` for plates 1–2. These are the
names already selected in the [peripheral reader](code/pilot_way_peripheral_onset.py).
Position comparisons use metres and force comparisons newtons after explicit
source-unit conversion; numerical HS units, force signs/offsets and conversion
evidence remain required configuration, not values inferred from magnitudes.

At a known start cue, the immediately preceding 500 ms must satisfy all of:

1. Both Euclidean digit–object distances, `||P2-P1||` and `||P3-P1||`, are at
   least `0.15 m` throughout that interval.
2. All five existing EMG and all three hand/wrist motion paths satisfy their
   existing low-threshold stillness rule, with complete causal histories.
3. Object position and backward motion are consistent with a supported-object
   reference, and each plate's force vector is consistent with released
   contact. Require each coordinate's absolute departure to be at most its
   configured tolerance; do not cancel opposite plate forces by summing them.

The supported/released references and tolerances must be frozen from permitted
development calibration before use, or established from already elapsed
admitted calibration data. Their values and admission remain unresolved;
quiet hand motion alone cannot supply them. Object/force checks only veto
inconsistent return; they never replace the registered primary onset paths.

After a reach, require an observed end cue before this return interval and a
subsequent new start cue. A break in any condition cancels accumulated return
support. At initial series entry there is no invented previous end cue: the
same fully supported pre-start interval is required. Missing units,
references, cue interpretation or sensor history leave return unknown and
prevent arming; later evidence never restores earlier decision points. Gaps
and series boundaries clear state.

This fixes a parameterized predicate, not its empirical qualification. Cue
interpretation, calibration values and the strictly-below-300-ms combined
timing bound remain unresolved. Development specification is already allowed
by GOAL's pre-score development/freeze rule and SEARCH_POLICY's
`onset_and_risk`; no new approval gate, scientific margin, source subset or
forecasting permission is introduced. No array-only implementation or new
test is selected by this clarification.

## 2. Recent and older-slow EEG: an exact construction

At decision `t`, set `c=t-200 ms`: older samples have `s<c`; recent samples
have `c<=s<t`. For the common 29-channel voltage stream `x`, one development-
fixed one-sided finite filter defines `O(s)=sum_k b_k x(s-k)`, approximating
0.5–8 Hz without shifting its output backward. On older support set
`C=x-O`; on recent support set `R=x`.

Use one shared training-derived voltage scale per channel, with
`mu_C+mu_O=mu_x`. For recent/older-slow indicators `r,o`, construct fixed slots
`Z_ro=[C_scaled+o*O_scaled, r*R_scaled]`. Thus Q11 reconstructs standardized
voltage before family features; no redundant raw-older branch is accessible.
Q00 retains complementary EEG and is **not** the primary zero-EEG baseline.

Mask/reconstruct before every feature and state update. Band-power and
covariance features use slot-local support; shrinkage/tangent references and
zero-slot handling are training-fixed. EEGNet/TCN use causal slot-local
operations; recurrent models replay only permitted inputs. Older raw support
stops at `c`; recent processing begins from a fixed boundary initialization.
No cached full-stream EEG features or state may bypass that boundary. All
conditions keep the same shape, eligibility, adapter, capacity, seeds and fit
allowance; gaps reset state and incomplete required history is unavailable.
Replay computation counts against the existing resources.
Freeze each family adapter before panel fitting, within that family's original
capacity and fitting allowance, without an extra hyperparameter search.
Failed equivalence does not permit expanding the adapter.

Voltage reconstruction does not prove model equivalence: boundary handling
changes feature operators. Keep the existing simultaneous Q11-versus-original
equivalence test at both horizons and both forbidden-block mutation tests.
Mutating constructed O while C stays fixed is a representation intervention,
not a raw-band intervention. C may retain slow information; interpret O's
effect conditional on that fixed complement, not as removal of all slow EEG
or evidence of independent physiological generators.

## 3. A strong source-specific non-neural baseline

| Source | Available past information |
| --- | --- |
| WAY | Raw cue state/elapsed time; EMG envelope/RMS; hand/wrist/object position and backward velocity; grip/load/force; past stillness, session time and availability |
| Self-paced | Causally detected audio/TRIG start/phase/elapsed time; wrist acceleration vector/magnitude and backward differences; four EOG channels; past stillness, session time and availability |

Never use a completed self-paced target/object choice or a future-corrected cue
timestamp. Physically absent sensors are omitted, not treated as quiet zeros;
transient missingness records availability and last-observed age. Start with
a 2-s past sequence and trailing 50/200/1000/2000-ms summaries on the decision
grid. Missing history has the same source-specific handling in every model.

Retain the existing linear discrete-hazard, boosted discrete-hazard and small
causal TCN candidates. Select one per source on development primary-horizon
log score before candidate EEG scores; keep its peripheral/context recipe
common across real, zero and surrogate EEG. WAY weights remain participant-
specific. The self-paced encoder and baseline are group fits on its 15
development people: the second task repeats the protocol, not WAY's weights.
Architecture and history follow-ups remain open inside existing allowances.

## 4. Four-parameter calibration from 32 complete movements

For stable 19-category logits or log probabilities `z`, use
`q_j=softmax(alpha*z_j+b_g(j))`, where `alpha=1/T>0`, the three six-bin horizon
groups have offsets `b_1,b_2,b_3`, and the no-onset category has offset zero.
Each model fits its own four values on the identical chronological whole-
stream prefix ending with the 32nd complete eligible movement. Include natural
at-risk stillness; do not replace the prefix by 32 positive event windows.

Use the existing censoring-aware likelihood: an observed onset contributes
`-log q_j`; complete 900-ms non-onset contributes `-log q_19`; censoring after
`m` complete bins contributes `-log sum_(j=m+1..19) q_j`. Right-censored examples
with `m=0` supply no likelihood information; an observed onset in the first bin
still contributes `-log q_1`. Use stable log-sum-exp throughout.
No outcome beyond the prefix, or from the guard/scored suffix, enters fitting.

Provisional engineering defaults: start `alpha=1,b=0`; minimize mean natural-
prevalence NLL plus `0.01*((alpha-1)^2+sum b_g^2)`, with `alpha` in `[0.25,4]`
and each offset in `[-3,3]`. Use float64 L-BFGS-B, at most 200 iterations,
projected-gradient tolerance `1e-6` and function tolerance `1e-9`. Accept only
a finite, bounded, converged solution whose objective is no worse than identity
by more than `1e-10`; otherwise keep identity and flag the fallback. Do not
drop a person/model, retune or try another calibration from that outcome.

Encoder, spatial projection, feature scaling and per-bin biases stay frozen.
No-label and calibrated conditions score the same later suffix with identical
inference-state initialization. The guard excludes every registered history/
filter-memory overlap with the calibration prefix; its labels never update a
model. These defaults still need development implementation checks and a
pre-score freeze. They are not scientific margins or runtime evidence.

## 5. Finite model recipes and grouped development folds

Prospective preparation, October 6, 2026. The coordinator delegated these
finite choices under the existing model families and development rules. The
[registry](model_recipe_registry.json) and [model/fold interface](code/model_recipes.py)
are implementation inputs, **not a qualified source, fitted backend, complete
method freeze or permission to forecast**. No timing, calibration, P1 offset,
valid-run designation or omitted sensor is supplied by a model default.

### Fixed grouping

WAY uses six whole-series inner folds for every one of P1–P12: validate on one
of series 1–6 and fit on the other five. Fit all scaling, feature references,
baselines and weights inside that fitting fold. Pool all out-of-fold eligible
grid points across the six series within each participant, preserving natural
prevalence, then weight participants equally to choose one shared recipe.
Do not instead equal-weight unequal-duration series. Series 7 remains aggregate feedback, not an inner
fold; final participant-specific refitting on series 1–7 occurs only after the
recipe is frozen. Series 8–9 stay jointly held out. State resets at every
series boundary, and no history crosses a role boundary. This is offline
whole-series cross-validation, not a claim that fitting always precedes its
validation series in calendar time.

Assign the numerically sorted 15 self-paced development IDs round-robin modulo
five, without outcome or support inspection:

| Validation fold | Whole participants |
| --- | --- |
| 0 | sub-02, sub-09, sub-17 |
| 1 | sub-03, sub-11, sub-18 |
| 2 | sub-04, sub-12, sub-20 |
| 3 | sub-06, sub-14, sub-21 |
| 4 | sub-08, sub-15, sub-22 |

Each fold fits on the other 12 people and validates on these three; all their
sessions, runs, calibration prefixes and stream state remain together. The
eight existing held-out people never enter a development fold. Final group
weights use all 15 development people. Unknown eligibility leaves the person
in the original fold with an unresolved result; it does not produce a smaller
successful cohort, replacement or reassignment. The 12-person WAY S1 support
panel remains complete and is not repeated by this preparation.
No partial fold or participant feedback is returned while a required fold or
participant result is unresolved; the existing complete-feedback rule remains.

### Complete reference configuration allowance

For each of the five families, use the Cartesian product of histories
`{1,2} seconds`, regularization `{0.001,0.01,0.1}` and the two variants below:
**exactly 12 configurations per family**, each with seeds 1901, 1902 and 1903.
Do not choose a seed or horizon-specific winner. Average the three normalized
19-category probability vectors before calibration/scoring; the existing
four-parameter calibration acts on that ensemble's stable log probabilities.

The [ensemble adapter](code/ensemble_forecast_backend.py) now connects this
ordering to the fitted backends. It requires all three registered seeds for
one matched condition, with the same recipe, canonical fitting scope, analysis
rate and original/factorial adapter; bandpower members also retain the same
five causal FIRs. Classical fits now retain their already-validated scope for
this interface, without changing fitting rules. Probability averaging reuses
the existing stable ensemble primitive. Saved predictions must have exactly
the supplied prefix's ordered participant/run/decision-time keys. The prefix's
censoring labels and natural-prevalence weights pass to the existing calibration
optimizer and flagged identity fallback. Prefix eligibility, chronological
first-32-event selection and the history guard remain upstream. Calibration
receives no encoder or scaler objects; after an explicit reset, paired no-label
and calibrated suffix predictions share one encoder evaluation. Five focused
integration checks, including actual tiny classical fits, passed in 0.112
seconds in allocation47085635. This does not establish source admission or
empirical forecasting.

The [horizon scorer](code/forecast_scoring.py), prepared October 9, connects
already aligned, post-ensemble predictions to one explicitly named horizon
at a time. It requires separate complete-follow-up coverage from the label
builder, including for observed events; an early event does not waive the
registered requirement to observe status through the horizon's upper edge.
Unknown or partly observed horizons receive no score. Positive and negative
binary probabilities use stable log-sum-exp over the appropriate categories,
without clipping. The score is natural log probability, with larger values
better. All matched conditions retain identical rows and coverage.

Pool eligible rows across series/runs within each participant, then weight
participants equally. Explicit scopes retain all 12 WAY people, all 15
self-paced development people or all eight held-out people, and enforce WAY
series roles. An absent person, absent required control, changed row order or
person with no observed support at the named horizon is unresolved. The
strongest surrogate is the best complete participant mean at that horizon,
not a per-row oracle or a comparator selected at another horizon. Development
uses both registered surrogates; held-out comparisons require all eight.
Five [focused checks](tests/test_forecast_scoring.py) passed in 0.029 seconds
inside the coordinator's Slurm job 47086577. Complete
fold execution, source admission, risk labels, natural-prevalence row coverage,
matching model recipes/adaptation conditions and empirical prediction remain
upstream; this scorer supplies no margins, bootstrap bounds or claim decision.

The separately prepared [factorial-equivalence reducer](code/forecast_uncertainty.py)
uses caller-supplied matched participant means for original/Q11/Q01/Q10/Q00
scores. It retains both held-out sources, all five families, both horizons and
the Q11-original and interaction endpoints: one forty-endpoint family. Each
of 10,000 draws resamples whole people independently within source, with shared
indices across that source's endpoints. A single maximum absolute centered
mean statistic supplies simultaneous two-sided 95% intervals. The prospective
implementation uses the upper empirical quantile (`higher`) and two PCG64
streams spawned in WAY/self-paced order from an explicit caller-frozen seed.

All twelve WAY and eight registered self-paced people are required, with
equal-person weighting; no incomplete person or endpoint is dropped. Margins
must be externally frozen positive values, or remain unresolved. Every relevant
interval must lie strictly within its margin to establish that equivalence
component. Non-containment does not establish non-equivalence, and neither
component alone establishes the complete temporal explanation. Matching of
rows, recipes and adaptation, source admission, mutation checks, decisive
contrasts and replication remain upstream or separate. Independent source
review is complete; all four [focused checks](tests/test_forecast_uncertainty.py)
remain unrun, with no empirical evaluation or new grant.

| Family | Two variants | Fixed representation and readout |
| --- | --- | --- |
| Filter-bank elastic net | L1 fractions 0.1 / 0.9 | Causal bands 0.5–4, 4–8, 8–13, 13–30, 30–45 Hz; per-channel mean squared filtered standardized voltage, then `log1p`; 19-output elastic-net correction |
| Riemannian tangent | Covariance shrinkage 0.1 / 0.5 | Population covariance shrunk toward trace/29 identity, plus `1e-6 I` in standardized voltage units; training-only affine-invariant mean; whitened matrix-log upper triangle with sqrt(2) off-diagonal weights; L2 correction |
| Compact causal EEGNet | 4 / 8 temporal filters, depth multiplier 2 | 128-ms causal temporal convolution, depthwise 29-channel spatial filter, 64-ms causal separable temporal convolution; temporal pools 4 then 8; ELU and dropout 0.25; 19-output linear correction |
| Dilated causal TCN | 16 / 32 channels | Five residual blocks with two kernel-3 causal convolutions, dilations 1/2/4/8/16, ReLU and dropout 0.25; temporal-summary linear correction |
| One-direction GRU | 16 / 32 hidden units | One unidirectional layer over the declared past sequence, dropout 0.25 on its output, 19-output linear correction |

All operations use the common 29-channel order. Fit feature centering/scaling
only on fitting data and freeze it at inference. The tangent mean uses at most
100 iterations and norm tolerance `1e-8`; failure is unresolved, not a fallback
reference chosen from validation scores. The covariance ridge is a declared
numerical SPD construction, not a source noise/calibration estimate. The
elastic-net objective is mean censored NLL plus
`lambda*((1-a)/2*||W||_F^2+a*||W||_1)`; intercepts are unpenalized. Tangent
readout uses `a=0`. Use proximal gradient/backtracking, at most 2,000 iterations
and relative objective tolerance `1e-6`. Nonconvergence remains visible.

The three neural families use AdamW, learning rate `0.001`, the configuration's
regularization as weight decay, batch size 128, gradient norm cap 1 and exactly
100 epochs. No validation-based epoch sweep or extra seed search is implied.
Within-source group fitting gives each participant equal total weight and
preserves natural prevalence within each person; there is no event/class
oversampling. WAY fits remain separate by person. All costs, including nested
fits and replay, stay inside the existing CPU/GPU/wall/scratch limits.

Kernel durations must be represented exactly on the supplied analysis grid.
Analysis sampling rate, anti-alias/resampling response, finite causal band
coefficients, warm-up and boundary state are explicit outstanding preprocessing
inputs; the registry deliberately provides no fake defaults. Required kernels
or histories that cannot be represented remain unresolved. Missing history is
not future-padded. The neural architectures above are precise prospective
recipes. GRU, EEG TCN and compact EEGNet array backends are implemented as
described below. Their 17 selected component checks passed on October 8;
empirical integration and production qualification remain incomplete.

### Baselines and matched comparison interface

Retain all three non-neural families separately in both sources, using the
existing two-second context and 50/200/1000/2000-ms summaries. Logistic hazard
uses 18 interval-specific linear logits and L2 `{0.001,0.01,0.1}`. Boosted hazard
uses 18 conditional logistic heads, depth `{2,3,4}`, 200 trees per head and
learning rate 0.05. The small causal TCN uses the TCN operator above, 16
channels, 18 hazard outputs and weight decay `{0.001,0.01,0.1}`. Its seeds and
training schedule match the declared neural schedule. Hazard probabilities are
`p_j=h_j*product_(k<j)(1-h_k)` and `p_19=product_(k<=18)(1-h_k)`; fit only the
observed complete risk intervals and the observed event, preserving the
existing censoring-aware 19-category likelihood. No unknown interval becomes
a negative. Select one baseline recipe per source before EEG candidate scores
using the registered primary score and tie-breaks.

**Complete development baseline composition, prepared October 9 before outcomes.**
[`development_baseline_driver.py`](code/development_baseline_driver.py) connects
all nine configurations above to the existing fitters, three-seed probability
ensemble, calibration and horizon scorer. One call handles one complete source:
1,944 WAY fits (nine recipes, three seeds, 72 participant/series folds), or
135 self-paced fits (nine recipes, three seeds, five whole-person folds).
Validation people do not multiply the number of self-paced group fits.
WAY pools all six validation-series rows within person before equal-person
selection; it does not average unequal-duration series scores equally.

The self-paced development selection follows the registered primary adaptation:
each validation person's first-32-event calibration, with no-label predictions
as a secondary result on the identical guarded suffix. The caller supplies the
qualified chronological prefix, labels truncated at that prefix endpoint and
the suffix start after the history guard. The library cannot certify that these
represent 32 real eligible movements or authentic clocks. Fitting people retain
their full admitted records; no validation person enters a group fit. All three
seed probabilities are averaged before each person's separate calibration.
Primary log score and tie-break censored NLL use the same adaptation. NLL pools
all scored natural-grid rows within person, including zero-information censored
rows, then weights people equally. Secondary horizons lacking support remain
unresolved; missing primary support prevents any selection.

For the existing parameter-count and latency tie-breaks, prospectively specify
a common counting convention, latency convention and measurement callable.
Immediately after each actual three-seed ensemble is fitted, before validation,
the callable supplies that ensemble's measured count and latency. It receives
the models and fitting scope, not validation rows or labels. Preserve every
fold measurement and use the **separate maxima across all required folds** for
the recipe's count and latency. This conservative operational aggregation is
distinct from equal-person scientific scoring. The prepared
[`baseline_ensemble_cost.py`](code/baseline_ensemble_cost.py) now supplies the
missing callable, using the actual fitted backends rather than recipe-level
size estimates. It counts all three seeds' inference coefficients/offsets and
scaler means/scales. Boosted trees contribute each learned split feature choice,
split threshold and leaf score; fixed topology bookkeeping is excluded. These
counting units are not equivalent statistical degrees of freedom across families.

The caller supplies explicit label-free fitting-fold probes, copied once and
reused in the same order across recipes, and prospectively fixes warmup and
measurement pass counts. Latency is the nearest-rank p95 of individual actual
three-seed predictions plus the existing probability mean. Raw feature
construction, resets, warmup, budget checks, calibration and I/O are excluded;
this does not qualify end-to-end latency. Every completed measurement retains
its per-seed count breakdown and observed durations. The callable checks declared
fold membership and availability but cannot authenticate source timing or probe
origin. Inference state resets even after a measurement failure and before the
driver's prefix/suffix prediction; both adaptation conditions share one suffix
evaluation. The three focused callback checks and changed ASTRA validation
passed in parent 47133348; see the
[execution evidence](experiment_log.md#2026-10-09--non_neural_selection-selects-cost-callback-companion-in-47133348).

The callable returns all recipe summaries and the selected complete OOF panel
only after every required fit and support check succeeds. It has no file reader,
CLI, final refit, fitted-weight publication, EEG-offset construction or access
release. Arrays and three seed models remain resident; classical fits have no
internal budget callback, so production memory and outer runtime admission
remain necessary. Five focused interface fixtures preserve the full registered
scopes without repeating numerical backend qualification. Those five driver
checks and their ASTRA validation passed in parent 47131800, as recorded in the
[execution log](experiment_log.md#2026-10-09--non_neural_selection-selects-prepared-baseline-driver-companion).
The cost callback's separate three checks and changed ASTRA validation also
passed. No empirical selection or method freeze occurred.

**Prospective small-TCN implementation, selected October 7 before outcomes.**
[`temporal_hazard_backend.py`](code/temporal_hazard_backend.py) accepts a
caller-admitted complete regular two-second sequence on **`[t−2,t)`**:
samples `t−2, …, t−1/fs`, with an explicit rate that represents two seconds
exactly. Every prepared row must be available by its sample time; the separate
prepared summary vector may use information available through `t`. Source
clock mapping, units, feature extraction and availability remain upstream
requirements. The backend supplies no rate, resampling or missing-value rule.

Each kernel-3 convolution extends its finite input with `2*dilation` left
zeros and no right padding. This fixed operator boundary applies only outside
an already complete window, never to a raw gap or missing history. It retains
all time positions even when the complete window has fewer than the model's
125-sample receptive field; it does not imply continuous pre-window history.
Each of the five width-16 blocks performs convolution/ReLU/dropout-0.25
twice, adds the same-time residual, then applies ReLU. The first shortcut uses
a 1-by-1 projection when needed. Mean all final hidden positions, concatenate
the separately standardized summary vector, and linearly produce 18 hazards.
No additional normalization layer, boundary indicator or hidden cache is used.

Fit channelwise centering/scaling over fitting windows with per-sample weight
`example_weight / window_samples`; fit summary scaling separately with the
same global example/person weights. Exact constant fitting features retain a
unit divisor. Uniform seeded epoch permutations preserve natural prevalence.
For a minibatch of size `B` from `N` examples, optimize
`(N/B) * sum(global_example_weight * censored_NLL)`, without within-batch
weight renormalization. Keep all observed single-class conditional heads and
report them; any entirely unobserved head leaves the fit unresolved before
network construction. No pseudo-event or probability floor is introduced.
This finite neural schedule does not claim an unconstrained maximum-likelihood
solution for single-class heads; the boosted classifier's separately recorded
single-class refusal remains unchanged.

Use CPU float32 and native Conv1d/Linear initialization under each existing
seed. AdamW settings are learning rate 0.001, betas `(0.9,0.999)`, epsilon
`1e-8`, configuration weight decay on **all parameters including biases**,
and `amsgrad/maximize/foreach/capturable/differentiable/fused=False`.
Run exactly 100 epochs with batches of at most 128, global gradient norm cap
1 and nonfinite-gradient refusal; no early stopping or schedule search.
A separate generator with the same declared seed supplies epoch permutations.
CPU RNG and deterministic-algorithm settings are restored after fitting. The
caller must configure one numerical thread and provide a resource/deadline
check called immediately before every optimizer step. Incomplete or nonfinite
fits cannot supply offsets; completed models are frozen in evaluation mode.

The implementation materializes the full fitting fold's windows. It is an
array-only backend, **not a streaming large-data fitter**: production readers,
memory qualification, final refits and empirical baseline selection remain
unfinished. Five focused checks, including one tiny actual 100-epoch fit,
passed with the boosted comparator checks on October 7 at 05:53 Pacific;
see the [execution evidence](experiment_log.md#2026-10-07-0553-pacific--past_only_controls--boosted-and-temporal-checks-pass).
These component results do not open source access or claim timing/calibration
qualification.

The selected baseline is fit strictly inside each training fold. Its frozen
19 log probabilities enter as offsets, with the family producing 19 additive
EEG correction logits followed by log-softmax. Real, zero and each surrogate
condition fit their EEG encoder/head from the same seed initializations, with
the same baseline offsets, architecture, history, optimization and calibration.
No real-EEG weights warm-start a surrogate. This is a declared fusion recipe,
not an inferred equivalence to a joint context/EEG optimizer. The standalone
baseline remains a separate required comparison, and zero-EEG noninferiority
remains required. No prototype condition replaces that check.

**October 9 final-refit/offset convention, selected before outcomes.** Fit the
already selected baseline recipe on exactly each EEG fitting cohort, freeze
its three seed models, and take the log of their uncalibrated probability
mean. Its predictions on the same fitting rows are training offsets for the
EEG correction; predictions on the untouched validation rows are evaluation
offsets. This does not require a new nested stacking partition. Training
offsets are never presented as OOF performance. Every matched real, zero and
surrogate branch receives identical baseline offsets. Each completed matched
model then follows the existing self-paced32-movement calibration separately;
no baseline-only calibration is applied before fusion. Baseline recipe choices
made on development folds make subsequent development feedback adaptive, not
an unbiased final performance estimate. Held-out WAY series8–9 and the eight
self-paced people retain their distinct final role.

After recipe freeze, final WAY baseline refits use series1–7 for each of all
12 people separately; the final self-paced baseline uses all15 development
people. A separate explicit final-refit fitting scope admits only those
cohorts. Existing development consumers retain their canonical fold checks.
Array-only implementation creates no reader release, empirical fit or
permission to score the final roles.

### Development-fold matched prediction and calibration

The October11 array-only `code/reference_development_prediction.py` composes
the existing fitted fitting-fold baseline and four separately fitted EEG
conditions (`real_EEG`, `zero_EEG`, `shift_1`, `phase_1`). It accepts exactly
one complete canonical development validation fold: one WAY person's held
series, or the three self-paced validation people. It does not construct
surrogates, select a recipe or certify source timing/voltage/prefix validity.

Compute the frozen baseline's uncalibrated three-seed probability mean once
per validation role segment and share its log probabilities across all four
EEG conditions. Each EEG ensemble then averages its own three probabilities.
For each self-paced participant and each completed model separately, fit the
existing four-parameter calibration only on the supplied qualified first32
movement prefix. Prefix and scored suffix inference reset independently;
guard rows are excluded. One suffix prediction supplies both zero-label and
calibrated outputs, preserving the exact suffix identities and natural grid.
WAY remains unadapted. All matched conditions retain the same recipe, grid,
adapter and existing bandpower FIR coefficients where applicable.

Return only internal aligned fold arrays, shared evaluation offsets and
calibration reports with explicit fallback flags. No partial-fold score,
ranking, aggregate feedback or empirical permission follows. Complete-source
scoring still requires every canonical fold/person under `forecast_scoring`;
the caller remains responsible for authenticated prepared rows, genuine
conditions, qualified prefix/guard and unchanged source limits. The prior
fitting/refit interfaces and numerical calibration implementation are unchanged.

### Complete-source development scoring composition

The selected array-only `code/reference_development_panel.py` connects the
complete validation-fold outputs above to the existing `forecast_scoring`.
For one registered baseline/EEG recipe, assemble every canonical fold: all72
WAY participant/held-series folds or all five self-paced folds. Require the
caller's complete expected scored-row identities and labels, exact matched
conditions, and unchanged recipes across folds. Reject missing or duplicate
folds, changed roles, mismatched row coverage or labels before exposing any
aggregate result. Neither a successful subset nor a fold score is feedback.

Reuse the existing scorer's three horizons, natural-prevalence pooling within
people, equal-person weighting and participant-specific strongest surrogate.
Keep the self-paced calibrated and zero-label outputs separate; WAY remains
unadapted. Preserve explicit calibration-fallback evidence. This composition
returns continuous complete-source scores and increments, without selecting a
winner, margin, claim, recipe or final model. It does not construct conditions,
qualify source clocks/labels/prefixes, or grant empirical access. Complete
empirical fitting orchestration remains dependent on those source facts.
The October11,17:12 `reference_panel` log entry records independent source
review, both focused tests and ASTRA validation passing in the closed compute
session47434740. Production memory/runtime remain unmeasured: supplied fold
arrays and the complete assembled panel remain resident together. The caller
owns allocation limits and the budget callback; this component is no empirical
resource qualification or permission to read source outcomes.

### Complete-source fixed-recipe development execution

The array-only `code/reference_development_driver.py` connects the existing
development fitting, prediction/calibration and complete-source scoring calls.
`run_reference_development_recipe` takes one registered baseline/EEG recipe,
all72 WAY or five self-paced fold inputs, and independent expected scored-row
identities and labels. Each fold supplies its already fitted frozen baseline,
the original prepared fitting participants, four genuine condition-window
maps, common preprocessing semantics and validation participants/windows.
The complete fold roster, condition coverage, baseline-offset bindings and
scored suffix labels bind before the first EEG fit. Common fitting rows
construct identical labels and offsets for every condition.

For each fold, fit the four registered development conditions using the
existing three-seed entrypoint, then apply the existing matched prediction and
calibration composition. Release that fold's EEG models before fitting the
next; preserve their fit reports. Only the complete-source scorer returns
feedback. No baseline is refitted, failed seed retried, participant dropped
or partial fold score returned. One recipe requires864 WAY or60 self-paced
EEG fits; those counts do not expand the original search allowance.

The caller still provides authentic conditions, qualified source facts and
prefixes, and the baseline's original fitting labels. Supplied arrays,
baselines and completed fold predictions remain resident; production resources
are unmeasured and classical fitting still requires an outer process limit.
This supplies no reader, surrogate generator, recipe selection, scientific
margin or empirical permission. All three focused integration checks and
changed ASTRA validation passed in the closed CPU step47375558.0; the
October11,23:54 `reference_panel` execution record gives evidence and shared
accounting. The separate pending final-refit checks in parent47373800 retain
their existing status; these integration fixtures do not replace them.

### Final EEG reference refit composition

The October 11 implementation in `code/reference_final_refit.py` supplies
`fit_final_reference_ensemble` for one frozen registered EEG recipe and one
already constructed matched condition. The two classical families, GRU, TCN,
compact EEGNet and their ensemble retain development as the default and require
explicit `fitting_stage="final_refit"` to admit the existing final cohort.
One call uses one WAY person's complete series 1–7 or all 15 self-paced
development people. It is not the complete five-family, participant or
condition panel; all 12 separate WAY fits and every required matched condition
remain necessary in the eventual orchestration.

The caller supplies `FrozenFittingOffsets` from the selected baseline recipe
on the identical fitting cohort. Exact participant, group, boundary and
decision-time identities join every supplied EEG row to one uncalibrated
19-category log-probability vector; existing offsets must agree. Packet order
does not set row correspondence. Training offsets remain explicitly non-OOF,
and source participant weights remain those of the canonical fitting scope.
The composition fits exactly seeds 1901, 1902 and 1903 without retries or
partial-ensemble fallback, returns their actual fit reports, and constructs
the existing frozen probability-mean ensemble. Models and numerical fitting
recipes are unchanged.

The caller still establishes the empirical recipe freeze, genuine matched
input conditions, qualified preprocessing and source/resource admission.
The wrapper adds no reader, calibration, surrogate construction, search,
scoring or CLI. Its budget callback does not interrupt classical fitting;
an outer process limit is also required. Production memory and runtime are
unmeasured. The initial three composition methods, five affected backend
scope/report checks and ASTRA passed in Slurm47360846. A later development-fold
extension shares this same loop through `fit_development_reference_ensemble`:
one five-series WAY fitting fold or twelve self-paced fitting participants,
with exact same-scope training offsets and three seeds. Validation roles,
WAY feedback series7 and final scopes are refused by this entrypoint. Only
the two registered development surrogates are admitted there; factorial
conditions still require the caller's frozen one-per-family recipe. It supplies
no partial fold feedback or recipe-selection result.

Independent source review of the extension found no actionable defect.
The three extended composition methods and changed ASTRA remain unrun as of
October11,05:19Pacific; the unchanged backend checks remain reusable. See the
`reference_panel` experiment-log entries for the separate evidence states.

Development uses the first block-shift and first coherent-phase surrogate;
final evaluation retains all four of each. Exact surrogate construction and
its source-role boundary settings still need internal specification. The
interface retains every mandatory identity rather than inventing those
settings or dropping a difficult surrogate.

**October 9 surrogate-method review; no construction adopted.** The operative
text does not require a finite DFT or a whole-session circular permutation;
exact construction and boundary settings are still prospective choices.
However, causally available input alone does not establish the required
alignment null. A common causal all-pass filter can retain predictive
information, and its finite startup is not an exact finite-record spectrum
preservation guarantee. No such filter is selected as a substitute here.

The [Prichard–Theiler multivariate construction](https://arxiv.org/pdf/comp-gas/9405002)
uses common Fourier phase changes to preserve auto- and cross-correlations
(Section II, equations4–5). For a single centered input matrix, a common
orthogonal time transform preserves its full-window channel covariance.
That observation must not be overextended to EP19's actual implementation:
`classical_forecast_backend.py::_slots` feeds separate older and final200-ms
covariances to the tangent feature model. An operator mixing samples across
that slot boundary need not preserve either slot covariance. Thus a whole-
window coherent phase transformation is not automatically vacuous for this
backend; slot-wise transformations would preserve the corresponding slot
covariances. This is an algebraic implementation inference, not an empirical
result or an accepted null-control recipe.

Any delay-based control must stay within its permitted participant/session/
fold/role segment, with no wrap into future samples or borrowing through gaps.
Any causal filter must reset at the calibration-prefix boundary as well as
the other declared boundaries: a finite guard does not remove exact influence
from an indefinitely carried IIR state. Both calibration conditions must use
the same subsequent initialization and eligible suffix rows. The secondary
search's finite state-age limit also needs an explicit exact rule, not an
assertion that an old tail is negligible.

Delay values, phase construction, seed bank, finite support/reset/warm-up,
common row eligibility and the scientific null remain unspecified. GOAL's
requirement that cue-stratified shuffles and circular-or-block shifts return
to the null is retained. The policy's legacy `circular_and_block_shifted_EEG`
label should be reconciled explicitly with its four session-preserving block
controls before selecting an implementation. No rows, controls, data roles or
scientific thresholds are changed by this review.

Original full references and their factorial adapters use the same two-summary
head, encoder capacity and parameter count. The original can propagate its
causal state through the cutoff; the factorial adapter reconstructs/masks
first and processes older and recent slots with the same family weights and
the fixed slot-boundary initialization. Concatenate the older and recent
summaries into that existing head. No unmasked cache or missing-block flag is
available. The non-neural offsets are identical across Q11/Q01/Q10/Q00. Fix
adapters before fitting; Q11 equivalence at both horizons and both mutation
checks remain necessary. The adapter is unavailable on failure, not expanded.

**GRU implementation selection, October 7 before outcomes.**
[`gru_forecast_backend.py`](code/gru_forecast_backend.py) reuses the existing
development-fold `FittingExample`/`PastWindow` interface, explicit analysis rate
and common 29-channel order. Supplied windows already have the common
fitting-only voltage transform and any mask/reconstruction applied. The GRU
does not refit a condition-specific scaler; masked zeros stay zero. Source
clock, acquisition, resampling and window availability remain upstream inputs.

Each slot contributes its last hidden state. Initialize to zero at each
complete window; original mode carries older state through the cutoff, while
the factorial adapter initializes the recent slot to zero. Both use the same
one-layer 16/32-unit GRU and concatenate the two states, apply dropout 0.25,
then a 19-output linear correction. These are selected implementation constants,
not additional hyperparameters. No hidden cache persists across windows.
Native CPU float32 GRU/Linear initialization, dropout and epoch permutations
use each declared seed; CPU RNG/deterministic settings are restored afterward.
The fixed 100-epoch AdamW settings, bias-inclusive weight decay, batch size,
gradient cap and immediately-before-step budget callback match the small-TCN
implementation above. Baseline log offsets and the existing censored likelihood
are accumulated in float64; only the EEG parameters are optimized. Uniform
epoch permutations use global participant weights without minibatch
renormalization. A wholly uninformative censored fold remains unresolved.

The backend materializes the fitting windows and is not a production streaming
reader. Six focused checks, including one tiny actual 100-epoch fit, are
defined in [`test_gru_forecast_backend.py`](tests/test_gru_forecast_backend.py)
and passed in the October 8 combined 17-check run; see the
[execution evidence](experiment_log.md#2026-10-08-2332-pacific--past_only_controls--all-seventeen-eeg-backend-checks-and-astra-pass).
No source access, empirical forecast, full-reference equivalence,
scientific margin or stage activation is established by this implementation.

**EEG TCN implementation selected October 7; documentation reconciled October 8.**
[`tcn_forecast_backend.py`](code/tcn_forecast_backend.py) implements the existing
16/32-channel family on the same canonical development `FittingScope`, complete
`PastWindow`, explicit sample-rate and 29-channel interfaces. It consumes the
shared fitting-only voltage transform after caller reconstruction/masking;
it supplies no condition-specific scaler, source clock or missing-value rule.

Five residual blocks use dilations 1, 2, 4, 8 and 16. Each block applies two
kernel-3 convolution/ReLU/dropout-0.25 operations, adds a same-time shortcut,
then applies ReLU. The first shortcut projects 29 channels to the registered
width with a learned 1-by-1 convolution; later shortcuts are identities.
Each convolution has exactly `2*dilation` fixed left zeros, no right padding,
and no normalization or weight normalization. The resulting receptive field
is 125 samples. This boundary applies outside a supplied complete finite
window or factorial slot; it neither imputes missing history nor imposes a
125-sample minimum on otherwise complete windows.

Original mode encodes the complete window before averaging final hidden
positions separately before and within the recent 200-ms slot. Factorial mode
encodes each reconstructed/masked slot separately with the same weights and
fresh fixed left boundaries. Concatenate the two means into the shared
19-output linear correction, with no extra head dropout or boundary indicator.
There is no persistent convolutional cache. Identical encoder/head capacity
does not establish the required Q11 equivalence or empirical mutation checks.

The backend reuses the GRU's fixed baseline log offsets, censored-category loss
and global participant-weight minibatch objective; it does not replace or
modify GRU training. Native CPU float32 Conv1d/Linear initialization and the
fixed 100-epoch AdamW schedule, three seeds, bias-inclusive weight decay,
batch size and gradient cap match the settings above. Offsets and likelihood
arithmetic are float64. The allocation/deadline callback runs immediately
before each optimizer step; CPU RNG and deterministic settings are restored.
The caller supplies one numerical thread. Completed parameters are frozen and
prediction requires an explicit reset boundary and increasing decision times.

Five [focused checks](tests/test_tcn_forecast_backend.py) cover the finite
causal operator, original/factorial summaries and masked invariance, exact
recipes and complete windows, unchanged offsets/person weights, and a tiny
100-epoch fit with budget refusal and frozen inference. **All five passed**
in the October 8 combined 17-check run, as did the changed ASTRA validation;
see the [execution evidence](experiment_log.md#2026-10-08-2332-pacific--past_only_controls--all-seventeen-eeg-backend-checks-and-astra-pass).
The backend materializes fitting windows; production readers/resources and
empirical family/control integration remain unfinished. These prepared checks
open no source or held-out access and provide no forecast result.

**Compact causal EEGNet implementation selected October 8 before outcomes.**
[`eegnet_forecast_backend.py`](code/eegnet_forecast_backend.py) uses the same
complete `PastWindow`, canonical development scope, common 29-channel order,
shared fitting-only voltage transform and fixed baseline offsets. The caller
supplies the analysis rate; the backend adds no resampling, condition-specific
scaler, source calibration, mask or context feature.

The selected order is a linear causal temporal convolution, a depthwise
29-channel spatial convolution with depth multiplier 2, ELU, causal average
pooling of width/stride 4 and dropout 0.25; then a depthwise temporal convolution
and linear pointwise convolution, ELU, causal average pooling of width/stride 8
and dropout 0.25. The temporal-filter count is the registered 4 or 8. The first
kernel is exactly 128 ms on the raw analysis grid; the separable temporal kernel
is exactly 64 ms on the **post-pool4 grid**. At 500 Hz these are respectively
64 raw samples and 8 coarse samples. Unsupported kernel or history grids are
rejected, without rounding or choosing another source rate.

All convolutions retain native CPU float32 initialization and enabled biases.
There is no normalization, weight constraint or additional head dropout.
Convolution and averaging operators extend only to the left with fixed zeros
outside the complete window/slot. Boundary averages divide by the full width,
including those zeros. For a full window of `N` raw samples, retained right
edges are `N-1-4k` after pool4 and `N-1-32k` after pool8, for in-window positions.
These phases remain anchored to the **full window** when the factorial adapter
resets every operator at the older/recent-200-ms cutoff; neither grid is
reanchored to an individual slot's end. Each slot must retain an encoded
position on the fixed grid.

Original mode encodes the complete window, then averages encoded positions
according to whether their right edge precedes or follows the cutoff.
Factorial mode separately encodes the reconstructed/masked slots with shared
weights and the same endpoint grids. Both concatenate two means into the same
linear 19-category correction head. No persistent state, missing-block flag
or forbidden pre-cutoff context enters the reset recent slot. Shared capacity
and component mask invariance do not establish empirical Q11 equivalence.

The fixed 100-epoch AdamW schedule, three seeds, global participant weights,
unchanged float64 baseline offsets/censored likelihood, immediate pre-step
budget check and frozen inference match the GRU/TCN settings above. Full fitting
windows are materialized, leaving production-memory qualification separate.
Independent source review found no blocking mismatch with these adopted
choices. Six [focused checks](tests/test_eegnet_forecast_backend.py), including
an independent finite-sum operator comparison and a tiny actual 100-epoch fit,
**passed** in the October 8 combined 17-check run. Changed ASTRA validation
also passed; see the [execution evidence](experiment_log.md#2026-10-08-2332-pacific--past_only_controls--all-seventeen-eeg-backend-checks-and-astra-pass).
No empirical forecast, source admission or held-out access follows.

`reference_jobs` enumerates 720 initial development identities
`(5*12 configurations)*3 seeds*(real,zero,two surrogates)`, before multiplying
by required source/fold fits. Final primary and factorial enumeration requires
**explicit one-per-family development-frozen IDs**: 150 and 60 identities,
respectively. These are job identities, not 720 new scientific search trials
or a launch request. Final/factorial stages never silently repeat the 12-way
configuration search. Preserve seed-level artifacts and report uncertainty at
the participant level, not at the seed/job level.

### Finite secondary changes and remaining work

The registry supplies three settings per required classical, temporal,
representation and fusion branch for the existing 12-trial initial coverage.
Classical changes its L2 linear readout; temporal changes the listed dilation
schedule; representation applies fitting-only PCA at rank 8/16/24; fusion uses
additive logits, linear concatenation or a width-16 gate over concatenated
EEG/context summaries. Context summaries must retain the identical permitted
inputs, training role and natural-prevalence objective. Follow-ups change one
recorded operator relative to their parent. Montage dropout and diagonal-state
settings remain conditional branches triggered only by a recorded development
limitation, never by held-out results. Every instantiated proposal must retain
the existing two-million-parameter, two-second-input/eight-second-state caps,
direct controls and 24–40-trial search requirements. A rank unsupported by its
fitting data is unresolved, not replaced by a new rank. The secondary winner
cannot join the five-family reference panel.

No source timing/units, P1 correction, onset detector, unknown valid-run status,
scientific margin or warning rule is selected here. Actual family backends,
the causal preprocessing coefficients and surrogate generator remain internal
implementation work. Their incompleteness is distinct from the missing
acquisition/calibration facts in `source_timing_handoff.md`. The existing
likelihood, four-parameter calibration and factorization primitives are reused;
their already-passing checks are not rerun by this preparation.

Implementation update, October 6: the two classical reference wrappers and
the source-specific logistic-hazard wrapper are prepared in
`code/classical_forecast_backend.py` and `code/non_neural_forecast_backend.py`.
The latter takes explicitly prepared non-neural vectors and preserves their
feature order, declared availability, participant identity and reset boundary;
it supplies frozen 19-category baseline probabilities. This does not implement
raw context-feature extraction or verify the source history used to make a
vector. Both wrappers cover development inner fitting folds only. Subsequent
boosted and non-neural temporal backends passed their selected component
checks; GRU, EEG TCN and compact EEGNet also passed their 17 selected component
checks as recorded above. The explicit baseline final-refit path was added
later, and the October 11 EEG final-refit composition is described above.
Empirical integration remains unfinished; see the experiment log for current
evidence and pending compute checks.
