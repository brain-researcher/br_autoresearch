# Can earlier recording days help predict neural activity on a new day?

Take the M1 part of the study. Imagine two rightward reaches made on a new
recording day. At the same moment after movement onset, the usual
direction-and-time response is the same for both trials. Yet their LFPs and
population spike counts may rise or fall together in different ways. Earlier
recording days contain many examples of these signals, but the recorded
neurons, electrode signals, and numeric scale have changed by the time the new
day begins. The same design is applied separately to PMd activity before
movement.

EP07 asks whether those earlier days are still useful after the new day has
provided its own small calibration set. Three earlier days from the same
animal, implant, and cortical region are used to help predict population spike
counts on untouched trials from a later day. The later day contributes only a
nominal 20% of its trials for calibration.

The first comparison is deliberately demanding. The source-day model must
beat both a direction-by-time average and an equally tuned model that receives
the same new-day calibration trials but no earlier-day data. Beating only the
average would show that neural signals help within a day; it would not show
that historical recordings add anything once today's calibration data are
available.

Even a win over both comparisons may have a simple explanation. More old data
could improve the average response for each reach direction, stabilize the
fit simply by adding more examples, or align the day's average population
pattern. None of those results shows that a trial-to-trial relationship
between LFP and population spiking survived across days. The first score must
therefore lead to a second question:

> After the usual response for reach direction and time is removed, can a
> relationship learned on earlier days still use the LFP to predict which
> new-day trial has more or less population spiking?

The intended paper should show more than a higher cross-day prediction score.
It should identify whether the useful part of an earlier day is an average
reach template, an aligned population pattern, or a relationship between LFP
and spiking on the same trial. It should also show whether the answer repeats
across target days and animals, whether it survives without 100--400 Hz
spike-rich features, and whether a rule fixed in advance can predict when
earlier days will help on a new session whose held-out outcomes have not been
examined.

Even a positive result would not show that the same neurons or electrode
weights remain stable, that LFP causes population spiking, or that an online
BCI will improve. It would support a narrower claim: in these two animals and
six fixed target days, earlier days add useful information after limited
same-day calibration for whichever prespecified setting the evidence supports—
M1 during movement, PMd before movement, or both.

The [paper plan](outputs/paper_plan.md) gives the follow-up tests, closest prior
work, result-dependent branches, and figure sequence. The primary transfer
test remains the first result. Later mechanism analyses cannot turn a failed
or ambiguous transfer result into a positive one.

## At a glance

| Question | EP07 design |
| --- | --- |
| What changes from day to day? | The recorded neurons, electrode signals, and signal scale can change even when the animal, implant, region, and reaching task stay the same. |
| What information is available? | Three earlier recording days plus a nominal 20% of the later day's trials for calibration. Neuron and numeric channel identities are not matched across days. |
| What is predicted? | Population spike counts on untouched later-day trials, separately for M1 during movement and PMd before movement. |
| What is the hard comparison? | The model using earlier days must beat both a direction-and-time average and an equally tuned model that uses the same later-day calibration trials but no earlier days. |
| What must repeat? | The improvement must survive across the six fixed later days and be supported in both animals, rather than depend on one favorable recording day. |
| What can the first test conclude? | Whether historical recordings add predictive information after limited same-day calibration. It cannot yet say what information transferred. |
| What would make a deeper finding? | Evidence that the transferable signal is a trial-specific LFP--population relationship, rather than only an average reach pattern, the stabilizing effect of extra training examples, or high-frequency spike contamination. |
| What is the next test? | Remove the direction-and-time average before fitting, then ask whether earlier days still improve prediction and which LFP and population components carry that gain. |
| What is the external prediction? | Before held-out neural outcomes are examined, use the permitted calibration data to predict whether earlier days will help on a new session and which population dimensions will benefit. |

## From a transfer gain to a scientific finding

Consider two reaches to the same target at the same time after movement onset.
Their mean response is identical by construction, but their LFP and population
spiking may differ from trial to trial. A source-bearing model could win because
old sessions estimate the direction-by-time mean more accurately. Alternatively,
it could win because old sessions teach a relationship that predicts which of
the two trials has higher or lower population activity. Those are different
findings and require different tests.

After the primary conclusion is fixed, the proposed follow-up creates a true
residual-only prediction task; it does not merely subtract the same mean from a
finished prediction and its target. Direction-by-time means are estimated from
permitted training/calibration trials. Those means are removed from both LFP
features and spike-count targets before either the source-bearing or matched
calibration-only residual model is fit. The residual models cannot see
uncentered neural values, a source-day mean template, or held-out outcomes.
Their predictions are scored directly against held-out spike residuals, while
the mean component is scored separately. It then asks which source-trained
population dimensions and LFP feature blocks carry reproducible residual gain.

The mechanism development set has already been outcome-exposed. Candidate
components, ties, margins, and ablations must therefore be frozen before any
new mechanism-specific score or diagnostic is computed or revealed—not
pretended to be frozen before the data were ever accessed. The consumed primary
audit can contribute only generic secondary outputs declared before its
single opening; it cannot select this mechanism. A visually appealing latent
trajectory is not enough.

A trial-specific field-potential interpretation additionally requires an
LMP/low-frequency-only result, explicit exclusion of 100--400 Hz power,
same-electrode/unit-intersection analysis, and spike-quality matching. If the
gain exists only in high-frequency or spike-rich features, report that boundary
rather than calling it general LFP--spike coupling.

The follow-up also has to make a forward prediction. Using only source days and
the target calibration set, it should estimate whether source information will
help that target day and which population dimensions will benefit. That rule is
trained and cross-checked on development days, frozen, and then tested on newly
sequestered sessions. Reusing the current audit trials after seeing their result
would be explanation of this corpus, not external validation.

The possible paper-level outcomes are deliberately different:

| Follow-up result | What it would mean | What not to claim |
| --- | --- | --- |
| Source gain remains after direction-by-time means are removed, localizes to prespecified LFP/population components, and predicts new sessions | A trial-specific LFP--population relationship is reusable across days within the tested scope | The same neural weights, electrodes, or single neurons are stable |
| Source gain is confined to the mean reach trajectory or aligned low-dimensional geometry | Earlier days reuse stable task structure, but evidence for transferable trial-specific coupling is absent | A general moment-to-moment neural mechanism |
| Source gain is explained by generic shrinkage, target calibration, one source day, or one target day | The proposed stable relationship is not supported; retain only the narrower result that survives the controls | A robust cross-day principle |
| No primary gain, or the result is imprecise | Stop or report the bounded negative/unresolved result | A mechanism rescued by post hoc component analyses |

## Authority and history boundary

This directory is the current local scientific contract. The contract alone
does not start compute or establish a finding; an explicit scientist
instruction in a Codex task starts the work. Earlier uses of this Dryad corpus
remain development history: no earlier score, winning setting, review, or
conclusion may seed the trial order or prior. At episode bootstrap, bind the
search policy, exposure record, input identities, and exact contract-byte
snapshot before any candidate-discriminating score is used.

EP05--EP08 share source dependence and therefore use one exposure ledger. A
new trial split inside this already exposed release is an internal audit of a
newly locked procedure, not independent replication.

## Exact unit of transfer

Within a fixed animal, implant, and cortical region, can a single bounded
policy use earlier recording days to improve prediction of trial-specific
target-day population activity after seeing only a nominal 20% target calibration
trials, by more than both a direction-by-time mean and an equally tuned model
that receives target calibration but no source-day information?

The selected configuration is one global policy shared by every primary
target day and both prespecified region/epoch strata. Its fitted parameters
remain animal-, implant-, and region-specific: neural observations are never
pooled or transferred across animals, implants, or regions. The search may not
select a different configuration for a session, animal, or stratum.

## Frozen forward day roles

Recording-day roles are global and chronological:

| Animal | `source_only` days | `target_only` days |
| --- | --- | --- |
| Mihili | 2014-02-17, 2014-02-18, 2014-03-03 | 2014-03-04, 2014-03-06, 2014-03-07 |
| Chewie-L | 2016-09-29, 2016-10-05, 2016-10-06 | 2016-10-07, 2016-10-14, 2016-10-21 |

M1 and PMd records from the same day always have the same role. A trial from
any `target_only` day is forbidden from every source fit, including when a
different target day is being scored. Target calibration may adapt a source
fit only through the declared target-adaptation operator; it does not convert
the target day into a source day. No day may change role or replace a named
day after any neural value or score is opened.

All six named days for each primary animal must be structurally eligible. Each
target day must contain the required M1 and PMd data, all eight authenticated
reach directions, complete event timing, support for the structural-only
target-neuron roster defined below, and at least 15 complete trials in every
direction. If either animal
lacks this support, stop for redesign before outcomes or scores are exposed; do not
silently weaken the threshold or substitute a day. Chewie-R is excluded from
the primary estimand and does not count as a third animal. Any later
Chewie-R analysis is explicitly labeled an implant sensitivity and cannot
choose, rescue, or promote a policy.

## Frozen windows and released-representation exception

Use the provider's native 30-ms bins without interpolation, rebinning, or
overlapping summaries:

- M1 execution: `[movement onset + 150 ms, movement onset + 450 ms)`, exactly
  10 native bins; and
- PMd preparation: `[go cue - 450 ms, go cue)`, exactly 15 native bins.

The half-open intervals above are the two prespecified strata. Region and task
epoch are confounded, so even a result in both strata is not evidence of a
general preparation-versus-execution effect.

The public payload contains provider-preprocessed LFP representations, not raw
voltage. Its session-wide centering and zero-phase filtering/smoothing cannot
be reconstructed separately within role. This is one frozen,
target-label-blind exception: the authenticated released arrays are used
unchanged and identically by the transfer stream and both comparators. No
candidate may tune or repeat that preprocessing. The exception must be
recorded in the lock bundle, and every claim is limited to this released,
offline representation; this is not a raw-signal or strictly inductive
benchmark.

## Frozen response, eligibility, and neuron roster

The scored response `y` is the provider-released native-bin spike-count array,
unchanged. A fit may center or project targets only inside a declared model
using source-only and target-calibration observations; every prediction must be
mapped back to the original target neuron's native count scale before scoring.
No square-root, log, smoothing, clipping, variance normalization, or other
response transform may change the primary scoring target.

Before role assignment, the episode's deterministic builder applies one fixed
validity rule to whole behavioral trials. `result` must equal `R`; `bin_size`
must equal 0.03 s;
animal, implant, day, direction, and simultaneous M1/PMd identities must join;
the required event indices must be finite integers; both frozen windows must
fit their native time axes; spike/LFP matrices and guides must have the declared
two-dimensional shapes; and every required LFP and spike value in both windows
must be finite. Spike counts must also be nonnegative integers. If either
stratum fails, the whole simultaneous trial is ineligible. This predicate may
inspect identities, structure, and validity, never variability, magnitude,
model residuals, or candidate/comparator scores.

Map `tgtDir` to labels 0--7 by the nearest circular multiple of `pi/4`, with
maximum absolute circular error `0.01` radians. Convert each provider MATLAB
event index from one-based to zero-based exactly once. If `m` and `g` are the
resulting movement and go-cue indices, the Python time-axis slices are
`M1[m+5:m+15]` and `PMd[g-15:g]`; each must contain exactly 10 and 15 bins.

For each target day and region, the target-neuron roster is every authenticated
unit-guide entry whose identity and ordering are identical across all eligible
trials and whose spike row is present in every required native window. Roster
construction reads unit-guide metadata and shapes only; it cannot use any
calibration, development, or audit spike value. A guide mismatch makes the day
structurally ineligible rather than inducing an intersection or outcome-based
neuron subset. The roster is ordered lexicographically by the canonical
unit-guide tuple, frozen before target-trial hashing, and shared by every model.
Individual low- or zero-variance neurons are never dropped. Only a zero joint
`R2_SSE` denominator for the full frozen roster is a technical failure.

## Deterministic target-trial roles

For each of the six `target_only` days and each reach direction, order complete
whole trials by ascending SHA-256 of the canonical-JSON UTF-8 array
`["ep07-target-role-split-v1", animal_id, session_date, direction_id,
authenticated_trial_id]`. Resolve a digest tie by the authenticated trial
identifier. For `n` eligible trials assign, in that order,

```text
n_calibration = floor(0.2*n + 0.5)
n_development = floor((n - n_calibration)/2)
n_audit       = n - n_calibration - n_development
```

The minimum `n = 15` therefore supplies at least 3/6/6
calibration/development/audit trials. All bins, neurons, input features, and
predictions belonging to a whole trial inherit its role. When M1 and PMd share
a behavioral trial, they inherit the same role. No resampling, seed search,
backfilling, exclusion based on neural values, or role-specific neuron
selection is allowed.

- **Calibration:** candidate-visible. Both model streams may fit target
  scaling, target latent alignment, residual means, and declared adaptation.
- **Adaptive development:** the episode evaluator keeps labels and
  candidate-discriminating diagnostics outside candidate fitting code and
  returns only the authorized aggregate feedback.
- **Locked audit:** trial identifiers, covariates, outcomes, predictions joined
  to outcomes, metrics, and candidate-discriminating diagnostics remain sealed
  until one policy, comparators, inference code, and terminal rule are locked.
  It opens once.

## Bounded scientific grammar

Each configuration is a declarative composition of only these operators:

1. **LFP representation:** authenticated LMP plus the eight stored band-power
   classes at native 30-ms resolution; electrode aggregation by moments and
   quantiles, robust histograms, or a low-rank permutation-invariant set
   encoder fitted on source days plus target calibration only. Numeric channel
   identity is never matched across days.
2. **Latent construction and alignment:** spike PCA rank in `{4, 8, 12, 16}`
   subject to structural support; optional training-only whitening;
   behavior-anchored orthogonal Procrustes, regularized CCA, or no rotation.
   Rank, shrinkage, eigenvalue floors, and reference construction are bounded
   configuration fields.
3. **Nuisance handling:** direction-by-time residualization, additive
   direction/time covariates, or their frozen combination. All nuisance
   quantities are fitted on source days or target calibration only.
4. **Source pooling:** equal-session pooling, reliability-weighted pooling,
   target-calibration similarity weighting, hierarchical coefficient
   shrinkage, or leave-one-source stacking. Fitted pooling weights may use only
   source observations and target calibration; authorized development feedback
   may select among already declared configurations but may not fit a hidden
   meta-weight.
5. **Mapping and adaptation:** ridge, elastic net, reduced-rank ridge, or a
   shallow two-layer MLP with width at most 64; target update as zero-prior
   fit, source-prior shrinkage, or convex source/target interpolation.
6. **Ensembling:** at most two already evaluated, diversity-qualified
   candidates and one weight from the frozen finite grid. The full ensemble is
   declared before scoring and consumes one ordinary successor evaluation.

The grammar excludes raw-waveform networks, non-native temporal summaries,
arbitrary feature code, cross-animal or cross-region inputs, numeric channel
matching, per-session or per-stratum winner selection, evaluation-fitted
preprocessing, and any representation selected from audit outcomes.

The full family set above is retained by scientist decision. Its equation-level
operator definitions are frozen in `NUMERIC_OPERATOR_GRAMMAR.md`; the finite
grid, 16 anchors per stream, legal-combination rules, seeds, and runtime are in
`ANCHOR_RUNTIME_CONTRACT.yaml`; and selection, budget, falsifier, retry, and
terminal semantics are in `NUMERIC_SEARCH_CONTRACT.yaml`. These three files are
one contract bundle: their content hashes must be locked together before any
development outcome is opened. Retaining the full grammar does not waive a
shape, legality, determinism, or leakage check.

## Comparators and exact estimand

Freeze two comparators for every target day and stratum:

1. `direction_time_mean`: the direction-by-native-time-bin mean fitted only
   from that day's calibration trials; and
2. `calibration_only_champion`: a model fitted only from that day's
   calibration trials, selected from the same legal representation, nuisance,
   mapping, target-adaptation, and capacity families as the transfer stream
   wherever those families remain meaningful without a source fit. It receives
   the exactly matched development-feedback budget below and no source-day
   observations or fitted source state.

Let `k` be M1 or PMd, `s` one of the six fixed target days, and `b` one of the
two comparators. For model `m`, compute one session-level score

```text
R2[s,k,m] = 1 -
  sum_(t,q,j) (y[s,k,t,q,j] - yhat[m,s,k,t,q,j])^2
  / sum_(t,q,j) (y[s,k,t,q,j] - ybar[s,k,j])^2
```

where `t`, `q`, and `j` index all eligible evaluation trials, native bins, and
the frozen target neurons, and `ybar[s,k,j]` is neuron `j`'s mean across those
evaluation trials and bins. This is one joint `R2_SSE`, not an average of
trial-, direction-, bin-, or neuron-level scores. Preserve negative values; a
zero joint denominator is a technical failure and cannot trigger post-hoc
neuron or day exclusion. Trials, directions, bins, neurons, latent dimensions,
seeds, folds, and calibration repeats are not independent audit units. Define

```text
Delta[s,k,b] = R2_transfer[s,k] - R2_comparator[b,s,k]

theta[k,b] = (1/2) * sum over animals a of
             ((1/3) * sum over the three target days s of animal a
                      Delta[s,k,b])

theta[k] = min over comparators b of theta[k,b]
```

The comparator axis is never collapsed by subtracting the best comparator
separately in each session. All estimates, uncertainty, sign checks, and
influence checks are computed and reported for each `b`; `theta[k]` is only
the conjunction summary. Development uses the same day- and animal-balanced
construction on development trials. Its primary policy-selection score is the
equal-stratum mean of the two development `theta[k]` values; session lower
tails, within-animal effects, retention relative to a same-split
within-session ceiling, and simplicity are report-only diagnostics except where
the frozen finalist tie order explicitly names a quantity. They do not define
an additional Pareto eligibility rule. During successor proposal, before the calibration-only champion is
known, the transfer stream instead receives only its equal-stratum increment
over the fixed direction/time mean; the calibration-only stream receives its
own equal-stratum absolute R2. After the calibration-only champion freezes,
all stored transfer predictions are deterministically rescored against both
comparators for the one final selection. Thus patience has no circular
dependence on a future champion.

## Matched adaptive-search budget

The transfer and calibration-only streams receive exactly matched tuning
opportunities:

1. evaluate 16 prespecified, diverse anchor pairs;
2. attempt `A` feedback-dependent successor pairs, where `2 <= A <= 14`, with
   one new transfer configuration and one new calibration-only configuration
   per atomic pair and at least two valid pairs required;
3. select and freeze the calibration-only champion first, then select one
   feasible source-bearing transfer finalist against that fixed champion and
   the fixed direction/time mean; and
4. run exactly 10 additional paired trials, each bound to those two frozen
   finalists and to one of the prespecified falsifier transforms `F01`--`F10`.

Only the `16 + A` anchors and successors in each stream are selectable. The 10
falsifier trials per stream are diagnostic and never enter either archive.
Therefore each stream receives `16 + A + 10 = 28..40` outcome-contacting
trials and the two streams jointly receive
`2 * (16 + A + 10) = 56..80`. The falsifier share after anchor coverage is
`10 / (A + 10)`, at least `10/24 = 41.67%`. The fixed
`direction_time_mean` consumes no tuning opportunity.

A pair is scientifically valid only after both streams return valid authorized
development evaluations; unilateral feedback is not released. Every attempted
pair still consumes two outcome contacts. Exact infrastructure retries repeat
the same hash and seed, emit no scientific feedback, do not consume patience,
and are capped at two per trial and 12 total. A candidate-specific numeric
failure after outcome contact consumes the atomic opportunity for both streams
but is not a valid selectable pair. No more than 14 successor pairs or 80 total
stream outcome contacts may be attempted. Failure to obtain two valid
successor pairs or to complete any required phase yields `incomplete_search`
unless an integrity violation requires `technical_failure`.

After two valid successor pairs, stop adaptive proposal on the first applicable
event:

- 14 paired successor opportunities have been attempted;
- the finite admissible configuration space is exhausted;
- seven consecutive valid successor pairs add less than `0.002 R2` to both
  streams' prior best eligible primary scores; or
- a declared resource or technical stop occurs.

Stopping successor proposal always leads to frozen finalist selection and all
10 falsifier pairs if the required search completed and a feasible finalist
exists. Falsifier trials neither reset nor advance patience. A failed
scientific falsifier makes the single finalist ineligible; it does not permit
runner-up substitution.

The `0.002 R2` value is only search resolution/patience; no falsifier,
promotion, or audit decision uses it. It is not the scientific audit margin.
A resource stop, authorized interruption, or failure
to finish the 16 anchors per stream and required coverage produces
`incomplete_search`; it never licenses audit access or a scientific negative.

Resource envelope: CPU only; at most 800 aggregate CPU-hours, 96 wall-clock
hours, 32 concurrent cores, and 500 GB scratch.

## Required falsifiers and robustness work

Before lock, every promotion-eligible policy must undergo:

- `F01` source-day-label shuffle;
- `F02` source-trial-correspondence shuffle;
- `F03` target-calibration-correspondence shuffle;
- `F04` zero-source ablation;
- `F05` direction/time-template-only reduction;
- `F06` equal-source-weight replacement;
- `F07`--`F09` omission of source-day positions 1, 2, and 3; and
- `F10` the first applicable active-component ablation in the frozen priority
  order (alignment, whitening, feature scope, aggregation, nuisance, then
  source-coordinate permutation).

Each of these is one atomic paired trial on the frozen transfer finalist and
calibration-only champion; applicability and pass mode are derived from their
operator graphs before scores open. Stable, drifting, null, and
latent-scale-shift synthetic recovery; deterministic full-refit replay; and
role, cache, link, and network-denial proofs are required nonbudgeted integrity
gates. If either finalist is a two-parent ensemble, both parent-removal scores
are recovered as two prespecified records from the parents' already stored
predictions; this adds no outcome contact and satisfies the component-ablation
requirement without bundling F10. Leave-one-target-day analysis is a no-refit recomputation from stored
locked audit predictions, not another search or falsifier trial.

A failed scientific negative control or robustness test makes a policy
ineligible. A failed access, split, identity, or leakage-integrity proof makes
the episode a technical failure.

## Lock and one-shot inference

Select one global transfer policy and one calibration-only champion using only
authorized development feedback. Re-run both from immutable inputs, then hash
a lock bundle containing the source and role manifests, structural support and
neuron rosters, provider-preprocessing exception, grammar, complete append-only
paired trial ledger, Pareto archive, fitted-state recipes, fixed comparator,
chosen configurations, predictions recipe, metric and bootstrap code,
multiplicity rule, falsifiers, environment, seeds, stopping event, and terminal
decision table. The bundle must include exact content hashes for
`NUMERIC_OPERATOR_GRAMMAR.md`, `ANCHOR_RUNTIME_CONTRACT.yaml`, and
`NUMERIC_SEARCH_CONTRACT.yaml`. No configuration, fit rule, exclusion, or
inference change is allowed after the lock.

For each `theta[k,b]`, test the practical-margin null
`H0: theta[k,b] <= 0.01 R2` with 9,999 prespecified paired-bootstrap
replicates generated by NumPy `PCG64DXSM` from unsigned 64-bit seed `7007999`.
Within every target day and direction, resample whole audit trials with
replacement, drawing the original number of audit trials in that cell. Use
the same sampled trial indices for transfer and both comparators and, for a
simultaneous behavioral trial, across M1 and PMd; keep every bin, neuron, and
prediction from a sampled trial together; refit nothing. Recompute each
session `R2_SSE`, `Delta[s,k,b]`, and the animal-balanced estimand in every
replicate. With `theta_hat` the observed estimate, use the centered plus-one
p-value

```text
p[k,b] = (1 + count(theta_star[k,b] - theta_hat[k,b]
                    >= theta_hat[k,b] - 0.01)) / 10000
```

For the intersection-union claim within a stratum, set
`p_IUT[k] = max_b p[k,b]`. Apply one-sided Holm correction at familywise
`alpha = 0.05` across the two fixed strata, M1 and PMd: test the smaller
`p_IUT` at `0.025`, and only if it rejects test the larger at `0.05`; equal
p-values are ordered M1 then PMd. This is paired whole-trial bootstrap
inference for exchangeable repeat trials within the fixed day-by-direction
cells, retaining their observed cell sizes, from these six fixed target days.
It is not an exact randomization test and does not sample sessions, direction
composition, or animals.

## Candidate rule and exact terminal semantics

A stratum `k` passes only if all of the following hold:

1. its Holm-adjusted `p_IUT[k] <= 0.05`, so the `0.01 R2` margin null is
   rejected against each comparator;
2. for each comparator separately, the within-animal mean `Delta` is positive
   in both Mihili and Chewie-L;
3. in each animal, at least two of its three target days have positive
   `Delta[s,k,b]` against both comparators;
4. every leave-one-target-day-out animal-balanced estimate is positive against
   each comparator, recomputing the affected animal mean over its two remaining
   days; and
5. every required scientific falsifier and robustness check passes, with all
   integrity checks valid.

The frozen local conclusion and outer status are assigned in this precedence
order:

| Condition | Local conclusion | Outer status |
| --- | --- | --- |
| Any forbidden access, broken day/trial role, input or lock mismatch, leakage-integrity failure, post-lock mutation, premature/duplicate audit opening, or invalid inference | `technical_failure` | `technical_failure` |
| A valid audit has at least one passing stratum | `candidate_ready_m1_only`, `candidate_ready_pmd_only`, or `candidate_ready_both` | `candidate_ready` |
| No stratum passes, but at least one has `min_b theta_hat[k,b] > 0.01 R2`, or a prespecified scientific robustness guard prevents a clean margin interpretation | `unresolved` | `closed_no_candidate` |
| A valid audit finishes, neither stratum has `min_b theta_hat[k,b] > 0.01 R2`, and no robustness ambiguity invokes `unresolved` | `audit_failed` | `closed_no_candidate` |
| Required development search completed, but no feasible finalist exists or the single frozen finalist fails any scientific falsifier | `development_exhausted_no_eligible_policy` | `closed_no_candidate` |
| Audit was not opened because minimum paired coverage, falsifiers, or an authorized resource/search requirement was not completed | `incomplete_search` | `closed_no_candidate` |

The current outer protocol has no separate incomplete state; its outward
mapping to `closed_no_candidate` is administrative and is not evidence of no
transfer signal. Likewise, `audit_failed` means only that this frozen
candidate rule did not pass; it is not a population null. Once any audit
metric or candidate-discriminating audit diagnostic is visible, the audit is
consumed and cannot trigger successor search.

If exactly one stratum passes, the claim names only that region/epoch stratum.
If both pass, report support in both prespecified strata, not a phase-general
effect. Even a successful audit supports only the added value of earlier-day
information for an already calibrated target day and future trials drawn from
the six fixed target days in these two animals, implants, windows, and the
provider-preprocessed representation. It does not establish generalization to
a new session, animal, implant, region, raw signal, online setting, or clinical
use.
