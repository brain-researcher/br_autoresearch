# EP07 paper plan: what part of an earlier recording day is still useful today?

Status: proposed study design, 2026-09-24. The public corpus and prior campaign
work are already outcome-exposed. No primary EP07 result, mechanism result, or
external validation is claimed here.

## The intended discovery

EP07 begins with a deliberately narrow test. A target day supplies its own
calibration trials. The question is whether earlier days from the same animal,
implant, and region still improve prediction on untouched trials, beyond both
a direction-by-time mean and an equally tuned calibration-only model.

The paper should not stop at "transfer improved R2." It should determine what
the source days contributed:

1. a better estimate of the average response for a reach direction and time;
2. a reusable low-dimensional population trajectory after day-specific
   alignment; or
3. a reusable relationship between same-trial LFP fluctuations and population
   spike-count fluctuations after the average reach response is removed.

The third result would be the strongest scientific contribution because it
says what information, not just what model, survived the change of recording
day. The first two would still be useful but support narrower claims.

The primary adaptive round in [GOAL.md](../GOAL.md) remains the authority for
the first question. The mechanism work below is a separate proposed round. It
cannot change the primary endpoint, replace either comparator, reopen the
audit, or rescue a failed primary result.

## A concrete example

Take two reaches to the same rightward target at 240 ms after movement onset.
The direction-by-time mean predicts the same population response for both.
Suppose the LMP is larger than usual on the first trial and smaller on the
second, and the population spike counts change in the same direction.

If source days teach that signed trial-to-trial relationship and it persists
on a new day after limited calibration, EP07 has found reusable residual
coupling. If source days improve only the common rightward-reach trajectory,
then EP07 has found reusable task geometry instead. A plot of two aligned mean
trajectories cannot distinguish these explanations; a separately fitted
residual-only prediction task can.

## What prior work already established

| Prior work | What it already showed | What EP07 must add |
| --- | --- | --- |
| Gallego-Carracedo et al., eLife 2022, [doi:10.7554/eLife.73155](https://doi.org/10.7554/eLife.73155) | In the source dataset, the relationship between LFPs and population latent dynamics is region- and frequency-dependent and can be stable across behavioral periods. | A chronological source-to-target-day test with target calibration, matched no-source tuning, untouched trials, and an account of which relationship transfers across days. |
| Hall et al., Nature Communications 2014, [doi:10.1038/ncomms6462](https://doi.org/10.1038/ncomms6462) | Multichannel low-frequency motor-cortical LFPs can estimate local single-neuron firing rates and support real-time biofeedback. | Show that an LMP/low-frequency relationship transfers across recording days under the EP07 roster and calibration rules; within-day or local-neuron prediction is not cross-day evidence. |
| Ahmadi et al., Scientific Reports 2021, [doi:10.1038/s41598-021-98021-9](https://doi.org/10.1038/s41598-021-98021-9) | Entire spiking activity can be inferred from LFP features, with LMP reported as a strong predictor. | Separate a genuinely low-frequency relationship from high-frequency spike-rich features and test source-day added value over a matched target-calibration model. |
| Gallego et al., Nature Neuroscience 2020, [doi:10.1038/s41593-019-0555-4](https://doi.org/10.1038/s41593-019-0555-4) | Motor-cortical population dynamics can remain similar across days after alignment despite unstable recorded units. | Evidence about LFP-to-population prediction, especially same-trial residual coupling, rather than another demonstration that average spiking trajectories align. |
| Degenhart et al., Nature Biomedical Engineering 2020, [doi:10.1038/s41551-020-0542-9](https://doi.org/10.1038/s41551-020-0542-9) | Aligning low-dimensional spiking activity can stabilize an online intracortical BCI under recording instability. | A target-calibrated LFP-to-spike transfer claim under EP07's fixed signals and days; offline internal evidence must not be described as online BCI stabilization. |
| Sussillo et al., Nature Communications 2016, [doi:10.1038/ncomms13749](https://doi.org/10.1038/ncomms13749) | Training spike-to-kinematic decoders on many historical conditions can improve robustness to future neural variability and electrode loss. | Show whether earlier LFP/spike days add information beyond a matched same-day model, and identify the source of that gain rather than relying on a larger training set alone. |

These papers make "cross-day transfer is possible" and "LFP predicts spikes"
insufficient novelty claims. EP07 needs a signal-specific result: a named,
falsifiable part of the LFP--population relationship that survives days, or a
useful boundary showing that only average task geometry survives.

## Exposure timeline and honest evidence labels

The mechanism plan is being written after the public source, source paper, and
earlier campaign analyses already exposed neural outcomes. It must not describe
the mechanism hypotheses as prospectively frozen before all data access.

| Time | What is or will be exposed | Evidential role |
| --- | --- | --- |
| Before this revision | Public Dryad outcomes, source-paper analyses, and prior EP05--EP08 campaign knowledge | Historical development exposure recorded in the shared ledger |
| Primary EP07 development | Source-day targets, target calibration, and target-development outcomes inside the trusted evaluator | Selection evidence for the original transfer policy, not fresh mechanism confirmation |
| This mechanism revision | Candidate mechanism list, residual task, signal-boundary controls, tie rules, and proposed predictions | Frozen before the first new mechanism-specific score or diagnostic is computed or revealed, not before the corpus was seen |
| Primary EP07 audit | One consumed opening under the original lock | Primary transfer evidence plus only generic secondary outputs declared before that opening; it cannot choose the mechanism |
| Mechanism development | Already exposed development roles | Hypothesis development and internal cross-fitting only |
| Newly sequestered target days | Outcomes hidden until the separate mechanism lock is complete | First prospective test of a mechanism selected from this corpus |

Every report should state which line supplied each figure. If a mechanism
diagnostic was computed before its rule was frozen, it becomes development
exposure and must not be reused as confirmatory evidence.

## Study sequence

### 1. Finish the primary forward-transfer test

Run the unchanged chronological day assignment, nominal 20% target calibration,
matched two-stream search, 16 anchors per stream, 2--14 adaptive successor
pairs, 10 finalist-bound falsifier pairs, and one locked audit. Preserve the
native-bin spike-count response and the exact animal-balanced estimands.

Report both comparators for all six target days and both region/epoch strata.
A source-bearing model must not be called successful because it beats only the
direction-by-time mean; it must also beat the equally tuned calibration-only
champion under the frozen terminal rule.

This round answers whether old days help on held-out trials from the six fixed
target days. It does not yet explain why, and it does not test a wholly unseen
day.

### 2. Fit a genuine residual-only prediction task

Simply subtracting one direction-by-time mean from a completed full prediction
and its target leaves their SSE unchanged. That algebraic recentering is not a
mechanism test. The follow-up instead fits new, capacity-matched source-bearing
and calibration-only models after residualization.

For each source day, estimate direction-by-time means for both LFP inputs and
spike-count targets from that day's permitted source-training trials only. For
the target day, estimate the corresponding means from target calibration only.
Subtract those means before fitting any residual model:

```text
x_res[t,q]   = x[t,q] - mean_allowed[x | direction(t), q]
y_res[t,q,j] = y[t,q,j] - mean_allowed[y_j | direction(t), q]
```

No residual model may see uncentered source values, a source-day mean template,
target-development targets, or audit outcomes. The source-bearing and
calibration-only streams receive matched representations, capacity, tuning
opportunities, and development feedback. Their predictions are scored directly
against held-out `y_res`; the mean component is fitted and reported separately.

#### Bridge different neuron rosters without matching neuron IDs

Each day keeps its own native neuron roster. Residual spike counts are first
formed within that native roster; source and target vectors are never joined by
unit number, waveform, electrode number, or a guessed neuron identity.

For each source day, fit its latent loading using only permitted source-training
residuals. Align source-day latent coordinates through the contract's
train-only, behavior-anchored rule. Behavior anchors and all alignment choices
must be computed from permitted training trials and fixed before any target
held-out outcome is available. The alignment maps low-dimensional coordinates,
not native neurons.

On the target day, fit a target-native loading and the permitted latent adapter
using target calibration residuals only. The frozen source-bearing model
predicts a target-day latent residual from residualized LFP; the target
calibration loading then maps that prediction back to the target day's native
neuron-count residuals. Add the target-calibration direction-by-time mean only
for the separately reported full-count prediction. The calibration-only model
uses the same target-native output construction but no fitted source state.

Thus the bridge is:

```text
native residual roster within each day
    -> train-only behavior-anchored latent coordinates/alignment
    -> target-calibration loading
    -> target-native residual spike-count prediction
```

Matching neuron IDs across days, intersecting source and target rosters,
directly applying a source-day native loading to target neurons, or fitting a
target loading/alignment from held-out outcomes is forbidden.

The proposed primary mechanism score is the animal-balanced difference in
held-out residual prediction between the source-bearing and matched
calibration-only residual models. Exact weighting, meaningful margin,
uncertainty, qualification, and invalid-model rules belong in the separate
follow-up contract; they are not inherited from the primary native-count
endpoint.

### 3. Establish that the signal is not high-frequency spike bleed

Predicting spike counts from 100--400 Hz power can reflect spike-rich content
on the same electrode rather than a reusable field-potential relationship.
Hall et al. and Ahmadi et al. make a low-frequency test plausible, but they do
not prove it for EP07. A trial-specific field-potential interpretation requires
all of the following prespecified views on identical trial roles:

- **LMP/low-frequency only:** refit the complete residual comparison without
  any 100--400 Hz power input;
- **high-frequency exclusion:** remove every authenticated 100--200 and
  200--400 Hz feature, rather than merely down-weighting it after fitting;
- **same-electrode/unit intersection:** repeat on the prespecified physical
  electrode/unit support where spike-rich proximity can be accounted for;
- **spike-quality matching:** compare subsets matched on unit yield, waveform
  or provider quality summaries, missingness, and available channel count; and
- **frequency-boundary controls:** report LMP, low-frequency, high-frequency,
  and combined models with matched capacity and no outcome-driven band choice.

If residual transfer exists only in 100--400 Hz power, on spike-rich
electrodes, or before quality matching, report a high-frequency/spike-bleed
boundary. Do not call it general LFP--spike coupling. A low-frequency-only
result still establishes prediction, not synaptic causality or a stable
single-neuron identity.

### 4. Identify the part of the model carrying the gain

The mechanism analysis should be small enough to interpret. Because generic
development outcomes are already exposed, freeze at most two candidate
mechanisms, every tie, and every ablation before any new mechanism-specific
score or diagnostic is returned. The allowed candidates are:

- direction/time template versus behavior-anchored population coordinates;
- LMP/low-frequency versus authenticated high-frequency families;
- source-day positions 1, 2, and 3; and
- population dimensions ordered by a source-only rule, not by target outcome.

For each retained candidate, run a fixed component or source-day omission on
identical target trials. A component supports the explanation only if removing
it reduces residual source gain in both primary animals without causing a
generic failure of both source and calibration-only models. Existing F01--F10
records are used only where they answer the exact mechanism question.

Three alternatives have to remain visible:

| Alternative | Decisive comparison | Interpretation if it wins |
| --- | --- | --- |
| Better behavioral averaging | Separately scored mean component; residual-only models show no source gain | Source days mainly improve a task template |
| Generic regularization or more rows | Capacity-matched calibration-only residual model, zero-source ablation, and synthetic sample-size controls | No evidence for a source-specific neural relationship |
| One influential source or target day | Each source-day omission and leave-one-target-day result | Restrict the finding or stop; do not call it a general cross-day property |
| High-frequency spike-rich content | LMP/low-frequency-only, 100--400 Hz exclusion, same-electrode/unit, and quality-matched checks | State the signal boundary; no general field-potential mechanism |
| Average aligned dynamics only | Aligned mean trajectories transfer but separately fitted residual models do not | Claim stable geometry, not trial-specific coupling |

### 5. Predict when historical data will help

An explanation becomes useful when it makes a prediction before target outcomes
are seen. Build a small transferability rule from quantities available after
source fitting and target calibration, such as:

- source-to-calibration agreement of direction/time population trajectories;
- disagreement among the three source-day residual predictions;
- reliability of the retained low-frequency feature block in target
  calibration; and
- alignment conditioning and the fraction of target-calibration residual
  activity represented by the source-fitted latent coordinates.

The outcome to predict is the target day's source-versus-calibration-only gain,
reported separately for native counts and the residual-only task. Development-
day cross-fitting is mandatory: the rule predicting one target day cannot be
fit using that day's held-out gain. Limit the rule to a frozen linear or
monotone form and at most two mechanism scores so a flexible predictor cannot
memorize six days.

This analysis is hypothesis development in the current exposed corpus. Six
target days are too few for a broad claim about prediction of transfer success.
Its first prospective test is the next stage.

### 6. Freeze the rule and test newly sequestered days

Before opening any new target-day spike outcomes, freeze the complete source
manifest, exposure ledger, chronology, target eligibility, calibration
fraction, native roster rules, LFP feature meanings, residualization, latent
alignment and back-mapping, models, transferability rule, signal-boundary
controls, comparators, margins, multiplicity, missing-data behavior, compute
budget, stopping rule, and result table.

The strongest validation would use a third animal with enough chronological
source and target sessions. Neural weights and target-native loadings would
still be fit separately within that animal; what transfers across animals is
the procedure and the mechanism prediction, not a common electrode or neuron
map. A genuinely unexposed later day in Mihili or Chewie-L would test new-day
transfer but not population generalization. Chewie-R is an implant sensitivity,
not a third animal.

For every new target session, the frozen procedure must predict:

1. native-bin population activity on untouched trials;
2. the within-direction/time residual activity;
3. whether adding source days will beat the calibration-only residual model;
   and
4. whether the result survives the locked low-frequency/spike-bleed controls.

Report all eligible target sessions. Do not retain only days the
transferability rule labels favorable. No compatible fresh source is currently
selected, so external validation remains a requirement rather than an implied
result.

## Result-dependent decisions

| Evidence after the primary round | Next step | Strongest defensible paper claim |
| --- | --- | --- |
| Primary transfer fails cleanly in both strata | Stop the mechanism story; report the fixed-corpus boundary if informative | Earlier days did not clear the prespecified added-value margin under this calibration regime |
| Primary estimate is positive but heterogeneous or imprecise | Examine prespecified day influence only; seek more target days before a mechanism claim | Source information may help some fixed days, but the stable relationship is unresolved |
| Gain is restricted to the separately scored direction/time component | Deepen the task-geometry explanation and test it prospectively | Earlier days improve estimation of stable reach structure, not trial-specific coupling |
| Separately fitted residual gain survives only high-frequency or spike-rich views | Report the signal boundary and stop the broad field-potential story | Spike-rich high-frequency features predict population residuals in this fixed corpus |
| Low-frequency residual gain survives controls but cannot predict new-day benefit | Report an exposed-corpus mechanism result with limited scope | Residual coupling is detectable internally; prospective usefulness remains unverified |
| Low-frequency residual gain and transferability prediction both hold on newly sequestered days | Develop the full cross-day relationship claim | A specified low-frequency LFP--population residual relationship transfers under limited calibration in the declared scope |
| M1 and PMd differ | Name the exact region/epoch stratum; do not turn the contrast into a preparation-versus-execution claim | Stratum-specific result only because region and epoch are confounded |

The mechanism route should be narrowed or stopped when its effect disappears
under the matched calibration-only residual model, is carried by one day,
cannot be separated from the mean component, or lacks a fresh target-day test.
Adding more latent models after such a failure is not a valid rescue.

## Paper figures, contingent on evidence

| Figure | Scientific judgment | Required content |
| --- | --- | --- |
| 1. Do old days help? | Source information adds value after same-day calibration | Chronological design, both comparators, every target day, animal-balanced effects, uncertainty, and negative scores |
| 2. What transfers? | Gain belongs to average task geometry, separately fitted residual coupling, or both | Mean-component result; residual-only source and calibration fits; native-roster-to-latent-to-target-native bridge; no algebraic recentering |
| 3. Is it a field-potential result? | Low-frequency information carries gain beyond spike-rich contamination | LMP/low-frequency-only model, 100--400 Hz exclusion, same-electrode/unit and quality-matched checks |
| 4. Can benefit be predicted before evaluation? | Calibration-visible measurements forecast when history is useful | Cross-fitted exposed-development predictions, prespecified score, failures as well as successes |
| 5. Does the prediction survive a new day? | The explanation transfers beyond the corpus that generated it | Frozen prospective procedure, all new sessions, activity and benefit predictions, exact animal/implant scope |

The abstract should ultimately say which stratum was supported, whether the
gain concerned means or separately fitted residuals, which frequency boundary
survived, and what kind of new session was tested. It must state that mechanism
development used an already exposed public corpus. It must also say that the
experiment is offline, uses provider-preprocessed features, and does not
establish clinical BCI performance, synaptic causality, or stable
single-neuron/electrode identity.
