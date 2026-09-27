# Can EEG forecast movement before peripheral sensors detect its onset?

Suppose a decoder raises an alarm 100 ms before the recorded movement onset.
That sounds like a forecast, but the hand or muscles may already have started
to change, and future-looking filtering can move those changes backward in
time. A high score may therefore reflect recognition of a movement already
under way rather than prediction of a future event.

EP19 asks a stricter question: **does EEG improve prediction of movement onset
300–600 ms in the future, beyond everything already known from the cue, task
context, and past peripheral sensors?** Every 50 ms while a participant is
still at risk of moving, the model predicts the probability of onset across the
next 900 ms. Only information available at the decision time may enter the
prediction.

Real EEG must beat three kinds of comparison on the same examples: a strong
non-neural model, the same model with a zero EEG input, and separately trained
models receiving capacity-matched surrogate EEG. An impressive absolute score
is not evidence for prospective neural information if cue timing, past muscle
or movement signals, or scrambled EEG can reproduce it.

The secondary question asks whether changing the prediction horizon changes
which model looks best. The same pre-specified model families are trained once
to predict all horizons, then compared at **0–300 ms** and **300–600 ms**.
Their ordering may persist, disappear, become visible only at the stricter
horizon, reverse, or remain uncertain.

Finally, the primary EEG increment must repeat in **eight held-out people**
performing a different self-paced reaching task. For the main transfer test,
each person contributes only the first **32 labelled movements** for a
four-parameter calibration; the neural encoder cannot be retrained.

In EP19, **causal means past-only, prospective information flow**: filtering,
normalization, and model state at time `t` use no sample after `t`. It does not
mean causal inference about motor preparation, conscious intention, or the
biological cause of movement.

![EP19 scientific question](outputs/ep19_question.svg)

Like the EP12 concept figure, this mockup uses synthetic traces and bars to make
four scientific judgments visible: genuine forecasting versus near-event
detection, the three real-EEG specificity increments, state changes in all ten
pairwise model-family edges, and the registered eight-person second-task test.
None of the traces, bars, or edge states is an EP19 result; exact margins and
source-construction details remain in the study text.

## The study at a glance

| Question | EP19 design |
| --- | --- |
| What is predicted? | At each 50-ms decision point, onset probability in eighteen 50-ms intervals across the next 900 ms, plus no onset within 900 ms |
| What is the primary horizon? | Onset 300–600 ms in the future |
| What is the near-event comparison? | Onset 0–300 ms in the future |
| What is the longer diagnostic horizon? | Onset 600–900 ms in the future |
| What must EEG beat? | A source-specific non-neural model, its zero-EEG counterpart, and eight separately trained surrogate-EEG counterparts |
| What pairwise order is studied? | All ten edges among five pre-specified model families, each fit once for all horizons and compared at 0–300 and 300–600 ms; no total rank is assumed |
| What is held out? | WAY-EEG-GAL series 8 and 9, plus eight whole participants from a second self-paced task |
| What adaptation is allowed in the second task? | Four calibration parameters fit from the first 32 complete movements; the encoder stays unchanged |
| What would success show? | Under the registered sensors and past-only analysis, EEG adds calibrated prospective information at 300–600 ms and the gain repeats in new participants |

## The primary question: does EEG add prospective information?

Two source-specific non-neural models are selected first: one for WAY-EEG-GAL
and one for the self-paced task. They may use causally available cue and task
context, past EMG where present, past kinematics, force or accelerometry, past
EOG where present, posture, recent motion, stillness duration, session time,
and missingness. They never receive EEG.

Every EEG model is evaluated as a matched set:

1. the selected non-neural model alone;
2. the same architecture with a real EEG branch;
3. the same architecture with a zero EEG branch; and
4. the same architecture trained separately on surrogate EEG.

The final surrogate set contains eight transformations: four
session-preserving block shifts and four cross-channel-coherent phase
randomizations. Each counterpart is trained from the beginning with the same
architecture, optimization, data budget, seeds, and calibration procedure.
The primary neural claim requires real EEG to beat the strongest surrogate for
each participant, not merely the average surrogate.

The three EEG comparisons remain separate:

- real EEG minus the non-neural model;
- real EEG minus the zero-EEG model; and
- real EEG minus the strongest capacity-matched surrogate model.

All three must clear margins chosen before any candidate EEG score is examined.
Those margins are still unset. Failing to resolve a margin is reported as
uncertain, not as proof that EEG contains no prospective information.

## Time, onset, and past-only prediction

At decision time `t`, the model receives only samples and state available at or
before `t`. It predicts 19 probabilities: eighteen left-closed, right-open
50-ms onset intervals spanning 0–900 ms, plus one probability for onset later
than 900 ms.

The primary score is a natural-prevalence binary log score for the probability
of onset inside each named horizon. Training uses the corresponding
censoring-aware 19-category likelihood. An example is scored at a horizon only
when onset status is observed through the upper edge of that horizon.
Administrative censoring uses only complete 50-ms intervals and never converts
an unknown future into a negative label.

The onset detector is built from raw non-EEG sensors. EEG cannot define the
label. A later confirmation sample may verify whether a candidate crossing was
real, but that future sample cannot enter the input or restore decision points
that occurred while the crossing was unresolved. The worst-case timing and
synchronization uncertainty must stay below the 300-ms lower edge of the
primary horizon.

All filters are one-sided, derivatives look backward, normalization uses
training data or accumulated past samples, and recurrent state resets at every
acquisition gap and series or session boundary. Centered filters, future
padding, whole-session normalization, bidirectional recurrence, and windows
crossing a data split are prohibited.

## The secondary question: does model order depend on the horizon?

The reference comparison contains five families:

- causal filter-bank band power with elastic net;
- causal covariance and Riemannian tangent features;
- a causalized compact EEGNet;
- a dilated causal temporal convolutional network; and
- a small one-directional GRU.

Each family receives the same 12-configuration search allowance and three
seeds. One joint 19-category model is fit and scored at both 0–300 ms and
300–600 ms. This avoids comparing two separately optimized winners and then
attributing their difference to forecast horizon.

For every well-resolved pair of model families, the conclusion can be:

- **retained:** the same direction of advantage at both horizons;
- **collapsed:** a near-event difference becomes practically equivalent;
- **emergent:** no near-event difference but a strict-horizon difference;
- **reversed:** the direction changes; or
- **uncertain:** the available participants do not distinguish these states.

A bounded search for one new model is secondary. It starts from the strongest
eligible reference and changes one operator at a time, with at most two million
parameters, two seconds of explicit input history, and eight seconds of stored
streaming state. A development winner cannot be inserted into the reference
panel or redefine the model-order question.

## Three data roles

### WAY-EEG-GAL: main forecasting and model-order study

For each of 12 participants, series 1–6 fit and select models, series 7 provides
rate-limited aggregate development feedback, and series 8 and 9 are used
together in the final held-out evaluation. Weights are participant-specific,
but the model family, preprocessing, hyperparameters, epoch rule, and seeds are
shared across participants.

### Self-paced reaching: new-participant replication

Fifteen participants develop the second-task procedure and group encoder.
Eight different participants are held out in full. Their primary comparison
uses the first 32 complete eligible movements for calibration and scores the
remaining data after a history guard.

This source has wrist accelerometry but no EMG. It can therefore repeat a claim
about forecasting before accelerometer-detectable wrist motion, not the
stronger WAY-specific claim about forecasting before any measured muscle
activation.

### Historical event detection: compatibility only

The earlier Kaggle grasp-and-lift task predicts six event labels around event
times and is not a strict premovement forecast. Before the final held-out test,
EP19 may check a legacy-compatible series-1–6 to series-7 implementation and
report `HandStart` separately. The exact series-1–8 to series-9 historical
comparison is deferred until after the scientific evaluation. It cannot
promote a model or rescue a failed prospective result.

## Thirty-two-event calibration in the second task

The first 32 complete eligible movements for each held-out participant are
taken in chronological order. Their full stream prefix and a history guard are
calibration-only. The no-label and 32-label conditions are scored on the same
later suffix.

The primary calibration has only four free values:

- one positive global temperature; and
- three offsets for the 0–300, 300–600, and 600–900 ms groups, with the
  `>900 ms` group as reference.

The encoder, spatial projection, and individual 50-ms category biases cannot
change. The regularization, bounds, optimizer, iteration limit, convergence
tolerance, and deterministic failure rule must be specified before these 32
events are used. Each scored model receives its own four-parameter fit under
the same procedure; one model's calibration values are not shared with
another.

No-label transfer is a stricter secondary result. A local refit using the first
half of complete runs is a descriptive upper bound only and cannot rescue the
primary 32-event result.

## Controls that can overturn the result

- changing any sample after time `t` must leave the prediction at `t`
  unchanged;
- sample-by-sample, variable-chunk, and batched streaming must agree;
- an impulse placed in the future must have zero effect on earlier outputs;
- filters, normalizers, and recurrent state reset at acquisition gaps and
  series or session boundaries;
- centered filtering, future padding, whole-session normalization, and split-
  crossing windows must be rejected;
- cue-stratified label shuffling and circular or block EEG shifts must return
  to the null;
- participant or session identity alone must not explain the result;
- all eight capacity-matched surrogate EEG models must be run for the final
  comparison;
- reasonable pre-set onset variations must not create the conclusion;
- a deliberately leaked future signal must be detected as abnormally strong;
- a synthetic prospective EEG motif must be recovered at the stated signal
  levels;
- the legacy near-event task must reach its pre-set positive-control range;
- raw voltage and causal filter-bank representations, receptive field, and
  low-frequency content must be ablated;
- leave-one-participant and leave-one-series influence checks must agree in
  direction; and
- no-label and identical 32-event calibration packets must be compared.

A failure of a shared timing, future-information, or positive-control check
invalidates the scientific comparison. It is not evidence that prospective EEG
information is absent.

## Development budget and stopping

Development begins with **12 complete coverage trials** spanning classical,
temporal, representation, and fusion approaches, with a direct control for
each. The full search requires **24–40 valid trials**, at least **two genuine
follow-up cycles**, at least **two recorded choices between the current best
model and a new proposal**, and at least **40% challenge tests, ablations,
influence checks, or direct replications** after coverage. Patience is eight
eligible opportunities after the minimum evidence is complete.

At most 40 aggregate development feedback calls are allowed. Up to 12
repairable engineering failures are allowed; the next one ends the study as a
technical failure.

Resource limits are **2,000 CPU core-hours**, **480 GPU-hours**, **240
wall-clock hours**, and **2 TiB of scratch space**. Nested reference fits and
all surrogate models count against those limits. No partial fold result may be
used to stop an unfavourable trial early. Reaching a resource limit does not
establish a positive or negative scientific result.

## What the study can conclude

| Evidence pattern | Scientific reading |
| --- | --- |
| Strict EEG increment repeats and model order is retained | Prospective EEG information is supported and the resolved reference advantages survive the stricter horizon |
| Strict EEG increment repeats but order changes | Prospective EEG information is supported, but conclusions about model families depend on prediction horizon |
| Strict EEG increment repeats but order is uncertain | The neural result is supported, but available participants do not resolve the model ordering |
| EEG helps only at 0–300 ms | The result is near-event recognition, not the primary strict forecast |
| Real EEG does not beat zero or surrogate EEG | Neural specificity is not supported even if absolute prediction is good |
| WAY succeeds but the eight-person second task fails | The result does not replicate under the registered second-task procedure |
| No new model improves on the references | The model-search result is negative; this does not by itself answer whether EEG adds information |
| A required estimate remains too uncertain | Report the unresolved comparison rather than choosing a winner |

## Relation to prior work

Premovement EEG prediction, WAY-EEG-GAL event detection, pre-onset kinematic
reconstruction, and cross-session neural benchmarks all predate EP19. The
study therefore makes no priority claim for predicting movement before onset
or for applying deep learning to these data.

The possible contribution is the conjunction of a continuous at-risk forecast,
a strong source-specific past-peripheral baseline, capacity-matched surrogate
EEG, a near-to-strict model-order comparison, and whole-participant replication
in a second task.

## Claim boundary

A positive result would show that, under the specified sensors, onset detector,
prediction horizons, calibration budget, and model panel, EEG adds prospective
information about sensor-detected movement onset. It would not establish:

- causal motor preparation or conscious intention;
- the earliest biological motor command;
- prediction before an unmeasured peripheral change;
- a universal model ordering;
- zero-shot transfer to a new task;
- online BCI usefulness, safety, or clinical benefit; or
- independent confirmation on a globally untouched dataset.

## Current status

The data sources are identified, but no EP19 signal analysis has started and no
held-out result has been examined. Before scoring, the study still needs
source-specific timing and onset checks, separated development and held-out
data views, the complete 32-event calibration procedure, all numeric margins,
the finite model-search ranges, and synthetic tests of past-only information
flow.

Dataset details are in [DATASETS.md](DATASETS.md). Compact methods and budgets
are in [SEARCH_POLICY.yaml](SEARCH_POLICY.yaml). The proposed paper story is in
[outputs/paper_plan.md](outputs/paper_plan.md).
