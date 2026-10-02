# Why does a human iBCI mapping work in one session but not another?

Train a direction-prediction mapping on an earlier BrainGate session, then
apply it unchanged to a later session from the same participant. If it performs
worse, the score alone does not tell us why. The later session might contain
less usable direction signal, the signal might still be present but differently
aligned, or ordinary changes in the recorded channels might be sufficient to
explain the loss.

EP16 first asks whether there is an extra long-gap problem at all. For each
participant, it compares session pairs separated by **1–30 days** with pairs
separated by **at least 180 days**. A transported mapping is judged against two
benchmarks on the same later-session trials: a mapping trained inside the later
session and a direction-balanced null. This separates a cross-session mismatch
from a later session that is simply difficult for every model.

The study then keeps two questions separate. A later session can contain less
locally recoverable direction signal even when the **fraction** of that signal
lost by transport does not increase. That local-signal route is therefore
tested whether or not an extra normalized transport loss is found. If an extra
transport loss is established, the same panel asks whether observable recording
change can reproduce it and whether a small labelled calibration set can repair
it. The three signatures can coexist:

1. the later session itself supports a weaker locally trained mapping;
2. changes visible without direction labels—missing or noisy channels, invalid
   samples, or changed channel statistics—can reproduce the loss; and
3. a bounded recalibration using exactly **32 labelled later-session trials**
   can repair a meaningful part of the loss.

A result must follow the same pre-set rule in **all three held-out
participants**. Different participant patterns count only when each pattern
repeats in two separate time-based partitions of that participant's sessions.
The study is therefore about reproducible failure signatures, not about finding
the most dramatic example of drift.

![EP16: transferred mapping, target-local signal and coexisting operational signatures](outputs/ep16_question_imagegen-v2.png)

The figure compares an unchanged earlier-session mapping, a target-local
mapping and a null on the same later-session trials. Cursor-to-target direction
is the task proxy, not a direct intention label. The three illustrated
signatures can coexist: less locally recoverable signal, a label-blind
recording-state emulator sufficient to reproduce loss, and recovery with a
fixed 32-label procedure. The earlier recording feeds the emulator; the
repair is not the unchanged transported mapping. Drawings show no observed
performance or identified biological mechanism. Exact margins and participant
rules remain in the study text; [generation and correction prompts](outputs/ep16_question_imagegen-v2-prompt.md)
are saved.

## The question in one table

| Question | EP16 answer |
| --- | --- |
| Data | Published version 6 of the BrainGate Dryad release `10.5061/dryad.x0k6djj1h` |
| Independent biological case | One participant; all arrays and sessions from that person stay together |
| Development and final evaluation | Six participants for development and three participants for one final held-out evaluation; the exact people are chosen before neural scores are used |
| Neural feature | Mean spike-band power (`sbp`) in one 300-ms onset window; `tx_4_5` is a required sensitivity check |
| Prediction | The cursor-to-target direction at the start of that window; this is a task-derived proxy, not a direct intention label |
| Main comparison | Mapping trained in one session and evaluated in another, relative to a later-session model and a null on the same trials |
| Time contrast | 1–30 days versus at least 180 days; 31–179 days are descriptive only |
| Main score | Direction-balanced negative two-dimensional mean-squared error; higher is better |
| Recalibration budget | Nested packets of 16, 32, and 64 trials; the primary low-budget decision uses 32 labels |
| Final evidence rule | The same conclusion must hold separately in all three held-out participants |

## What is actually measured

The public release contains preprocessed 10-ms neural features, cursor and
target positions, trial and block boundaries, and electrode information. It
does not contain the historical online decoder, raw voltage, the complete
online preprocessing state, target size, or a direct intention label.

The analysis therefore studies an **offline mapping built in a source
session**. It does not reconstruct what the historical online decoder would
have done if moved to another day. Because the participant's neural and cursor
trajectories were generated under the decoder used on that day, every result is
an offline diagnostic rather than a closed-loop counterfactual.

For each complete trial, the predictor is mean neural activity in one fixed
300-ms window. The outcome is a unit vector from cursor to target at the start
of that window. A single onset offset is chosen from task timing and
physiological considerations before any neural comparison. Later cursor
corrections cannot change either trial eligibility or the label.

The primary model is a capacity-matched multi-output ridge regression.
Direction octants receive equal total weight. Ten-millisecond bins are never
treated as separate observations, and whole blocks remain together when trials
are split.

Each eligible session contributes one 128-trial packet. Whole blocks are
divided into two 64-trial sides; each side serves once for fitting and once for
evaluation. Nested 16-, 32-, and 64-trial subsets provide equal exposure for
all recalibration methods. Trials beginning within 5% of the schedule's median
nonzero target spacing are excluded because their direction is unstable.

Channels are aligned by physical array and electrode identity, not by column
number in a file. A channel missing from the later session is represented as
missing after source-based scaling rather than silently dropped. Repeating the
analysis on electrodes observed in both sessions is a required control.

## Establishing an extra long-gap loss

Every later-session evaluation uses three scores on exactly the same trials:

- **null benchmark:** what can be predicted without a direction-dependent
  neural mapping;
- **later-session benchmark:** what the same model class can recover when fit
  inside the later session; and
- **transported mapping:** what an earlier-session mapping recovers without
  retraining.

For a chronological source-to-target pair `s -> t`, let `B_t` be the null
score on target trials, `L_t` the target-local score, and `F_st` the unchanged
source mapping's score on those same trials. Each score is first averaged
equally over the two reversed outer folds. Only then define

`R_st = (L_t - F_st) / (L_t - B_t)`.

`R_st` is the fraction of locally recoverable above-null performance lost in
transport. It is not clipped: a value below zero means transport beat the local
benchmark, and a value above one means it fell below the null. If
`L_t - B_t` is nonfinite or does not exceed the pre-set readiness minimum, the
pair and its registered connected component are unresolved; the pair is not
dropped, and no epsilon replacement or per-fold ratio is allowed. The
primary estimate compares long-gap and near-gap `R_st` values while accounting
for general source-session and target-session difficulty. A simpler
target-balanced median contrast must also clear its pre-set positive margin.
Session pairs are repeated measurements; the participant remains the
biological unit.

For that robust contrast, first take the median `R_st` across eligible
chronological sources for each target within a gap class and connected
component. Then take the median of those target-level values, form long minus
near within the component, and aggregate schedule groups and components with
the same equal-weight hierarchy as the adjusted estimate. Thus a target with
many sources cannot dominate the check.

### The adjusted contrast and when it can be estimated

For every participant and connected target-schedule component, `R_st` is fit
as:

`transport loss = component average + source-session effect + target-session effect + long-gap effect`.

The coefficient on the long-gap indicator is the adjusted long-versus-near
contrast. It asks whether long-gap pairs lose more after allowing some sessions
to be generally harder sources or harder targets.

Having six near and six long pairs is necessary but not sufficient. The
contrast is used only when the actual source-to-target pairing pattern contains
both gap classes and the constrained design has full rank. In particular, the
long-gap indicator must retain non-zero variation after the source and target
session terms are removed. It must also pass pre-set limits on numerical
conditioning, residual long-gap information, and the influence of any one
session. These limits are fixed before neural scores are examined. A component
that fails them is an unresolved support failure; it is not dropped in favour
of a more convenient set of pairs.

Uncertainty uses 9,999 session-node bootstrap draws. Each sampled occurrence of
a session receives its own bootstrap identity; all source-to-target pairs
implied by the sampled endpoints are rebuilt, and whole blocks/trials are then
resampled inside each occurrence. Pairs are never sampled as independent
observations. A draw that loses fixed-effect rank or a required denominator is
not quietly discarded or redrawn: it enters the lower tail as minus infinity
and the upper tail as plus infinity. If the required 95% bound is therefore
unbounded, that participant's result is unresolved. This makes weak session
support visible instead of conditioning uncertainty on the convenient draws.

The proposed practical anchor for an important extra long-gap loss is **0.10
normalized units**. The exact threshold and its uncertainty rule must be set
before neural scores are examined. A failed or unstable denominator remains an
unresolved comparison rather than being removed after the fact.

The primary estimand uses only chronological pairs: the source session must
precede the target session. Reversing an eligible pair is a required transport-
asymmetry sensitivity, but that reverse score does not enter the adjusted
long-gap contrast or either local-signal contrast. Sessions are compared only
when they share the same participant, physical array set, and observed target
schedule. Schedule matching may remove
translation and scale, but not rotation or reflection. A task name is not
assumed when the release does not provide one. Size-dependent schedules are
excluded when target size is unavailable.

Each held-out participant must retain at least six eligible near-gap and six
eligible long-gap session pairs in a common schedule group. If that support is
not available, the study cannot answer the question for that participant and
does not substitute another person after seeing outcomes.

## Three explanations tested with the same panel

[Component details](outputs/component_details.md) specify the shared fitting
exposure, local learning curves, label-blind emulator construction and
32-label recovery quantities. They keep method exploration open and identify
the remaining scientific choices without changing this study's estimands or
numerical anchors.

### 1. Less locally recoverable signal

For session `u`, let `J_u` be its fold-averaged locally fitted score minus its
fold-averaged null score. For a chronological source-to-target pair, the local fractional change is
`(J_target - J_source) / J_source`; the source must first clear the fixed
readiness threshold. The accompanying raw quantity is the transported mapping
score on the target minus that target's null score.

Within each lag class and connected schedule component, every target session
gets equal total weight, divided equally among its eligible source sessions.
The two diagnostic contrasts are the long-gap mean minus the near-gap mean for
the fractional local change and for the raw transported above-null score. They
use exactly the same components, which must contain both gap classes; schedule
groups and then components receive equal weight. A signal-reduction result
requires simultaneous upper bounds for both contrasts to fall below their
negative margins in every held-out participant. The proposed fractional anchor
is a **15% larger reduction** at long gaps; the raw-score margin still must be
fixed before scoring.

This result would mean that less direction-related signal is recoverable under
the specified feature, trial, and model choices. It would not show that motor
cortex has lost movement information.

### 2. Observable recording-state change is sufficient

A label-blind emulator receives only channel identity, availability,
finite-value rates, and pre-specified summaries of channel quality. Where
available, same-day yield or impedance can be added, but missing values cannot
define who enters the study.

The emulator may drop channels, attenuate them, or add noise to reproduce a
later recording state. It may not invent a missing channel or improve a worse
source recording. It must reproduce both the near- and long-gap changes in
locally recoverable signal and mapping loss. Proposed absolute error anchors
are **0.10 normalized units** for both quantities.

A matched-random control changes the same number and quality distribution of
channels while scrambling which channel is affected. This separates a generic
loss of channel quantity or quality from a spatially specific channel pattern.
A performance correlation alone cannot pass this test.

### 3. A 32-label mismatch that can be repaired

The adaptation panel separates target-data amount from model flexibility:

| Later-session information | Examples |
| --- | --- |
| No target data | Earlier-session mapping unchanged |
| Unlabelled activity only | Past-only normalization, bounded covariance adjustment, or orthogonal alignment |
| 32 labelled trials | Output gain/bias/rotation, orthogonal or low-rank remapping, or bounded supervised alignment |
| 64 labelled trials | Later-session benchmark and capacity-matched upper comparisons |

The main decision uses one method chosen on development participants and then
applied unchanged in form to the three held-out participants. It succeeds only
if 32 labels recover at least **50% of the transport gap** and improve by at
least **0.10 of the locally recoverable above-null signal**. The same 16/32/64
curve is reported so that a favourable 32-trial result cannot hide an unstable
budget response.

## Results the study can distinguish

| Evidence pattern | Scientific reading |
| --- | --- |
| Extra normalized transport loss is not established | The positive long-gap margin was not cleared; this is not evidence of equivalence or absence, and the separate local-signal route can still pass |
| Later-session benchmark also declines | Less usable direction signal is locally recoverable in later sessions |
| Recording-state emulator reproduces the deficit | Observed channel and recording changes are sufficient for the eligible chronological transitions |
| Thirty-two labels repair the deficit | A meaningful part of the mismatch is recoverable with a fixed low label budget |
| More than one explanation passes | The operational signatures are reported together; no additive causal percentage is claimed |
| Fully resolved patterns differ and repeat within person | These participants show reproducibly different profiles, not population subtypes |
| Required comparisons remain uncertain | The study remains unresolved rather than choosing the most attractive story |

## Controls that can change the conclusion

- intercept-only and block-preserving shifted-label nulls;
- a deliberately invalid random-bin split to expose leakage;
- `sbp` compared with the pre-set `tx_4_5` feature sensitivity;
- source- and target-session difficulty controls;
- reverse-direction scores reported only as a transport-asymmetry control;
- direction and observed-target-schedule balance checks;
- identical 16/32/64 trial-budget curves;
- past-only normalization compared with a descriptive whole-session upper bound;
- output-only, orthogonal, low-rank, and bounded supervised recalibration;
- shuffled target labels for every supervised method;
- identity-aware recording emulation compared with matched-random degradation;
- survivor-electrode analysis;
- later-session learning curves;
- influence of missing impedance and yield measurements;
- leave-one-development-participant and leave-one-schedule checks; and
- a negative control showing that correlation without reproduction is not
  enough.

## Development, stopping, and final evaluation

The first stage determines whether the nine participants contain comparable
target schedules and enough near- and long-gap pairs. Participant roles are
then set without using neural scores. Six people are used for development and
three for the final held-out evaluation.

Before real neural comparisons, synthetic examples must demonstrate correct
block splitting, score direction, missing-channel handling, denominator
behaviour, known mismatch and signal-loss cases, and all leakage controls.
Synthetic examples may reject an infeasible threshold but cannot choose a more
favourable scientific threshold.

Development begins with **20 complete coverage panels** spanning every
recalibration family, both recording-state emulators, the survivor-electrode
analysis, the second neural feature, and the major cross-panel controls. The
full search requires at least four development participants with sufficient
data support and **40–96 valid panels**, at least **two genuine follow-up
cycles**, at least **two recorded choices between the current best panel and a
new proposal**, and at least **40% tests designed to challenge the result,
ablations, influence checks, or direct controls** after coverage. Patience is
16 eligible panels after the minimum evidence is complete.

Resource limits are **1,200 CPU core-hours**, **168 wall-clock hours**, no GPU
time, at most **16 CPU cores**, **96 GiB memory**, and **8 hours per panel**.
Reaching a resource limit is not scientific success or scientific failure.

One complete diagnostic panel is chosen using reliability, leave-one-person
stability, survival of the required challenge tests, and then lower complexity
and runtime. It is not
chosen because it produces a preferred biological explanation. The three
held-out participants are then evaluated together once. Their results cannot
change the methods, thresholds, or selected panel.

## Relation to earlier work

Prior BrainGate work already shows that neural recordings and decoder
performance change over time, that calibration can help, and that latent
alignment sometimes improves stability. Those observations are not the new
claim.

The possible contribution here is a common task-, model-, data-budget-, and
participant-matched panel that distinguishes three operational explanations:
less locally recoverable signal, sufficiency of observable recording changes,
and low-budget remappability.

## Claim boundary

A positive result supports only an operational statement about these released,
preprocessed recordings under the specified direction proxy, feature window,
model family, and participant set. It cannot establish:

- how the historical online decoder would have performed;
- what would have happened in closed-loop use with a different decoder;
- hardware failure, neuron loss, or representational drift as a cause;
- additive causal percentages when several signatures coexist;
- conscious intention or disappearance of movement information;
- stable neuron identity behind a stable electrode number;
- population prevalence from three held-out participants;
- clinical readiness or deployment benefit; or
- independent confirmation outside this publication-exposed release.

## Current status

The scientific design is specified, but no EP16 experiment has started and no
held-out neural result has been examined. The remaining pre-run work is to
complete the target-schedule support check, choose the six development and
three held-out participants, set the still-missing numeric thresholds, and
test the analysis on synthetic cases.

Dataset details are in [DATASETS.md](DATASETS.md). Exact compact methods and
resource rules are in [SEARCH_POLICY.yaml](SEARCH_POLICY.yaml). The proposed
paper story is in [outputs/paper_plan.md](outputs/paper_plan.md).
