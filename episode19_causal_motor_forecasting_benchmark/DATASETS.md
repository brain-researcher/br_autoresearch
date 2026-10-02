# Data used in Episode 19

EP19 combines two EEG movement datasets for different scientific purposes.
WAY-EEG-GAL supplies the main forecasting and model-order comparison. A second
self-paced reaching dataset tests whether the EEG increment repeats in new
people on a different task. Historical Kaggle material provides an
implementation check only, and AJILE12 is outside the primary study.

No held-out signal has been examined for EP19.

## Sources and roles

| Source | Scientific role | Current state |
| --- | --- | --- |
| WAY-EEG-GAL Figshare collection v2 | Main development, model-order comparison, and held-out series evaluation | Acquired but not prepared as episode-specific data views |
| Self-paced/free-choice EEG reaching Figshare v1 | Fifteen-person development and eight-person replication | Acquired but not prepared as episode-specific data views |
| Kaggle Grasp-and-Lift EEG Detection | Historical task and implementation compatibility only | No scientific outcome role |
| AJILE12 DANDI release | Deferred exploratory ECoG extension | Not eligible for the primary claim |

## WAY-EEG-GAL

### Source facts

| Item | Value |
| --- | --- |
| Dataset | WAY-EEG-GAL multimodal grasp-and-lift recordings |
| Figshare collection | `10.6084/m9.figshare.c.988376.v2` |
| Participant records | 12, each currently version 1 |
| Participant archive size | 10,339,259,047 bytes total |
| Dataset article | `10.1038/sdata.2014.47` |
| Official utilities | `https://github.com/luciw/way-eeg-gal-utilities` |
| Conservative reuse rule | CC BY 4.0 attribution |

The dataset article reports CC BY 4.0, whereas current child records may show a
different public license. EP19 follows the more conservative CC BY 4.0 rule.

The expected continuous source contains:

- 108 files covering 12 participants and series 1–9;
- 12 participant event/metadata files;
- 32 EEG channels sampled at 500 Hz;
- five EMG channels sampled at 4 kHz;
- three-dimensional hand, wrist, and object kinematics at 500 Hz;
- force-related signals at 500 Hz; and
- cue state and event timing.

Exact clocks, units, missing channels, and alignment must be checked directly
before analysis. Third-party filtered arrays, notebook caches, or previously
normalized tensors cannot substitute for the continuous source.

### Series roles

Every WAY participant follows the same time order:

| Series | Role |
| --- | --- |
| 1–6 | Participant-specific fitting, inner selection, and timing checks |
| 7 | Rate-limited aggregate development feedback |
| 8 | No-feedback held-out series |
| 9 | Additional public held-out series, evaluated together with series 8 |

After development, each participant's final weights may be refit on that
person's series 1–7. Series 8 and 9 are then evaluated together with no partial
result returned. The public status of these series makes them campaign-held-out
rather than globally untouched data.

The scientific models use this common 29-channel intersection:

`Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, C4, T8, TP9,
CP5, CP1, CP2, CP6, TP10, P7, P3, Pz, P4, P8, O1, Oz, O2`.

The historical compatibility analysis may use the original 32-channel montage,
but it cannot be mixed into the scientific 29-channel comparison.

### Onset and past-only history

The primary onset is rebuilt from the raw non-EEG sensors. Provider
`HandStart` is an agreement check, not the primary onset. EEG cannot define the
event it is asked to forecast.

The detector must specify, before candidate EEG scoring:

- the sensor hierarchy and combination rule;
- one-sided filters and their group delays;
- baseline, threshold, dwell, and confirmation rules;
- cross-sensor clock reconciliation;
- how ambiguous or missing onsets are excluded; and
- a worst-case timing bound smaller than 300 ms.

Object motion, grip force, and load force are consistency checks rather than
alternative outcomes chosen because they favour a model.

The [component design](outputs/component_details.md) selects the earliest
independently confirmed EMG-or-hand-motion crossing and an online armed/pending
state. All primary sensor paths remain required. Published `HandStart` and
the provider's synchronization figure are agreement/context information, not
proof that the EP19 detector meets its full timing bound.
[WAY acquisition and event extraction](https://pmc.ncbi.nlm.nih.gov/articles/PMC4365902/).

## Self-paced/free-choice reaching

### Source facts

| Item | Value |
| --- | --- |
| Dataset DOI | `10.6084/m9.figshare.28632599.v1` |
| Dataset article | `10.1038/s41597-025-06039-9` |
| File | `Freewill_EEG_Reaching_Grasping.zip` |
| Figshare file ID | `57518986` |
| Archive size | 13,591,548,048 bytes |
| License | CC BY 4.0 |

The release contains 23 people, 49 sessions, and 6,808 trials. It includes raw
continuous BrainVision EEG, four EOG channels, audio and trigger channels, and
three-axis wrist accelerometry. Twenty-one participants were recorded at
250 Hz; `sub-13` and `sub-15` were recorded at 1,000 Hz.

All 49 session headers contain the same first 31 EEG channels. The six sessions
from `sub-13` and `sub-15` include extra numbered empty physical channels, not
additional EEG signals. The archive contains 240 raw EEG runs, while the paper
reports 238 valid runs; eligibility must therefore follow the event and
valid-run information rather than filename count.

The scientific analysis uses the same 29-channel intersection listed for WAY.
Resampling and channel order must be set before held-out evaluation, and no
interpolation may use future samples.

### Participant roles

Fifteen people develop the second-task procedure and group encoder:

`sub-02, sub-03, sub-04, sub-06, sub-08, sub-09, sub-11, sub-12, sub-14,
sub-15, sub-17, sub-18, sub-20, sub-21, sub-22`.

Eight whole participants are held out:

`sub-01, sub-05, sub-07, sub-10, sub-13, sub-16, sub-19, sub-23`.

All sessions, runs, EEG, EOG, triggers, and accelerometry from a participant
stay on the same side. A person is never replaced because of their result.

During development, the 15 people use one five-fold whole-participant split.
The final group model is fit on all 15 only after the model recipe has been
chosen. The eight held-out participants cannot enter feature learning,
self-supervised training, model selection, onset tuning, or normalization-state
initialization.

### What the second task can establish

The self-paced source has wrist accelerometry but no EMG. Its onset is rebuilt
from raw three-axis accelerometry with the same past-only principles used for
WAY. A successful result supports forecasting before accelerometer-detectable
wrist motion. It cannot establish forecasting before any measured muscle
activation.

The primary held-out condition uses each participant's first 32 complete
eligible movements in chronological order for four-parameter calibration. The
whole prefix and a sufficient history guard are excluded from scoring. The
remaining suffix is shared by the zero-label and 32-label comparisons.

The four parameters are one positive temperature and three coarse-horizon
offsets for 0–300, 300–600, and 600–900 ms, with `>900 ms` as reference. The
encoder and spatial projection remain unchanged. A full local refit on the
first half of complete runs is descriptive only.

Provider onset and cue annotations use zero-phase processing/manual review;
rebuild the primary onset and cue availability from raw accelerometry/TRIG.
The provider annotation remains an agreement check. Initial detector choices
and the complete calibration recipe are in the [component note](outputs/component_details.md).
The calibration packet is the whole chronological prefix, including eligible
stillness; no suffix or guard label can enter its censoring-aware likelihood.
[Self-paced acquisition and processing](https://www.nature.com/articles/s41597-025-06039-9).

## Data needed for the explanation and warning analyses

The source descriptions indicate that both main releases contain the pieces
needed for the proposed follow-up: continuous raw EEG, event timing, and enough
stream history to reconstruct past-only inputs. WAY also contains the
peripheral streams needed for its onset and risk set; the self-paced release
contains wrist accelerometry for the corresponding role. These facts make the
analyses plausible, but the episode-specific data views have not yet confirmed
their usable coverage.

The recent-by-older-slow prediction requires, for every scored decision point:

- at least 200 ms of valid EEG immediately before the decision and enough
  earlier history for the unchanged model input;
- raw or losslessly represented continuous EEG from which three fixed blocks
  can be constructed: an older complementary background, final-200-ms raw or
  broadband EEG, and causal 0.5–8 Hz EEG ending before that recent segment;
- acquisition-gap and series or run boundaries, so no factorial condition carries
  state across a discontinuity; and
- the same cue, context, peripheral history, onset label, and risk-set status
  for all four factorial conditions.

One causal filter bank, including its delay, warm-up, gap reset, and boundary
rule, is fixed on development data. Older blocks use raw samples strictly
before `t - 200 ms`; the recent block uses `[t - 200 ms, t)`, so the cutoff
sample belongs only to the recent block. No feature window may cross that
boundary. The [exact construction](outputs/component_details.md) sets O to the
causal filtered older component and C=x−O, then reconstructs each masked
voltage slot before family features/state. A shared training scale preserves
Q11 voltage reconstruction. C may retain slow information; constructed-O
mutation with C fixed is a representation intervention, not raw-band removal.
Q00 still contains complementary EEG. All four conditions use the same tensor
shape and development-frozen scaling, with an absent block replaced by fixed zeros and
no missingness flag. In every model family, the factorized full input must first
reproduce its original full-reference result within a pre-set equivalence margin.

Those equivalence calls use participant-bootstrap, two-sided 95% simultaneous
intervals rather than point estimates. A single max-statistic family covers
both held-out sources, all five families, both horizons, the Q11-minus-original
differences, and the factorial interactions. The full-reference and
interaction margins are fixed on development data before this family is read.

This representation needs an explicit state audit. A recent-absent model may
carry state formed before the cutoff, but its filters, window summaries,
normalizers, convolutions, and recurrent updates may not depend on final-200-ms
raw samples. Arbitrarily mutating that segment must leave both recent-absent
predictions unchanged. Mutating only the constructed older-slow block must
likewise leave both older-slow-absent predictions unchanged. The episode data
view must retain enough intermediate state to run these tests, not only final
feature matrices.

The event-warning analysis additionally requires continuous eligible-stillness
duration, complete movement onsets, and enough negative time to estimate false
alarms per hour. Development data choose one threshold, persistence rule,
refractory period, and numerical false-alarm ceiling. Held-out data may only
report detection, false alarms, and first-warning lead time under that frozen
rule.

The factorial prediction is developed using WAY series 1–7 and the 15
self-paced development participants. It is then tested without modification in
WAY series 8–9 and in the eight whole held-out self-paced participants. If raw
coverage, pre-onset history, or negative at-risk duration is inadequate in a
source, retain the primary log-score analysis but mark the affected explanation
or warning result unavailable; do not substitute another band, cutoff, or
false-alarm rule after outcomes are seen.

## Historical compatibility

The Kaggle task used participant series 1–8 for training, series 9–10 for test,
six event labels expanded approximately 150 ms on either side of event times,
and mean column-wise event AUROC.
The public WAY multimodal source contains only series 1–9; the private series 10
does not provide the full public multimodal record needed by EP19.

The public winning repository is
`https://github.com/alexandrebarachant/Grasp-and-lift-EEG-challenge`. Its compact
`Safe1` analysis is the preferred positive control. Any native Python 2.7 and
Theano execution must be distinguished from a modern semantic port.

Before the scientific final evaluation, EP19 may run a series-1–6 to series-7
compatibility check and report `HandStart` separately. A series-1–8 to series-9
historical comparison may run only afterward and has no role in the prospective
EEG conclusion. EP19 cannot claim to reproduce the private Kaggle leaderboard
without the original entrants' complete code, features, weights, and
predictions.

## Deferred AJILE12 extension

AJILE12 published version `10.48324/dandi.000055/0.220127.0436` remains an
exploratory ECoG extension. The available ECoG and pose-event products do not
currently establish the raw filtering and timing information required for the
primary strict-streaming claim. AJILE12 cannot replace either main source if
one fails.

## Data separation during the study

Development analyses receive WAY series 1–7 and the 15 self-paced development
participants. WAY series 8–9 and all data from the eight self-paced held-out
participants are used only after the onset rules, models, calibration procedure,
scores, margins, and controls have been set.

The two held-out components are evaluated together. Development receives no
participant-level, fold-level, surrogate-specific, file-order, calibration, or
partial quality information from that evaluation. A held-out result cannot
trigger a replacement model, participant, onset rule, or second evaluation.

## Current readiness

The named archives have been acquired but remain unprepared for EP19 analysis.
Before signal scoring, the study still needs:

- episode-specific development and held-out data views;
- direct clock, channel, sampling, gap, and series-boundary checks;
- executable/frozen onset and stillness rules from the selected component design;
- the five executable reference model recipes;
- source-specific non-neural baselines;
- development validation and freezing of the selected four-parameter recipe;
- confirmation of valid recent and older EEG history at every scored decision
  point, including a boundary that no derived feature crosses;
- the source-compatible causal factorization into older complementary,
  final-200-ms, and older 0.5–8 Hz blocks, its full-reference equivalence check,
  and both block-mutation tests;
- eligible-stillness exposure for participant-level false-alarm rates;
- a development-fixed warning threshold, persistence rule, refractory period,
  and numerical false-alarm ceiling;
- a finite set of allowed model changes and all numeric margins;
- the five-fold assignment for the 15 self-paced development participants; and
- synthetic tests that expose future-information leakage and recover a known
  prospective EEG signal.

No scientific forecast, model ordering, or second-task replication has yet
been evaluated.
