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
- exact onset and stillness code for each source;
- the five executable reference model recipes;
- source-specific non-neural baselines;
- the complete four-parameter 32-event calibration procedure;
- a finite set of allowed model changes and all numeric margins;
- the five-fold assignment for the 15 self-paced development participants; and
- synthetic tests that expose future-information leakage and recover a known
  prospective EEG signal.

No scientific forecast, model ordering, or second-task replication has yet
been evaluated.
