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
