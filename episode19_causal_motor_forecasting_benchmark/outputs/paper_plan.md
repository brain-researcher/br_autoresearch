# EP19 paper plan

[Current conceptual figure](ep19_question_imagegen-v2.png) · [Exact image-generation record](ep19_question_imagegen-v2-prompt.md).

## Paper question

Does EEG contain prospective information about movement onset 300–600 ms in
the future after accounting for cue, task context, and past peripheral signals?
If it does, do model advantages seen near onset survive at that earlier horizon,
and does the EEG increment repeat in new participants on a second task? The
explanatory question is whether near-onset gain depends more on a final-200-ms
input block while the 300–600 ms increment depends more on a non-overlapping,
earlier causal 0.5–8 Hz block.

The paper should begin with the difference between detecting a movement already
starting and forecasting one that remains in the future. Historical competition
compatibility and the bounded new-model search belong in methods or appendices,
not in the headline.

The [component design](component_details.md) supplies the selected onset/risk
state, exact EEG construction, source-specific baseline inputs and provisional
calibration recipe. These are design choices, not tested implementations.

## Prior work and the specific gap

| Prior work | Already established | What EP19 would need to add |
| --- | --- | --- |
| [Prediction of human voluntary movement before it occurs](https://pmc.ncbi.nlm.nih.gov/articles/PMC5558611/) | EEG can support prediction before measured movement onset. | Establish the specified 300–600 ms increment beyond available past context/peripheral signals on a continuous natural-risk sequence. |
| [Crell et al. (2025)](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2025.1540155/full) | Cued-to-self-paced asynchronous EEG detection and false-alarm operating points are already studied. | Repeat the sensor-conditioned forecast and registered input-support contrasts in held-out participants under a second-task procedure. |

The gap is this conjunction, not first premovement prediction, slow EEG or
cross-task testing. Model-family ranking and new-architecture search remain
secondary. The second source repeats the procedure with its own development-
trained encoder; it does not test zero-shot transfer of WAY weights.

## Figure 1 — Why detection is not necessarily forecasting

**Scientific judgment:** the analysis defines a genuinely prospective target
and excludes future information from preprocessing and model state.

**Panels:**

- a timeline showing decision time, 0–300, 300–600, and 600–900 ms horizons;
- the 19-category onset distribution at 50-ms resolution;
- sensor crossing, suspended decisions and later confirmation, alongside the
  full timing uncertainty bound and causal cue availability; and
- examples of valid one-sided filtering versus invalid centered filtering,
  future padding, whole-session normalization, and state carried across gaps.

**Decision:** future-input mutation, future-impulse, streaming consistency,
state-reset, timing, and deliberately leaked-signal checks must all pass.

**If a check fails:** there is no scientific forecast result. The failure is a
timing or information-flow problem, not evidence for or against premovement EEG.

## Figure 2 — Does real EEG add information at 300–600 ms?

**Scientific judgment:** real EEG improves the strict-horizon forecast beyond
the source-specific non-neural model, a zero-EEG model, and the strongest of
eight capacity-matched surrogate-EEG models.

**Panels:**

- absolute forecast score for context and peripheral history alone;
- real-minus-non-neural, real-minus-zero, and real-minus-surrogate increments
  at all three horizons;
- participant-level effects and leave-one-participant influence; and
- finite, normalized probability and longer-horizon checks.

Show which peripheral/context inputs each source actually provides. Select
its baseline before candidate EEG scores; the second task repeats this
procedure with development-group weights, not transferred WAY weights.

**Decision:** all three primary-horizon increments must clear their own pre-set
margins. Probability validity, participant robustness, and 600–900 ms
non-degradation must also pass. Calibration intercept, calibration slope,
reliability curves, and Brier score remain secondary summaries.

**If only 0–300 ms passes:** call the result near-event recognition, not strict
forecasting.

**If absolute performance is good but surrogate EEG matches it:** do not make a
neural-information claim.

## Figure 3 — What separates a near-event signal from a strict forecast?

**Scientific judgment:** test one signed prediction using non-overlapping recent
and older-slow input blocks instead of using receptive-field plots as a post
hoc explanation.

**Panels:**

- a factorized past-only input: older complementary EEG using samples strictly
  before `t - 200 ms`, raw/broadband EEG on `[t - 200 ms, t)`, and causal
  0.5–8 Hz EEG using raw support before that final segment;
- the four fixed-shape conditions with recent and older-slow blocks present or
  absent, including their interaction;
- the recent and older-slow contributions at 0–300 and 300–600 ms;
- all ten model-family edges in one predeclared simultaneous family; and
- WAY series 8–9 and all eight held-out self-paced participants, with the six-
  of-eight directional rule shown explicitly.

Show the exact older construction `O=causal_filter(x)`, `C=x-O` and shared
voltage scaling before family features. Q11 reconstructs voltage but still
needs model equivalence; Q00 retains complementary EEG and is not zero-EEG.
Keep the adapters within the original family allowances. C can retain slow
information, so O's effect is conditional on C, not removal of all slow EEG.

**Prediction:** average each block's factorial contribution over the state of
the other block. The recent contribution is positive and larger at 0–300 ms;
the older-slow contribution is positive and larger at 300–600 ms. Simultaneous
one-sided lower bounds for those four statements must exceed zero. A separable
claim also requires the factorial interaction to remain inside its pre-set
equivalence margin.

Both equivalence decisions use two-sided 95% participant-bootstrap intervals,
not point estimates. One max-statistic family covers both held-out sources, all
five model families, both horizons, every Q11-minus-original difference, and
every factorial interaction. The full-reference and interaction margins are
fixed on development data; each relevant interval must fit entirely inside
its margin.

**Competing explanation:** the same input block drives both horizons, a large
interaction makes the two-block story inseparable, or recent samples leak into
filter, normalization, covariance, convolutional, or recurrent state.

**Decision:** first require each family's factorized full input to match that
family's original full reference within an equivalence margin at both horizons.
Then train all four
conditions from the beginning for all five families with identical tensor
shape, parameter count, optimization, and seed bank. Mutating final-200-ms raw
EEG must not change either recent-absent prediction. All four contrasts need
positive simultaneous lower bounds in WAY and the self-paced macro result, and
the same at least six of eight held-out people must have all four positive
signs. Failure leaves the primary log-score conclusion unchanged.

Also show constructed-O mutation invariance for older-slow-absent models;
this is a representation intervention with C fixed, not a raw-band mutation.

For model ordering, orient all ten edges by the fixed reference-family order.
Before held-out scoring, development data assign each edge a signed recent or
older-slow prediction, or no explanation. All edge-by-block contrasts share a
simultaneous inference family. A changed edge without its predeclared,
repeated component contrast is documented but not explained.

## Figure 4 — Does the model ordering change with prediction horizon?

**Scientific judgment:** determine whether conclusions among five pre-specified
model families are retained, collapsed, emergent, reversed, or uncertain when
the endpoint moves from 0–300 to 300–600 ms.

**Panels:**

- a pairwise model-comparison matrix at each horizon;
- a transition diagram linking each near-event edge to its strict-horizon
  state;
- the prespecified recent-by-older-slow factorial result from Figure 3; and
- the one separately searched model shown outside the reference panel.

**Decision:** each reference family is fit once for all horizons, with equal
12-configuration and three-seed allowances. One simultaneous uncertainty rule
covers all pre-specified pairs.

**If order changes:** the paper's contribution is that benchmark conclusions
depend on how early prediction is required, not that one architecture is
universally best.

**If edges are uncertain:** retain the EEG increment conclusion, if supported,
but do not manufacture a model winner.

Architecture remains a supporting result. A changed edge matters scientifically
only if its development-fixed block prediction distinguishes why it changed; otherwise the
paper reports the ranking change without a mechanistic explanation.

## Figure 5 — Does the EEG increment repeat in new people?

**Scientific judgment:** test the pre-set forecast in eight held-out
participants performing self-paced reaching, with only the first 32 complete
movements used for four-parameter calibration.

**Panels:**

- fifteen-person development versus eight-person held-out design;
- no-label, 32-event calibration, and descriptive local-refit conditions;
- the three EEG increments for every held-out participant; and
- macro, 6-of-8, and leave-one-person robustness summaries;
- probability reliability, calibration intercept and slope, and Brier score for
  every participant; and
- event detection and first-warning lead time under one development-fixed
  threshold, persistence rule, refractory period, and false-alarm ceiling.

Depict the complete prefix ending at movement 32, including naturally occurring
eligible stillness, then the history guard and shared scored suffix. Annotate
identity fallback if the development-fixed four-parameter fit fails; never
drop a person/model or expose calibration quality separately from the joint
evaluation. No-label and calibrated models have identical suffix-state starts.

**Decision:** each neural comparison must be positive in at least six of eight
people, clear its participant-macro margin, and remain positive in every
leave-one-person mean. The encoder cannot change during the 32-event fit.

**If WAY passes but this figure fails:** report a dataset- or procedure-specific
forecast that did not repeat in the second task.

Probability reliability and event warnings are secondary. The held-out warning
analysis reports false alarms per hour of eligible stillness, fraction of
movements warned 300–600 ms before onset, and first-warning lead time. No
held-out operating-point adjustment is allowed, and these results cannot rescue
a failed primary log-score endpoint.

## Result-dependent next steps

| Result | Next scientific step |
| --- | --- |
| Strict EEG increment repeats | Test the pre-set forecast prospectively in an attended online study; do not infer mechanism from prediction alone |
| Strict increment and factorial block prediction both repeat | Test the recent-versus-older-slow dependence prospectively with a task or acquisition perturbation |
| Strict increment repeats but the factorial prediction fails | Preserve the forecasting result; do not use the proposed input-block explanation |
| Near-event only | Improve onset measurement and artifact controls rather than enlarging the model |
| Real EEG matches surrogates | Identify which temporal or spectral nuisance carries the score before trying another architecture |
| Model order changes | Build benchmarks that report forecast horizon explicitly instead of one pooled leaderboard |
| Model order is uncertain | Add participants or repeated sessions; do not select a family from point estimates |
| Second-task nonreplication | Examine task, sensor, and calibration differences in development data from a future study, without adapting to these eight outcomes |
| No new model improves | Retain the reference result; architecture search is secondary to the neural-information question |
| Timing or future-information failure | Repair and retest the measurement pipeline before any biological interpretation |

## Main claim boundary

The strongest EP19 paper can claim incremental prospective EEG information
under the stated sensors, onset detector, horizons, tasks, and calibration
budget. If the factorial prediction repeats, it may additionally claim that the
tested models depend more on recent input near onset and on older slow input at
the strict horizon under the prespecified factorization. It cannot identify two
physiological generators, and it still cannot claim
causal motor preparation, conscious intention, the earliest biological command,
universal model superiority, zero-shot transfer, online BCI utility, or clinical
benefit.

A null means the tested models did not establish the specified increment
beyond the measured sensor/context history, not that EEG contains no motor
information. Lack of significance is not equivalence. A positive increment
also cannot prove cortical origin or exclude an unmeasured peripheral change.
