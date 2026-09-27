# EP19 paper plan

## Paper question

Does EEG contain prospective information about movement onset 300–600 ms in
the future after accounting for cue, task context, and past peripheral signals?
If it does, do model advantages seen near onset survive at that earlier horizon,
and does the EEG increment repeat in new participants on a second task?

The paper should begin with the difference between detecting a movement already
starting and forecasting one that remains in the future. Historical competition
compatibility and the bounded new-model search belong in methods or appendices,
not in the headline.

## Figure 1 — Why detection is not necessarily forecasting

**Scientific judgment:** the analysis defines a genuinely prospective target
and excludes future information from preprocessing and model state.

**Panels:**

- a timeline showing decision time, 0–300, 300–600, and 600–900 ms horizons;
- the 19-category onset distribution at 50-ms resolution;
- sensor-derived onset and its uncertainty bound; and
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
- calibration and longer-horizon checks.

**Decision:** all three primary-horizon increments must clear their own pre-set
margins. Calibration, participant robustness, and 600–900 ms non-degradation
must also pass.

**If only 0–300 ms passes:** call the result near-event recognition, not strict
forecasting.

**If absolute performance is good but surrogate EEG matches it:** do not make a
neural-information claim.

## Figure 3 — Does the model ordering change with prediction horizon?

**Scientific judgment:** determine whether conclusions among five pre-specified
model families are retained, collapsed, emergent, reversed, or uncertain when
the endpoint moves from 0–300 to 300–600 ms.

**Panels:**

- a pairwise model-comparison matrix at each horizon;
- a transition diagram linking each near-event edge to its strict-horizon
  state;
- representation-transfer and receptive-field diagnostics; and
- the one separately searched model shown outside the reference panel.

**Decision:** each reference family is fit once for all horizons, with equal
12-configuration and three-seed allowances. One simultaneous uncertainty rule
covers all pre-specified pairs.

**If order changes:** the paper's contribution is that benchmark conclusions
depend on how early prediction is required, not that one architecture is
universally best.

**If edges are uncertain:** retain the EEG increment conclusion, if supported,
but do not manufacture a model winner.

## Figure 4 — Does the EEG increment repeat in new people?

**Scientific judgment:** test the pre-set forecast in eight held-out
participants performing self-paced reaching, with only the first 32 complete
movements used for four-parameter calibration.

**Panels:**

- fifteen-person development versus eight-person held-out design;
- no-label, 32-event calibration, and descriptive local-refit conditions;
- the three EEG increments for every held-out participant; and
- macro, 6-of-8, and leave-one-person robustness summaries.

**Decision:** each neural comparison must be positive in at least six of eight
people, clear its participant-macro margin, and remain positive in every
leave-one-person mean. The encoder cannot change during the 32-event fit.

**If WAY passes but this figure fails:** report a dataset- or procedure-specific
forecast that did not repeat in the second task.

## Result-dependent next steps

| Result | Next scientific step |
| --- | --- |
| Strict EEG increment repeats | Test the pre-set forecast prospectively in an attended online study; do not infer mechanism from prediction alone |
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
budget. It cannot claim causal motor preparation, conscious intention, the
earliest biological command, universal model superiority, zero-shot transfer,
online BCI utility, or clinical benefit.
