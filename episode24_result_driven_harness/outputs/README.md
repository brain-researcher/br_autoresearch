# EP24 outputs

October 16, 2026: the scientist adopted a revised design. A blinded, paired
comparison tests whether the three result rules adopted that day change the
next action agents propose at 23 README decision points. The scientist
approved the [key](arm_comparison/key.md), and the materials in
[arm_comparison/](arm_comparison/) are frozen.

**Attempt001 result: underidentified.** Two model raters scored 168 proposals.
On 10 primary items the rules raised targeting by +0.13 (95% interval −0.04 to
+0.33, p = 0.19). The directive control reached +0.43 but p = 0.0625: old
proposals were already at ceiling on 5 of the 10 items. Secondary framing
measures rose: R2 +0.33 (p = 0.008), R5 and R6. Read the
[three lines and figure](figures/arm_comparison_attempt001/3lines.md) and the
[report](arm_comparison/results/attempt001/report.md). No human rating
validates the scores.

October 13, 2026: the complete 636,747,571-byte session prefix has been parsed
on compute, and a focused harness instruction change is applied. Prospective
behavior and real-result validation remain open; EP24 is not complete.

October 14: conversation exports were removed. Only aggregate history counts
and analyst-written case summaries are retained; source-line references point
to the original private session outside this repository.

Read [GOAL.md](../GOAL.md), [DATASETS.md](../DATASETS.md), and
[SEARCH_POLICY.yaml](../SEARCH_POLICY.yaml) for the question, source boundary,
authority and acceptance criteria.

Start with the [baseline findings and limits](history_baseline_report.md),
[exact counts](runs/history_baseline/summary.json),
[execution log](experiment_log.md), and [experiment program](astra/v0.0.14/astra.yaml).
The [next-result objective](next_result_goal.md) is prepared for the original
coordinator; it has not been silently installed in another thread.

**October 17, attempt002 result: underidentified.** These items are 7 resume
points from episode logs; the floor was lowered from 8 to 7 after screening.
Old proposals never tested the alternative there. The rules raised targeting
by +0.12 (p = 0.5); the directive control raised it by +0.48 (p = 0.0625, the
minimum possible for its 5 informative items). Read the
[three lines and figure](figures/arm_comparison_attempt002/3lines.md) and the
[report](arm_comparison/results/attempt002/report.md).

Across both attempts, the rules changed how proposals are framed more than
which action they propose. Neither attempt can confirm or exclude an effect on
targeting.

**October 17, attempt003 stopped at screening.** Alternatives were keyed only
when testable, leaving 14 candidates.
- On 6, old and directive screening proposals both scored R1 = 1.
- On 3, both scored 0.
- Only 5 gap items in 4 episodes were retained, below the required 10 in 7.

Of the 11 items that met the directive threshold, 6 already had old R1 = 1,
and the retained gap set is too small to power the comparison. See the
[log](experiment_log.md) and [retained.json](arm_comparison/attempt003/retained.json).

**October 17, descriptive look at the 5 gap items.** This is effect sizes only.
On items where the directive makes proposals test the alternative 90% of the
time, the rules did so 13% of the time, against 0% under the old instructions.
Episode-weighted, that is +0.125 for the rules against +0.875 for the
directive. The rules raised advance naming and positive-control requirements
much more than targeting. Read the
[three lines and figure](figures/arm_comparison_attempt003/3lines.md) and the
[report](arm_comparison/results/attempt003/report.md).

Next: no further run is planned without a scientist decision.
Observations of real episodes, including EP21's continuation, are descriptive
context only, as are the October 13 next-result objective and its EP21 focus.
