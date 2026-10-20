# EP24 arm comparison, attempt001 — result

Analyzed 2026-10-16 with the prespecified analysis
([arm_comparison_analysis.py](../../../tools/arm_comparison_analysis.py)) and
the rating amendment adopted before unsealing (GOAL decision 6).
[Figure and three lines](../../../figures/arm_comparison_attempt001/3lines.md) ·
[all numbers](analysis.json) · [per-item means](per_item.csv).

## What ran

- **Proposals.** 168 next-action proposals (job 48167798): 23 items × old and
  new arms × 3 draws, plus the directive control on the 10 type-A items. All
  were written by gpt-6-astra (effort ultra) in fresh tool-free sessions.
- **Ratings.** Claude Opus 5.5 (job 48180775) and gpt-6-astra at max effort
  (job 48185013) each scored all 168 against the frozen key and rubric, with
  arm labels withheld. No blinding check was run. Each proposal's score is the
  mean of the two raters.
- **Unsealing.** The scientist waived the blind 25% check, so no human rating
  validates these scores. The mapping was unsealed only by the analysis (srun
  job 48187790).

## Primary result and decision

| Comparison over 10 type-A items | Estimate | 95% interval | Exact p |
| --- | --- | --- | --- |
| Three result rules − old (R1) | +0.133 | −0.041 to +0.333 | 0.188 |
| Directive control − old (R1) | +0.433 | 0.000 to +0.833 | 0.0625 |

The frozen decision map evaluates the directive control first. It must reach
the 0.25 margin with p < 0.05. It reached the margin but not significance, so
the decision is **underidentified**: no conclusion about the rules' effect on
targeting follows.

The diagnosis points to the item set, not to the raters. The raters agree on
R1 (κ = 0.92, 96.7% agreement over 90 type-A proposals). Under the old
instructions, proposals already target a keyed alternative in every or nearly
every draw for EP06, EP17, EP18, EP22 and EP23; across all 10 items the
old-arm rate is 0.48. Paired differences on those five items are zero or
negative. That leaves 5–6 items that can move, and with 5 informative items
an exact two-sided sign-flip test cannot go below p = 0.0625. The planned
detectability assumed a 20% old-arm rate across all 10 items.

## Secondary measures (descriptive)

| New − old | Items | Estimate | Exact p |
| --- | --- | --- | --- |
| R2: an outcome would weaken one keyed explanation (type A) | 10 | +0.333 | 0.008 |
| R5: result and weakening outcomes named in advance | 13 | +0.474 | 0.008 |
| R6: known-positive control before reading a null | 11 | +0.909 | 0.002 |
| R4: estimated cost (0–3) | 23 | +0.029 | 0.83 |

- **Boundaries.** Rater-mean R3 failures are 5.0 for old, 3.5 for new and 5.0
  for directive proposals. New-arm failures fall on EP11, EP12, EP14 and EP23.
  Old proposals also failed on EP12 and EP23, so the new-only failures are on
  EP11 (0.5) and EP14 (1.0).
- **Rater agreement.** κ is 0.56 for R3 (93.5% agreement), below the 0.6 level
  used for R1.
- **Harm conditions** of the "retain" rule all hold. The surface-compliance
  pattern (R5 rising without R2) does not occur, because R2 rose too.
- **Each rater alone** gives the same decision: Claude +0.167 (p = 0.125) and
  astra +0.100 (p = 0.5) for the rules; +0.467 (p = 0.0625) and +0.400
  (p = 0.094) for the directive control.

## Competing accounts

| Account | Status after attempt001 |
| --- | --- |
| Rule effect on targeting | Not established; underidentified |
| Ceiling: the record drives the choice | Partly supported: old proposals target the alternative at ceiling in 5 of 10 items |
| Dilution: the rules go unheeded | Weakened for framing (R2, R5 and R6 rose), not resolved for targeting |
| Surface compliance | Weakened: R2 rose along with R5 |
| Over-application | Weakened: no rise in boundary failures or cost |

**Post hoc observation, formed after seeing results and not confirmatory.** On
the five items where old proposals were not at ceiling (EP01, EP07, EP08,
EP14, EP15), the rules raised R1 by +0.30 on average and the directive control
by +0.90.

## Claim limits

These are written proposals from one proposer model, prompt and short packet,
not executed actions or scientific outcomes. Scores come from two model
raters; the second shares the proposer's model family, and no human rating
validates them. The ten primary items were already examined by the review that
motivated the rules. R5 and R6 partly restate rules 1 and 3, so their rise is
largely compliance. R2 is the closer measure of decision quality.

## Next step under the decision map

The decision map asks to diagnose before concluding. Neither the raters nor
the rubric explains the failure; the item set does. An informative follow-up
needs decision points where old instructions do not already test the
alternative, for example packets that leave the alternative less explicit, or
more items. Either would change the item set and is the scientist's decision.
