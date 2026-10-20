# EP24 arm comparison, attempt002 — result

Analyzed 2026-10-17 with the same prespecified analysis as attempt001
([arm_comparison_analysis.py](../../../tools/arm_comparison_analysis.py)).
[Figure and three lines](../../../figures/arm_comparison_attempt002/3lines.md) ·
[all numbers](analysis.json) · [per-item means](per_item.csv). Codex
(gpt-6-astra, max) reviewed this write-up against the numbers; it found no
numerical errors, and its wording corrections are applied.

## What ran

- **Items.** Each candidate is a resume point: an episode's newest log entry at
  `960ab41` that reports a scientific result, names a surviving alternative or
  limitation, and states a next action of 300 words or fewer. Codex
  (gpt-6-astra, max) proposed the picks, and every heading, length and quote
  was verified against the log. Nine episodes qualified.
- **Screening** (job 48221066). Two old-arm draws per candidate. The rule
  dropped EP08 and EP14, where every screening proposal was scored as testing
  the alternative. Seven were retained: EP01, EP04, EP05, EP07, EP17, EP18 and
  EP23. The screening draws were not reused.
- **Floor.** The frozen floor was 8. The scientist lowered it to 7 after
  screening, before any attempt002 new- or directive-arm proposal existed.
- **Test** (job 48254701, 32m12s). 63 fresh proposals: 7 items × old, new and
  directive arms × 3 draws, written by gpt-6-astra (effort ultra). Claude Opus
  5.5 and gpt-6-astra (max) rated every proposal with arm labels withheld; each
  score is the mean of the two raters. The planned human blinding check was
  waived, so blinding itself is unverified.

## Primary result and decision

| Comparison over 7 items | Mean paired difference | 95% interval | Exact p | Items with a positive difference |
| --- | --- | --- | --- | --- |
| Three result rules − old (R1) | +0.119 | 0.000 to +0.333 | 0.50 | 2 (EP05, EP18) |
| Directive control − old (R1) | +0.476 | 0.000 to +1.000 | 0.0625 | 5 |

"Items with a positive difference" counts items whose model-rated item mean
was higher than the old arm's. It does not mean an executed change in action.

No old-arm proposal was scored as testing the keyed alternative (R1 = 0 on all
7 items), so attempt001's ceiling does not recur. The directive control
reached the 0.25 margin but not p < 0.05. All five of its nonzero items were
positive, and with five informative items an exact two-sided sign-flip test
cannot go below p = 0.0625. Under the frozen decision map the result is again
**underidentified**.

Under the directive, EP01 and EP07 still scored 0. One possible reason, to be
checked by reviewing those keys, is that their keyed alternatives are
claim-scope statements rather than testable explanations. EP01's reads "not a
mechanism, novelty or fresh confirmation"; EP07's reads "not held-out
evidence, a scientific margin test, final model selection or proof of
trial-specific coupling".

## Secondary measures (descriptive)

| New − old | Items | Mean difference | Exact p |
| --- | --- | --- | --- |
| R2: an outcome would weaken one keyed explanation | all 7 | +0.119 | 0.50 |
| R5: result and weakening outcomes named in advance | 5 where both arms proposed new analysis | +0.533 | 0.25 |
| R6: known-positive control before reading a null | the 2 null-flagged items | +1.000 | 0.50 |
| R4: estimated cost (0–3) | all 7 | +0.143 | 1.00 |

- **Boundaries.** Rater-mean R3 failures are 3.5 for old, 4.0 for new and 3.0
  for directive proposals, so the "no rise in boundary failures" harm
  condition fails; the decision does not turn on it. The +0.5 net change
  breaks down by item:
  - **EP18: +2.0.** One rule-arm proposal failed for both raters (1.0), and
    two were failed by astra alone (0.5 each). The dispute is whether
    proposing a calibration and positive-control diagnosis after the log
    declared the deliverable complete improperly reopens a stop.
  - **EP07: −1.5.**
  - **EP04: 0.**

  R3 agreement is κ = 0.60 (88.9%).
- **Rater agreement.** κ for R1 is 0.75 (92.1% over 63 proposals). Agreement
  is not validity, and no human rating checks these scores.
- **Each rater alone** gives the same decision: Claude +0.190 (p = 0.5) and
  astra +0.048 (p = 1.0) for the rules; +0.524 (p = 0.0625) and +0.429
  (p = 0.125) for the directive control.

## Both attempts together (descriptive)

| Attempt | Items | Old-arm R1 | Rules − old | Directive − old | Decision |
| --- | --- | --- | --- | --- | --- |
| 001: README rows | 10 | 0.48 | +0.13 (p = 0.19) | +0.43 (p = 0.0625) | underidentified |
| 002: log resume points | 7 | 0.00 | +0.12 (p = 0.50) | +0.48 (p = 0.0625) | underidentified |

In both attempts, the rules' mean difference on targeting (R1) was a quarter
to a third of the directive control's. Framing measures were higher in the
rule arm: R5 and R6 in both attempts (R6 on only 2 items in attempt002), and
R2 in attempt001. Neither attempt can
confirm or exclude an effect of the rules on targeting. This comparison across
attempts was formed after seeing both results, and the attempts share
episodes, so it is not an independent test.

## Claim limits

These are written proposals from one proposer model, prompt and short packet,
not executed actions or scientific outcomes. Two model raters score them and
no human rating validates them; the second rater shares the proposer's model
family. The items are past states of the same episodes the motivating review
examined. Retained items were selected on old-arm screening draws, and the
floor was lowered after screening; neither step saw a rules-arm or
directive-arm proposal.

## What a next attempt would need

Under the decision map an underidentified result calls for diagnosis, not a
conclusion. The working hypothesis is that too few decision points carry an
alternative that is both untested by default and testable by a next action.
The lower bound on informative items is fixed by the exact test. Five
same-direction items give at best p = 0.0625, six give p = 0.031, and about
ten would leave room for mixed directions. Ten items still do not guarantee
identification.

One possible attempt003 design adds a directive-arm screen that keeps only
items the directive can move. Its comparison would then use separate fresh
draws, and its result would describe directive-responsive decision points
only. Changing the item set again is the scientist's decision.
