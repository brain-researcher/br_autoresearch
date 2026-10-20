# EP24 arm comparison, attempt003 — screening result and descriptive look

Recorded 2026-10-17. [Figure and three lines](../../../figures/arm_comparison_attempt003/3lines.md) ·
[descriptive numbers](descriptive.json) · [per-item means](per_item.csv) ·
[screening means](../../attempt003/retained.json). Codex (gpt-6-astra, max)
reviewed this write-up against the numbers. It found no numerical errors, and
its wording corrections are applied.

The arms are the old instructions, the old instructions plus the three result
rules, and the old instructions plus an explicit directive to test the
strongest named alternative.

## Screening (job 48279121)

Codex (gpt-6-astra, max) built the candidate pool from the README rows and up
to three log resume points per episode at `960ab41`. It tagged each named
alternative as testable or claim-scope, and every quote was verified verbatim.
Fourteen candidates in 9 episodes had a testable alternative. Each got two
old-arm and two directive draws, scored by Claude Opus 5.5 and gpt-6-astra
(max) with arm labels withheld. The retention rule kept items with old R1 ≤
0.25 and directive R1 ≥ 0.75.

| Screening means (rater mean over 2 draws) | Items |
| --- | --- |
| Old R1 = 1 and directive R1 = 1 | EP06-R, EP06-L1, EP08-L3, EP14-L1, EP18-R, EP22-R |
| Old R1 = 0 and directive R1 = 0 | EP05-L3, EP07-R, EP14-R |
| Retained: old R1 = 0 and directive R1 ≥ 0.75 | EP01-L3, EP05-L1, EP05-L2, EP08-L2, EP15-R (directive 0.75; the others 1.0) |

The design required 10 retained items in 7 episodes. Only 5 items in 4
episodes were retained, so the frozen test did not run. Of the 11 items that
met the directive screening threshold, 6 already had old R1 = 1.

## Descriptive look at the retained items (job 48293879)

At the scientist's request, the 5 retained items received fresh draws: 3 per
item per arm, 45 proposals, rated by both raters. The figures below are
effect sizes only, with no test, interval or decision.

| Mean over items | Old | Old + three result rules | Old + directive |
| --- | --- | --- | --- |
| R1: tests a testable named alternative | 0.00 | 0.13 | 0.90 |

| Difference from old, episode-weighted (4 episodes unless noted) | Rules | Directive |
| --- | --- | --- |
| R1: tests a testable alternative | +0.125 | +0.875 |
| R2: an outcome would weaken a keyed explanation | +0.167 | +0.875 |
| R5: result and weakening outcomes named in advance (3 items with paired scores, 3 episodes) | +0.889 | +1.000 |
| R6: positive control before reading a null (2 null items, 1 episode) | +0.750 | 0.000 |
| R4: estimated cost (0–3) | +0.083 | +0.417 |

- **Per item.** The rules' R1 difference was +0.33 on EP05-L1 and EP08-L2 and
  0 on the other three. The directive's was +1.0 on four items and +0.5 on
  EP15-R.
- **Per rater.** Both raters give +0.125 for the rules. For the directive,
  Claude gives +0.917 and astra +0.833.
- **Boundary failures.** Rater-mean failures were 0.0 for old, 1.0 for the
  rules and 2.0 for directive proposals, out of 15 proposals each.

The ratio of the two R1 differences, 0.125 / 0.875, is about one seventh. The
rules' item-mean R1 share was 13.3%.

## Reading across the three attempts (descriptive, formed after the results)

| Attempt | Rules − old (R1) | Directive − old (R1) |
| --- | --- | --- |
| 001: README rows | +0.13 | +0.43 |
| 002: log resume points | +0.12 | +0.48 |
| 003: gap items, descriptive | +0.125 | +0.875 |

The rules' R1 difference is about a quarter to a third of the directive's in
attempts 001–002, and about one seventh here. In every attempt, rule-arm
proposals scored higher on the framing measures (R5, and R6 where null items
existed) than on R1. These are observations about written proposals from one
proposer, scored by model raters only, on few episodes. They are not tests,
and the attempts share episodes.
