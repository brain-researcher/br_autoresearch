# EP24 — Result-driven research harness

**Adopted 2026-10-16.** On 2026-10-16 the scientist adopted this revision
and the five decisions listed below. It supersedes the 2026-10-13 contract
(commit `960ab41`). The scientist approved the key the same day. The packets,
key, arm texts, prompts and run plan are frozen in
[`outputs/arm_comparison/`](outputs/arm_comparison/).

## Question

Do the three result rules adopted on 2026-10-16 change the next action an agent
proposes at a pending research decision? Given the same record, does the agent
more often propose to test the strongest surviving alternative that the record
already names, without more boundary violations or cost?

## Why this test

The 2026-10-13 work stands: a complete 160,721-record history baseline and an
applied patch for authority carryover and remaining-cost decisions. A
2026-10-16 portfolio review found two problems. Agents name each result's
strongest surviving alternative but rarely make it the next test, and the
2026-10-13 goal counted any estimate with an interval as a result. The same
day the scientist adopted three rules in `AGENTS.md`, `run-episode` and
`design-episode`. They were still uncommitted when this revision was written:

1. An estimate counts as a result only if its record named, before it ran, the
   explanations it separates and the outcome that would weaken each.
   Otherwise it is calibration or description.
2. Once a result names its strongest surviving alternative, the default next
   action tests that alternative.
3. A null or equivalence is interpreted only after the same pipeline has
   detected a known positive effect at the same margin.

A fourth rule adopted the same day says that missing source facts are not
stopping points. It governs a different decision, and the scientist excluded
it from every arm.

Watching later episodes under the new text would be an unmatched before/after
comparison, which cannot identify the rules' effect. This test holds the
record, model and prompt fixed and varies only the instruction text.

## Competing accounts

| Account | Predicts | Weakened by |
| --- | --- | --- |
| Rule effect | New-arm proposals target the named alternative much more often | A New − Old difference below the margin while the directive control clears it |
| Ceiling: the record, not the rules, drives the choice | Old-arm proposals already target the alternative when it is in view | A low Old-arm rate that leaves room for the margin |
| Dilution: ~40 added lines in ~40 KB of instructions go unheeded | Both arms rarely target it | A New − Old difference at or above the margin |
| Surface compliance | Advance naming (R5) rises; the action does not change | R1 and R2 rising with R5 |
| Over-application | Targeting rises with boundary failures, reopened stops, cost or weaker tests | No New-arm rise in R3 failures or cost, and R2 not lower |

## Design

**Arms.** *Old*: `AGENTS.md` and the three `SKILL.md` files as committed at
`960ab41` (HEAD on 2026-10-16), read with `git show`, never from the working
tree. *New*: Old plus only the three result-rule passages, applied as exact
hunks to `960ab41` at freeze. The amendment's new scenario in
`behavioral_evals.md` is left out of both arms. Agents load it only for skill
maintenance, and it nearly states the scoring target. *Directive control*: Old plus one task-prompt sentence
requiring a test of the strongest surviving alternative the summary names. This
is the known positive that rule 3 demands before EP24 can interpret its own
possible null.

**Items.** One per row of the root README "Current episode navigation" table,
EP01–EP23 as committed at `960ab41` (EP24 excluded). That gives 23 items: the
whole universe, not a sample. The commit pins rows that other sessions are
editing, and keeps out three rows already rewritten under the new rules. Each
packet holds the episode's GOAL title as its question, plus the row's two
cells, and is identical in every arm. Before any proposal exists, a key built
from the packet text alone quotes each named alternative and types the item.
Draft key:

| Type | Definition | Items |
| --- | --- | --- |
| A (primary) | Completed result with a named surviving alternative | EP01 (borderline), EP06, EP07, EP08, EP14, EP15, EP17, EP18, EP22, EP23 |
| B | No completed result; source or support facts pending | EP02, EP03, EP09, EP19 |
| C | Closed, valid stop, or next action fixed by an active job chain | EP04, EP05, EP10–EP13, EP16, EP20, EP21 |

Rows are not split, because items from one episode would be dependent. That
leaves about 10 primary items, below a 12–20 target, so precision comes from
three draws per arm instead. B and C items detect over-application.

**Proposals and scoring.** Each draw is a fresh, tool-free session of the
production proposer (Codex gpt-6-astra). It writes one next-action proposal of
at most 200 words from the packet and the arm text. Proposals are text only:
they are never executed or sent to episode owners. Raters blind to arm score
each proposal against the frozen key with the policy's six-item rubric. R1,
targeting a named surviving alternative, is primary. The other items cover
discrimination, boundaries, cost, advance naming and positive controls. Claude
Opus 5.5 scores every proposal. It runs outside the repository, so the live,
amended instructions do not load. The planned blind scientist check of a
stratified 25% was replaced by a second model rater (decision 6).

**Primary comparison.** Over type-A items, the mean paired difference
(New − Old) in the proportion of draws with R1 = 1. Test: exact two-sided
sign-flip test over items (exact McNemar at one draw). The 95% interval comes
from inverting that test, and the item (one per episode) is the independent
unit. Approximate analytic power for 10 primary items, with homogeneous rates
and α = 0.05:

| Draws per arm | 20% → 70% shift | 20% → 50% shift |
| --- | --- | --- |
| 3 | ≈ 0.85 | ≈ 0.4 |
| 1 | ≈ 0.36 | ≈ 0.1 |

A miss therefore rules out only large effects, and only if the directive
control clears the margin.

## Result and decision

Report the primary difference with its interval, R2–R6, the directive control,
rater agreement and arm-guess accuracy, as one figure (per-item paired rates)
and three lines. The scientist adopted this decision map with a 0.25 margin.

- **Retain**: difference ≥ margin with p < 0.05, no rise in R3 failures, mean
  R4 up by at most 0.5 level, and R2 not lower.
- **Revise**: targeting rises along with a harm, or only R5 rises.
- **No detectable effect**: difference below the margin while the directive
  control clears it. Revise the wording or placement, or revert.
- **Underidentified**: the directive control fails. Diagnose the key, rubric
  or raters before concluding anything.

## Decisions adopted 2026-10-16

1. **Revision adopted.** It replaces the 2026-10-13 prospective-validation
   acceptance (a new real-data result plus unmatched process evidence). Later
   real-episode observations stay descriptive.
2. **New arm: the three result rules only.** The fourth 2026-10-16 rule
   (missing source facts are not stopping points) is excluded from every arm,
   and claims concern the three rules.
3. **Items: all 23 rows, three draws per arm.** The key needs the scientist's
   approval before freeze.
4. **Raters.** Claude Opus 5.5 is primary; the scientist blind-scores a
   stratified random 25%.
5. **Thresholds.** A 0.25 margin at p < 0.05, with the harm tolerances above.
6. **Blind check replaced by a second model rater.** Later on 2026-10-16,
   before unsealing, the scientist skipped the blind 25% scoring. Instead,
   gpt-6-astra at max effort scores all 168 proposals with the same prompt. The
   primary averages both model raters, and each rater alone is a sensitivity
   analysis. A kappa for R1 below 0.6 makes the primary rater-dependent. No
   human rating validates the scores, and arm-guess accuracy is unavailable.

## Attempt001 result and proposed attempt002

**Attempt001 (2026-10-16) was underidentified.**
- Rules − old: +0.13 [−0.04, +0.33], p = 0.19.
- Directive control: +0.43, p = 0.0625.

The README packets spell out each claim limit, so under the old instructions
proposals already tested the named alternative at ceiling on 5 of the 10
primary items. [Result](outputs/figures/arm_comparison_attempt001/3lines.md).

**Attempt002 was adopted on 2026-10-16.** The scientist approved this design
and delegated its execution through to the result, asking that Codex
(gpt-6-astra, max) do as much of the work as possible. It tests the rules where
the old instructions do not already test the alternative. These are resume
points, where the agent's own logged plan competes with the alternative.

- **Items.** One per episode, EP01–EP23. Take the latest log entry at
  `960ab41` that was written while the episode was active, reports a
  scientific result, names a surviving alternative or limitation, and states a
  next action. The packet is the GOAL title plus the entry's date and text.
  Entries over 300 words give way to the next-latest qualifying entry.
- **Screening.** Two old-arm draws per candidate, scored by both model
  raters. A candidate is dropped when both draws already test a keyed
  alternative (rater-mean R1 ≥ 0.75). Screening draws are never reused. If
  fewer than 8 candidates remain, the attempt stops as infeasible.
- **Test.** Three fresh draws per arm (old, new, directive) on the retained
  items. Raters, analysis, margin and decision map are attempt001's,
  unchanged.
- **Detectability.** With 12 retained items and an old-arm rate near 0.2, a
  0.2 → 0.7 shift is detected with probability about 0.9; with 8 items, about
  0.7.
- **Limits.** The estimand is conditional on the screen: the rules' effect
  where the old instructions do not already test the alternative. The items
  are past states of the same episodes. Regression of the old-arm rate after
  screening works against finding an effect.

**Attempt002 stopped at screening (2026-10-17).** Nine episodes had a
qualifying resume point. Old-arm proposals never tested the alternative on 7
of them (0 of 14 draws) and always did on EP08 and EP14. That left 7 retained
items, below the floor of 8, so no comparison ran.

**Floor lowered to 7 (2026-10-17).** The scientist lowered the floor from 8 to
7 after screening, when only old-arm screening draws had been generated and
before any new- or directive-arm proposal existed. The test ran on the 7
retained items, where the smallest attainable exact two-sided p is 0.0156.

**Attempt002 result (2026-10-17): underidentified.**
- Rules − old: +0.12 [0.00, +0.33], p = 0.5.
- Directive control: +0.48 [0.00, +1.00], p = 0.0625. Only five items were
  informative, and two items stayed unmoved even under the directive.

[Result](outputs/figures/arm_comparison_attempt002/3lines.md).

## Attempt003 — adopted 2026-10-17

The scientist approved the four design points below and delegated execution
through to the result, using Codex (gpt-6-astra, max) wherever possible.

Both attempts were underidentified for one reason: too few items where the
old instructions do not test the named alternative but an explicit directive
does. Attempt003 selects those items on purpose and analyzes them by episode.
Only about 12 episodes have a real result with a named alternative, so the
number of episodes, not items, sets the test's resolution.

- **Item pool.** Every real decision point at `960ab41`: the 10 type-A README
  rows from attempt001, plus up to the three newest qualifying log entries per
  episode. Log entries use attempt002's criteria with a 400-word limit.
  Decision points already used in attempts 001–002 are included and disclosed.
- **Testable alternatives only.** The key lists an alternative only if a next
  analysis on permitted data could weaken it: an explanation, confound or
  limitation. Claim-scope disclaimers, such as "not a mechanism" or
  "exploratory", move to a separate line that R1 does not count. Items left
  without a testable alternative leave the pool. Codex drafts these
  classifications, and each one is checked against the source.
- **Screening with a directive check.** Each candidate gets two old-arm and
  two directive draws, scored by both model raters. An item is retained when
  old proposals rarely test the alternative (rater-mean R1 ≤ 0.25) and
  directive proposals usually do (≥ 0.75). Screening draws are never reused.
  If fewer than 10 items or fewer than 7 episodes remain, the attempt stops.
- **Test.** Three fresh draws per arm (old, new, directive) on retained items,
  with the same raters, rubric, 0.25 margin and decision map.
- **Unit of analysis.** The episode. Item differences are averaged within each
  episode before the exact two-sided sign-flip test over episodes. The
  smallest attainable p is 0.0156 with 7 episodes and 0.002 with 10.
- **What it can show.** The directive already moved every retained item in
  screening. If it still clears the margin in the fresh test draws, a small or
  absent rules effect reads as "no detectable effect", not "underidentified".
  The result describes decision points where the default fails and explicit
  instruction succeeds.
- **Size and cost.** At the earlier attempts' pass rate (about 40%), a pool of
  35–40 decision points would leave about 14–16 items across 7–10 episodes.
  Screening would take about 150 proposals and 300 ratings, the test about 9
  proposals and 18 ratings per retained item, roughly three hours of jobs.

**Attempt003 stopped at screening (2026-10-17).** Codex tagged alternatives as
testable or claim-scope, which left 14 candidates in 9 episodes. Screening
means were:
- old R1 = 1 and directive R1 = 1 on 6 items;
- old and directive R1 both 0 on 3;
- old R1 = 0 and directive R1 ≥ 0.75 on 5 items in 4 episodes, which were
  retained.

That is below the required 10 items in 7 episodes, so no rules-arm proposal
was generated in the frozen test. Of the 11 items that met the directive
threshold, 6 already had old R1 = 1, so the retained gap set was too small to
power the comparison.

**Descriptive look at the 5 gap items (scientist decision, 2026-10-17).** The
scientist asked for fresh draws on the 5 retained items: 3 per arm for old,
new and directive. Only effect sizes are reported, with no test, interval or
decision. With 4 episodes the smallest attainable exact p would be 0.125, so
nothing here can be inferential.

Result: the mean model-rated share testing the alternative was 0.00 under the
old instructions, 0.13 under the rules and 0.90 under the directive. The
episode-weighted differences were +0.125 for the rules and +0.875 for the
directive. [Result](outputs/figures/arm_comparison_attempt003/3lines.md).

## Claim limits

The result concerns how instruction text changes written proposals from one
proposer and one prompt, given a short packet rather than a long working
session. The review that motivated the rules had already examined these 23
decision points, so the rules may fit them better than future ones.

The test does not measure executed actions, scientific outcomes or
productivity, and it does not separate rule content from added length. R1
partly measures compliance with rule 2 by construction, so decision quality
rests on R2–R4. Model ratings are judgments; two model raters check each
other, but no human rating validates them. The second rater is the
proposer's model family. The history baseline stays descriptive. No external communications,
persistent services or system-configuration changes belong to this episode.
