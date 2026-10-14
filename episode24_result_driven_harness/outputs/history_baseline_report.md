# EP24: full-history baseline and first harness change

Observed October 13, 2026. This is an operational analysis, not a neuroscience
result or proof that the changed harness improves research productivity.

## Coverage

The original thread's entire recorded prefix was parsed on Slurm job 47703984.
It contains 160,721 complete JSONL records and 636,747,571 bytes, from
2026-10-03 17:24:59.953 UTC through 2026-10-13 23:33:05.435 UTC
(October 3 10:24 through October 13 16:33 Pacific). No complete line failed
JSON parsing and the prefix ended on a complete record. Later appends are not
part of this baseline. The source remained read-only; raw staging was temporary
node-local storage. [Exact summary](runs/history_baseline/summary.json).

Every record was structurally processed. Interpretation concerns permitted
operational events and selected user-facing messages, not a semantic reading
of hidden reasoning or every tool-output body. Those bodies and system
instructions were not exported. Goal packets were classified separately.
On October 14, derived conversation and objective exports were removed from
the project. Aggregate counts and source-line references below remain; the
original private source is unchanged. The exporter now emits aggregates only.

## Observed counts

| Observable | Count | Interpretation limit |
| --- | ---: | --- |
| Explicit goal-continuation packets | 447 | Automatic continuation markers, not 447 new human approvals. |
| Context-compaction records | 188 | Context management occurred; this alone does not measure its cost. |
| Distinct explicit question-tool calls | 38 | Calls, not a count or rate of unnecessary approvals. |
| Direct sleep-tool calls | 2,699 | Invocations, not measured idle hours or proof of waste. |
| Other user-role messages / structured replies | 50 / 7 | Message categories; no claim of verified human presence throughout. |
| Turn contexts with danger-full-access / never | 664 / 664 | Recorded settings, not evidence of who configured them. |

Tool counts use unique call IDs; two duplicate calls were excluded. Nested
tools inside `exec` are not separately counted. Keyword references to checks,
documentation, scheduler commands or permission are overlapping mentions and
must not be presented as executed actions or waste. Token snapshots are not
converted to prices or billed usage. [Continuation counts and definitions](runs/history_baseline/continuation_counts.json).

## What the history actually supports

1. **The objective kept its old framing.** All 447 goal packets repeat one
   objective covering EP01–EP21, allowing valid terminal results or unavoidable
   blockers and requiring unactivated follow-ups to remain separate. This
   supports narrowing the next-result target and carrying later user decisions
   explicitly; it does not prove that no science was done.
2. **There is a concrete avoidable approval candidate.** User source line 57682
   approved EP21 fitting at 1 CPU/7 GiB/2h. At line 58755, the agent asked whether
   the same existing job could start with ready sub-01 data while an independent
   export continued, explicitly preserving samples, models, resources and job
   count. Given those invariants, this is scheduling discretion. The judgment
   of redundancy is interpretive, not a property of all 38 questions.
3. **A stale authorization report is documented.** At source line 66486 the
   agent acknowledged that the EP21 approval had been received and acted on,
   while its earlier blocked-goal status had been conflated with an unanswered
   decision. Authorization, dependency state and app-goal status must stay
   distinct.
4. **Some questions protect a changed scientific comparison.** The resource
   flexibility request at line 110687 does not authorize the session exclusion
   proposed at 111450. Likewise, real-data continuation at 156176 does not
   automatically authorize the changed spatial support proposed at 157278.
5. **Automatic continuation and settings are observable; uninterrupted absence
   is not.** The history confirms goal-triggered continuations and the recorded
   sandbox settings. Session age does not establish ten days of continuous
   unattended login-node execution. This analysis changes no host configuration
   and starts no supervisor or automatic renewal.

Case evidence is located by the source line numbers above in the original
private record identified in [DATASETS.md](../DATASETS.md); no conversation
copies are retained in this project. Historical cases informed the patch and
cannot count as unseen prospective validation.

## Applied change and what remains unproven

The repository AGENTS now requires a named next result, authority carried
across resumes, direct scheduling/repair within that authority, and distinction
between a blocker and delivery. It preserves valid scientific stops and later
explicit resource ceilings. The run-episode skill now uses measured remaining
cost to distinguish per-job limits from cumulative limits; its maintenance
scenarios exercise continuation and actual changed scientific boundaries.

These edits are applied instructions, not measured runtime improvement. They
do not rewrite the active objective in the other thread or guarantee that an
already-running agent has reloaded them. A [concrete next-result objective](next_result_goal.md)
is prepared for the owning coordinator's next goal revision.

EP21's Finding A is the strongest prospective real-data endpoint; its existing
owner and dependency chain remain intact. EP20's equivalence/timing work and
EP18's control-bank continuation measure prerequisite progress. EP15's current
certificate stop is not erased by spare compute. EP24 is not complete until
the changed behavior is observed and its required new real-data result exists.
