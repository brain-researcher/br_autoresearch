# EP24 inputs and evidence boundaries

## Operational baseline

The authorized primary source is the existing session record for thread
`01a102cc-2ffb-7120-bd17-e442a0a313e8`:

`/home/users/zijiao/.codex/sessions/2026/10/03/rollout-2026-10-03T10-24-58-01a102cc-2ffb-7120-bd17-e442a0a313e8.jsonl`

The coordinator will record the capture time and fixed byte endpoint before
processing. Analyze the complete event history inside that prefix, not a
tail-only sample. If the endpoint cuts an incomplete JSONL record, disclose
the excluded trailing bytes. Later appends are outside this baseline unless a
subsequent capture is explicitly identified. The live file must not be edited,
truncated, copied into the episode, or treated as stationary during reading.

This is already exposed conversational/process history, not a fresh scientific
holdout. Authorization covers that one source, not other sessions, application
configuration, credentials, or arbitrary home-directory searches. Bulk parsing
and focused parser checks run in the coordinator's bounded Slurm allocation.

## Analysis units and disclosure

Use recorded events, turns, and linked task transitions as operational units.
Keep source line or event offsets and timestamps sufficient to inspect a
decision. Different task segments and episodes are not independent randomized
replicates. Distinguish observed scheduler/tool events from plans, reported
claims, and inferred behavioral classifications. Missing events or ambiguous
ownership remain unknown. Thread age does not establish active or unattended
agent time.

Only derived, redacted process metadata, relevant user requests and user-visible
assistant messages belong under `outputs/`. Preserve their source line and
timestamp. Do not export internal reasoning, system/developer instructions,
credentials, scientific payloads, or bulk tool-output bodies. Do not forward the raw record
to an external service. Exact counts must state their event definitions and
coverage; causal explanations and redundancy judgments require explicit
interpretive labels and case evidence.

Historical case labels are EP01, EP04, EP08, and EP14. Their occurrence in the
source is not permission to read their raw inputs or closed outputs.

## Prospective process evidence

EP15, EP18, EP20, and EP21 are possible cases, not activated experiments under
EP24. Their owners may provide permitted, redacted summaries of authorizations,
dependencies, execution transitions, resources, and released result references.
Store those small process observations under EP24 outputs with their source and
observation time. Direct access to another episode still requires reading its
current contract and establishing its positive read boundary in that scope.

No other episode's scientific source payloads are EP24 inputs. EP24 cannot
change their splits, models, eligibility, metrics, stopping rules, claim limits,
or holdout status. Scientific execution and resource accounting remain owned
by those episodes; do not submit duplicate jobs or count their consumption
twice.

## Evidence strength

The historical record supports an operational baseline and candidate patches.
Prospective task traces can demonstrate applied behavior and real result
completion. Neither an unmatched before/after comparison nor one successful
case proves that the harness caused a general improvement. This episode has no
new neuroscience endpoint and no independent scientific confirmation set.
