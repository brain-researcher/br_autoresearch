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

Only aggregate process counts, content-free source locators, and concise
analyst-written English case summaries belong under `outputs/`. Source line
numbers and timestamps may locate evidence in the original private record.
Do not export conversation bodies, goal objectives, question text, raw tool
arguments, free-text runtime fields, internal reasoning, system/developer
instructions, credentials, scientific payloads, or tool-output bodies. Redacting
secrets does not make a conversation export suitable for the repository.
Do not forward the raw record to an external service.
Exact counts must state their event definitions and
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

## Arm comparison — adopted 2026-10-16

The adopted design uses four kinds of material, stored under
`outputs/arm_comparison/` unless noted.

- **Instruction arms.** Old: `AGENTS.md` and the three `SKILL.md` files from
  `git show 960ab41:<path>`. New: Old plus the three result-rule hunks of the
  2026-10-16 amendment, saved as a patch at freeze. These are repository
  instructions, not scientific data.
- **Item packets** ([items.md](outputs/arm_comparison/items.md)). Rows
  EP01–EP23 of the root README navigation table and each GOAL title, both at
  `960ab41`. These are released summaries. Reading other episodes stops at
  these two sources; no episode logs, run outputs, or closed, held-out or audit
  material enter a packet.
- **Key** ([key.md](outputs/arm_comparison/key.md)). Quotations from the
  packets, with types and flags. Raters see it; proposers never do.
- **Observations.** Generated proposals and their ratings. Raw texts stay on
  Oak under `outputs/runs/arm_comparison/` and are not tracked. Released score
  tables, the figure and the summary are tracked.

The unit is the item, one per episode; draws are repeats within an item. Items
are not a random sample of decision points: the review that motivated the
rules had examined them, and each packet is a short snapshot. Proposers and
raters get no file, tool or web access. The prospective process evidence
above is descriptive context only.
