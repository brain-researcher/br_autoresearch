# Campaign protocol

## Ownership

| Role | Owns | Cannot own |
| --- | --- | --- |
| Scientist / PI | Umbrella question, frozen reward, budget, launch approval | Evidence verdicts or outcome labels |
| Discovery and native goal lane | Candidate recall, exploratory evidence, provenance | Reward, selection, launch authority |
| Direction Society | Reward-blind scientific eligibility and uncertainty | Reward, portfolio selection, authorization |
| Portfolio selector | Transparent selection among eligible directions | Scientific verdicts |
| Confirmation runner | Approved independent experiment | Rewriting frozen task, reward, or approval |

## State and gates

```text
TASK_DRAFT
  -> DISCOVERING
  -> REVIEWING
  -> AWAITING_REWARD
  -> SELECTING
  -> AWAITING_LAUNCH_APPROVAL
  -> EXECUTING
  -> SYNTHESIZING
  -> COMPLETE
```

`AWAITING_REWARD` and `AWAITING_LAUNCH_APPROVAL` are successful pauses. Do
not skip either gate. Reward ranks only directions that Society marked
`eligible`; it cannot turn `revise`, `reject`, or `abstain` into an eligible
direction.

Keep discovery and Society review blind to the scientist-authored reward. Stop
at `AWAITING_REWARD` and `AWAITING_LAUNCH_APPROVAL`. Only the scientist
supplies reward and launch approval.

Freeze candidate and review evidence before the scientist authors reward. A
changed candidate set requires a new reward decision. Before launch, freeze a
claim, primary contrast, controls, primary outcome, positive control, Figure
target, stop rule, budget, and evidence tier. Only an approved, independently
prepared confirmation episode may execute canonical confirmation work.

## Native goal lane

`Launch Autoresearch Goal` is a client-native exploration lane that supplies a
terminal CandidateBundle. A `candidate_ready` submission leaves the persisted
loop in `DISCOVERING` for later `REVIEWING`; a `closed_no_candidate` or
`technical_failure` submission first records a private exact-terminal
reservation while the loop remains `DISCOVERING` with no active review action.
Only the same persisted submission may reconcile that reservation to
`COMPLETE`, without review or execution evidence. A `candidate_ready`
submission must not bypass Society review, reward, portfolio selection, launch
approval, independent confirmation, claim adjudication, or Landscape update.
Negative and technical terminals are Society-ineligible and do not call
Society.

The lane name is conventional, not a magic phrase: any unambiguous scientist
request to start this native-goal exploration, or the scientist's own launcher
invocation, is sufficient. Drafting, readiness, or setup requests alone do not
authorize launch, and an agent must not invoke the launcher to manufacture that
authority.

Prepare only against the authenticated owner’s `codex_autoresearch_v1` loop at
`DISCOVERING`, and bind the exact source revision into the handoff. Submit must
revalidate that binding before it records the terminal bundle.

The enhanced native-goal path explicitly requests
`candidate_bundle_contract_version: "v3"` and verifies the returned frozen
version before creating the host goal. Omitted or explicit `v1`, and an
already-frozen explicit `v2`, remain their exact compatibility contracts;
neither side upgrades a frozen V1 or V2 bundle. An enhanced terminal is
`CandidateBundleV3` with `GoalExplorationTraceV2`; a frozen V2 terminal retains
its `CandidateBundleV2` with `GoalExplorationTraceV1`. Before
candidate-discriminating target outcome access, the same Goal runs an
outcome-blind ideation prelude. Outcome-blind dataset/QC inventory and
already-observed prior-episode phenomenon evidence are permitted inputs, not
candidate-discriminating target outcome access: start from the phenomenon and
prior evidence; inspect the exposed schema and attempt high-level
`kg_hypothesis_workflow` or `kg_hypothesis_candidate_cards`; record an
unavailable, rejected, or failed attempt in `outputs/idea_search.json` as
non-blocking client-maintained workspace provenance; screen the canonical
fifteen `PatternId` values exactly once in canonical order; record one through
three advisory ideation pattern steps per candidate. `official_submode` is
optional: use it only when the actual C00--C30 card informed formulation and an
authoritative mapping from that card to a canonical `PatternId` can be verified
from the available source. Otherwise, use the schema-supported, non-blocking
`parent_pattern_only` classification and omit `submode_id`; use
`unknown_pattern` only when no canonical `PatternId` fits. Do not force or
fabricate C00--C30 lineage; then
freeze its typed trace and V3 selection record before candidate-discriminating
target outcome access.
The V3 trace starts phenomenon-first by default
(`research_mode: phenomenon_driven`); `theory_driven` and
`instrument_validation` are explicit choices. The ordinary bounded strategy
records at least two competing explanations and ranked candidates (except
explicit instrument validation), selecting no more than two input-ready
discriminating tests before outcome observation. That timing is
`client_attested`, not independently preregistered. The trace also records
novelty classification, nearest prior references and delta, plus terminal
self-reflection: belief update, strongest surviving alternative, failed
assumptions, disposition, and any successor question. A registered program may
use its frozen `registered_program_policy`, but it must name its
`search_policy_ref`; generic Goal exploration never becomes unbounded.

Use the persisted MCP loop and `$brain-autoresearch-loop` to display or advance
a campaign after the handoff. The native goal itself belongs to the current
Codex task; it is not a server-runner, an episode, or an execution
authorization.

For a `candidate_ready` submission, the client explicitly declares primary
result artifacts in `output_artifacts` and uploads only those declared review
artifacts into the owner-scoped goal handoff. `outputs/experiment_log.md` may be
declared as optional supplemental context. `outputs/memory.md`,
`outputs/governance.md`, `outputs/society.md`, `outputs/loop.md`,
`outputs/landscape.md`, and `outputs/verification.md` are client delivery
projections, not required or implied Society review artifacts, and must never
be declared in `output_artifacts`. Brain Researcher freezes a reward-blind Goal
Society review packet matching the frozen CandidateBundle version before the
native Society panel runs.
The complete client panel bound to that exact packet is immutable supplemental
evidence only. Brain Researcher then runs its server-owned strict Codex-primary
Society router over the frozen packet and supplemental objections. Only that
trusted typed decision has direction authority: `eligible` advances to
`AWAITING_REWARD`; `revise`, `reject`, `abstain`, and `panel_incomplete` remain
in `REVIEWING` without reward authority. A router failure freezes a
`panel_incomplete` refusal, and exact replay resumes without rerouting an
already persisted trusted outcome.
Negative and technical terminal bundles do not call Society. Before persistence,
their recovery surface accepts only the same terminal bundle; after persistence,
`autoresearch_goal_get` returns `retry_terminal_reconciliation` until the typed
completion binds the Goal handoff, accepted submission, and terminal status.
It leaves review, reward, portfolio, approval, authorization, execution,
synthesis, scientific transition, and scientific acceptance empty.
If a `COMPLETE` native-goal terminal belongs to a different handoff,
`autoresearch_goal_get` returns `closed_by_other_handoff`; it is a terminal,
non-recoverable result for this handoff.

At launch, the client workspace initializes `outputs/experiment_log.md`,
`outputs/memory.md`, `outputs/governance.md`, `outputs/society.md`,
`outputs/loop.md`, `outputs/landscape.md`, and `outputs/verification.md` as
human-readable projections. They are not server authority, Brain Researcher
memory, or Society evidence by default. `outputs/experiment_log.md` is optional
supplemental review context only when the candidate bundle explicitly declares
it. The other six projection files are never required, inferred, uploaded, or
reviewed Society artifacts. `outputs/society.md` is stage-aware: before packet
freeze it is only a not-called stub, and its review content begins after freeze
and copies only observed MCP facts. For negative and
technical terminals, both governance and Society must state `Society: not
eligible and not called`. `outputs/loop.md` copies only persisted loop facts;
`outputs/landscape.md` separates proposed, permitted, and applied transitions;
`outputs/verification.md` reports mechanical checks and states that scientific
validity is not mechanically verified. None of these Markdown projections may
grant or replace persisted authority.

These are exactly the seven workspace Markdown projections. Do not add an
eighth usage, trajectory, scientific-review, report, or terminal-status
projection: V3 exploration traces and server-owned scientific-learning records
are typed JSON records, not additional workspace authority or delivery files.
`outputs/idea_search.json` is a structured client-maintained workspace
provenance artifact, not a standalone top-level schema. V3 server authority is
the `GoalExplorationTraceV2` embedded in `CandidateBundleV3`. An unavailable KG
attempt recorded there remains client provenance without becoming
server-validated lineage. It records selection provenance only and cannot grant
a Goal gate, Society, reward, approval, execution, confirmation, or Landscape
authority; it is not a confirmation or execution `PatternProgram`.
Before client-native discovery completes, the Goal host reports its observed
token/cost/compute telemetry through `autoresearch_goal_usage_submit` and reads
the folded view through `autoresearch_goal_usage_get`. Missing telemetry is
`null` with `unavailable` measurement, not zero; the host cannot assert owner,
episode, run, reward, or scientific identities.

After an eligible direction receives explicit scientist reward and deterministic
selection, `autoresearch_goal_confirmation_prepare` freezes the independent
confirmation episode at `AWAITING_LAUNCH_APPROVAL`. A separate explicit
scientist action may then call `autoresearch_goal_confirmation_launch` with only
the server-issued `episode_id`, `action_id`, current revision, and
`scientist_confirmation=true`. The bridge derives canonical authority and may
advance the loop to `EXECUTING` only after a distinct persisted provider run is
accepted. This transition is execution lineage, not scientific acceptance;
result review and Landscape transition remain later closeout work.
Expired pre-provider authority creates a structurally identical, server-owned
successor that requires a new explicit approval. A persisted accepted run is
reconciled after interruption and must never be launched twice.

Only a registered adopter that freezes `GoalConfirmationContractV2` with
`ScientificLearningPolicyV1` enters the generic scientific-learning terminal
consumer. A V2 discovery bundle alone does not. For that opt-in path the
server re-resolves `GoalConfirmationTerminalObservationV1` and canonical
terminal evidence, records a server-owned `ScientificReviewVerdict` on success
or `TechnicalDispositionV1` on failed, cancelled, timed-out, or skipped runs,
then persists `ScientificLearningCloseoutIntentV1`. It materializes exactly
`write_report`, `revise_report`, or `none`, collects the create-once
`ScientificLearningUsageSummaryV1`, persists `ScientificLearningCloseoutV1`,
and only then performs the outer-loop `COMPLETE` CAS. External OAK/Sherlock
review is supplemental client-attested evidence only: it cannot choose report
action, complete this transaction, accept science, or update Landscape. This
consumer also cannot automatically reward, approve, launch, create a ClaimCard,
or create a Landscape transition.

The Goal review snapshot is labeled `client_submitted_native_snapshot`. Each
child memo repeats the exact packet/direction binding, and only the integrator
memo carries a decision. This is an admitted client-side Society record, not a
claim that BR independently queried the Codex app-server. Its integrator
decision cannot open the reward gate.
