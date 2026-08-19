---
name: brain-autoresearch-loop
description: Run or resume the single persistent, human-gated Brain Researcher autoresearch loop from the current Codex conversation, including Society review, scientist reward and launch approval, visible MCP checkpoints, and an explicitly requested Launch Autoresearch Goal in an already-open GOAL.md/DATASETS.md workspace. Use for outer-loop control, approved registered dispatch, reviewed next-round continuation, or explicit native-goal exploration; not for implicit goals, direct compute, or background service launch.
---

# Brain Autoresearch Loop

Treat the current Codex conversation as the UI/control shell for one visible,
human-gated campaign. It can show work and coordinate agents, but it is not an
authority store, issuer, scheduler, or execution runtime.

This is the single canonical outer-loop skill. Read
[the campaign protocol](references/campaign_protocol.md) before starting or
advancing a campaign. It covers human ownership, the state machine, and the
fact that a native goal submission is a handoff for later review, not
confirmation.

## Establish the authoritative loop

1. Inspect `server_info`, then inspect `loop_profile_get` for
   `codex_autoresearch_v1`. Use only MCP tools exposed by the active server. If
   the profile or a required tool is unavailable, report that integration gap
   and stop rather than substituting another profile.
2. Resume with `autoresearch_loop_get` and the persisted `loop_id` whenever it
   exists. Use `autoresearch_loop_init` only to begin a new loop.
3. Treat persisted MCP loop state, its action records, and its returned
   `next_action` as authority. Treat chat history, copied text, and a plan view
   as display context only; none can grant reward, approval, authorization, or
   execution.

The required state order is:

`TASK_DRAFT → DISCOVERING → REVIEWING → AWAITING_REWARD → SELECTING → AWAITING_LAUNCH_APPROVAL → EXECUTING → SYNTHESIZING → COMPLETE`.

A submitted native Goal with terminal status `closed_no_candidate` or
`technical_failure` takes the explicit early-close branch
`DISCOVERING → (exact terminal reservation, no review action) → COMPLETE`.
The loop remains `DISCOVERING` until the exact validated terminal submission
is persisted; then reconciliation records no review, reward, launch approval,
execution, scientific transition, or acceptance.

Apply normal progression with `autoresearch_loop_apply_action`. Persist the
review, reward, and launch checkpoints with the exposed loop tools, including
`autoresearch_loop_checkpoint` and `autoresearch_loop_record_reward`.

## Launch Autoresearch Goal

Use this lane only when the scientist explicitly asks to **Launch Autoresearch
Goal** or resume that explicit native-goal exploration. It is a client-native
discovery lane bound to the outer campaign, not a canonical execution lane.

### Optional goal-prompt Stage 1 prelude

Use this optional, host-only authoring prelude only when the scientist
explicitly invokes `$goal-prompt`, or clearly asks to draft or review `/goal`
text before launch. The phrase **Launch Autoresearch Goal** alone does not
implicitly enable this prelude.

1. Use the already-open `GOAL.md` and `DATASETS.md` as the `goal-prompt`
   research scenario and produce a scientist-confirmed research-intent brief.
2. This is Stage 1, read-only authoring only. Do not render a final `/goal`,
   enter Stage 2, create `.goal-task/`, create or update a host goal, call MCP,
   or write the workspace.
3. After the scientist confirms the brief, `$goal-prompt` immediately exits.
   The separate launch lane below later passes that confirmed brief verbatim as
   the only `research_intent` to `autoresearch_goal_prepare`. Preserve the
   server-issued `goal_objective` unchanged and use it only for the single
   `create_goal(...)` call in the launch lane below.
4. This optional prelude does not change CandidateBundle, Society, reward,
   approval, confirmation, or Landscape boundaries. If `$goal-prompt` is
   unavailable, report `optional prelude unavailable` and continue with the
   scientist's intent; do not simulate the prelude or block the launch lane.

1. Verify the already-open local workspace contains `GOAL.md`, `DATASETS.md`,
   `inputs/`, and `outputs/`. Read the two Markdown files, treat `inputs/` as
   read-only, and write every new artifact under `outputs/`. Do not create a
   workspace or substitute another path.
2. First call `get_goal({})` in the current Codex task. If it reports an
   unfinished goal, preserve it and ask whether to continue that objective;
   do not prepare or create another goal.
3. Read the authoritative persisted loop and require profile
   `codex_autoresearch_v1` at `DISCOVERING`. Inspect the exposed
   `autoresearch_goal_prepare` schema, then call it with that `loop_id`, its
   current `expected_revision`, the research intent, an idempotency key, and
   `candidate_bundle_contract_version: "v3"`. Retain the server-issued
   `goal_handoff_id`, `goal_objective`, and returned
   `candidate_bundle_contract_version`; this enhanced launch continues only
   when that version is exactly `v3`. An omitted version or explicit `v1`, and
   an already-frozen explicit `v2`, remain their respective compatibility
   contracts. For a frozen V2 handoff, continue only when that version is
   exactly `v2`; never silently upgrade a frozen V1 or V2 handoff. If the loop
   binding is stale, unavailable, rejected, or returns another contract
   version, stop without creating a goal.
4. Only after successful preparation, call `create_goal(...)` **exactly once**
   in the same task, passing the server-issued `goal_objective` unchanged. Do
   not retry, replace an unfinished goal, or call `create_goal` from a new
   task, subagent, server, or `codex exec` process. This creates no workspace
   and no server `GoalRunner`.

### Required V3 outcome-blind ideation prelude

Before candidate-discriminating target outcome access, run this prelude inside
the same Goal. Outcome-blind dataset/QC inventory and already-observed
prior-episode phenomenon evidence are permitted inputs; neither is
candidate-discriminating target outcome access. The prelude is selection
provenance only, not a separate Goal, a review, or an execution program.

1. Start from the stated phenomenon, available prior evidence, and any
   outcome-blind dataset/QC inventory. Record their references and every
   unavailable or unresolved premise.
2. Inspect the actual exposed schemas for `kg_hypothesis_workflow` and
   `kg_hypothesis_candidate_cards`, then attempt one high-level
   candidate-ideation call that its schema permits. Do not invent parameters or
   substitute a lower-level route. If neither is exposed, the schema rejects
   the attempt, or the call fails, record that KG ideation is `unavailable`
   with the observed reason in `outputs/idea_search.json`; this is non-blocking
   client-maintained workspace provenance, not evidence for or against a
   candidate.
3. Screen the canonical fifteen `PatternId` values exactly once, in canonical
   order, as one pre-outcome pass: `assumption_audit_and_pivot`,
   `architectural_operator_substitution`, `generative_process_redesign`,
   `controlled_diagnostic_design`, `unify_into_shared_representation`,
   `reframe_as_solvable_object`, `self_supervised_signal_engineering`,
   `structural_prior_encoding`, `algebraic_equivalence_unification`,
   `heterogeneous_decomposition`, `decompose_and_delegate`,
   `relax_discrete_search_to_continuous`, `adapt_via_conditioning`,
   `characterize_limit_then_surpass`, and
   `targeted_self_supervised_objective`. Do not repeat, score after
   candidate-discriminating target outcome access, or treat an omitted pattern
   as screened.
4. For every candidate, record one through three advisory ideation pattern
   steps. Record `official_submode` only when the actual C00--C30 card informed
   formulation; then record that card's `submode_id` and its matching canonical
   `PatternId` through the official C00--C30 subpattern-to-`PatternId` mapping in
   [`autoresearch_goal_ideation_v3.md`](../../docs/specs/autoresearch_goal_ideation_v3.md).
   Otherwise use `parent_pattern_only` for canonical fifteen-`PatternId`
   classification. Use `unknown_pattern` only when no canonical `PatternId`
   fits. Do not force a C00--C30 ID for a canonical candidate or fabricate a
   31-submode lineage.
5. Freeze the typed `GoalExplorationTraceV2` and the pre-outcome selection
   portion of `CandidateBundleV3` before candidate-discriminating target
   outcome access. A later terminal record may add observations but must not
   alter the frozen ideation record, candidate set, selected steps, or screen
   result.

`outputs/idea_search.json` is a structured client-maintained workspace
provenance artifact, not a standalone top-level schema, an authority store, a
Goal gate, or an eighth Markdown projection. V3 server authority is the
`GoalExplorationTraceV2` embedded in `CandidateBundleV3`. An unavailable KG
attempt recorded there remains client provenance without becoming
server-validated lineage. The resulting ideation program records why candidates
were selected; it is distinct from, and cannot become, a confirmation or
execution `PatternProgram`.

5. Let the coding-agent harness explore the stated question and data contract:
   inspect, code, execute, observe, and repair within the workspace. Before a
   terminal state, write and parse `outputs/candidate_bundle.json` as the
   terminal `CandidateBundleV3` required by this enhanced handoff. Do not use a
   V1 or V2 bundle body with a V3 handoff. An already-frozen V1 or V2 handoff
   continues with its exact historical contract, including terminal
   `CandidateBundleV2` and `GoalExplorationTraceV1` where applicable. At launch,
   initialize the matching
   projection stubs under `outputs/` from `templates/`. Replace every `not
   observed` only when evidence exists, and maintain these human-readable
   workspace projections as well:
   - `outputs/experiment_log.md`: an append-oriented lab notebook of attempted
     analyses, failures, deviations, result references, and remaining
     uncertainty;
   - `outputs/memory.md`: a concise synthesis of reusable local lessons,
     approaches not carried forward, observed failure modes, unresolved
     questions, and the evidence references that support each item; and
   - `outputs/governance.md`: the latest observed Goal/outer-loop state,
     handoff and terminal status, Society applicability or frozen outcome,
     open human gate, and explicit non-authority boundary.
   - `outputs/society.md`: a stage-aware projection whose review content begins
     only after freeze and records observed Society facts, objections, and
     required changes. Before a review packet is frozen, retain its stub and
     state that Society has not been called; after `candidate_ready` review
     begins, update it only from observed MCP responses to
     `autoresearch_goal_get`, `autoresearch_goal_review_prepare`, or
     `autoresearch_goal_review_submit`.
   - `outputs/loop.md`: a projection of the observed persisted outer-loop
     state, revision, actions, and transitions; and
   - `outputs/landscape.md`: a projection that separates a proposed Landscape
     change from a permitted change and an applied change. Do not infer or
     record a transition that MCP has not returned.
   - `outputs/verification.md`: a mechanical-verification record for artifacts,
     commands, and numerical checks. It must state that scientific validity is
     not mechanically verified.
   These files are client-maintained delivery projections, not MCP authority
   stores. `outputs/memory.md` is not Brain Researcher memory, and
   `outputs/governance.md`, `outputs/society.md`, `outputs/loop.md`,
   `outputs/landscape.md`, and `outputs/verification.md` cannot grant reward,
   approval, execution, Society, ClaimCard, memory, or Landscape authority.
   `outputs/governance.md` must say `Society: not eligible and not called` for
   a negative or technical terminal instead of inventing a review;
   `outputs/society.md` must say the same. Do not promote
   `outputs/memory.md` into Brain Researcher memory unless the scientist
   separately requests that write. For `candidate_ready`, use
   `output_artifacts` to explicitly declare the primary result artifacts for
   Society review. `outputs/experiment_log.md` may be declared only as optional
   supplemental context. Do not require or infer `outputs/memory.md`,
   `outputs/governance.md`, `outputs/society.md`, `outputs/loop.md`,
   `outputs/landscape.md`, or `outputs/verification.md` as Society review
   artifacts; never add any of those six projection files to the enhanced
   `CandidateBundleV3.output_artifacts`. The identical exclusion remains in
   force for already-frozen `CandidateBundleV1.output_artifacts` and
   `CandidateBundleV2.output_artifacts`.
   Before candidate-discriminating target outcome access, write the required
   `GoalExplorationTraceV2` inside this V3 bundle, including the frozen
   outcome-blind ideation record above.
   It retains the V1 exploration fields: use `phenomenon_driven` unless the
   scientist explicitly chose
   `theory_driven` or `instrument_validation`; record the phenomenon statement,
   source, and any required evidence references. For the ordinary
   `bounded_ranked_candidates` strategy, record at least two competing
   explanations and two contiguous ranked candidates (except explicit
   instrument validation), with a bounded candidate set of at most eight. Each
   candidate has an explanation, discriminating test, falsifier, input
   readiness, estimated cost, selection bit, and `pre_outcome_timing`. Select
   no more than two candidates, and only candidates with
   `input_readiness: ready`; their timing assurance is
   `client_attested`, not server-verified preregistration. A registered program
   may instead freeze `registered_program_policy` with its
   `search_policy_ref`, but this is never an implicit escape to unbounded
   exploration. Record `novelty_classification`, `nearest_prior_refs`,
   `novelty_delta`, and `revision_requirement`; a `novel_increment` needs named
   nearest prior references and a concrete delta. Finish a complete trace with
   `belief_update`, `strongest_surviving_alternative`,
   `failed_assumptions`, `proposed_disposition`, and a `successor_question` for
   every non-`stop` disposition. The `candidate_ready`
   `terminal_candidate_id` must be a pre-outcome selected candidate; a
   `closed_no_candidate` terminal must not name one. This is an advisory,
   client-attested exploration record: it neither proves novelty nor grants a
   gate. `outputs/idea_search.json` may preserve supporting client-maintained
   workspace provenance, but it is not a `PatternProgram`, an authority record,
   or a substitute for the frozen trace.
   Use exactly the seven existing Markdown projections listed above. Do not add
   an eighth status/provenance projection such as `outputs/trajectory.md`,
   `outputs/usage.md`, `outputs/review.md`, or `outputs/report_status.md`; the
   trace, usage ledger, and closeout records are typed JSON.
   `outputs/idea_search.json` is the structured client-maintained workspace
   provenance artifact rather than a new delivery projection. This does not
   forbid ordinary primary scientific output artifacts that are explicitly
   declared for review.
   Before the native goal is closed, submit the host's one discovery invocation
   through `autoresearch_goal_usage_submit`. Supply the server-issued
   `goal_handoff_id`, a stable `invocation_id` and idempotency key, and only
   observed host token, LLM-cost, compute-actual, or reservation telemetry. The
   server derives owner and loop identity: never send caller-asserted owner,
   episode, run, reward, scientific outcome, or authority. If the host has no
   observed value, use `null` with its measurement marked `unavailable`; do not
   turn missing usage or price into zero. If the native Society panel later
   runs, submit its separate `direction_society_native` invocation the same
   way. Use `autoresearch_goal_usage_get(goal_handoff_id)` to inspect coverage,
   not to infer an absent charge.
6. Inspect the exposed `autoresearch_goal_submit` schema and submit the
   server-issued `goal_handoff_id` plus the parsed `candidate_bundle`. The
   server records the handoff but does not inspect the client workspace. If
   submit is unavailable or rejects the bundle, leave the native goal active
   and report the integration gap. After an accepted response, call
   `autoresearch_goal_get(goal_handoff_id)` and refresh
   `outputs/governance.md` and `outputs/loop.md` only from those returned MCP
   facts; do not infer a stage, Society result, reward, approval, execution, or
   scientific status. For a `candidate_ready` submission, do not update
   `outputs/society.md` with a verdict until review facts exist.
7. Call `update_goal({status: "complete"})` only after a terminal bundle exists
   and `autoresearch_goal_submit` accepts it. This completes the client-native
   exploration, not its governance review. For `closed_no_candidate` or
   `technical_failure`, require that returned bridge state to show the outer
   loop as `COMPLETE` with `next_action: null`; refresh `outputs/governance.md`
   and `outputs/loop.md` from that response, set `outputs/society.md` to
   `Society: not eligible and not called`, record no applied transition in
   `outputs/landscape.md`, preserve the zero-execution and
   zero-scientific-transition boundary across those projections, and then stop.
   If submit errors before the bridge shows this state, retry only the exact
   same terminal submission and inspect `autoresearch_goal_get` again. Its
   `resume_exact_terminal_submission` or `retry_terminal_reconciliation`
   action is the only allowed recovery; never change the bundle or advance
   review. If it returns `closed_by_other_handoff`, a different handoff already
   closed the loop: this handoff is non-recoverable, so do not retry it or call
   Society. A pre-reservation legacy negative or technical terminal may use
   `retry_terminal_reconciliation` only by replaying its exact persisted bundle;
   the server rechecks the original `DISCOVERING` revision before reconstructing
   the reservation. Do not call Society or open a reward gate.
8. For `candidate_ready`, inspect `autoresearch_goal_get` and upload each file
   named by `output_artifacts` with `autoresearch_goal_artifact_put`. Send file
   content, its exact `outputs/`-relative reference, and media type; never send
   an absolute host path, `inputs/`, model weights, or undeclared artifacts.
   The server-owned snapshot is immutable review evidence, not an EpisodePaths
   execution artifact.
9. Call `autoresearch_goal_review_prepare(goal_handoff_id)`. Retain the
   server-issued `review_packet_id`, `direction_id`, and exact immutable packet.
   This freezes reward-blind exploratory evidence and advances the bound outer
   loop from `DISCOVERING` to `REVIEWING`; it does not create an episode. Update
   `outputs/society.md`, `outputs/governance.md`, and `outputs/loop.md` only
   from the observed MCP response. The projection may report a frozen packet,
   but must not author a verdict or grant a gate.
10. Run the native Codex Society panel below against that exact packet. Submit
    the complete observed panel payload with `autoresearch_goal_review_submit`;
    never submit a bare caller-authored verdict or eligible direction ID. The
    server validates the fixed topology, packet binding, typed memos, and
    reviewer provenance, but stores this caller snapshot as supplemental only.
    It then runs its strict Codex-primary Society router over the frozen packet
    and panel objections. Only that server-owned typed decision may open the
    reward gate; a router failure freezes `panel_incomplete` in `REVIEWING`.
11. If the server-owned decision is `eligible`, verify the authoritative loop is
    exactly `AWAITING_REWARD` and stop for scientist-authored reward. For
    `revise`, `reject`, `abstain`, or `panel_incomplete`, preserve the refusal
    and stop at terminal v1 action `review_blocked` in `REVIEWING`; these
    outcomes cannot open the reward gate or silently reopen discovery. Refresh
    `outputs/society.md`, `outputs/governance.md`, and `outputs/loop.md` from
    the observed MCP outcome only; do not treat any Markdown projection as a
    persisted Society record.
12. After the scientist explicitly records reward and the loop deterministically
    selects one direction, call `autoresearch_goal_confirmation_prepare` with the
    server-issued `goal_handoff_id` and current revision. Retain the returned
    `episode_id` and exact `next_action`; preparation still grants no execution
    authority.
13. Only when the scientist explicitly approves this frozen episode, inspect the
    `autoresearch_goal_confirmation_launch` schema and call it with exactly
    `episode_id`, the returned `action_id`, its `expected_revision`, and
    `scientist_confirmation=true`. Do not separately call generic canonical
    authorize or launch tools. A successful response must contain distinct
    `run_id` and `canonical_run_id` and authoritative state `EXECUTING`.
    If the response instead supplies `successor_episode_id` and `next_action`,
    stop for a new explicit scientist approval of that successor; do not retry
    the predecessor or reuse its expired authority.
14. Watch the returned MCP `run_id` with `autoresearch_run_watch`. Entering
    `EXECUTING` means only that the approved provider run was accepted and bound;
    it is not a scientific result, Society judgment, or Landscape transition.

### Scientific-learning terminal boundary

An enhanced V3 discovery bundle does not itself authorize a scientific-learning
closeout. Only a registered adopter that freezes `GoalConfirmationContractV2` with
`ScientificLearningPolicyV1` enters that terminal consumer; a V1 confirmation
contract and any already-frozen V2 compatibility contract retain their existing
program-specific closeout. For the explicit confirmation V2 contract, continue
to watch the same canonical run and report only observed terminal facts. The
server-side reconciliation sequence is:

```text
canonical GoalConfirmationTerminalObservationV1 and terminal evidence
  -> succeeded: server-owned ScientificReviewVerdict
     non-success (failed/cancelled/timeout/skipped): TechnicalDispositionV1
  -> ScientificLearningCloseoutIntentV1
  -> write_report / revise_report / none materialization
  -> ScientificLearningUsageSummaryV1
  -> ScientificLearningCloseoutV1
  -> existing outer-loop COMPLETE CAS
```

The closeout intent binds the future usage-summary reference before report
materialization so the final summary can include review and report-generation
usage. A report is produced only for `write_report`; `revise_report` writes a
revision handoff and `none` writes neither. `autoresearch_run_usage(run_id)` is
the run-scoped read surface for observed operational coverage. Client-native
OAK or Sherlock review is supplemental client-attested evidence only: it
cannot replace the server-owned review, choose report action, complete the
transaction, establish scientific acceptance, or update Landscape.

Do not treat this terminal consumer as an automatic reward, portfolio choice,
approval, launch, successor episode, ClaimCard, or Landscape transition. Its
final closeout remains non-promoting; any later scientific or Landscape action
requires its own persisted authority.

Submission and review by themselves are never authorization, canonical episode
creation, dispatch, scientific acceptance, confirmation, or Landscape transition.
Workspace Markdown is a reconstructable display surface and cannot grant or
replace persisted Goal, Society, reward, approval, execution, ClaimCard, memory,
or Landscape authority.

Workspace isolation in v0 is logical and `client_unattested`; it does not prove
a mount, sandbox, capability boundary, or confirmation-data firewall.

### Candidate bundle boundary

Require `outputs/candidate_bundle.json` to contain:

- for this enhanced launch, an explicitly supplied
  `schema_version: br.autoresearch_goal_candidate_bundle.v3` and its required
  `exploration_trace: GoalExplorationTraceV2`, including the frozen
  outcome-blind ideation record; an already-frozen compatibility handoff
  accepts only its frozen `CandidateBundleV1` with
  `schema_version: br.autoresearch_goal_candidate_bundle.v1` or terminal
  `CandidateBundleV2` with
  `schema_version: br.autoresearch_goal_candidate_bundle.v2` and
  `GoalExplorationTraceV1`;
- `terminal_status`: `candidate_ready`, `closed_no_candidate`, or
  `technical_failure`;
- a non-empty `research_question` for every terminal status;
- non-duplicated `outputs/`-relative `output_artifacts` that explicitly name
  primary result artifacts, plus `limitations`, `deviations`, and
  `failed_attempts` lists. `outputs/experiment_log.md` is optional supplemental
  context only when explicitly declared; `outputs/memory.md`,
  `outputs/governance.md`, `outputs/society.md`, `outputs/loop.md`,
  `outputs/landscape.md`, and `outputs/verification.md` are delivery
  projections, not Society review artifacts, and must never be declared in
  `output_artifacts`;
- for `candidate_ready`, the hypothesis, prediction, falsifier, primary test,
  controls, `development_summary`, and `reproduction`;
- `isolation_assurance: client_unattested`;
- `scientific_status: exploratory_only`;
- `confirmation_requirement: independent_fresh_confirmation_required`;
- `confirmation_eligible: false`; and
- `scientific_acceptance: false`.

A negative or technical close is a valid terminal exploration result, but no
bundle supports a claim that the hypothesis is confirmed or supported. A V3
technical trace may be `partial` or `unavailable_before_freeze`, but must record
its technical reason and failed attempts and may propose only `stop` or
`retry_technical`; it must not invent a phenomenon, novelty position, belief
update, or outcome-derived ideation mapping. Never call
`autoresearch_episode_authorize`, `autoresearch_episode_launch`, or an
equivalent dispatch route from this lane.

### Goal review bridge boundary

The review bridge freezes only artifacts explicitly declared in the accepted
frozen CandidateBundle version (`CandidateBundleV1`, `CandidateBundleV2`, or
`CandidateBundleV3`).
Treat `autoresearch_goal_get` as the resumable phase
projection. Exact artifact, packet, and panel replays may resume an interrupted
handoff; changed content or a panel bound to another packet must fail closed.

Direction Society answers only whether an exploratory candidate is eligible
for the scientist's later reward ranking. `reward_eligible` is not
`confirmation_eligible`. Society cannot supply reward, select a portfolio,
approve launch, create a ClaimCard, authorize execution, or update Landscape.

Before running the Goal panel, inspect the published `society_panel` schema.
Every child memo must repeat the exact server-issued `review_packet_id` and
`direction_id`, its own fixed `role_id`, a bounded summary, strongest
objection, and required confirmation changes. Only the `society_integrator`
memo may contain `decision` and `reward_eligible`; do not add those fields at
the panel top level or to another reviewer. Submit the observed task-tree state
with `provenance_assurance: client_submitted_native_snapshot`. This accurately
labels the handoff: BR validates the closed snapshot and memo bindings but does
not claim it independently queried the Codex app-server. The snapshot and its
integrator decision remain supplemental; only the subsequent server-owned
strict-router decision has direction-eligibility authority.

## Make the control shell visible

At the start of a turn, use `update_plan` to show the current phase and the
next human or agent action. Use concise `commentary` updates after meaningful
work to identify actual agent roles, tools invoked, files opened or changed,
the current diff, and the persisted checkpoint or `loop_id` when one exists.

Declare roles only when they are actually active. A compact role map may name
the outer-loop coordinator, Society reviewers, scientist gate, and registered
predictive dispatcher, but must distinguish an assigned sub-agent from a
future role. Surface actual paths and `git diff` output, not a reconstruction
of invisible background activity.

Before yielding or handing off, show the loop state, returned `next_action`,
open human gate, relevant artifact references, and changed files. This is
observability for the user, not a replacement for persisted authority.

## Run the native Codex Society panel

When the current Codex client exposes native collaboration tools, use the
fixed `codex-native-society-topology-v1` contract instead of simulating a panel
with repeated `route_chat` calls. The current task is the root conductor and is
not counted as a child. Spawn exactly ten child tasks: six primary reviewers,
then three cross-review or red-team tasks, then one integrator. Use the exact
roles, dependencies, and peer-review edges from
`brain_researcher.autoresearch.society.codex_native_topology`.

Use the custom-namespace operations by their generic model-facing names:
`spawn_agent`, `followup_task`, and `wait_agent`. Spawn every child with its
matching exact `task_name=<role>`, `model=gpt-5.6-sol`,
`reasoning_effort=xhigh`, `fork_turns=none`, and a
`message` containing its role contract, immutable packet, and only its declared
dependency memos. Treat those requested calls as intent, not runtime evidence.

Register and display only observed native state. Take `agentRole` only from the
persisted `Thread` source. Count a peer edge only from V2
`SubAgentActivity(kind=interacted)`, using the enclosing notification's thread
ID as sender and the activity's child thread ID as target. Verify each child's
actual model and reasoning effort through `thread/resume`. Report each child
task's real thread ID, role, parent thread ID, native status, display phase,
model, reasoning effort, and completed-turn state, plus every observed native
interaction or scheduling edge. Respect the client concurrency limit: with a
four-slot tree, the root may run at most three children concurrently. Launch
roles in dependency-ordered waves, waiting for and reusing freed slots; the
completed tree must still contain exactly ten descendants and all three required
peer `followup_task` routes. Address every sibling by its canonical
`/root/<recipient_role_id>` target:

- `evidence_cross_reviewer → /root/result_gate_auditor`
- `methods_scope_cross_reviewer → /root/method_assumptions_auditor`
- `adversarial_red_team → /root/statistics_spin_auditor`

Never invent a child, status transition, or review exchange from the requested
topology alone.

Keep two execution surfaces distinct:

- In the current Codex Desktop task, native children belong to this task and
  may appear in its task tree.
- In a GCE or pod-owned `codex app-server`, the children belong to that remote
  parent. Mirror their observed state into Brain Researcher events, but do not
  claim they are attached to the current Desktop task unless both clients are
  verified to share the same app-server session tree.

The native Society panel is collaboration-tools-only: no shell, file changes,
MCP, web, apps, or compute. Keep the strict zero-tools `CodexCLIRouter` as a
separate typed closeout layer. The explicit Launch Autoresearch Goal lane above
uses the host goal outside this panel. If native collaboration is unavailable
or the observed tree is incomplete, report `PANEL_INCOMPLETE`; do not replace
missing children with prose personas or treat panel failure as a scientific
null.

For a completed registered Sydnor episode, persist the panel at the fixed
episode-owned path `society/codex-native-panel-v1.json`, then call
`sydnor_episode_ingest_native_society_review` with only the server-issued
`episode_id` and the frozen `source_loop_revision`. The bridge revalidates the
registered adjudication, exact canonical packet binding, and observed native
topology before writing an immutable supplemental record. Its automatic
canonical roll-forward must remain `admission_required` while the reviewed
episode has no registered successor question. To continue, call
`sydnor_episode_admit_next_round` only with the exact scientist-approved
`next_question`, the frozen source revision, and
`scientist_confirmation=true`. The server derives the authenticated owner,
next-round identities, and one-candidate discovery policy, then materializes a
canonical episode only to `AWAITING_REWARD`. Never replace that human gate with
a caller path, panel prose, chat-derived question, or another spin run; neither
admission nor candidate materialization authorizes compute or scientific
acceptance.

## Preserve the human and Society boundaries

Use Society only for reward-blind review. Society may produce review evidence
and criticism, but cannot rank for reward, select a portfolio, approve launch,
or issue authorization. The scientist supplies reward and launch approval
through the persisted loop actions.

Use `autoresearch_episode_authorize` only when the active profile presents the
scientist approval action. It records `authorization_pending`; then use
`autoresearch_episode_dispatch_start` only when the active full MCP server
exposes that exact tool and the returned `next_action` is revision-bound to it.
The primary call order is `autoresearch_episode_dispatch_start` ->
`autoresearch_run_watch`. The start tool queues the registered control run and
returns its `run_id`; poll it with `autoresearch_run_watch`. `run_get` is only a
compatibility view for this path.

The queued registered dispatch re-reads the persisted approval, handoff, frozen
slice, preseal authority, program, commitment binding, and batch before it
advances. A persisted expired-worker lease can be reclaimed and only the exact
registered dispatch can be requeued. A separate program-specific terminal
reconciler acts only on already-bound natural-v2 canonical runs. These
control-plane recoveries neither create approval nor establish scientific
acceptance.

`autoresearch_episode_dispatch` remains the synchronous compatibility tool.
Its legacy registered closeout requires strict `CodexCLIRouter` for its typed
Society review lane and may retry only the exact persisted `EXECUTING` or
`SYNTHESIZING` recovery path.

Use the existing exact registered predictive authority rather than a chat
approval or cached packet. The dispatcher re-reads the persisted approval,
handoff, frozen slice, preseal authority, program, commitment binding, and
batch at dispatch. Never equate an approval checkpoint, a plan, or a local
test with a launched run.

The ten-child native panel is a supplemental review and scheduling layer; it
cannot overwrite canonical claim cards or deterministic adjudication. Neither
layer is the main outer-loop agent, a reward source, an authority issuer, or a
compute runner. In particular, the native panel is not the main outer-loop
agent.

## Close and continue honestly

In `SYNTHESIZING`, preserve the current evidence and ask the authoritative
loop/profile for its closeout action. Create or conduct a next round only from
a persisted reviewed transition. Do not derive a next round from chat,
unreviewed findings, or a summary alone.

The control shell may use either native children in the current task or a
deployed headless supervisor with mirrored events. Neither surface authorizes
live compute: do not invoke dispatch, start a service, launch a scientific
model, or run an experiment unless the user explicitly requests that
already-approved live action. Control-plane evidence remains distinct from a
production receipt, scientific validity, and report readiness.

## Handoff

Return a compact block with the loop or review state, terminal status, bundle
path, submission result, exploratory-only boundary, open human gate, and exact
next invocation. Write the Brain Researcher session snapshot before ending the
research turn.

Client metadata and the default invocation prompt live in
[`agents/openai.yaml`](agents/openai.yaml).
