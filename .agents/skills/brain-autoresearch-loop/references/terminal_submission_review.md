# Terminal submission and Goal review bridge

Use this reference when a native Goal in DISCOVERING has a terminal bundle to
submit, when an exact terminal replay is required, or when a candidate_ready
handoff is entering Goal Society review.

Always inspect the live tool schema and re-read autoresearch_goal_get before a
state-changing call. Local files and an earlier terminal do not establish the
current bridge state.

## CandidateBundle boundary

outputs/candidate_bundle.json must contain:

- the exact schema version frozen by the handoff:
  - br.autoresearch_goal_candidate_bundle.v3 with GoalExplorationTraceV2 for
    the enhanced V3 lane;
  - br.autoresearch_goal_candidate_bundle.v2 with GoalExplorationTraceV1 for a
    frozen V2 compatibility handoff; or
  - br.autoresearch_goal_candidate_bundle.v1 for a frozen V1 handoff;
- terminal_status equal to candidate_ready, closed_no_candidate, or
  technical_failure;
- a non-empty research_question for every terminal status;
- unique outputs/-relative output_artifacts that name primary result artifacts;
- limitations, deviations, and failed_attempts lists;
- for candidate_ready: hypothesis, prediction, falsifier, primary test,
  controls, development_summary, and reproduction;
- isolation_assurance equal to client_unattested;
- scientific_status equal to exploratory_only;
- confirmation_requirement equal to
  independent_fresh_confirmation_required;
- confirmation_eligible equal to false; and
- scientific_acceptance equal to false.

experiment_log.md may be an explicitly declared optional supplemental
artifact. Never declare memory.md, governance.md, society.md, loop.md,
landscape.md, or verification.md. See
[workspace projections](workspace_projections.md).

A negative or technical close is a valid terminal exploration result, but no
bundle can claim that a hypothesis is confirmed or supported. A V3 technical
trace may be partial or unavailable_before_freeze only when it records the
technical reason and failed attempts and proposes only stop or retry_technical.
It must not invent a phenomenon, novelty position, belief update, or
outcome-derived ideation mapping.

Never call autoresearch_episode_authorize, autoresearch_episode_launch, or an
equivalent dispatch route from native discovery.

## Submit and reconcile

1. Inspect the live autoresearch_goal_submit schema.
2. Submit the server-issued goal_handoff_id and the parsed candidate_bundle.
   The server records the handoff; it does not inspect the local workspace.
3. If submission is unavailable or rejected, leave the host goal active and
   report the exact integration gap.
4. After acceptance, call autoresearch_goal_get.
5. Refresh governance.md and loop.md only from returned MCP facts. Do not infer
   a stage, Society result, reward, approval, execution, or scientific status.
6. Call update_goal({status: "complete"}) only after a terminal bundle exists
   and submission was accepted. This completes client-native exploration, not
   its governance review.

For candidate_ready, do not put a verdict in society.md until review facts
exist.

## Negative or technical early close

For closed_no_candidate or technical_failure, require the bridge to show:

- outer loop COMPLETE;
- next_action null;
- Society not eligible and not called;
- no reward, launch approval, execution, scientific transition, acceptance, or
  applied Landscape transition.

Refresh governance.md, loop.md, society.md, and landscape.md from that response
and stop.

If submission fails before this state appears:

1. Re-read autoresearch_goal_get.
2. Retry only the identical frozen terminal submission when next_action is
   resume_exact_terminal_submission or retry_terminal_reconciliation.
3. Never edit the bundle to make a replay pass and never advance to Society.
4. If closed_by_other_handoff is returned, stop. This handoff is
   non-recoverable.

A pre-reservation legacy negative or technical terminal may replay only its
exact persisted bundle. The server rechecks the original DISCOVERING revision.

## Candidate-ready artifact freeze

1. Re-read autoresearch_goal_get and require the candidate-ready submission.
2. For every declared output_artifact, call autoresearch_goal_artifact_put with
   its exact outputs/-relative reference, content, and media type.
3. Do not upload absolute paths, inputs/, weights, or undeclared files.
4. Call autoresearch_goal_review_prepare with goal_handoff_id.
5. Retain review_packet_id, direction_id, and the exact immutable packet.

Review preparation freezes reward-blind exploratory evidence and advances the
bound loop from DISCOVERING to REVIEWING. It creates no episode and grants no
reward or execution authority. Update society.md, governance.md, and loop.md
only from the observed response; do not author a verdict.

## Goal Society review

Before running a panel, inspect the published society_panel schema and read
[native Society panel](society_panel.md).

Submit the complete observed native panel payload with
autoresearch_goal_review_submit. Never submit a bare caller-authored verdict or
eligible direction ID. The service validates packet binding, fixed topology,
typed memos, and reviewer provenance, then runs its strict server-owned router.
The caller panel is supplemental; only the server-owned typed decision controls
direction eligibility.

- If the server decision is eligible, require loop state AWAITING_REWARD and
  stop for scientist-authored reward.
- For revise, reject, abstain, or panel_incomplete, preserve the refusal and
  stop at review_blocked in REVIEWING.

The latter outcomes cannot open the reward gate or silently reopen discovery.
Refresh society.md, governance.md, and loop.md only from MCP.

## Review bridge invariants

The bridge freezes only artifacts declared by the accepted CandidateBundle.
autoresearch_goal_get is the resumable phase projection. Exact artifact,
packet, and panel replays may resume an interrupted handoff; changed content or
a panel bound to another packet must fail closed.

Direction Society decides only whether a candidate may enter later scientist
reward ranking. reward_eligible is not confirmation_eligible. Society cannot
supply reward, select a portfolio, approve launch, create a ClaimCard,
authorize execution, or update Landscape.

Every child memo repeats the exact review_packet_id and direction_id, its own
fixed role_id, a bounded summary, strongest objection, and required
confirmation changes. Only society_integrator may contain decision and
reward_eligible. The submitted task-tree snapshot must use
provenance_assurance client_submitted_native_snapshot.

## Confirmation bridge

After the scientist explicitly records reward and the loop deterministically
selects one direction:

1. Call autoresearch_goal_confirmation_prepare with goal_handoff_id and the
   current revision.
2. Retain episode_id and next_action. Preparation grants no execution
   authority.
3. Only after the scientist explicitly approves that frozen episode, inspect
   the live autoresearch_goal_confirmation_launch schema.
4. Call it with exactly episode_id, returned action_id, expected_revision, and
   scientist_confirmation true.
5. Do not separately call generic canonical authorize or launch tools.

A successful launch response must contain distinct run_id and canonical_run_id
and state EXECUTING. If it instead returns successor_episode_id and
next_action, stop for a new explicit scientist approval; never retry the
predecessor or reuse expired authority.

Watch the returned MCP run_id with autoresearch_run_watch. EXECUTING means only
that the approved provider run was accepted and bound. For terminal handling,
read [confirmation and closeout](confirmation_closeout.md).
