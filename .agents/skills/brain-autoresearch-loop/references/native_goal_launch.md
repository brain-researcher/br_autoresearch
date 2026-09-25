# Native Goal launch

Read this reference only when the scientist has explicitly asked to start or
resume native-goal exploration in the current Codex task. A request to draft,
inspect, repair, organize, or make an episode runnable is not launch authority.
An explicit start or resume instruction needs no second confirmation.

This is a client-native discovery lane bound to the outer campaign. It is not
canonical confirmation or execution.

## Optional goal-prompt prelude

Use this host-only, read-only authoring prelude only when the scientist
explicitly invokes goal-prompt or clearly asks to draft or review goal text
before launch. A launch request alone does not enable it.

1. Use the open GOAL.md and DATASETS.md as the research scenario and produce a
   scientist-confirmed research-intent brief.
2. Do not render a final host goal, enter a second authoring stage, create
   .goal-task, create or update a host goal, call MCP, or write the workspace.
3. After confirmation, exit the prelude. The launch below passes the confirmed
   brief verbatim as the only research_intent to autoresearch_goal_prepare.
4. Preserve the server-issued goal_objective unchanged for the single
   create_goal call.
5. If goal-prompt is unavailable, report optional prelude unavailable and use
   the scientist's stated intent. Do not simulate the prelude or block launch.

This prelude changes no CandidateBundle, Society, reward, approval,
confirmation, or Landscape boundary.

## Launch sequence

1. Verify that the already-open episode contains GOAL.md, DATASETS.md, inputs/,
   and outputs/. Read the two Markdown files. Treat inputs/ as read-only, write
   all new work under outputs/, and do not create or substitute a workspace.
2. Call get_goal({}) first in the current Codex task. If it reports an
   unfinished host goal, preserve it and ask whether to continue it. Do not
   prepare or create another goal.
3. Re-read the authoritative loop and require profile
   codex_autoresearch_v1 in DISCOVERING.
4. Inspect the live autoresearch_goal_prepare schema. Call it with loop_id,
   current expected_revision, research_intent, an idempotency key, and
   candidate_bundle_contract_version v3.
5. Retain the server-issued goal_handoff_id, goal_objective, and returned
   contract version. Continue the enhanced lane only if that version is exactly
   v3.
6. An omitted version or explicit v1 follows its compatibility contract. An
   already-frozen v2 handoff follows exactly v2. Never silently upgrade a
   frozen v1 or v2 handoff.
7. If binding is stale, unavailable, rejected, or returns another version, stop
   without creating a host goal.
8. Only after successful preparation, call create_goal exactly once in the same
   task with the server-issued goal_objective unchanged.

Do not retry create_goal, replace an unfinished goal, or call it from a new
task, subagent, server, or codex exec process. It creates neither a workspace
nor a server GoalRunner.

## Next phase

After successful launch, read [V3 discovery](discovery_v3.md) and
[workspace projections](workspace_projections.md) before accessing
candidate-discriminating target outcomes.
