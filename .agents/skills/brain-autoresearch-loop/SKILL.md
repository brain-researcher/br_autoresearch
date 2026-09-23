---
name: brain-autoresearch-loop
description: Run or resume the single persistent, human-gated Brain Researcher autoresearch loop from the current Codex conversation, including Society review, scientist reward and launch approval, visible MCP checkpoints, and explicitly requested native-goal exploration in an already-open GOAL.md/DATASETS.md workspace. Use for outer-loop control, approved registered dispatch, reviewed next-round continuation, or explicit native-goal exploration; not for implicit goals, direct compute, or background service launch.
---

# Brain Autoresearch Loop

Use the current Codex conversation as the visible control shell for one
human-gated campaign. The canonical Brain Researcher MCP service owns campaign
state and actions. Chat and local Markdown are projections, never authority.

## Detect state first

At the start of every campaign turn:

1. Inspect server_info.
2. Inspect loop_profile_get for codex_autoresearch_v1.
3. Use only tools exposed by that server. If the profile or a required tool is
   absent, report the integration gap; do not simulate it.
4. Resume an existing campaign with autoresearch_loop_get and its persisted
   loop_id. Use autoresearch_loop_init only when the scientist is explicitly
   starting a new campaign.
5. When a Goal handoff exists, inspect autoresearch_goal_get before acting.
6. Treat returned state, revision, actions, terminal reservation, and
   next_action as authoritative. Re-read them before every mutation or replay.

The normal state order is:

TASK_DRAFT → DISCOVERING → REVIEWING → AWAITING_REWARD → SELECTING →
AWAITING_LAUNCH_APPROVAL → EXECUTING → SYNTHESIZING → COMPLETE.

A submitted closed_no_candidate or technical_failure terminal instead takes
the server-validated early-close path from DISCOVERING to COMPLETE, with no
Society, reward, approval, execution, or scientific transition. Until the
exact terminal submission is persisted and reconciled, state remains
DISCOVERING.

## Route by intent and observed state

Read only the reference selected below, plus any reference it directly routes
to. Read each selected reference completely before acting.

| Intent or observed state | Required reference | Route |
| --- | --- | --- |
| Inspect, explain, diagnose, or report status | This entrypoint | Query canonical state read-only and hand off the observed next action. |
| TASK_DRAFT | [Campaign protocol](references/campaign_protocol.md) | Follow only the returned drafting or scientist-decision action. |
| Start a new outer campaign or apply a generic loop action | [Campaign protocol](references/campaign_protocol.md) | Follow the persisted profile, state machine, and human gate. |
| Explicitly start or resume native-goal exploration; no handoff yet | [Native Goal launch](references/native_goal_launch.md) | Verify the open episode, prepare the handoff, and create the host goal exactly once. |
| DISCOVERING, V3 handoff, before terminal outcomes | [V3 discovery](references/discovery_v3.md) and [workspace projections](references/workspace_projections.md) | Freeze outcome-blind selection provenance, then run bounded exploration. |
| DISCOVERING, already-frozen V1 or V2 handoff, before terminal outcomes | [V1/V2 compatibility discovery](references/discovery_compatibility.md) | Preserve the exact historical contract and skip V3-only requirements. |
| DISCOVERING with a terminal bundle, rejected submission, or interrupted bridge | [Terminal submission and review](references/terminal_submission_review.md) | Validate, submit, or replay only the exact state-authorized terminal. |
| candidate_ready, or REVIEWING with panel work requested by next_action | [Terminal submission and review](references/terminal_submission_review.md), then [native Society panel](references/society_panel.md) | Freeze declared artifacts or run the exact reward-blind panel; preserve and stop on a persisted review block. |
| REVIEWING with review_blocked, panel_incomplete, or no panel next_action | [Terminal submission and review](references/terminal_submission_review.md) | Preserve the persisted refusal or incomplete panel and stop. |
| AWAITING_REWARD | [Campaign protocol](references/campaign_protocol.md) | If this message supplies scientist-authored reward, persist the exact action; otherwise stop for it. |
| SELECTING or AWAITING_LAUNCH_APPROVAL | [Terminal submission and review](references/terminal_submission_review.md) | Prepare or launch only when the current scientist message supplies the required decision; otherwise stop at the gate. |
| EXECUTING or SYNTHESIZING | [Confirmation and closeout](references/confirmation_closeout.md) | Watch the bound run and follow only its registered terminal route. |
| Registered predictive dispatch or recovery | [Confirmation and closeout](references/confirmation_closeout.md) | Re-read approval and dispatch authority; execute only the exact returned action. |
| COMPLETE with a registered Sydnor panel-ingestion or next-round-admission next_action | [Native Society panel](references/society_panel.md) | Follow this specific registered continuation before the generic COMPLETE route. |
| COMPLETE | [Campaign protocol](references/campaign_protocol.md) | Report the terminal record; do not manufacture a successor. |

If intent and state disagree, state wins. Within one state, the most specific
state-plus-next_action row wins; generic COMPLETE applies only when no narrower
registered route exists. Explain a mismatch and surface the server's
next_action. Never use local notes or an old terminal to replay an action
without querying canonical state again.

## Authority boundaries

- The scientist alone supplies reward and launch approval.
- Society is reward-blind. It may review evidence but cannot rank rewards,
  select a portfolio, approve launch, authorize compute, create a ClaimCard, or
  update Landscape.
- Native-goal launch requires an unambiguous scientist request or a
  scientist-invoked project launcher. Drafting, checking, organizing, or making
  an episode runnable is not launch authority.
- A native Goal is exploratory. Submission and review do not constitute
  confirmation, canonical execution, scientific acceptance, or Landscape
  transition.
- Never dispatch, start a service, launch a scientific model, or run an
  experiment unless the user explicitly requests that already-approved live
  action and canonical state authorizes it.
- Negative and technical terminals are legitimate closeouts and bypass
  Society. Do not turn them into review candidates.
- Frozen contracts, bundles, packets, decisions, and terminal observations are
  immutable. Recovery may replay only the exact persisted content authorized
  by next_action.
- The active episode is the only writable scientific workspace. Treat inputs/
  as read-only and put new scientific artifacts under outputs/. Follow
  [workspace projections](references/workspace_projections.md).

Use autoresearch_loop_apply_action, autoresearch_loop_checkpoint, and
autoresearch_loop_record_reward only when the live profile exposes the exact
tool and current state requests it. Tool availability is not permission;
returned revision-bound next_action is the gate.

## Keep the control shell visible

Use update_plan at the start of work. After meaningful work, send concise
commentary that identifies:

- the current phase and active roles;
- tools invoked and observed MCP checkpoint or loop_id;
- files opened or changed and the actual diff; and
- the next human or agent action.

Declare a role only while it is active. Distinguish an assigned subagent from a
future reviewer. Show observed paths and task-tree state; never reconstruct
invisible background activity.

## Required handoff

Before yielding, re-read the relevant canonical state and return one compact,
self-contained handoff containing:

- loop_id or review identity and authoritative state;
- terminal status, if any;
- CandidateBundle path and immutable hash, when applicable;
- submission, artifact-freeze, panel, or run result actually observed;
- the exploratory-only and non-acceptance boundary;
- the open human gate;
- exact returned next_action or exact next invocation;
- changed files and relevant artifact references; and
- unresolved integration gaps or failed attempts.

Write the Brain Researcher session snapshot before ending a research turn.
Client metadata and the default prompt are in
[agents/openai.yaml](agents/openai.yaml).
