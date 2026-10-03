---
name: manage-research-campaign
description: >-
  Manage explicitly requested formal Brain Researcher campaign state, Society
  review, scientist reward, registered confirmation, or Landscape transitions.
  Not for ordinary episode design, execution, or advisory scientific critique.
---

# Manage a Research Campaign

## Applicability gate

This is the optional formal state-management workflow, not the entrypoint for
generating or running ordinary episodes. Use `design-episode` for new studies
and redesign, and `run-episode` for standalone execution and analysis. Neither
advisory scientific critique nor an ordinary Codex goal binds a campaign.

Use this skill only for a scientist's explicit request about the canonical
Brain Researcher campaign, Society/reward, registered handoff, confirmation or
Landscape. Naming this skill alone does not approve every action: a status
question stays read-only and each mutation still needs its own current authority.

Use the conversation as the visible control shell for one human-gated campaign.
The canonical MCP service owns state and actions; conversation and local
Markdown make them visible but grant no authority.

## Scientific decisions in an authorized campaign

Read [scientific decisions](references/scientific_decisions.md) when authorized
discovery, design critique or result interpretation needs scientific assistance.
A status-only turn does not activate that work. Its advisory review does not
replace formal Society or any frozen selection/confirmation contract.

## Detect state first

At the start of every canonical campaign turn:

1. Inspect server_info and loop_profile_get for codex_autoresearch_v1.
2. Resume an existing campaign with autoresearch_loop_get and its persisted
   loop_id. Use autoresearch_loop_init only when the scientist is explicitly
   starting a new campaign.
3. When a Goal handoff exists, inspect autoresearch_goal_get before acting.
4. Treat returned state, revision, actions, terminal reservation, and
   next_action as authoritative. Re-read them before every mutation or replay.

Use only exposed tools. An absent profile or required tool leaves its step
blocked: report the integration gap without simulating it, substituting a
profile, or treating another workflow as authorized fallback.

The normal state order is:

TASK_DRAFT -> DISCOVERING -> REVIEWING -> AWAITING_REWARD -> SELECTING ->
AWAITING_LAUNCH_APPROVAL -> EXECUTING -> SYNTHESIZING -> COMPLETE.

A closed_no_candidate or technical_failure terminal takes the server-validated
DISCOVERING -> COMPLETE path without Society, reward, approval, execution, or
scientific transition. State remains DISCOVERING until the exact terminal
submission is persisted and reconciled.

## Route by intent and observed state

Read only the reference selected below, plus any reference it directly routes
to. Read each selected reference completely before acting.

| Intent or observed state | Required reference | Route |
| --- | --- | --- |
| Inspect, explain, diagnose, or report status | This entrypoint | Query canonical state read-only and hand off the observed next action. |
| TASK_DRAFT | [Campaign protocol](references/campaign_protocol.md) | Follow only the returned drafting or scientist-decision action. |
| Start a new outer campaign or apply a generic loop action | [Campaign protocol](references/campaign_protocol.md) | Follow the persisted profile, state machine, and human gate. |
| Explicitly bind native-goal exploration to the canonical campaign; no handoff yet | [Native Goal launch](references/native_goal_launch.md) | Verify the open episode, prepare the handoff, and create the host goal exactly once. |
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

State restricts which requested actions are legal; it never expands a status
question into permission to act. Within one state, the most specific
state-plus-next_action row wins; generic COMPLETE applies only when no narrower
registered route exists. Explain a mismatch and surface the server's
next_action without executing an unrequested action.

## Authority boundaries

- The scientist alone supplies reward and launch approval.
- Society is reward-blind. It may review evidence but cannot rank rewards,
  select a portfolio, approve launch, authorize compute, create a ClaimCard, or
  update Landscape.
- A native Goal is exploratory. Submission and review do not constitute
  confirmation, canonical execution, scientific acceptance, or Landscape
  transition.
- Never dispatch, start a service, launch a scientific model, or run an
  experiment unless the user explicitly requests that already-approved live
  action and canonical state authorizes it.
- Do not turn negative or technical closeouts into review candidates.
- Frozen contracts, bundles, packets, decisions, and terminal observations are
  immutable. Recovery may replay only the exact persisted content authorized
  by next_action.
- The active episode is the only writable scientific workspace. Treat inputs/
  as read-only and put new scientific artifacts under outputs/. Follow
  [workspace projections](references/workspace_projections.md).

Use autoresearch_loop_apply_action, autoresearch_loop_checkpoint, and
autoresearch_loop_record_reward only when exposed by the live profile and
requested by the returned revision-bound next_action.

## Keep the control shell visible

Use a host planning tool when available and useful; otherwise state the next
authorized action in commentary. After meaningful changes, report the observed
phase, active roles, MCP checkpoint or loop_id, relevant file changes, and next
human or agent action. Distinguish assigned workers from future reviewers;
never reconstruct unobserved background activity.

## Required handoff

Before yielding, re-read the relevant canonical state and return one compact,
self-contained handoff containing:

- loop_id or review identity, authoritative state, and any terminal status;
- observed submission, artifact-freeze, panel, or run result, with CandidateBundle
  path and immutable hash when applicable;
- the exploratory-only/non-acceptance boundary, open human gate, and exact
  returned next_action or next invocation;
- changed files and relevant artifact references; and
- unresolved integration gaps or failed attempts.

Write the Brain Researcher session snapshot before ending a state-changing
canonical research turn. A read-only status/explanation request does not
authorize writing a snapshot. If a required snapshot tool is unavailable,
report the unmet handoff requirement rather than inventing a substitute.
Client metadata and the default prompt are in
[agents/openai.yaml](agents/openai.yaml).
