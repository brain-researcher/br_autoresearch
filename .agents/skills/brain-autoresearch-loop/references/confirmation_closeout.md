# Confirmation execution and closeout

Use this reference only for an already-authorized confirmation or registered
episode, or when an authoritative run has entered EXECUTING or SYNTHESIZING.
Re-read the persisted loop, episode, approval, and returned next_action before
every mutation.

## Registered dispatch

Use autoresearch_episode_authorize only when the active profile presents the
scientist approval action. It records authorization_pending.

Then use autoresearch_episode_dispatch_start only when:

- the full live MCP server exposes that exact tool; and
- current next_action is revision-bound to it.

The primary order is:

1. autoresearch_episode_dispatch_start
2. autoresearch_run_watch

The start call queues the registered control run and returns run_id. run_get is
only a compatibility view for this path.

The queued dispatcher re-reads persisted approval, handoff, frozen slice,
preseal authority, program, commitment binding, and batch before advancing. A
persisted expired-worker lease may be reclaimed, but only the exact registered
dispatch may be requeued. A program-specific terminal reconciler acts only on
already-bound natural-v2 canonical runs.

These recoveries create neither approval nor scientific acceptance.

autoresearch_episode_dispatch remains the synchronous compatibility tool. Its
legacy registered closeout requires strict CodexCLIRouter for typed Society
review and may retry only the exact persisted EXECUTING or SYNTHESIZING
recovery path.

Never substitute chat approval, a cached packet, a plan, or a local test for
registered predictive authority or a launched run.

## Scientific-learning terminal boundary

An enhanced V3 discovery bundle does not authorize scientific-learning
closeout. Only a registered adopter that freezes GoalConfirmationContractV2
with ScientificLearningPolicyV1 enters this terminal consumer. A V1
confirmation contract and an already-frozen V2 compatibility contract retain
their program-specific closeout.

For explicit confirmation V2, watch the same canonical run and report only
observed terminal facts. Server reconciliation is:

1. canonical GoalConfirmationTerminalObservationV1 plus terminal evidence;
2. on success, server-owned ScientificReviewVerdict; on failed, cancelled,
   timeout, or skipped, TechnicalDispositionV1;
3. ScientificLearningCloseoutIntentV1;
4. write_report, revise_report, or none materialization;
5. ScientificLearningUsageSummaryV1;
6. ScientificLearningCloseoutV1; and
7. existing outer-loop COMPLETE compare-and-swap.

The closeout intent binds the future usage-summary reference before report
materialization so the summary may include review and report-generation usage.
A report is produced only for write_report. revise_report writes a revision
handoff; none writes neither.

autoresearch_run_usage is the run-scoped read surface for observed operational
coverage. Client-native OAK or Sherlock review is supplemental
client-attested evidence. It cannot replace server review, choose report
action, complete the transaction, establish scientific acceptance, or update
Landscape.

The terminal consumer is not an automatic reward, selection, approval, launch,
successor episode, ClaimCard, or Landscape transition. Its closeout is
non-promoting; later scientific or Landscape actions need their own persisted
authority.

Workspace isolation v0 is logical and client_unattested. It proves no mount,
sandbox, capability boundary, or confirmation-data firewall.

## Synthesis and continuation

In SYNTHESIZING, preserve current evidence and ask the active loop and profile
for the closeout action. Create or conduct a next round only from a persisted,
reviewed transition. Never derive it from chat, unreviewed findings, or a
summary alone.

The control shell may use native children or a deployed headless supervisor
with mirrored events. Neither authorizes live compute. Do not dispatch, start a
service, launch a scientific model, or run an experiment unless the user
explicitly requests that already-approved live action.

Control-plane evidence remains distinct from a production receipt, scientific
validity, and report readiness. Submission, review, and execution state are not
scientific acceptance or Landscape transition unless the server explicitly
records those facts.
