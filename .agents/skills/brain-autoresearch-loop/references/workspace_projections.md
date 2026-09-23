# Workspace projections and artifact layout

This reference applies whenever an episode workspace is initialized, explored,
submitted, reviewed, or handed off.

## Workspace boundary

- The open episode must already contain GOAL.md, DATASETS.md, inputs/, and
  outputs/.
- Treat inputs/ as read-only.
- Put every new scientific artifact under outputs/.
- Initialize the seven Markdown projections from templates/ at launch.
- Replace a not-observed placeholder only when supporting evidence exists.
- These files are readable delivery projections, never MCP authority.

## Exactly seven Markdown projections

1. outputs/experiment_log.md is an append-oriented lab notebook of attempted
   analyses, failures, deviations, result references, and uncertainty.
2. outputs/memory.md summarizes reusable local lessons, rejected approaches,
   failure modes, unresolved questions, and evidence references.
3. outputs/governance.md projects the latest observed Goal and loop state,
   handoff and terminal status, Society applicability or frozen outcome, open
   human gate, and its non-authority boundary.
4. outputs/society.md is stage-aware. Before a review packet is frozen, retain
   the stub and state that Society has not been called. During review, update it
   only from observed goal-get, review-prepare, and review-submit responses.
5. outputs/loop.md projects observed loop state, revision, actions, and
   transitions.
6. outputs/landscape.md separates proposed, permitted, and applied Landscape
   changes. Never infer a transition that MCP did not return.
7. outputs/verification.md records mechanical checks and explicitly states that
   scientific validity is not mechanically verified.

Do not add an eighth status or provenance Markdown file such as trajectory.md,
usage.md, review.md, or report_status.md. The typed trace, usage ledger, and
closeout records are JSON. outputs/idea_search.json is structured
client-maintained provenance, not an eighth Markdown projection. Ordinary
primary scientific outputs remain allowed.

## Projection authority

- memory.md is not Brain Researcher memory and must not be promoted unless the
  scientist separately requests that write.
- governance.md, society.md, loop.md, landscape.md, and verification.md cannot
  grant Society, reward, approval, execution, ClaimCard, memory, or Landscape
  authority.
- For a negative or technical terminal, governance.md and society.md must say
  Society: not eligible and not called.
- Refresh projections only from observed evidence. Do not reconstruct a
  missing canonical event from local notes.

## CandidateBundle artifacts

For candidate_ready, output_artifacts must explicitly declare the primary
result artifacts used for Society review. experiment_log.md may be declared
only as optional supplemental context.

Never declare these six projections in CandidateBundle V1, V2, or V3
output_artifacts:

- outputs/memory.md
- outputs/governance.md
- outputs/society.md
- outputs/loop.md
- outputs/landscape.md
- outputs/verification.md

The server-owned artifact snapshot is immutable review evidence. It is not a
local EpisodePaths execution artifact. Upload only declared outputs/-relative
files; never upload absolute host paths, inputs/, model weights, or undeclared
files.
