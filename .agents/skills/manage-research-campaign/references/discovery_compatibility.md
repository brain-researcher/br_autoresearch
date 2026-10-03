# Frozen V1 and V2 discovery compatibility

Use this reference only when autoresearch_goal_get proves that the active
handoff was already frozen under CandidateBundle V1 or V2.

## Preserve the frozen contract

- Never upgrade, translate, or wrap the handoff as V3.
- Do not run the V3-only ideation prelude or create
  GoalExplorationTraceV2.
- Inspect the live schema and retained handoff before constructing a terminal.
- A V1 handoff accepts only CandidateBundleV1.
- A V2 handoff accepts only CandidateBundleV2 with its required
  GoalExplorationTraceV1.
- Preserve the original research intent, candidate boundary, and every frozen
  field. Do not infer V3 fields or rewrite pre-outcome decisions.

## Explore and record

The coding-agent harness may perform the bounded inspect, code, execute,
observe, and repair loop authorized by the frozen handoff. Stay within the open
episode, keep inputs/ read-only, put durable artifacts under outputs/, use the
episode scratch directory for transient work, and use Slurm for expensive
compute.

Maintain the seven files in
[workspace projections](workspace_projections.md). Before closing, write and
parse outputs/candidate_bundle.json against the exact frozen version. Preserve
limitations, deviations, failed attempts, reproduction details, and evidence
for terminal status.

Submit observed discovery usage using the handoff's stable identifiers and the
same null-means-unavailable rule described by the live usage schema. Never
assert owner, reward, scientific outcome, or authority as usage telemetry.

When the exact compatibility bundle reaches a genuine terminal, follow
[terminal submission and review](terminal_submission_review.md). Its common
terminal, replay, artifact, review, and human-gate rules apply, while the
CandidateBundle schema remains the frozen V1 or V2 version.
