# Society projection

> Client-maintained, stage-aware projection of observed MCP facts.
> Its review content begins only after packet freeze.
> Persisted MCP Goal and Society records remain authoritative. This file does
> not grant Society, reward, approval, execution, scientific, ClaimCard, memory,
> or Landscape authority. It is not a CandidateBundle review artifact and must
> never be declared in a frozen CandidateBundle's `output_artifacts`.

This projection does not grant Society authority.

## Binding

- Goal handoff: not observed
- Outer loop: not observed
- Review packet: not observed
- Direction: not observed
- observed_at: not observed
- source: not observed

## Society status

- Applicability: not observed
- Called: not observed
- Native panel snapshot: not observed
- Server-owned decision: not observed
- Strongest objections: not observed
- Required confirmation changes: not observed

## Stage rule

- Before `autoresearch_goal_review_prepare`: Society has not been called; do
  not write a verdict.
- For `closed_no_candidate` or `technical_failure`: replace the status above
  with `Society: not eligible and not called`.
- After `candidate_ready` review begins: update this projection only with
  observed MCP facts returned by `autoresearch_goal_get`,
  `autoresearch_goal_review_prepare`, or `autoresearch_goal_review_submit`.
  Do not infer, compose, or promote a verdict from local prose.

## Boundary

- A Society decision is not scientist reward, portfolio selection, launch
  approval, execution, scientific acceptance, or Landscape transition.
- This projection is a readable delivery artifact, not a second Society record.
