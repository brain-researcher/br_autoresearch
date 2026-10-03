# Governance projection

> Reconstructable display of observed MCP facts only. Persisted MCP state remains authoritative.
> Record `observed_at` as the client observation timestamp. `source` names the
> MCP tool (usually `autoresearch_goal_get`); `schema_version` is that response's
> schema version.
> `canonical_confirmation_episode_created` is a scope description, not an
> `autoresearch_goal_get` field. Copy the exact response keys below and do not
> infer a confirmation episode.
> This file grants no reward, approval, execution, scientific,
> memory, Society, ClaimCard, or Landscape authority.

## Observed MCP record

- observed_at: not observed
- source: not observed
- schema_version: not observed
- goal_handoff_id: not observed
- loop_id: not observed
- revision: not observed
- outer_stage: not observed
- next_host_action: not observed
- submission_state: not observed

## Society

- Applicability: not observed
- Status: not observed
- Packet or outcome refs: not observed
- Detail projection: `outputs/society.md` (not observed)

For `closed_no_candidate` or `technical_failure`, replace this section with:
`Society: not eligible and not called`. Keep the same status in
`outputs/society.md`; neither projection grants Society authority.

## Human gates

- Scientist reward: not observed
- Launch approval: not observed

## Execution and confirmation boundary

- Client-native exploratory execution: see `outputs/experiment_log.md`; not asserted by MCP.
- server_execution: not observed
- canonical_episode_created: not observed
- landscape_transition_created: not observed
- scientific_acceptance: false
