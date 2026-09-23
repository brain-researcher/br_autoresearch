# Optional canonical-confirmation workflow

This workflow is retained for episodes that later request a canonical
confirmation, Society decision, reward, or shared campaign claim. It is not a
prerequisite for `bin/codex-episode launch`. An episode-managed run performs
its own bootstrap, validation, falsification, and held-out evaluation.

## Files and authority

- `READINESS_MODEL.yaml` defines readiness, execution, and evidence states.
- `EPISODE_READINESS.yaml` selects the active review and summarizes every
  formal episode.
- Episode-specific readiness evidence is optional during provisioning and is
  referenced from the root ledger only when a later readiness state requires
  it.
- `ADAPTIVE_EVIDENCE_POLICY.yaml` governs post-launch evidence acceptance.
- Brain Researcher MCP alone may attest `canonical_ready` for this optional
  confirmation lane and may govern accepted claims or Landscape transitions.

`SEARCH_POLICY.yaml` (or EP02's `SEARCH_POLICY.json`) is immutable contract
content. Its deprecated `status: planned_unregistered` and
`canonical_binding.launch_blocked: true` fields record the authoring-time
snapshot and are never edited during promotion. Current binding and readiness
belong only in the root readiness/registry projections backed by a canonical
receipt; this keeps the bytes named by the receipt stable.

The legacy `launch_blocked` boolean is canonical-confirmation compatibility
metadata. The local episode launcher ignores it.

## Optional prospective confirmation sequence

An already completed local episode is exploratory evidence. If canonical
confirmation is desired, apply this sequence before opening a genuinely new
held-out source; do not relabel outcomes already seen during the local run as
prospective confirmation.

1. **Independent audit.** Review scientific sufficiency, source integrity,
   role leakage, evaluator isolation, compute feasibility, prior exposure, and
   canonical identity collision. No candidate-discriminating outcome access.
2. **Scientist decision.** Resolve every choice that changes the estimand,
   audit unit, role assignment, margin, comparator, terminal rule, or claim
   boundary.
3. **Trusted provisioning.** Build content-addressed development and audit
   handoffs. Candidate workers never receive a mixed raw tree or evaluator-only
   labels.
4. **Synthetic qualification.** Exercise the controller, ledger, replay,
   configuration lock, audit-open rule, denial paths, and resource ceilings on
   fixtures.
5. **Local promotions.** Promote successively to `development_ready`,
   `audit_ready`, and `canonical_preflight_ready`; record evidence references
   for every passed gate.
6. **Canonical binding.** Obtain a fresh server receipt binding the unique
   episode identity, exact registered program version, policy hash, contract
   revision, Git commit, and full contract-manifest hash. A trusted verifier,
   not a human-edited receipt string, must confirm freshness and supersession.
   Only that verified receipt permits `canonical_ready`.
7. **Keep claims separate from execution.** `bin/codex-episode launch` needs an
   explicit user invocation, not `canonical_ready`. A later canonical receipt
   governs only the confirmation/claim workflow it explicitly binds.

If a pre-outcome contract changes, increment its revision and invalidate older
preflight receipts. If candidate-discriminating audit information is exposed
or a post-audit design change is proposed, do not roll the episode backward;
invalidate it or begin a new episode with a new audit source.

## Parallel-agent boundary

For the active episode, parallelize these read-only or disjoint workstreams:

- scientific estimand, split, comparator, and decision-rule review;
- source/provenance and role-firewall design;
- controller/evaluator and synthetic-test design; and
- canonical identity/history review.

Integrate them into one readiness manifest before any state transition. Never
let separate agents independently assign audit roles or create competing split
manifests.

## Current use

There is no portfolio-level readiness queue for local execution. EP07's
scientific split and operator contracts are frozen; its remaining data-value,
runtime, controller, validation, and held-out checks run inside EP07.
