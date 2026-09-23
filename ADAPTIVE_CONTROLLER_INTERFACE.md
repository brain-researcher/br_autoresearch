# Trusted adaptive controller interface

## Purpose

The episode-owned controller converts agent-authored scientific hypotheses
into validated configurations inside that episode's frozen grammar. It
schedules trials, scores held-out development roles, updates the
incumbent/Pareto archive, creates a configuration lock, and invokes the
episode's held-out evaluation path.

The agent proposes; the controller validates and executes. Arbitrary candidate
code is not an acceptable substitute for grammar enforcement.

## Required operations

```text
load_policy(policy, source_manifest, split_manifest)
validate_config(config) -> accepted | rejected(reason)
register_hypothesis(parent, proposal_mode, proposal_context_hash,
                    outcome_evidence_refs, successor_cycle_id, prediction,
                    falsifier, changed_operator, unchanged_operators,
                    expected_information_gain, expected_cost, config)
run_development_trial(trial_id) -> blinded_evaluator_outputs
apply_falsifiers(trial_id) -> control_status
score_trial(trial_id) -> environment_metrics, constraints
update_archive(trial_id) -> retired | challenger | incumbent
run_full_search_null(generator_lock_hash, replicate_manifest)
    -> null_distribution, monte_carlo_p, replay_receipts
check_stop() -> continue | stop(reason)
lock_configuration(incumbent_id) -> immutable_lock_hash
run_audit(lock_hash) -> one_shot_audit_receipt
```

## Trust boundary

- Every episode must enforce its declared role visibility through the
  controller/loader API. Search fitting functions receive only their permitted
  inputs and configuration; held-out labels are excluded from fitting and
  proposal decisions.
- Physical process, Unix-identity, network, cache, or mount separation is
  required only when the episode policy claims that stronger boundary. A
  same-user logical boundary is allowed for an internal episode but must be
  disclosed and cannot be described as cryptographically sealed or independent
  confirmation.
- If an episode chooses a separated audit environment, audit features and
  labels remain unavailable to proposal, search, cache, log, and development
  evaluator processes except for a declared feature-schema preflight.
- The controller rejects unregistered operators, data-dependent config schema
  changes, undeclared outputs, and any transform that requests forbidden
  held-out metadata.

## Persistence and recovery

The source of truth is a hash-chained JSONL ledger conforming to
`TRIAL_LEDGER.schema.json`. A derived SQLite database may accelerate queries but
is rebuildable and non-authoritative. Resume replays the ledger, verifies every
hash and referenced artifact, reconstructs the archive, and continues from the
same frozen policy. It never infers success from scheduler state alone.

For an `outcome_adaptive_successor`, the controller also verifies that every
outcome-evidence reference names a scored record committed before the proposal
timestamp, that at least one parent trial is named, and that the proposal
context hash commits to the complete visible ledger prefix. Prespecified grid
rows cannot be relabeled as adaptive successors after execution.

Lock and audit transitions are ledger records as well. A lock record commits
the immutable configuration hash; every audit record must name that same hash
and carry an access receipt. This makes audit-open count, mechanical failures,
and any forbidden post-audit development mutation replayable from the ledger.

For episodes that require a full-search null, `run_full_search_null` starts each
replicate from an empty ledger and reruns the complete controller under a
locked null generator. The grammar, budgets, stopping rule, proposal model,
prompt, sampling parameters, and seed manifest are identical to the observed
search. The controller emits one replay receipt per replicate and refuses to
open outcomes unless a pre-outcome dry profile shows that all registered
replicates fit the episode's total compute ceiling. A null that reruns only the
winning model, or reuses the observed search trajectory, is invalid.

## Multi-objective selection

The episode policy defines hard scientific constraints before preference
ordering. The controller preserves a Pareto archive whenever collapsing metrics
could reward a degenerate solution. A composite score may break ties only after
all hard constraints are visible. Complexity and resource use are explicit
tie-breakers, not hidden penalties.

## Optional canonical integration

The Brain Researcher `codex_autoresearch_v1` profile can provide outer-loop
review, reward, and claim governance. It does not need to host or attest this
inner controller before local episode execution. If a later confirmation is
registered, its receipt may bind the realized program, policy, and artifacts;
that optional binding does not retroactively authorize the run.
