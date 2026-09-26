# Trusted adaptive controller interface

## Scope

This interface applies only when an episode has entered candidate-scoring
adaptive execution and actually implements a trusted controller.  It does not
govern data acquisition, source qualification, preliminary readiness, or a
simple non-adaptive episode.  Operations below are required only for features
activated by that episode's scientific policy; implementing the whole
interface is never a startup gate.

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
load_policy(policy, source_ref, split_roles)
validate_config(config) -> accepted | rejected(reason)
register_hypothesis(parent, proposal_mode, proposal_context_ref,
                    outcome_evidence_refs, successor_cycle_id, prediction,
                    falsifier, changed_operator, unchanged_operators,
                    expected_information_gain, expected_cost, config)
run_development_trial(trial_id) -> blinded_evaluator_outputs
apply_falsifiers(trial_id) -> control_status
score_trial(trial_id) -> environment_metrics, constraints
update_archive(trial_id) -> retired | challenger | incumbent
run_full_search_null(generator_lock_ref, replicate_plan)
    -> null_distribution, monte_carlo_p
check_stop() -> continue | stop(reason)
lock_configuration(incumbent_id) -> lock_id
run_audit(lock_id) -> one_shot_audit_result
```

These names describe scientific operations, not required artifact formats.  A
plain policy file, split table, ordered log, and uniquely identified write-once
or versioned lock record are sufficient by default; custom schemas,
cryptographic hashes, manifests, and receipts are not implied.

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
  evaluator processes except for the minimal input-readability check required
  by the actual evaluator.
- The controller rejects unregistered operators, data-dependent scientific
  configuration changes, undeclared outputs, and any transform that requests
  forbidden held-out metadata.

## Persistence and recovery

The source of truth is the simplest durable ordered trial log that contains the
fields the active controller consumes; JSONL, CSV, or SQLite are all acceptable.
`TRIAL_LEDGER.schema.json` is optional implementation support, not a readiness
gate.  Resume reads the recorded state and the particular outputs needed to
reconstruct the archive; it does not verify every file or hash by default.  It
never infers scientific success from scheduler state alone.

For an `outcome_adaptive_successor`, the controller also verifies that every
outcome-evidence reference names a scored record committed before the proposal
timestamp, that at least one parent trial is named, and that the proposal
record identifies the prior evidence visible at that point. Prespecified grid
rows cannot be relabeled as adaptive successors after execution.  A timestamp
or monotone sequence number is sufficient; a prefix hash is not required.

Lock and audit transitions are trial-log records as well. The audit record
names the selected lock identity, which resolves to the exact immutable or
versioned lock record, and records whether outcomes were exposed. This makes
audit-open count, mechanical failures, and any forbidden post-audit development
mutation inspectable without requiring a configuration hash or access receipt.

For episodes that require a full-search null, `run_full_search_null` starts each
replicate from an empty ledger and reruns the complete controller under a
locked null generator. The grammar, budgets, stopping rule, proposal model,
prompt, sampling parameters, and seed rule are identical to the observed
search when those elements affect the null. The controller records each
replicate's result and checks compute feasibility only when it is genuinely in
doubt. A null that reruns only the winning model, or reuses the observed search
trajectory, is invalid.

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
