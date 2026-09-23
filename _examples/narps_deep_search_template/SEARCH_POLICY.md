# Reference search policy: `narps_adaptive_smoothing_search_v1`

## Purpose

This policy makes multi-round development the registered program rather than
asking generic Goal discovery to select two terminal tests. It has no canonical
authority until the Brain Researcher service registers it and returns a bound
`search_policy_ref`.

## Immutable policy summary

```yaml
policy_id: narps_adaptive_smoothing_search_v1
policy_status: draft_unregistered
development_dataset: ds001734
audit_dataset: ds000005
minimum_valid_trials: 30
maximum_valid_trials: 60
maximum_failed_attempts: 15
patience_valid_trials: 12
minimum_incumbent_delta_esc: 0.01
maximum_composed_operators: 3
cpu_core_hour_ceiling: 2500
scratch_ceiling_tb: 1.5
wall_clock_ceiling_hours: 96
gpu_ceiling: 0
audit_open_count: 1
```

## Trial state machine

```text
PROPOSED
  -> READINESS_REJECTED
  -> RUNNING
  -> ENGINEERING_FAILED
  -> INVALID
  -> SCORED
  -> RETIRED | CHALLENGER | INCUMBENT
```

`INCUMBENT` is nonterminal. Only configuration lock followed by the one-shot
audit may create a terminal CandidateBundle.

## Required append-only trial fields

Each line of `experiments.jsonl` must contain:

- trial ID, timestamp, parent trial, stage, and hypothesis family;
- scientific prediction, falsifier, strongest alternative, and expected cost;
- exact code/config/environment/source/split hashes;
- changed operator and all unchanged operators;
- fold- and environment-level metrics, aggregate score, and uncertainty;
- negative-control and ablation status;
- runtime, memory, CPU-hours, exit state, and failure reason;
- belief update, branch disposition, and successor proposals; and
- incumbent/challenger decision with the frozen rule that produced it.

Rows may be superseded by later rows but never rewritten or deleted.

## Scheduler

1. Satisfy branch coverage with two valid trials in each admissible family.
2. Prefer the highest expected information gain per CPU-hour, estimated only
   from development history.
3. After branch coverage, allocate at least 40% of remaining trials to
   falsification, ablation, replication, or negative controls rather than
   score improvement.
4. A composed mechanism is admissible only if each component has an interpretable
   single-operator result.
5. Engineering failures may be repaired without consuming a valid trial only
   when scientific configuration is byte-identical.
6. Search stops only under the budget, qualified-patience, or queue-exhaustion
   rules in `GOAL.md`.

## Configuration lock and audit

The lock freezes the winner, runner-up, baselines, code, environment, data
manifests, metrics, thresholds, uncertainty, and report template. The audit
runner accepts only that lock hash. It must reject a second invocation and any
configuration that differs from the lock.

Audit output may assign one of the four predeclared conclusion classes. It may
not enqueue a trial, promote the runner-up, or mutate the ledger.

## Canonical integration requirement if reused

Before any future run, the service must expose a registered-program route
that binds this policy. If only generic V3 `bounded_ranked_candidates` is
available, the proposed run remains blocked; silently approximating this
program by two selected candidates would recreate the historical depth failure.
