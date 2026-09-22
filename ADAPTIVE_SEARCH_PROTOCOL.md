# Common adaptive-search protocol

## Authority boundary

This protocol specifies the requirements for a registered program. A local
contract, controller implementation, or successful development run is not
canonical registration, launch approval, confirmation, scientific acceptance,
or a Landscape update.
The episode-specific policy must bind non-null `canonical_program_id`,
`registered_program_policy_ref`, and `search_policy_ref` before launch.

## Why search is long but not unbounded

An agent can generate hypotheses indefinitely; a finite dataset cannot supply
independent information indefinitely. Repeated adaptation to the same scores
eventually overfits the development environments even when ordinary CV is
correct. Therefore:

- agent reasoning and proposal generation may be persistent;
- each scientific round has a finite grammar, trial range, resource ceiling,
  and qualified-patience rule;
- development outcomes may guide successor hypotheses only inside that round;
- audit outcomes never guide another trial in the same round; and
- continuing after audit requires a new round, a new exposure record, and a
  genuinely new audit source.

## Immutable before candidate-discriminating development

Freeze and hash:

1. scientific estimand, observation and leakage units;
2. source identities, eligibility, development roles, audit identity, and
   exposure ledger;
3. operator grammar and prohibited operations;
4. metrics, constraints, environment aggregation, uncertainty, and meaningful
   improvement thresholds;
5. trial schema, branch-coverage requirement, promotion and tie rules;
6. minimum/maximum valid trials, failed-attempt allowance, patience, CPU/GPU,
   memory, storage, and wall ceilings;
7. synthetic calibration, negative controls, mandatory falsifiers, and
   ablations;
8. configuration-lock manifest, trusted audit command, retry policy, terminal
   classes, and claim boundary.

The exact sequence of admissible hypotheses is deliberately not frozen.

## Program stages

### Stage 0: readiness and synthetic calibration

Validate schemas, split isolation, evaluator behavior, nulls, metrics,
determinism, resource metering, and audit rejection on synthetic fixtures.
Freeze tolerances without reading candidate-discriminating audit outcomes.

### Stage 1: branch coverage

Run a space-covering set of valid development trials. Every required grammar
family must be exercised or receive a documented input-readiness rejection.
An invalid or duplicate configuration does not satisfy coverage.

### Stage 2: adaptive hypothesis search

Each proposed challenger must name its parent, directional prediction,
falsifier, changed operator, unchanged operators, expected information gain,
and expected cost. Development outcomes may update beliefs and choose the next
admissible challenger. A counted successor must also reference at least one
previously committed scored record and hash the complete ledger prefix visible
when it was proposed; a prespecified grid row cannot be relabeled as
outcome-adaptive later. The incumbent is a development state, never a terminal
candidate.

### Stage 3: falsification, ablation, and replication

At least 40% of trials after branch coverage must test an alternative,
negative control, ablation, influence guard, synthetic recovery, or direct
replication. A composite candidate requires all component ablations. Branch
failure retires that branch while another admissible branch and budget remain.

### Stage 4: configuration lock

After a valid stop event, freeze exactly one winner, declared baselines,
diagnostic runner-up if allowed, code, environment, data/split manifests,
operator DAG, hyperparameters, seeds, metrics, thresholds, retry rules, report
template, complete ledger, and audit command. A runner-up cannot replace the
winner after audit access.

### Stage 5: one-shot audit

A trusted, permission-separated runner accepts only the lock hash. It opens the
audit payload once, emits only predeclared outputs, records access receipts,
and permanently rejects subsequent development trials for the round. An audit
result may change the conclusion class but cannot enqueue a challenger.

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

Engineering repair may reuse a trial identifier only when the scientific
configuration and all data roles are byte-identical. Otherwise it is a new
trial. Records are append-only and hash-chained; correction creates a new
record that supersedes rather than rewrites an old one.

## Minimum realized-search evidence

Before an episode can count as adaptive-autoresearch evidence, it must have:

- satisfied its minimum valid-trial and branch-coverage requirements;
- completed at least two outcome-adaptive successor cycles;
- verified that counted successor proposals reference prior scored outcomes
  and were committed after those outcomes but before their own execution;
- recorded at least two incumbent/challenger decisions;
- satisfied the episode's minimum 40% post-coverage falsifier/ablation share;
- verified ledger integrity and deterministic score replay;
- locked configuration before audit access;
- opened the audit no more than once; and
- recorded zero post-audit mutations.

Passing these mechanical gates establishes search depth, not scientific truth.

## Stop events

A development round stops only on the first applicable frozen event:

- maximum valid trials;
- CPU/GPU/wall/storage ceiling;
- qualified patience after minimum trials and branch coverage;
- exhaustion of all admissible configurations; or
- a technical condition that makes every branch non-executable.

The first positive result, first negative result, appearance of an incumbent,
or failure of one branch is not a stop event. A scientist may stop a run, but
that interrupted run does not become verified adaptive-search evidence unless
the registered terminal contract explicitly permits it.

## Shared-source dependence

Episodes using the same participants, neurons, sessions, or audit groups must
share an exposure/split ledger. Separate questions do not create independent
replications. EP05--EP08 share LFP source dependence; EP09--EP11 share
SEU-A1876 dependence. Their reports must expose correlated evidence and may not
multiply one audit bank into several independent confirmations.

## Audit retry rule

A mechanical retry is allowed only when the runner proves that no audit label,
metric, prediction join, or candidate-discriminating diagnostic was emitted,
and the frozen policy names the exact infrastructure failure. Once any audit
metric is visible, the audit is consumed.

## Terminal encoding

Episode documents may name scientific `conclusion_class` values such as
`hybrid_supported`, `policy_improved`, `audit_failed`, or `unresolved`. Those
labels do not invent new outer-loop states. Unless a future registered program
explicitly defines a compatible typed extension, the canonical bundle status
remains one of:

- `candidate_ready` for an eligible positive conclusion;
- `closed_no_candidate` for valid negative, unresolved, exhausted, or
  development-only conclusions; or
- `technical_failure` for invalid inputs, execution, leakage, policy, or audit
  integrity.

The episode policy must map every conclusion class to one of these outer
statuses before launch. A richer local label cannot bypass Society, reward,
approval, or terminal validation.
