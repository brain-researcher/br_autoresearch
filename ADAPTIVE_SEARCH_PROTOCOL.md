# Common adaptive-search protocol

## Authority boundary

This protocol is executed inside each explicitly started episode. A local contract,
controller implementation, or successful development run is not scientific
acceptance or a Landscape update, but no external registration, readiness
promotion, evaluator service, or canonical receipt is required to begin the
episode. Brain Researcher registration remains optional post-run governance for
review, reward, or confirmation claims.

This protocol governs candidate-discriminating adaptive execution only after
the episode reaches that stage.  It is not an acquisition or preliminary
readiness checklist.  Apply the minimum-necessary rule in `AGENTS.md`: preserve
the scientific decisions and role boundary, but do not add hashes, schemas,
receipts, manifests, reruns, or synthetic checks unless the active scientific
stage actually needs them.

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

Record and freeze the scientifically consequential decisions in the lightest
durable form that makes their pre-outcome timing clear.  A normal versioned
document or ordered trial log is sufficient; cryptographic hashing is not a
generic requirement:

1. scientific estimand, observation and leakage units;
2. source identities, eligibility, development roles, audit identity, and
   exposure ledger;
3. operator grammar and prohibited operations;
4. metrics, constraints, environment aggregation, uncertainty, and meaningful
   improvement thresholds;
5. trial fields consumed by the executor, branch-coverage requirement,
   promotion and tie rules;
6. minimum/maximum valid trials, failed-attempt allowance, patience, CPU/GPU,
   memory, storage, and wall ceilings;
7. any activated synthetic calibration, negative controls, mandatory
   falsifiers, and ablations;
8. configuration-lock record, held-out invocation, retry policy, terminal
   classes, and claim boundary.

The exact sequence of admissible hypotheses is deliberately not frozen.

## Program stages

### Stage 0: targeted bootstrap and any activated calibration

Check only the inputs and behavior needed by the episode's next executable
step, plus direct leakage and role-isolation risks.  Use a small synthetic
fixture only when executor logic, a metric, or a null implementation genuinely
needs a smoke test.  Do not build a generic schema, determinism, resource-
metering, or rejection battery.  Freeze any scientifically meaningful
tolerance without reading candidate-discriminating held-out outcomes.  A
failed item produces the direct proceed/revise/stop action or, when execution
is genuinely impossible, an episode-local technical terminal; it does not
justify a larger bootstrap framework or retroactively prevent task startup.

### Stage 1: branch coverage

Run a space-covering set of valid development trials. Every required grammar
family must be exercised or receive a documented input-readiness rejection.
An invalid or duplicate configuration does not satisfy coverage.

### Stage 2: adaptive hypothesis search

Each proposed challenger must name its parent, directional prediction,
falsifier, changed operator, unchanged operators, expected information gain,
and expected cost. Development outcomes may update beliefs and choose the next
admissible challenger. A counted successor must also reference at least one
previously recorded score and be entered before its own execution.  Ordinary
ordered records are enough to establish this sequence; no ledger-prefix hash
is required.  A prespecified grid row cannot be relabeled as outcome-adaptive
later. The incumbent is a development state, never a terminal candidate.

### Stage 3: falsification, ablation, and replication

At least 40% of trials after branch coverage must test an alternative,
negative control, ablation, influence guard, synthetic recovery, or direct
replication. A composite candidate requires all component ablations. Branch
failure retires that branch while another admissible branch and budget remain.

### Stage 4: configuration lock

After a valid stop event, freeze exactly one winner, declared baselines,
diagnostic runner-up if allowed, scientifically consequential operator choices,
hyperparameters, seeds when relevant, metrics, thresholds, split roles, retry
rules, report fields, and held-out invocation in one simple lock record.  Bind
code, environment, or file inventories only when their variation can change
the result or an observed inconsistency requires it.  A runner-up cannot
replace the winner after held-out access.  The lock record must be uniquely
identified and write-once or explicitly versioned before that access; a
cryptographic hash is optional.

### Stage 5: held-out evaluation

The episode's locked evaluation path accepts only the selected lock identity,
which resolves to the exact immutable/versioned record, and checks the
scientifically consequential settings against that record.
It applies the episode's declared held-out or audit rule, emits only predeclared
outputs, records access, and rejects subsequent development trials for the
round. A cryptographic lock hash or access receipt is optional unless the
episode's actual trust boundary requires it. An evaluation result may change
the conclusion class but cannot enqueue a challenger. Whether this path uses
process/permission separation is an episode design choice, not a root task-
startup requirement.

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
configuration, scientifically relevant source/release and consumed inputs, and
all data roles are unchanged. Otherwise it is a new trial. Scored adaptive
records are append-only; a correction creates a new record that supersedes
rather than rewrites an old one. A hash chain is not required.

## Minimum realized-search evidence

Before an episode can count as adaptive-autoresearch evidence, it must have:

- satisfied its minimum valid-trial and branch-coverage requirements;
- completed at least two outcome-adaptive successor cycles;
- verified that counted successor proposals reference prior scored outcomes
  and were committed after those outcomes but before their own execution;
- recorded at least two incumbent/challenger decisions;
- satisfied the episode's minimum 40% post-coverage falsifier/ablation share;
- verified the chronological trial record and directly verified or recomputed
  the score supporting the selected comparison, without replaying every trial;
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
the frozen terminal contract explicitly permits it.

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
metric is visible, the audit is consumed.  A trusted runner's log is sufficient
when it exhaustively covers every declared output channel; otherwise inability
to establish non-exposure consumes the audit.  Do not create a separate
receipt or attestation layer by default.

## Terminal encoding

Episode documents may name scientific `conclusion_class` values such as
`hybrid_supported`, `policy_improved`, `audit_failed`, or `unresolved`. Those
labels do not invent new outer-loop states. Unless a future frozen program
explicitly defines a compatible typed extension, the canonical bundle status
remains one of:

- `candidate_ready` for an eligible positive conclusion;
- `closed_no_candidate` for valid negative, unresolved, exhausted, or
  development-only conclusions; or
- `technical_failure` for invalid inputs, execution, leakage, policy, or audit
  integrity.

The episode policy must map every conclusion class to one of these outer
statuses before candidate selection completes. A richer local label cannot
bypass later Society, reward, approval, or claim review.
