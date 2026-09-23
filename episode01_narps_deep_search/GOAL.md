# Episode 01: Adaptive Search for Transportable Smoothing Mechanisms

## Authority and status

This is the current local contract for formal Episode 01. It replaces the old
EP01/EP02 *episode roles* with one deeper search program, but it does not
rewrite or supersede their evidence records. The two runs archived under
`../_examples/historical_prior_records/` remain immutable, fully exposed
development priors.

This contract is `planned_unregistered`. It does not authorize data access,
computation, canonical Goal creation, Society review, reward, launch, audit
opening, or a scientific claim. A new canonical program, Goal handoff, and loop
identity are required; neither legacy loop may be reused. The common adaptive
search requirements in
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md) apply.

## Research question

Which spatial-operator, masking, scaling, first-level estimation, and
group-aggregation mechanisms explain the change from unsmoothed to
8-mm-smoothed gain/loss effect maps, and does the selected explanation
transport from NARPS `ds001734` to the independently collected mixed-gambles
dataset `ds000005`?

The objective is not to rediscover that smoothing changes maps. It is to build
and falsify mechanistic predictors of *where the S0-to-S8 change comes from*,
separate the deterministic consequence of the exact 8-mm operator from
noncommuting pipeline operations, and evaluate one locked explanation on data
that did not select it.

## Relationship to the two prior runs

The legacy first run evaluated a frozen `confound x QC x smoothing` multiverse.
The legacy second run evaluated two bounded mechanism tests. Their contracts,
results, failures, review state, and selected small artifacts are declared in
[`inputs/prior_lineage/PRIOR_LINEAGE_MANIFEST.json`](inputs/prior_lineage/PRIOR_LINEAGE_MANIFEST.json).
They may motivate hypotheses and test ingestion, but they:

- count as zero new trials and zero independent evidence;
- cannot satisfy branch coverage, adaptive-successor, lock, or audit gates;
- cannot be described as confirmation because `ds001734` outcomes are exposed;
- cannot donate a candidate or canonical status to this episode; and
- cannot be read from live sibling outputs during an authorized run.

The new unit of work is one append-only program:

```text
readiness -> branch coverage -> adaptive successors -> falsification
          -> one configuration lock -> one external audit -> one terminal
```

The audit is part of Episode 01. It is not a separate EP02, because splitting
selection and audit would break the single ledger/lock boundary and could count
one dependent result twice. Formal EP02 now independently denotes the Dudman
dopamine learning-rate contract and has no evidential role in this episode.

## Scientific objects and estimands

For each contrast, NARPS task-version environment, subject fold, and frozen
analysis pathway, define:

- `M0`: the group effect map from the unsmoothed pathway;
- `M8`: the group effect map from the 8-mm pre-GLM pathway;
- `K8_ref(M0)`: one globally frozen exact 8-mm Gaussian operator applied to
  `M0` after group aggregation, with one fixed grid, boundary, normalization,
  and resampling DAG; and
- `Mhat8(h)`: the predicted smoothed map under mechanism hypothesis `h`.

Within a frozen comparison mask, the primary map score is:

```text
ESC(h) = 1 - SSE(M8, Mhat8(h)) / SSE(M8, M0)
DeltaESC(h) = ESC(h) - ESC(K8_ref(M0))
```

`ESC=0` means no improvement over predicting no smoothing change; `ESC=1`
means exact prediction. `DeltaESC` is always the paired improvement over this
single global exact-kernel reference. Kernels applied at other pipeline stages
are mechanism hypotheses, not moving baselines. If the denominator is zero,
nonfinite, or below the
synthetically calibrated floor, that environment is invalid under a frozen
degeneracy rule; it is never silently dropped or assigned a favorable score.

The four primary development environments are `gain x EI`, `gain x ER`,
`loss x EI`, and `loss x ER`, each evaluated through deterministic,
task-version-stratified subject folds. NARPS EI and ER are environments, not
independent replications. The controller ranks valid trials
lexicographically by:

```text
(minimum-environment DeltaESC,
 median-environment DeltaESC,
 lower value under the frozen complexity functional,
 lexicographically lower canonical configuration hash)
```

Thus a large gain in one environment cannot compensate for a direction
reversal in another. Spatial correlation, normalized RMS difference,
spatial-frequency residuals, edge-distance profiles, sign agreement,
matched-rank overlap, and thresholded sets are diagnostic or secondary unless
frozen otherwise before development begins.

## Development hypothesis grammar

Every scored hypothesis must declare its parent trial IDs (an empty list for a
root proposal), strongest alternative, directional prediction, falsifier,
expected information gain, expected cost, changed operator, and unchanged
operators. It may change one interpretable family at a time unless it is a
declared composition or ablation.

### A. Spatial operator and stage

- Exact matched Gaussian operators at BOLD, run-effect, subject-effect, or
  group-map stage.
- Frozen FWHM levels `{0, 2, 4, 5, 6, 8, 10}` mm with role-specific access.
- Boundary convention, normalization, interpolation, and resampling order.
- Explicit commutation tests between smoothing and GLM/group aggregation.

### B. Mask and boundary support

- Fixed coverage masks and kernel-aware erosions.
- Millimetre-defined distance-to-boundary shells.
- Coverage and low-intensity-support diagnostics.
- Whole-mask versus interior scoring under the same metric.

### C. Scaling and numerical stabilization

- Native versus explicitly matched signal scaling.
- Variance floors calibrated only on synthetic/null fixtures.
- Outcome-blind handling of constant or near-constant series.
- Floating-point and resampling precision controls.

### D. First-level and group estimation

- Historical C1 nuisance model as anchor; C0/C2 only as declared sensitivity
  environments, never an unrestricted confound leaderboard.
- Equal-run, inverse-variance, stabilized inverse-variance, and robust group
  aggregation.
- AR(1) and declared alternative noise-model checks when executable.

### E. Residual representation and reliability

- Spatial-frequency decomposition of `M8-K8_ref(M0)`.
- Split-half and leave-one-subject-out reliability weighting.
- Low-dimensional residual summaries learned only inside grouped folds.
- Spatial-phase, subject-label, and mismatched-kernel negative controls.

At most three compatible operators may be composed. Each component must first
have an interpretable single-operator result; every composition requires all
single-operator ablations and a negative control. Arbitrary ROI fishing,
threshold tuning, post-audit feature creation, unrestricted architecture
search, and unlogged manual edits are prohibited.

## Program stages

### Stage 0: readiness and synthetic calibration

Without opening audit neural arrays, validate source inventories, folds, map
algebra, exact-kernel implementation, masks, degeneracy handling, uncertainty,
ledger transitions, resource metering, and audit rejection on synthetic
fixtures. Freeze the final audit thresholds and simultaneous-interval family
only after outcome-blind calibration and scientist signoff.

### Stage 1: branch coverage

Run 12--20 valid broad-screen trials and complete exactly twelve primary
coverage slots: two trials assigned to each of families A--E, one no-change
baseline, and one global exact-`K8_ref` baseline. A trial fills at most one slot;
additional valid screen trials fill no second slot. Invalid, duplicate,
readiness-rejected, or engineering-failed attempts do not count.

### Stage 2: adaptive hypothesis search

Run 12--24 valid targeted-recomputation and composition trials. Development
outcomes may choose admissible successors. Each counted successor
must reference a prior scored record, commit to the complete ledger prefix
visible at proposal time, and be proposed before its own execution. At least
two outcome-linked successor cycles and two incumbent/challenger decisions are
required. The first improvement is an incumbent, not a terminal candidate.

### Stage 3: falsification, ablation, and replication

Run 6--16 valid full-development falsification, ablation, and replication
trials. The denominator is every valid trial committed after the unique
coverage-satisfied record. The numerator is the subset assigned exactly one
predeclared alternative, negative-control, ablation, influence, synthetic-
recovery, or direct-replication credit; the required count is
`ceil(0.40 * denominator)`. Stage-2 trials may qualify when their credit is
committed before execution, every valid Stage-3 trial must qualify, and invalid,
duplicate, retry, or engineering-failed records never count. Thus the minimum
30-trial schedule needs at least two qualifying Stage-2 trials in addition to
six Stage-3 trials. Full-development shortlisting compares one incumbent, at
most two challengers, no-change, and exact-`K8_ref` baselines across gain/loss,
EI/ER, deterministic halves, and leave-one-subject-out influence. Before
eligibility is evaluated, freeze a nonempty comparator set containing every
nonkernel member of the full-development shortlist, including the
highest-ranked fully developed nonkernel configuration. Omitting that set or
its highest-ranked member leaves the depth gate unsatisfied. A locked 6-mm
interpolation check may open once. The 2-, 5-, and 10-mm levels are a predeclared
dose-response sensitivity series; they may affect the pre-lock shortlist only
through the frozen ranking rule and can never alter thresholds. An independent
implementation must replay the primary metric before lock.

### Stage 4: configuration lock

After a valid stop event, first apply one frozen, simultaneous-interval
eligibility contract. Select the highest-ranked nonkernel finalist whose four
environment DeltaESC lower bounds clear the calibrated useful-effect margin
with no direction reversal. If none passes, `K8_ref` may lock only if all four
development environments clear a calibrated ESC lower-bound floor and every
member of the frozen nonkernel comparator set has a DeltaESC upper bound below
the useful-effect margin in all four environments. Otherwise close
`closed_no_development_lock` without opening the audit. These rules are
mutually exclusive by construction and do not fall back to an ineligible raw
incumbent. The floor, margin, interval family, and multiplicity rule are
launch-blocking until outcome-blind calibration and scientist signoff.

For an eligible mechanism, freeze exactly one winner, one diagnostic runner-up,
the complete nonkernel comparator set, all baselines, the global `K8_ref` DAG,
the complexity functional, code, environment, data/split/operator manifests,
masks, hyperparameters, seeds, metrics, thresholds, uncertainty procedure,
retry rules, report template, audit command, and the complete hash-chained
ledger. When `K8_ref` wins, the runner-up is the highest-ranked member of the
nonkernel comparator set. The runner-up cannot replace the winner after audit
access.

### Stage 5: one-shot external transport audit

A permission-separated runner opens `ds000005` in one lock-hash-bound audit
transaction. It recomputes the frozen S0/S8 pathways for `paragain` and
`paraloss` without mechanism tuning or candidate substitution, then emits only
declared point estimates, simultaneous paired subject-bootstrap intervals,
deterministic-half checks, leave-one-subject-out guards, controls, and access
receipts. Any visible candidate-discriminating metric consumes the audit.
Audit output cannot enqueue a trial in this program. A frozen, eligible
infrastructure retry that emitted no candidate-discriminating value is a
receipt-linked continuation of that same transaction, not a second scientific
opening; its exact failure list and maximum attempt count must be signed before
launch.

## Incumbent, budget, and stopping rules

- A challenger replaces the incumbent only if the first lexicographic score
  coordinate improves by at least `0.01 DeltaESC`, or does not decrease it and
  lies within `0.01` while resolving a declared falsifier with lower frozen
  complexity. Remaining ties use median DeltaESC, then complexity, then the
  lexicographically lowest configuration hash.
- Minimum valid development trials: **30**.
- Maximum valid development trials: **60**.
- Stage minima are **12 coverage, 12 adaptive, and 6 falsification trials**;
  surplus trials in one stage cannot satisfy another stage's minimum.
- Maximum engineering failures: **15**; the sixteenth is technical failure.
- Qualified patience: **12 consecutive valid trials** without material
  improvement, only after every development depth gate is met. Patience resets
  only when the frozen challenger rule replaces the incumbent.
- Resource ceilings: **2,500 CPU core-hours**, **0 GPU-hours**, **1,536 GiB
  scratch**, and **96 wall-clock hours**.

A policy violation maps to `policy_violation` at any time; the sixteenth
engineering failure or all branches becoming technically impossible maps to
`technical_failure` at any time. Resource ceiling or authorized interruption
before a valid audit completes maps to non-scientific `incomplete_search`
regardless of trial count. Reaching 60 valid trials or exhausting all
admissible scientific configurations while any development depth gate remains
unsatisfied also maps to `incomplete_search`. Only after the minimum and every
depth gate are satisfied do maximum trials, qualified patience, or scientific
exhaustion invoke the development-lock eligibility classifier. A positive
result, a negative contrast, one branch failure, or the appearance of an
incumbent is never itself a stop event.

## Audit decision contract

The following numerical values are provisional and therefore launch-blocking
until synthetic calibration and scientist signoff freeze them.

### Kernel-sufficient positive

If and only if the locked winner is `K8_ref`, then for both audit contrasts:

- the simultaneous paired 95% lower bound for exact-`K8_ref` `ESC` is at least
  `0.80`;
- every member of the locked nonkernel comparator set has a simultaneous
  paired 95% upper bound for `DeltaESC` strictly below `0.05`;
- the result is stable in deterministic halves and every leave-one-subject-out
  analysis; and
- mismatched-kernel and spatial-phase controls fail as predicted.

### Pipeline-noncommutative positive

If and only if the locked winner is nonkernel, then for both audit contrasts:

- the selected mechanism has point `DeltaESC >= 0.10`;
- its simultaneous paired 95% subject-bootstrap lower bound is greater than
  zero;
- direction agrees in deterministic halves and every leave-one-subject-out
  analysis; and
- required component ablations and negative controls behave as predicted.

The simultaneous family and bootstrap draw count are launch-blocking until
frozen. The two positive classes are mutually exclusive. A selected
`K8_ref` mechanism definitively fails transport if either contrast's ESC upper
bound is below `0.80`, any member of the nonkernel comparator set has a
DeltaESC lower bound of at least `0.05`, or a required stability/control check
fails without an integrity breach. A selected nonkernel mechanism definitively
fails if either contrast's DeltaESC upper bound is below `0.10` or a required
stability/ablation/control check fails without an integrity breach. Those cases
are `closed_development_only`; an audit that is neither positive nor
definitively negative is `closed_unresolved`. Neither case permits runner-up
substitution, threshold changes, or a resumed search.

## Terminal classes

Terminal precedence is fixed:

| Precedence | Conclusion class | Outer status | Meaning |
| ---: | --- | --- | --- |
| 1 | `policy_violation` | `technical_failure` | Leakage, forbidden mutation, or policy breach invalidates the round. |
| 2 | `technical_audit_integrity_failure` | `technical_failure` | Audit firewall, lock binding, or receipt integrity failed. |
| 3 | `technical_failure` | `technical_failure` | Inputs, execution, or every branch became technically unusable. |
| 4 | `incomplete_search` | `closed_no_candidate` | Resource/interruption prevented audit, or max/exhaustion arrived before all depth gates; not a scientific negative. |
| 5 | `closed_no_development_lock` | `closed_no_candidate` | All development depth gates completed but no mechanism was eligible to lock. |
| 6 | `closed_development_only` | `closed_no_candidate` | A locked development mechanism did not transport. |
| 7 | `closed_unresolved` | `closed_no_candidate` | A valid audit could not distinguish a predeclared positive class. |
| 8 | `candidate_ready_pipeline_noncommutative` | `candidate_ready` | A named noncommuting mechanism passed the locked audit rule. |
| 9 | `candidate_ready_kernel_sufficient` | `candidate_ready` | The exact kernel passed the locked sufficiency rule. |

Only precedence 8 or 9 may create a candidate-ready bundle, and only after a
valid audit. Every other valid scientific terminal preserves a useful negative
or unresolved result without pretending it is confirmation.

## Required artifacts

The registered program must produce immutable source, exposure, split, mask,
operator, and environment manifests; an append-only `experiments.jsonl` ledger;
hypothesis genealogy and incumbent history; synthetic calibration; the full
score/failure table; resource accounting; configuration lock; audit receipt;
audit report; ablations and controls; a conservative `RESULT.md`; a
schema-valid terminal bundle; and the seven local workspace projections.

## Claim boundary

Even a positive result supports only a transportable *methods mechanism*
across two small public mixed-gamble datasets. It does not establish a neural
mechanism of gain/loss processing, universality across tasks or pipelines,
scientific acceptance, or permission to update the Landscape. The 16-person
audit makes precision and leave-one-subject sensitivity central limitations.
