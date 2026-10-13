# EP12 V3 coverage-estimand audit

Date: 2026-09-27

Scope: generated data only. No real or final connectivity was accessed.

## Verified execution evidence

- Numerical predecessor `45496112`: `COMPLETED 0:0`.
- V3 interval array `45496314`: all 24 tasks completed.
- Aggregate `45496332`: `COMPLETED 0:0`.
- Split-block calibration `45496381`: `COMPLETED 0:0` in 15 seconds.
- The 24 shard reports total 13,518.779 process CPU seconds (3.7552
  core-hours). The complete numerical-gate-to-calibration Slurm chain totals
  14,065 allocated one-CPU seconds (3.9070 core-hours) and 25:08 elapsed from
  first start to final completion.
- Aggregate inventory: 24/24 shards, 1,728/1,728 records, zero numerical
  failures.
- Aggregate terminals: 0 M-supported, 187 adequate-before-sensitivity, 1,541
  unresolved.
- Calibration/evaluation identities and seed namespaces are disjoint;
  evaluation fitted-delta records are not reused.
- Finite-procedure gain classifications across 12 cells: 2 above, 6 below,
  and 4 indeterminate relative to the frozen `0.02` margin.
- Independent evaluation-block terminals: 0 M-supported, 94
  adequate-before-sensitivity, 770 unresolved, and 0 numerical failures.
- Both residual-group cells are `unresolved_for_adequacy`; no false-adequacy
  pass/fail is assigned.

Artifact identities:

- `aggregate_report.json` SHA-256:
  `7fc7ac0e6615d5aabb5e9532dee277933b20a19aee5cf4af5a4d7751974b6b9f`
- `finite_procedure_measured_gain_calibration.json` SHA-256:
  `9e668743195291fca60e4c4f33b7972d661e48a90d3845cf48549997e854c10a`

## Why coverage is not qualified

For each 72-replicate cell/block, the pilot compares replicate `r`'s interval
with the mean fitted delta from the other 71 replicates. The 72 reference
centers overlap. Consequently, the coverage indicators are dependent and the
reported Clopper-Pearson limits are only binomial reference summaries; they are
not exact confidence intervals for coverage of an infinite-design parameter.
The artifacts correctly set `scientific_gate_applied=false`, but their
descriptive coverage fields cannot adjudicate the contract's frozen `0.95`
coverage gate.

## Target implied by the governing files

The endpoint, contract, qualification specification, and executable fitter most
strongly imply this target for each frozen synthetic cell and direction:

> the expected whole-type M-minus-selected-T/U held-out log-predictive-density
> gain over fresh generated types, source samples, target samples, and counts,
> while holding the finite sample size and complete fitting/CV/selection/
> scoring procedure fixed.

The bilateral aggregate is the equal mean of the directional targets. This is
an infinite-design expected **finite-procedure** gain. It is neither the
population-optimal KL gap between abstract model families nor a conditional
mean over the 576 rows already generated. Formalizing it in an additive
successor contract is a clarification of the existing endpoint; replacing it
with either alternative is a consequential estimand change.

## Why the frozen exact-binomial rule still cannot run

For 72 independent outer intervals, Clopper-Pearson inference is exact only if
each success indicator compares its interval with the same fixed, known target.
For the frozen two-sided 95% rule at `n=72`, the interval contains `0.95` only
for 64 through 71 successes; even 72/72 is rejected because its lower limit is
slightly above `0.95`.

- A finite independent oracle-bank mean is a random plug-in target. Conditional
  on that bank, the outer indicators are independent, but they concern coverage
  of the bank mean rather than the infinite-design expectation.
- A high-confidence stochastic enclosure of the infinite target can give
  conservative certain/possible coverage counts, but it introduces an oracle
  error probability and multiplicity allocation. Retaining an overall 95%
  claim therefore requires a versioned error-budget rule.
- A deterministic certified enclosure refined until every coverage indicator
  is invariant would preserve the target and exact rule without an added error
  probability. No such oracle exists in the executor, and continuous
  generators plus nonlinear fitting and model selection make its feasibility
  and cost unproved.
- A frozen finite generated bank permits exact conditional coverage but narrows
  the claim to that arbitrary bank.

The old qualification path now marks the overlapping-center indicators as
dependent and fails closed. No threshold was changed.

## Consequential choice

The remaining defensible paths are:

1. Preserve the implied infinite-design finite-procedure target and version
   the gate to include a conservative stochastic-oracle error budget and
   stopping rule.
2. Preserve both that target and the unchanged exact-binomial rule, but require
   a deterministic certified oracle whose feasibility and cost are not yet
   established.
3. Freeze a finite generated bank and retain exact conditional binomial
   coverage, explicitly accepting the narrower estimand.
4. Leave coverage unresolved; under the frozen spec, adequacy cannot qualify.

The executor did not select a path, change a threshold, or submit a successor
chain.
