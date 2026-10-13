# Interval-refinement V3 autonomous chain

- Submitted: `2026-09-27T03:10:50-0700`
- Integration predecessor: `45496112`
- 24-shard array: `45496314`, dependency `afterok:45496112`
- Descriptive aggregate: `45496332`, dependency `afterok:45496314`
- Split-block measured-gain calibration: `45496381`, dependency `afterok:45496332`
- Array wrapper: `outputs/executor/slurm/pilot_interval_refinement_v3_2048_array.sbatch`
- Aggregate wrapper: `outputs/executor/slurm/pilot_interval_refinement_v3_2048_aggregate.sbatch`
- Calibration wrapper: `outputs/executor/slurm/pilot_measured_gain_calibration_v3_2048.sbatch`
- Durable root: `outputs/scientific_qualification/pilots/interval_refinement_v3_2048/`
- Scope: generated data only; no real or final connectivity access

The array preserves the predeclared 12 cells, two independent 72-replicate
blocks, eight whole types per replicate, 1,999 paired studentized bootstrap
resamples, and two source-CV folds, using 2,048 predictive-integration draws.
Shards exit zero after computational completion regardless of scientific
coverage. The aggregate is descriptive and threshold-free. The final job uses
block 01 only to classify finite-procedure measured gain relative to the
existing `0.02` margin and block 02 only for terminal counts; it does not claim
population-oracle truth or assign a scientific pass/fail.

If integration gate `45496112` fails, all three downstream stages are expected
to dependency-cancel at zero runtime. No duplicate launch should be submitted.

## Terminal result and decision boundary

- Integration gate `45496112` completed `0:0` and released this chain.
- Array `45496314` completed all 24 tasks. Aggregate `45496332` completed
  `0:0`; calibration `45496381` completed `0:0` in 15 seconds.
- The 24 shards measured 13,518.779 process CPU seconds (3.7552 core-hours).
  Across the numerical gate, array, aggregate, and calibration, Slurm allocated
  14,065 one-CPU seconds (3.9070 core-hours); end-to-end elapsed time was
  25:08.
- `aggregate_report.json` verifies 24/24 shards and 1,728/1,728 records with
  zero numerical failures. Its generated-only terminal counts are 0
  M-supported, 187 adequate-before-sensitivity, and 1,541 unresolved.
- `finite_procedure_measured_gain_calibration.json` verifies disjoint block
  identities and seed namespaces, does not reuse evaluation fitted deltas,
  and classifies the 12 cells relative to the frozen `0.02` margin as 2 above,
  6 below, and 4 indeterminate. The two residual-group cells remain
  `unresolved_for_adequacy`. It explicitly applies no scientific gate and
  assigns no pass/fail.
- Artifact SHA-256 identities are
  `7fc7ac0e6615d5aabb5e9532dee277933b20a19aee5cf4af5a4d7751974b6b9f`
  for `aggregate_report.json` and
  `9e668743195291fca60e4c4f33b7972d661e48a90d3845cf48549997e854c10a`
  for `finite_procedure_measured_gain_calibration.json`.

The aggregate's leave-one-replicate-out coverage indicators reuse overlapping
observations and are therefore dependent. Its reported Clopper-Pearson limits
are descriptive, not exact coverage confidence intervals for the intended
infinite-design estimand, and cannot adjudicate the frozen `0.95` coverage
gate. The governing files most strongly imply the infinite-design expectation
of the fixed finite-sample fitter/CV/selection/scoring procedure. A finite-bank
conditional target would change that estimand; a stochastic approximation to
the implied target would instead require new oracle-error accounting, while a
deterministically certified oracle is not currently implemented or costed.
Those alternatives require scientist direction. No successor full
qualification or readiness chain was launched.

A subsequent conformance probe also established that the operational
count-aware/shrinkage U-linear covariance is full-rank and U-curved adds an
unfrozen full-rank residual tube. The V3 chain therefore measured numerical
behavior and cost for that implementation, not qualification of the declared
rank-two/one-dimensional bounded-U families. The active qualifier now fails
fast until that model-family choice is versioned.
