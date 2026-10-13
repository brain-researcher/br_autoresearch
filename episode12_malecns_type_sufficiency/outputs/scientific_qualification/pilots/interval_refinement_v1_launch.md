# Interval-refinement pilot launch

- Submitted: `2026-09-27T02:36:37-0700`
- Integration predecessor: `45494509`
- 24-shard array: `45494516`, dependency `afterok:45494509`
- Aggregate: `45494518`, dependency `afterok:45494516`
- Array wrapper: `outputs/executor/slurm/pilot_interval_refinement_array.sbatch`
- Aggregate wrapper: `outputs/executor/slurm/pilot_interval_refinement_aggregate.sbatch`
- Durable root: `outputs/scientific_qualification/pilots/interval_refinement_v1/`
- Scope: generated data only; no real connectivity access

The array contains 12 predeclared cells × two independent 72-replicate blocks
with eight whole types per replicate, 1,999 paired studentized bootstrap
resamples, 512 integration draws, and two source-CV folds. Shards report
L→R/R→L/aggregate/joint coverage and exit zero after computational completion
regardless of scientific coverage; the aggregate verifies all 24 shards and
1,728 records. It applies no new threshold and does not interpret false
adequacy before margin-relative truth calibration is repaired.

## Dependency outcome

The predecessor `45494509` failed its unchanged numerical-stability gate.
Slurm consequently cancelled array `45494516` and aggregate `45494518` at
zero runtime on `2026-09-27T02:40:25-0700`. No interval shard or aggregate was
created. This chain must not be resubmitted until a replacement numerical gate
passes.
