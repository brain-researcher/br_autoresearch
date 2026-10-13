# 2,048-to-4,096 integration-refinement launch

- Submitted: `2026-09-27T03:06:15-0700`
- Slurm job: `45496112`
- Wrapper: `outputs/executor/slurm/pilot_integration_refinement_2048_4096.sbatch`
- Durable result directory: `outputs/scientific_qualification/pilots/integration_refinement_common_random_2048_4096/`
- Scope: generated data only; no real or final connectivity access
- Draw grid: `1024`, `2048`, `4096`
- Gate comparison: `2048` versus `4096`
- Frozen maximum directional delta change: `0.002`
- Selected-model criterion: exact source-fit metadata equality
- Deterministic replay: required

This bounded full six-fixture gate follows diagnostic `45495714`. It retains
the original model family, current one-standard-error candidate-selection
rule, fixtures, seeds, numerical tolerance, and exact-identity requirement.
Failure will not be bypassed by further blind draw escalation; it will expose
a consequential selection-rule choice for scientist review.

## Terminal result

- State: `COMPLETED`
- Exit: `0:0`
- Runtime: `00:02:16`
- Node: `sh02-01n03`
- Finished: `2026-09-27T03:12:53-0700`
- Deterministic replay: exact
- Maximum directional delta change: `0.0010515592092518577`
- Frozen tolerance: `0.002`
- Selected-model metadata equality: passed for all six fixtures
- Overall gate: passed

The successful predecessor automatically released interval array `45496314`.
This pilot supports 2,048 predictive-integration draws for the next generated
qualification stage; it is not by itself a scientific qualification pass or
authorization to read final connectivity.
