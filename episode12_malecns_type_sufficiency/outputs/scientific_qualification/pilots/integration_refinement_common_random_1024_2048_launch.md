# High-draw integration-refinement launch

- Submitted: `2026-09-27T02:52:49-0700`
- Slurm job: `45495442`
- Wrapper: `outputs/executor/slurm/pilot_integration_refinement_successor.sbatch`
- Durable result directory: `outputs/scientific_qualification/pilots/integration_refinement_common_random_1024_2048/`
- Scope: generated data only; no real or final connectivity access
- Draw grid: `512`, `1024`, `2048`
- Gate comparison: `1024` versus `2048`
- Frozen maximum directional delta change: `0.002`
- Selected-model criterion: exact source-fit metadata equality
- Deterministic replay: required

This is the full six-fixture successor gate supported by the completed
selection-boundary diagnostic `45495135`. It changes only the numerical draw
count being tested; the model family, candidate-selection rule, tolerance,
fixture set, seeds, and replay requirement remain unchanged.

## Terminal result

- State: `FAILED` by the wrapper's deliberate non-passing exit
- Exit: `1:0`
- Runtime: `00:01:04`
- Node: `sh04-13n31`
- Finished: `2026-09-27T02:55:50-0700`
- Deterministic replay: exact
- Maximum directional delta change: `0.000544475711655577` (passes `0.002`)
- Selected-model metadata equality: failed in one of six fixtures

The only failed row was `curved_continuum_dense`, fixture seed `1201`,
R-to-L: the right-source M candidate changed from M2 at 1,024 draws to M3 at
2,048 draws, while the directional delta changed by only
`0.0001691792756738364`. The gate was not bypassed. This is a second discrete
M2/M3 selection-boundary failure and requires a focused diagnostic before any
further full-gate or interval launch.
