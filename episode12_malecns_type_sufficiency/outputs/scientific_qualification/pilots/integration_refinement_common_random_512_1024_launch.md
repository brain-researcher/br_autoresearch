# Common-random integration-refinement launch

- Submitted: `2026-09-27T02:36:06-0700`
- Slurm job: `45494509`
- Initial scheduler state: `PENDING`
- Wrapper: `outputs/executor/slurm/pilot_integration_refinement.sbatch`
- Durable result directory: `outputs/scientific_qualification/pilots/integration_refinement_common_random_512_1024/`
- Scope: generated data only; no real connectivity access
- Candidate comparison: `512` versus `1024` integration draws
- Frozen criteria: maximum absolute directional delta change `<= 0.002` and exact selected-model metadata equality
- Numerical repair: candidate-specific fit/GMM seeds retained; integration streams shared within each source-CV fold and across final T/U/M target scoring
- Deterministic replay: required

This reruns the unchanged numerical gate after a bounded variance-reduction
repair. It neither overwrites the first pilot nor authorizes qualification,
readiness, or real-data progression by itself.

## Terminal result

- State: `FAILED` by the wrapper's deliberate non-passing exit
- Exit: `1:0`
- Elapsed: `00:01:34`
- Finished: `2026-09-27T02:40:13-0700`
- Deterministic replay: passed exactly
- Maximum observed directional delta change: `0.003157227455292663`
- Frozen tolerance: `0.002`
- Selected-model metadata equality: failed in one of six fixtures
- Downstream interval array `45494516`: dependency-cancelled at zero runtime
- Downstream interval aggregate `45494518`: dependency-cancelled at zero runtime

The sole failure was `curved_continuum_dense`, fixture seed `1202`, L-to-R:
the selected M candidate changed from M2 at 512 draws to M3 at 1,024 draws.
All other fixture comparisons retained model identity and had directional
changes no larger than `0.00067216`. A focused generated-only diagnostic was
therefore launched as job `45495135`; the failed gate was not rerun unchanged.
