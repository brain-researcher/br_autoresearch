# Integration-refinement pilot launch

- Submitted: `2026-09-27T02:26:25-0700`
- Slurm job: `45494117`
- Wrapper: `outputs/executor/slurm/pilot_integration_refinement.sbatch`
- Durable result directory: `outputs/scientific_qualification/pilots/integration_refinement_512_1024/`
- Initial scheduler state: `PENDING`
- Scope: generated data only; no real connectivity access
- Candidate comparison: `512` versus `1024` integration draws
- Frozen acceptance criteria: maximum absolute directional delta change `<= 0.002` and exact selected-model metadata equality
- Deterministic replay: required

This is a real T/U/M numerical-refinement execution, not a fixture-routed or
placeholder job. It cannot unlock readiness by itself; its result informs a
separately versioned scientific-qualification repair.

## Terminal result

- Slurm state: `FAILED`, exit `1:0`, elapsed `00:01:36`
- Durable report: `integration_refinement_512_1024/integration_refinement.json`
- Deterministic replay: passed
- Maximum 512→1024 directional delta change: `0.001403143430177588`
- Unchanged delta tolerance: passed (`<= 0.002`)
- Exact selected-model metadata: failed
- Identity mismatch: `curved_continuum_dense`, seed `1201`, right-source M
  selection (`M3` at 512 versus `M2` at 1024)

The nonzero exit is the pilot's intended fail-closed scientific/numerical
signal. The result is not being retried unchanged. The next bounded repair
separates fit initialization seeds from common nested integration streams and
then repeats this same gate.
