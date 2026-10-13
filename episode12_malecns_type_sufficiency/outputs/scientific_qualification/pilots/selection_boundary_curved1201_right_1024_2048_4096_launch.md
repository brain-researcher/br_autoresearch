# Second selection-boundary diagnostic launch

- Submitted: `2026-09-27T02:59:55-0700`
- Slurm job: `45495714`
- Wrapper: `outputs/executor/slurm/pilot_selection_boundary_curved1201_right_1024_2048_4096.sbatch`
- Durable result directory: `outputs/scientific_qualification/pilots/selection_boundary_curved1201_right_1024_2048_4096/`
- Scope: generated data only; no real or final connectivity access
- Fixture: `curved_continuum_dense`, seed `1201`, R-to-L
- Integration draws: `1024`, `2048`, `4096`
- M candidates: M2 and M3 under the unchanged production CV rule
- Deterministic replay: required
- Scientific threshold or qualification gate: none

This bounded job diagnoses the sole identity failure in full successor gate
`45495442`. It records candidate fold scores, means, standard errors,
selection-boundary distances, and fixed-model target deltas without changing
the model family, tie rule, margin, or numerical tolerance.

## Terminal result

- State: `COMPLETED`
- Exit: `0:0`
- Runtime: `00:00:33`
- Node: `sh02-10n54`
- Finished: `2026-09-27T03:04:33-0700`
- Deterministic replay: exact
- Selected M candidate at 1,024/2,048/4,096: M2/M3/M3
- Fixed M3 R-to-L delta change, 2,048 to 4,096: `0.0001955758223813915`
- Fixed M2 R-to-L delta change, 2,048 to 4,096: `0.00014444323414577726`

The observed 2,048-to-4,096 identity is stable within this job, but M2 is only
`0.00007781387440401765` outside the current near set at 4,096 draws. The
result therefore supports one bounded full six-fixture 2,048-versus-4,096
gate; it does not justify changing or bypassing the exact-identity criterion.
