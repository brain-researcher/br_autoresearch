# Selection-boundary diagnostic launch

- Submitted: `2026-09-27T02:48:01-0700`
- Slurm job: `45495135`
- Wrapper: `outputs/executor/slurm/pilot_selection_boundary_diagnostic.sbatch`
- Durable result directory: `outputs/scientific_qualification/pilots/selection_boundary_curved1202_512_1024_2048/`
- Scope: generated data only; no real or final connectivity access
- Fixture: `curved_continuum_dense`, seed `1202`, L-to-R
- Integration draws: `512`, `1024`, `2048`
- M candidates: M2 and M3 under the unchanged production CV rule
- Output: candidate fold scores, means, standard errors, selection-boundary
  distances, and fixed-M2/fixed-M3 target deltas against the selected T/U
  reference
- Deterministic replay: required
- Scientific threshold or qualification gate: none

This bounded job diagnoses the only failing row from job `45494509`. It does
not change the candidate family, tie rule, `0.002` numerical tolerance, or
scientific margin. A full six-fixture successor gate will be launched only if
this diagnostic supports a stable higher-draw comparison.

## Terminal result

- State: `COMPLETED`
- Exit: `0:0`
- Runtime: `00:00:19`
- Node: `sh04-17n32`
- Finished: `2026-09-27T02:49:23-0700`
- Deterministic replay: exact
- Selected M candidate at 512/1,024/2,048: M3/M3/M3
- Fixed M3 L-to-R delta change, 1,024 to 2,048: `0.000544475711655577`
- Fixed M2 L-to-R delta change, 1,024 to 2,048: `0.0001980891838832457`
- Frozen tolerance retained for the successor gate: `0.002`

At 512 draws, M2 lay only `0.00006095704117078959` outside the production
near set, confirming a numerically fragile discrete selection boundary. At
1,024 and 2,048 draws, the M2 gaps outside that boundary were about `0.002219`
and `0.006594`, respectively. This supports a full six-fixture 1,024-versus-
2,048 successor gate without changing the model family or selection rule.
