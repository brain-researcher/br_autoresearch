# Experiment log

## 2026-08-17 — Goal and governance initialization

- Verified native `get_goal` was empty before initialization.
- Created fresh `codex_autoresearch_v1` loop
  `arl_b622657e9fbd000075c5ad5f4cf30c44`; parent loop
  `arl_8cbd25a7a8610a008c2830ba624052c4` was not reused.
- Persisted server-issued `begin_discovery`; loop entered `DISCOVERING` at revision 1.
- Prepared Goal handoff `goal_handoff_daf31dc2e8aa7a9965a9a8e7` and called native
  `create_goal` exactly once with the unchanged server-issued objective.
- Read `GOAL.md`, `DATASETS.md`, and `inputs/prior_episode/LINEAGE.md` in full.
- Began Stage 0 on Sherlock compute node `sh03-06n06` inside Slurm job 39398079.
- No new neural outcome values had been accessed when this entry was written.

## Execution ledger

| Stage | Action | Status | Evidence |
|---|---|---|---|
| 0 | Source and resource inventory | retrying after optional raw-annex audit repair | `outputs/code/inventory_sources.py` |

### Inventory repair

The first header scan stopped at a dangling optional raw-BOLD annex link for
`sub-093` run 4. It read no voxel data and wrote no inventory result. The repair
records dangling raw links separately and defines readiness from the actual test
inputs (fMRIPrep BOLD/masks/confounds, events, and FitLins designs). The frozen
tests will not depend on optional raw BOLD.

The completed retry also showed two storage dtypes among otherwise identical
fMRIPrep BOLD grids (419 float32 files, 13 int16 files). Spatial compatibility is
defined from shape, spatial zooms, and affine, not storage dtype; the inventory
code records both full-header and spatial-header signatures.

## 2026-08-17 — Prospective mechanism selection

- Stage 0 passed for 432/432 required runs; 109 optional raw-annex BOLD links are
  dangling, but the selected tests use complete fMRIPrep derivatives.
- Before source neural voxel access, retained exactly two mechanism tests:
  amplitude/shape decomposition and mask-edge/weighting decomposition.
- Task-version stratification, substantive spatial persistence, and fixed
  continuous-versus-thresholded comparisons are embedded alternatives/controls,
  not extra mechanism tests.

## 2026-08-17 — Contract freeze

- Froze `outputs/frozen_successor_contract.{md,json}` at
  2026-08-17T15:21:34Z before new neural outcome access.
- Fixed C1 only, S0 versus S8 only, Q0/all 108 subjects, separate gain/loss,
  outcome-blind common/core/periphery masks, three run aggregations, PCG64 seed
  20260817, and explicit candidate/stop rules.
- Contract files are immutable from this point forward.
- Frozen hashes: Markdown
  `63cb26e5b57ca520c27fc39c6c38b53d52dae0672f20917ce5186a24a8ad5ed0`;
  JSON `1c7f712a8c5ced45a97f312794df15fa68d83f117b4f0dddf63cc9d35c6f5496`.

### Pre-outcome mask-feasibility repair

The initial frozen 100%-coverage-intersection core failed its outcome-blind
validity guard: 28,881 core voxels versus 221,116 common voxels (<50%). No BOLD,
effect, variance, t, or group-map values had been read. A prospective mask-only
check showed that one erosion of the already fixed common mask gives 204,421 core
and 16,695 periphery voxels. The contract was transparently versioned to v2 and
re-frozen with that core definition; endpoints, thresholds, subjects, contrasts,
and candidate rules did not change. The superseded v1 hashes remain above.
Final re-frozen v2 hashes: Markdown
`62aea3f7a884630aeff50c9fc713ffa9f0831a7c6d939c85692328cd40ac3aa3`;
JSON `27a58a09ee043e71b7d1fca93de0ef727b79fc6ecbd9fef2eaf800208317829f`.

## 2026-08-17 — Engineering validation and pilot

- Python syntax and three direct pipeline/math tests passed. `pytest` itself was
  unavailable in the existing environment, so the test functions were invoked
  directly without installing or changing dependencies.
- Pilot job 39400893 stopped before checkpoint creation because one run omitted
  the optional `trial_type.missed` design column when it had no missed trials.
  The implementation now preserves task columns present in each unchanged
  FitLins design, requires both target contrasts and the intercept, and does not
  synthesize an absent nuisance event regressor. This engineering repair does not
  change the frozen estimand, endpoints, or subject set.

## 2026-08-17 — Subject execution and pre-aggregation audit

- Pilot retry job `39401155` completed `sub-001` in 10:37 with a valid hashed
  checkpoint; peak resident memory was 8.8 GB.
- Production job `39401778` completed `sub-002`. Its never-started pending
  indices were canceled solely to reduce the requested memory from 48 GB to the
  pilot-supported 20 GB; no running or completed work was canceled.
- Replacement array `39401862` completed the remaining 106 subject indices.
  The concurrency cap was increased from 16 to 24 only after 48 hashed
  checkpoints had accumulated without failure; observed completed-task peak
  memory remained below the 20 GB request (maximum observed approximately
  12.7 GB).
- Exactly 108 subject `.npz` checkpoints and 108 sidecars are now present on
  scratch. The production array is no longer queued or running.
- A deterministic code audit before group aggregation corrected three gaps:
  it tied alternative-aggregation support to the same strategy's
  leave-one-out guard, added split-half threshold/rank readouts, and completed
  the declared leave-one-out `D_raw` and equal-run core influence summaries.
  The frozen v2 contract, outcomes, thresholds, SESOIs, and candidate hierarchy
  were not changed.
- Python compilation and all five direct tests passed. The first synthetic
  threshold-helper probe used a null-mean sample and correctly produced no
  rejected voxels (undefined Jaccard), so an assertion expecting a defined
  Jaccard failed; a non-null synthetic probe then passed. This was a test-input
  issue, not an analysis failure.

## 2026-08-17 — Frozen aggregation and terminal result

- Initial aggregation submission `39502281` requested 96 GB. Sherlock mapped
  that request to 13 schedulable CPUs and no node had sufficient free memory;
  the job was canceled with exactly zero runtime. The resource request alone
  was reduced to 24 GB and 4 CPUs.
- Aggregation attempt `39502592` ran for 54 seconds and stopped at the declared
  affine negative control. The explicit correlation was within `4e-16` of one,
  but deriving the OLS residual as `centered_ss*(1-r^2)` produced cancellation
  error above the `1e-10` residual-RMS guard. The implementation was changed to
  evaluate the identical fitted OLS residual directly (and use its
  Gram-matrix Schur complement for bootstrap summaries). A sixth permanent
  numerical-stability test was added; all six direct tests passed.
- Unchanged retry `39502945` completed on `sh03-06n09` in 55 seconds with exit
  code 0 and peak RSS 2,376,268 KiB. Both affine negative controls passed at
  residual-RMS ratios below `6e-16`.
- The frozen hierarchy returned `candidate_ready` with the single selected
  candidate `residual_substantive_spatial_sensitivity`.
- Full IVW S0-versus-S8 `D_raw` was 0.9039 (gain) and 0.9101 (loss), spatial
  Pearson r was 0.4410 and 0.4024, and scale attenuation was only 0.0089 and
  0.0068. Amplitude/offset support therefore failed.
- Periphery change was enriched 3.026x and 2.985x, but core/full ratios were
  1.000 and 0.998. Equal-run aggregation changed `D_raw` by -2.98% for each
  contrast (slightly larger rather than attenuated), while stabilized IVW was
  effectively unchanged. Edge/weighting support therefore failed.
- Equal-run core `D_raw` remained 0.928/0.932 with r 0.385/0.355. Required
  directions held in EI, ER, both deterministic split halves, and every
  leave-one-subject-out result.
- Full BH-FDR-set Jaccards were approximately 0.001, but matched top-|t|
  Jaccards were also low (0.081/0.077), so the fixed threshold-disconnect rule
  did not pass; the pattern was broad rank/spatial relocation rather than a
  stable-rank threshold crossing.
- Wrote and locally validated `outputs/RESULT.md` and schema-shaped
  `outputs/candidate_bundle.json`; every declared artifact exists. All claims
  remain exploratory-only within ds001734 and require independent fresh
  confirmation.
