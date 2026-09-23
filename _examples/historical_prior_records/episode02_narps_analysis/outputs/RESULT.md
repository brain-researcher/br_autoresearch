let# NARPS smoothing-mechanism successor result

## Outcome

Terminal status: **`candidate_ready`**  
Selected candidate: **`residual_substantive_spatial_sensitivity`**

The frozen hierarchy selected `residual_substantive_spatial_sensitivity` as one reproducible exploratory methods candidate. Here, substantive means a material spatial sensitivity to the analysis choice, not a biological mechanism. This is an internal ds001734 result, not independent confirmation.

## Mechanistic interpretation of the v1 smoothing dominance

This successor did not rerun or re-estimate the v1 89.5% variance share; it tested why smoothing could dominate it. The fixed S0-versus-S8 normalized differences were 0.904 for gain and 0.910 for loss, while affine scale/offset alignment removed only 0.89% and 0.68% of those differences; spatial correlations were only 0.441 and 0.402. Periphery change was enriched by about 3.03x/2.99x, but core/full ratios were 1.000/0.998, and equal-run aggregation attenuated the difference by -2.98%/-2.98% (negative means slightly larger). Thus neither the mask edge nor inverse-variance weighting explains the dominant change. Finally, matched top-|t| Jaccards were only 0.081/0.077, so the very low FDR-set overlap was not a stable-rank threshold-crossing effect. The bounded explanation is therefore a broad analysis-induced spatial-pattern change that persists in the well-covered core, under non-extreme weighting, and within both task-version strata.

This is exploratory-only analysis in ds001734. It is not scientific acceptance,
does not establish a biological mechanism, and does not provide independent
confirmation. Gain and loss were analyzed separately.

## Frozen primary readouts

| Contrast | IVW full D_raw | Scale attenuation (95% CI) | Pearson r (95% CI) | Edge enrichment (95% CI) | Core/full D ratio (95% CI) | Equal-run attenuation | Stabilized-IVW attenuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| gain_demean | 0.9039 | 0.0089 [0.0081, 0.0110] | 0.4410 [0.3658, 0.4281] | 3.026 [2.703, 3.468] | 1.000 [0.998, 1.003] | -0.030 | -0.000 |
| loss_demean | 0.9101 | 0.0068 [0.0062, 0.0111] | 0.4024 [0.3459, 0.4022] | 2.985 [2.769, 3.378] | 0.998 [0.996, 1.002] | -0.030 | 0.000 |

Amplitude dominance required scale attenuation and its lower bound at least 0.50,
Pearson r and its lower bound at least 0.80, and both point thresholds in EI and
ER for both contrasts. Edge/weighting support required edge enrichment/lower
bound at least 2.0, core ratio/upper bound no greater than 0.75, at least 25%
attenuation under a matched aggregation control, and both EI/ER stratum rules.

## Fixed threshold/rank control

| Contrast | S0 BH-FDR voxels | S8 BH-FDR voxels | FDR Jaccard | matched k | top-|t| Jaccard |
|---|---:|---:|---:|---:|---:|
| gain_demean | 20 | 19238 | 0.0010 | 20 | 0.0811 |
| loss_demean | 21 | 20029 | 0.0010 | 21 | 0.0769 |

BH-FDR q=0.05 was fixed and secondary. The matched-cardinality top-|t| control
distinguishes threshold crossing from wholesale rank relocation; neither was
tuned after outcome access.

## Frozen decision audit

- `amplitude_small_subject_guard`: `true`
- `amplitude_split_direction`: `true`
- `amplitude_support`: `false`
- `edge_small_subject_guard`: `false`
- `edge_split_direction`: `false`
- `edge_support`: `false`
- `residual_small_subject_guard`: `true`
- `residual_support`: `true`
- `threshold_disconnect_small_subject_guard`: `true`
- `threshold_disconnect_split_direction`: `false`
- `threshold_disconnect_support`: `false`

The deterministic EI/ER checks, split halves, 1,000 paired cohort-stratified
bootstraps (PCG64 seed 20260817), leave-one-subject-out threshold guard, and
amplitude-alignment negative control are recorded across
`outputs/mechanism_results.tsv`, `outputs/threshold_stability.tsv`, and
`outputs/robustness_results.tsv`. Within-ds001734 robustness is not independent
confirmation.

## Deviations and failed attempts

- Before neural outcome access, the first outcome-blind 100%-coverage core
  failed its own feasibility guard. The contract was transparently re-frozen as
  v2 using one erosion of the fixed common mask; no endpoint or SESOI changed.
- The first pilot wrote no checkpoint because a run lacked the optional
  `trial_type.missed` column. The implementation was repaired to preserve all
  task columns present in the unchanged FitLins design while requiring both
  target contrasts and the intercept.
- `pytest` was absent, so the six test functions were invoked directly;
  all passed, and no package was installed.
- The first completed aggregation attempt stopped at the frozen affine negative
  control because the algebraic `1-r^2` residual lost precision near `r=1`.
  Direct OLS-residual evaluation fixed the numerical implementation; the
  unchanged negative control then passed at residual-RMS ratios below `6e-16`.

## Reproduction

From the workspace root on Sherlock:

```bash
module load python/3.12.1
/home/users/zijiao/pcvenv/bin/python outputs/code/inventory_sources.py
/home/users/zijiao/pcvenv/bin/python outputs/code/build_masks.py
mkdir -p /scratch/users/zijiao/episode02_narps_analysis/{logs,checkpoints}
sbatch outputs/code/fit_subjects.sbatch
# After exactly 108 valid checkpoints:
sbatch outputs/code/aggregate_results.sbatch
/home/users/zijiao/pcvenv/bin/python outputs/code/finalize_results.py
```

The frozen v2 contract SHA-256 is
`27a58a09ee043e71b7d1fca93de0ef727b79fc6ecbd9fef2eaf800208317829f`.
High-frequency checkpoints remain on scratch; durable summaries and code are in
`outputs/`.
