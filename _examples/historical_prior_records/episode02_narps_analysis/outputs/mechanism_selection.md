# Prospective mechanism selection

Status: **selected before new source neural-outcome access** on 2026-08-17.
Selection used only `GOAL.md`, source filenames/metadata/NIfTI headers, and the
curated parent packet. No source BOLD, contrast, variance, t, or group-map voxel
values were read.

## Ranking

| Rank | Candidate | Specificity and falsifiability | Input readiness | Cost | Decision |
|---:|---|---|---|---|---|
| 1 | Mask-edge and estimator/weighting mechanism | Directly targets the parent's documented near-zero edge variances and distinguishes spatial support from inverse-variance leverage with outcome-blind masks and matched aggregation controls. Clear enrichment and attenuation thresholds can falsify it. | Complete fMRIPrep masks/BOLD, confounds, and run designs; bounded 0/8 mm recomputation needed. | Moderate | **Retain as Test 2** |
| 2 | Amplitude-versus-shape dependence | Directly decomposes the smoothing difference into affine map-scale/offset and residual spatial-pattern components. Scale alignment has an exact, predeclared estimand and falsifier. | Same bounded 0/8 mm outputs as Test 2. | Low incremental | **Retain as Test 1** |
| 3 | Substantive spatial sensitivity | Scientifically important but is the residual explanation after scale and edge/weighting mechanisms are challenged. It is therefore adjudicated by both retained tests rather than consuming a third test. | Supported by the same maps and controls. | No incremental | Control/alternative, not a separate test |
| 4 | Continuous-versus-thresholded disconnect | Important for interpretation, but threshold stability alone cannot explain the continuous functional-ANOVA share. A fixed BH-FDR and matched-cardinality control will be embedded in Test 1. | Supported after bounded recomputation. | Low incremental | Secondary control, not a separate test |
| 5 | Task-version or sample-composition mechanism | EI/ER have different designs, while all data come from the same ds001734 sample. A direct task-version result would be associative and cannot provide independent confirmation. EI and ER stratification is nevertheless a strong internal falsifier for both retained mechanisms. | Complete 54/54 metadata and run coverage. | Low incremental | Robustness/control, not a separate test |

## Frozen shortlist rationale

The two retained tests share one minimal source-derived computation: the fixed
C1 first-level model at only 0 and 8 mm FWHM for every subject. This is not the
parent's 3 x 3 x 3 multiverse. Test 1 asks whether raw smoothing sensitivity is
mostly removable map amplitude/offset and whether fixed-threshold instability
tracks or departs from continuous spatial change. Test 2 asks whether residual
sensitivity is concentrated in an outcome-blind mask periphery or depends on
very small run variances/inverse-variance fixed effects.

Substantive spatial sensitivity is supported only as the residual account if
scale alignment fails to remove the change and the change persists in the mask
core under equal-run and variance-stabilized aggregation. Neither mechanism may
become a candidate if it appears in only one task-version stratum. EI/ER checks
are internal robustness within ds001734, never independent confirmation.

No unselected candidate may replace either retained test after outcome access.
