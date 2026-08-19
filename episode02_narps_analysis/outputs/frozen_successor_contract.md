# Frozen successor contract: NARPS smoothing mechanisms

Status: **re-frozen before new neural-outcome access** at
2026-08-17T15:26:35Z. The JSON contract is authoritative and neither contract
may change after source BOLD/effect/variance/t/group-map values are read.

Pre-outcome amendment: the v1 outcome-blind 100%-coverage intersection yielded
only 28,881 eroded core voxels and failed its own core-at-least-50% feasibility
guard. Before any neural outcome access, the core was repaired to the one-voxel
erosion of the fixed 95%-coverage common mask (204,421/221,116 voxels). No
endpoint, prediction, SESOI, subject rule, or outcome-dependent choice changed.
The superseded v1 JSON hash is
`1c7f712a8c5ced45a97f312794df15fa68d83f117b4f0dddf63cc9d35c6f5496`.

This is exploratory-only internal mechanism testing in ds001734. It is not
independent confirmation, scientific acceptance, a biological-mechanism claim,
or a causal QC claim. Independent fresh confirmation remains required.

## Fixed source and model

Use all 108 technically eligible subjects (54 EI, 54 ER), four runs each, with no
QC exclusion or censoring. Analyze `gain_demean` and `loss_demean` separately.
Do not compare their magnitudes and do not pool them.

Fit only C1 (true Friston24 plus unchanged FitLins task regressors, native cosine
terms, and intercept) at 0 and 8 mm FWHM. Smooth the 4D MNI152NLin2009cAsym 2 mm
fMRIPrep BOLD before a Nilearn voxelwise AR(1) GLM with `drift_model=None`,
`standardize=False`, `signal_scaling=0`, and `minimize_memory=True`. Contrast the
unchanged `gain_demean` or `loss_demean` column. This two-level, one-confound
design is not a rerun of the v1 3 x 3 x 3 grid.

For every subject/smoothing/contrast retain the four run effect and variance
arrays and construct:

1. inverse-variance fixed effect (IVW);
2. equal-run mean; and
3. stabilized IVW, flooring each run variance map at its common-mask 0.1st
   percentile before inversion.

All required arrays must be finite and variances strictly positive. Missing or
non-estimable required cells are technical failures unless an engineering-only
repair can preserve this contract exactly.

## Outcome-blind masks and group maps

The common mask requires at least 95% coverage across all 432 run masks and in
each 216-run task-version cohort. The core is the common mask after one
six-neighbor erosion. The periphery is common minus core. The common
mask must have at least half the median run-mask voxel count, the core at least
half the common mask, and the periphery at least 1% of the common mask.

The primary group map is 0.5 times the EI subject mean plus 0.5 times the ER
subject mean. Repeat all decision metrics within EI and ER separately as internal
task-version/sample-composition controls. Absence of a required direction in
either stratum disallows a candidate; these checks are not independent
confirmation.

## Test 1: amplitude versus spatial shape

For each contrast, compare IVW S0 and S8 maps over the common mask:

- `D_raw = RMS(Y8-Y0)/RMS(Y0)`;
- fit `Y0 = alpha + beta*Y8` across voxels;
- `D_aligned = RMS(Y0-alpha-beta*Y8)/RMS(Y0)`;
- `aligned_ratio = D_aligned/D_raw`;
- `scale_attenuation = 1-aligned_ratio`; and
- spatial Pearson correlation `r`.

Amplitude/offset dominance predicts `scale_attenuation >= 0.50` and `r >= 0.80`
for both contrasts. The full-sample point estimates and paired-bootstrap 95%
lower bounds must exceed both thresholds, and EI and ER point estimates must do
so separately. Failure of any component falsifies amplitude/offset dominance.

Matched negative control: apply the same alignment to
`Y_control = 0.1*SD(Y0) + 1.25*Y0`. It must yield `r` within 1e-12 of 1 and
residual RMS no greater than `1e-10*RMS(Y0)`.

Secondary fixed threshold control: two-sided BH-FDR q=0.05 on group intercept t
maps, plus a matched-cardinality top-|t| set with
`k=min(n_rejected_S0,n_rejected_S8)`. FDR Jaccard below 0.25 with top-|t|
Jaccard at least 0.50 flags threshold crossing despite stable ranks. Both below
0.25 flag broad spatial relocation. No disconnect claim is allowed if `k=0`.

## Test 2: mask edge and inverse-variance weighting

For every aggregation, region, group/stratum, and contrast define
`Delta=Y8-Y0`, `edge_enrichment=mean(Delta^2_periphery)/mean(Delta^2_core)`,
`core_ratio=D_raw_core/D_raw_full`, and
`aggregation_attenuation=1-D_raw_alternative_full/D_raw_IVW_full`.

Edge/weighting support requires both contrasts to have:

- full IVW `edge_enrichment >= 2.0`, with bootstrap lower bound meeting 2.0;
- full IVW `core_ratio <= 0.75`, with bootstrap upper bound meeting 0.75;
- at least 25% attenuation, by point estimate and lower bound, under equal-run
  or stabilized IVW; and
- EI and ER point estimates each showing enrichment at least 1.5 and core ratio
  no greater than 0.85.

Equal-run and stabilized-IVW maps are the matched controls. Failure of any
required component falsifies the mechanism. Persistence in the core under both
controls instead supports residual substantive spatial sensitivity.

## Frozen uncertainty, robustness, and decision hierarchy

Use 1,000 paired cohort-stratified ordinary subject bootstraps, PCG64 seed
20260817, reusing sampled indices across contrasts, smoothings, masks, and
aggregations. Exact Gram-matrix evaluation is allowed. Deterministic split halves
alternate sorted subject IDs within EI and ER; candidate direction must agree in
both halves. Leave-one-subject-out influence must not move a relevant full-sample
estimate across its SESOI threshold.

Select at most one terminal candidate in this order:

1. edge/weighting mechanism if Test 2 passes completely;
2. amplitude/offset mechanism if Test 1 passes and Test 2 does not;
3. residual substantive spatial sensitivity if both preceding mechanisms fail
   and, in the equal-run core for both contrasts and EI/ER, `D_raw >= 0.25` and
   `r <= 0.80`; or
4. continuous-threshold disconnect if, for both contrasts and EI/ER, `r >= 0.80`,
   FDR Jaccard `<0.25`, and top-|t| Jaccard `>=0.50`.

Every candidate also requires agreeing split-half directions and the small-subject
guard. If none passes after complete tests, close with `closed_no_candidate`.
Use `technical_failure` only for a genuine source, estimability, mask, runtime,
storage, or checkpoint blocker. Undeclared metrics are post-hoc descriptive and
cannot alter terminal status; no failed candidate may be replaced.

## Resource and stop rule

Run only on Sherlock compute nodes. Use
`/scratch/users/zijiao/episode02_narps_analysis` for high-frequency work and
`outputs/` for durable artifacts. Cap execution at 108 subject tasks, two retries
per failed subject, 1,800 requested core-hours, 1 TB scratch, and 48 wall-clock
hours. Stop after these two tests, their controls/robustness checks, and the
terminal bundle; do not widen the search.
