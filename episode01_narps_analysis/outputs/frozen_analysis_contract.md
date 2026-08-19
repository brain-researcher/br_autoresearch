# Frozen analysis contract: NARPS analysis-choice attribution

Status: **frozen before neural outcome access** at 2026-08-14T14:22:00Z. This file and its JSON counterpart must not be changed after BOLD or neural contrast/effect-map access.

This is an exploratory analysis-choice attribution study using public NARPS ds001734. It is not independent confirmation, does not establish a biological effect, and does not identify a causal effect of QC. Independent fresh confirmation remains required.

## Scientific question and estimand

The question is whether continuous gain- and loss-related task-fMRI group effect maps are more sensitive to confound handling and outcome-blind subject exclusion than to spatial smoothing. `gain_demean` and `loss_demean` are analyzed separately; magnitudes are never compared between the two contrasts.

For every confound × QC × smoothing cell, each subject contributes a four-run inverse-variance fixed-effect contrast map. At each voxel, the group model is OLS with an intercept and task-version code `+0.5` for equalIndifference (EI) and `-0.5` for equalRange (ER). The primary map is the intercept, which is the equally weighted mean of the EI and ER cohort means regardless of cohort-size imbalance. A direct EI-versus-ER coefficient is not primary and, if shown, is labeled task-version-associated and exploratory.

The BOLD signal is scaled voxelwise to mean 100 in the first-level model. Contrast maps therefore retain percent-signal units per unchanged design-regressor unit. Primary maps are continuous and unthresholded. There is no per-cell spatial centering, z-scoring, winsorization, thresholding, or L2 normalization, and mask voxels have equal weight.

## Frozen data and design scope

All sources are read-only; every new artifact goes under `outputs/`.

- Raw participants/events: `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/input/ds001734`
- fMRIPrep derivatives: `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/fmriprep/ds001734/derivatives`
- FitLins run-level designs: `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses/ds001734/task-MGT/node-runLevel`
- Required imaging grid: MNI152NLin2009cAsym, 2 mm
- Inventory: 108 subjects (EI 54, ER 54), four runs each, 432 usable runs

Technical eligibility is based on estimability of both frozen contrasts in all four runs under every confound model, not full design-matrix rank. Nuisance rank deficiency alone does not exclude a subject. A required nonestimable contrast causes `technical_failure`; it never changes a QC set.

## Frozen QC sets

| Set | Criteria | EI | ER | Total | Excluded from Q0 |
|---|---|---:|---:|---:|---|
| Q0_all | all technically eligible subjects | 54 | 54 | 108 | none |
| Q1_lenient | weighted mean FD ≤ 0.30 mm and every run p(FD > 0.5 mm) ≤ 0.20 | 54 | 52 | 106 | sub-030, sub-100 |
| Q2_strict | Q1 plus weighted mean FD ≤ 0.20 mm and every run p(FD > 0.5 mm) ≤ 0.10 | 54 | 49 | 103 | sub-030, sub-100, sub-016, sub-018, sub-088 |

Framewise displacement is the fMRIPrep `framewise_displacement` column. The definition-driven first-frame NaN is excluded, every other defined frame is in the denominator, subject mean FD is weighted by defined frame count across all four required runs, and no volumes are censored. `std_dvars` is diagnostic only. The implementation must verify `Q2 ⊂ Q1 ⊂ Q0` and the stated counts.

Because all five strict exclusions are ER subjects, Q differences are interpreted as analysis-set/sample-composition sensitivity, not a causal effect of QC.

## Frozen confound and smoothing grid

The complete 3 × 3 × 3 grid has 27 cells per contrast.

| Axis | Level | Frozen definition |
|---|---|---|
| Confound | C0 | six native motion parameters + native run-specific cosine/DCT columns + unchanged task regressors + intercept |
| Confound | C1 | C0 task/DCT/intercept terms with true Friston24: six originals, six first derivatives, six squared originals, six squared derivatives |
| Confound | C2 | C1 + `a_comp_cor_00`…`a_comp_cor_05`, each metadata-verified as combined-mask and `Retained=true` |
| Smoothing | S0 | 0 mm FWHM |
| Smoothing | S1 | 5 mm FWHM |
| Smoothing | S2 | 8 mm FWHM |
| QC | Q0 | frozen Q0_all membership |
| QC | Q1 | frozen Q1_lenient membership |
| QC | Q2 | frozen Q2_strict membership |

The Friston derivatives use first difference with the first row set to zero; squares are computed after originals/derivatives. A selector called `motion24` is never trusted. Nuisance columns remain in native units and are not z-scored. Any nonfinite required value beyond the definition-driven derivative first row, or any missing/non-retained/non-combined required aCompCor component, causes `technical_failure`.

The first-level fit uses the frozen FitLins task regressors, native run-specific DCT columns, and intercept, replacing only the nuisance block. The cell's Gaussian smoothing is applied to the 4D BOLD before fitting; S0 is unsmoothed. Estimation uses a Nilearn voxelwise AR(1) GLM with `drift_model=None`, `standardize=False`, and `signal_scaling=0`. The unchanged FitLins contrast vector defines `gain_demean` or `loss_demean`.

No cell may be imputed, omitted, replaced, or silently reduced. Any missing fitted cell yields `technical_failure`.

## Frozen common mask

Mask construction is outcome-blind. The reference grid is the lexicographically first Q0 run brain mask in the required space/resolution. A run mask is nearest-neighbor resampled only if needed. A voxel is included when covered by at least 95% of all 432 run masks and at least 95% of the 216 run masks in each task-version cohort. There is no erosion, dilation, hole filling, or outcome-based trimming.

The same `outputs/common_analysis_mask.nii.gz` is used for every cell and both contrasts. It must be nonempty and contain at least 50% of the median run-mask voxel count; otherwise the outcome is `technical_failure`.

## Exact variance attribution

For each contrast, let `Y[c,q,s](v)` be the 27 group effect maps on the common mask. A balanced, sum-to-zero functional ANOVA decomposes them into the seven mutually orthogonal map terms:

`C`, `Q`, `S`, `C×Q`, `C×S`, `Q×S`, and `C×Q×S`.

For every term, the squared term maps are summed over their level combinations with the appropriate replication multiplicity and then uniformly over voxels. These sums of squares partition the total across-cell map sum of squares around the grand map exactly. No inferential model treats the 27 overlapping cells as independent.

Interactions are allocated equally among participating factors (a Shapley allocation):

- `A_C = SS_C + 1/2 SS_CQ + 1/2 SS_CS + 1/3 SS_CQS`
- `A_Q = SS_Q + 1/2 SS_CQ + 1/2 SS_QS + 1/3 SS_CQS`
- `A_S = SS_S + 1/2 SS_CS + 1/2 SS_QS + 1/3 SS_CQS`

Factor shares are `A_factor / SS_total`. The primary comparisons are `A_C/A_S`, `A_Q/A_S`, and descriptively `(A_C+A_Q)/A_S`. An interaction is material when its SS is at least 5% of total SS; every interaction is reported regardless.

Uncertainty uses 2,000 paired cohort-stratified Bayesian-bootstrap replicates with PCG64 seed 20260815. In each replicate, one iid Exp(1) weight is drawn for every Q0 subject and reused for all C, Q, S, and both contrasts. Weights are renormalized within each task-version cohort among subjects eligible for each Q set before forming the equal-cohort group map. Ordinary outputs use percentile 95% intervals. The two C/S and Q/S primary ratios use Bonferroni-simultaneous 97.5% intervals (1.25th–98.75th percentiles).

## Continuous map-comparison metrics

The primary endpoint is the functional-ANOVA/Shapley attributed variance above. Pairwise descriptive metrics are spatial Pearson correlation, cosine similarity, sign agreement, and normalized RMS difference. For the last metric:

`NRMSD(a,b) = sqrt(mean((a-b)^2)) / sqrt(mean(reference^2))`,

where the fixed reference is C0-Q0-S0 within the same contrast. A zero denominator is a technical failure.

## Secondary BH-FDR stability

For each cell and contrast, a two-sided OLS t test of the task-version-adjusted group intercept uses `df = N_Q - 2`. Benjamini–Hochberg correction at `q ≤ 0.05` is applied across all common-mask voxels within that one cell/contrast family. Report rejected count/proportion, positive and negative rejected counts, any-rejection status, and pairwise Jaccard stability. This is secondary only: the threshold is not a fourth factor and is never tuned.

## Matched-size random-deletion calibration

Use PCG64 seed 20260814 and 1,000 replicates:

- Q0→Q1: remove a simple random sample without replacement of 0 EI and 2 ER subjects from Q0.
- Q1→Q2: remove a simple random sample without replacement of 0 EI and 3 ER subjects from Q1.

The exact sampled subjects in a replicate are reused across all nine C×S cells and both contrasts. For each transition, contrast, C, and S, calculate NRMSD and `1 - Pearson r` for the observed frozen-Q change and random-deletion changes. Report the null median, 2.5th/97.5th percentiles, observed/null-median ratio, and upper-tail empirical `p = (1 + #null ≥ observed) / 1001`. This calibrates sample composition only and is not a causal QC estimate.

## Frozen decisions, stop rules, and exploratory diagnostic

The full objective is supported only if, for at least one contrast, both `A_C/A_S > 1` and `A_Q/A_S > 1`, and both Bonferroni-simultaneous interval lower bounds exceed 1. If only one ratio or only `(A_C+A_Q)/A_S` exceeds 1, report that exact partial pattern without claiming both confound and exclusion sensitivity exceed smoothing.

`candidate_ready` additionally requires all 27 cells for both contrasts, both matched-size calibrations, every required artifact, and reproducibility checks. A scientifically valid but nonqualifying or unstable result is `closed_no_candidate`. Missing data, invalid aCompCor metadata, contrast nonestimability, invalid mask, insufficient compute, or an incomplete grid is `technical_failure`; frozen rules are never relaxed.

Engineering repairs may correct code or resume computation only when this contract remains byte-for-byte unchanged. Failures and deviations are recorded.

After the primary analysis, the required exploratory diagnostic is ER-cohort leave-one-out continuous-map influence for each of the five frozen excluded subjects, summarized across C×S and contrasts and compared descriptively with retained ER subjects. It is labeled `exploratory_post_primary` and cannot modify the primary analysis.

## Frozen figures and outputs

Primary figures are Shapley-attributed C/Q/S variance-share bars with uncertainty and 27-cell continuous-map correlation heatmaps, each faceted by contrast. Secondary figures show BH-FDR count/Jaccard stability and matched-size deletion calibration. The subject-influence panel is exploratory.

Required terminal artifacts are the two frozen contracts, `qc_subject_sets.tsv`, `multiverse_manifest.tsv`, `variance_attribution.tsv`, `conclusion_stability.tsv`, `matched_n_calibration.tsv`, `figures/`, `RESULT.md`, and a schema-valid `candidate_bundle.json` retaining the exploratory-only and independent-confirmation boundary.
