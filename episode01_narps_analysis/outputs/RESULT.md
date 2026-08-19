# NARPS analysis-choice attribution result

## Scope and claim boundary

This is an **exploratory-only analysis-choice attribution study** in public NARPS ds001734 data. It is not independent confirmation, does not establish a biological effect, and does not estimate a causal effect of QC. Independent fresh confirmation remains required.

Gain (`gain_demean`) and loss (`loss_demean`) were analyzed separately. No pooled EI/ER gain-minus-loss contrast is used as a primary result.

## Completion and estimand

All 54 contrast-by-grid cells (27 per contrast) used the frozen common mask of 221,116 voxels and the frozen group intercept estimand: the equally weighted mean of EI and ER cohort means. All multiverse cells overlap in subjects and are not treated as independent observations.

## Primary continuous-map variance attribution

| Contrast | C share (95% interval) | Q share (95% interval) | S share (95% interval) | C/S (97.5% simultaneous interval) | Q/S (97.5% simultaneous interval) | (C+Q)/S (95% interval) |
|---|---:|---:|---:|---:|---:|---:|
| gain_demean | 0.06782 [0.05683, 0.08654] | 0.03821 [0.003381, 0.1277] | 0.894 [0.8062, 0.9314] | 0.07587 [0.06087, 0.1025] | 0.04274 [0.00244, 0.207] | 0.1186 [0.07365, 0.2405] |
| loss_demean | 0.07731 [0.06218, 0.0956] | 0.02723 [0.002338, 0.09135] | 0.8955 [0.8341, 0.9287] | 0.08634 [0.06618, 0.1119] | 0.03041 [0.001731, 0.1354] | 0.1168 [0.07682, 0.1989] |

The exact functional-ANOVA term maps partitioned total across-cell map sum of squares within numerical tolerance. Interactions were allocated equally to their participating factors (the frozen Shapley rule).

Material interactions (observed SS share >= 0.05):
none.

## Frozen decision rule

No contrast met the complete frozen rule requiring both C/S and Q/S point estimates and both simultaneous interval lower bounds to exceed 1. The exact partial patterns are the table values above; they must not be described as showing that both confound and exclusion sensitivity exceed smoothing.

## Descriptive continuous and BH-FDR stability

One-axis pair summaries (median over matched cell pairs):

| Contrast | Axis changed | Pearson r | cosine | NRMSD | FDR Jaccard |
|---|---|---:|---:|---:|---:|
| gain_demean | C | 0.9737 | 0.9665 | 0.0916 | 0.6427 |
| gain_demean | Q | 0.9829 | 0.9818 | 0.07596 | 0.7186 |
| gain_demean | S | 0.607 | 0.6082 | 0.8197 | 0.002235 |
| loss_demean | C | 0.953 | 0.9551 | 0.1233 | 0.6364 |
| loss_demean | Q | 0.9837 | 0.9814 | 0.06852 | 0.7369 |
| loss_demean | S | 0.5859 | 0.5941 | 0.8225 | 0.00419 |

BH-FDR at q<=0.05 is secondary only and was not tuned or treated as a fourth factor.
For gain_demean, rejected counts ranged from 14 to 29,721; 27/27 cells had at least one rejection.
For loss_demean, rejected counts ranged from 12 to 20,029; 27/27 cells had at least one rejection.

## Matched-size deletion calibration

The frozen-Q changes were calibrated against 1,000 cohort-stratified random deletions using the same sampled subject sets for every C×S cell and both contrasts (PCG64 seed 20260814). This is sample-composition calibration, not a causal estimate of QC.
- gain_demean, Q0_to_Q1, normalized_rms_difference: median observed/null-median ratio across C×S = 1.679; 9/9 cells had upper-tail empirical p<=0.05.
- gain_demean, Q0_to_Q1, one_minus_pearson_r: median observed/null-median ratio across C×S = 2.593; 9/9 cells had upper-tail empirical p<=0.05.
- gain_demean, Q1_to_Q2, normalized_rms_difference: median observed/null-median ratio across C×S = 1.142; 2/9 cells had upper-tail empirical p<=0.05.
- gain_demean, Q1_to_Q2, one_minus_pearson_r: median observed/null-median ratio across C×S = 1.349; 2/9 cells had upper-tail empirical p<=0.05.
- loss_demean, Q0_to_Q1, normalized_rms_difference: median observed/null-median ratio across C×S = 1.649; 9/9 cells had upper-tail empirical p<=0.05.
- loss_demean, Q0_to_Q1, one_minus_pearson_r: median observed/null-median ratio across C×S = 2.689; 9/9 cells had upper-tail empirical p<=0.05.
- loss_demean, Q1_to_Q2, normalized_rms_difference: median observed/null-median ratio across C×S = 1.198; 3/9 cells had upper-tail empirical p<=0.05.
- loss_demean, Q1_to_Q2, one_minus_pearson_r: median observed/null-median ratio across C×S = 1.458; 6/9 cells had upper-tail empirical p<=0.05.

## Exploratory post-primary diagnostic

ER leave-one-out influence was computed for all 54 ER subjects and compared the five frozen exclusions (sub-016, sub-018, sub-030, sub-088, sub-100) descriptively with retained ER participants. This diagnostic does not alter the frozen primary analysis.
- gain_demean: median influence across subjects and C×S was 0.09076 for frozen-excluded ER and 0.0774 for retained ER.
- loss_demean: median influence across subjects and C×S was 0.1487 for frozen-excluded ER and 0.1101 for retained ER.

## Exploratory edge-scaling diagnostic

A diagnostic-only audit of all 1,944 subject × C × S × contrast fixed-effect variance arrays found 30,186 array-voxel entries below 1e-20 and 18,057 below 1e-30; the global minimum was 6.89e-33. All stored variances remained finite and strictly positive. Float64 temporal-mean-below-1 logging diagnostics were available for 94/108 subjects; the 14 earlier compatible sidecars predated diagnostic logging and were not recomputed. The float64 counts are not asserted to equal Nilearn's internal float32 clamp decisions in every rounding-edge case. These thresholds and counts are exploratory engineering diagnostics, not a change to, filter on, or sensitivity redefinition of the frozen primary analysis.

## Limitations and reproducibility

All five exclusions are ER participants, so Q sensitivity conflates the prespecified analysis set with sample composition. EI and ER are task versions with different gain support; any direct task-version coefficient would be exploratory. Bayesian-bootstrap intervals quantify paired subject-sampling sensitivity of map-level metrics and are not voxelwise inference. The same dataset generated every exploratory result here. The frozen 95%-coverage mask permits low-intensity edge voxels in individual runs; exact Nilearn signal_scaling=0 behavior clamps temporal means below 1 before percent scaling. Those run/voxel estimates are therefore not literally scaled by their own temporal mean. An exact-zero series becomes constant -100 and can have numerically tiny within-run contrast variance, which can pull its inverse-variance subject fixed effect toward zero. Smoothing reduces many such edge cases, so this boundary behavior can contribute specifically to S attribution. The diagnostic audit is reported separately; no post-hoc mask trimming or variance floor was applied.

Reproduce after all immutable subject checkpoints exist with:

```bash
module load python/3.12.1
/home/users/zijiao/pcvenv/bin/python outputs/code/audit_edge_scaling.py
/home/users/zijiao/pcvenv/bin/python outputs/code/aggregate_results.py --gram-voxel-chunk 8192
/home/users/zijiao/pcvenv/bin/python outputs/code/finalize_candidate.py
```

Frozen randomness: paired Bayesian bootstrap PCG64 seed 20260815 (2,000 replicates); matched-size deletion PCG64 seed 20260814 (1,000 replicates). The Gram-matrix implementation is an exact algebraic evaluation of the frozen map-space sums, correlations, and distances; it does not change the estimand. No frozen-analysis deviation was introduced by this aggregation stage.
