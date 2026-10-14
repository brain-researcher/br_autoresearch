# Exploratory NARPS conditional-estimand sensitivity

All **108 participants × four runs** are present (54 EI, 54 ER). The summaries retain participant pairing and fixed common spatial support.

The spatial support was amended with scientist approval after observed input-validity failures: **37,607** of **37,615** original common-mask voxels are finite at every frame and have a positive run mean in all 432 runs. The original-support complete-cohort analysis remains unavailable. All summaries here use this one amended support.

Design-quality failures affect **103 participants** and **463 run/arm combinations**. Finite estimates remain exploratory; failed gates do not establish the original specificity claim.

Those failures include 362 relative-efficiency and 213 VIF failures, with
overlap; they are not 463 failed scans. Two gain coefficients are not estimable
for sub-080 (ER), run 4, in S01 and R11. Consequently 24 of 252 scalar summaries
and 16 of 108 group-map panels remain unavailable. The participant is retained;
no reduced-cohort result replaces these panels. All other 228 scalar summaries
and 92 group-map panels are available.

| Stratum | Slope | Cross-half displacement cosine, mean [t95 CI] | Symmetric relative displacement, mean [t95 CI] |
|---|---|---|---|
| all | gain | 0.0188 [0.0046, 0.0330] | 0.9434 [0.8953, 0.9915] |
| all | loss | 0.0207 [0.0058, 0.0355] | 0.8110 [0.7680, 0.8541] |
| EI | gain | 0.0237 [0.0041, 0.0433] | 1.0345 [0.9638, 1.1052] |
| EI | loss | 0.0278 [0.0078, 0.0478] | 0.7812 [0.7295, 0.8329] |
| ER | gain | 0.0139 [-0.0071, 0.0349] | 0.8523 [0.7947, 0.9099] |
| ER | loss | 0.0135 [-0.0090, 0.0360] | 0.8409 [0.7712, 0.9106] |

Intervals describe participant means without multiplicity correction. Spatial voxels are not treated as independent observations. These ratios do not test the original equivalence margins.

[Full report and all six contrasts](exploratory_fmri_summary.json) · [Participant metrics](participant_metrics.csv) · [Offer-sequence influence](offer_sequence_omissions.csv)

A required nonestimable arm makes its dependent full-stratum summaries unavailable. No participant or voxel subset is selected to recover a positive result. Team maps remain closed; this analysis does not identify a psychological or temporal neural mechanism.

Both the fully adjusted maps and their specification differences show weak
cross-half spatial agreement. The relative RMS values above describe coefficient
differences relative to map magnitude, not explained variance or a proportion
of changed voxels. Large differences may include estimation noise and reduced
precision. Raw coefficient magnitudes depend on the fixed SPM impulse basis
(oversampling 50); they are not peak evoked BOLD amplitudes.

Complete six-axis figures, using only the saved summary values:
[cross-half agreement](../../../figures/exploratory_fmri_axes_20261013_002/exploratory_fmri_cross_half_cosine.png)
and [coefficient magnitude](../../../figures/exploratory_fmri_axes_20261013_002/exploratory_fmri_four_run_map_rms.png).
PDFs are alongside the PNGs. Earlier plot attempt 001 remains preserved; 002
clarifies coefficient units without changing any value.
