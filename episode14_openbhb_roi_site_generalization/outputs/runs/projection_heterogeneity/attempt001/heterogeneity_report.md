# EP14 cached projection heterogeneity

External development predictions only; alpha1000 fixed. All13 domains, all ages and every appearance are retained. No new model, support threshold, significance test or causal claim.

Each participant has equal total weight within its domain, divided among appearances; domain means are then equally weighted. Centering is recomputed per model/domain for each population. Predictions are never averaged across views.

For correction c = prediction_other − prediction_base and baseline residual e = prediction_base − age, the exact identity is ΔMSE = 2 E[e c] + E[c²]. The first term describes error/correction alignment; the second is correction energy. Centered MSE uses centered e and c. MAE has no such decomposition.

## Desikan

| Comparison | Δ MAE, years | Δ centered MAE, years | Δ MSE, years² | Alignment, years² | Correction energy, years² |
| --- | ---: | ---: | ---: | ---: | ---: |
| retained_minus_raw | 0.361338 | 0.149076 | 4.497944 | 1.550692 | 2.947252 |
| rescaled_minus_retained | 0.014516 | -0.000156 | 0.119688 | 0.094074 | 0.025615 |
| rescaled_minus_raw | 0.375854 | 0.148920 | 4.617632 | 1.749994 | 2.867638 |
| restored_minus_raw | -0.000000 | -0.000000 | -0.000000 | -0.000000 | 0.000000 |

| Omitted domain | Participants | Domain Δ MAE | Remaining-domain Δ MAE | Domain Δ centered MAE | Remaining-domain Δ centered MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14 | 32 | 0.372178 | 0.376160 | 0.013565 | 0.160199 |
| 2 | 14 | 0.457350 | 0.369062 | 0.332919 | 0.133586 |
| 21 | 10 | 0.440226 | 0.370489 | 0.400288 | 0.127972 |
| 24 | 151 | 0.646176 | 0.353327 | 0.201218 | 0.144561 |
| 33 | 7 | 0.553020 | 0.361090 | -0.061010 | 0.166414 |
| 39 | 21 | 1.123280 | 0.313568 | 0.708516 | 0.102287 |
| 41 | 21 | -0.294333 | 0.431702 | -0.521700 | 0.204805 |
| 5 | 40 | -0.271477 | 0.429798 | 0.202887 | 0.144422 |
| 51 | 12 | -0.045048 | 0.410929 | 0.030607 | 0.158779 |
| 52 | 17 | 0.345132 | 0.378414 | 0.281779 | 0.137848 |
| 54 | 26 | -0.144823 | 0.419243 | -0.319647 | 0.187967 |
| 55 | 24 | 1.158483 | 0.310634 | 0.347622 | 0.132361 |
| 8 | 41 | 0.545931 | 0.361680 | 0.318910 | 0.134754 |

### Common support and changed weights

Covered domains: 14, 2, 21, 24, 33, 39, 41, 5, 51, 52, 54, 55, 8. Uncovered domains remain visible: none.

Supported/complement mixtures use conditional original appearance weights. Actual common-support scores rebalance participants and their eligible appearances. The exact difference below retains a separate appearance-reweighting term; it does not equate these two weighting schemes.

| Contrast/metric | Common − all | Domain reweighting | Within-domain restriction | Appearance reweighting |
| --- | ---: | ---: | ---: | ---: |
| retained_minus_raw/delta_mae | -0.153281 | 0.000000 | -0.166867 | 0.013585 |
| retained_minus_raw/delta_mse | -3.903095 | 0.000000 | -4.052117 | 0.149022 |
| rescaled_minus_retained/delta_mae | 0.010505 | 0.000000 | 0.013381 | -0.002876 |
| rescaled_minus_retained/delta_mse | 0.125624 | 0.000000 | 0.156055 | -0.030431 |
| rescaled_minus_raw/delta_mae | -0.142776 | 0.000000 | -0.153485 | 0.010709 |
| rescaled_minus_raw/delta_mse | -3.777470 | 0.000000 | -3.896062 | 0.118592 |
| restored_minus_raw/delta_mae | 0.000000 | 0.000000 | 0.000000 | -0.000000 |
| restored_minus_raw/delta_mse | 0.000000 | 0.000000 | 0.000000 | -0.000000 |

The all-row and supported summaries on identical covered domains are included in JSON for absolute and independently recentered metrics. No additive support-mixture interpretation is assigned to recentered losses. No new common-support eligibility threshold is imposed.

## Destrieux

| Comparison | Δ MAE, years | Δ centered MAE, years | Δ MSE, years² | Alignment, years² | Correction energy, years² |
| --- | ---: | ---: | ---: | ---: | ---: |
| retained_minus_raw | -0.003243 | 0.119563 | 0.985809 | -1.625504 | 2.611313 |
| rescaled_minus_retained | 0.015714 | 0.011528 | 0.197523 | 0.181846 | 0.015677 |
| rescaled_minus_raw | 0.012471 | 0.131091 | 1.183331 | -1.424454 | 2.607786 |
| restored_minus_raw | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

| Omitted domain | Participants | Domain Δ MAE | Remaining-domain Δ MAE | Domain Δ centered MAE | Remaining-domain Δ centered MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14 | 32 | 0.561002 | -0.033240 | 0.236637 | 0.122296 |
| 2 | 14 | -0.462071 | 0.052016 | -0.028848 | 0.144420 |
| 21 | 10 | 0.094472 | 0.005638 | 0.156876 | 0.128943 |
| 24 | 151 | 0.580770 | -0.034887 | 0.223538 | 0.123387 |
| 33 | 7 | -1.698147 | 0.155023 | 0.070058 | 0.136177 |
| 39 | 21 | 0.163033 | -0.000076 | 0.202605 | 0.125132 |
| 41 | 21 | -0.612394 | 0.064543 | 0.249578 | 0.121217 |
| 5 | 40 | 0.098853 | 0.005273 | 0.144524 | 0.129972 |
| 51 | 12 | -0.014045 | 0.014681 | 0.098599 | 0.133799 |
| 52 | 17 | 0.403162 | -0.020086 | 0.344897 | 0.113274 |
| 54 | 26 | 0.084620 | 0.006459 | 0.281477 | 0.118559 |
| 55 | 24 | 0.906077 | -0.061996 | -0.090718 | 0.149575 |
| 8 | 41 | 0.056795 | 0.008778 | -0.185036 | 0.157435 |

### Common support and changed weights

Covered domains: 14, 2, 21, 24, 33, 39, 41, 5, 51, 52, 54, 55, 8. Uncovered domains remain visible: none.

Supported/complement mixtures use conditional original appearance weights. Actual common-support scores rebalance participants and their eligible appearances. The exact difference below retains a separate appearance-reweighting term; it does not equate these two weighting schemes.

| Contrast/metric | Common − all | Domain reweighting | Within-domain restriction | Appearance reweighting |
| --- | ---: | ---: | ---: | ---: |
| retained_minus_raw/delta_mae | -0.106717 | 0.000000 | -0.104805 | -0.001911 |
| retained_minus_raw/delta_mse | -2.360918 | 0.000000 | -2.290833 | -0.070085 |
| rescaled_minus_retained/delta_mae | 0.006078 | 0.000000 | 0.006343 | -0.000264 |
| rescaled_minus_retained/delta_mse | 0.097684 | 0.000000 | 0.106570 | -0.008886 |
| rescaled_minus_raw/delta_mae | -0.100638 | 0.000000 | -0.098463 | -0.002176 |
| rescaled_minus_raw/delta_mse | -2.263234 | 0.000000 | -2.184262 | -0.078971 |
| restored_minus_raw/delta_mae | -0.000000 | 0.000000 | -0.000000 | -0.000000 |
| restored_minus_raw/delta_mse | -0.000000 | 0.000000 | -0.000000 | 0.000000 |

The all-row and supported summaries on identical covered domains are included in JSON for absolute and independently recentered metrics. No additive support-mixture interpretation is assigned to recentered losses. No new common-support eligibility threshold is imposed.

Fixed age bins are [0,10), …, [90,100], including100 in the last bin. Out-of-range ages receive a separate visible category. Every domain/bin cell, unique-participant count and appearance count is retained in JSON; empty cells are marked. Age-bin macros can have different contributing domain sets, and sparse cells are descriptive only.

The original frozen R0/R3 summaries are retained as context. Previously diagnosed cross-run numerical drift is not relabeled as a passed strict replay. The final holdout remains closed, and this report does not change candidate selection or identify a causal mechanism.
