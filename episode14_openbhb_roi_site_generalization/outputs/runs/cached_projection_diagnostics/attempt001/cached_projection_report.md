# EP14 cached projection diagnostics

Exploratory secondary description of exposed development predictions; no fitting or confirmation.

Residual = prediction minus age. Positive values indicate overprediction. All cached ages, domains, roles and views are retained.

For pooled-domain summaries, each participant has total weight 1 within its domain, divided equally among that participant's appearances. Residuals are centered using that same pooled-domain weighting. Per-view summaries instead center within view × domain. Predictions are never averaged across views.

The exact weighted identity is MSE = squared mean residual + centered residual variance. Domain-macro components average domains equally. Centered MAE is a separate diagnostic: MAE has no such additive decomposition. Centering uses evaluation ages and is not a deployable correction. These components do not identify a causal explanation.

JSON includes every domain in both roles and all three per-view summaries. No intervals or significance claims are added.

## Desikan external domains

| Domain | N / appearances | Age range | R0 offset | R3 offset | R0 MAE | R3 MAE | R0 centered MAE | R3 centered MAE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 32 / 32 | 18.00–40.00 | -1.9639 | -3.0798 | 4.1034 | 4.4755 | 3.5574 | 3.5709 |
| 2 | 14 / 14 | 19.00–37.00 | 1.2603 | 1.4899 | 2.9137 | 3.3711 | 2.6124 | 2.9454 |
| 21 | 10 / 10 | 21.00–40.00 | 0.4959 | 0.1595 | 2.7412 | 3.1814 | 2.7811 | 3.1814 |
| 24 | 151 / 453 | 20.17–81.94 | -7.4127 | -8.4763 | 9.6039 | 10.2501 | 7.9747 | 8.1759 |
| 33 | 7 / 7 | 8.00–14.20 | -7.5715 | -7.3661 | 7.6857 | 8.2387 | 5.1442 | 5.0832 |
| 39 | 21 / 42 | 9.44–33.24 | -2.9154 | -3.9814 | 4.5584 | 5.6817 | 4.0053 | 4.7138 |
| 41 | 21 / 42 | 8.17–42.00 | 6.2291 | 5.7374 | 7.0533 | 6.7589 | 5.0221 | 4.5004 |
| 5 | 40 / 80 | 19.00–59.00 | -4.0610 | -3.5277 | 5.2569 | 4.9854 | 4.0175 | 4.2204 |
| 51 | 12 / 24 | 17.00–56.20 | 0.0632 | -1.8375 | 7.8846 | 7.8396 | 7.8636 | 7.8942 |
| 52 | 17 / 51 | 12.04–24.83 | -0.4578 | -0.1043 | 3.7815 | 4.1266 | 3.8505 | 4.1323 |
| 54 | 26 / 52 | 7.00–48.00 | 2.5883 | 3.3768 | 7.8052 | 7.6604 | 7.2734 | 6.9538 |
| 55 | 24 / 24 | 18.00–64.00 | -7.0278 | -8.1265 | 8.0926 | 9.2510 | 6.1248 | 6.4724 |
| 8 | 41 / 123 | 9.95–39.25 | -2.5348 | -2.3451 | 4.3242 | 4.8701 | 4.1713 | 4.4902 |

| Role / view | Δ MAE (years) | Δ centered MAE (years) | Δ MSE (years²) | Δ squared domain offset (years²) | Δ centered variance (years²) |
| --- | ---: | ---: | ---: | ---: | ---: |
| external / pooled | 0.375854 | 0.148920 | 4.617632 | 3.145599 | 1.472033 |
| internal / pooled | -0.090267 | 0.083995 | -0.158473 | -0.779021 | 0.620548 |
| external / 0 | 0.281538 | 0.114315 | 3.415788 | 2.222166 | 1.193622 |
| internal / 0 | -0.010845 | -0.011920 | 0.762701 | 1.869663 | -1.106962 |
| external / 1 | 0.330166 | 0.227235 | 4.642569 | 2.938830 | 1.703739 |
| internal / 1 | -0.129852 | 0.248774 | -0.866593 | -4.015860 | 3.149268 |
| external / 2 | 0.412898 | 0.105002 | 6.175021 | 4.325690 | 1.849331 |
| internal / 2 | 0.058319 | -0.003230 | 3.065208 | 2.130355 | 0.934853 |

## Destrieux external domains

| Domain | N / appearances | Age range | R0 offset | R3 offset | R0 MAE | R3 MAE | R0 centered MAE | R3 centered MAE |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | 32 / 32 | 18.00–40.00 | -2.1848 | -3.2299 | 4.3449 | 4.9059 | 3.6007 | 3.8373 |
| 2 | 14 / 14 | 19.00–37.00 | 1.7885 | 1.3217 | 3.6474 | 3.1854 | 3.0370 | 3.0081 |
| 21 | 10 / 10 | 21.00–40.00 | -0.0479 | -0.4078 | 4.1367 | 4.2312 | 4.1559 | 4.3127 |
| 24 | 151 / 453 | 20.17–81.94 | -5.9563 | -6.8879 | 8.1533 | 8.7340 | 6.9107 | 7.1342 |
| 33 | 7 / 7 | 8.00–14.20 | -8.2977 | -6.5995 | 8.2977 | 6.5995 | 4.4665 | 4.5366 |
| 39 | 21 / 42 | 9.44–33.24 | -2.4872 | -2.6219 | 4.6923 | 4.8553 | 4.6322 | 4.8348 |
| 41 | 21 / 42 | 8.17–42.00 | 5.5222 | 4.3964 | 6.5085 | 5.8961 | 4.4312 | 4.6808 |
| 5 | 40 / 80 | 19.00–59.00 | -4.6655 | -4.7946 | 5.1914 | 5.2903 | 4.2680 | 4.4125 |
| 51 | 12 / 24 | 17.00–56.20 | 0.7805 | 0.3512 | 4.7599 | 4.7458 | 4.5762 | 4.6748 |
| 52 | 17 / 51 | 12.04–24.83 | -0.4585 | 0.0623 | 3.4320 | 3.8351 | 3.4792 | 3.8241 |
| 54 | 26 / 52 | 7.00–48.00 | 3.5084 | 3.2249 | 7.6352 | 7.7198 | 6.5171 | 6.7986 |
| 55 | 24 / 24 | 18.00–64.00 | -5.4771 | -6.5255 | 6.5742 | 7.4803 | 5.2171 | 5.1263 |
| 8 | 41 / 123 | 9.95–39.25 | -2.4456 | -2.2999 | 4.5721 | 4.6289 | 4.2782 | 4.0932 |

| Role / view | Δ MAE (years) | Δ centered MAE (years) | Δ MSE (years²) | Δ squared domain offset (years²) | Δ centered variance (years²) |
| --- | ---: | ---: | ---: | ---: | ---: |
| external / pooled | 0.012471 | 0.131091 | 1.183331 | -0.686462 | 1.869793 |
| internal / pooled | -0.103247 | 0.029515 | 1.874082 | -1.710205 | 3.584287 |
| external / 0 | 0.074293 | 0.120192 | 2.076881 | 0.798013 | 1.278868 |
| internal / 0 | 0.091011 | 0.126224 | 18.469946 | 1.503363 | 16.966583 |
| external / 1 | 0.236698 | 0.214256 | 4.519364 | 1.355757 | 3.163607 |
| internal / 1 | -0.334810 | -0.052622 | -4.865704 | -4.717235 | -0.148468 |
| external / 2 | -0.049999 | 0.097744 | 0.645220 | -1.682331 | 2.327551 |
| internal / 2 | -0.033727 | 0.007635 | 0.119602 | 0.150107 | -0.030505 |

The frozen R0–R3 selection and its primary absolute-MAE endpoint are unchanged. Destrieux and repeated RAMP views are sensitivity descriptions, not independent replications. Mean-offset and centered-error differences can reflect several correlated mechanisms; model geometry, support, and biological explanations remain unresolved.

Sources: the twelve R0/R3 outer JSON files in `../../fixed_development_comparison/attempt001/outer/`, plus the existing development and controls reports used to verify exact reproduction of their external/internal MAEs.
