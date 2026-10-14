# EP14 discarded-direction diagnostics

Exploratory development evidence. Original no-candidate selection and closed public holdout are unchanged.

Discarded-score predictors use fold-local directions/scales and nested leave-one-site-out tuning. Geometry uses five fixed penalties without outcome selection; alpha1000 is the prespecified anchor. Every evaluation row is retained in the all-row panels.

Negative contrasts favor the first named model. Absolute utility compares against the outer-training age median. Centered utility compares against zero prediction of centered age; centering is evaluator-only and is recomputed within each bootstrap sample. It is not a deployable correction. Domains are equally weighted after participant-balanced appearance losses; predictions are not averaged across views.

Intervals are 10,000 paired hierarchical bootstrap percentiles, conditional on observed support/domains. They are descriptive, with no equivalence claim or omnibus success criterion. All paired models use identical participants. Common-support centered summaries require at least10 unique participants and ageSD>=2.5years per domain; all exclusions are explicit. The absolute_on_centered_eligible_domains companion uses exactly those same domains/participants for joint absolute/centered interpretation; the broader absolute panel remains descriptive of all covered domains. Per-view panels are reported without intervals in JSON.

## Desikan

Maximum restored/raw prediction discrepancy across alphas: 3.41e-13 years.

### External all rows

| Metric | Comparison | Δ MAE, years | Descriptive 95% interval | Domains |
| --- | --- | ---: | --- | ---: |
| absolute | discarded_site − age_median | -0.4511 | [-1.7908, 0.8285] | 13 |
| absolute | discarded_random − age_median | -1.4310 | [-3.0093, -0.0839] | 13 |
| absolute | discarded_permuted − age_median | -0.0780 | [-1.0942, 0.9083] | 13 |
| absolute | discarded_site − discarded_random | 0.9800 | [-0.4171, 2.7066] | 13 |
| absolute | discarded_site − discarded_permuted | -0.3731 | [-1.1597, 0.4468] | 13 |
| absolute | geometry_retained_alpha1000 − geometry_raw_alpha1000 | 0.3613 | [0.0274, 0.6808] | 13 |
| absolute | geometry_rescaled_alpha1000 − geometry_retained_alpha1000 | 0.0145 | [-0.0143, 0.0436] | 13 |
| absolute | geometry_restored_alpha1000 − geometry_retained_alpha1000 | -0.3613 | [-0.6808, -0.0274] | 13 |
| centered | discarded_site − centered_zero | -0.4224 | [-0.9009, 0.1654] | 13 |
| centered | discarded_random − centered_zero | -0.1985 | [-0.6887, 0.2829] | 13 |
| centered | discarded_permuted − centered_zero | 0.0770 | [-0.0880, 0.2224] | 13 |
| centered | discarded_site − discarded_random | -0.2240 | [-0.7583, 0.4761] | 13 |
| centered | discarded_site − discarded_permuted | -0.4994 | [-0.9349, 0.0309] | 13 |
| centered | geometry_retained_alpha1000 − geometry_raw_alpha1000 | 0.1491 | [-0.1013, 0.3906] | 13 |
| centered | geometry_rescaled_alpha1000 − geometry_retained_alpha1000 | -0.0002 | [-0.0281, 0.0229] | 13 |
| centered | geometry_restored_alpha1000 − geometry_retained_alpha1000 | -0.1491 | [-0.3906, 0.1013] | 13 |

| Domain | Supported / all participants | Supported / all appearances | Age range | Age SD | Common centered eligible |
| --- | ---: | ---: | --- | ---: | --- |
| 14 | 32 / 32 | 32 / 32 | 18.00–40.00 | 5.037 | True |
| 2 | 14 / 14 | 14 / 14 | 19.00–37.00 | 4.838 | True |
| 21 | 10 / 10 | 10 / 10 | 21.00–40.00 | 5.477 | True |
| 24 | 151 / 151 | 453 / 453 | 20.17–81.94 | 16.949 | True |
| 33 | 7 / 7 | 7 / 7 | 8.00–14.20 | 2.006 | False |
| 39 | 21 / 21 | 42 / 42 | 9.44–33.24 | 6.909 | True |
| 41 | 21 / 21 | 42 / 42 | 8.17–42.00 | 10.127 | True |
| 5 | 40 / 40 | 80 / 80 | 19.00–59.00 | 7.652 | True |
| 51 | 12 / 12 | 24 / 24 | 17.00–56.20 | 11.383 | True |
| 52 | 17 / 17 | 51 / 51 | 12.04–24.83 | 3.404 | True |
| 54 | 26 / 26 | 52 / 52 | 7.00–48.00 | 9.860 | True |
| 55 | 24 / 24 | 24 / 24 | 18.00–64.00 | 14.736 | True |
| 8 | 41 / 41 | 123 / 123 | 9.95–39.25 | 6.877 | True |

### External common support

| Metric | Comparison | Δ MAE, years | Descriptive 95% interval | Domains |
| --- | --- | ---: | --- | ---: |
| absolute | discarded_site − age_median | 0.9155 | [-0.0785, 2.0398] | 13 |
| absolute | discarded_random − age_median | -0.3174 | [-1.9354, 1.0179] | 13 |
| absolute | discarded_permuted − age_median | 0.4382 | [-0.5244, 1.3480] | 13 |
| absolute | discarded_site − discarded_random | 1.2329 | [-0.1448, 2.9453] | 13 |
| absolute | discarded_site − discarded_permuted | 0.4773 | [-0.4270, 1.3935] | 13 |
| absolute | geometry_retained_alpha1000 − geometry_raw_alpha1000 | 0.2081 | [-0.1282, 0.5308] | 13 |
| absolute | geometry_rescaled_alpha1000 − geometry_retained_alpha1000 | 0.0250 | [-0.0049, 0.0551] | 13 |
| absolute | geometry_restored_alpha1000 − geometry_retained_alpha1000 | -0.2081 | [-0.5308, 0.1282] | 13 |
| absolute_on_centered_eligible_domains | discarded_site − age_median | 0.5346 | [-0.4025, 1.5978] | 9 |
| absolute_on_centered_eligible_domains | discarded_random − age_median | -0.0878 | [-1.1667, 0.9894] | 9 |
| absolute_on_centered_eligible_domains | discarded_permuted − age_median | 0.1007 | [-0.9664, 1.1008] | 9 |
| absolute_on_centered_eligible_domains | discarded_site − discarded_random | 0.6225 | [-0.2386, 1.5325] | 9 |
| absolute_on_centered_eligible_domains | discarded_site − discarded_permuted | 0.4339 | [-0.5811, 1.3689] | 9 |
| absolute_on_centered_eligible_domains | geometry_retained_alpha1000 − geometry_raw_alpha1000 | 0.0957 | [-0.3098, 0.5017] | 9 |
| absolute_on_centered_eligible_domains | geometry_rescaled_alpha1000 − geometry_retained_alpha1000 | 0.0341 | [0.0062, 0.0659] | 9 |
| absolute_on_centered_eligible_domains | geometry_restored_alpha1000 − geometry_retained_alpha1000 | -0.0957 | [-0.5017, 0.3098] | 9 |
| centered | discarded_site − centered_zero | -0.3327 | [-0.7431, 0.0771] | 9 |
| centered | discarded_random − centered_zero | -0.1381 | [-0.6770, 0.3536] | 9 |
| centered | discarded_permuted − centered_zero | 0.1278 | [-0.0071, 0.2698] | 9 |
| centered | discarded_site − discarded_random | -0.1946 | [-0.7021, 0.4360] | 9 |
| centered | discarded_site − discarded_permuted | -0.4606 | [-0.8649, -0.0579] | 9 |
| centered | geometry_retained_alpha1000 − geometry_raw_alpha1000 | 0.1584 | [-0.1917, 0.4531] | 9 |
| centered | geometry_rescaled_alpha1000 − geometry_retained_alpha1000 | 0.0004 | [-0.0244, 0.0335] | 9 |
| centered | geometry_restored_alpha1000 − geometry_retained_alpha1000 | -0.1584 | [-0.4531, 0.1917] | 9 |

| Domain | Supported / all participants | Supported / all appearances | Age range | Age SD | Common centered eligible |
| --- | ---: | ---: | --- | ---: | --- |
| 14 | 29 / 32 | 29 / 32 | 18.00–29.00 | 3.390 | True |
| 2 | 12 / 14 | 12 / 14 | 19.00–25.00 | 1.700 | False |
| 21 | 6 / 10 | 6 / 10 | 21.00–27.00 | 2.494 | False |
| 24 | 49 / 151 | 125 / 453 | 20.17–34.37 | 3.904 | True |
| 33 | 7 / 7 | 7 / 7 | 8.00–14.20 | 2.006 | False |
| 39 | 21 / 21 | 39 / 42 | 9.44–33.24 | 6.909 | True |
| 41 | 18 / 21 | 34 / 42 | 8.17–34.43 | 8.064 | True |
| 5 | 20 / 40 | 40 / 80 | 19.00–29.00 | 2.657 | True |
| 51 | 10 / 12 | 19 / 24 | 17.00–34.10 | 5.192 | True |
| 52 | 17 / 17 | 51 / 51 | 12.04–24.83 | 3.404 | True |
| 54 | 22 / 26 | 39 / 52 | 7.00–33.00 | 7.214 | True |
| 55 | 10 / 24 | 10 / 24 | 18.00–25.00 | 2.441 | False |
| 8 | 40 / 41 | 114 / 123 | 9.95–34.82 | 6.347 | True |

## Destrieux

Maximum restored/raw prediction discrepancy across alphas: 2.33e-12 years.

### External all rows

| Metric | Comparison | Δ MAE, years | Descriptive 95% interval | Domains |
| --- | --- | ---: | --- | ---: |
| absolute | discarded_site − age_median | -0.5220 | [-1.8764, 0.7038] | 13 |
| absolute | discarded_random − age_median | -0.6762 | [-2.0445, 0.6648] | 13 |
| absolute | discarded_permuted − age_median | -0.0839 | [-1.0387, 0.8244] | 13 |
| absolute | discarded_site − discarded_random | 0.1542 | [-1.3340, 1.5250] | 13 |
| absolute | discarded_site − discarded_permuted | -0.4381 | [-1.1437, 0.0989] | 13 |
| absolute | geometry_retained_alpha1000 − geometry_raw_alpha1000 | -0.0032 | [-0.4472, 0.3535] | 13 |
| absolute | geometry_rescaled_alpha1000 − geometry_retained_alpha1000 | 0.0157 | [-0.0072, 0.0415] | 13 |
| absolute | geometry_restored_alpha1000 − geometry_retained_alpha1000 | 0.0032 | [-0.3535, 0.4472] | 13 |
| centered | discarded_site − centered_zero | -0.2943 | [-0.6375, 0.0270] | 13 |
| centered | discarded_random − centered_zero | 0.5441 | [-0.3720, 1.4037] | 13 |
| centered | discarded_permuted − centered_zero | 0.0188 | [-0.0717, 0.1312] | 13 |
| centered | discarded_site − discarded_random | -0.8384 | [-1.6193, -0.0017] | 13 |
| centered | discarded_site − discarded_permuted | -0.3131 | [-0.6544, -0.0032] | 13 |
| centered | geometry_retained_alpha1000 − geometry_raw_alpha1000 | 0.1196 | [-0.0822, 0.3066] | 13 |
| centered | geometry_rescaled_alpha1000 − geometry_retained_alpha1000 | 0.0115 | [-0.0054, 0.0258] | 13 |
| centered | geometry_restored_alpha1000 − geometry_retained_alpha1000 | -0.1196 | [-0.3066, 0.0822] | 13 |

| Domain | Supported / all participants | Supported / all appearances | Age range | Age SD | Common centered eligible |
| --- | ---: | ---: | --- | ---: | --- |
| 14 | 32 / 32 | 32 / 32 | 18.00–40.00 | 5.037 | True |
| 2 | 14 / 14 | 14 / 14 | 19.00–37.00 | 4.838 | True |
| 21 | 10 / 10 | 10 / 10 | 21.00–40.00 | 5.477 | True |
| 24 | 151 / 151 | 453 / 453 | 20.17–81.94 | 16.949 | True |
| 33 | 7 / 7 | 7 / 7 | 8.00–14.20 | 2.006 | False |
| 39 | 21 / 21 | 42 / 42 | 9.44–33.24 | 6.909 | True |
| 41 | 21 / 21 | 42 / 42 | 8.17–42.00 | 10.127 | True |
| 5 | 40 / 40 | 80 / 80 | 19.00–59.00 | 7.652 | True |
| 51 | 12 / 12 | 24 / 24 | 17.00–56.20 | 11.383 | True |
| 52 | 17 / 17 | 51 / 51 | 12.04–24.83 | 3.404 | True |
| 54 | 26 / 26 | 52 / 52 | 7.00–48.00 | 9.860 | True |
| 55 | 24 / 24 | 24 / 24 | 18.00–64.00 | 14.736 | True |
| 8 | 41 / 41 | 123 / 123 | 9.95–39.25 | 6.877 | True |

### External common support

| Metric | Comparison | Δ MAE, years | Descriptive 95% interval | Domains |
| --- | --- | ---: | --- | ---: |
| absolute | discarded_site − age_median | 0.7253 | [-0.5594, 2.1284] | 13 |
| absolute | discarded_random − age_median | 0.3808 | [-0.7998, 1.5902] | 13 |
| absolute | discarded_permuted − age_median | 0.4844 | [-0.4085, 1.3151] | 13 |
| absolute | discarded_site − discarded_random | 0.3445 | [-1.1717, 1.8445] | 13 |
| absolute | discarded_site − discarded_permuted | 0.2410 | [-0.5035, 1.1975] | 13 |
| absolute | geometry_retained_alpha1000 − geometry_raw_alpha1000 | -0.1100 | [-0.5129, 0.2038] | 13 |
| absolute | geometry_rescaled_alpha1000 − geometry_retained_alpha1000 | 0.0218 | [0.0002, 0.0461] | 13 |
| absolute | geometry_restored_alpha1000 − geometry_retained_alpha1000 | 0.1100 | [-0.2038, 0.5129] | 13 |
| absolute_on_centered_eligible_domains | discarded_site − age_median | 0.2197 | [-1.1157, 1.4925] | 9 |
| absolute_on_centered_eligible_domains | discarded_random − age_median | -0.2468 | [-1.1751, 0.7285] | 9 |
| absolute_on_centered_eligible_domains | discarded_permuted − age_median | 0.1493 | [-0.9342, 1.1914] | 9 |
| absolute_on_centered_eligible_domains | discarded_site − discarded_random | 0.4665 | [-0.5682, 1.5784] | 9 |
| absolute_on_centered_eligible_domains | discarded_site − discarded_permuted | 0.0705 | [-0.4041, 0.4919] | 9 |
| absolute_on_centered_eligible_domains | geometry_retained_alpha1000 − geometry_raw_alpha1000 | 0.1068 | [-0.1275, 0.3409] | 9 |
| absolute_on_centered_eligible_domains | geometry_rescaled_alpha1000 − geometry_retained_alpha1000 | 0.0115 | [-0.0076, 0.0308] | 9 |
| absolute_on_centered_eligible_domains | geometry_restored_alpha1000 − geometry_retained_alpha1000 | -0.1068 | [-0.3409, 0.1275] | 9 |
| centered | discarded_site − centered_zero | -0.0971 | [-0.3503, 0.1997] | 9 |
| centered | discarded_random − centered_zero | 0.4083 | [-0.3644, 1.1868] | 9 |
| centered | discarded_permuted − centered_zero | 0.0256 | [-0.0874, 0.1293] | 9 |
| centered | discarded_site − discarded_random | -0.5054 | [-1.1970, 0.2201] | 9 |
| centered | discarded_site − discarded_permuted | -0.1227 | [-0.3823, 0.1837] | 9 |
| centered | geometry_retained_alpha1000 − geometry_raw_alpha1000 | 0.1516 | [-0.0498, 0.3571] | 9 |
| centered | geometry_rescaled_alpha1000 − geometry_retained_alpha1000 | 0.0128 | [-0.0052, 0.0314] | 9 |
| centered | geometry_restored_alpha1000 − geometry_retained_alpha1000 | -0.1516 | [-0.3571, 0.0498] | 9 |

| Domain | Supported / all participants | Supported / all appearances | Age range | Age SD | Common centered eligible |
| --- | ---: | ---: | --- | ---: | --- |
| 14 | 29 / 32 | 29 / 32 | 18.00–29.00 | 3.390 | True |
| 2 | 12 / 14 | 12 / 14 | 19.00–25.00 | 1.700 | False |
| 21 | 6 / 10 | 6 / 10 | 21.00–27.00 | 2.494 | False |
| 24 | 49 / 151 | 125 / 453 | 20.17–34.37 | 3.904 | True |
| 33 | 7 / 7 | 7 / 7 | 8.00–14.20 | 2.006 | False |
| 39 | 21 / 21 | 39 / 42 | 9.44–33.24 | 6.909 | True |
| 41 | 18 / 21 | 34 / 42 | 8.17–34.43 | 8.064 | True |
| 5 | 20 / 40 | 40 / 80 | 19.00–29.00 | 2.657 | True |
| 51 | 10 / 12 | 19 / 24 | 17.00–34.10 | 5.192 | True |
| 52 | 17 / 17 | 51 / 51 | 12.04–24.83 | 3.404 | True |
| 54 | 22 / 26 | 39 / 52 | 7.00–33.00 | 7.214 | True |
| 55 | 10 / 24 | 10 / 24 | 18.00–25.00 | 2.441 | False |
| 8 | 40 / 41 | 114 / 123 | 9.95–34.82 | 6.347 | True |

The complete JSON retains both roles, every domain, all view-specific results, all fixed-alpha curves, tuning choices, support counts and paired contrasts. No support fallback or age/domain selection was made from prediction outcomes. Composition, regularization geometry, redundancy and biological explanations remain distinct; predictive utility alone does not identify causality or a biological aging mechanism.
