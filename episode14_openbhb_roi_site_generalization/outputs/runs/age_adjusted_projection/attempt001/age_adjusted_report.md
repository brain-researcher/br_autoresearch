# Linear-age-adjusted direction diagnostic

Exploratory exposed-development evidence. Original no-candidate decision and public holdout remain unchanged.

Training ROI~linear age OLS precedes direction estimation only. Evaluation transforms project original standardized ROIs with no ages or post-projection scaling. The sham permutes fitting ages within each site once. Downstream ridge always fits actual training ages.

Alpha1000 is fixed; all five penalties are sensitivity descriptions without outcome selection. Negative age contrasts favor the first arm. Intervals use 10,000 paired domain/participant bootstrap draws with recentering, conditional on observed domains. Common-support eligibility and evaluator-only centering match the preceding diagnostic. No biological, causal, equivalence or age-preservation conclusion follows automatically.

## Desikan

### External all rows

| Metric | Comparison, alpha1000 | Δ MAE, years | Descriptive 95% interval | Domains |
| --- | --- | ---: | --- | ---: |
| absolute | age_adjusted − original | -0.1599 | [-0.4286, 0.1358] | 13 |
| absolute | sham_adjusted − original | -0.0052 | [-0.3309, 0.3640] | 13 |
| absolute | age_adjusted − sham_adjusted | -0.1547 | [-0.3981, 0.0668] | 13 |
| absolute | age_adjusted − raw | 0.2015 | [-0.0695, 0.4994] | 13 |
| absolute | original − raw | 0.3613 | [0.0274, 0.6808] | 13 |
| centered | age_adjusted − original | -0.0818 | [-0.2914, 0.0827] | 13 |
| centered | sham_adjusted − original | -0.0843 | [-0.3216, 0.0832] | 13 |
| centered | age_adjusted − sham_adjusted | 0.0025 | [-0.1368, 0.1642] | 13 |
| centered | age_adjusted − raw | 0.0673 | [-0.0893, 0.1705] | 13 |
| centered | original − raw | 0.1491 | [-0.1013, 0.3906] | 13 |

| Domain | Supported/all participants | Supported/all appearances | Age range | Age SD | Centered eligible |
| --- | ---: | ---: | --- | ---: | --- |
| 14 | 32/32 | 32/32 | 18.00–40.00 | 5.037 | True |
| 2 | 14/14 | 14/14 | 19.00–37.00 | 4.838 | True |
| 21 | 10/10 | 10/10 | 21.00–40.00 | 5.477 | True |
| 24 | 151/151 | 453/453 | 20.17–81.94 | 16.949 | True |
| 33 | 7/7 | 7/7 | 8.00–14.20 | 2.006 | False |
| 39 | 21/21 | 42/42 | 9.44–33.24 | 6.909 | True |
| 41 | 21/21 | 42/42 | 8.17–42.00 | 10.127 | True |
| 5 | 40/40 | 80/80 | 19.00–59.00 | 7.652 | True |
| 51 | 12/12 | 24/24 | 17.00–56.20 | 11.383 | True |
| 52 | 17/17 | 51/51 | 12.04–24.83 | 3.404 | True |
| 54 | 26/26 | 52/52 | 7.00–48.00 | 9.860 | True |
| 55 | 24/24 | 24/24 | 18.00–64.00 | 14.736 | True |
| 8 | 41/41 | 123/123 | 9.95–39.25 | 6.877 | True |

### External common support

| Metric | Comparison, alpha1000 | Δ MAE, years | Descriptive 95% interval | Domains |
| --- | --- | ---: | --- | ---: |
| absolute | age_adjusted − original | -0.1078 | [-0.3751, 0.1954] | 13 |
| absolute | sham_adjusted − original | -0.0585 | [-0.3585, 0.2944] | 13 |
| absolute | age_adjusted − sham_adjusted | -0.0493 | [-0.2253, 0.1214] | 13 |
| absolute | age_adjusted − raw | 0.1002 | [-0.1837, 0.4216] | 13 |
| absolute | original − raw | 0.2081 | [-0.1282, 0.5308] | 13 |
| absolute_on_centered_eligible_domains | age_adjusted − original | -0.2073 | [-0.4726, 0.0448] | 9 |
| absolute_on_centered_eligible_domains | sham_adjusted − original | -0.2127 | [-0.4929, 0.0675] | 9 |
| absolute_on_centered_eligible_domains | age_adjusted − sham_adjusted | 0.0054 | [-0.1596, 0.1910] | 9 |
| absolute_on_centered_eligible_domains | age_adjusted − raw | -0.1115 | [-0.3825, 0.1220] | 9 |
| absolute_on_centered_eligible_domains | original − raw | 0.0957 | [-0.3098, 0.5017] | 9 |
| centered | age_adjusted − original | -0.0860 | [-0.3269, 0.1714] | 9 |
| centered | sham_adjusted − original | -0.0232 | [-0.2573, 0.2052] | 9 |
| centered | age_adjusted − sham_adjusted | -0.0628 | [-0.1722, 0.0586] | 9 |
| centered | age_adjusted − raw | 0.0724 | [-0.0872, 0.2019] | 9 |
| centered | original − raw | 0.1584 | [-0.1917, 0.4531] | 9 |

| Domain | Supported/all participants | Supported/all appearances | Age range | Age SD | Centered eligible |
| --- | ---: | ---: | --- | ---: | --- |
| 14 | 29/32 | 29/32 | 18.00–29.00 | 3.390 | True |
| 2 | 12/14 | 12/14 | 19.00–25.00 | 1.700 | False |
| 21 | 6/10 | 6/10 | 21.00–27.00 | 2.494 | False |
| 24 | 49/151 | 125/453 | 20.17–34.37 | 3.904 | True |
| 33 | 7/7 | 7/7 | 8.00–14.20 | 2.006 | False |
| 39 | 21/21 | 39/42 | 9.44–33.24 | 6.909 | True |
| 41 | 18/21 | 34/42 | 8.17–34.43 | 8.064 | True |
| 5 | 20/40 | 40/80 | 19.00–29.00 | 2.657 | True |
| 51 | 10/12 | 19/24 | 17.00–34.10 | 5.192 | True |
| 52 | 17/17 | 51/51 | 12.04–24.83 | 3.404 | True |
| 54 | 22/26 | 39/52 | 7.00–33.00 | 7.214 | True |
| 55 | 10/24 | 10/24 | 18.00–25.00 | 2.441 | False |
| 8 | 40/41 | 114/123 | 9.95–34.82 | 6.347 | True |

### Ordinary seen-site probe

| Representation | Balanced accuracy |
| --- | ---: |
| raw | 0.635875 |
| original | 0.620515 |
| age_adjusted | 0.618035 |
| sham_adjusted | 0.618515 |

| Comparison | Δ balanced accuracy | Descriptive 95% interval |
| --- | ---: | --- |
| age_adjusted_minus_original | -0.002480 | [-0.034166, 0.026672] |
| sham_adjusted_minus_original | -0.001999 | [-0.031967, 0.025921] |
| age_adjusted_minus_sham_adjusted | -0.000480 | [-0.014632, 0.010127] |
| age_adjusted_minus_raw | -0.017840 | [-0.051080, 0.011144] |
| original_minus_raw | -0.015360 | [-0.051129, 0.017778] |

### Direction alignment

Squared fraction of training marginal age-covariance norm removed; this is a diagnostic, not an age-preservation guarantee.

| View | Original | Actual-age adjusted | Within-site sham |
| --- | ---: | ---: | ---: |
| 0 | 0.028014 | 0.030836 | 0.006335 |
| 1 | 0.022602 | 0.027920 | 0.007476 |
| 2 | 0.027105 | 0.029336 | 0.004708 |

## Destrieux

### External all rows

| Metric | Comparison, alpha1000 | Δ MAE, years | Descriptive 95% interval | Domains |
| --- | --- | ---: | --- | ---: |
| absolute | age_adjusted − original | 0.0594 | [-0.2917, 0.5444] | 13 |
| absolute | sham_adjusted − original | 0.0447 | [-0.2868, 0.4510] | 13 |
| absolute | age_adjusted − sham_adjusted | 0.0147 | [-0.2500, 0.2542] | 13 |
| absolute | age_adjusted − raw | 0.0562 | [-0.1271, 0.2555] | 13 |
| absolute | original − raw | -0.0032 | [-0.4472, 0.3535] | 13 |
| centered | age_adjusted − original | -0.1402 | [-0.3134, 0.0374] | 13 |
| centered | sham_adjusted − original | -0.1500 | [-0.3148, 0.0322] | 13 |
| centered | age_adjusted − sham_adjusted | 0.0097 | [-0.1152, 0.1238] | 13 |
| centered | age_adjusted − raw | -0.0207 | [-0.1293, 0.0809] | 13 |
| centered | original − raw | 0.1196 | [-0.0822, 0.3066] | 13 |

| Domain | Supported/all participants | Supported/all appearances | Age range | Age SD | Centered eligible |
| --- | ---: | ---: | --- | ---: | --- |
| 14 | 32/32 | 32/32 | 18.00–40.00 | 5.037 | True |
| 2 | 14/14 | 14/14 | 19.00–37.00 | 4.838 | True |
| 21 | 10/10 | 10/10 | 21.00–40.00 | 5.477 | True |
| 24 | 151/151 | 453/453 | 20.17–81.94 | 16.949 | True |
| 33 | 7/7 | 7/7 | 8.00–14.20 | 2.006 | False |
| 39 | 21/21 | 42/42 | 9.44–33.24 | 6.909 | True |
| 41 | 21/21 | 42/42 | 8.17–42.00 | 10.127 | True |
| 5 | 40/40 | 80/80 | 19.00–59.00 | 7.652 | True |
| 51 | 12/12 | 24/24 | 17.00–56.20 | 11.383 | True |
| 52 | 17/17 | 51/51 | 12.04–24.83 | 3.404 | True |
| 54 | 26/26 | 52/52 | 7.00–48.00 | 9.860 | True |
| 55 | 24/24 | 24/24 | 18.00–64.00 | 14.736 | True |
| 8 | 41/41 | 123/123 | 9.95–39.25 | 6.877 | True |

### External common support

| Metric | Comparison, alpha1000 | Δ MAE, years | Descriptive 95% interval | Domains |
| --- | --- | ---: | --- | ---: |
| absolute | age_adjusted − original | 0.0765 | [-0.2633, 0.5505] | 13 |
| absolute | sham_adjusted − original | -0.0220 | [-0.3418, 0.3785] | 13 |
| absolute | age_adjusted − sham_adjusted | 0.0985 | [-0.0755, 0.2871] | 13 |
| absolute | age_adjusted − raw | -0.0334 | [-0.1965, 0.1459] | 13 |
| absolute | original − raw | -0.1100 | [-0.5129, 0.2038] | 13 |
| absolute_on_centered_eligible_domains | age_adjusted − original | -0.1322 | [-0.3637, 0.0980] | 9 |
| absolute_on_centered_eligible_domains | sham_adjusted − original | -0.1315 | [-0.4046, 0.1497] | 9 |
| absolute_on_centered_eligible_domains | age_adjusted − sham_adjusted | -0.0007 | [-0.1694, 0.1707] | 9 |
| absolute_on_centered_eligible_domains | age_adjusted − raw | -0.0254 | [-0.1867, 0.1394] | 9 |
| absolute_on_centered_eligible_domains | original − raw | 0.1068 | [-0.1275, 0.3409] | 9 |
| centered | age_adjusted − original | -0.1339 | [-0.3361, 0.0444] | 9 |
| centered | sham_adjusted − original | -0.0837 | [-0.2745, 0.0917] | 9 |
| centered | age_adjusted − sham_adjusted | -0.0502 | [-0.1826, 0.0814] | 9 |
| centered | age_adjusted − raw | 0.0177 | [-0.0946, 0.1234] | 9 |
| centered | original − raw | 0.1516 | [-0.0498, 0.3571] | 9 |

| Domain | Supported/all participants | Supported/all appearances | Age range | Age SD | Centered eligible |
| --- | ---: | ---: | --- | ---: | --- |
| 14 | 29/32 | 29/32 | 18.00–29.00 | 3.390 | True |
| 2 | 12/14 | 12/14 | 19.00–25.00 | 1.700 | False |
| 21 | 6/10 | 6/10 | 21.00–27.00 | 2.494 | False |
| 24 | 49/151 | 125/453 | 20.17–34.37 | 3.904 | True |
| 33 | 7/7 | 7/7 | 8.00–14.20 | 2.006 | False |
| 39 | 21/21 | 39/42 | 9.44–33.24 | 6.909 | True |
| 41 | 18/21 | 34/42 | 8.17–34.43 | 8.064 | True |
| 5 | 20/40 | 40/80 | 19.00–29.00 | 2.657 | True |
| 51 | 10/12 | 19/24 | 17.00–34.10 | 5.192 | True |
| 52 | 17/17 | 51/51 | 12.04–24.83 | 3.404 | True |
| 54 | 22/26 | 39/52 | 7.00–33.00 | 7.214 | True |
| 55 | 10/24 | 10/24 | 18.00–25.00 | 2.441 | False |
| 8 | 40/41 | 114/123 | 9.95–34.82 | 6.347 | True |

### Ordinary seen-site probe

| Representation | Balanced accuracy |
| --- | ---: |
| raw | 0.630284 |
| original | 0.601399 |
| age_adjusted | 0.618646 |
| sham_adjusted | 0.606163 |

| Comparison | Δ balanced accuracy | Descriptive 95% interval |
| --- | ---: | --- |
| age_adjusted_minus_original | 0.017248 | [-0.005604, 0.045850] |
| sham_adjusted_minus_original | 0.004764 | [-0.014401, 0.027343] |
| age_adjusted_minus_sham_adjusted | 0.012483 | [-0.002090, 0.035475] |
| age_adjusted_minus_raw | -0.011638 | [-0.038802, 0.013815] |
| original_minus_raw | -0.028886 | [-0.066663, 0.005975] |

### Direction alignment

Squared fraction of training marginal age-covariance norm removed; this is a diagnostic, not an age-preservation guarantee.

| View | Original | Actual-age adjusted | Within-site sham |
| --- | ---: | ---: | ---: |
| 0 | 0.003580 | 0.044970 | 0.011975 |
| 1 | 0.001860 | 0.038851 | 0.015001 |
| 2 | 0.011905 | 0.025644 | 0.007666 |

The JSON preserves all domains, roles, overlapping views, fixed penalties, paired contrasts, support and site-class recalls. Per-row predictions and correctness and fitted direction state are retained in the six outer checkpoints. The single sham realization conditions this comparison; it is not a permutation test or independent replication.
