# Provider-group compression: exposed development only

All three views, six fixed compression dimensions, 24 grouped arms, six global-PCA controls and raw ridge are retained.
All 757 public-validation rows remain closed; the original no-candidate result is unchanged.

| Arm | Dimensions | Domain MAE | Participant MAE | Worst-domain MAE | Oldest-quartile MAE | Site BAcc |
|---|---:|---:|---:|---:|---:|---:|
| raw | 476 | 5.4685 | 7.5209 | 11.9765 | 17.7006 | 0.6359 |
| pca34 | 34 | 5.6413 | 7.7264 | 12.2251 | 18.0932 | 0.5062 |
| anatomical_provider_k1 | 34 | 6.3032 | 8.9376 | 14.6594 | 21.5006 | 0.3080 |
| anatomical_shuffle14021_k1 | 34 | 6.8358 | 9.1984 | 14.6719 | 21.3146 | 0.3602 |
| anatomical_shuffle14022_k1 | 34 | 6.3850 | 8.8241 | 14.1100 | 20.6231 | 0.3391 |
| anatomical_shuffle14023_k1 | 34 | 7.0728 | 9.6171 | 15.7308 | 22.4408 | 0.3805 |
| pca68 | 68 | 5.5969 | 7.6587 | 12.0979 | 18.0142 | 0.5543 |
| anatomical_provider_k2 | 68 | 5.7607 | 8.2580 | 13.5275 | 19.8627 | 0.4506 |
| anatomical_shuffle14021_k2 | 68 | 6.5408 | 9.1578 | 14.9218 | 22.0513 | 0.4868 |
| anatomical_shuffle14022_k2 | 68 | 6.1580 | 8.5380 | 13.6544 | 19.7285 | 0.4287 |
| anatomical_shuffle14023_k2 | 68 | 6.1647 | 8.7892 | 14.7200 | 20.5719 | 0.4569 |
| pca136 | 136 | 5.4792 | 7.5108 | 11.9192 | 17.6124 | 0.6003 |
| anatomical_provider_k4 | 136 | 5.5099 | 7.6843 | 12.4660 | 18.2818 | 0.6086 |
| anatomical_shuffle14021_k4 | 136 | 5.9046 | 8.1810 | 13.1877 | 19.4383 | 0.5758 |
| anatomical_shuffle14022_k4 | 136 | 5.7142 | 8.2288 | 13.7093 | 20.2027 | 0.5660 |
| anatomical_shuffle14023_k4 | 136 | 5.7583 | 8.2817 | 13.8135 | 20.3130 | 0.5811 |
| pca14 | 14 | 5.8101 | 7.9908 | 12.9144 | 18.9774 | 0.4762 |
| measurement_provider_k2 | 14 | 6.8770 | 9.4654 | 15.8325 | 23.6517 | 0.5329 |
| measurement_shuffle14021_k2 | 14 | 6.3915 | 8.9776 | 14.7016 | 21.8382 | 0.2870 |
| measurement_shuffle14022_k2 | 14 | 6.4853 | 9.4073 | 15.9887 | 23.5436 | 0.2766 |
| measurement_shuffle14023_k2 | 14 | 6.5602 | 9.3971 | 15.6375 | 22.6399 | 0.3225 |
| pca28 | 28 | 5.6372 | 7.7061 | 12.2474 | 18.0164 | 0.5019 |
| measurement_provider_k4 | 28 | 5.9312 | 8.4763 | 14.1895 | 21.1587 | 0.6176 |
| measurement_shuffle14021_k4 | 28 | 6.2740 | 8.9927 | 14.9276 | 22.0193 | 0.4266 |
| measurement_shuffle14022_k4 | 28 | 6.3034 | 9.0749 | 15.1788 | 22.3140 | 0.4236 |
| measurement_shuffle14023_k4 | 28 | 6.1057 | 8.8499 | 14.8847 | 21.8719 | 0.4403 |
| pca56 | 56 | 5.6289 | 7.7184 | 12.1846 | 18.0992 | 0.5251 |
| measurement_provider_k8 | 56 | 5.5442 | 7.8802 | 12.9914 | 19.1583 | 0.6084 |
| measurement_shuffle14021_k8 | 56 | 5.8076 | 8.0800 | 13.0785 | 19.2070 | 0.5132 |
| measurement_shuffle14022_k8 | 56 | 5.9044 | 8.1754 | 13.1900 | 19.3552 | 0.5114 |
| measurement_shuffle14023_k8 | 56 | 5.7113 | 7.9771 | 12.9143 | 18.9922 | 0.5096 |

Paired provider-minus-reference differences (years; negative favors provider groups):

| Family | Dimension | Reference | Difference | Descriptive 95% interval |
|---|---:|---|---:|---|
| anatomical | 34 | raw | 0.8347 | [0.2631, 1.4479] |
| anatomical | 34 | pca34 | 0.6619 | [0.0992, 1.2714] |
| anatomical | 34 | anatomical_shuffle14021_k1 | -0.5326 | [-1.2839, 0.1574] |
| anatomical | 34 | anatomical_shuffle14022_k1 | -0.0817 | [-0.7574, 0.5758] |
| anatomical | 34 | anatomical_shuffle14023_k1 | -0.7696 | [-1.9059, 0.2712] |
| anatomical | 68 | raw | 0.2922 | [-0.1472, 0.7149] |
| anatomical | 68 | pca68 | 0.1638 | [-0.2612, 0.5830] |
| anatomical | 68 | anatomical_shuffle14021_k2 | -0.7801 | [-1.5536, -0.1422] |
| anatomical | 68 | anatomical_shuffle14022_k2 | -0.3973 | [-1.0637, 0.1657] |
| anatomical | 68 | anatomical_shuffle14023_k2 | -0.4040 | [-1.1132, 0.2717] |
| anatomical | 136 | raw | 0.0413 | [-0.1736, 0.2707] |
| anatomical | 136 | pca136 | 0.0306 | [-0.2002, 0.2797] |
| anatomical | 136 | anatomical_shuffle14021_k4 | -0.3947 | [-0.8443, 0.0279] |
| anatomical | 136 | anatomical_shuffle14022_k4 | -0.2044 | [-0.6153, 0.2114] |
| anatomical | 136 | anatomical_shuffle14023_k4 | -0.2484 | [-0.7357, 0.2529] |
| measurement | 14 | raw | 1.4085 | [0.4871, 2.4006] |
| measurement | 14 | pca14 | 1.0668 | [0.3311, 1.8555] |
| measurement | 14 | measurement_shuffle14021_k2 | 0.4855 | [-0.1439, 1.2928] |
| measurement | 14 | measurement_shuffle14022_k2 | 0.3916 | [-0.0346, 0.8985] |
| measurement | 14 | measurement_shuffle14023_k2 | 0.3168 | [-0.2668, 1.0775] |
| measurement | 28 | raw | 0.4626 | [-0.0299, 1.0005] |
| measurement | 28 | pca28 | 0.2940 | [-0.1689, 0.7992] |
| measurement | 28 | measurement_shuffle14021_k4 | -0.3428 | [-0.7591, 0.0601] |
| measurement | 28 | measurement_shuffle14022_k4 | -0.3723 | [-0.7996, 0.0587] |
| measurement | 28 | measurement_shuffle14023_k4 | -0.1745 | [-0.5562, 0.2217] |
| measurement | 56 | raw | 0.0756 | [-0.1988, 0.3620] |
| measurement | 56 | pca56 | -0.0847 | [-0.3223, 0.1694] |
| measurement | 56 | measurement_shuffle14021_k8 | -0.2634 | [-0.5159, -0.0032] |
| measurement | 56 | measurement_shuffle14022_k8 | -0.3603 | [-0.6634, -0.0711] |
| measurement | 56 | measurement_shuffle14023_k8 | -0.1671 | [-0.4243, 0.0928] |

Intervals use 10,000 paired domain/participant resamples and are not multiplicity adjusted. Overlapping views are not independent cohorts.
Dimensions are output dimensions; covariance fitting complexity differs between global and block PCA and is not equated by dimension matching.
All fitting, including scaling, PCA and alpha tuning, uses the corresponding training role only. No whitening or post-PCA rescaling.
Each sham is retained individually. Anatomical claims require comparison against both same-size block shams and global PCA; a raw-ridge win alone is insufficient.
