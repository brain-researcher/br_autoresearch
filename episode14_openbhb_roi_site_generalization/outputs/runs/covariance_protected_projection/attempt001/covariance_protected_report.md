# Exact fitting-covariance protection

Exploratory exposed-development counterfactual. Original fit scale and four site directions are reused. Projecting those directions into the true or within-site-shuffled fitting covariance's orthogonal complement preserves rank four. No final scaling. All downstream ages are true fitting ages; evaluation transforms receive features only.

Raw/original predictions and ordinary seen-site correctness are copied exactly from the completed source. Protecting fitting covariance does not preserve ridge predictions or held-out utility. True and sham rotate the original subspace by potentially different amounts; principal angles, projector displacement and removed feature energy are reported in JSON.

Fixed alpha1000 anchor; all five penalties retained without selection. Equal participants within domain, then equal domains. Descriptive 95% intervals use 10,000 paired hierarchical domain/participant draws with centers recomputed. Common-support eligibility stays fixed; zero-support domains remain visible. Final holdout stays closed and original no-candidate unchanged.

## desikan

| Population | Metric | Comparison | Δ MAE years | Descriptive 95% interval | Domains |
| --- | --- | --- | ---: | --- | ---: |
| all_rows | absolute | covariance_protected − original | -0.0303 | [-0.0821, 0.0207] | 13 |
| all_rows | absolute | sham_protected − original | -0.0344 | [-0.0884, 0.0193] | 13 |
| all_rows | absolute | covariance_protected − sham_protected | 0.0041 | [-0.0018, 0.0098] | 13 |
| all_rows | absolute | covariance_protected − raw | 0.3310 | [0.0268, 0.6274] | 13 |
| all_rows | absolute | original − raw | 0.3613 | [0.0274, 0.6808] | 13 |
| all_rows | centered | covariance_protected − original | -0.0082 | [-0.0530, 0.0323] | 13 |
| all_rows | centered | sham_protected − original | -0.0103 | [-0.0574, 0.0312] | 13 |
| all_rows | centered | covariance_protected − sham_protected | 0.0021 | [-0.0014, 0.0068] | 13 |
| all_rows | centered | covariance_protected − raw | 0.1409 | [-0.0834, 0.3567] | 13 |
| all_rows | centered | original − raw | 0.1491 | [-0.1013, 0.3906] | 13 |
| common_support | absolute | covariance_protected − original | -0.0206 | [-0.0768, 0.0349] | 13 |
| common_support | absolute | sham_protected − original | -0.0225 | [-0.0814, 0.0355] | 13 |
| common_support | absolute | covariance_protected − sham_protected | 0.0019 | [-0.0036, 0.0071] | 13 |
| common_support | absolute | covariance_protected − raw | 0.1875 | [-0.1253, 0.4871] | 13 |
| common_support | absolute | original − raw | 0.2081 | [-0.1282, 0.5308] | 13 |
| common_support | centered | covariance_protected − original | 0.0036 | [-0.0477, 0.0534] | 9 |
| common_support | centered | sham_protected − original | 0.0006 | [-0.0521, 0.0529] | 9 |
| common_support | centered | covariance_protected − sham_protected | 0.0030 | [-0.0015, 0.0070] | 9 |
| common_support | centered | covariance_protected − raw | 0.1620 | [-0.1511, 0.4195] | 9 |
| common_support | centered | original − raw | 0.1584 | [-0.1917, 0.4531] | 9 |
| common_support | absolute_on_centered_eligible_domains | covariance_protected − original | 0.0101 | [-0.0460, 0.0725] | 9 |
| common_support | absolute_on_centered_eligible_domains | sham_protected − original | 0.0094 | [-0.0489, 0.0738] | 9 |
| common_support | absolute_on_centered_eligible_domains | covariance_protected − sham_protected | 0.0007 | [-0.0064, 0.0072] | 9 |
| common_support | absolute_on_centered_eligible_domains | covariance_protected − raw | 0.1059 | [-0.2812, 0.4952] | 9 |
| common_support | absolute_on_centered_eligible_domains | original − raw | 0.0957 | [-0.3098, 0.5017] | 9 |

## destrieux

| Population | Metric | Comparison | Δ MAE years | Descriptive 95% interval | Domains |
| --- | --- | --- | ---: | --- | ---: |
| all_rows | absolute | covariance_protected − original | -0.0137 | [-0.0500, 0.0192] | 13 |
| all_rows | absolute | sham_protected − original | -0.0127 | [-0.0568, 0.0347] | 13 |
| all_rows | absolute | covariance_protected − sham_protected | -0.0010 | [-0.0182, 0.0118] | 13 |
| all_rows | absolute | covariance_protected − raw | -0.0170 | [-0.4317, 0.3175] | 13 |
| all_rows | absolute | original − raw | -0.0032 | [-0.4472, 0.3535] | 13 |
| all_rows | centered | covariance_protected − original | -0.0116 | [-0.0296, 0.0097] | 13 |
| all_rows | centered | sham_protected − original | -0.0160 | [-0.0374, 0.0082] | 13 |
| all_rows | centered | covariance_protected − sham_protected | 0.0044 | [-0.0017, 0.0100] | 13 |
| all_rows | centered | covariance_protected − raw | 0.1079 | [-0.0857, 0.2871] | 13 |
| all_rows | centered | original − raw | 0.1196 | [-0.0822, 0.3066] | 13 |
| common_support | absolute | covariance_protected − original | -0.0025 | [-0.0241, 0.0247] | 13 |
| common_support | absolute | sham_protected − original | 0.0016 | [-0.0288, 0.0428] | 13 |
| common_support | absolute | covariance_protected − sham_protected | -0.0041 | [-0.0206, 0.0076] | 13 |
| common_support | absolute | covariance_protected − raw | -0.1125 | [-0.4942, 0.1862] | 13 |
| common_support | absolute | original − raw | -0.1100 | [-0.5129, 0.2038] | 13 |
| common_support | centered | covariance_protected − original | -0.0161 | [-0.0330, 0.0028] | 9 |
| common_support | centered | sham_protected − original | -0.0214 | [-0.0399, 0.0004] | 9 |
| common_support | centered | covariance_protected − sham_protected | 0.0053 | [-0.0013, 0.0103] | 9 |
| common_support | centered | covariance_protected − raw | 0.1355 | [-0.0537, 0.3311] | 9 |
| common_support | centered | original − raw | 0.1516 | [-0.0498, 0.3571] | 9 |
| common_support | absolute_on_centered_eligible_domains | covariance_protected − original | -0.0123 | [-0.0294, 0.0060] | 9 |
| common_support | absolute_on_centered_eligible_domains | sham_protected − original | -0.0149 | [-0.0368, 0.0089] | 9 |
| common_support | absolute_on_centered_eligible_domains | covariance_protected − sham_protected | 0.0026 | [-0.0070, 0.0112] | 9 |
| common_support | absolute_on_centered_eligible_domains | covariance_protected − raw | 0.0945 | [-0.1284, 0.3184] | 9 |
| common_support | absolute_on_centered_eligible_domains | original − raw | 0.1068 | [-0.1275, 0.3409] | 9 |
