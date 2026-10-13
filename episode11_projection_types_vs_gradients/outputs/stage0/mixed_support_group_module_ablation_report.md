# EP11 synthetic group-module attribution

- Real projection outcomes read: **no**
- Outcome-bearing search authorized: **no**
- Common backbone: known synthetic group-free density
- Audit labels: integrated for every score
- Effect scale: repaired subset-invariant generator, but still the projected-C proxy rather than full-family calibration

| Fixture | Arm | Replicates | Mean oracle regret | MC SE | Mean group accuracy |
|---|---|---:|---:|---:|---:|
| strong_projected_proxy_target | MOD_DISC | 256 | 0.7282 | 0.0141 | 0.637 |
| strong_projected_proxy_target | MOD_LABEL | 256 | 0.3981 | 0.0096 | 0.853 |
| strong_projected_proxy_target | MOD_SIGNATURE | 256 | 0.0284 | 0.0029 | 0.920 |
| strong_projected_proxy_target | MOD_GATE_FIT | 256 | 0.0636 | 0.0047 | 0.914 |
| strong_projected_proxy_target | MOD_GATE_TRUTH | 256 | 0.0000 | 0.0000 | 0.924 |
| zero_group | MOD_DISC | 256 | 0.1511 | 0.0061 | NA |
| zero_group | MOD_LABEL | 256 | 0.0687 | 0.0036 | NA |
| zero_group | MOD_SIGNATURE | 256 | 0.0000 | 0.0000 | NA |
| zero_group | MOD_GATE_FIT | 256 | 0.0000 | 0.0000 | NA |
| zero_group | MOD_GATE_TRUTH | 256 | 0.0000 | 0.0000 | NA |

## Strong-fixture paired attribution

| Contrast | Mean rescue/regret | MC SE | Paired one-sided 95% bounds (central 90% interval) | Material (>0.005 lower bound) |
|---|---:|---:|---:|---:|
| label_discovery_rescue_including_downstream_signature_and_intercept_refit | 0.3301 | 0.0108 | [0.3123, 0.3479] | unresolved_MC_precision |
| signature_rescue_given_true_labels | 0.3697 | 0.0094 | [0.3542, 0.3852] | unresolved_MC_precision |
| position_gating_need | 0.0284 | 0.0029 | [0.0237, 0.0331] | unresolved_MC_precision |
| smooth_gate_remaining_regret | 0.0636 | 0.0047 | [0.0559, 0.0713] | unresolved_MC_precision |

## Fixture diagnostics

| Fixture | Oracle H - projected-C proxy | MC SE | Development discovery accuracy | Gate success rate |
|---|---:|---:|---:|---:|
| strong_projected_proxy_target | 0.0880 | 0.0038 | 0.642 | 1.000 |
| zero_group | 0.0000 | 0.0000 | NA | 1.000 |

The nominal strong-fixture target was 0.100 nats/neuron, but the independent
projected-proxy calibration realized 0.0811 (MC SE 0.0090); the diagnostic
replicates realized 0.0880. The proxy is not the complete fitted C family, so
neither value qualifies the production effect scale.

## Current finite-D12 reference

| Fixture | Replicates | Mean H - C | MC SE | Detection rate | Mean one-sided 95% half-width | Mean group accuracy |
|---|---:|---:|---:|---:|---:|---:|
| strong_projected_proxy_target | 32 | -0.0344 | 0.0114 | 0.000 | 0.1145 | 0.568 |
| zero_group | 32 | -0.0224 | 0.0099 | 0.000 | 0.0944 | NA |

This reference is the historical overconstrained fitter that freezes the fitted
`C` backbone into `H`; it is not the production-contract `H - C` comparison.

`MOD_GATE_TRUTH` is required to reproduce the exact oracle density
cellwise. The attribution is conditional and ordered; it is not a variance
decomposition. Sequential MC-SE target met: **False**; 
maximum monitored MC SE: **0.0108**.
Component-level interpretation is disabled if that target is false.
All reported group accuracies are pooled-cell, truth-assisted,
permutation-invariant diagnostics; they are not equal-unit evidence that a
fixed development label transports to audit units.
Before any recovery suite, EP11 still needs population-optimal
complete-family C/H calibration and a finite-D12 estimator that uses equal-unit
objectives, smooth development-only gating, marginal group likelihood, and
separate but budget-matched C and H optimization.
