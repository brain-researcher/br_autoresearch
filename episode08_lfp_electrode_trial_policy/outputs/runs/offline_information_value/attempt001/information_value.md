# Exploratory conditional electrode-value prediction

Complete eight-session development analysis. The same 40 evaluation trials per session
were exposed during the earlier 41-comparison search; this is not new confirmation.

Positive MSE reduction means that adding the conditional cross-validated gain improves
prediction of subsequent fixed-decoder removal loss beyond the fixed quality/marginal
association baseline. All quantities are model-specific predictive values.

| Fitting-prefix trials | Transfer | Chewie MSE reduction | Mihili MSE reduction | Equal-animal mean |
| --- | --- | ---: | ---: | ---: |
| 32 | leave_one_session_out | -5.3116184e-09 | -1.9957054e-09 | -3.6536619e-09 |
| 32 | opposite_animal | -4.0100248e-09 | 5.9855544e-09 | 9.877648e-10 |
| 64 | leave_one_session_out | 1.7701567e-08 | 1.9948101e-08 | 1.8824834e-08 |
| 64 | opposite_animal | 2.0359297e-08 | 8.3103322e-08 | 5.173131e-08 |

The 32-trial leave-one-session-out comparison is primary. The 64-trial information state
and opposite-animal transfers are prespecified sensitivities, not opportunities to select
the most favorable result. Session omissions use fixed predictions and are not confidence intervals.
Electrodes, contexts and sessions within two animals do not establish new-animal replication.
Signal-quality covariates are recorded-feature proxies; artifact fraction, line noise,
physical sensor noise and spike contamination are not separately quantified here.
Four reserved sessions remain closed. No acquisition-efficiency or causal-importance claim follows.

Full electrode/context values, predictions, per-session errors and unavailable rank correlations
are retained in `information_value.json`. Historical policy-search results remain unchanged.
