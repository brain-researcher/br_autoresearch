# Adopted exploratory electrode information-value analysis — October 13, 2026

The scientist approved a distinct real-data continuation asking which electrodes
add predictive information beyond the others, and whether conditional
complementarity predicts subsequent removal loss beyond quality and marginal
target association. This implements the already permitted offline information-
value route. The original policy search remains complete at 41 comparisons with
`development_only_no_initializer_robust_policy`; no policy winner is reopened.
Next-direction allocation, sensor–trial coupling, sequential efficiency and the
four reserved sessions are outside this selected analysis.

## Evidence and exposure

Use the same eight development sessions, original 96 fitting trials and 40
evaluation trials per session, through `development_reader.py`. Preserve the
existing metadata/role reports, all 96 physical electrodes, three LFP features,
the ten-bin 150–450 ms movement-window mean and every native population target.
No new trial, channel, animal, split or feature source is admitted. Every source
neural element is read once in this run: 768 all-electrode fitting trials and
320 all-electrode evaluation trials. Record logical bytes and trial counts.

The evaluation trials were repeatedly scored during the completed search and
helped motivate this question. They remain excluded from the fitting algorithms,
but are not fresh confirmation for the analyst. Report this as exploratory
development evidence after 41 prior comparisons, including the positive average
correspondence and failed initializer robustness. Same-account procedural reader
separation is not OS isolation. Four reserved sessions remain closed.

The existing design notes locate this question in established conditional sensor
selection, not a novel information-theoretic estimator. Prior BR retrievals in
the execution log supplied no relevant M1 selector evidence. A focused BR
`pipeline_plan_review` of this branch returned no findings (no KG rules consulted);
that generic advisory result is not empirical qualification or formal acceptance.
The contribution tested here is transfer of a training-derived conditional-value
predictor beyond recorded quality proxies, not discovery of a biological mechanism.

## Fixed information states and electrode contexts

The primary predictor information state is the original balanced 32-trial prefix
(four trials/direction); 64 trials is a prespecified sensitivity. These are
existing fitting-pool prefixes, not new mandatory pilots or additional data.
Both predict removal value after the decoder receives all 96 fitting trials.
"Subsequent" refers to this information sequence, not chronological trial order:
the existing within-direction pools use seeded random orders.

Use one full 96-electrode context and three outcome-independent partitions into
six disjoint 16-electrode contexts each. For each partition, shuffle ascending
electrode positions with Python `random.Random.seed(version=2)` and literal
`ep08-information-value-context-v1|20261013|<filename>|<seed>`, seeds 0,1,2.
Every electrode appears once per partition and once in the full context:
384 electrode/context rows per session and information state. No context is
chosen by quality, response, a previous winner or removal outcome.

## Predictors known from the fitting prefix

Use four whole-trial folds: within each direction, prefix position modulo four.
Each fold retains every direction. Within each fold's fitting rows, subtract
direction-specific means, standardize LFP residual columns and scale targets by
their raw fitting SD. Exactly zero SD becomes one and no column is removed.
Predict the held fold with ordinary multioutput ridge, fixed alpha 1 under
`||Y-XB||²/n + alpha||B||²`. No alpha, rank, representation or model search occurs.

The conditional score is the mean over four folds of
`(SSE_drop_electrode - SSE_full_context) / SST_fold`. Each pair refits the same
fixed ridge procedure; `SST_fold` uses held-fold targets around the fitting
target mean in fitting-SD units. Negative gains are retained. Target directions,
means, normalizers and coefficients never use held-fold responses for fitting.

The comparison predictor includes direction-repeatability, the log native
residual RMS separately for each of the three bands (floor 1e-12), constant-band
fraction, mean squared residual marginal target correlation, matching fourfold
cross-validated marginal ridge gain, mean squared correlation with the other
context electrodes, and log context size. These are recorded-signal quality and
redundancy proxies. Artifact rate, line noise, hardware noise and calibrated SNR
are unavailable in this representation; no claim of matching all signal quality
is made. The augmented model adds only the conditional cross-validated score.

## Removal outcome and matched decoder

Before any evaluation-array read, freeze every predictor and every full/drop
decoder fitted on all 96 fitting trials. The decoder uses the same direction
residualization, raw target SD, alpha 1, target support and representation for
both members of a removal pair. The outcome is
`(SSE_drop - SSE_full) / SST_evaluation` on all original 40 evaluation trials,
using one common training-derived target mean/SD denominator per session.
Refitting after removal measures conditional predictive utility for this fixed
regularized model; it is not an intervention on brain tissue, mutual information,
or a claim of irreducible unique neural information. Population targets retain
their original overlap with LFP electrodes; high-frequency spike contamination
is not resolved by this comparison. Keep negative and zero values.

## Transfer, endpoint and complete reporting

Fit a fixed ridge value-prediction model with an unpenalized intercept to donor
sessions. Exclude the entire target session before any predictor centering,
scaling, target centering, weight calculation or coefficient fit. In training,
give animals equal weight, sessions equal weight within animal, and context
families equal weight: half to full96 and half across the three partitions.
Within each partition give all 96 electrode rows equal weight. Use these same
within-session weights for evaluation. Standardize predictors using donors only;
zero SD becomes one. Before augmentation, project the standardized conditional
score onto the intercept and standardized baseline predictor span using weighted
least squares on donors only, without removal outcomes. Add only its residual,
without restandardizing that residual. Apply the saved donor projection to the
target session. This keeps baseline ridge shrinkage unchanged and makes an
exactly duplicate extra feature unable to create apparent additional value.
The regression objective is weighted MSE plus the squared coefficient norm,
with fixed penalty 1 and no tuning.

The primary endpoint is baseline-minus-augmented removal-value prediction MSE
at 32 prefix trials, averaged first over sessions within animal and then over
the two animals. Report both models' MSE, every session, both animal means and
positive-session counts. Repeat the complete fixed comparison at 64 prefix
trials and with the other animal as the only donor, without choosing a winner
among these four reports. Within-context Spearman correlations are secondary;
constant ranks are explicitly unavailable and do not exclude primary rows.
Report fixed-prediction leave-one-session omissions as sensitivity ranges, not
confidence intervals. Electrodes, contexts and sessions within two animals do
not create independent-animal replicates; no electrode-level p-values or
confirmatory significance claim is selected.

Positive primary MSE reduction in both animal means is an exploratory directional
description, not a meaningfulness threshold or promotion criterion. Mixed or
nonpositive results weaken this particular predictor/model account. Do not
replace it with a more favorable information state, context or subset.

## Stopping, resources and executable handoff

Stop after this complete fixed eight-session analysis and report every specified
comparison. This is one additional offline evaluation against the shared
112-evaluation allowance (41 historical policy comparisons + 1), not restarted
policy-search patience. Any missing session, nonfinite required input, failed
identity/support check, invalid denominator or resource stop makes the new
analysis incomplete; preserve the failure and do not publish a selected subset.
No adaptive successor or held-session opening follows automatically.

The implementation makes 35,160 fixed small ridge fits and uses the existing
selective source reader. Prior 45,600 inner fits plus 2,280 refits completed in
289 seconds, but their dimensions differ; this is only a rough several-minute
cost basis. Select at most 1 CPU/6 GiB/35 minutes, one numerical thread, an
1800-second work limit and 5-GiB process RSS guard. Output should be below
20 MiB and no raw arrays or fitted neural coefficients are written. Preserve
all historical shared CPU/wall charges and the original 1200 CPU-hour/120
wall-hour ceilings; no allowance or historical clock is reset. Root records
actual allocation admission and once-only accounting before execution.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
timeout --signal=TERM --kill-after=10s 1860s python3.12 outputs/tools/development_information_value.py \
  --source-root "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --metadata-report outputs/runs/source_support/attempt002/metadata_support.json \
  --trial-roles outputs/runs/source_support/development_trial_roles_attempt001/development_trial_roles.json \
  --output-dir outputs/runs/offline_information_value/attempt001 --max-seconds 1800 --max-rss-gib 5
```

The existing Python 3.12 NumPy/SciPy/h5py runtime suffices. Matplotlib is optional
for the saved-number figure; missing plotting support does not invalidate the
complete numerical report. Root owns compute validation/execution. No source
scan, synthetic pilot, scheduler action or execution occurred during preparation.
