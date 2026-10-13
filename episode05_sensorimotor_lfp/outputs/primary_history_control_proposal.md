# Fixed four-bin LMP history and trial-specific channel-label control

Prepared 2026-10-03 after the complete eight-comparison temporal/control panel.
**Root adopted this exact pair and directional control screen before any
new source values or fits. Implemented; root static review, all nine focused
fixtures, independent review and ASTRA validation passed. The complete panel
finished 23:46:21 UTC; primary +0.002039 fails 0.01 while identity +0.005499
passes 0.005. The label-control screen passes; no promotion.**
The prior models, results and failed margins remain intact. No GOAL, DATASETS,
SEARCH_POLICY, source role, animal grouping or endpoint is amended.

## Hypothesis and unchanged endpoint

The quadratic time-varying map improved its trial-identity readout but reduced
primary performance in both animals relative to stationary full-channel
ridge. Correct training pairing helped that map, without rescuing the failed
primary result. Test a different temporal representation: a fixed short
history of each physical electrode, under a stationary coefficient map.
Pair it with a control that removes stable electrode labels across trials
while retaining their complete multichannel temporal content.

Use all sixteen frozen sessions, original complete trials, eight held-direction
folds, stable physical LMP electrodes and the existing outer-session selector.
The target remains native XY velocity at **offsets 0:+14 inclusive**, fifteen
30 ms samples. Keep the equal-direction first-harmonic template, shared
training residual intercept, native centered residual loss, pooled joint XY
SSE denominator, exact evaluation wrong-pair mean and equal-session then
equal-animal aggregation. Chewie implants remain one animal.

The feature input is stored LMP at **offsets -3:+14 inclusive**, eighteen bins,
entirely inside the already structurally qualified -4:+14 source window.
Only LMP from the frozen physical electrodes and XY at 0:+14 are read.
No new source qualification by outcome, trial deletion, high-frequency value,
spike field, raw-voltage filtering, resampling or future-session access occurs.
Unexpected nonfinite or missing history stops the comparison; do not shorten
the history, drop a trial or change the evaluated endpoint.

## H: one fixed history representation

Recipe ID: `lmp_history4_ridge_v1`.

For evaluated target time t in {0,...,14}, concatenate physical LMP vectors
at offsets `[t, t-1, t-2, t-3]`, in lag order `[0,-1,-2,-3]`, then original
physical-electrode order. There are exactly 4E columns. The history spans
0–90 ms before each evaluated time. No lag, lag count, temporal average or
window is selected from scores. No per-animal or per-session winning history
is permitted. Offline source preprocessing means these past-index inputs do
not establish an online or causal prediction claim.

Within each held-direction split, calculate a separate population mean and
SD for each **lag × physical electrode** using all fifteen bins of only the
seven fitting directions. Exactly zero SD becomes one, retaining the column
identity. Standardize training and evaluation history with those fitting
parameters. Concatenate without whitening, PCA, block scaling or division
by sqrt(4). Subtract the resulting fitting-column means and apply those same
means to evaluation columns. This last centering is numerically nearly zero
for H but makes the shared-intercept convention explicit.

Fit the same normalized objective with lambda in `[0.001,0.01,0.1,1,10]`,
ties preferring the larger lambda. Each outer session's lambda uses only
the other fifteen H score banks, as in previous comparisons. Setting the
three past-lag coefficient blocks to zero embeds the current-bin stationary
anchor at the same lambda. More columns do not guarantee a held-out gain;
retain the existing stationary reference without refitting or charging it.

## L: eight per-trial physical-channel-label randomizations

Recipe ID: `lmp_history4_channel_labels8_v1`.

Replicate IDs are fixed at 0 through 7. For each original trial and replicate,
start from the physical-electrode position list `[0,...,E-1]` in its frozen
guide order. Initialize Python `random.Random()` with `seed(literal, version=2)`:

`ep05-lmp-history-channel-label-v1|20261003|<session>|<original_trial_id>|<replicate>`.

Use the exact original trial-ID string and unpadded decimal replicate. Apply
one ordinary uniform `shuffle`; accept chance unchanged permutations. There
is no rejection, reseed, favorable-seed choice or permutation based on neural
values. A trial has the same map in every held-direction split and lambda.
All maps are recorded as zero-based source-electrode positions alongside the
original trial IDs and frozen physical-channel list.

First use H's fitting-derived **physical-channel** lag means/SDs to standardize
the actual source history. Then apply that trial's single channel permutation
identically at every time and every lag, in **both fitting and evaluation
features**. Thus each source electrode's trajectory and cross-lag relationship
move together; within-trial values are not temporally scrambled. Training XY,
evaluation XY, actual trial pairing and behavior baselines remain unchanged.

For each replicate, subtract its own shuffled fitting-column means and apply
those same means to its evaluation features. Do not restandardize the shuffled
columns or refit physical-channel SDs. Fit a separate ridge map for each
replicate. Unlike whole-trial row shuffling, per-trial channel permutations
generally change the global Gram matrix; each replicate needs its own
covariance/eigendecomposition. No invalid shared-Gram shortcut is allowed.

This control deliberately retains information in physical-channel
normalization and tests the usefulness of stable electrode labels **after
that normalization**. It also changes global feature covariance. Its result
does not isolate a causal channel-identity effect from every covariance or
regularization interaction. A single fixed global feature permutation would
merely relabel an isotropic ridge model and is not this control.

For outer session s, use the lambda selected by H from the other fifteen
paired H banks for H and all eight L refits. No L outcome chooses a lambda,
permutation or source channel. Keep seed-level correct and exact wrong-pair
SSEs. Average the eight **scores**, not predictions, using the identical
denominator; this remains one control comparison rather than an ensemble or
eight biological observations. Report seed spread descriptively without a
permutation-tail p-value.

## Comparison and adopted control screen

Report H and mean L for every session/animal, all selected lambdas, the reused
stationary anchor, and H-minus-L primary and identity increments. The
directional label screen passes only if both animal means of both H-minus-L
contrasts are strictly positive and the primary contrast is positive in
strict majorities of both animals (at least 4/6 and 6/10). Ties fail. This
prospective directional screen is not a significance test or a replacement
for the existing **0.01 primary / 0.005 identity** practical margins.

A failing screen blocks promotion under this screen and bounds the
interpretation of stable electrode labels for this normalized history model.
It does not establish absence of neural trial information. Passing cannot
rescue a failed primary endpoint or imply causal, online, raw-voltage,
population-mechanism or new-animal generalization. All data remain exposed
exploratory development; the rest of the required controls remain open.

## Reuse, resources and count

Preserve earlier source readers and science files. The new reader explicitly
reports separate native LMP and target windows. After all new session fits,
verify those declared windows and use the existing post-fit reference check
for original physical channels, trial counts/IDs, directions, actual target
and common baseline. Never rewrite the new input-window metadata as 0:+14
to satisfy an older helper. The old reference's target window remains 0:+14.

Root selected one complete Chewie_20150313 pilot, both arms/all eight
replicates, **1 CPU, 4 GiB, five minutes**, with 270-second internal and
300-second external bounds and numerical threads one. Use a verified
task-owned allocation only if at least five minutes remain. Choose the
remaining fifteen-session bound from measured pilot cost without opening
partial scientific scores. No aggregate bound is selected yet.

A 96-electrode history has 384 columns; about 15,570 rows occupy 48 MB in
float64. Process replicates sequentially and preserve all completed banks.
This is a design-array estimate, not measured total memory or runtime.

Two complete sixteen-session recipes add two valid comparisons, eight to
ten, including one new alternative-explanation control. The resulting
post-initial balance would be two controls in four comparisons (50%). Seeds,
lambdas, fits, the reused anchor, pilot and infrastructure retries do not
increase scientific depth. Every required session/seed must complete. All
28–64, patience, family/challenger, budget and promotion requirements persist.

Focused fixtures must verify exact source/lag indexing including target t=0
and t=14, stationary current-bin embedding, fitting-only scaling and arm
centering, whole-electrode maps shared across lags/times, chance unchanged
maps, unchanged targets, independent control Gram matrices, mean-SSE
aggregation and shared lambda's outer-row exclusion. Root owns tests,
source execution and scheduler actions; workers only edit/review code.
