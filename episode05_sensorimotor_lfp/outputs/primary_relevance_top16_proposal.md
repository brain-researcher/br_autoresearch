# Training-only held-direction electrode relevance and a matched rank-label control

Prepared 2026-10-04 from the thirteen released primary comparisons while the
PCA90 panel was closed. Root subsequently released the complete PCA result;
the added interpretation below changes no drafted ranking, k, control or gate.
**Root adopted this exact recipe at00:39UTC after root and independent reviews,
before implementation or new fits.** New code and focused checks are authorized;
source execution remains coordinator-owned within the separately selected
bounded pilot. No existing result, contract or practical margin changes.

## Question and authority

Does retaining electrodes with individually reproducible residual-velocity
prediction improve on fitting every LMP coordinate? The stationary full-LMP
anchor has small primary gain, about0.00194, despite positive trial-identity
and temporal-alignment contrasts. Fixed history and LMP-plus-high-band models
have primary gains around0.00204 and0.00225, still below0.01, and their small
advantages over stationary LMP do not generalize consistently across both
animals. More coordinates or temporal flexibility have not supplied the
missing practical gain. These observations motivate testing whether weak or
inconsistent electrode mappings dilute the available predictive signal.

This is **supervised predictive-relevance selection**, not an estimate of
intrinsic recording reliability, neuronal quality or artifact contamination.
It uses native XY only inside allowed fitting directions. The operative GOAL
permits deterministic training-only top-k selection, and SEARCH_POLICY names
a short prespecified k list; propose the singleton list **[16]**. Those rules
do not restrict channel ranking to input-only statistics. The exact target-
supervised ranking rule below still requires root adoption before fitting.
It is distinct from the input-covariance PCA rule; no PCA score,
rank or diagnostic was consulted in choosing it. No direction-rotated target
or new direction-by-neural transform is introduced.

The subsequently released complete PCA panel strengthens this distinction.
PCA primary gain was0.001378 and its identity gain0.002837, failing both
practical margins. It beat its permuted-loading control by0.000454 primary,
but lost to full LMP by0.000559 primary and0.000559 identity, with losses in
both animals under independently selected and matched penalties; all fixed
omission contrasts against full LMP were negative. The128 retained widths
were24–44, and control projected traces were only25.02–57.31% of the paired
trace despite equal nominal rank. Therefore the loading-label result supports
aligned retained variance, not a pure spatial-identity mechanism or useful
PCA denoising. Training predictive relevance tests an alternative to selecting
large input variance. These results do not guarantee that marginally useful
individual electrodes will retain the jointly useful information PCA lost.

## Fixed scope and candidate R

Candidate ID: `lmp_inner_lodo_top16_ridge_v1`.

Keep all sixteen original sessions, their original complete trials and common
physical LMP channels; current LMP and native XY offsets0:+14; outer held-
direction folds; cyclic direction/time baseline and shared residual intercept;
the five final ridge penalties; native joint XY SSE, exact within-direction
wrong-trial identity score, and equal-session/equal-animal aggregation.
Use the unchanged LMP reader and post-fit baseline/target/trial replay. No
extra source interval, power band, source cohort or target is needed.

For outer held direction h, let A contain the other seven directions. Build
an electrode ranking using only A as follows, separately for every h:

1. For each inner held direction d in A, fit on A excluding d, containing six
   complete directions. Refit the original equal-direction first-harmonic
   direction/time baseline on these six directions only. Obtain its pooled
   fitting residual intercept, also from these fitting trials only. The
   seventh direction's native XY must not affect either fitted quantity.
2. Compute each physical channel's original population mean/SD across the
   six-direction fitting trials and fifteen bins. Replace a zero SD with one;
   apply those parameters unchanged to the inner held trials. Neither outer
   held-direction features nor inner held-direction features determine these
   statistics. Do not residualize or otherwise transform neural direction
   means. Retain the original channel coordinate.
3. Fit one scalar-input/two-native-output ridge correction for every electrode,
   with **fixed normalized lambda1** and an unpenalized shared residual
   intercept already handled by the baseline. For standardized fitting
   coordinate z_e and centered native residual matrix U, use
   `b_e = z_e' U / (z_e' z_e + n_fit * 1)`.
   Apply `B_d + z_test,e b_e` to all complete inner held trials. There is no
   channel-specific, direction-specific or inner-validation penalty search.
   The fixed ranking penalty matches the established stationary anchor's
   scale and is frozen before opening PCA results.
4. Sum `SSE(B_d,Y_d) - SSE(B_d + z_test,e b_e,Y_d)` over all seven inner held
   directions. Divide by their common total number of trial-time rows
   `15 * N_A`; the numerator sums both native XY coordinates. This is a signed
   native mean squared-error gain, with the same row weighting and denominator
   for every electrode. It is a ranking statistic, not an additional primary
   endpoint or independent replicate. Keep negative and zero values.
5. Order channels by decreasing signed gain, breaking exact ties by ascending
   physical channel ID. Select exactly `K=min(16,E)` channels. Do not select
   only positive channels, alter K, replace tied/negative channels, or choose
   different k values by direction, session or animal. Record all signed
   electrode gains, per-inner-direction contributions and the strict order.

After freezing the ranking for h, fit the final outer model on all seven
directions A. Recompute the original physical-channel means/SDs using only A,
before retention. Retain R's K columns, center using fitting rows only and
do not rescale after retention. Fit the same native residual ridge model at
all five original penalties. The outer h features and native XY are used only
for evaluation after every transformation and ranking are frozen.

The outer session penalty is selected from the other fifteen R session banks
using the existing equal-animal selector and larger-lambda tie rule. This
selects one fixed recipe across sessions, not per-session winning methods.

## C: matched retained width with eight frozen physical-rank-label maps

Control ID: `lmp_inner_lodo_top16_rank_labels8_v1`.

For each session and replicate0..7, construct an ordinary physical-channel
permutation p by seeding Python `random.Random` with version2 and literal

`ep05-lmp-inner-lodo-top16-rank-label-v1|20261004|<session>|<replicate>`.

Shuffle a fresh ascending zero-based physical-position list exactly once.
Accept identity, fixed points and chance overlaps. Reuse the same map in all
eight outer held-direction fits, all trials/times and all five penalties;
record destination/source positions and physical IDs.

Within each outer fit, first compute R's strict training-only rank vector
`rank[e]` after its deterministic tie break. Assign
`rank_control[e] = rank[p[e]]`; retain its K lowest ranks. Equivalently, this
randomly relabels the learned electrode order. The use of strict ranks means
that even tied raw scores have a fully specified map distribution. Conditional
on a fixed ranking this gives a uniform K-electrode subset for each seed,
with chance overlap with R permitted. The maps are shared across outer folds,
but selected sets may change because the training-only rankings change.

Apply the original outer-fitting physical means/SDs **before** changing retained
labels. Then use the selected physical columns in both fitting and evaluation,
with original XY, trial IDs and time bins unchanged. Fit each control's own
K-column Gram and ridge solution; a different retained set generally changes
covariance. Do not permute all columns while retaining all of them, refit the
ranking after its label map, move target values, or reuse R's Gram.

Use R's outer-session-selected lambda for all eight controls. Average correct
and wrong-trial SSEs across the eight fitted models with the original shared
denominator; never average predictions or choose a favorable seed. The eight-
seed mean constitutes **one** control comparison.

This asks whether the relevance-selected set is more useful than same-width
random retention under the same decoder and tuning opportunity. It does not
remove neural information. Equal K does not equalize covariance, redundancy,
effective ridge degrees of freedom or useful signal energy. Physical training
normalization still conveys channel information. Both R and C condition on the
same fitted rankings and spend the same ranking computation; label assignment
is the controlled change. The full-channel anchor changes dimension and is
a separate practical comparison, not the capacity-matched falsifier.

## Interpretation, finite diagnostics and stopping

Report R, C, R-minus-C, and R minus the preserved full-LMP anchor under both
its independently selected lambda and R's matched lambda. Reuse the old
five-lambda reference banks without a new fit or scientific-trial charge.
The proposed R-minus-C screen requires strictly positive primary and identity
contrasts in both animals and primary-positive majorities of at least4/6
Mihili and6/10 Chewie. Ties fail. A claim that relevance selection improves
on all channels additionally requires the same directional screen against
both anchor selections. These are directional descriptions, not significance
tests or replacements for the frozen **0.01 primary / 0.005 identity** gates.

Retain raw ranking-score ranges, number positive, cutoff ties and selected
physical IDs per outer fit; control overlaps with R; and projected fitting
covariance trace, numerical rank and ridge effective df at the selected
lambda. Use the already declared PCA diagnostic tolerance
`64 * eps_float64 * K * max(1,max(abs(eigenvalues)))`, counting eigenvalues
strictly above it. These describe rank/energy differences only; no diagnostic
changes membership, K, penalty or scoring. Zero-energy or deficient controls
remain valid positive-lambda baseline-only fits. Tied/no-skill rankings cannot
support a learned-relevance interpretation even if the deterministic physical-
ID tie-break subset happens to score well; report that limitation without
inventing a new ranking or omitting the fold.

Marginal single-electrode usefulness is not conditional complementarity.
The rule can retain redundant channels and discard individually weak channels
that are jointly useful. A positive result could reflect ordinary variance
reduction from sparse retention rather than a biological reliability property.
A failure would reject this particular ranking/width, not all sparse models
or all LFP trial information. Generalization of the ranking is assessed with
every outer held direction; an inner score is never treated as independent
confirmation. Movement phase or unmodeled behavior may still contribute, so
the stationary anchor's timing-control success does not automatically certify
R or satisfy every remaining finalist control.

If E is at most16 in a session, report that session's full-retention identity
without dropping it. If every eligible session is full-retention, the pair
is structurally duplicate full-channel ridge and contributes zero new valid
comparisons. Otherwise two complete fixed panels add exactly two comparisons,
one candidate and one control, to the ledger at execution time. After the
released PCA panel there are15 valid comparisons; this pair would reach17,
with6 controls among11 post-initial comparisons. Inner folds, single-channel fits,
retained electrodes, seeds, contrasts and infrastructure do not count as
extra trials. The28–64 budget, patience and control fraction remain unchanged;
there is no early favorable-score stop or minimal-count padding.

## Proposed implementation and resource boundary

If root adopts this exact specification, preserve existing science and write
new core/runner/fixtures. Focused fixtures should establish nested exclusion
before inner baseline/scaling/ranking, the signed alpha1 scalar formula and
common row weighting, deterministic tie handling, exact frozen rank-label
maps, same-width control invariants with separate Grams, untouched outer
targets and baselines, mean-SSE aggregation, shared outer lambda, and structural
full-retention identity. Reuse qualified original readers and scoring checks.

There are56 inner six-direction ranking fits per session, each vectorizing E
independent scalar-input/two-output regressions, plus the final8 directions
times9 arms times5 penalties with at most16 columns. This is not an E-channel
matrix solve repeated E times or an extra lambda grid. Source exposure and
prediction-array sizes match the current-bin control families. Runtime is
unmeasured; propose the complete predetermined Chewie_20150313 pilot on1CPU
within4GiB,270seconds internal/300seconds external after independent review
and focused checks. Root chooses the actual allocation and the remaining
fifteen-session bound from completed pilot resource facts. Preserve failures;
no automatic retry, session/seed omission, or outcome-based resource choice.

Root's read-only BR query `local field potential motor decoding reliability`
at00:37UTC succeeded with four Publication titles about MRI reliability,
human optic flow, cued-language LFP and orientation fMRI. It supplied no
directly relevant motor-LFP method evidence, reliability formula or endorsement.
No method assertion is inferred from those titles and no retry is needed.
The proposal follows the existing grammar and released comparisons; the
advisory retrieval is neither a launch gate nor scientific validation.
