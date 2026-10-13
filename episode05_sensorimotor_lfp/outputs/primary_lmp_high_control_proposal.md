# Current-bin LMP plus high-frequency power and matched trial-label control

Prepared 2026-10-03 while the fixed-history panel is running, without opening
its partial scientific results or new source values. **Root adopted the full
recipe and both screens at 23:48 UTC before any new joint fits. New core,
runner and nine focused fixtures are prepared. Root and independent core/runner
reviews pass; all nine fixtures passed in 0.394 seconds in allocation 46438625.
ASTRA validation passed and the full panel completed 2026-10-04 00:03:52 UTC.
J passes identity but fails the primary margin; trial alignment passes while
both LMP-complementarity screens fail. No promotion.** This pair asks whether a fixed high-frequency
block improves the existing full-channel LMP decoder, and whether that gain
requires the high block from the correctly matched trial. Previous failures,
the frozen endpoint, margins and remaining search requirements are unchanged.

## Fixed support and motivation

Use the same sixteen exposed development sessions, original complete trials,
eight held-direction folds and native XY target at offsets **0:+14 inclusive**
(fifteen 30 ms bins). Retain the equal-direction first-harmonic direction/time
template, common fitting-residual intercept, joint native XY SSE denominator,
exact direction-wise evaluation wrong-pair mean, equal-session then
equal-animal aggregation and the two-animal grouping.

Completed multichannel banks already document identical physical-electrode
support for LMP and both high bands: 86 electrodes for Chewie_20150313 and
Chewie_20150319, and 96 for each other session. The exact ordered identities
are each preserved bank's `source.common_LMP_electrodes`, not a newly chosen
intersection. Each selected original trial was qualified for one exact guide
column per electrode and band before the old reader loaded any values.
Metadata evidence is in
`runs/adaptive_search/primary_multichannel_attempt001/<session>/score_banks.json`
(`status`, `session_id` and `source` fields only). The guide contract is in
`tools/multichannel_reader.py` and `tools/primary_multichannel.py:guide_columns`.
No channel or trial may be dropped if replay fails.

The earlier high-only decoder's small increment does not resolve whether high
power adds information conditional on LMP. This is a feature-combination
comparison with an explicit alternative-explanation control, not another
penalty grid. It does not supply a new nonlinear model family, and must not be
counted as one for outstanding family/challenger coverage.

## J: one fixed joint feature block

Recipe ID: `lmp_high_joint_ridge_v1`.

At each evaluated time, concatenate the E current-bin LMP coordinates followed
by the 2E high-power coordinates. LMP uses physical electrodes in their frozen
ascending order. Within the high block, use electrode first, then bands
`[(100,200),(200,400)]` Hz. There are exactly 3E columns. Use stored powers as
qualified; no log, square, new filter, lag, temporal average, band weighting,
PCA, whitening or outcome-selected electrode set is introduced.

For each held direction, estimate a separate population mean and SD for every
physical electrode/band coordinate from all fifteen bins of only the seven
fitting directions. Exactly zero SD becomes one. Apply those same parameters
to fitting and evaluation values, then subtract the fitting-column means and
apply the same means to evaluation columns. Do not rescale blocks or divide
the high block by sqrt(2). This retains the existing LMP coordinate metric and
gives each standardized high coordinate the same ridge penalty as each LMP
coordinate. The high block has twice as many columns, hence typically twice
the summed standardized feature energy of LMP; this explicit capacity and
regularization difference is addressed by the same-width control below, not
concealed as equal block capacity.

Fit the existing normalized objective
`mean_row ||centered_residual - Z beta||^2 + lambda ||beta||^2`
with `[0.001,0.01,0.1,1,10]`. Each outer session selects J's lambda using only
the other fifteen J score banks, with equal-session/equal-animal means and
larger-lambda tie breaking. No per-animal, per-session or seed-specific recipe
choice is allowed. Zero high coefficients embed the stationary LMP anchor at
the same lambda; finite-sample held-out improvement is not guaranteed.

## D: eight within-direction high-block trial derangements

Recipe ID: `lmp_high_trial_derangement8_v1`.

Fix replicate IDs 0 through 7. For each session, direction and replicate,
sort the original trial-ID strings lexicographically into a recipient list.
Initialize Python `random.Random()` with `seed(literal, version=2)` using

`ep05-lmp-high-trial-label-v1|20261003|<session>|<direction>|<replicate>`.

Direction and replicate use unpadded decimal integers. Repeatedly shuffle a
fresh copy of that canonical list until every recipient receives a different
donor. Accept the first derangement, with a hard maximum of 10,000 draws;
failure stops this comparison without reseeding. All frozen directions have
at least two trials. Record every recipient/donor map, seed literal and draw
count. Maps depend only on metadata, and remain fixed across folds and
lambdas. Seeds are not biological replications or a source of model selection.

First standardize the physical coordinates using J's fitting parameters.
Then substitute the donor's **entire high block** for each recipient: both
bands, all physical electrodes and all fifteen bins move together. Apply the
same direction-specific map when that direction supplies fitting trials and
when it is the held evaluation direction. Since whole directions are held
out, a donor is always in the same fitting/evaluation partition as its
recipient; never move a high trajectory across that boundary. Evaluation
targets are not used to fit any transform, map or coefficient. This is an
offline evaluation control, not a deployable trial-matching algorithm.

Recipient LMP, XY targets, direction labels and behavior baseline remain
unchanged. For each replicate, center the shuffled joint columns using only
their fitting means; do not refit coordinate SDs or rescale blocks. Whole-trial
permutation preserves each high coordinate's fitting mean/SD and the high
block's Gram matrix, while retaining within-high electrode/band/time
relationships. LMP's block Gram is also unchanged. The **cross-block Gram
generally changes**, so each replicate requires its own joint Gram/solve.
The fitted LMP coefficients can change in this joint refit even though its
input columns remain fixed.

For outer session s, apply J's lambda selected from the other fifteen J banks
to J and all eight D refits. D never selects a penalty. Retain each replicate's
correct SSE and exact wrong-pair SSE, and average those SSEs before computing
the control's increments with the common denominator. Do not average
predictions into an ensemble, select favorable seeds or treat the eight seeds
as eight comparisons. The usual evaluation wrong-pair statistic is computed
from each model's full joint prediction; it is distinct from this input-block
control. D can retain substantial identity information through recipient LMP.

## Contrasts, proposed screens and claim limits

Report J and mean D by session and animal, selected lambdas, seed-level scores
and J-minus-D primary and identity contrasts. The proposed directional
trial-alignment screen follows the previous pair: both animal means of both
contrasts must be strictly positive, and the primary contrast must be positive
in at least 4/6 Mihili and 6/10 Chewie sessions. Ties fail. This is a directional
screen, not a significance test or permutation-tail p-value.

Also report J against the preserved independently selected LMP anchor, and
against that anchor's retained five-lambda score bank at **J's selected lambda**
for each outer session. The latter keeps the LMP penalty identical when asking
what the added high block contributes; the former states whether J improves
on the already selected usable anchor. These are reused reference contrasts,
not additional fits or charged comparisons. Their outer lambda choices must
remain exactly as declared; no contrast chooses another lambda.

A claim of added trial-specific predictive value requires positive J-minus-LMP
primary and identity means in both animals under both reference selections,
with strict primary session majorities, as well as the J-minus-D screen.
Report all contrasts even when that directional complementarity screen fails.
There is no new practical margin for these contrasts. Promotion still requires
J to satisfy the unchanged **0.01 primary / 0.005 identity** margins, animal
signs and majority gates, plus every other required control/search condition.
A passing alignment control cannot rescue a failed joint primary result.

This derangement destroys both high-to-target trial pairing and LMP-to-high
trial covariance, while preserving within-direction/time distributions. It
therefore tests whether a correctly aligned extra high block helps this joint
decoder. It does **not** isolate cross-block coupling from high's own
trial-specific information. The additive linear model has no explicit
multiplicative cross-block interaction. Avoid claims of causal synergy,
absence of neural information after a null result, or uniquely field-potential
mechanisms: spike contamination remains an outstanding high-frequency
alternative explanation. These exposed sessions do not support confirmation
or new-animal generalization.

## Prospective implementation and bounded execution

Preserve the existing science files. Reuse the unchanged qualified
`multichannel_reader.load_session` and its exact guide/target checks. That
reader loads its already qualified LMP, three low bands and two high bands;
immediately discard the low block without fitting or scoring it. Record this
actual read scope explicitly rather than claiming only LMP/high values were
decoded. This proposal introduces no new band or window and no spike reads.

Before reading old prediction arrays for replay, complete all new session
fits. Then use the unchanged post-fit reference checks for exact original
trial IDs, targets, directions, baseline, physical channels and 0:+14 window.
Read the preserved five-lambda LMP score rows for the two reference contrasts;
never overwrite its banks. Require all sixteen sessions and all eight control
replicates before releasing a scientific summary.

If adopted, use one complete Chewie_20150313 pilot with both arms/all seeds,
provisionally 1 CPU, 4 GiB and five minutes (270-second internal/300-second
external bounds, numerical threads one). Root must select the actual allocation
and confirm adequate remaining time. Choose the remaining-session bound from
measured pilot cost without opening pilot scientific scores. No source reads,
code execution, tests, pilot or submission are performed by workers; root owns
the selected compute handoff following implementation review.

Required focused fixtures cover matched physical electrode/band ordering;
fitting-only scaling and the nested same-lambda LMP prediction; deterministic
whole-block derangements independent of input row order; no fixed points or
partition crossings; unchanged recipient LMP/XY; fixed high Gram with changing
cross Gram and a separate joint solve; mean-SSE rather than ensemble scoring;
and candidate-selected outer lambda shared by control and matched anchor.

Two complete sixteen-session recipes add two comparisons, including one
alternative-explanation control. If the currently running history pair
completes, this pair would advance ten to twelve valid comparisons with three
controls among six post-initial comparisons (50%). Running outcomes are not
assumed complete here. All 28–64, patience, coverage and promotion requirements
remain unchanged; seeds, penalties, contrasts and retries do not increase depth.
