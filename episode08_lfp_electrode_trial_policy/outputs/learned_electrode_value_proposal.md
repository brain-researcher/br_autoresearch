# Learn which initial electrode signals generalize to later calibration trials

Prospective development proposal, 2026-10-03. **Adopted by root at approximately
00:04 UTC on 2026-10-04, before new labels/fits; implementation prepared,
unrun.** Prepared from the operative open-exploration contracts and completed
comparison records. The running mapping-grouping control was not opened or
used to select this design. No source access, computation, submission, held
opening or optional named explanatory endpoint is activated by this document.

## Operative authority and transfer boundary

The approved [open-exploration amendment](open_exploration_design.md) expressly
includes "static geometry/quality and development-trained electrode rules"
and requires offline labels for transferable rules to be session-cross-fitted.
`SEARCH_POLICY.yaml:open_exploration` keeps source-trained selection open;
its shared required coverage includes simple linear learned ranking. The
proposal instantiates those delegated choices without changing any canonical
contract or the optional reference's one-shot retention grammar.

This is leave-one-session-out development within **two already exposed
animals**. Other sessions of the target animal remain donors. It is not
leave-one-animal-out transfer, new-animal evidence, chronological forecasting
or a causal prospective acquisition experiment. Reverse-animal transfer remains
a separate missing check. Target s is excluded from every ranker training
label, feature statistic, fitting record and any hyperparameter choice; the
three predictors and fixed penalty are chosen here, not from s's outcomes.

## Why this is the next distinct question

Initial-only residual marginal selection improved the acquired-only comparison,
but required an explicitly paid 32-trial all-electrode initializer. Conditional
fitting gain did not beat marginal selection in the earlier residual-selector
comparison. Neither method has learned from other sessions whether an early
apparent electrode signal survives later trials. The failed mapping proxy and
fragile target-mean allocator do not answer that electrode-value question.

Test one simple **session-cross-fitted learned electrode ranking**, with a
matched permuted-value-label control. This fills the previously untested
linear learned-selector/transfer family. It is not another decoder upgrade,
penalty grid, variation of the mapping design matrix, or claim that the
minimum search count has been met. It does not fill shallow/nonlinear learned
ranking, all allocator families, quality-matched swaps or coupling coverage.

The learned labels below concern future **single-electrode predictive gain**.
They are neither conditional removal values nor a newly activated named
explanation outcome. A useful ranking may motivate a later joint policy; this
balanced-trial comparison cannot establish sensor-dependent trial ordering.

## Same one-shot acquisition grammar and evaluation

Keep the exact eight development sessions, fixed 96 fitting trials per session
(12 per direction), original queues, common 40 evaluation trials, current
three-band ten-bin-mean LFP features and the separate ten-bin time mean for
every original native population unit. Never average across neurons.
Four held sessions remain closed.

For the session being scored, acquire the first four fitting trials per
direction on all 96 electrodes, totaling 32 trials. Select one nested physical
electrode ordering once, with retained prefixes 8/16/32 and the common all-96
branch. Subsequently acquire only those retained electrodes along the same
balanced original queues. Evaluate all 32/48/64/80/96 checkpoints, for the same
20 cells as the completed acquired-only comparison. No later reranking,
channel revision, historical backfill or new trial action is introduced.

The decoder, initial four whole-trial folds, fixed rank/alpha grid, tie rules,
later initializer-conditioned CV, fitting transformations and evaluation Q
are exactly those of `outputs/acquired_only_proposal.md`. In particular, later
CV keeps all 32 initializer trials in every inner fitting set and partitions
only newly acquired rows. Balance is outcome-independent, so this reuses that
qualified later-CV protocol rather than the separate adaptive-history recipe.
No decoder is chosen by session, animal, learned score or control seed.

## Training labels come from other sessions' fitting pools

For every donor development session, prepare five information views: its full
32-trial initializer and each of its four exact 24-row initial inner-fitting
subsets. These have four or three trials per direction respectively. A view
never includes its eight initial validation rows when it is an inner view.

The donor's subsequent 64 fitting trials, queue positions 4 through 11 in
each direction, form a fixed later-calibration label set. They exclude all
32 initializer trials and all 40 evaluation trials. This is supervised donor
training, not a new external or untouched evaluation claim.

For each donor view and physical electrode, fit a full three-coordinate
linear direction-residual ridge with **alpha 1**, without tuning or response
rank truncation. Use the existing residual convention: direction X/Y means,
residual X population SD, raw Y population SD and residual intercept are all
fitted solely on that view; zero SD becomes one without dropping columns.
Predict the donor's later 64 target rows using only that electrode.
The exact qualified helper is `ridge_training.ridge_coefficients`, minimizing
`||Y_residual-X_residual B||_F^2 / n + alpha*||B||_F^2`; its normal equation
uses `X'X + n*alpha*I`. `direction_training.residual_statistics` and
`residual_coordinates` provide the unchanged raw-per-unit-Y scaling and
intercept convention. Alpha1 is not a sum-loss penalty1.

The signed training label is

`[SSE_scaled(direction-only) - SSE_scaled(single-electrode ridge)] / D`,

where scaling is the view's raw target SD and
`D = SSE_scaled(later64 Y - view global Y mean)`.
The direction-only prediction is the same view-fitted direction mean and
intercept used by the residual decoder. No negative label is truncated; no
label-dependent electrode exclusion is allowed. A nonfinite or zero
denominator makes that required donor view unresolved and stops the complete
comparison. No epsilon, label clipping or session omission is a fallback.

This small full-linear donor label is a declared prediction target for the
ranker, not the final reduced-rank decoder's realized multielectrode value.
It need not be additive across selected electrodes. No correspondence among
neurons or electrode numbers across animals is assumed.

## Three fixed initializer predictors and one fixed linear ranker

Compute exactly these predictors from each permitted donor or target view:

1. The existing residual marginal target-association gain with selector
   penalty 1, residual-X scaling and raw-Y scaling, unchanged from the
   residual-selector recipe.
2. Mean direction repeatability over the electrode's three raw LFP features,
   using the existing balanced-group between/within variance estimator and
   its declared zero/zero convention. This is repeatability, not a measured
   artifact, missingness or line-noise quality score.
3. Residual-X redundancy with the other 95 electrodes: for standardized,
   direction-residual blocks `X_e` and `X_f`, average
   `||X_e' X_f / n||_F^2 / 9` over all `f != e`. All required X values were
   actually acquired in the initializer. Zero-SD coordinates use the same
   replacement-one convention; no electrode is dropped.

No feature selection, nonlinear terms, transforms, extra penalty grid,
geometry surrogate or post-outcome predictor revision is included.

For target session s and information view v, train the ranker on **only the
other seven sessions' matching view-v electrode records**. Each donor animal
gets half the total loss weight, sessions within each donor animal get equal
weight, and its 96 electrodes share that session weight equally. Thus the
three-versus-four donor-session imbalance does not give one animal more weight.
Fit predictor means/population SDs using these weighted donor records alone;
exactly zero SD becomes one.

Fit an unpenalized intercept and three coefficients minimizing

`sum_records w * (label - intercept - standardized_predictors @ beta)^2
 + ||beta||^2`,

with weights summing to one and fixed penalty **1**. There is no ranker CV or
session-specific penalty. Apply those donor-fitted transformations and weights
to the target session's corresponding permitted view. Sort predicted signed
gain descending; exact ties use ascending physical electrode ID. The score
need not be positive to retain an electrode.

At initial decoder CV fold j, target predictors use only that fold's 24
fitting rows and the donor ranker trained on matching 24-row views. Neither
target validation X nor Y may enter selection. For the outer initial fit,
use the donor full-32 ranker and all 32 acquired target rows; freeze that
ordering for subsequent balanced acquisition. Ranker training for s must not
receive any of s's donor labels or future 64 fitting rows. Source-role checks
must enforce that session exclusion before a ranker is fitted.

## Required learned-label control

Keep all predictors, the decoder recipe and true target labels unchanged;
each retained set has its own fitted coefficients. For fixed
control labels 0/1/2, permute the 96 signed training labels once within each
donor session and information view, using an ordinary uniform permutation
including fixed points. Pair labels with canonical ascending physical IDs.
Use Python `random.Random().seed(literal, version=2)` with

`ep08-learned-electrode-label-v1|20261003|<donor_filename>|<view>|<replicate>`.

View is the literal `outer` or the unpadded initial-fold integer 0/1/2/3;
replicate is 0/1/2. Shuffle one fresh canonical label-index list once, without
rejection, reseeding or dependence on the target session being left out. Only
the permitted seven donor sessions enter each target's control ranker.
Retain every donor permutation and predicted/retained target ordering.

Each control seed has its own frozen target set and decoder history. Average
its three correct/wrong SSE scores within each matched resource cell before
session/animal aggregation; never average predictions or select a favorable
seed. Their scoring denominators are identical because target trial queues
are balanced and unchanged. This is one complete control comparison, not three.

The control asks whether learned donor value labels matter beyond the fixed
features, width and fitting procedure. It is not a significance test, and
permuted labels do not imply that the unchanged final decoder lacks signal.

## References, costs and report

The principal reference is the completed initial-only marginal selector in
`runs/development_search/acquired_only_attempt001/acquired_only.json`.
It has the same current-session all-channel initializer, frozen sets,
balanced trials, decoder/CV and scoring normalization. Read its outcome packet
only after all new fits, and authenticate source roles, physical electrode
identities, folds, checkpoints and target support before pairing scores.
Require the common all-96 models/scores to replay within the existing 1e-10
Q tolerance. Existing reports contain scores; do not claim old prediction-array
identity if those arrays are unavailable. Reuse identical all-96 computation
across new methods/seeds, with exact source-array identity before reuse.

This is **equal target-session acquisition cost**, not equal total training
information or total cost. The learned selector additionally receives
supervised training from seven donor sessions, including their later
64×all-96-electrode calibration exposure; the within-session marginal
selector does not receive that information. Charge every donor read, fit and
stored training/method record separately. A win can justify that explicitly
larger training investment, but cannot be attributed to a better formula
under identical total information. Report total storage and retained artifact
bytes alongside source logical exposure and CPU/wall time.

Fixed random selection from that report is useful context but has a cheaper
initializer. Do not label the learned-versus-random Q difference an
equal-acquisition-cost gain. For retained E and acquired n, both learned and
initial marginal policies consume

`32*96 + (n-32)*E` LFP electrode-trials,

plus `n*U*10` target values. Report separate electrode/trial axes, all-channel
initial exposure, donor-training cost, actual repeated reads and unique logical
source elements. Donor label creation needs all 96 fitting trials on all 96
electrodes in each donor session; it is not free deployment information. The
retrospective evaluator will have encountered each session's future rows when
it serves as a donor for other sessions. Trusted role-separated numerical
inputs, not claims of OS isolation or newly blinded data, enforce the target
ranker's information boundary.

Keep the existing acquired-only 20-cell Q normalization and equal-cell,
session-within-animal, then equal-animal aggregation. Report learned minus
marginal, learned minus random, learned minus label control, all seed scores,
every session/cell, both animals, selected coefficients/scales and all
retained sets. Use the existing 0.001 search-progress resolution on this
matching support; introduce no biological margin, p-value, new stopping rule
or named explanation decision. Two complete panels charge **two** shared
valid evaluations; neither donor fits, 40 rankers, seeds nor resource cells
inflate this count. Required family coverage and the 40–112 allowance remain
incomplete and unchanged.

There are 8×5×96 = **3,840** small three-coordinate donor fits. One target
candidate and three control seeds require 4×3×5 scarce-cell models plus five
shared all-96 models per session: **520** actual target decoder models over
eight sessions, with the unchanged nested rank/alpha work. This is distinct
from the 760 models in the completed acquired-only run, which took 133.452525s;
donor-label and ranker cost is additional and unmeasured.

Propose a **complete predetermined target-session pilot** for
`Mihili_CO_20140217.mat`, the first session in the frozen development-role list,
with all seven donors, all four initial CV views, every target cell and all
three control seeds. Root selected one CPU within the attended4GiB parent,
with 270s internal and 300s external, requiring at least five minutes remaining
after review and focused checks. Fit
all 65 target models before reading its evaluation rows and preserve the
complete compact score/method record. Release only identity, completion,
timing, memory and exposure fields for resource selection, withholding pilot
scientific scores until the full eight-session comparison. No source arrays
or serialized target decoder coefficients are needed for reuse.

The remaining seven sessions reuse the completed pilot's score record only
when the full comparison is assembled. Keep all scientific choices unchanged;
the pilot is part of the eventual two complete panels, not a separate valid
comparison. For a simple implementation without a new cache or qualification
chain, the remaining run may recompute the eight sessions' compact donor
training records rather than serialize fitted source objects: charge the
pilot's 7×5×96 = 3,360 donor fits plus the full pass's 3,840, totaling **7,200**
donor fits, and every repeated read. The 520 target decoder fits already
include the pilot and are not repeated. Old acquired-only outcome references
and pilot scientific-score reuse are reporting inputs only after the other
seven sessions' new fits have completed; all-96 reference replay is then
required before releasing any complete comparison. Choose the remaining finite
bound from measured pilot cost. No automatic retry, requeue, skipped seed or
session is authorized.

Focused checks should establish target-session exclusion from donor labels,
matching 24/32-row fold information, direction-residual label weighting,
donor-only ranker scaling and animal weights, ordinary label permutations,
frozen target electrode sets and chosen-channel source boundaries, and
mean-SSE control aggregation. No generic preflight or new source copy is needed.

Any later claim about jointly useful electrodes and trials still needs an
independently specified sensor-dependent trial rule and its discriminating
comparisons. Success here would support transfer of a marginal electrode-value
ranking under this paid information state; failure would challenge this fixed
ranker, feature set and donor-label definition, not the existence of neural
complementarity or learnable acquisition value in general.
