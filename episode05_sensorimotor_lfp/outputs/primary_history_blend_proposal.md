# Can a training-selected mixture of current and historical LMP improve HW?

Prepared by root, 2026-10-04. **Adopted after independent scientific review
at 08:51 UTC; isolated implementation and focused fixtures authorized.** No
new fit, prediction combination or mixture score has been computed. This is
separate from the already adopted intersection and capacity controls.

## Motivation and allowed scope

The completed original-support W and HW recipes are complementary at the
animal level: adding history improves Chewie's primary score but lowers
Mihili's under every omission. HW has the best comparable primary score,
0.0023412544731769847, still below 0.01. A training-selected convex mixture
asks whether partial use of historical predictions improves this compromise.
It is motivated by the released W/HW results, not by an unseen mixture score.

GOAL.md permits a convex blend of at most two previously evaluated
complementary pipelines, with weights learned inside training sessions/folds.
Use only the complete original W and HW five-penalty native-XY prediction
banks. Neither reduced support, changed baseline, new source feature,
spike-population outcome nor incomplete elastic-net bank is included.
All sixteen original sessions remain exploratory development; Chewie's two
implants remain one animal. No fresh validation source is implied.

## One fixed recipe and finite tuning rule

Proposed ID: `lmp_current_history_repeatability_convex_blend_v1`.
For original penalty lambda in `[0.001,0.01,0.1,1,10]` and historical weight
alpha in `[0,0.25,0.5,0.75,1]`, define the full XY prediction

`P(lambda,alpha) = (1-alpha) * P_W(lambda) + alpha * P_HW(lambda)`.

The two components use the same nominal penalty in this proposal. This keeps
one interpretable temporal-mixture choice and does not claim their ridge
capacities are equal. Do not add independent component penalties, negative
weights, extra blend components or a finer weight grid after outcomes.
For alpha=0 or 1, reuse the literal saved endpoint arrays; do not recompute
them through floating-point multiply/add. For interior alpha, use the exact
expression above, with no normalization, clipping or recentering.

Because both components add back the same original behavioral baseline B,
the blend is B plus a convex combination of their native residual predictions.
Do not add B twice. Physical normalization, repeatability priors, history
states and coefficients remain exactly those of the completed original
fits. There is no new within-session neural fit.

For each held outer session, score all 25 combinations using only the other
fifteen sessions' already held-direction predictions. Optimize the original
equal-session-within-animal, then equal-animal primary increment. Choose the
maximum tuple `(training_primary_score, lambda, -alpha)`: exact score ties
favor stronger regularization, then less history. This tie convention is
prospective and does not introduce a numerical tolerance. Apply that single
chosen lambda/alpha to every direction/trial in the held session. The held
session's score, animal-level result or identity gain cannot tune either
parameter. The same global recipe and selector apply to all sessions.

## Exact saved-support replay before combining predictions

Use these episode-local, complete originals under `runs/adaptive_search/`:

- W: `primary_repeatability_weight_control_attempt001/<session>/`, recipe
  `lmp_repeatability_weight_ridge_v1` and its complete summary under
  `primary_repeatability_weight_summary_attempt001/`.
- HW: `primary_history_repeatability_weight_control_attempt002/<session>/`,
  recipe `lmp_history4_repeatability_weight_ridge_v1` and its complete summary
  under `primary_history_repeatability_weight_summary_attempt002/`.

The original runners write `<recipe>_predictions.npz`; root confirmed the
literal W recipe and naming from the existing source before implementation.
Read only the original candidate predictions, not weight-label controls.
Require complete original session status, five-lambda order, all source-qualified
trial IDs/order, physical LMP IDs, directions, target interval, native XY
targets and B arrays to agree exactly. W's input interval is current 0:+14
and HW's is -3:+14 by design; their scored targets remain identical.
Replay each component's baseline and all five candidate/wrong-pair SSE rows
against its immutable bank with the existing scorer. A mismatch stops the
attempt rather than relaxing equality, copying a different bank or refitting.
Validate the original summaries' other-fifteen-session selectors independently
of new mixture outcomes. Original components and their figures are preserved.

## Original endpoint, anchors and scientific depth

Use unchanged joint native-XY whole-session D, original B, negative R-squared,
and exact within-direction whole-trial evaluation-pairing derangement mean.
An interior convex mixture is intentionally a model ensemble: score its
combined predictions, not the average of its component SSEs. This differs
from the mean-SSE convention for repeated negative-control draws.

Report all sixteen selected mixture rows, lambda/alpha choices, both animal
means, primary-positive counts, full fixed-prediction session omissions,
original 0.01/0.005 margins and directional requirements. Compare the blend
to both W and HW at their independent original outer-selected penalties and
at the blend's selected shared penalty, using direct same-target prediction
SSE and identity differences. These are existing-anchor diagnostics with no
added margin, pass condition or noninferiority claim. No anchor is refit.

One distinct complete recipe adds at most one valid scientific candidate;
the 25 parameter cells, two saved anchors and omissions add none. It is not
a new literal null control. Do not claim it increases the conservative
alternative-check count merely because the mixture has endpoints.
If all selected predictions exactly reproduce the original selected W panel
or original selected HW panel across all sixteen sessions, record the global
structural duplicate and charge zero. Literal endpoint choices and exact
selected-array equality are valid duplicate evidence; close scores are not.
Do not force an interior alpha to obtain another scientific count.

If distinct, the candidate uses the identical original full trial/target/B/D
support and may update the comparable incumbent under the adopted patience
rule. Any strictly greater complete primary score resets patience even if
promotion fails. A mere parameter change or diagnostic-anchor gain does not.
The original practical and identity gates, missing future-session evidence,
required controls, 28–64 trial bounds, twelve-trial patience and resource
ceilings are unchanged. Root records ordinals in actual completion order.

## Bounded artifact computation after adoption

No provider source read is needed. Stream one pair of saved session prediction
banks at a time; preserve all complete outputs. The largest original session
has 1,038 trials, so two float64 five-penalty XY arrays use about 2.5 MB;
25 temporary full predictions need not coexist. Save each session's 25 score
cells, exact endpoint/replay flags and selected final predictions. Generate
the full summary only after every original session completes.

Propose a one-thread, 2 GiB, 60/90-second fixed Chewie_20150313 resource pilot
covering all 25 cells and both anchor replays; its scores stay closed until
the whole cohort completes. Select a finite full-cohort bound from complete
pilot resource facts, preserve the pilot, and reuse a suitable task allocation.
Focused fixtures should check endpoint identity, intentional ensemble SSE
cross-terms, unchanged B/targets, exclusion of each held session from tuning,
both levels of exact ties, wrong-pair identity, native-XY scaling and complete
global duplicate accounting. Tests and artifact scoring run only in root's
verified compute allocation. No new allocation or remaining bound is selected
before complete pilot resource review.

Root resource selection,09:05UTC: fixed pilot completed09:04:27.835798 to
09:04:29.037626 in1.201828 seconds,1.156490 saved-artifact read/replay and
0.045011 scoring,67,584KiB peak,1,038 trials/86 electrodes, both exact
component replays passed and zero source reads/refits. Scientific mixture
scores stayed closed. Preserve this pilot; run fifteen fresh sessions with
one thread and2GiB,180-second internal/210-second external aggregate bounds,
unchanged60/90 children. Fifteen pilot equivalents18.027422 seconds, a full
90-second child reserve and sixteen replay equivalents19.229250 leave
52.743328 seconds for summary writing and variation. Require240seconds
actual allocation time remaining. Fresh summary and parent outputs use
`primary_history_blend_summary_attempt001` and
`primary_history_blend_panel_attempt001`. No automatic retry or extension.
