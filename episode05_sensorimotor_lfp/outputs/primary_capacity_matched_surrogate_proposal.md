# A capacity-matched training-feature surrogate for the fixed HW model

Prepared 2026-10-04 on root's explicit request. **Adopted by root after
independent scientific review at 08:45 UTC; implementation and focused checks
authorized, fixed source pilot follows only after their review.** No new source
value or partial scientific score informed this choice. Implementation and
focused fixtures are being prepared; this document makes no execution or
result claim. GOAL, DATASETS and SEARCH_POLICY remain unchanged. Root owns
compute and the scientific record. Matched-budget and elapsed-only scientific
scores were not inputs to this choice.

The 08:08 UTC coverage audit in `experiment_log.md` leaves an explicit
capacity-matched random-feature or permitted-surrogate check open. This
proposal supplies one narrowly defined **whole-trial label-randomized-feature
surrogate** of the original full-support history-plus-repeatability model HW.
It is an instance of the allowed LMP/history/ridge and whole-trial pairing
control families. It does not generate raw-voltage phase, posit a Gaussian
neural process, or claim that a renamed earlier control closes this gap.

## Why this check and what is reused

Fix the already completed `lmp_history4_repeatability_weight_ridge_v1`
representation and its original five penalties. Its released animal-balanced
primary increment, 0.0023412544731769847, remains below the adopted 0.01 gate.
Its existing eight weight-label controls reassign a physical repeatability
prior and have their own changed covariance/df. They do not hold the full
ridge design's capacity fixed.

The earlier quadratic T/S8 control also preserved its own fitting Gram under
whole-trial training shuffles. It remains a completed control of the 3E
quadratic-time map, with its original role and count. It is not an HW control,
and is not relabeled or recounted here. The present check fixes HW's 4E lag
representation, physical repeatability metric, shared behavior baseline and
original selected penalty, then randomizes only the training-trial assignment
of the completed feature trajectories. The exact Gram/df identity is an
explicit premise of this prospective comparison.

Use all sixteen original M1 sessions, complete original trials, their original
qualified physical LMP electrodes and eight held-direction folds. Preserve
Mihili's six sessions and Chewie's ten, with both Chewie implants one animal.
The native input is LMP offsets -3:+14 with lag order [0,-1,-2,-3]; native,
unscaled XY targets stay at offsets 0:+14, fifteen 30-ms bins. No new source
interval, target, feature family, channel subset, trial subset, unit rule,
filter, phase reconstruction or window search is introduced. No spike values
or future-session source is needed.

## Preserve the complete original HW fitting state

For every held direction h, reproduce HW's original state using only its
seven permitted fitting directions:

- Equal-direction-weighted first-harmonic direction/time template B0 and its
  original pooled residual intercept c. Keep B=B0+c as both comparator and
  prediction add-back. Preserve the original numerical residual convention,
  `(Y_fit-B0_fit).reshape(-1,2)-c`.
- W's original current-bin physical mean/population SD and T/P/r/a statistics
  on the unchanged native current view, preserving the memory/reduction path.
  Zero SD maps to one; `r=clip(P/T,0,1)`, T=0 gives r=0, and `a=0.1+r`.
- H's original separate lag/electrode fitting mean/population SD and final
  fitting-column centering. Keep original lag-major physical-ID order and
  the distinct current-weight and history-decoder states.
- Original full-history HW physical weights
  `w_e^2=a_e*trace(G_H)/sum_j a_j*q_j`,
  `q_e=sum_lag G_H[(lag,e),(lag,e)]`, with the same zero-Gram/equal-a identity
  conventions and one physical weight repeated over all four lags. Add no
  post-weight normalization or centering.

Call the completed weighted fitting trajectories F, with shape
`N_fit_trials x 15 x 4E`, their row-flattened matrix Z, and the unchanged
evaluation matrix Z_eval. Keep the true centered native XY residual matrix R.
F and Z are computed once before any surrogate mapping. The true target,
baseline, fitting feature pool and preprocessing are the same for every arm.
Do not permute physical weight labels, reestimate a surrogate-specific prior,
change history layout, or replace any of these states with elapsed-only or
matched-budget states.

## S_HW: one eight-draw, whole-trial training-feature surrogate

Proposed recipe ID:
`lmp_history4_repeatability_weight_training_surrogate8_v1`.

Use exactly eight fixed replicate IDs 0 through 7. Within each session,
held direction and each of the other seven fitting directions, sort the
original session-qualified trial IDs by their validated one-based numeric
source suffix. Preserve the full serialized IDs in outputs.

For replicate r and fitting direction d, initialize Python `random.Random`
with `seed(literal, version=2)`, where the literal is

`ep05-lmp-hw-capacity-training-feature-surrogate-v1|20261004|<session>|<held>|<replicate>|<direction>`.

Use the canonical integer labels, unpadded decimal h/r/d, and one ordinary
`shuffle` of the ordered donor positions. **Accept fixed points, full identity
maps and repeated maps; never redraw or choose a seed by an outcome.** This
uses the ordinary permutation distribution, not the earlier T/S8 rejection-
sampled derangement distribution. A draw can retain some true training pairs;
that limitation is explicit, and all eight prescribed draws remain in the
mean, including duplicate/identity draws.

Each recipient fitting trial receives the donor's entire completed
15-by-4E feature trajectory. Move all target times, all lag blocks and all
physical channels together, retaining time and coordinate order. Never cross
directions, import a held-direction trial, permute individual bins, or alter
the targets. Save recipient and donor native IDs, fitting direction, replicate
and literal seed for every mapping. A non-bijective or misgrouped map stops
the attempt; no fallback shuffle is authorized.

Evaluation features and targets retain their original correct trial pairing.
There is no evaluation shuffle in this training-surrogate recipe. Its ordinary
fixed-prediction wrong-trial evaluation readout is still calculated separately
using the existing exact within-direction derangement mean.

## Exact conditional ridge-capacity matching

Let P_r be the row permutation induced by the whole-trial map, preserving each
trial's fifteen-row time order. The surrogate fitting design is

`Z_r=P_r Z`, therefore `Z_r'Z_r=Z'Z`.

Use **one original symmetric Gram and eigensystem per held-direction fold**
for every surrogate and original penalty. Reuse the original HW positive-ridge
solver equations, including its unnormalized Gram and denominators
`eigenvalue + n*lambda`, with `n=15*N_fit_trials` and
`lambda in [0.001,0.01,0.1,1,10]`. No negative-eigenvalue clipping, rank cutoff,
new solver tolerance, target scaling or penalty rescaling enters the fit.
Keep covariance/rank/df diagnostic conventions separate from the solver.

Only the cross-product and coefficients change:

`C_r=Z_r'R`,

`beta_r,lambda=(Z'Z+n*lambda*I)^(-1) C_r`,

`P_r,lambda = B_eval + Z_eval beta_r,lambda`.

The last P denotes a predicted XY trajectory, not the row permutation matrix.
An implementation should use distinct variable names for these objects.
Sharing the Gram avoids manufacturing tiny apparent capacity differences
from summing the same permuted rows in another floating-point order. It does
not justify sharing the true-pair cross-product or neural coefficients.

This matches more than width or nominal lambda. For the normalized Gram
`G=Z'Z/n`, both true-pair and surrogate heads have exactly the same Gram
spectrum, rank, trace and ridge effective degrees of freedom at every penalty:

`df(lambda)=trace[G(G+lambda I)^(-1)]`.

Their fitting ridge smoother matrices are related by permutation conjugacy,
`H_r=P_r H P_r'`, so their smoother spectra and squared-smoother trace also
match algebraically. This is a conditional statement about the fixed HW
design and ridge head. It is not matched biological information, equal
generalization performance, or a statement that the surrogate is a realistic
neural recording. No different model is made capacity-matched merely by using
the same feature count or penalty.

Keep the original `projected_covariance` diagnostics for the shared weighted
design, including their declared diagnostic-only PSD tolerance. Reuse those
same diagnostic values for the surrogate; do not claim numerical rank is an
additional fit truncation. Report width, trace, rank and df for all five
penalties, identifying the originally selected penalty. Since the same Gram
is used, the capacity identity is by construction and verified mapping,
not a newly optimized matching tolerance or a score-selected df target.

## Saved anchor, fixed tuning and evaluation

Reuse the original completed HW candidate, without refitting it, from these
paths under `outputs/runs/adaptive_search/`:

- `primary_history_repeatability_weight_control_attempt002/<session>/score_banks.json`;
- `primary_history_repeatability_weight_control_attempt002/<session>/lmp_history4_repeatability_weight_ridge_v1_predictions.npz`;
- `primary_history_repeatability_weight_summary_attempt002/primary_history_repeatability_weight.json`.

Require literal replay of original source support, trial IDs/order, directions,
targets, original B, current-weight state, history normalization/centering,
repeatability statistics and physical weights after the new surrogate fits.
Replay the saved original full prediction banks' scoring as in the elapsed-only
runner. No current matched-budget or elapsed-only bank is an anchor. Computing
the true-pair cross-product for identity diagnostics does not authorize another
HW neural fit or another charged candidate.

Fit every surrogate at all five original penalties. Then use the original
complete HW summary's other-fifteen-session-selected lambda for each outer
session, and apply that same lambda to every surrogate replicate. Preserve
its original equal-session/equal-animal selector and stronger exact ties.
The surrogate does not choose its own lambda, a seed, or a new anchor from
its scores; this is a control of fixed HW rather than a tuned rival null.
Reading the selected original penalty after all new banks are fitted makes
the numerical boundary explicit without pretending the corpus is fresh.

Use the unchanged positive finite full-session denominator
`D_s=sum_(i,t,k)(Y_i(t,k)-mean_(original session trials/bins)Y_k)^2`, solely for
scoring. Keep full joint native XY trajectories and negative R2. Calculate
the ordinary correct SSE and existing exact within-direction whole-trajectory
wrong-pair mean separately for every surrogate prediction. Aggregate the
eight **correct SSEs** and eight **wrong-pair SSEs**, then form

`delta_S=(SSE_B-mean_r SSE_correct,r)/D_s`,

`identity_S=(mean_r SSE_wrong,r-mean_r SSE_correct,r)/D_s`.

Do not average the predictions into an unintended ensemble. Report original
HW minus mean surrogate using direct same-denominator SSE differences and
identity differences. Retain all eight seed-level contrasts; no permutation
tail p-value or confidence interval is inferred from eight draws.

Collapse to one estimate per session, equal-session means within each animal,
then equal weights for the two animals. Report every session, animal means,
positive-session counts, all fixed-prediction within-animal omissions, all
seed contrasts and the common selected lambdas. Omissions do not refit or
retune anything. The original HW practical and identity gates remain as
recorded. **No new promotion threshold or pass/fail gate is introduced.**

## Interpretation, depth and failure rules

A positive HW-minus-surrogate difference would support a benefit from the
correct training-trial assignments in this fixed representation at matched
ridge capacity. A nonpositive difference would limit that narrower learning-
correspondence claim. Neither sign alone establishes absence or presence of
a biological mechanism, causality, online decoding or new-animal robustness.

The surrogate preserves the full fitting feature distribution, its direction
membership, elapsed-time trajectories, lag consistency and electrode
covariance, and evaluation features remain correctly paired. It therefore
retains direction/time structure and may predict some true evaluation-trial
variation. It is **not a signal-free random-neural null**, and a positive
surrogate score is not automatically evidence of leakage. This is a permitted
feature-label surrogate, not a phase-randomized raw-voltage reconstruction.

The eight-seed mean is **one literal null-control recipe**. If distinct and
complete across all sixteen sessions, it adds one valid scientific comparison
and one literal null control, also qualifying as an alternative-explanation
check. Root applies these increments in actual completion order; no ordinal
total is inferred here. The saved HW anchor, new diagnostic arithmetic,
individual seeds, five penalties, folds, omissions and resource pilot add no
comparison. Earlier T/S8 and HW weight-label counts remain unchanged.

Before outcomes are released, define duplicate accounting by exact structure:
if every surrogate cross-product equals the true-pair `Z'R` literally for
every prescribed seed, fold and session, the shared Gram and baseline make
the whole mean surrogate algebraically duplicate HW and its charge is zero.
This includes all-identity maps, identical mapped trajectories, and a fully
zero design. Record the direct map/design identity flags as well. A single
identity or repeated seed remains in the fixed mean and does not by itself
remove the recipe's charge. Do not drop an identity draw, replace it, apply a
near-equality tolerance, or infer duplication from similar evaluation scores.

As a null control it cannot update the comparable true-pair incumbent or
reset patience under `primary_patience_interpretation.md`. A distinct complete
negative control still counts once. Technical failures and incomplete panels
count zero; preserve their artifacts. Completing this comparison would
perform the explicit capacity-surrogate coverage task, not certify promotion
or completion of the wider search. Keep original margins, all remaining
coverage, the 28–64 bound, twelve-trial patience, control allocation, 2,000
core-hour ceiling and 120-hour clock unchanged.

## Bounded implementation proposal after adoption

Use new isolated core/runner files and a fresh attempt. Reuse the qualified
history reader and original HW preprocessing helpers; do not modify the old
HW fitter or banks. One source read supplies all eight surrogates. Sequential
per-replicate cross-products can share each fold's Gram/eigensystem. There
are 8x8x5=320 surrogate penalty fits per session, without a new HW candidate
fit. Save all five predicted XY banks per seed and complete ID maps.

Focused compute fixtures should establish exact within-direction bijections
and whole-time/lag/channel movement; unchanged original fitting state and
held-direction exclusion; the Gram/smoother conjugacy identity; shared-Gram
predictions against an independent direct solve on a small permuted design;
correct target/add-back and unchanged evaluation features; saved-anchor
lambda exclusion; mean-SSE rather than prediction-ensemble scoring; and
exact all-seed duplicate accounting. Small algebra checks run only through
root's verified compute allocation. No generic qualification chain is added.

Propose one complete fixed **Chewie_20150313** resource pilot, all eight
surrogates and five penalties, with **one CPU, 4 GiB, numerical threads one,
270 seconds internal and 300 seconds external**. Its original 1,038 trials
and 86 physical electrodes imply a 42,848,640-byte full float64 history design
and 9,964,800 bytes for eight five-penalty XY prediction banks before copies.
The completed original HW candidate-plus-eight-weight-controls pilot used
428,072 KiB peak RSS and 23.340652 seconds; the elapsed-only pilot's resource
report used 428,472 KiB and 59.634049 seconds, including 56.241556 seconds
source reading. These prior observations motivate a bounded pilot, not an
assumption that source cost or total runtime is constant.

Keep pilot effects closed. Root would select a finite remaining-panel bound
from completion/support/resource facts, preserve and reuse the pilot, and
verify actual allocation CPU/memory/time before launch. No remaining bound,
parallel worker count, retry, solver-cap increase or new allocation is
selected by this pilot proposal. Every required session/seed must finish;
no dropped cell, source workaround or alternate seed is authorized.
