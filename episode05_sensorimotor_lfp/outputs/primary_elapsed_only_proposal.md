# Does the fixed HW result survive an elapsed-time-only comparator?

Prepared 2026-10-04 on the coordinator's explicit request. **Adopted by root
2026-10-04 08:25 UTC after root and independent scientific review.**
Implementation and focused fixtures are being prepared; this document makes
no execution or scientific-result claim. Root owns compute and execution.
The parallel matched-trial/channel work has separate ownership;
neither its unreleased scores nor an incomplete elastic-net bank informed
this proposal. Existing GOAL, DATASETS and SEARCH_POLICY remain unchanged.

GOAL requires an elapsed-time-only behavioral comparison, and
`primary_reference_proposal.md` explicitly left it open. Use the already fixed
full-original history-plus-repeatability recipe HW,
`lmp_history4_repeatability_weight_ridge_v1`, whose released complete primary
increment is 0.0023412544731769847. It remains below the 0.01 primary margin.
This check addresses one required alternative explanation; it cannot turn
that failed primary gate into a pass.

There is one new neural recipe: HW fit after subtracting an elapsed-only
template, called ET-HW below. The standalone time template and new contrasts
of saved original HW predictions are necessary comparators, not additional
scientific trials. No harmonic, temporal basis, lag, channel, penalty or
window sweep is proposed. No extra weight-label ensemble is introduced.

## Original support and the fixed time-only model

Keep all sixteen original M1 sessions, their original complete trials and
physical LMP electrodes, and their eight held-direction fits. Mihili has six
sessions and Chewie ten; its two implants remain one animal. Use the qualified
LMP offsets -3:+14, fixed lag order [0,-1,-2,-3], and unscaled native XY
targets at offsets 0:+14: fifteen 30-ms bins, timestamps 0 through 420 ms.
No source interval, target, trial role, matched subset or source field changes.
Spikes, PMd, other animals and future sessions remain outside this check.

For held direction h, use only the other seven fitting directions. At each
fixed native time t and coordinate k, define

`T0_h(t,k) = (1/7) sum_(d != h) [(1/n_d) sum_(i:direction_i=d) Y_i(t,k)]`.

This is the **intercept-only, equal-direction-weighted counterpart** of the
original first-harmonic template. Implement it through the existing
`DirectionTime.fit(..., harmonics=0, direction_labels=fit_labels)` numerical
path. The displayed direction-mean formula specifies its estimand; an
independent small fixture may check algebraic agreement to ordinary numerical
tolerance. Do not take the original first-harmonic model's fitted constant
coefficient: the missing eighth direction means its constant and harmonic
columns are not generally orthogonal. Do not replace the equal-direction
fit with a trial-pooled per-bin mean, which would also change weighting.

As in the original baseline, apply one common, unpenalized, pooled
fitting-trial/bin residual intercept per coordinate:

`c_T,h = mean_(fitting trials and bins) [Y_i(t,:) - T0_h(t,:)]`,

`T_h(t,:) = T0_h(t,:) + c_T,h`.

The same T is the ET-HW comparator and the term added back to its residual
prediction. Preserve the original reshape/mean convention for c_T. Do not
introduce a separate intercept per direction, time or trial, and do not assume
c_T is zero under unequal direction counts. T has fifteen unrestricted time
coefficients per coordinate plus this prescribed common calibration; there
is no smoothing or temporal model selection.

T is independent of the predicted trial's angle within a fitted fold. The
direction labels still define the seven fitting groups, their weights and the
held-direction split. Its fitted values can differ across held-direction
folds because the permitted training observations differ. This is an
elapsed-only predictor under the original split, not one template fitted to
all eight directions. Evaluation behavior does not enter its coefficients.

## ET-HW: change the residual baseline, preserve the neural recipe

Proposed recipe ID: `lmp_history4_repeatability_weight_elapsed_only_sensitivity_v1`.

For each held direction, reconstruct exactly the original HW neural state:

1. Preserve the native current-bin view `x[:, 3:18, :]` and W's original
   fitting mean/population SD, zero SD mapped to one. Preserve its memory and
   reduction path; do not insert a contiguous copy. Compute the same
   current-only T/P/r/a repeatability statistics from the seven fitting
   directions, with `r=clip(P/T,0,1)`, T=0 giving r=0, and `a=0.1+r`.
2. Preserve H's separate lag-by-electrode fitting means/population SDs and its
   final fitting-column centering, applied unchanged to evaluation features.
   Keep the 4E columns in original lag-major/physical-ID order.
3. For the resulting unweighted history Gram `G=Z_H'Z_H/n`, use the original
   physical energy `q_e=sum_lag G[(lag,e),(lag,e)]` and HW weights
   `w_e^2=a_e*trace(G)/sum_j a_j*q_j`. Preserve its exact zero-Gram and equal-a
   identity conventions. Repeat each physical weight across all four lags;
   add no post-weight normalization or centering.

Elapsed-only residualization changes neither this neural design nor its
repeatability prior. It changes the fitting response to

`R_T = (Y_fit - T_h).reshape(-1,2)`.

Preserve the original numerical subtraction order: first form
`r0_T=(Y_fit-T0_h).reshape(-1,2)`, obtain `c_T=r0_T.mean(axis=0)`, then
use `R_T=r0_T-c_T`. This is the displayed residual algebraically and retains
the original HW arithmetic convention when the fitted baselines coincide.

Fit the existing normalized ridge objective

`||R_T - Z_HW beta||_F^2/n + lambda*||beta||_F^2`

at exactly `[0.001,0.01,0.1,1,10]`, using the unchanged positive-ridge
eigensystem path in `primary_history.ridge_predictions`. Predict

`P_ET-HW(lambda) = T_h + Z_HW,eval beta_lambda`.

Do not pass the old `Y-B` residual returned by the original history helper
into this fit. Reusing HW's original residual coefficients and only replacing
the term added back would define a different, invalid comparison. Preserve
native XY scaling and the same training-row normalization n. Fit one neural
arm per fold; there are 40 penalty fits per session and 640 across the panel,
with one Gram decomposition reused across the five penalties in each fold.

For each outer evaluation session, choose ET-HW's lambda using only the other
fifteen ET-HW banks. Preserve equal-session means within each available animal,
then equal weight for the two animals, and stronger-penalty exact ties. Use
its increment over T as the selection score; the T term is constant across
lambda, so this ranks the same complete native-XY predictions. Neither that
outer session's scores, old HW's scores, anchor contrasts nor identity scores
select ET-HW's penalty. There is one global recipe, not a per-session choice
between direction/time and elapsed-only baseline families.

## Full-original saved anchors and common scoring

The immutable HW anchor is the candidate recipe above, using these existing
complete outputs under `outputs/runs/adaptive_search/`:

- `primary_history_repeatability_weight_control_attempt002/<session>/score_banks.json`;
- `primary_history_repeatability_weight_control_attempt002/<session>/lmp_history4_repeatability_weight_ridge_v1_predictions.npz`;
- `primary_history_repeatability_weight_summary_attempt002/primary_history_repeatability_weight.json`.

The NPZ contains `actual_xy`, `baseline_xy`, `directions`, `trial_ids` and
`candidate_xy_by_lambda`. Use all five saved original HW prediction banks,
and the original complete summary's already selected outer-session penalties.
Do not refit HW, overwrite its outputs, import the matched-support arm or
replace it with a later leader. Reconstruct original B only to establish
support/state replay and score the fixed comparisons.

Require literal replay of the original full trial IDs/order, directions,
targets, original B and physical support. Recompute and compare original HW
current-weight state, history normalization/centering, repeatability
statistics and candidate weights against its saved fold metadata. The
original B replay helper receives **B**, not the deliberately changed T.
Do not demand equality of T and B as a validity gate. Anchor prediction
arrays and complete status are required; no fallback to an incomplete bank.

Every score uses the same original whole-session denominator

`D_s = sum_(i,t,k) [Y_i(t,k) - mean_(all original session trials and bins)Y_k]^2`.

This scoring-only reference mean never enters any fitted transform or head.
Require finite positive D and exact agreement with the saved original bank;
retain negative R2. Score each full XY trajectory jointly on identical trials.
For a prediction P, report `R2(P)=1-SSE(P)/D` and the exact existing mean
within-direction whole-trajectory derangement SSE. The identity increment is
`I(P)=[SSE_wrong(P)-SSE(P)]/D`; preserve the original no-fixed-point evaluation
pairing rule and its complete-trial support. No Monte Carlo draw is added.

Report these predeclared comparisons separately:

1. **Original HW versus T**, from its saved independently selected full
   prediction: `[SSE(T)-SSE(P_HW)]/D`. Also report B versus T,
   `[SSE(T)-SSE(B)]/D`, and replay the original HW versus B result. The sum of
   those latter two increments equals HW versus T algebraically. This
   comparison preserves the original primary model completely.
2. **ET-HW versus T**, its own baseline: `[SSE(T)-SSE(P_ET-HW)]/D`, together
   with I(ET-HW). Also score its full prediction against original B for a
   clearly labeled common-reference descriptive comparison.
3. **ET-HW versus original HW**, using full prediction SSE differences
   `[SSE(P_HW)-SSE(P_ET-HW)]/D` and `I(ET-HW)-I(HW)`. Report both original
   HW's independently selected penalty and its saved prediction at ET-HW's
   selected penalty. No HW refit is needed for either anchor contrast.

Do **not** subtract ET-HW's increment over T from HW's increment over B and
call it a full-model contrast: that includes the changed comparator. Both
direct model contrasts above share the same D and prediction scoring rule.
Same-lambda comparisons match the neural design and ridge operator, while
the fitted response and supplied behavioral prediction deliberately differ.

Collapse to one score per session, equal-session means within Mihili/Chewie,
then equal animal means. Report all sixteen sessions, both animals, positive
session counts and all fixed-prediction leave-one-session means without
retuning penalties. Show both anchor penalty contrasts. Baseline T and B
are identical across trials within a held direction, so their trial-identity
increment is algebraically zero; they are not trial-specific predictors.

## Prospective interpretation and completion

The fixed original-HW elapsed-baseline screen requires a strictly positive
HW-minus-T increment in each animal, primary-positive session counts of at
least 4/6 Mihili and 6/10 Chewie, and strictly positive within-animal means
under each omission. Report its unchanged original identity result alongside
that screen. A gain over T alone can include the original direction template's
benefit and is not evidence that LFP supplies that entire gain.

The ET-HW residual sensitivity has its own clearly labeled screen: its
increment over T and its exact identity increment must both be strictly
positive in each animal, its primary-positive session counts at least 4/6
and 6/10, and both increments strictly positive under every within-animal
omission. Ties fail. The full-model ET-HW-minus-HW contrasts are descriptive;
neither sign is a newly invented noninferiority margin or a promotion rule.

Report execution completion separately from passed/failed screens. A complete
negative sensitivity closes the planned comparison as performed, while a
failed required positive-result gate still blocks that claim. The original
0.01 primary and 0.005 identity gates remain attached to the original primary
direction/time estimand. No new practical margin is borrowed for ET-HW.

ET-HW is a **nonpromotable elapsed-baseline sensitivity**, not a replacement
primary candidate. A favorable score can reflect direction information now
left for the neural head to predict. It does not establish that the new
baseline is preferable, isolate a biological mechanism, overcome HW's failed
primary margin, or show baseline-independent trial-specific information.
The check remains exposed two-animal exploratory development, offline and
noncausal. It supplies no new session, animal, spike-population or online claim.

## Duplicate rules, counts and bounded next implementation

Only a distinct complete ET-HW recipe across all sixteen original sessions
can add **one** valid scientific comparison and **one** post-initial
alternative-explanation stress comparison. It introduces no null recipe:
the literal null-control count remains eleven. Time-only template fitting,
saved-HW comparisons, five penalties, eight direction folds, omissions and
anchor contrasts add zero. No ordinal total is frozen here because the
separate adopted matched-support work may complete first.

Record an exact original-HW duplicate if the fitted T and B values are
literally identical over both fitting and evaluation support in every
required fold/session, with the original neural state unchanged. This makes
their residuals and predictions the same fixed recipe; charge zero. Also
record zero depth if every ET-HW weighted-design/response cross-product is
exactly zero, so every positive-ridge coefficient is zero and the recipe
reduces algebraically to the uncharged T comparator throughout. Preserve
per-fold identity diagnostics. Do not infer duplication from close scores,
an equality tolerance, or one zero/identity fold, and do not drop such folds.
Incomplete attempts, replays and technical retries add zero.

Under the adopted `primary_patience_interpretation.md`, this nonpromotable
changed-baseline recipe cannot update the comparable primary incumbent or
reset patience. If distinct and complete, count it once in the actual
completion order and apply the existing post-minimum counter. Preserve the
28–64 comparison contract, required coverage/control fraction, 2,000
core-hours, 120-hour clock and all resource ceilings. No new stopping floor,
frontier reset or budget extension is proposed.

After adoption, use separate elapsed-only tools and fresh attempt directories;
reuse the original reader and HW numerical components without mutating them.
Focused compute fixtures should establish: equal-direction time fitting
under unequal counts; exclusion of held-direction outcomes from T and all
fitted states; the shared residual intercept; unchanged HW neural state;
proper replacement of both residual response and added-back baseline;
full-trajectory exact wrong-pair scoring; common-denominator anchor algebra;
outer-session lambda exclusion; and the two exact duplicate rules. These
checks have a direct validity purpose; no generic qualification chain is
proposed. Root owns tests in a verified compute allocation.

Propose one fixed **Chewie_20150313** pilot with the complete single ET-HW arm
and saved-anchor scoring, **one CPU, 4 GiB, all numerical threads one,
270 seconds internal and 300 seconds external**. Reuse a suitable task-owned
allocation if its verified remaining resources suffice. No GPU is needed.
The earlier completed HW pilot used 1,038 trials and 86 physical electrodes:
its full 15-bin by 344-column float64 design is 42,848,640 bytes before copies,
and its five XY predictions are 1,245,600 bytes. The measured original
candidate-plus-eight-controls pilot took 23.340652 seconds, with 10.116058
seconds source reading, 12.911586 seconds fit/score, and 428,072 KiB peak RSS.
These are prior resource observations, not new ET-HW measurements. Shared
source reading and replay costs prevent assuming a ninefold speedup.

Root subsequently measured the fixed ET-HW pilot from 08:30:46.494475 to
08:31:46.128530 UTC: 59.634049416 seconds total, 56.241555933 seconds source,
3.211057669 seconds fit/score, 428,472 KiB peak RSS, 1,038 trials and 86
electrodes. Both exact original replays passed and HW was not refit. Scientific
scores remained closed during the resulting resource-only decision.

Root selected **four single-thread workers in 4 CPU/8 GiB, 720 seconds internal
and 750 external** for the fifteen remaining sessions. Four pilot-duration
waves cost 238.536198 seconds, with a full 300-second child reserve plus
181.463802 seconds for variation. Four measured peaks total about 1.635 GiB.
Reuse the unchanged reviewed fixed-wave dispatcher and actual-CPU guard;
stop-only aggregate signaling prevents future waves and drains bounded active
children. Preserve/reuse the completed pilot. Session caps stay 270/300 and
each launch requires its full 300-second reserve. Every remaining session and
summary path must be fresh. Root verifies the allocation's remaining time
before launching this panel, with no automatic renewal. Resource arithmetic
includes elapsed allocation time once, shared with coordinated work.

Release effects only after the fixed sixteen-session panel and its complete
saved-anchor/support checks succeed. Failure preserves the attempt and its
diagnostic without automatic retry, cap increase, session deletion or altered
recipe. This document neither requests nor submits an allocation.
