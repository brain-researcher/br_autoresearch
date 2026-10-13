# Does successive supervised compression preserve useful weighted history?

Prepared 2026-10-04; **ADOPTED at21:17UTC after root and independent scientific
review.** Implementation and the bounded resource-only pilot below are selected;
the fifteen-session remainder still requires a resource decision from that pilot.
This is one prospective adaptive comparison under the existing PLS/ridge
grammar. It does not adopt the independent-penalty blend amendment, extend
the RBF grid, relax elastic-net convergence or create a new stopping rule.

## Question, prediction and decision

The complete HW105 panel leaves 42 valid comparisons and patience 9/12. Its
primary increment is 0.0014729045689071778 and identity increment is
0.003169133294973225; both practical margins fail. Correct training pairing
beats the eight controls, but the nonlinear replacement loses to independently
selected HW and the constant W/HW blend. The comparable incumbent remains
0.0026788544310999856. All sixteen HW105 selectors choosing the original
smallest penalty does not authorize adding smaller penalties.

The [prospective coverage audit](primary_post_intersection_coverage_audit.md)
already identified a different unresolved question: whether successive,
covariance-aware supervised compression can recover useful directions that
the [nondeflated covariance-SVD transform](primary_covariance_svd_proposal.md)
misses. That earlier transform restricted its input basis to the two columns
of the initial input–target cross-covariance. The present predictor instead
constructs its second direction after removing the first score's fitted
input and target contributions. Correlated inputs can make this new effective
direction leave the original cross-covariance span.

Use **exactly two components at most**, with no component-count search. This
tests a deliberately compact alternative to full HW using the same nominal
component cap as the earlier covariance-SVD reference. Two native outputs
do not imply that two predictive input components must suffice. Failure
would retire this fixed compact recipe, not PLS at every rank. The previous
covariance-SVD recipe used stationary, unweighted LMP; any empirical difference
from that historical result also changes representation. The present controlled
comparison is against HW on the same weighted historical input, while a
synthetic fixture must establish the algorithmic distinction from raw SVD.

The motivating prediction is conditional: if redundant weighted history is
obscuring a compact covariance-adjusted predictive state, this two-component
map should preserve trial-specific prediction and may improve the independent
HW comparison. If it loses or misses the practical gates, that explanation
has not supplied an adequate candidate. Full HW already performs better than
HW105, so there is no presumed need for more flexibility or presumed PLS gain.

GOAL's train-only PLS transform and PLS/ridge predictor choices, and
SEARCH_POLICY's partial-least-squares grammar, permit this prospective operator
change. Existing required family coverage is complete. This proposal adds a
specific compression test; it is not required merely to reach patience 12.

## Fixed source, input and scoring state

Candidate ID: `lmp_history4_repeatability_weight_pls2_ridge_v1`.

Keep all sixteen original sessions, both animals, complete original trials,
original physical LMP channels, LMP offsets -3:+14, target offsets 0:+14 and
lag-major order [0,-1,-2,-3]. Preserve all eight held-direction fits, native
unscaled XY, the equal-direction first-harmonic baseline and its original
shared fitting-residual intercept, the common pooled-XY denominator, exact
whole-trajectory wrong pairing within direction, and equal-session then
equal-animal aggregation. No target, source, history, band or support changes.

Reuse the original HW `prepare_fold` and `history_trace_weights` arithmetic:
current-only W normalization and T/P/r/a with a=0.1+r; separate per-lag
fitting mean/population SD and final history centering; one physical weight
shared over four lags; original full-history trace normalization. Do not
restandardize after weighting, replace a floor, zero evaluation coordinates
using the RBF-specific convention or reweight native XY. All of this is fitted
on the seven allowed directions only and is identical for candidate/controls.

Let X be the resulting n-by-D fitting matrix, D=4E, and X_eval its held-direction
counterpart. Let U be the original n-by-2 centered native fitting residual;
B is the unchanged full baseline added back at evaluation. The new scientific
choice is solely the supervised two-component transform before the ridge head.

## Exact two-component regression-deflation transform

The [official cross-decomposition description](https://scikit-learn.org/stable/modules/cross_decomposition.html#plsregression)
distinguishes one-shot PLSSVD from successive deflation and identifies
regression deflation by approximating targets with input scores. The explicit
equations below define this implementation: thin-SVD directions, regression
deflation, and a separate ridge head. They do not call package defaults,
scale the response, or run an iterative power-method convergence search.
The method reference is not motor-LFP evidence or an endorsement of this trial.

All arithmetic is float64. Initialize X_0=X and U_0=U. For h=0,1:

1. Compute C_h = X_h' U_h / n, then its thin SVD. If its leading singular
   value is at or below

   `tau_h = 64 * eps_float64 * max(D,2) * max(1, sigma_max(C_h))`,

   stop component construction and retain the already constructed K=h
   components. This reuses the earlier covariance-SVD numerical-zero formula
   with D historical coordinates; it is not a score-selected rank rule.
   Record both singular values, tau_h, the stop reason and K. Do not test a
   different tolerance or a larger component count after seeing the result.
2. Select a unit input weight w_h from the leading left singular subspace.
   For a unique largest singular value, that subspace has one column. For an
   **exact** largest-singular-value tie, include every tied leading column;
   do not merge nearly equal values. Call this orthonormal matrix A. For each
   physical lag/channel coordinate j calculate d_j=sum_k A[j,k]^2. Choose the
   smallest j attaining the largest d_j, and set

   `w_h = A @ A[j,:] / norm(A @ A[j,:])`.

   This fixes a direction and sign using the existing lag/channel order even
   at a tied leading singular value. It is invariant to sign changes or
   rotations of the supplied tied basis algebraically. The chosen coordinate
   has positive weight. Record j, tie multiplicity and w_h; do not let a
   favorable native-X/native-Y direction choose the tie convention.
3. Compute t_h = X_h w_h and s_h=t_h' t_h. A finite, strictly positive s_h
   is required after a nonzero cross-covariance direction has been selected.
   Otherwise stop the whole attempt as a numerical failure. Do not fabricate
   a component, add an epsilon to s_h, or turn a failed division into a valid
   K=0 branch. Store the finite loadings

   `p_h = X_h' t_h / s_h`, and `q_h = U_h' t_h / s_h`.

4. Deflate both matrices with that same **input score**:

   `X_(h+1) = X_h - t_h p_h'`

   `U_(h+1) = U_h - t_h q_h'`.

   Do not deflate U with a separately normalized target score, which would
   define canonical rather than regression deflation. Do not recompute a
   baseline, residual intercept or target scaling after deflation.

There are at most two SVDs per transform and no optimizer iterations. Nonfinite
input/intermediate/output, a failed SVD/linear solve, overflow or nonpositive
score energy after an accepted direction is a technical failure with no
valid-comparison credit. No solver fallback, jitter, clipping, relaxed cutoff,
alternate BLAS path or automatic retry is selected. A rank-zero or rank-one
stop caused by the fixed C_h threshold is instead a valid deterministic
low-rank outcome of the specified transform; no fold/session is dropped.

### Projection and native regression head

Inference must use the stored sequential deflation map. Initialize E_0=X_eval.
For each retained component, compute

`z_eval,h = E_h w_h`, then `E_(h+1) = E_h - z_eval,h p_h'`.

Training scores are t_0,...,t_(K-1). Subtract each score's fitting mean from
its training and evaluation score column **after** the complete sequential
transform has been constructed. Retain those means. Do not normalize score
variance, whiten, multiply scores by singular values or refit the transform
at different penalties. The raw weights W are not the inference rotation:
simply using X_eval W is incorrect. Algebraically the effective rotation is
R=W(P'W)^(-1); an independent synthetic fixture checks that identity, but the
deployed arithmetic remains the sequential map, without a pseudoinverse or
an alternative inference branch to rescue discrepancies.

For centered score matrix H and each original penalty
`lambda in [0.001,0.01,0.1,1,10]`, fit

`(H'H + n*lambda*I_K) beta = H'U`.

The head predicts **original U**, not the final deflated U_K and not a target
score. Deflation loadings q_h construct the input transform; they are not a
second additive predictor. Evaluation is B + H_eval beta, with B added once.
Use the original small symmetric ridge solver arithmetic, with no eigenvalue
clipping in the solver. At K=0 the neural correction is exactly zero and the
prediction is literal B for all five penalties; at K=1 retain only that score.

The candidate's other-fifteen-session, equal-animal primary score selects one
of the original five penalties; exact ties select the larger lambda. Neither
identity, the held session, anchor contrasts, rank diagnostics nor a control
score selects a penalty or the component cap. No independent component
penalties, rank grid, smaller-lambda extension or blending parameter exists.

## S: eight relearned whole-trial target-label controls

Control ID: `lmp_history4_repeatability_weight_pls2_target_labels8_v1`.
Replicate labels 0..7 constitute **one mean-control recipe**.

Reuse `primary_covariance_svd.trial_map` literally: numeric original IDs;
seven fitting directions only; sorted recipient and donor IDs within each
direction; Python seed version 2; one ordinary shuffle using

`ep05-lmp-covariance-svd-basis-label-v1|20261004|<session>|<held>|<replicate>|<direction>`.

The retained literal contains its historical "basis-label" name; it supplies
the same deterministic donor maps here, not the old control's scientific
semantics. Each recipient gets its mapped donor's complete 15-by-2 U block.
Accept fixed points, identity maps and repeated draws. Never shuffle bins,
coordinates, directions or held-direction trials independently; never redraw.
Save exact recipient/donor IDs, literals and all eight maps.

For each draw, keep X, X_eval, B and all target-independent HW state unchanged,
replace the fitting target with U_r, and **relearn every supervised PLS weight,
loading, deflation, rank stop and score-centering parameter using X and U_r**.
Fit that draw's own five ridge heads against **U_r**, then evaluate its full
prediction against the original, correctly paired held-direction XY. The
baseline/intercept are not re-estimated from a permutation. The same map is
used throughout that draw's transform and all five heads.

This is a full target-correspondence control. The earlier covariance-SVD
control shuffled labels for basis construction but fitted its head to true U;
that basis-only control is different. The present control asks whether this
entire supervised prediction procedure benefits from correct trial pairing.
It does not isolate the value of supervised projection from the value of
correct head targets. Ordinary within-direction permutations retain conditional
direction/time structure and are not a signal-free recording.

Every control uses the candidate's outer-selected lambda. Average its eight
correct SSEs and eight exact whole-trial wrong-pair SSEs under the unchanged
denominator; never average predictions, tune a control or select a favorable
draw. Report every candidate-minus-draw contrast. Draws are neither additional
animals nor eight independent biological replications or a permutation p-value.

The candidate/control nominal component cap, algorithm and ridge grid match,
but their realized ranks, score Gram matrices and effective ridge degrees of
freedom can differ. Record those quantities with the existing diagnostic-only
covariance conventions. **No exact capacity match is claimed.** A two-column
ridge head does not account for all flexibility used to learn the supervised
projection. This also differs from HW105's fixed-feature, shared-Gram control.

## Replay, comparisons and claim limits

After each session's new fits are fixed, require literal replay of original
HW physical/current/history state, T/P/r/a, history weights/covariance,
baseline, denominator, trials, directions and original five-penalty scores.
Reuse the original time-fast Y/B restoration and preserved prediction layout;
do not change an array convention or use approximate equality to rescue a
reference mismatch. Save every new five-penalty candidate/control prediction
bank and enough fitting metadata to reconstruct the sequential map.

Before any complete-panel scientific release, require all sixteen sessions,
all original/new saved-score checks and independent replay of the preserved
HW five-penalty selector and constant blend's original 25-cell selector.
Report candidate/control primary and identity effects, all sessions, both
animals, majority counts, all fixed-prediction session omissions and all
within-animal omissions. Candidate-minus-mean-control and every draw must
retain their signs transparently; no seed or animal is omitted.

The anchor comparisons are candidate-minus-independently-selected HW and
candidate-minus-independently-selected constant blend. Also report HW at the
candidate's selected nominal lambda as a descriptive shrinkage comparison.
Matching the numeric lambda does not match different score scales or capacity
and cannot override the independently selected policies. The historical
stationary covariance-SVD outcome is context, not an isolated empirical
deflation contrast or an extra counted trial.

For a claim of improvement over an anchor, require positive primary and identity
contrasts in both animals, primary-positive majorities at least 4/6 and 6/10,
and positive within-animal fixed-prediction omission contrasts. Ties fail.
The same directional checks apply to the mean target-label control. Disclose
the matched-lambda HW comparison without converting it into a capacity claim.
None replaces the original absolute primary >=0.01 and identity >=0.005 gates,
original required finalist-specific controls, or absent future-session test.
Recipe-specific support/temporal/contamination checks do not transfer merely
because other predictors used the same input. No spike-population follow-up,
future-session opening, neural mechanism or cross-animal population claim is
activated by this development comparison.

## Literal duplicates, counters and retirement

Check selected complete predictions against the baseline, original selected
HW and blend, completed HW105, and the original selected covariance-SVD
candidate. These are frozen original-support panels; their score contrasts
are not additional trials. If a structural equivalence with another completed
original-support recipe is identified, resolve its literal full-panel identity
before counting. A candidate is a duplicate only when every selected array in
all sixteen sessions equals one same prior panel. Similar scores, equality
in some folds or changing comparison recipes across sessions are insufficient.

Apply the same conservative rule to the mean control: every one of its eight
selected prediction arrays in every session must equal one same complete
panel, including the new candidate, to establish a duplicate. Do not compare
averaged predictions or average scores to declare a duplicate. K=0 and
identity-map cases must preserve these rules, without artificial trial credit.

Process candidate first, mean control second. Charge at most two distinct
complete valid recipes. Only the mean control adds null/alternative credit;
only a comparable candidate with a strictly higher complete primary score can
reset patience. Five penalties, two components, eight draws, anchor contrasts,
pilots, technical failures and exact duplicates do not add depth.

Incoming state is count 42, patience 9, alternatives 22/36 and literal nulls
14/36 after the initial six recipes. If both new recipes are distinct and
neither updates the incumbent, the result would be count 44, patience 11,
alternatives 23/38 and nulls 15/38. That arithmetic is not a reason to run the
comparison and is not a patience stop. A strict candidate improvement resets
the counter before the separate control is counted, even if practical gates
still fail, exactly as the adopted patience interpretation requires.

Retire this fixed recipe after the complete panel if it supplies neither a
better comparable candidate nor the original practical adequacy. Report the
bounded compression question as negative or unresolved as appropriate; do
not rescue it by adding components, whitening, changing score scaling,
moving numerical thresholds or reopening the RBF/independent-blend grids.
The original minimum 28, maximum 64, patience 12, >=40% post-initial alternative
allocation, 2,000 core-hours and 120-hour clock from 2026-10-03 22:23 UTC remain
unchanged. Root reconciles consumed resources; no stopping decision is adopted
by this proposal.

## Prospective implementation checks and resource pilot

Only after scientific adoption may new uniquely named `primary_history_pls`
files implement this recipe; preserve every completed/failed earlier attempt.
Implementation and these focused numerical checks are now authorized by the
adoption above; they run only in a verified task-owned compute allocation.

The essential source-free fixtures should establish:

- a correlated-input example whose second effective deflated direction leaves
  the original cross-covariance span, unlike the earlier raw-SVD construction;
- training-score orthogonality and the sequential held-out map versus the
  algebraic effective rotation, with an example that rejects naive X_eval W;
- original-U ridge targets and direct small-system predictions at all five
  penalties, preserving add-back and target layout;
- exact tied leading subspaces, valid K=0/K=1 stops and technical failures for
  nonfinite or invalid accepted-score arithmetic, with no padded components;
- held-direction exclusion; complete within-direction target-trajectory maps;
  control projection and head relearning; exact identity-map candidate replay;
- other-session selection, strong-lambda ties, mean SSE rather than mean
  predictions, native support replay and whole-panel duplicate/count rules.

Propose a resource-only pilot on the previously fixed largest session
Chewie_20150313: 1 CPU, 6 GiB, no GPU, one numerical thread, 600-second internal
bound and 900-second external envelope. It should read history once, build
the candidate and all eight supervised controls over every held direction,
fit all five heads per transform, save complete banks and replay original HW.
Record source/transform/head/replay/write times, child and allocation memory,
actual retained ranks and success/failure. Keep scientific effects closed.
The nine two-component transforms use thin D-by-2 SVDs and small ridge solves;
the pilot, not that complexity argument, determines safe resources.

Root must adopt this pilot and select any fifteen-session remainder resources
from its measured cost before launch. No full-panel CPU count, walltime,
retry, solver extension or automatic renewal is selected now. Reuse a complete
successful pilot and release scientific effects only after the unchanged
full sixteen-session validation. The absence of a current attended allocation
does not move any test, source read or scientific calculation to a login node.
