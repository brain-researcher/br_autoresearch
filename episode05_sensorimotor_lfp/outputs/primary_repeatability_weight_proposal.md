# Does target-free repeatability weighting improve full-channel LMP prediction?

Prepared 2026-10-04 from the complete nineteen-comparison development release.
**Adopted by root at 2026-10-04 06:48 UTC after root and independent scientific
review.** Root owns compute and execution records. No elastic-net partial scientific
scores were released to prepare this proposal. Its incomplete attempts add
no scientific-trial depth and do not motivate a new model from their outcomes.

## Scientific question and permitted scope

The complete PCA, marginal top16 and target-covariance SVD comparisons each
lost to full-channel LMP ridge in both animals. The covariance-SVD transform
beat its correspondence control but failed both practical margins; the
current-bin full-LMP anchor also remains below the required primary margin.
These complete results motivate keeping every physical input coordinate while
testing a different, target-free allocation of ridge shrinkage.

Training-only reliability weighting is explicitly in the permitted grammar.
This proposal uses a precisely defined **repeatable direction/time trajectory
energy ratio**, without claiming that it is an unbiased estimate of intrinsic
electrode reliability. Here target-free means that no velocity target or
prediction score enters the weights; original reach-direction labels remain
inputs. The question is whether emphasizing LMP channels whose
15-bin trajectories repeat across trials to the same target improves the
native residual-velocity prediction relative to assigning those same weights
to arbitrary electrode identities and relative to ordinary full-LMP ridge.

A repeatable task-locked trajectory can have little trial-specific predictive
information. Conversely, trial-varying LMP may contain exactly the information
EP05 seeks. The weighting can therefore downweight useful signal; the outcome
must decide whether this fixed input prior helps. Repeatability is not a
biological quality gate and does not define trial variance as noise.

## R: positive repeatability weights on every physical LMP coordinate

Candidate ID: `lmp_repeatability_weight_ridge_v1`.

Retain all sixteen original M1 sessions, original complete trials and physical
LMP electrodes, and the current-bin native offsets 0:+14. Keep the original
eight outer held-direction fits, equal-direction first-harmonic baseline plus
common fitting-residual intercept, native XY residual target, five original
ridge penalties, exact within-direction wrong-trial evaluation, whole-session
SSE denominator and equal-session then equal-animal summaries. No spike,
high-frequency, new time-window or additional source interval is used.

For each held direction h, use only the other seven directions for the
physical mean/population-SD normalization and every weight below. Zero SD
becomes one exactly as before; retain the column. Let `z_di,e` be the resulting
15-bin vector for trial i, fitting direction d and physical electrode e, and
let N_d be the complete fitting trial count in that direction. The existing
support rule guarantees N_d>=2 for every direction. The arithmetic mean over
the seven directions below gives each direction equal weight, irrespective
of trial count:

`T_e = mean_d [ sum_i ||z_di,e||² / (15*N_d) ]`

`P_e = mean_d [ (||sum_i z_di,e||² - sum_i ||z_di,e||²)
                 / (15*N_d*(N_d-1)) ]`.

P is the average cross-trial dot product of the entire time-aligned trajectory,
computed without enumerating all trial pairs. T is the corresponding mean
single-trial energy. P can be negative. By the sum-of-vectors inequality,
P<=T in exact arithmetic. This does not make the ratio an unbiased reliability
estimator: normalization was estimated from the same fitting data, and common
direction/time means contribute directly to P.

Use the fixed rule `r_e = clip(P_e/T_e, 0, 1)` when T_e>0 and r_e=0 when T_e=0.
No epsilon, trial exclusion, channel deletion, significance test or target
quantity enters this rule. Define the strictly positive base weights
`a_e = 0.1 + r_e`; the floor 0.1 is fixed before the new result, with no floor
grid or later score-based adjustment. Negative estimated agreement receives
the floor while its signed P and unclipped ratio remain reported.

Flatten the original fitting z into Z with n trial-time rows and E columns,
and define `G = Z'Z/n`. Preserve the total standardized input energy by setting

`w_e² = a_e * trace(G) / sum_j a_j*G_jj`.

When trace(G)>0 the denominator is strictly positive because a_e>=0.1.
Take the positive square roots w_e and form `H = Z diag(w)`. Apply those
same fitted weights to the original physically standardized evaluation
features. Do not restandardize H, whiten it, truncate its rank or fit another
intercept; any of those would change or undo the selected input prior.
When G is exactly all zero, set w=1 for every column and retain the fold as
an identity/baseline case. Nonfinite energies, weights or matrices are
unresolved arithmetic failures, not permission to add an epsilon or drop data.

Fit the original native-residual ridge head at every original
`lambda in [0.001, 0.01, 0.1, 1, 10]`:

`(H'H + n*lambda*I) beta = H'U`.

Here U is the unchanged centered native XY fitting residual; add the same
baseline B back to obtain complete native velocity predictions. Select each
outer session's lambda using only the other fifteen candidate banks under the
original equal-animal selector and stronger-penalty tie rule. No session may
choose a weighting family or use its own evaluation scores to select lambda.

All weights are positive, so the physical feature span and its exact rank are
retained. In original physical standardized coordinates, the model instead
penalizes `lambda * sum_e ||coefficient_e||² / w_e²`. This is a fixed diagonal
generalization of ridge, not feature removal. It changes the ridge metric and
effective shrinkage despite preserving the total Gram trace.

## C: eight frozen physical weight-label controls

Control ID: `lmp_repeatability_weight_labels8_v1`.
Replicates 0..7 form one mean-eight control comparison.

For each session and replicate, start with the canonical ascending original
physical electrode IDs. Use ordinary Python `random.Random().seed(literal,
version=2)` and shuffle one fresh donor-ID list with the literal

`ep05-lmp-repeatability-weight-label-v1|20261004|<session>|<replicate>`.

Map each recipient electrode to the corresponding donor electrode. Accept
identity permutations, fixed points and chance alignments; do not redraw.
The same physical-ID map is reused across all held directions, trial/time
rows and penalties within that session. Save exact recipient/donor physical
IDs and the seed literal. No numerical outcome chooses a map.

For each outer held direction, retain the candidate's fitting-only T, P, r,
a, original physical normalization, Z and U. Reassign the **base weights** as
`a_control[e] = a[donor(e)]`, then apply that control's own trace normalization:

`w_control[e]² = a_control[e]*trace(G) / sum_j a_control[j]*G_jj`.

Do not permute the signal columns, targets or an already normalized w vector.
The control gets its own H, Gram and true-U ridge coefficients. Its ordinary
evaluation uses the original LMP and native XY with those fitted weights.
Both candidate and controls retain E coordinates and the same total weighted
Gram trace. Their covariance spectra, effective ridge degrees of freedom,
predictive energy and active numerical rank diagnostics can differ. This is
not equal-effective-capacity evidence or a pure causal intervention.

Use the candidate's outer-session-selected lambda for all eight controls.
Average the eight correct SSEs and the eight exact wrong-trial SSEs with the
original common denominator. Do not average predictions, select a seed or
retune a control lambda. The unchanged evaluation-pairing check remains
separate from this physical weight-label control.

## Structural identity and counting

Check identity separately for each candidate and each control arm in every
fold, before scientific summaries. If G is all zero, the weighting is an
identity transform. Otherwise, if every positive-energy physical column
(`G_ee>0`) receives the same base a, trace normalization makes its effective
w=1 algebraically. Weight values on exactly zero columns cannot alter the
predictor. Use exact equality of the base a values for this structural rule;
do not invent a near-uniform tolerance. An implementation may use all-one
effective weights in this algebraic identity case to avoid meaningless
normalization roundoff, while retaining the original a diagnostics and maps.

An identity candidate does **not** imply identity controls. For example, a
zero-variance column has a=0.1; shuffling that label onto one of several
positive-energy columns can create nonuniform active weights even if the
candidate's active a values were equal. Each control must therefore be
checked using its own reassigned a and the original positive-energy support.

Charge the candidate zero additional depth only if its effective weighting is
the identity in every fold of every required session. Charge the mean-control
recipe zero if all eight control arms are identities in every fold and
session. Root additionally adopted, before new data, the conservative rule
that the control adds zero if all its arms equal the candidate's effective
weighting in every fold/session. Record `all_controls_equal_candidate` using
literal equality of the actual applied weight vectors restricted to
positive-training-energy columns; empty support is equal. Do not use a
tolerance or compare near-equal outcomes. A zero-training-energy column has
an exactly zero ridge coefficient and cannot distinguish the predictors.
Thus a nonidentity candidate with identical controls adds one, not two.
Otherwise each complete fixed recipe contributes one comparison;
the pair can add 0, 1 or 2, never extra charges for seeds, directions or
penalties. Record the per-recipe identity evidence. A new negative result or
a mixture of identity and nonidentity folds remains a valid comparison.
Reusing the preserved full-LMP anchor contributes no new depth.

## Prospective contrasts, diagnostics and limits

Report R, mean C, R-minus-C and R-minus-full-LMP under both independently
selected and R-matched original lambda. Reuse the complete full-LMP five-
lambda banks after the existing exact source/trial/baseline/target replay.
Retain every session, both animal means, all eight control seeds and every
fixed-prediction session omission. Seeds and sessions do not add animals.

The prospective weight-label screen requires strictly positive R-minus-C
primary and identity contrasts in both animals and primary-positive session
majorities of at least 4/6 Mihili and 6/10 Chewie. Ties fail. A claim that the
weighting improves full LMP additionally requires the same directional screen
against both full-LMP anchor selections. These do not replace the unchanged
0.01 primary / 0.005 identity practical gates, remaining required falsifiers
or the absent future-session evaluation.

Save each fold's fitting direction counts, original physical mean/SD,
T/P/unclipped ratio/clip category/r/a, all candidate/control weights and exact
physical maps. For T=0, report the unclipped ratio as undefined rather than
fabricating a value. Also report each arm's trace, spectrum, numerical rank
and `sum_j eigenvalue_j/(eigenvalue_j+lambda)` ridge df at the selected lambda,
using the existing projected-covariance diagnostic convention. Diagnostics
never change the weights, rank, penalties, seeds or eligibility. Preserve
signed negative P even when the model uses its fixed clipped weight.

Any gain remains exploratory evidence about this weighting of the exposed
corpus. It does not show that trial-specific LMP variance is noise, identify
an intrinsic channel property, certify spike-free LMP, establish a causal
mechanism, or support online/new-animal/cross-session-coefficient transfer.
A failure rejects this fixed input prior and floor, not all reliability-based
models. A positive identity contrast would still require all original primary
and alternative-explanation conditions before promotion.

## Finite implementation and resource selection

After adoption, use new core/runner files and the original current-bin LMP
reader, baseline, physical scaling, direct positive-penalty ridge solve,
wrong-trial score and outer selector. Do not extend the stalled elastic-net
iteration limit or mix its partial banks into this family. No new compiler,
library installation, data copy, source window or source scan is required.

Focused compute fixtures should check T/P against explicit ordered distinct-
trial pairs with unequal direction counts; negative agreement, exact-zero T,
clipping and the fixed floor; exclusion of held-direction features and every
velocity value from the weight calculation; trace preservation and the
equivalent diagonal ridge penalty; physical-ID maps shared across folds,
including an accepted identity draw; distinct control Grams and true-U
heads; the identity candidate/nonidentity control zero-column edge; per-arm
0/1/2 accounting; mean-eight SSE and outer-session lambda exclusion.

The full panel has 128 fitting-only repeatability constructions and
5,760 ridge fits, arranged as nine arms times five penalties in each of eight
folds and sixteen sessions. Matrix decompositions can be reused across the
five penalties within one arm; no iterative convergence relaxation is
needed. Resource scale is that of prior current-bin full-channel controls,
but those timings do not measure this new recipe. Propose a complete fixed
Chewie_20150313 pilot on one CPU within 4 GiB, 270 seconds internal and
300 seconds external after review and focused fixtures. Keep scientific
pilot scores closed while root selects a finite remaining-panel bound from
resource/support facts. No automatic retry or favorable-session omission.

If both recipes are distinct and complete, depth advances 19 to 21 with eight
controls among fifteen post-initial comparisons. If one is a structural
duplicate, count only the other and update its actual control contribution;
if both are duplicates, add zero. Preserve the 28–64 scientific-depth bounds,
12-trial post-minimum patience, >=40% post-initial control fraction, 2,000
core-hour ceiling and 120-hour clock starting 2026-10-03 22:23:02.824 UTC.
This proposal introduces no material-improvement cutoff or early-stop rule.
