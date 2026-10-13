# Can conditional coefficient shrinkage preserve useful full-channel LMP information?

Prepared 2026-10-04 after the coordinator released all nineteen completed
primary comparisons. **Adopted by root at 2026-10-04 06:08 UTC after root and
independent scientific review; implementation is written.** Root owns compute
and experiment records. Adoption and code
creation alone add no scientific-trial charge and change neither the episode
contract nor the existing practical gates.

## Why this comparison follows the released result

The complete target-covariance SVD comparison yielded primary gain
0.000290014223082 and identity gain 0.000602383365873. It beat its mean basis
correspondence control in both animals, but lost to full LMP in both animals
under independently selected and matched penalties. It failed both practical
margins, and omitting Mihili_20140307 made its Mihili primary gain negative.
These are coordinator-released complete-panel observations; this proposal did
not open partial banks or recompute them.

PCA truncation, hard marginal top16 selection and the two-dimensional
target-covariance span have therefore all discarded useful prediction relative
to full-channel ridge in this corpus. That does not show that coefficient
sparsity is beneficial or that every compact representation fails. The next
question is whether **conditional soft coefficient shrinkage**, while leaving
all physical coordinates available, improves on the full-channel ridge
reference. Correlated channels enter the joint prediction rather than being
ranked individually or projected into a fixed low-dimensional span.

Elastic net is already in the permitted predictor grammar. The
[official linear-model documentation](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)
defines its squared-error loss with combined L1 and L2 penalties and
distinguishes coordinatewise elastic net from a shared-support multitask
penalty. The explicit objective below is the selected proposal. The reference
supports the method's definition, not a claim that elastic net improves motor
LFP decoding. Earlier BR retrieval supplied no directly relevant validation
of this successor; optional advisory review is not execution authority.

## Candidate E: full physical LMP with a fixed elastic-net penalty

Candidate ID: `lmp_full_channel_elastic_net_v1`.

Keep all sixteen original M1 sessions, original complete trials and physical
LMP electrodes, the native current-bin window 0:+14, and both native XY
coordinates. Retain the original eight held-direction fits, equal-direction
first-harmonic baseline, shared fitting-residual intercept, exact within-
direction wrong-trial score, pooled native XY SSE denominator, outer-session
selector, and equal-session then equal-animal aggregation. Do not read spikes,
change the movement baseline, select a window or add a power band.

For a held direction h, construct every fitted quantity using only the other
seven directions. Let B be the unchanged cyclic baseline plus its common
fitting-residual intercept, and let U be the resulting centered native XY
residual, flattened to n trial-time rows by 2 coordinates. Let Z be the full
physical LMP matrix after the original fitting mean/population-SD scaling;
exactly zero input SD becomes one. Evaluation inputs use those same fitting
parameters. There is no channel selection, whitening, PCA or target rotation.

Define a **single common fitting-only velocity scale**

`s = sqrt(sum(U * U) / (2*n))`.

This scalar sets the units of the L1 coefficient penalty. It does not divide
the observed targets, weight XY differently, alter the native prediction or
change the score. Both coordinates retain their native equal-weight loss.
Under a common change of velocity units, s and the fitted coefficient matrix
change by the same scale and the predictions change consistently. Separate
X/Y scales are forbidden. Save s for every fold, without using it to exclude
folds or alter the grid.

For each original `lambda in [0.001, 0.01, 0.1, 1, 10]`, fit the E-by-2 matrix
W by minimizing

`F(W) = ||U - ZW||_F^2/(2*n) + lambda*||W||_F^2/2
        + lambda*s*sum_e,k(abs(W[e,k]))`.

The L2 term is exactly the full-LMP ridge term at the same lambda; the only
new term is the fixed L1 contribution. For s>0, expressing coefficients and
responses in units of this one common s gives the standard elastic-net
objective with `alpha=2*lambda` and `l1_ratio=0.5`. This algebraic equivalence
defines the fixed mixing choice; no mixing-ratio grid or alpha-max path is
selected from outcomes. The implemented native objective remains as written.

Choose **coordinatewise L1**, equivalent to two scalar-response problems with
the same Z, lambda and s. X and Y may have different coefficient supports.
Do not substitute a row-group/multitask L21 penalty, which would impose a
different shared-support assumption. Native XY axes stay fixed throughout;
coordinatewise L1 is not invariant to arbitrary output rotations, and no
axis-invariance claim is made. The common scale preserves equal-coordinate
loss but does not erase this penalty property.

The predictor is `B + Z_eval W`. Do not refit an extra intercept. Select each
outer session's lambda using only the other fifteen candidate banks with the
original equal-animal selection and larger-lambda tie rule. No fitting or
selection object receives held-direction values. The same model recipe and
five candidate penalties apply to every session.

Zero coefficients are valid: an all-zero fit predicts exactly B. If s=0,
the centered U is zero and the unique ridge-regularized solution is W=0;
retain the entire fold and its ordinary score. Do not reduce the penalty,
change the feature count or drop a zero fit to obtain a gain.

## Fixed numerical solution and failure rule

Use float64 and the native sufficient statistics `G=Z'Z/n`, `C=Z'U/n`.
Every lambda is strictly positive, so the objective is strongly convex and
has a unique minimizer. A cyclic coordinate-descent implementation can solve
the two scalar responses with the update

`W[e,k] = soft_threshold(C[e,k] - sum_j!=e G[e,j]*W[j,k], lambda*s)
          / (G[e,e] + lambda)`.

Visit physical electrodes in canonical source-ID order and X then Y; no
random coordinate schedule. Start the lambda=10 fit at zero, then warm-start
each smaller original lambda from the completed fit at the next larger
lambda within that same fold and arm. Warm starts cannot cross sessions,
outer directions or control seeds. A zero feature still has the positive
lambda denominator and must remain available.

After each complete sweep, recompute the gradient from the original G and C,
not only an incrementally maintained residual:

`g = G W - C + lambda W`.

For a nonzero coefficient, the KKT residual is
`abs(g[e,k] + lambda*s*sign(W[e,k]))`; at exactly zero it is
`max(abs(g[e,k]) - lambda*s, 0)`.
Stop only when the maximum residual is at most
`1e-8 * max(1, max(abs(C)))`. The bound is computed separately from each
arm's own C and is a numerical convergence criterion, not a scientific
selection threshold. Record the maximum and Frobenius norm of the minimal
subgradient residual, tolerance and completed sweep count for every fit.
Strong convexity gives the additional diagnostic objective-error upper bound
`||residual||_F^2/(2*lambda)`; record it without selecting a fit from it.

The fixed cap is **10,000 full sweeps per lambda and arm**, counting a sweep
over both coordinates as one. The terminal KKT certificate is required even
if coefficient changes are tiny or a library reports convergence. Nonfinite
statistics, arithmetic failure, an exceeded sweep cap or an exceeded selected
process walltime makes the attempt unresolved. Do not publish an unconverged
partial bank as scientific completion, omit that lambda/seed/fold, enlarge
the cap, loosen the tolerance or choose a solver from visible scores. A
diagnosed implementation defect may be repaired under the existing policy;
this document authorizes no automatic retry or resource extension.

## S: whole-training-trial LMP correspondence control

Control ID: `lmp_full_channel_elastic_net_training_labels8_v1`.
Fixed replicates 0..7 constitute one mean-eight control comparison.

For each session, outer held direction h, replicate r and fitting direction
d, sort the original fitting trial IDs numerically. Seed Python
`random.Random().seed(literal, version=2)` with

`ep05-lmp-elastic-net-training-label-v1|20261004|<session>|<h>|<r>|<d>`.

Shuffle a fresh canonical donor-ID list once, align it to the sorted recipient
IDs, and map back to the original fitting order. Accept fixed points, identity
permutations and chance alignments; do not redraw. Move each donor trial's
**entire 15-by-E standardized physical LMP trajectory together**. Never shuffle
bins, individual electrodes, coordinate targets or directions separately.
Save exact recipient/donor original IDs and seed literals. The held direction
does not enter the maps or any fitted numerical quantity.

Compute the original physical normalization first. Permute those already
standardized whole fitting trials to obtain Z_r, and fit the control against
the original true U with the same lambda, s and solver rules. The control
cross-covariance is `C_r = Z_r' U/n`; its Gram is algebraically the same G
because complete rows are merely reordered. Reuse the exact candidate G so
roundoff from accumulation order cannot introduce a different input metric.
Evaluation uses the original correctly paired physical LMP and true native
XY, with the control coefficients. B and its intercept remain unchanged.

The control therefore preserves fitting trial counts, per-direction LMP
trajectory distributions, all input covariances, feature width, target
distribution, regularization opportunity and physical scaling while breaking
training-trial correspondence. It preserves direction-level structure and
does not create a decoder with no possible task information. Sparsity,
coefficients, fitted signal energy and realized model complexity may still
differ; equal G is not proof of equal effective capacity or a causal mechanism.

Every control uses the candidate's selected outer-session lambda. Average
eight correct SSEs and eight exact wrong-trial SSEs with the common session
denominator. Do not average predictions or select a favorable seed. This is
separate from the unchanged evaluation-pairing check applied with each fitted
model fixed.

## Comparisons and interpretation

Report candidate, mean control, candidate-minus-control and candidate-minus-
full-LMP ridge. Reuse the preserved full-channel five-lambda banks for both
independently selected and candidate-matched lambda anchor comparisons; no
new anchor fit or scientific-trial charge is needed. The matched anchor has
the identical physical inputs and identical L2 coefficient. Its absent L1
term is the intended difference, not a claim of equal realized complexity.

The prospective training-correspondence screen requires strictly positive
candidate-minus-control primary and identity contrasts in both animals, and
primary-positive session majorities of at least 4/6 Mihili and 6/10 Chewie.
Ties fail. An improvement claim over full-channel ridge additionally requires
the same directional screen against both anchor selections. Keep every
session, animal summary, all eight control-seed summaries and all fixed-
prediction session omissions. None of these screens replaces the unchanged
0.01 primary / 0.005 identity practical gates, required falsifiers or future
session requirement.

Retain each fold/arm/penalty's common scale, convergence diagnostics, exact
zero-coefficient count, X/Y active counts, active-channel union/intersection,
and coefficient L1/Frobenius norms. An active coefficient means exactly
nonzero after soft thresholding; no later threshold changes its support.
These diagnostics do not choose penalties, seeds, channels, exclusions or
new thresholds. Do not claim that an electrode is biologically irrelevant
because its fitted coefficient is zero in a correlated-input model.

A positive result would support this fixed conditional regularizer on the
exposed corpus, subject to every existing control and influence limit. A
negative result limits this one mixing rule and grid; it does not establish
that every elastic-net or sparse model fails. Neither outcome establishes
causality, spike-free LMP, online decoding, cross-session coefficient transfer
or new-animal generalization.

## Bounded implementation, resource pilot and accounting

After adoption, preserve earlier tools and use new core/runner files. Reuse
the current-bin LMP reader, baseline, physical scaling, exact wrong-trial
score and outer selector. Original trial/order/baseline/target replays remain
required. No extra source interval, source copy or directory scan is proposed.

Focused compute fixtures should check the explicit native objective and KKT
certificate against an independently solved small correlated example; the
orthogonal-design soft-threshold closed form; equivalence to the original
ridge equation when the L1 coefficient is set to zero **in a fixture only**;
common velocity-unit scaling; zero target/zero-feature/all-zero coefficients;
fitting-only normalization and s; deterministic whole-trial original-ID maps,
including an accepted identity permutation; identical Gram and genuine
cross-covariance changes under the control; true held targets remaining unused
until scoring; mean-eight SSE and outer-session lambda exclusion; and the
explicit nonconvergence failure path. These check concrete new mathematics,
not a generic infrastructure sequence.

The full panel has 128 baseline/statistic constructions, 1,152 candidate-or-
control covariance pairs, and 5,760 two-response penalized fits (11,520 scalar
response fits). Iteration count is data dependent but bounded; previous SVD
timings do not measure its cost. The source/prediction support matches prior
full current-bin eight-control panels. Propose the predetermined complete
Chewie_20150313 pilot on one CPU within 4 GiB, with 270 seconds internal and
300 seconds external after review and scientific fixtures. Release only its
resource/support/convergence facts before choosing a finite remaining-panel
bound. Do not use pilot scores to alter the penalty or abandon a valid
negative recipe. Root owns the allocation and any subsequent selection.

After the complete pilot's resource-only review, root selected the remaining
fifteen sessions as four fixed waves of at most four one-thread session
processes, within one verified 4-CPU/12-GiB allocation. The aggregate bounds
are 1,260 seconds internal and 1,290 seconds external; every child retains
270/300-second bounds. Four complete 300-second wave reserves plus 60 seconds
for aggregation define the bound. The wrapper checks the actual CPU
allocation, explicit thread limits and remaining reserve before dispatch.
It settles every active child before starting another wave or returning a
failure, stops dispatch after any failure or aggregate alarm, and reports
completed sessions in original source order. This engineering parallelism
does not change any fit, score, data role or scientific comparison; it
authorizes no automatic retry.

At 2026-10-04 06:32 UTC, root adopted a diagnosed engineering repair after
Mihili_20140307 exhausted the unchanged internal deadline in the Python
coordinate update with source loading complete. A reviewed C99 helper now
performs one physical-order cyclic sweep per call; Python retains its
deadline check and the original NumPy KKT calculation after every sweep,
the same objective, common scale, lambda grid, warm starts and 10,000-sweep
cap. The kernel has no solver loop, convergence decision, threads, allocation
or file I/O. It is compiled only on compute with `-O3 -std=c99 -fPIC -shared
-ffp-contract=off`, without fast-math. Floating-point reduction order can
differ from BLAS; independent objective/certificate and bounded coefficient/
prediction comparisons are required, without claiming universal bitwise
identity. Original Python tools and failed artifacts remain unchanged.

The repair reuses the eight completed original session banks in place and
first repeats only the failed Mihili_20140307 session into a fresh attempt002
bank under the same 270/300-second bounds. Seven previously unlaunched
sessions may follow only after resource-only review of that native pilot.
An explicit sixteen-session source map binds the eight preserved banks and
eight native banks; no existence-based fallback, copying, completed-session
refit or outcome-based source choice is allowed. Engineering recovery adds
no scientific-trial depth.

Two complete nonduplicate panels would advance 19 to 21 valid comparisons,
with 8 controls among 15 post-initial comparisons. The eight seeds, two
coordinates, five lambdas, direction fits and anchor contrasts are not extra
scientific trials. If the fixed outputs are structurally identical to an
already evaluated model throughout all required sessions/folds, record that
identity and add zero depth; a newly obtained null fit under this distinct
penalty is otherwise a valid negative outcome. Preserve the 28–64 depth,
12-trial post-minimum patience, control fraction, 2,000 core-hour and 120-hour
ceilings, with the search clock starting 2026-10-03 22:23:02.824 UTC. No
material-improvement threshold or early-stop rule is invented here.
