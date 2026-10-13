# Can a distributed training-target subspace preserve useful LMP information?

Prepared 2026-10-04 after the complete seventeenth primary comparison was
released. **Prospective proposal only: not adopted, implemented or run.**
No new source read, fit, ASTRA execution recipe or scientific-trial charge is
authorized by this document. Existing margins, source support and budgets stay
fixed. Root owns any adoption, validation and bounded execution.

## Motivation and the distinct question

The complete [top16 relevance result](runs/adaptive_search/primary_relevance_summary_attempt001/primary_relevance.json)
has primary gain0.000565575 and identity gain0.001844182, below both practical
margins. Its primary gain is negative in Mihili. The primary increment over
its rank-label control is0.000118282 overall but negative in Mihili, while the
identity contrast is positive in both animals. It loses to full LMP in both
animals under independently chosen and matched penalties; all fixed-omission
contrasts against full LMP remain negative. These are released coordinator
observations; this draft did not open partial banks or recompute results.

Earlier covariance-PCA compression also lost to full LMP. Neither result
establishes that all compact representations fail. PCA selects input variance;
top16 selection retains physical coordinates with individually reproducible
prediction. A useful distributed combination can have modest input variance
and require electrodes that are weak individually. Conversely, neither failure
guarantees such a distributed combination exists.

The next question is whether an input subspace selected by its **training
covariance with native XY residuals** can preserve a useful distributed signal.
This uses the operative train-only PLS feature-transform and ridge grammar.
The [official PLSSVD definition](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSSVD.html)
describes an SVD of the input–target cross-covariance with input projection on
its left singular vectors. The explicit equations below define this proposed
covariance-SVD transform, including its null-rank handling and separate ridge
head. No iterative PLS regression, target whitening, component search or new
software installation is implied. The documentation establishes the method's
definition, not evidence that it improves motor-LFP decoding. The earlier BR
PLS retrieval supplied no directly relevant motor-LFP support or endorsement.

## Fixed scope and candidate P

Candidate ID: `lmp_target_covariance_svd_ridge_v1`.

Keep all sixteen original M1 sessions, both animals, complete original trials,
physical LMP coordinates and native XY at offsets0:+14. Retain the original
outer held-direction folds, equal-direction first-harmonic direction/time
baseline, shared fitting-residual intercept, five ridge penalties, native
pooled-XY SSE, exact within-direction wrong-trial evaluation and equal-session
then equal-animal aggregation. No power-band, time-window, direction-by-neural
term, spike read or target change is introduced.

For outer held direction h, use only the other seven directions to construct
every quantity below:

1. Fit the unchanged cyclic baseline B0. Let c be the pooled mean of its
   fitting residuals, and let U be the native two-coordinate fitting residual
   after subtracting c. The baseline for fitting and evaluation is B=B0+c.
2. Compute the original physical LMP fitting means and population SDs across
   trials and fifteen bins; replace zero SD by one. Z is the resulting fitting
   matrix with n trial-time rows and E physical columns. Keep all original
   columns. Evaluation features use exactly those fitting parameters.
3. Compute the float64 cross-covariance **C = Z' U / n**, of shape E-by-2.
   Do not residualize neural direction means, scale/whiten native XY, weight
   directions again, or let held-direction values enter C.
4. Take a thin SVD of C. Retain **every singular direction whose singular
   value is strictly greater than**
   `64 * eps_float64 * max(E,2) * max(1, largest_singular_value)`.
   There are at most two retained directions, with no component-count grid.
   This is a fixed numerical-zero rule, not an outcome threshold. Save both
   singular values, the tolerance, retained rank K and E-by-K orthonormal
   input basis W. All nonzero directions at an exact tie are retained. Sign
   choices and rotations within an equal-singular-value retained subspace
   must not change predictions under the isotropic ridge head.
5. Project training and evaluation features as ZW. Subtract each projected
   coordinate's fitting mean, retaining it for evaluation, and **do not
   divide by projected SD, whiten scores or multiply scores by singular
   values**. Fit the original native-residual ridge head with normal equations
   `(H'H + n*lambda*I) beta = H'U` at each original penalty. Add B back to
   predict full native velocity.

The outer session's lambda is selected from the other fifteen candidate
session banks using the original equal-animal selector and larger-lambda tie
rule. No evaluated session chooses its own feature rule or component count.
Both the supervised basis and its head may use the same seven fitting
directions: all evaluation directions remain outside both fits. The basis
does not receive evaluation trials through tuning or diagnostics.

K=0 or1 remains a valid result of the fixed transform. At K=0, the neural
correction is identically zero and every penalty returns B. Do not invent an
arbitrary null-space vector, pad to two coordinates, select another seed,
drop the fold or replace the model. The original target support and scoring
must remain complete. Nonfinite inputs or a failed factorization are genuine
unresolved attempts, not permission to change this rule.

This subspace is the span of the two native input–residual cross-covariance
vectors after numerical-null removal. It is a strong, explicit restriction;
it does not span all supervised information or all solutions of iterative
PLS. Full ridge generally has coefficient directions proportional to
`(G+lambda*I)^(-1) C`, which need not remain in the span of C. Thus this is not
a disguised repeat of full-channel ridge. It can discard covariance-adjusted
directions that full ridge needs. A failure would reject this fixed compact
transform, not distributed prediction in general.

## S: basis-only training-trial correspondence control

Control ID: `lmp_target_covariance_svd_basis_labels8_v1`.
Fixed replicate labels0..7 form one mean-eight control comparison.

For every session, outer held direction h, replicate r and each of the seven
fitting directions d, sort the original fitting trial IDs numerically. Seed
Python `random.Random().seed(literal, version=2)` with

`ep05-lmp-covariance-svd-basis-label-v1|20261004|<session>|<h>|<r>|<d>`.

Shuffle a fresh canonical donor-ID list once and align it with the sorted
recipient list, then map back to the original fitting order. Accept identity,
fixed points and chance alignments. Each source trial's entire15-by-2 U block
moves together; never shuffle bins, XY coordinates or directions separately.
The held direction is absent from the map and every numerical fitting input.
Reuse each map at all five penalties and save exact recipient/donor original
IDs and seed literals. No redraw, derangement or favorable-seed selection.

Compute `C_r = Z' U_permuted / n`, and fit that control's own covariance-SVD
basis with the **same** singular-value rule. Original physical scaling,
baseline, residual intercept and Z remain fixed. Project Z through this
control basis, center its projected training features, and fit its own ridge
head against **true unpermuted U**. Evaluate against true original XY. Only
the correspondence used to learn the input basis is randomized; target signal
remains available to the downstream head.

Each control has its own projected Gram and fitted coefficients. It shares
the candidate's outer-session-selected lambda. Average eight correct SSEs
and eight wrong-trial SSEs with the unchanged common denominator; never
average predictions or select the strongest control draw. Ordinary
within-direction permutations preserve conditional means and the full
multivariate trajectory distribution. They randomize residual trial
correspondence while retaining any direction-level residual structure.

Candidate and control have the same nominal component cap, fitting support,
normalization, algorithm, ridge grid and final true targets. Their realized
K, covariance, signal energy and effective ridge degrees of freedom can
differ; report those differences. Supervised basis estimation also consumes
training information, so a two-column head is not proof of only two total
estimated degrees of freedom. This is not a pure causal intervention on
neural information or a no-signal decoder.

## Comparisons, diagnostics and interpretation

Report candidate, mean control, candidate-minus-control, and candidate minus
the preserved full-LMP anchor under both independently selected and
candidate-matched lambda. Reuse the original full-channel five-lambda banks;
do not refit or recharge the anchor. Retain each session, both animal means,
all eight seed summaries and all fixed-prediction session omissions.

The prospective basis-correspondence screen requires strictly positive
candidate-minus-control primary and identity contrasts in **both** animals
and primary-positive majorities of at least4/6 Mihili and6/10 Chewie. Ties
fail. A claim that this compression improves on full LMP additionally requires
the same directional screen against both anchor selections. These screens
do not replace the unchanged **0.01 primary / 0.005 identity** practical
gates, other required controls or the absent future-session evaluation.

For every fold and arm retain singular values/tolerance/K and the projected
covariance eigenvalues, trace, numerical rank and effective ridge df at the
selected penalty. Use the already declared projected-covariance diagnostic
tolerance `64 * eps_float64 * K * max(1,max(abs(eigenvalues)))`; for K=0 the
spectrum is empty and trace/rank/df are zero. Covariance diagnostics do not
change the fitted subspace, rank, penalty or scoring. Report
candidate/control projector overlap `trace(W W' W_r W_r')` as a raw value
between zero and the smaller retained rank, alongside both actual ranks;
do not divide by zero or call unequal-rank overlaps equivalent.

These diagnostics can distinguish a correspondence contrast accompanied by
large capacity/energy changes from one with similar realized projection
properties. They do not identify a biological mechanism. A positive result
would support this fixed training-target input subspace in the exposed
corpus, conditional on its controls and influence limitations. It would not
establish spike-independent physiology, causal velocity coding, online
decoding or new-animal generalization. A negative result would limit this
specific covariance-SVD representation without proving that more flexible
PLS or distributed features cannot work.

## Finite implementation, resources and accounting

After adoption, preserve earlier files and create only new core/runner and
focused scientific fixtures. Reuse the original current-bin LMP reader,
baseline, physical scaling, ridge head, exact wrong-trial scoring and outer
selector. All post-fit original baseline/target/trial replays remain required.

Fixtures should cover fitting-only C and physical scaling, agreement with
the direct cross-covariance SVD, orthogonal/sign invariance of projected ridge,
the explicit zero/rank-one cases, ordinary original-ID whole-trial maps,
true-U heads for both arms, distinct projected Grams, untouched held
direction targets, mean seed SSE, and shared outer lambda. No source-size
scan, generic infrastructure gate or new copy of the source is proposed.

Each session requires8 outer baselines,72 E-by-2 cross-covariance SVDs and
360 at-most-two-coordinate ridge fits, plus the existing source/scoring work.
The sixteen-session panel therefore has1,152 such SVDs and5,760 ridge fits.
Source support and prediction-array scale match the already completed
current-bin eight-control families; no new source interval is read.

The top16 panel's released resource observations were50.381seconds for the
predetermined pilot and131.428seconds for the remaining fifteen sessions.
This supports feasibility but does not measure the new recipe. Propose the
same complete Chewie_20150313 resource pilot on1CPU within4GiB,270seconds
internal/300seconds external after review and scientific fixtures. Root may
choose a suitable task-owned allocation and must select any remaining-panel
bound from resource facts while pilot scientific scores stay closed. No
automatic retry, omitted seed/session, score-based budget change or generic
indefinite optimization loop is authorized.

Two complete fixed panels would advance17to19 valid comparisons, with7
controls among13 post-initial comparisons. The8 seeds,128 outer direction
fits, component directions and anchor contrasts are not additional trials.
If both transforms are structurally identical to an already evaluated model
throughout every required session/fold, record that identity and add zero
depth; do not count a duplicate as scientific coverage. Otherwise a complete
null or negative result still counts under the fixed recipe. Preserve the
28–64 minimum/maximum, patience, control fraction and shared resource budget.
The study cannot stop early because this next comparison happens to pass.
