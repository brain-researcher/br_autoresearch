# Training-only LMP covariance denoising and PCA-loading-label control

Prepared 2026-10-04 after thirteen complete comparisons. **Root adopted this
exact recipe and projected-covariance diagnostic clarification at00:19UTC,
after independent review and before new code/fits. Implementation is authorized;
tests and source execution remain root-owned.** This uses the existing training-only PCA
grammar. Direction-rotated targets or new direction-by-neural tensors are not
activated or treated as delegated model choices.

## Scientific question and fixed scope

Does retaining the bulk of training LMP covariance improve held-direction
velocity prediction, or does compression remove useful low-variance electrode
contrasts? The full-channel anchor retains every standardized coordinate;
the prior rank-one reduced-rank model constrains response rank rather than
denoising the input covariance. These are different hypotheses. The joint
high-band model's improvement over LMP was inconsistent across animals, so
this branch returns to LMP-only representation rather than combining further
bands or selecting different animal-specific decoders.

Keep all sixteen sessions, original whole trials and common physical LMP
electrodes; current LMP/XY offsets **0:+14**; seven fitting directions and one
held direction; the original direction/time baseline and shared fitting
residual intercept; native joint XY SSE and exact wrong-trial identity score;
and equal-session then equal-animal aggregation. No source window, target,
fold, baseline, session or practical margin changes. Reuse the unchanged
qualified LMP-only reader and post-fit original target/baseline/trial replay.

## P: one covariance-denoising rule, no rank search

Recipe ID: `lmp_pca90_ridge_v1`.

For each held-direction fit, use only the other seven directions to obtain
the original physical-electrode population means/SDs; zero SD becomes one.
Apply those same parameters to fitting and evaluation LMP. Let Z denote these
standardized fitting rows (all fifteen bins of every allowed fitting trial).
Compute the symmetric covariance `Z'Z/n` and its orthonormal eigenvectors V,
ordered by decreasing eigenvalue. This transform uses no target value or
evaluation feature. No whitening or post-PCA component standardization occurs.

Choose the smallest K whose cumulative nonnegative eigenvalues reach at least
**90% of their positive total**. This is the sole, prospectively fixed retained-
variance rule, not a grid or a score-selected threshold. Include any additional
eigenvectors with eigenvalue exactly equal to the cutoff eigenvalue so that
an exact tied eigenspace is not arbitrarily split. Eigenvector signs and
rotations within a fully retained tied space have no effect on isotropic ridge.
For float64 eigenvalues, set the fixed numerical bound
`tol = 64 * eps_float64 * E * max(1, max(abs(eigenvalues)))`.
An eigenvalue below `-tol` stops the attempt; remaining negative eigenvalues
are set to zero solely for this PSD covariance's retained-variance calculation.
Zero positive total also stops, without a new rank or feature fallback.

Use `Z @ V[:,:K]` for fitting and the original physical-normalized evaluation
rows projected by the same V. Subtract each arm's fitting component means and
apply those same means at evaluation; do not rescale components. Fit the same
ridge objective and lambda list `[0.001,0.01,0.1,1,10]` to the original centered
world-coordinate XY residual. K can differ across fitting folds because its
input-only rule is fixed; it is never selected by session/animal outcomes.

Each session's outer lambda uses only the other fifteen P score banks, with
the existing animal-balanced selector and larger-lambda tie rule. Report K,
eigenvalues/cumulative retained fraction and constant physical coordinates
for every held-direction fit. This branch intentionally gives more feature
energy to leading covariance directions, unlike a whitened PCA decoder.

## C: eight matched physical-label permutations of the learned projection

Recipe ID: `lmp_pca90_loading_labels8_v1`.

For each session and replicate0..7, start with source positions `[0,...,E-1]`
in the original ascending physical-electrode list. Seed Python
`random.Random().seed(literal, version=2)` using

`ep05-lmp-pca90-loading-label-v1|20261004|<session>|<replicate>`.

Perform one ordinary uniform shuffle, accepting chance fixed points and
identity. Record every destination-to-source position map and its physical
channel identities. The same map is used across all eight held-direction
fits, all trials, all times and all lambdas for that session/replicate.

First fit P's physical means/SDs and PCA projection using the permitted fitting
rows. **Keep that learned V and K fixed.** For map p, replace the projection
with `V_control[e,j] = V[p[e],j]`, then project the original physically
standardized feature rows through V_control in fitting and evaluation.
Neither LMP trial labels nor XY values move. Recenter the control's fitting
components without rescaling, and fit its own ridge map. A row permutation
preserves orthonormal projection columns and nominal K, but generally changes
the projected covariance, so each arm needs its own Gram/eigensystem.

For P and every control/held-direction fit, retain the projected fitting
covariance eigenvalues and trace. Report numerical rank using
`tol_projected = 64 * eps_float64 * K * max(1, max(abs(projected_eigenvalues)))`
and counting eigenvalues strictly above that fixed bound. Also retain ridge
effective degrees of freedom `sum(mu/(mu+lambda))` at each original lambda,
where mu are the projected covariance eigenvalues with roundoff negatives
treated as zero; summarize at the shared selected lambda without using this
diagnostic for selection. A materially negative eigenvalue is a numerical
failure under the same stated convention. Rank-deficient or zero-energy
controls remain valid ridge (possibly baseline-only) controls. Do not redraw,
refit PCA, change K, replace the arm or omit a session because its effective
rank/variance differs. Nominal K can exceed the control's effective rank.

Do not refit PCA on permuted feature coordinates: that would merely relabel
the original PCA model and produce an algebraically invariant control. Likewise
a full-span unwhitened orthogonal projection is equivalent to full-channel
ridge. Report any full-span fitting fold as an uninformative label-control
fold, without dropping it or inventing a replacement. If every fold is full
span, this is structural duplication of the anchor, not new scientific depth;
do not charge candidate/control duplicates toward the minimum.

For each outer session use P's selected lambda for P and all eight C refits.
Retain each seed's correct/wrong SSE, then average SSEs with the original
denominator; never ensemble predictions or select a favorable seed. One
eight-seed aggregate is one control comparison.

This tests whether a truncated covariance projection helps because its
loadings are aligned with the actual physical channels. The control retains
physical normalization information and changes retained signal variance and
effective covariance. It is matched in nominal dimension and fitting recipe,
not every effective degree of freedom, and does not establish absence of
neural information or a pure causal electrode-label effect.

## Contrasts and proposed interpretation gates

Report P and mean C, P-minus-C, and P minus the preserved full-channel anchor
under both its independently selected lambda and P's matched lambda. Reuse
the original five-lambda anchor banks, without refitting or charging them.
The P-minus-C directional screen requires both animal means of both primary
and identity contrasts to be strictly positive, plus primary-positive
majorities of at least4/6 Mihili and6/10 Chewie. Ties fail.

A claim that denoising improves on the anchor additionally requires the same
directional conditions for both P-minus-anchor reference selections. These
are prospective directional screens, not significance tests or new practical
margins. Promotion still requires P's unchanged **0.01 primary / 0.005 identity**
gates and every outstanding control/search requirement. Success of the earlier
stationary model's timing control does not automatically cover a new finalist.
Failure may mean covariance-dominant dimensions omit useful residual signal;
it does not reject all learned denoising or trial-specific LFP information.

## Implementation boundary, cost and advisory limitation

Preserve existing science. Before any source/fits, root must adopt the exact
recipe and its stated numerical covariance convention. Focused
fixtures should verify fitting-only scaling/PCA/K, the90% cutoff and full tied
space rule, no whitening, full-span equivalence, fixed projection-label maps
without PCA refit, unchanged target/baseline, separate control Grams, seed-SSE
aggregation and shared outer lambda/anchor contrasts. No new code or source
access is authorized by this proposal.

If adopted, propose the complete predetermined Chewie_20150313 pilot, both
arms/all eight controls, provisionally1CPU/4GiB/five minutes with270s internal
and300s external bounds. The matrices are at most96×96 before projection and
K×K afterward; this is a modest design size, not measured runtime. Root selects
the actual allocation and sizes the remaining fifteen-session run only from
completed pilot cost. Preserve every original session, all seeds and failed
attempts; no score-based abandonment, retry or automatic renewal.

Two distinct complete recipes would advance thirteen to fifteen comparisons,
with five controls among nine post-initial comparisons. No rank, eigenvector,
fold, seed, reference contrast or infrastructure attempt is a separate trial.
The 28–64, patience, adaptive/challenger and remaining-control rules persist.

Root's optional BR `kg_search_nodes` lookup for partial-least-squares motor
decoding returned only four fMRI/gray-matter/visuomotor PLS publication titles
(DOIs10.1016/j.neuroimage.2004.05.018,10.1016/j.cortex.2016.04.011,
10.1016/j.neuroimage.2005.09.010,10.1002/hbm.20664), without directly relevant
LFP evidence or method fields. The initial quoted Boolean request produced a
parser ParseException; one plain-language correction succeeded. This was not
a KG outage or validation of this branch. No retry or scientific endorsement
is inferred; the proposal follows the allowed grammar and observed model
limitations independently of that advisory retrieval.
