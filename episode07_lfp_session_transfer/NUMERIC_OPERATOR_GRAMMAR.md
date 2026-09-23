# EP07 full numeric operator grammar v2

Contract schema ID: `ep07.numeric_operator_grammar.v2`.

Status: **scientist approved retaining the full operator families on
2026-09-22.** This declarative contract does not by itself authorize
development-outcome access; episode-bootstrap validation, runtime artifacts,
the full synthetic dry profile, and execution qualification remain pending.

## 1. Frozen invariants

- Direction assignment uses the nearest circular multiple of `pi/4`, with
  maximum error `0.01` radians.
- One regression row is one native 30-ms bin within one whole trial. Every
  split and cross-validation fold keeps all bins from a trial together.
- Configuration choices are global across animals, target days, and strata.
  Fitted parameters may differ by target day and stratum.
- Transfer fitting may use the three same-animal/source-region source days and
  the current target day's calibration role only.
- Calibration-only fitting may use the current target day's calibration role
  only.
- No input transform, latent transform, nuisance template, meta-weight, or
  model parameter may use development or audit outcomes.
- Predictions are finite `float64` values reconstructed onto the target
  neuron's native spike-count scale. They are never clipped, rounded,
  rectified, smoothed, or variance-normalized before scoring. Negative
  predictions remain valid predictions.
- Numeric electrode identity is never matched across days.

## 2. Canonical row and state construction

For target day `t`, stratum `k`, trial `i`, native bin `q`, feature class `c`,
and electrode `e`, let `x[i,q,c,e]` be the authenticated released LFP feature
value. Let `y[i,q,j]` be the native spike count for frozen target neuron `j`.

The transfer-side permitted data universe is:

```text
F_T(t,k) = three source-only days of the same animal and region
           union current target-day calibration trials
```

The calibration-only permitted data universe is:

```text
F_C(t,k) = target-day calibration trials
```

All fit-state objects are target-day- and stratum-specific. The declarative
configuration is nevertheless identical for every target day and stratum.
`F_T` is an access ceiling, not permission to pool source and target responses
in an arbitrary supervised fit. Operator-specific roles are fixed as follows:

| State | Source-bearing transfer | Calibration-only / transfer `zero_prior_fit` |
| --- | --- | --- |
| LFP aggregation edges, scaling, and set-basis PCA | source predictor rows plus current target-calibration predictor rows; fold-local source rows during source OOF fits | current target-calibration predictor rows only |
| source spike PCA and source nuisance response templates | that source day's response rows only; fold-local during source OOF fits | absent |
| target spike PCA and target nuisance response templates | current target-calibration responses only | current target-calibration responses only |
| cross-day alignment | fold-permitted source response anchors plus fixed target-calibration response anchors | absent |
| each source base mapping | that source day's response rows only; target-calibration labels prohibited | absent |
| source pooling/meta-weights | source-only OOF responses, plus target-calibration predictor/anchor similarity only where declared | absent |
| source-prior or convex target adaptation | frozen source prediction plus current target-calibration responses exactly as in Section 11 | target-calibration responses only |

Development predictor rows may be used only to emit predictions and to check
the label-free ensemble-diversity rule. They never fit an aggregation edge,
scaler, PCA, template, alignment, coefficient, meta-weight, or adaptation
state. Development and audit responses are prohibited everywhere above.

Class order is `LMP`, followed by the eight authenticated power-band semantic
IDs in the frozen provider-manifest order. Trial order, neuron order, and
direction order come only from their frozen manifests.

## 3. Deterministic numerical primitives

All linear algebra is `IEEE-754 float64` in the pinned single-thread runtime.

For a symmetric matrix, first replace it by `(A + A.T)/2`. Eigenvalues are
ordered descending. Adjacent eigenvalues belong to one degenerate block when

```text
abs(lambda[a] - lambda[b]) <= 1e-12 * max(1, abs(lambda[0]))
```

A degenerate eigenspace is canonicalized from its projector: scan ordinary
coordinate axes in ascending index, project each axis into the eigenspace,
apply two-pass modified Gram-Schmidt against already accepted vectors, and
accept vectors with norm greater than `1e-12` until the block dimension is
filled. For every resulting vector, make the loading with largest absolute
value positive; ties use the lowest coordinate index.

`canonical_svd` first calls the pinned full-matrix SciPy/LAPACK `gesvd`
implementation. Singular values are ordered descending. Adjacent values are a
degenerate block under the same relative `1e-12` rule above. Within each block,
canonicalize the right-singular subspace from its projector using the stated
coordinate-axis procedure, then construct every positive-singular-value left
vector as `A v / sigma`. Any required right null space is canonicalized from
its projector by the same procedure. This is the only SVD algorithm; an
eigendecomposition of `A.T @ A` is not an alternative implementation.

Quantiles use the linear interpolation rule. Exact objective ties are resolved
by the ascending canonical configuration hash or, inside a support-enumeration
solver, by ascending binary support mask.

## 4. LFP representation operators

The searchable feature scopes are:

```text
lmp_only
eight_power_classes
all_9_classes
```

For every row and feature class, aggregation operates on the unordered finite
electrode set. It is therefore invariant to electrode permutation.

### 4.1 `moments_quantiles`

Emit, in this order:

```text
mean, population_sd, q10, q25, q50, q75, q90
```

for each selected class, then concatenate classes in canonical class order.

### 4.2 `robust_histogram_8`

For each class, fit seven edges `q12.5, q25, ..., q87.5` from all electrode
values in that operator's permitted fit set. Emit eight per-row electrode
fractions. Bins are defined by

```text
bin = searchsorted(internal_edges, value, side="right")
```

so repeated edges produce deterministic empty bins and an edge value enters
the bin immediately to its right. Counts are divided by the authenticated
number of electrodes.

### 4.3 `low_rank_set_encoder_K`, `K in {8,16}`

For each class and row, form the 17-dimensional invariant basis:

```text
[the 7 moments/quantiles,
 the 8 robust-histogram fractions,
 log1p(number_of_electrodes),
 q90 - q10]
```

Concatenate selected classes, giving dimension `17 * number_of_classes`.
Fit a columnwise median and IQR on permitted fit rows; replace a zero IQR by
one. Fit deterministic centered PCA to the scaled basis and emit its first
`K` scores. Thus both `K=8` and `K=16` are dimensionally legal even for
`lmp_only`; the invalid seven-input/eight-component v1 case is removed. Set-PCA
legality is `K <= min(number_of_basis_columns, number_of_fit_rows)`. It does
not require `K <= matrix_rank`: exact linear dependencies create a
zero-eigenvalue block whose eigenvectors are fixed by the projector
canonicalization in Section 3 and whose emitted scores are exactly zero.

For `moments_quantiles` and `robust_histogram_8`, apply the same columnwise
median/IQR scaling after aggregation. Scaling state is fit only on `F_T` or
`F_C`, as appropriate. A nonfinite output is an invalid configuration; there
is no alternate transform.

## 5. Spike PCA and exact inverse

For each source day separately and for target calibration separately, arrange
the spike matrix as rows `(whole_trial, native_bin)` and columns in frozen
neuron order. Fit

```text
mu[j] = (1/N) * sum_i Y[i,j]
C  = (Y - mu).T @ (Y - mu) / N
```

and take the first `r` deterministic eigenvectors, where

```text
r in {4,8,12,16}
r <= min(number_of_neurons, N - 1)
```

When whitening is enabled, let eigenvalues be `lambda[l]` and define

```text
d[l] = sqrt(max(lambda[l], epsilon * lambda[0]))
epsilon in {1e-6, 1e-4}
```

`epsilon` is present only when `whiten=true`; it is absent, not silently
ignored, when `whiten=false`.

The latent coordinates are:

```text
unwhitened: Z = (Y - mu) @ V
whitened:   Z = (Y - mu) @ V @ diag(1/d)
```

The exact target inverse is:

```text
unwhitened: Yhat = mu_target + Zhat @ V_target.T
whitened:   Yhat = mu_target
                    + Zhat @ diag(d_target) @ V_target.T
```

`Yhat` is returned unchanged. In particular, there is no lower clipping at
zero.

## 6. Cross-day latent alignment

For every source day, create equally weighted direction-by-native-bin anchor
rows:

```text
A_s[d,q] = mean over source trials of Z_s[trial,q]
A_t[d,q] = mean over target-calibration trials of Z_t[trial,q]
```

There are `H = 8Q` matched rows. Let `X=A_s`, `Y=A_t`.

### 6.1 `none`

```text
M = I_r
b = 0
```

This is a registered negative-control alignment, not a claim that neuron axes
match across days.

### 6.2 `orthogonal_procrustes`

Center anchors by `xbar` and `ybar`. Compute

```text
U, S, Vt = canonical_svd((X-xbar).T @ (Y-ybar))
M = U @ Vt
b = ybar - xbar @ M
```

Reflection is allowed. The aligned source latent is `Z_s @ M + b`.

### 6.3 `regularized_cca(rho)`

`rho` is one of `{1e-4, 1e-2, 1}`. Center the anchors and compute:

```text
Sxx = Xc.T @ Xc / H
Syy = Yc.T @ Yc / H
Sxy = Xc.T @ Yc / H

ax = rho * trace(Sxx) / r
ay = rho * trace(Syy) / r
Rx = Sxx + ax I
Ry = Syy + ay I
```

Zero trace or a failed positive-definite solve makes the configuration invalid.
Using deterministic symmetric inverse square roots:

```text
T = Rx^(-1/2) @ Sxy @ Ry^(-1/2)
U, K, Vt = canonical_svd(T)
V = Vt.T
Wx = Rx^(-1/2) @ U
Wy = Ry^(-1/2) @ V
```

Form canonical anchor coordinates:

```text
Cx = Xc @ Wx
Cy = Yc @ Wy
eta = rho * trace(Cx.T @ Cx) / r
G = solve(Cx.T @ Cx + eta I, Cx.T @ Cy)
```

The complete source-PCA to target-PCA map is:

```text
M = solve(Wy.T, (Wx @ G).T).T
b = ybar - xbar @ M
Z_source_in_target_coordinates = Z_source @ M + b
```

This explicitly returns target-PCA coordinates and repairs the undefined v1
CCA back-map. The written solve uses the pinned general linear solver and no
explicit inverse or pseudoinverse.

## 7. Nuisance operators

Direction uses deterministic seven-column effect coding: directions `0..6`
have their corresponding unit entry and direction `7` is all `-1`. Native
time uses the same construction with `Q-1` columns.

Within every permitted fit day, define full direction-by-time templates
`T_X[d,q]` and `T_Z[d,q]`. Source `Z` is first aligned into target-PCA
coordinates. At target prediction time, target-calibration templates are used.

The three legal nuisance modes are:

1. `direction_time_residual`

   ```text
   design = X - T_X[d,q]
   response = Z - T_Z[d,q]
   prediction = fitted_residual + target_T_Z[d,q]
   ```

2. `additive_direction_time`

   ```text
   design = [X, direction_effect_code, time_effect_code]
   response = Z
   prediction = fitted_response
   ```

3. `combined`

   ```text
   design = [X - T_X[d,q],
             direction_effect_code,
             time_effect_code]
   response = Z
   prediction = fitted_response
   ```

All templates are refit inside every source cross-validation fold.

## 8. Mapping operators

Every mapping includes an unpenalized intercept. Let `N`, `p`, and `r` denote
the number of rows, non-intercept design columns, and latent outputs.

### 8.1 Ridge

For `lambda in {1e-4,1e-2,1,100}`:

```text
argmin_(a,B)
  ||Z - 1a - XB||_F^2 / (N r)
  + lambda ||B||_F^2 / (p r)
```

Let `Xc=X-mean(X)` and `Zc=Z-mean(Z)`. The unique implementation is

```text
B = scipy.linalg.solve(Xc.T@Xc + (N*lambda/p)*I,
                       Xc.T@Zc, assume_a="pos")
a = mean(Z) - mean(X)@B
```

in the pinned runtime. Do not use a pseudoinverse or solver fallback.

### 8.2 Elastic net

For

```text
alpha in {1e-3,1e-2,0.1}
l1_ratio in {0.1,0.5,0.9}
```

fit each latent output independently:

```text
||z_l - a_l - X b_l||_2^2 / N
+ alpha * [
    l1_ratio * ||b_l||_1 / p
    + (1-l1_ratio) * ||b_l||_2^2 / (2p)
  ]
```

For each output, call the pinned scikit-learn `ElasticNet` implementation with
`alpha_solver=alpha/(2p)`, the declared `l1_ratio`, `fit_intercept=true`,
`precompute=false`, `copy_X=true`, `max_iter=10000`, `tol=1e-8`,
`warm_start=false`, `positive=false`, and `selection="cyclic"`. This scaling
makes the library objective equal to the written objective up to multiplication
by the positive constant `1/2`. The pinned library's centering, convergence,
and dual-gap definitions are authoritative. A convergence warning, iteration
cap, or nonfinite coefficient is an invalid configuration; no other solver or
penalty scaling is allowed.

### 8.3 Reduced-rank ridge

For `lambda in {0.01,0.1,1}`, first fit ridge as above. Center `X` and form

```text
F = X_centered @ B_ridge
h = max(1, floor(r/2))
V_h = first h right singular vectors of canonical_svd(F)
B_rrr = B_ridge @ V_h @ V_h.T
a_rrr = mean(Z) - mean(X) @ B_rrr
```

### 8.4 Shallow MLP

Search:

```text
width in {16,32,64}
learning_rate in {3e-4,1e-3}
weight_decay in {1e-4,1e-3,1e-2}
```

The model has exactly two affine layers:

```text
h = GELU(X W1 + b1)
Zhat_standardized = h W2 + b2
```

where exact GELU is `0.5*x*(1+erf(x/sqrt(2)))`. Standardize every response
coordinate by its training mean and population SD, replacing zero SD by one;
undo this standardization after prediction.

For a minibatch containing row set `B` and `r` outputs, the data loss is

```text
L_B = sum_(i in B,l) (Z_standardized[i,l] - Zhat[i,l])^2 / (|B| r)
```

Backpropagate this exact mean loss in float64. The GELU derivative is
`Phi(x) + x*phi(x)`, evaluated with the pinned SciPy special functions.

Initialize from a fresh PCG64DXSM generator using seed component
`mlp_weight_initialization`. Draw `W1` then `W2` in C row-major order from
NumPy's half-open uniform distribution with bounds
`sqrt(6/(p+width))` and `sqrt(6/(width+r))`, respectively; initialize both
biases to exact zero. Run exactly 300 epochs of AdamW with:

```text
beta1=0.9, beta2=0.999, epsilon=1e-8
whole-trial batch size=16
global gradient-norm clip=5
constant learning rate
no early stopping
```

For zero-based epoch `e`, instantiate a fresh PCG64DXSM generator from seed
component `mlp_epoch_permutation` and `epoch=e`. Permute the canonical whole-
trial list once, take consecutive groups of 16 trials, retain the final smaller
group, and concatenate every selected trial's native bins in ascending `q`
order. Thus a whole trial never crosses batches. The Adam step counter starts
at one and advances once per minibatch across epoch boundaries.

Initialize first and second moments to zero for every parameter. Compute the
single global L2 norm across all four data-gradient arrays. If it exceeds five,
multiply every gradient by `5/norm` before moment updates. At step `h`, use

```text
m_h = beta1*m_(h-1) + (1-beta1)*g_h
v_h = beta2*v_(h-1) + (1-beta2)*(g_h*g_h)
mhat = m_h / (1-beta1^h)
vhat = v_h / (1-beta2^h)

W <- (1 - learning_rate*weight_decay)*W
     - learning_rate*mhat/(sqrt(vhat)+epsilon)
b <- b - learning_rate*mhat/(sqrt(vhat)+epsilon)
```

where the last two lines use the moments corresponding to that parameter.
Only `W1` and `W2` receive decoupled weight decay; biases do not. Operations
occur in the written order with no fused or foreach kernel. Any nonfinite loss,
gradient, moment, parameter, or prediction is invalid; there is no retry with a
different seed or optimizer.

## 9. Inner source cross-validation

All meta-estimation is nested wholly inside source-only data. Development
evaluation is the outer held-out evaluation and never participates here.

Within each source day and direction, sort whole trials by:

```text
SHA256("ep07-inner-v2\0" || policy_hash || config_hash ||
       animal || source_date || direction_id || authenticated_trial_id)
```

Assign sorted trials round-robin to folds `0..4`. All bins of a trial share its
fold. Every representation state, source PCA, alignment anchor, nuisance
template, and base mapping that could use a held-out source response is refit
inside that fold. Target-calibration state may remain fixed because it is an
authorized fit role. Fewer than five source trials in any direction is a
structural readiness failure.

## 10. Source pooling

Let `p_s(x)` be the full target-latent prediction from source day `s`, including
the applicable target nuisance restoration.

### 10.1 Equal session

```text
p(x) = (p_1(x) + p_2(x) + p_3(x)) / 3
```

### 10.2 Reliability weighted

Generate five-fold source out-of-fold predictions and aligned source latent
targets. For each source day:

```text
R_s = 1 - SSE_s / sum_i ||z_s[i] - mean(z_s)||^2
```

A zero denominator invalidates this pooling configuration. For temperature
`tau in {0.05,0.2}`:

```text
w_s = exp((R_s - max_u R_u)/tau)
      / sum_v exp((R_v - max_u R_u)/tau)
```

and `p(x)=sum_s w_s p_s(x)`.

### 10.3 Calibration similarity

For each source day, use its aligned anchor and the target-calibration anchor:

```text
D_s = ||A_s M_s + b_s - A_t||_F^2 / (H r)
```

For `tau in {0.05,0.2}`:

```text
w_s = exp(-(D_s - min_u D_u)/tau)
      / sum_v exp(-(D_v - min_u D_u)/tau)
```

and `p(x)=sum_s w_s p_s(x)`.

### 10.4 Hierarchical shrinkage

This is legal only with ridge or reduced-rank ridge. Let `D_s=[1, X_s]`, day
coefficients be `Theta_s`, global coefficients `Theta_0`, `P` select
non-intercept rows, and `lambda_h in {0.1,1,10}`. Solve the single convex
objective below. `lambda_map` is exactly the selected ridge `lambda`; for a
reduced-rank-ridge configuration it is that family's selected first-stage
ridge `lambda`.

```text
(1/3) sum_s [
  ||Z_s - D_s Theta_s||_F^2 / (N_s r)
  + lambda_map ||P Theta_s||_F^2 / (p r)
]
+ lambda_h * sum_s ||Theta_s - Theta_0||_F^2
    / (3 (p+1) r)
+ lambda_map ||P Theta_0||_F^2 / (p r)
```

The target prior is `D_target Theta_0`.

For `direction_time_residual`, `D_target Theta_0` is a residual prediction and
the target-calibration `T_Z[d,q]` is added before adaptation or inversion. The
`combined` and `additive_direction_time` designs already emit full latent
predictions and receive no additional restoration.

For reduced-rank ridge, solve this objective first, form the equal-session
weighted stacked global fitted matrix

```text
vertcat_s(D_s Theta_0 / sqrt(3 N_s)),
```

take its first `floor(r/2)` right singular vectors, and right-project
`Theta_0` onto that output subspace.

### 10.5 Leave-one-source stacking

For each of the five inner folds, refit every source-day expert after excluding
that fold from every source day, then predict the excluded rows from all three
experts. Thus no meta-response was used by any prediction supplied to the
stacking fit. A `direction_time_residual` OOF expert prediction restores the
held-out row's fold-fitted, aligned source-day `T_Z[d,q]`; prediction on a
target row restores the target-calibration `T_Z[d,q]`.

For `eta in {0.01,0.1,1}`, solve:

```text
minimize over w >= 0, sum_s w_s = 1:
  (1/3) sum_day [
    mean_rows_outputs ||Z - sum_s w_s P_s||^2
  ] + eta ||w - (1/3,1/3,1/3)||_2^2
```

Enumerate all seven nonempty supports. On each support solve the
equality-constrained quadratic exactly, discard negative-weight solutions, and
select minimum objective followed by ascending support mask. Refit all three
experts on complete source data and apply the frozen weights to target rows.

## 11. Target adaptation

Let `p_S(x)` be the pooled source prediction in target-PCA coordinates.

### 11.1 `zero_prior_fit`

Fit the declared representation, nuisance, and mapping using target
calibration only. No source state is constructed. In the transfer stream this
is an explicit source-free control and all alignment and pooling fields must
be `null`.

### 11.2 `source_prior_shrinkage`

For `lambda_A in {0.1,1,10}`, compute source predictions `P_S` on target
calibration and solve:

```text
argmin_(A,c)
  ||Z_t - P_S A - 1c||_F^2 / (N r)
  + lambda_A (||A-I||_F^2 + ||c||_2^2) / (r^2+r)
```

Prediction is `p_S(x) A + c`.

### 11.3 `convex_source_target`

Fit an independent target-calibration model `p_T` using the same
representation, nuisance, and mapping family. For source weight
`gamma in {0.25,0.5,0.75}`:

```text
p(x) = gamma p_S(x) + (1-gamma) p_T(x)
```

## 12. Ensembles

An ensemble is a terminal configuration containing exactly two previously
valid, non-ensemble configurations from the same stream and a first-parent
weight in `{0.25,0.5,0.75}`. It introduces no refit:

```text
Yhat = alpha Yhat_parent_1 + (1-alpha) Yhat_parent_2
```

Combination occurs after each parent has independently inverted its target
PCA onto native neuron order. Therefore parents may use different latent
ranks.

Diversity uses development inputs and predictions but no development labels.
Within each of the 12 target-day/stratum cells, center each parent's flattened
native-scale prediction vector and normalize it to unit L2 norm. Both norms
must be nonzero. The diversity statistic is the absolute value of the
equal-cell mean inner product; it must not exceed `0.95`.

Parent hashes and weight are part of the ensemble configuration before
evaluation. There is no uncounted optimization of the ensemble weight. The
ensemble consumes one ordinary stream evaluation.

The submitted `parent_hashes` array must already be in ascending lexicographic
order and the controller rejects, rather than sorts, any other order. `alpha`
always multiplies the lexicographically first parent hash. This prevents
canonicalization from changing a prediction.

## 13. Legal combinations

- All feature scopes are legal with all four aggregations:
  `moments_quantiles`, `robust_histogram_8`,
  `low_rank_set_encoder_8`, and `low_rank_set_encoder_16`.
- All three nuisance modes are legal with all mapping families.
- `none`, Procrustes, and regularized CCA are legal only for source-bearing
  transfer configurations.
- Equal, reliability, similarity, and stacking pooling are legal with ridge,
  elastic net, reduced-rank ridge, and MLP.
- Hierarchical pooling is legal only with ridge or reduced-rank ridge.
- `source_prior_shrinkage` and `convex_source_target` require a source-bearing
  configuration.
- `zero_prior_fit` requires `alignment=null` and `source_pooling=null`.
- Calibration-only configurations always have `alignment=null`,
  `source_pooling=null`, and `adaptation=zero_prior_fit`; they retain the full
  representation, nuisance, rank, whitening, ridge, elastic-net,
  reduced-rank-ridge, and MLP families.
- An ensemble has only parent hashes and a weight; all ordinary operator fields
  must be null.
- A base transfer configuration is source-bearing exactly when its adaptation
  is not `zero_prior_fit`. A transfer ensemble is source-bearing when at least
  one parent is source-bearing; F04 zeros only those source-bearing parent
  contributions and never relabels a source-free parent.
- Unused parameter fields are forbidden rather than silently ignored.
- Spike-PCA rank must satisfy Section 5. Set-encoder rank follows the distinct
  Section 4.3 rule and may include deterministic zero-eigenvalue components.
  Any other rank or shape failure invalidates the configuration. There is no
  automatic reduction of rank, feature scope, neuron set, or model family.

## 14. Canonical configuration identity and invalidity

Serialize configurations using canonical JSON with sorted keys, exact decimal
grid literals, explicit nulls only where required above, and no derived seeds.
The SHA-256 of those bytes is the configuration identity.

A solver non-convergence, nonfinite fitted state, nonfinite prediction,
unsupported rank, failed CCA positive-definite solve, or illegal combination
is an invalid configuration. It receives no score and cannot fall back to
another operator or hyperparameter. The paired-round controller applies the
registered replacement and retry rules.

Every implementation must pass synthetic tests for:

- exact target-PCA reconstruction with whitening on and off;
- regularized CCA output in target-PCA coordinates;
- permutation invariance of every electrode aggregation;
- fold-local refitting and whole-trial fold containment;
- hierarchy and stacking weight recovery;
- deterministic MLP replay;
- native-scale negative predictions remaining unclipped; and
- rejection at direction error greater than `0.01`, but not at the observed
  `0.0027871747` radians.
