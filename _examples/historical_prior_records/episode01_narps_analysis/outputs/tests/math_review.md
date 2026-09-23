# Independent frozen-mathematics review

Scope: synthetic/read-only audit of the frozen functional ANOVA, equal-cohort
group coefficient, paired Bayesian bootstrap, and matched-size deletion
calculations. No neural outcome was accessed by these tests, and no production
or frozen-contract file was edited.

## Verdict

No blocking mathematical defect was found in `outputs/code/aggregate_results.py`.
Its internal cell order is `(C,S,Q)` (Q varies fastest), and all projection,
labeling, Gram, OLS-t, and deletion-metric calculations consistently use that
order.

## Exact functional ANOVA

Let `J = 11'/3` and `H = I-J`. For contract order `(C,Q,S)`, the grand-mean
projector is `P0 = J⊗J⊗J` and the seven term projectors are

- `P_C = H⊗J⊗J`, `P_Q = J⊗H⊗J`, `P_S = J⊗J⊗H`;
- `P_CQ = H⊗H⊗J`, `P_CS = H⊗J⊗H`, `P_QS = J⊗H⊗H`;
- `P_CQS = H⊗H⊗H`.

For production order `(C,S,Q)`, swap the Q and S Kronecker positions, exactly as
the implementation does. Each matrix is symmetric and idempotent, distinct
terms are orthogonal, ranks are `2,2,2,4,4,4,8`, and
`P0 + sum(P_term) = I_27`.

For a 27-by-V map matrix `Y` and raw map Gram `G=YY'`,

`SS_term = ||P_term Y||_F^2 = tr(P_term G)`.

Equivalently, the marginal-effect formulas require replication factors 9 for
each main effect, 3 for each two-way interaction, and 1 for the three-way
interaction. The exact partition is

`sum_term SS_term = tr((I-P0)G) = sum_cell,voxel (Y_cell-grand_voxel)^2`.

No spatial mean subtraction belongs in this primary ANOVA.

## Shapley allocation

For each factor, allocate every orthogonal interaction equally among factors in
that interaction:

- `A_C=SS_C+SS_CQ/2+SS_CS/2+SS_CQS/3`;
- `A_Q=SS_Q+SS_CQ/2+SS_QS/2+SS_CQS/3`;
- `A_S=SS_S+SS_CS/2+SS_QS/2+SS_CQS/3`.

Thus `A_C+A_Q+A_S=SS_total`. This equals the Shapley value obtained by
enumerating all six factor-entry permutations for the game whose coalition
value is the sum of ANOVA terms contained in that coalition. Shares divide by
`SS_total`; ratios divide the replicate-level attributed quantities, not their
separately summarized intervals.

## Equal-cohort group coefficient

With OLS design rows `[1,+0.5]` for EI and `[1,-0.5]` for ER,

`alpha_hat = (mean_EI + mean_ER)/2`,
`beta_hat = mean_EI - mean_ER`.

The intercept coefficient on subject `i` in eligible Q set is therefore
`1/(2 n_EI,Q)` for EI and `1/(2 n_ER,Q)` for ER. It is not the unstratified
mean when cohort sizes differ, and subject first-level precision must not be
used as a group weight. For the frozen secondary OLS t statistic,

`SSE = sum_EI ||y_i-mean_EI||^2 + sum_ER ||y_i-mean_ER||^2`,
`df=n_EI+n_ER-2`, and
`Var(alpha_hat)=SSE/df * (1/4)(1/n_EI+1/n_ER)` voxelwise.

## Paired Bayesian-bootstrap Gram algorithm

For replicate `r`, draw one `e_i ~ Exp(1)` for each Q0 subject. For every Q and
cohort `g`, define

`w[r,q,i] = (1/2) I(i eligible in q,g) e_i / sum_{j eligible in q,g} e_j`.

The same raw draw must be reused for all C, Q, S, and both contrasts; only the
within-cohort eligible-set denominator changes.

Let `E[i,b,:]` be a subject map for base cell `b=(C,S)` and
`K[b,i,d,j]=<E[i,b],E[j,d]>`. The exact replicate group-map Gram is

`G[r,(b,q),(d,p)] = sum_i,j w[r,q,i] K[b,i,d,j] w[r,p,j]`.

ANOVA SS then follows from `tr(P_term G[r])`, without materializing replicate
voxel maps. Production implements precisely this identity. A further exact
reduction useful for auditing chooses orthonormal level basis
`U=[1/sqrt(3), (1,-1,0)/sqrt(2), (1,1,-2)/sqrt(6)]`, transforms subject maps
along C and S, aggregates four subject Grams for constant/contrast C and S
subspaces, and uses `h=U'W`. It yields twelve subject quadratic forms per
replicate and matched the full projections in every synthetic trial.

For bootstrap pairwise Pearson metrics, group spatial sums must be propagated
alongside raw Grams. The replicate centered Gram entry is
`G_ab - sum(a)sum(b)/V`.

## Matched-size deletion metrics

For one C/S cell, let source and target coefficient vectors be `a` and `b`,
raw subject Gram `K=EE'`, subject spatial-sum vector `s=E1`, and
`Kc=K-ss'/V`. Let `R` be the squared norm of the fixed observed
C0-Q0-S0 group reference map for the same contrast. Then

- `NRMSD = sqrt((a-b)'K(a-b) / R)`;
- `Pearson = a'Kc b / sqrt((a'Kc a)(b'Kc b))`;
- the reported second metric is `1-Pearson`.

The V factors in both RMS quantities cancel. Raw K is required for NRMSD;
spatially centered Kc is required for Pearson. Production implements both
identities and uses the observed fixed reference. Its random subject sets are
drawn once in recorded replicate-major transition order and reused across all
nine C/S cells and both contrasts. The upper-tail p-value is exactly
`(1 + count(null >= observed))/1001`.

## Executed tests

- `outputs/tests/test_math_review.py`: projection symmetry/idempotence/
  orthogonality/ranks/completeness; marginal replication factors; six-
  permutation Shapley equality; unbalanced group coefficients; reduced-Gram
  versus direct bootstrap ANOVA; raw/centered deletion Grams versus direct
  voxel metrics. Twenty randomized repetitions passed.
- `outputs/tests/test_aggregate_results_math.py`: black-box production audit
  with 25 randomized term-label/order cases, all bootstrap term SS and ratios,
  unbalanced OLS intercept/t, and all matched-deletion distributions for both
  transitions and nine C/S cells. Passed.
- Production `aggregate_results.py --self-test`. Passed.

## Implementation pitfalls and residual minor risk

- A `(C,Q,S)` versus `(C,S,Q)` flattening mismatch silently relabels Q and S;
  production is internally consistent and independently tested.
- Normalize bootstrap weights within each eligible task-version cohort, never
  across the whole Q set. Reuse raw draws rather than independently drawing for
  each cell/Q/contrast.
- Do not spatially center maps for ANOVA, cosine, or NRMSD. Do center for the
  definition of Pearson correlation.
- Form percentile intervals from each replicate's share or ratio directly.
  The two primary 97.5% intervals are percentiles 1.25 and 98.75.
- Use float64 accumulation for Grams, spatial sums, SSE, and quadratic forms.
- Treat nonpositive reference norms, Pearson variances, total SS, or smoothing
  attribution as undefined rather than adding an epsilon.
- Minor robustness improvement: `matched_calibration` could clip correlations
  to `[-1,1]` after validating a small numerical tolerance and explicitly check
  all null metric arrays are finite before percentile/p-value summaries. The
  general pairwise helper already does this. No substantive discrepancy was
  observed in synthetic tests.
