# Temporal information versus decoder capacity within reach direction

Proposed October 3, 2026 after D/W/R completed and **adopted prospectively by
the coordinator at21:36UTC**. All three panels and both focused checks completed
in job46430982,step12 at21:40:46UTC. Actual evidence and three evaluation charges
are recorded in the execution log.
This is three genuinely distinct development model/representation panels under
open exploration, not a selector comparison or named coupling test. The prior
mean-feature direction-residual random ridge is the unchanged reference.

## Why compare these now

Residual ridge improves direction-only Q in all eight sessions, but its gain
is modest (0.006486; Chewie 0.010520, Mihili 0.002452). Trial pairing accounts
for most of that gain. This makes two explanations actionable: collapsing ten
source bins to one mean discards useful temporal structure, or a full-rank
linear fit misses or inefficiently estimates the within-direction relation.
Temporal ridge changes the representation alone; reduced-rank ridge changes
the response-space complexity; a radial-basis kernel changes linearity while
retaining the mean features. None selects electrodes using evaluation targets.

Residual-informed conditional electrode selection is a valuable subsequent
comparison, but performing it now would entangle a new target-sensitive selector
with uncertainty about the shared decoder. First choose or retain a supported
representation/decoder from these development comparisons, then compare residual
marginal association, residual conditional gain and its within-direction
training-target permutation control under that same decoder, target, information
and tuning opportunity. Do not count an unchanged raw selector with a new name
as a method. No scientific promise to run that successor is made here.

## Common roles, preprocessing and score

Keep exactly the eight current development sessions, 96 maximum fitting trials,
balanced nested32/64/96 prefixes, four fixed inner folds, forty evaluation rows
(five per direction), fixed five electrode permutations and nested8/16/32/all96
sets. All96 is fitted once per trial amount, not five times. Held sessions remain
closed. Reuse the same qualified ten-bin150–450ms window and all native M1 units.
The target is unchanged: each unit's mean over those ten bins. Every temporal
trial remains **one row**, never ten independent samples or CV units.

For every inner fitting set, and again independently on the full outer prefix,
fit direction-specific arithmetic X/Y means. Subtract those fitting means from
both fitting and validation/evaluation rows of the corresponding direction.
Standardize each residual input column by its fitting residual population SD;
zero becomes1, no column/trial/unit is dropped. Scale residual Y by **raw fitting
Y population SD**, retaining the D/W/R target weighting; zero SD becomes1.
Fit a global unpenalized residual intercept using fitting rows only. Add the
fitted direction mean and intercept back to each final raw-Y prediction.
No transform, bandwidth, rank, alpha, or direction mean uses outer evaluation.

All inner tuning minimizes summed true-target prediction SSE in each inner
fitting raw-Y SD coordinates after adding back fitted direction means. Exact
ties use the rules below, without tolerance-based score grouping. The existing
alpha grid remains(0.0001,0.01,1,100,10000), mean-loss penalty n*alpha. Do not
expand it after seeing boundary choices; report such choices explicitly.

Report the unchanged raw-target Q denominator from the outer training global
mean, paired delta Q from direction-only and prior residual-mean ridge, and
exact within-direction wrong-pair Q/increment. Every replicate, resource cell,
session and animal remains visible; average five scarce-set replicates, twelve
cells, four sessions/animal and two animals equally. Secondary pairing does not
create another evaluation charge or become a new stopping endpoint. Preserve
the 0.001-Q development progress resolution; no biological margin is added.

## T: temporal residual ridge

Retain all ten ordered source bins for each selected electrode/band, making30
coordinates/electrode in electrode-major, band-major, time-major order. This is
an invertible flattening of the same permitted source slice, with no additional
window or neural exposure. Fit the common direction means/residual scaling
separately for each coordinate. Then use the unchanged full-rank mean-loss ridge
and five-penalty inner tuning; exact alpha ties choose larger alpha.

This compares available temporal detail with the existing three mean coordinates
per electrode while holding target, random sets and data roles fixed. It does
not change the target to time-resolved firing or infer time bins as replicates.
It has five tuning candidates per inner split, as does the prior mean ridge.

## L: reduced-rank residual ridge on mean features

Use the prior three mean features per electrode and common residual transforms.
For each alpha, compute ordinary ridge B_alpha. Select the response projection
from descending eigenvectors of the symmetric matrix
`C = Y_fit.T @ X_fit @ B_alpha` (symmetrize for floating arithmetic), not from
an SVD of the shrunken predictions. This is the training response-space metric
for the **regularized reduced-rank** objective. At rank r use
`B_alpha @ V_r @ V_r.T`. The full-rank option is ordinary B_alpha exactly.
No projection sees inner validation or evaluation targets.

The finite rank candidates are1,2,4,8,and full. For each fitting set cap feasible
numeric rank by min(input width,target width,n_fit-8), reflecting eight fitted
direction means. Use the intersection of legal ranks across all inner fitting
sets and the full outer prefix, so every candidate has complete common-fold
support. A rank exceeding that bound is unavailable and reported; the
full option remains ordinary ridge rather than a data-dependent truncation.
Exact ties prefer the smaller allowed rank, then larger alpha; full sorts last.
Use the same fixed native target order and descending symmetric-eigensolver
order throughout; no direction/target is removed to obtain a desired rank.

The rank/alpha grid has at most25 candidate scores per inner split versus five
for ordinary ridge. Alpha solutions/eigendecompositions may be shared across
ranks; this is computational reuse, not extra method panels. Report the actual
allowed grid, chosen rank/alpha and boundary frequencies. This comparison has
more tuning opportunity; it cannot be described as equal tuning capacity or a
pure selector improvement. All choices are nevertheless nested in training.

## K: radial-basis residual kernel ridge on mean features

Use the same three mean features, direction residuals/scales and target as L.
For each inner fitting set, compute all distinct-row squared Euclidean distances
in its standardized residual input coordinates. Fix
`sigma_squared = median(strictly positive off-diagonal squared distances)`.
The raw kernel is `exp(-distance_squared/(2*sigma_squared))`. This bandwidth
rule is fixed: no bandwidth multipliers, evaluation-distance tuning or additional
kernel search. If all fitting rows coincide, the centered training and prediction
kernel is identically zero: use that exact constant-feature limit and retain
the fitted direction mean/intercept, without choosing an arbitrary bandwidth.
Record the case; no trial or target is dropped.

Center K using fitting column/row/grand means (H K H), and center prediction
cross-kernels using the same fitting column/grand means plus each prediction
row's mean over fitting anchors. This latter mean is an input transformation,
not fitting from evaluation targets or other evaluation rows. Solve
`(K_centered+n*alpha*I) A = Y_residual_centered`. Add the fitting residual
intercept and direction target mean at prediction. Only the same five alphas
are tuned; exact ties choose larger alpha. Recompute the fixed bandwidth and
centering on the outer prefix for its single refit. This is nonlinear kernel
ridge with five tuning candidates, not a new per-session kernel family search.

## Execution, cost and decisions

This proposal charges **three** panels only if all complete, moving9to12. The
rank/alpha candidates, electrode replicates, cells and reused reference scores
are not additional panels. Read the same source slices once per session into
an evaluator-owned10-bin tensor. Fit all T/L/K models for that session before
reading its40 evaluation rows. The fitting components receive only copied
allowed rows. The actual all96-electrode offline exposure remains explicit;
there is no acquired-only or sequential-efficiency claim. Existing same-UID
filesystem limitations remain, and no candidate plugin is introduced.

A one-CPU/2GiB/ten-minute bounded step with an internal540-second deadline is
proposed. The source slice volume is unchanged30,094,080 logical bytes. At most
384 outer models/panel and8064 ridge/kernel systems/panel are needed; rank
projections reuse alpha solutions while producing the extra nested validation
scores. T uses2880 coordinates at all96 but still at most96 trial rows; its
solve can remain dual. Temporary arrays are modest relative to2GiB. Actual cost
is unmeasured; preserve partial counts and stop at the finite limit rather than
silently enlarge it. Existing passing role/reader/ridge checks are reused.
If implemented, focused independent checks should verify the reduced-rank
regularized objective against an augmented-design reference and centered-kernel
prediction/intercept against a direct finite-feature reference. No generic
preflight, source copy or unchanged full test suite is needed.

Interpretation is conditional on the results. A temporal gain supports retained
within-window detail; a low-rank gain supports a useful response-complexity
restriction; a kernel gain supports this nonlinear predictor, not a nonlinear
biological mechanism. None makes a coupling claim. If none exceeds the unchanged
residual mean ridge, retain it before testing residual-informed selection or
trial coverage. Reverse-animal/session omission, acquired-only replay, coverage,
spike-contamination and the other required controls remain later work; this
proposal is not an exhaustive method menu or a shortcut to the held opening.
