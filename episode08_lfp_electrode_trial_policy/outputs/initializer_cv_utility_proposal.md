# Does internal validation improve single-electrode utility ranking?

Prepared after35 completed comparisons; **root adopted at2026-10-04
07:01UTC** after root and independent scientific review passed. The fixed
implementation and bounded resource pilot are authorized. Remaining work is
selected from pilot resources before score release; all held and named
explanatory endpoints remain closed.

## Question and fixed comparisons

The supervised initial marginal selector remains ahead of repeatability,
target-linked greedy conditional selection and target-free residual-input
logdet selection. The latter does not beat matched uniform selection in
Chewie even under session omissions. A distinct possibility is that the
existing marginal score rewards associations that fit a small initializer
but do not predict another trial from the same information state.

Test a single-electrode utility score estimated by **validation strictly
inside each existing selection view**. This changes the ranking estimator,
not the final decoder, target, representation, total acquisition or main
inner/outer information boundary. It does not tune another proxy penalty or
claim to measure biological electrode necessity.

Candidate: `initial_crossvalidated_utility_v1`.
Control: `initial_crossvalidated_utility_fit_labels5_v1`, fixed seeds0–4.
The matched control disrupts target correspondence only inside each internal
fitting partition, while retaining its true validation targets. Its mean-five
SSE panel is one comparison, not five.

The open policy permits prospective model/selection comparisons under fixed
information and cost commitments. All outcomes remain the already adopted
prediction Q and its ordinary method contrasts. Internal validation utility
is a selector statistic, not a newly activated empirical explanatory endpoint.
Named removal-value, next-direction-value and coupling tests remain closed.

## Original roles and strictly internal balanced folds

Keep the exact eight development sessions,96 physical electrodes,96 fitting
trials and40 reserved evaluation trials/session, original within-direction
queues and32-trial all-electrode initializer. Preserve native-unit target
means, three native LFP features/electrode, ten-bin150–450ms means,
E=8/16/32/96 and n=32/48/64/80/96: all20 original equal cells.

The outer selector still receives exactly32 rows, four per direction. Each
of the four main inner selectors still receives exactly its original24
fitting rows, three per direction; its eight main-validation rows never
enter any part of that selector. These main roles do not change.

Inside a main view with m rows/direction, m is3 or4. For each direction,
sort its original trial IDs ascending and assign positions j=0,...,m−1.
Internal validation fold k contains the one row/direction satisfying
`(j + direction_index) mod m == k`; all remaining rows are internal fitting.
Thus each24-row view has three internal folds of16 fitting +8 validation
rows; the32-row view has four folds of24 fitting +8 validation rows.
Every row is validated exactly once. Assignments depend only on original
IDs and direction metadata, never neural values, fit error or seed.

All internal fitting, validation and normalization stay inside the current
main view. No internal-validation X or Y may contribute to that internal
fold's direction means, scales, intercept or proxy coefficients. Neither
future acquisition nor reserved40-trial evaluation data are used.

## Exact signed validation utility

Each internal fitting partition F defines all normalization. For X, subtract
its eight fitting direction means, subtract the fitting residual mean and
divide by its residual population SD. For Y, keep the qualified raw-Y fitting
population SD and eight fitting direction means; retain the fitting mean of
the standardized direction residual as the residual intercept b. Exact zero
input or target SD becomes one, with every column/unit retained. Apply these
same fitting quantities to the internal validation rows V. No normalization
is recomputed on V or the enclosing24/32 rows.

Let Z_F and Z_V be these centered/scaled residual inputs. Let
`R_F = (Y_F - mean_F(Y | direction_F)) / sd_F(Y) - b`, and
`R0_V = (Y_V - mean_F(Y | direction_V)) / sd_F(Y)`.
For each physical electrode's three-coordinate block e, fit exactly one
full three-coordinate alpha1 mean-loss ridge proxy:

`G_e = sym(Z_F,e' Z_F,e / |F|)`

`C_e = Z_F,e' R_F / |F|`

`B_e = solve(I_3 + G_e, C_e)`.

The baseline prediction is the direction mean estimated in F; the proxy
adds `sd_F(Y) * (b + Z_V,e B_e)`. Its signed validation benefit is

`u_e,F = sum_units,sum_rows_in_V [ R0_V^2 - (R0_V - b - Z_V,e B_e)^2 ]`.

Sum u across the m internal folds and divide by **m*8*U**, where U is the
unaltered native target-unit count. This is mean standardized squared-error
reduction per validated trial/unit. Each fold has eight validation rows;
all directions, folds and target units receive the stated equal contribution.
The tiny retained residual intercept follows the qualified fit convention;
it is never estimated from V. Do not divide by a validation-derived noise
variance or an evaluation denominator, choose a winning internal fold, fit a
proxy reduced rank, or search alpha. The fixed alpha1 and three coordinates
match the original proxy's convention; its utility is now assessed on rows
excluded from the proxy fit.

Keep the exact signed mean utility, including negative values. There is no
zero floor, shrink-to-positive operation, significance filter or positive-gain
stopping rule. Nonfinite normalization, coefficients, losses or score is a
technical failure with no exclusion/fallback. Rank all96 physical electrodes
by decreasing signed utility, exact ties by ascending physical ID, and freeze
one nested top32 order for E=8/16/32. No greedy conditioning or reranking.

## Matched internal-fitting target-label control

A permutation of the whole24/32 main view could carry an internal-validation
target into its fitting state. This control instead permutes whole native-
unit Y vectors **only within each direction of the current internal F**.
There are two donors/direction for a16-row F or three for a24-row F. The eight
internal validation targets remain true and never become fitting donors.

For each main view, internal fold and seed0–4, use an ordinary Python random
permutation of canonically sorted original fitting trial IDs separately in
each direction, with explicit version2 string seed:

`ep08-initial-cv-utility-fit-label-v1|20261004|<filename>|<main_view>|<internal_fold>|<seed>|<direction>`.

Main view is0/1/2/3/outer; internal fold is0,...,m−1. Identity maps and fixed
points are allowed without redraw or derangement. Whole vectors move all
units together, preserving within-F direction means and raw-Y SD. Compute
normalizers/intercept from true F once, then permute centered standardized
R_F only for the control C_e. Gram matrices, X, b, validation Z/Y, fold losses,
units and aggregation are otherwise identical. Every control seed has its
own coefficients and aggregate ordering; no proxy coefficient is ensembled.
Save complete original recipient/donor IDs, seed literal and fold membership.

These are **new internal-fitting maps and a different null** from the earlier
whole-main-view label controls. Do not claim map identity with that prior
null or attribute a change in their contrast solely to the utility estimator.
The current null asks whether a proxy learned from fitting correspondence
ranks electrodes better on true internal validation. With only two/three
trials per fitting direction, many ordinary maps remain partly or wholly
unchanged; report that limitation, never force stronger perturbation.

## Actual decoding, timing and reference reuse

All30 candidate/control main-view orders/session must be complete before
any post-initial source row. Each main inner decoder uses its own24-row
selector's order, and the outer32-row order remains fixed throughout later
acquisition. Internal proxy fits are not actual population decoder fits.

Every actual decoder and its main CV uses **true, unpermuted Y**, the existing
direction-residual reduced-rank ridge procedure, actual3E rank limits,
original alpha/rank grid and lower-rank/stronger-alpha ties. Later conditioned
folds retain every initializer row and validate only newly acquired rows.
Five identical full96 fits may be shared across the six branches; all95
distinct models/120 references/session must precede reserved evaluation.

Only after all new fits, read the completed initializer repeatability report
and reuse its initial-marginal, matched-uniform and repeatability panels.
No historical score, rank or selected set enters internal folds or selection.
Require exact recorded roles/physical IDs/trials/target counts, matching20
cells and physical/numeric/target exposure. Current unit-guide equality stays
with the reader; do not claim unsaved historical unit-ID equality. Replay
recorded target denominators against uniform/repeatability with the existing
1e-10 relative/absolute tolerance, direction-null Q against all references
at1e-10 absolute, and new full96 support/capacity/prediction scores against
qualified original full96 records. Legacy marginal SSE/denominators were not
saved and are not reconstructed. No reference fit/comparison is recharged.

Mean the five **actual decoder SSEs** at every cell for the control panel,
never their predictions, preserving each seed. Report every cell/session,
equal20-cell session means, equal sessions within each animal and equal
animal means. Candidate-minus-own-control is the correspondence contrast;
candidate-minus-initial-marginal tests the changed ranking estimator.
Uniform, repeatability, direction-only and wrong-pair differences remain
specified descriptive references. Preserve all eight fixed-prediction session
omissions and each control seed; do not choose a seed or remove a session.

Two prospective screens are separate: candidate-minus-own-control Q must be
strictly positive in each animal mean, and candidate-minus-initial-marginal Q
must be strictly positive in each animal mean to pass the incumbent challenge.
Exact ties fail. Keep the existing0.001 search-progress resolution. No new
biological margin, significance/equivalence, automatic promotion or joint
mechanism claim is introduced.

Internal fitting has only two or three trials/direction, so these utility
estimates can be noisy. A win would support this fixed cross-validated ranking
recipe, while a loss would not prove that the original score is free of fitting
optimism. The changed estimator also changes sample reuse and estimation
variance; do not assign its performance difference to one cause alone.

## Resource pilot, exact counts and accounting

Each branch pays `32*96+(n-32)*E` physical electrode-trials,30 actual-dtype
LFP values per electrode-trial and `n*U*10` target values. Main information,
source exposure, decoder capacity and tuning opportunity match between the
candidate and its own controls. Extra proxy CV is computational cost and is
reported separately; do not hide it inside the old decoder count.

Per session the four24-row main views contribute4*3=12 internal folds and
the outer32-row view contributes4, totaling16. At96 electrodes and6 branches
this is **9,216 single-electrode proxy fits/session**, or **73,728 overall**.
The five controls produce16*8*5=640 direction maps/session, or **5,120 overall**.
There are16 unique true-fitting normalization views/session,128 overall,
reused without validation leakage across seeds. The actual decoder workload
remains760 target fits,15,200 inner ridge fits,76,000 capacity scores,
240 main selector orders and40 direction-null models across eight sessions,
with9,984 fitting-trial read calls and320 evaluation trials. Shared Gram
factorizations or batched solves do not erase those logical proxy models.
Record proxy and decoder work separately, selected-order/normalization timing,
source bytes, CPU/wall time, peak memory and retained report bytes.

The new internal-CV workload warrants a measured first-session resource pilot.
Propose the predetermined full Mihili20140217 session, all six branches and
all95 actual models, on1CPU within4GiB with270seconds internal/300seconds
external. It must finish9,216 proxy fits,1,900 actual inner fits,9,500 actual
capacity scores,30 orders,640 maps,16 normalization views,five null models,
1,248 fitting-trial read calls and40 evaluation trials. No scientific scores
or old-reference JSON are released from that
pilot. It charges no comparison by itself.

After its measured resource/support evidence, root selects a finite remaining-
seven bound before opening pilot scientific scores, within the existing
resource allowance. Proposed ceiling is600seconds internal/630seconds
external, selected only if the completed pilot supports it and the actual
allocation has reserve; a new attended allocation may be used when needed.
Do not extend a running deadline or automatically retry. Preserve and reuse
the completed pilot exactly once after all seven new sessions finish, requiring
its recipe, roles, counts and barrier flags. No pilot refit or duplicate charge.
This pilot is for the new73,728-proxy workload, not a universal preflight.

Focused checks should independently establish the internal-fold exclusion
and fitting-only normalization; direct signed validation-SSE utility including
negative/zero/tied cases; maps restricted to F with identity allowed and true
validation Y; original main inner exclusion; true-target actual decoders and
30/95 timing; exact proxy/map counts; historical schema/full96 replay; and
pilot preservation with old-reference reads only after all new fits.

One complete candidate plus one mean-five control adds **two comparisons**,
35to37. Proxy fits, seeds, folds, checkpoints, pilot, reused references and
omissions add none. Existing coverage,40–112 limits, post-coverage alternative
check fraction, patience and shared resources remain binding. This comparison
does not finish coverage or open any held or named explanatory endpoint.
