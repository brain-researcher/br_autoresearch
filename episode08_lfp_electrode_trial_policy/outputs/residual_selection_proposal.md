# Electrode selection for within-direction population information

Proposed2026-10-03 after the completed T/L/K comparison and **adopted by the
coordinator at21:46UTC before new fits**, with the ordinary-permutation correction
adopted before execution. All three panels and the focused check completed in
job46430982,step15 at21:50:17UTC. Three new development panels address residual association,
conditional redundancy and the direction-preserving target-link falsifier.
They do not activate removal-value/coupling endpoints or sequential acquisition.

## Fixed comparison

Use the completed reduced-rank **mean-feature** residual ridge as the one global
decoder because it has the largest current animal-balanced Q. Its weaker Mihili
gain remains visible; never choose a decoder separately by animal or session.
Retain its common legal rank grid1/2/4/8/full, five alphas, smaller-rank/larger-alpha
exact tie rule, raw-Y SD, inner-fold direction means and intercept, unchanged
three mean LFP coordinates/electrode and native population-mean target. Keep
all current roles, direction balance,32/64/96prefixes, four folds,8/16/32/all96
cells and four closed held sessions.

Every inner selector is fitted on that fold's fitting rows **before** its
validation inputs/targets enter decoder CV. The outer selector refits on the
complete allowed prefix. Selection cannot reuse an all-prefix ordering inside
inner CV. Decoder selection/scaling/means always use true training targets;
the target permutation changes only the selector's input. No source path or
outer evaluation array enters a numerical fitting component.

## Selection scores and three panels

Within each permitted selector fitting set, compute direction-specific raw X/Y
means. Subtract them and scale X by residual fitting SD, Y by **raw** fitting Y
SD, zeros replaced by1without removal. Center resulting residual columns by
their fitting global means to retain the intercept convention. Do not call the
old selector's extra global Y standardization: it would replace raw-Y weighting
with residual-Y weighting. Keep each electrode's three features together.

Let n be fitting rows, U target columns, A_S=(I+X_S X_S'/n)^-1 for retained set S.
For candidate three-column block E define
`G_E=X_E' A_S X_E/n`, `C_E=X_E' A_S Y/n` and
`gain(E|S)=trace[C_E' (I+G_E)^-1 C_E]/U`.
This is a fixed mean-loss regularized linear proxy with selector penalty1,
not an information-theoretic estimator or the final decoder's rank choice.
The resulting predictor is still the same tuned reduced-rank decoder in all
panels. Exact score ties choose ascending physical channel ID; build one nested
ordering through32electrodes, with8/16/32prefixes.

1. **Residual marginal association:** compute every gain from empty S once,
   then sort; no conditional updates.
2. **Residual conditional association:** greedily update A_S after each chosen
   block using the existing exact rank-three update; capture nonredundant gain
   under the declared proxy.
3. **Within-direction permuted-target conditional selection:** same conditional
   score, but replace only selector Y rows with a whole-population-vector
   ordinary uniform permutation within each direction, including fixed points. Do not shuffle
   units/bins independently or cross directions. Direction target means and
   raw-Y SD are unchanged by the permutation. Final decoder fitting/validation
   always uses the true Y rows.

For control replicate0..4, each selector fitting set/direction draws one ordinary
uniform permutation, including fixed points. The existing smallest inner group
has3rows, so finite-group behavior matters.
Use Python Random's fixed seeded version2with the string
`ep08-residual-selector-permutation-v1|20261003|filename|training_total|replicate|fold_or_outer|direction`.
Seed once per group and shuffle once. Use the same resulting ordering across
its electrode prefixes.
Before any tests, fits or scores, the coordinator corrected the initial
derangement proposal: for a centered group of size m, derangements have expected
partner value -Y_i/(m-1), creating negative covariance that a squared-gain
selector could exploit. Ordinary permutations instead have zero expected
centered covariance. This prospective correction does not change D/W/R's
already completed **evaluation pairing** control, which asks a different
descriptive question and does not claim to be a null selector-training scheme.
The five complete control replicates are averaged within each scarce cell,
not counted as five methods. The first two panels use one deterministic order.

## Decoder tuning, exposure and reporting

Fit all selected models for a session before its forty evaluation rows are read.
For each trial amount, share one unchanged full96reduced-rank model among all
three panels/control replicates. Replay its Q against the completed capacity
report within absolute1e-10 before releasing this comparison's scientific
packet; discrepancy is an engineering problem, never a retuning opportunity.
The prior random-electrode reduced-rank panel is the matched decoder reference.
Only after all fitting/scoring read its historical scores for paired reporting.

Retain every cell/replicate/session/animal, Q, gain over matched random reduced-
rank and direction-only, and exact within-direction prediction wrong-pair Q.
Averaging and denominators are unchanged. Report rank/alpha choices and equal
rank/alpha tuning opportunity across these selector panels. Global Q improvement
uses the existing0.001resolution; pairing is descriptive, no new p/margin.
The control targets a selector's dependence on true trial pairing, not the
final decoder's target signal, and cannot establish sensor–trial coupling.

Offline selection observes all96training electrodes and all allowed targets.
This is no lower-acquisition-cost claim. Source read volume remains the fixed
768training+320evaluation rows,30,094,080logical bytes. Same-UID callable boundary
is disclosed. Three complete panels charge3and would move12to15; data cells,
control seeds and decoder tuning are not separate panels. Expected528actual
outer refits,11,088ridge systems,52,800nested decoder scores and840selector fits.
One CPU/2GiB/ten-minute step with540sinternal bound is proposed; actual cost is
unmeasured. Reuse qualified readers, rank decoder and prior matrix-update checks.
If implemented, one focused boundary/equivalence check should establish fold-
local residual weighting and group-preserving ordinary selector-only
permutation; no repeated generic preflight is needed.

Further trial-coverage/value, reverse-animal/session robustness, low-frequency/
contamination and acquired-only replay remain distinct work. This recipe is
not a closed menu, an explanation acceptance rule, or permission to open held
sessions.
