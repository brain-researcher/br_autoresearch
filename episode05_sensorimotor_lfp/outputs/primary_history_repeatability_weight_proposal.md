# Does the fixed repeatability prior improve the fixed four-bin history model?

Prepared 2026-10-04 after release of all twenty-one completed comparisons.
**Adopted by root 2026-10-04 07:08:33 UTC after root and independent review.**
Implementation and focused fixtures are prepared for coordinated compute
validation; this paragraph makes no execution claim. Root owns compute and
execution records. No incomplete elastic-net score or
new source value was read to choose this pair. Existing failed comparisons,
eligibility, margins, budgets and stopping conditions remain unchanged.

## Why combine these two already fixed choices?

The complete current-bin repeatability-weighted model W produced primary
gain 0.002174850827889 and identity gain 0.003504405470203. It improved on
full LMP by 0.000237466082668 and 0.000108080924405 respectively, with positive
animal means and fixed-omission means for both contrasts. It also beat its
weight-label controls in both animals. Nevertheless, it failed both fixed
practical margins, 0.01 and 0.005.

The previously completed four-bin history model H had primary gain
0.002038782088692 and identity gain 0.005498615738149. Its primary increment
over current-bin LMP was small and not consistent across animals and session
omissions. Thus W supplied a more consistent current-bin primary increment,
while H supplied the stronger identity readout. Neither result shows that
the two changes will combine successfully.

The next question is whether **the already fixed current-bin repeatability
prior improves the already fixed history representation**, and whether the
history adds useful prediction beyond W. This combines two permitted choices
without searching a lag count, window, repeatability floor, time-specific
weight or new target. More history coefficients can overfit; the complete
candidate and its matched control must decide the result.

## HW: one physical weight shared by all four original lags

Candidate ID: `lmp_history4_repeatability_weight_ridge_v1`.

Keep all sixteen original M1 sessions, complete original trials and frozen
physical LMP channels. Neural input remains offsets -3:+14, inside the already
qualified -4:+14 source interval; the native XY target remains offsets 0:+14.
For target bin t, use exactly `[t, t-1, t-2, t-3]`, in lag order
`[0,-1,-2,-3]` followed by original physical-electrode order. There are 4E
columns. No later lag/window search, truncation, new neural interval or
velocity diagnostic target is introduced. Released offline features do not
support an online or causal interpretation from these past indices.

Preserve all other primary choices: eight held-direction fits; equal-direction
first-harmonic baseline and its common fitting-residual intercept; native XY
residuals and pooled session SSE; exact evaluation-pairing mean; original
five penalties; other-fifteen-session equal-animal lambda selection; and
equal-session then equal-animal summaries. All preprocessing below excludes
the evaluated direction. Neither velocity targets nor scores enter weights.

For each outer held direction h, construct two explicitly defined input
states from its seven fitting directions:

1. **Weight state:** apply exactly W's existing current-bin preprocessing to
   the raw current slice at offsets 0:+14, using its fitting physical means
   and population SDs, with zero SD mapped to one. Use this state to compute
   the unchanged T/P/r/a recipe from
   [the adopted W proposal](primary_repeatability_weight_proposal.md). For
   each electrode, T is equal-direction mean trajectory energy and P is
   equal-direction mean ordered distinct-trial trajectory dot product,
   both averaged over fifteen bins. Use `r=clip(P/T,0,1)`, T=0 gives r=0,
   and `a=0.1+r`. Invoke the existing current-bin functions directly rather
   than deriving a new statistic from past lags or H's numerical centering.
   Save the current-only physical state and signed/undefined diagnostics.
2. **Decoder state:** preserve H's original separate fitting mean/population
   SD for each lag-by-electrode column, zero SD mapped to one. Apply those
   parameters to fitting and evaluation blocks; flatten in the original
   lag-major order; subtract H's resulting fitting-column means from both
   fitting and evaluation matrices. This final centering is the existing H
   convention, even though its means are numerically nearly zero. Do not
   replace it with the current-only state, whiten scores or divide by sqrt(4).

The weight state and the lag0 decoder state are mathematically the same
physical normalization before H's final centering, but their numerical paths
remain those of the corresponding preserved recipes. This prevents an
unrecorded preprocessing change when comparing either anchor.

Let Z_H be the resulting centered history fitting matrix with n rows and 4E
columns, and `G_H=Z_H'Z_H/n`. For each physical electrode define its **total
history energy**

`q_e = sum_lag G_H[(lag,e),(lag,e)]`.

Using the current-only base a, compute one positive physical-electrode weight
shared across all four lags:

`w_e² = a_e * trace(G_H) / sum_j a_j*q_j`.

Multiply every history column `(lag,e)` by w_e, in fitting and evaluation
matrices. The full weighted history Gram has the same trace as G_H. If G_H
is exactly all zero, set every weight to one and retain the fold. Otherwise
the fixed positive floor ensures a positive denominator. Nonfinite values
or invalid arithmetic stop the attempt; do not add an epsilon or delete a
column. Do not restandardize or recenter after applying these fixed weights;
linear weighting of the already centered H state preserves its convention.

The same physical prior is shared across lags; there is no per-lag repeatability
estimate or per-lag weight search. The normalization uses the **whole history
Gram**, not the current-only Gram. A channel constant at lag0 can still carry
past-bin information and therefore contribute q_e>0 while receiving a_e=0.1.
Do not declare that electrode inactive using only T_e or current-bin energy.

Fit the unchanged native-residual ridge head at all original
`lambda in [0.001,0.01,0.1,1,10]`, add the same baseline back, and select
lambda for each outer session using only the other fifteen HW banks.
Every physical weight is positive, so the exact history feature span remains
available. In the original history coordinates, the induced penalty is

`lambda * sum_e sum_lag ||coefficient_(lag,e)||² / w_e²`.

When lag-specific zero-variance support differs, history-trace normalization
can change the common scale of w relative to W. Matched nominal lambda across
HW and W therefore does not establish matched capacity, an identical penalty
on the current-only submodel, or a strict same-lambda embedding. Across HW and
H the nominal history width and total input energy are matched, while the
penalty metric, covariance spectrum and effective df can change.

## C: the same eight physical weight-label maps, shared across lags

Control ID: `lmp_history4_repeatability_weight_labels8_v1`.

Reuse exactly the current W control's ordinary physical-ID permutations,
replicates 0..7, with literal

`ep05-lmp-repeatability-weight-label-v1|20261004|<session>|<replicate>`.

The canonical physical IDs, Python seed version2 and one ordinary shuffle
remain unchanged. Accept fixed points and identity maps; do not redraw.
Each map is fixed per session, shared across outer directions, times and
all four lags. Save the original recipient/donor physical IDs. Do not reuse
H's older per-trial channel-signal maps: that was a different control.

For each fold, reassign the same current-only a values to the permuted physical
IDs, then apply **that control's own full-history trace normalization** using
the original q. Repeat its physical weights across all four lags. Original
history signals, normalization, centering, true fitting targets and evaluation
pairings remain unchanged. Each control requires its own weighted history
Gram and ridge head. Permuting already normalized current-only weights or
shuffling the signal columns would test a different recipe.

Use HW's selected outer-session lambda for all eight controls. Average their
correct SSEs and their exact wrong-trial SSEs with the common denominator;
do not average predictions, retune a control or select a seed. Control draws
are descriptive repeated fits, not additional animals or a confidence interval.

## Anchors and prospective screens

Use both preserved complete candidate anchors without refitting them:

- H: `lmp_history4_ridge_v1`, from `primary_history_control_attempt001` and
  `primary_history_control_summary_attempt001/primary_history_control.json`.
- W: `lmp_repeatability_weight_ridge_v1`, from
  `primary_repeatability_weight_control_attempt001` and
  `primary_repeatability_weight_summary_attempt001/primary_repeatability_weight.json`.

Report HW-minus-H and HW-minus-W under both their independently selected
penalties and HW-matched original lambda: four anchor contrasts. Report
HW-minus-mean-control, every seed, every session, both animals and all fixed-
prediction omissions. The H comparison asks whether the weight prior helps
with the same history opportunity. The W comparison asks whether this history
recipe adds useful prediction to the successful current-bin weighting; it
does not isolate lag information from all capacity and shrinkage changes.

The weight-label screen requires positive primary and identity contrasts
in both animals and primary-positive counts >=4/6 Mihili and >=6/10 Chewie.
A claim of improvement over either anchor requires that same directional
screen under **both** anchor penalty comparisons. Ties fail. A claim that
the combination improves both component recipes requires all four anchor
screens. None replaces the unchanged 0.01/0.005 practical gates, original
omission requirements, required falsifiers or absent future-session test.

Check original physical/trial/direction/target/baseline identities after fitting,
using the qualified history-input/current-target replay helper. Preserve the
distinct source windows in metadata. Require recomputed current-only T/P/r/a,
weight-state normalization and the fixed physical maps to replay W's matching
fold records, with no outcome-based alternate normalization. Do not demand
that the new history-normalized weights equal W's current-only weights.

## Diagnostics and structural accounting

Retain the original lag-specific means/SDs and H centering, current-only weight
state and T/P/raw-ratio/r/a, q, all applied weights and physical maps. Report
candidate/control full-history trace, spectra, numerical rank and selected-
lambda ridge df. The same E physical electrodes and matched history trace do
not establish matched effective df or an isolated channel-quality effect.

For each arm, an exact zero G_H is an identity case. Otherwise, if all a
values on physical electrodes with q_e>0 are equal, full-history normalization
is an identity weighting on the fitted information; use all-one effective
weights and record the structural identity. Use exact base-value equality,
without a near-uniform tolerance. Check each control independently after its
label assignment. Compare candidate/control applied physical weight vectors
by literal equality only on q_e>0 to record controls identical to HW.

Candidate identity throughout all required folds/sessions duplicates H and
adds zero depth. A control identity throughout every seed/fold/session also
duplicates H and adds zero. If every control equals HW throughout, charge
the mean control zero rather than treating identical fits as a new comparison.

Also preserve the degenerate no-history case: if all three past-lag blocks
have exactly zero fitting energy in every fold/session, HW and its controls
are algebraically the already evaluated W and W-control recipes, respectively;
record that structural duplicate and add zero. This follows from zero ridge
coefficients for the extra columns, the same current-only a, the same maps,
and trace normalization reducing to the current-only case. No outcome
comparison or numerical near-equality threshold establishes duplication.
This is an algebraic recipe identity, not a promise of identical floating-
point arrays: H's preserved final centering can introduce rounding-level
differences from W's numerical path. That centering does not justify a new
scientific comparison or a claim of exact numerical embedding.

Otherwise each distinct complete recipe contributes one comparison, up to
two. A new negative result still counts; individual lags, weights, seeds,
penalties and anchor contrasts do not add depth. No control may be declared
duplicate merely because the unpermuted candidate is an identity.

## Bounded implementation and interpretation

After adoption, use new core/runner files and reuse the qualified history
reader, fixed lag layout, H fitting preprocessing/ridge head and W statistics/
physical maps. Do not modify old H/W tools or banks. The window remains the
previously qualified -3:+14 LMP support and 0:+14 XY target.

Focused fixtures should check unchanged lag/target indexing, separate original
current-weight and H-decoder preprocessing, exclusion of held-direction and
velocity values from a, positive full-history trace normalization, the
diagonal-penalty equivalence, one weight/map repeated across lags, own control
Grams and true targets, current-zero/history-active electrodes, independent
identity/accounting including the all-past-zero case, mean SSE and outer-
session lambda exclusion for both reused anchor families.

Propose one complete fixed Chewie_20150313 resource pilot, candidate plus all
eight controls, on one CPU within 4 GiB, 270 seconds internal/300 external.
A 96-electrode history has 384 columns; the largest observed trial support
gives roughly 48 MB for one float64 history design. This is not a measurement
of total RSS. Process arms sequentially; reuse each arm's decomposition
across the five original penalties where applicable. Keep pilot scientific
scores closed while root selects a finite remaining-panel bound from actual
resource/support facts. No automatic retry, solver-cap increase or dropped
session is authorized.

Two distinct complete recipes advance 21 to 23, with nine controls among
seventeen post-initial comparisons; apply the actual per-recipe duplicate
counts when needed. Preserve the 28–64 depth, 12-trial post-minimum patience,
>=40% post-initial control fraction, 2,000 core-hour ceiling and 120-hour clock
starting 2026-10-03 22:23:02.824 UTC. No material-improvement cutoff or early
stop is invented. A positive result remains an exposed-corpus comparison of
this input metric and history recipe, without intrinsic sensor-quality,
causal/online, spike-free, cross-session coefficient-transfer or new-animal
claims.
