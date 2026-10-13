# Does direction-inclusive association improve a frozen initializer ranking?

Prepared2026-10-04 after37 completed comparisons. **Root adopted at07:18UTC
after root and independent scientific reviews passed.** The fixed new
implementation and bounded full-cohort comparison are authorized, subject to
focused checks and root's verified allocation reserve. Adoption is not an
observed launch or completion. Four held sessions and named endpoints remain
closed.

## Distinct question and prior-method overlap

The cross-validated single-electrode utility candidate loses to the residual
marginal incumbent in both animal means and fails its own control screen in
Mihili. Together with the previous target-free and greedy comparisons, this
leaves a distinct preprocessing question: should a small paid initializer
retain direction-related LFP-to-population association when choosing a fixed
set for later **residual** prediction? Strong direction-related structure may
identify channels that also remain useful for within-direction decoding; it
may instead distract the selector from the decoder's residual target. Neither
possibility is established by the completed comparisons.

The original [selector comparison](selector_comparison_proposal.md) already
used this raw marginal formula, including a32-row fitting prefix. It observed
all96 electrodes and recomputed selection separately at32/64/96 fitting rows,
then used ordinary raw-target ridge heads. Therefore the claim is **not** that
raw marginal association or32-row selection is new. The prospective branch
fixes the original24/32-row initializer orders through every later acquisition
checkpoint and uses the current direction-residual reduced-rank decoder and
initializer-conditioned CV. The [acquired-only comparison](acquired_only_proposal.md)
activated that information state only for residual marginal selection. The
new paired result tests direction-inclusive versus residualized ranking under
the same paid information, fixed acquisition and actual decoder.

Use one candidate, `initial_raw_marginal_v1`, and one mean-five-control panel,
`initial_raw_marginal_target_labels5_v1`, with fixed seeds0,1,2,3,4. No proxy
penalty grid, adaptive combination of raw/residual scores, or extra panel is
selected. This lies within the operative open policy's prospective selector
comparisons. The named electrode-removal, next-direction-value and coupling
endpoints remain unactivated; the ranking statistic is not a new empirical
explanation endpoint.

## Exact original raw marginal statistic

Preserve the original eight development sessions and physical channel order,
96 fitting trials,40 reserved evaluation trials, balanced direction queues,
32-trial all96 initializer and four24-row main inner-fitting views. The outer
view has four trials/direction; each inner view has three. Native target units
stay separate. Inputs remain the three actual LFP means per electrode:
LMP,(100,200)Hz,(200,400)Hz, over the same ten150–450ms bins. No representation,
time-window, target or source-support change is proposed.

In each allowed view independently, compute the **global** fitting column
means and population SDs of raw X and true Y, with exactly zero SD replaced
by one and every column retained. Define

`Z = (X - mean_view(X)) / sd_view(X)`

`T = (Y - mean_view(Y)) / sd_view(Y)`.

There is no subtraction of direction means for this selector. All normalizers
exclude the main inner validation rows and every future/evaluation row. These
are the original `selector_training.py` raw-marginal coordinates, including
raw population target SD. Their input scaling differs from the incumbent's
residual-X SD as well as their centering; the contrast is the complete fixed
preprocessing recipe, not an isolated causal intervention on direction means.

For each physical electrode's three-coordinate block e, with n allowed rows
and U native target units, compute once from the empty retained set:

`G_e = sym(Z_e' Z_e / n)`

`C_e = Z_e' T_selector / n`

`gain_e = sum(C_e * solve(I_3 + G_e, C_e)) / U`.

The dimensionless mean-loss penalty is exactly one, independently of the
actual decoder's tuning. There is no conditional update, whitening, internal
validation utility or positive-gain stopping. Compute in float64; fail on
nonfinite coordinates, solved coefficients or scores rather than dropping
data or falling back. Keep finite signed roundoff values as the original
formula does, without clipping. Rank all96 electrodes by descending gain,
with exact ties by ascending physical channel ID, and retain a nested top32
order with8/16/32 prefixes. Record all96 scores and constant-SD diagnostics.

## Matched controls preserve direction-level association

The true candidate uses T directly. Each control uses the **exact original**
canonical within-direction whole-target-vector maps from
[initializer target-label control](initializer_target_label_control_proposal.md).
For each allowed main view and direction separately, sort original integer
trial IDs, seed an ordinary Python `random.Random` with version2 and literal

`ep08-acquired-marginal-target-label-v1|20261004|<filename>|<view>|<seed>|<direction>`.

View is0/1/2/3/outer. Shuffle one fresh canonical donor-ID list once, align
with canonical recipient IDs, and map back to supplied row order. Preserve
identities and fixed points without rejection, derangement or redraw. Use one
map for the entire native-unit vector, never separate unit/bin permutations.
No map crosses directions or leaves its24/32-row information view.

Compute normalizers from true fitting X/Y once, then set
`T_selector = T[donor_map]` only for control cross-products. Global Y means,
SDs and within-direction means are mathematically unchanged. All five seeds
share the view's input Gram; each has its own cross-product, scores and order.
Save exact map records and require agreement with the completed original
control's mappings after all new fits are finished.

This control **retains direction-related association**: its expected raw
cross-covariance need not vanish because direction means remain paired. It
tests the usefulness of the additional within-direction trial pairing for
this raw ranking, not total target dependence or a signal-free selector. Its
ordinary small-group shuffles can have many fixed points. The inherited
residual control uses the same maps but different selector coordinates; their
contrast magnitudes are not interchangeable nulls or calibrated p-values.
An across-direction shuffle would answer another question and is not added.

## Acquisition, true-target heads and timing

Every candidate and control pays all32x96 initializer electrode-trials.
Generate all30 orders/session (six branches times five main views) before any
post-initial neural source read. Each main inner decoder uses its own24-row
order; the outer32-row order remains immutable through later acquisition.
Acquire only its retained electrode set afterward along the original balanced
queues. Preserve E=8/16/32/96 and n=32/48/64/80/96, all20 resource cells.
No discarded-channel backfill, revision, later reranking or adaptive action.

The actual decoder remains the qualified direction-residual reduced-rank
ridge model, with true unpermuted Y for every training row, direction mean,
scale, alpha/rank selection and final refit. Use actual3E inputs, the original
five penalties, legal rank menu, low-rank/strong-alpha exact tie preference
and fitting-only normalizers. Later CV keeps all initializer rows in every
training split and validates only post-initial trials. The raw selector does
not replace decoder residualization or direction-null predictions.

Fit all95 distinct models/session before any of its40 evaluation rows:
six branches times15 scarce cells plus five shared all96 checkpoint heads.
There are120 model references/session; identical all96 models are shared
across all six branches after current support equality, not fitted six times.
The existing selected-channel trajectory and native-unit reader contracts
apply unchanged. The same-UID boundary remains logical, not an OS sandbox.

## Scoring and deferred historical reference replay

After all760 new models across eight sessions are complete, read only the
completed `initializer_target_label_control_attempt001/initializer_target_label_control.json`
for the residual `initial_marginal` incumbent and its five-seed control.
Do not read old reports into selection, fitting, stopping or seed choice.
Require exact sessions/animals, original recipe, source-role records,
physical channel/trial/fold identities, target-unit counts and exposure,
all20 cell identities and all25 per-session original view/seed map sets,
comprising200 individual direction maps/session. Historical
role records do not contain unit-guide IDs: do not assert historical unit-ID
equality; preserve current qualified within-run guide checks and matched
recorded target support.

Use the unchanged training-global-null denominator and direction-only D
predictions. Replay all recorded full96 Q/capacity metadata against the old
candidate and every original control seed with the existing1e-10 Q tolerance,
and require exact within-run common-full96 prediction/score capacity identity.
Original legacy `initial_marginal` cells omit `null_SSE` and standardized SSE:
do not fabricate or claim replay of those absent fields. Require the complete
persisted SSE/denominator fields in the original target-label control and
replay new target denominators against it at every matched cell. Known legacy
absence is explicit; unexpected missing or partial required schema fails.

For each new control cell, average the five saved standardized correct and
wrong SSEs with their common denominator, then derive Q/pairing increments.
Never average predictions or choose a favorable seed. Strip seed-variable
channel/rank/alpha metadata from averaged cells and retain per-seed model
records. Use all20 equal cells, then four equal sessions/animal, then two equal
animal means. Full96 identity cells remain included.

The primary paired contrast is raw candidate minus residual marginal
incumbent under matched cost/decoder, with a prospective challenger screen
requiring strictly positive means in both animals; ties fail. The own-control
screen similarly requires raw candidate minus mean raw-label control to be
positive in both animals. Report both contrasts overall/by animal/session/
cell, all five seed contrasts, fixed-prediction leave-one-session omissions,
and raw candidate minus existing D using the existing metric only. Report
the residual candidate/control contrast descriptively for context. There is
no new success margin, confidence interval or automatic incumbent promotion;
the existing0.001-Q progress resolution and adaptive bookkeeping remain.

Positive own-control evidence concerns within-direction correspondence in
this ranking. A positive challenger contrast would support this early
preprocessing choice for residual decoding, not prove biological direction
mechanisms, electrode necessity, coupling or a joint optimal acquisition rule.
Only two animals and exploratory development outcomes remain available.

## Fixed work, resources and comparison count

The scientific branch matches `32*96 + (n-32)*E` physical LFP electrode-trials
and `n*U*10` target values; three native bands contribute30 numeric values per
electrode-trial. Reusing a historical panel neither erases its acquisition
cost nor creates a new charge.

Expected complete work:760 actual models,15,200 inner ridge fits,76,000
capacity scores,240 orders,1,600 direction-level maps and40 direction-null
fits. The240 orderings each evaluate96 marginal blocks once:23,040 logical
three-coordinate score solves, irrespective of batched numerical execution.
There are40 true-X/Y normalization views; reuse each for its five controls.
Fit source reads are8x1,248=9,984 trial views plus320 evaluation trial views.
The previous identical acquisition pattern measured113,959,680 logical and
30,094,080 unique native bytes; recompute actual dtype accounting rather than
hardcoding observed values. Preserve selection time, fitted constant-SD
occurrences and per-seed score/map records.

The proposed bound is270seconds internal/300seconds external, one CPU with
all numerical threads one, within4GiB branch memory, in a coordinator-owned
attended allocation. Several completed same-size six-branch panels used
roughly87–107seconds and under104MiB measured process RSS; this ranking is
the already tested fixed3x3 marginal solve, so no new resource pilot is
required by the current evidence. Root must still select an actual attempt
after focused checks and verify the full reserve before launch, or obtain a
new attended allocation. There is no automatic retry or bound extension.

Required focused checks are raw/global normalization and original-formula
equivalence, canonical maps and allowed identities, direction-mean retention,
fitting-view exclusion, deterministic physical-ID ties, true-Y decoder paths,
30-order/95-model evaluation barriers, mean-five-SSE aggregation and complete
reference/schema replay. Root owns execution; an independent worker owns the
scientific fixtures. Existing code/output paths remain preserved.

Only a complete eight-session candidate and mean-five control add two valid
comparisons,37 to39. Five seeds, cells, proxy scores and historical replays
are not extra comparisons. Failure keeps its resources and artifacts without
claiming completion. The next decision remains outcome-responsive within the
existing search policy; reaching a count is not a scientific rationale.
