# Does a single preceding LMP bin improve the primary history compromise?

Prepared 2026-10-04 from released complete W, HW, matched-budget, elapsed-only
and capacity-surrogate evidence. **ADOPTED by root at09:08UTC after root and
independent scientific review; isolated implementation is authorized.** No source,
partial blend/intersection outcomes, new fit or scientific score was read or
computed for this proposal. Root may adopt and prepare isolated implementation
prospectively while the current blend and intersection work proceeds; their
unreleased outcomes are not prerequisites for that decision. Any numerical
launch must be serialized with current work and fit the live stopping budget.
This creates no new required-coverage obligation or automatic successor run.

## Why this one comparison is worth considering

Completed W has primary/identity increments 0.002174850827889 /
0.003504405470203; HW has 0.0023412544731769847 / 0.0056143839673429. HW
improves identity substantially, but its primary contrast to W is negative
in Mihili (−0.000228909235730 under every omission) and positive in Chewie
(+0.000561716526307). Only 2/6 Mihili primary contrasts are positive. This
motivates a simpler temporal model; it does not establish that distant lags
cause the loss or that a shorter history will recover it. Evidence is in
the complete [HW panel](runs/adaptive_search/primary_history_repeatability_weight_summary_attempt002/primary_history_repeatability_weight.json)
and its 07:25 closeout in the experiment log.

The [matched-budget evidence](runs/adaptive_search/primary_matched_budget_summary_attempt001/compact_evidence.json)
also shows sensitivity to fitting support: channel matching retains primary
0.002213016 and identity 0.005271333, while trial matching lowers them to
0.001030769 / 0.001632066 and selects lambda10 throughout. These different-
support diagnostics suggest estimation burden may matter; they do not measure
the effect of removing lags. The
[capacity control](runs/adaptive_search/primary_capacity_matched_surrogate_summary_attempt001/compact_evidence.json)
shows a real advantage of correctly paired training over its conditional
surrogate, with shared Gram and effective df: HW-minus-null primary
0.003707855328 and identity 0.004853258310, positive in all sixteen sessions.
This supports retaining a training-alignment test alongside the new decoder.

The [elapsed-only result](runs/adaptive_search/primary_elapsed_only_summary_attempt001/compact_evidence.json)
does not motivate weakening the behavioral comparator: its large identity
score coexists with an original-B-relative primary loss of −0.250247782.
Keep the original first-harmonic direction/time baseline and native XY target.

Recommend one fixed two-bin model, H2W, plus its matching training surrogate.
The current blend instead combines two completed predictions; H2W would fit
a new stationary map using only the current and immediately preceding bin.
Their predictions need not coincide. This recommendation is independent of
unreleased blend outcomes and makes no assumption about their eventual
count, incumbent or patience transition.

GOAL and SEARCH_POLICY permit a prospectively finite historical representation.
The older HW proposal's fixed four-lag/no-later-truncation commitment continues
to govern HW and all its preserved outcomes. This document proposes a separate
recipe with one new fixed lag set, not a modification, repair or retuning of
that completed recipe. Do not add three-bin histories, longer histories,
lag subsets, lag-specific penalties or a history-length selector to this run.

## H2W: exact original support and one new temporal representation

Candidate ID: `lmp_history2_repeatability_weight_ridge_v1`.
For target bin t use `[t,t-1]`, in lag order `[0,-1]` followed by the original
physical-electrode order. This means 2E features with a maximum offset of
30ms before t. The fifteen native XY targets remain offsets0:+14; all sixteen
original sessions, original complete trials, directions and physical LMP
electrodes remain unchanged. No newly eligible trial or electrode is added
because less history is used. Chewie's implants remain one animal.

Use the qualified original eighteen-bin history reader (source offsets−3:+14)
and its original array layout for complete support/state replay. The new
model uses only offsets−1:+14; earlier read bins support unchanged anchor
preprocessing/replay and enter no H2W predictor. Record both the reader's
interval and the effective model interval without calling them identical.
Never enlarge the trial set, shorten the target, inspect spike values, or
reinterpret offline stored features as online/causal recordings.

For each held direction, reuse the original training-only four-lag
`prepare_fold` preprocessing without running the old ridge fits. Its
first-harmonic direction-by-time template uses equal-direction means from
the seven fitting directions; its common residual intercept is their pooled
native XY residual mean. Preserve the numerical two-step residual convention
`(Y-template).reshape(-1,2)-intercept`. Add exactly the same B back to every
new held-direction residual prediction.

The original raw current slice `x[:,3:18,:]` must retain its time-fast layout.
Reuse the original W fitting physical mean/population-SD calculation and
current-only T/P/r/a statistics, with `r=clip(P/T,0,1)`, T=0 giving r=0, and
`a=.1+r`. No target values, held-direction trials or past-lag repeatability
enter a. Require exact replay of this W state against original fold records.

For the decoder, take the first 2E columns of the original *unweighted*,
training-normalized and fitting-centered four-lag matrix, before new metric
weighting. Use the matching original evaluation columns. This deliberately
retains the original retained-column means/SDs, zero-SD→1 handling and final
H centering arithmetic; do not reconstruct a C-order raw array or recompute
the retained normalization by a different reduction path. Older-lag values
do not enter the retained-column formulas. A source-free leakage fixture must
verify that changing the omitted two lag blocks does not change this state.

Let Z2 denote the resulting fitting matrix, n its bin rows, G2=Z2'Z2/n, and
`q_e=G2[(0,e),(0,e)]+G2[(-1,e),(-1,e)]`. Apply the unchanged trace-normalized
physical prior to this new history support:

`w_e² = a_e * trace(G2) / sum_j a_j*q_j`.

Repeat w across both lags and apply it to fitting and evaluation features.
The new weighted trace equals trace(G2), not the four-lag trace. Do not reuse
HW's already normalized four-lag weights, divide features by sqrt(2), or
standardize/center again after weighting. For exactly zero G2, use all-one
weights and retain the fold. If all a on q>0 electrodes are exactly equal,
use all-one effective weights and record the structural identity of the
weighting. Identity weighting alone does not duplicate W or HW: a distinct
two-bin representation remains a distinct candidate.

Fit ridge against the unchanged native XY residual with the existing
objective `mean(||R-Z2w C||²)+lambda*||C||²` and original five lambdas
`[.001,.01,.1,1,10]`. Select one lambda for each held outer session using
only the other fifteen sessions' held-direction primary scores, equal
sessions within animal then equal animals, with stronger-lambda exact ties.
No per-animal/session history choice, held-session tuning, or identity-based
selection is introduced. Save all five held-direction prediction banks.

## One matching conditional-capacity training surrogate

Control ID: `lmp_history2_repeatability_weight_training_surrogate8_v1`.
Use exactly the eight already declared HW-capacity destination-to-donor maps:
ordinary Python random version2 shuffles of numerically ordered original
trial IDs independently within each fitting direction, using the literal
`ep05-lmp-hw-capacity-training-feature-surrogate-v1|20261004|<session>|<held>|<replicate>|<direction>`.
Reuse the existing map helper unchanged and verify IDs/maps against the
completed HW-capacity records. Ordinary fixed points, identity maps and
repeated draws are retained; no redrawing or seed selection is permitted.

Each map moves an entire weighted 15×2E fitting trajectory. Keep the true
response residual in its original recipient order, and keep evaluation
features, B, all fitted preprocessing and repeatability weights unchanged.
Thus the surrogate changes training feature/response pairing only. It
preserves the exact mathematical Gram, spectrum, rank and conditional ridge
df of H2W. Compute one Gram/eigensystem per direction and share it for the
candidate and all eight control right-hand sides; do not refit eight Grams
and accept different numerical capacity. At fixed fitting state this is
the same permitted surrogate family already used for HW, now challenging
the new two-bin decoder. It is not phase randomization or signal-free data.

Use H2W's outer-selected lambda for every draw, without tuning the null.
Average correct SSE and wrong-pair SSE across eight draws, using common D;
do not average their predictions. Report every draw, both animals and fixed-
prediction omissions. Draws are not extra animals, confidence intervals or
a new significance test. This control retains direction/time structure and
the correctly paired evaluation features, so a positive contrast only
supports dependence on true training alignment conditional on that state.

## Original endpoint, anchors and the decision the comparison can support

Use unchanged whole-session joint native-XY denominator
`D=sum((Y-mean_coordinate(Y))²)`, negative R² values, identical B and exact
within-direction whole-trial evaluation-derangement mean. Primary is
`(SSE_B-SSE_candidate)/D`; identity is `(SSE_wrong-SSE_correct)/D`.
Do not average direction R² values or choose favorable bins.

Replay the complete original W/HW arrays, native trial IDs/order, labels,
target/B and five-lambda correct/wrong-pair scores, using the qualified exact
saved-prediction layout convention. Require original preprocessing/state
replay as above, but never require the *new* two-lag normalized weight vector
to equal the four-lag vector. Reuse immutable originals without neural refits:

- W banks: `runs/adaptive_search/primary_repeatability_weight_control_attempt001/`;
  summary: `primary_repeatability_weight_summary_attempt001/primary_repeatability_weight.json`.
- HW banks: `runs/adaptive_search/primary_history_repeatability_weight_control_attempt002/`;
  summary: `primary_history_repeatability_weight_summary_attempt002/primary_history_repeatability_weight.json`.
- Fixed control maps: `runs/adaptive_search/primary_capacity_matched_surrogate_attempt001/`,
  the preserved bank root recorded by the complete capacity summary.

Report H2W-minus-W and H2W-minus-HW both at independently selected original
anchor penalties and at H2W-matched original lambda: four direct same-D
prediction-SSE/identity contrasts. These compare complete model recipes;
matching lambda, physical channels or trace does not match W/HW effective df.
If the already adopted blend has completed before the H2W summary is published,
also compare to its fixed original outer-selected saved predictions, after
exact support/score replay. This comparison is declared now without seeing
its outcomes. Do not retune its alpha/lambda to H2W or construct another blend.
It ensures the pending plausible incumbent is represented alongside the two
fixed component anchors; it adds no recipe or gate. If that blend remains
technically incomplete, leave its comparison explicitly unavailable.
Also report H2W-minus-mean-null and all eight draws. Report all sixteen
session rows, animal means/sign counts, within-animal omissions, selected
lambdas, trace/rank/spectrum and selected-lambda df for the new shared Gram.

The motivating prediction is improved Mihili primary accuracy relative to
HW without sacrificing the animal-balanced primary objective. Report the
actual signs and magnitudes even if that prediction fails. Reduced identity
gain is an anticipated tradeoff. These diagnostics add no new margin, gate
or noninferiority claim. The original primary .01 and identity .005 criteria,
both-animal/majority/omission requirements and required finalist falsifiers
remain controlling. Old HW checks do not automatically establish that a new
finalist passes its own required controls. This pair does not itself authorize
promotion, a follow-up spike-population analysis or a future-session test.

## Duplicate, trial and patience accounting

Process the complete candidate first, then the mean-eight control. Each
distinct valid recipe contributes at most one scientific comparison; five
lambdas, eight draws, exact anchors and omission reports add no extra depth.
H2W is a candidate, with zero literal-null and zero conservative alternative-
check increment. The distinct mean control adds one literal null and one
alternative check. Do not relabel existing HW controls or use candidate
ablation language to inflate the alternative fraction.

Check structural identities separately. Exactly zero retained past-lag fitting
energy throughout all folds/sessions triggers examination of a possible W
identity; exactly zero discarded −2/−3 lag energy triggers examination of a
possible HW identity. Neither energy condition alone establishes duplication.
To charge zero by algebra, explicitly derive unchanged retained preprocessing,
residual/intercept, effective trace-normalized metric and ridge head for all
five penalties, followed by the unchanged outer selector. Extra zero columns
have zero fitted coefficients, but numerical centering or normalization must
not be dismissed as an unspecified rounding allowance. If that derivation is
not established, do not automatically collapse the comparison. Require exact
state/array equalities wherever used; no near-zero, near-uniform, allclose or
score-tolerance fallback is permitted. The completed HW record already reports
positive energy in all four lags. Separately, a complete selected prediction
panel literally identical to original selected W/HW predictions adds zero.
Apply the same literal selected-panel duplicate check to the completed blend
when its saved predictions are available. No interior/nonduplicate outcome
is forced.

The control adds zero if all surrogate cross-products equal the true H2W
cross-product in all required draws/folds/sessions. It also duplicates the
completed HW-capacity control if the discarded lag blocks are structurally
zero and an explicit derivation establishes identical retained preprocessing,
trace metric, each surrogate head and selected-lambda rule with the preserved
maps/fitting support. The energy flag alone does not establish that identity;
if the proof is absent, do not auto-collapse or use numerical closeness. Candidate
duplication with W, or uniform weighting, does not alone prove that this
training surrogate duplicates any previous W weight-label control. Diagnose
and record its own algebraic identity rather than borrowing candidate flags.

A distinct H2W may update the full-precision comparable original-support
incumbent under the [adopted patience rule](primary_patience_interpretation.md).
Only a strictly higher complete primary score resets patience; the control
cannot reset it. Use the incumbent and counter actually in effect at
completion, not the current HW score or an assumed ordinal. Preserve the
28–64 bounds, twelve-trial patience, applicable required coverage, >=40%
post-initial alternative allocation, 2,000 core-hours, 120-hour clock and
all resource ceilings. This optional pair is not a reason to continue past
a valid stopping condition or to extend any cap.

## Resource pilot and bounded implementation if adopted

Use new isolated tools and preserve all original code/banks. Reuse the exact
qualified reader, four-lag preprocessing, W statistics, ridge algebra and
capacity-map helpers; implement only the two-lag extraction/trace weighting,
new candidate head and matching surrogate heads. Focused fixtures should
check t=0/14 indexing, omitted-lag and held-direction exclusion, unchanged
current prior/B/XY numerical replay, two-lag trace normalization, shared
Gram/df, whole-trial maps, own control right-hand sides, wrong-pair algebra,
mean SSE, outer-session selection and separate duplicate rules.

Propose fixed Chewie_20150313, one numerical thread, 1CPU/4GiB,
270 seconds internal/300 seconds external, covering the complete candidate,
all eight controls, five penalties, eight directions and exact anchor replays.
Its scientific scores stay closed until the full cohort completes. The known
maximum1038trials×15bins×192two-lag columns uses 23,915,520 bytes (~22.8MiB)
for one float64 design; the retained four-lag preprocessing state is roughly
twice that, while one 192² float64 Gram is ~0.28MiB. These are array estimates,
not an RSS guarantee. Process one direction/control at a time and release
temporary blocks; no parallel BLAS is needed.

The completed eight-draw four-lag capacity pilot took17.073seconds with
437260KiB peak RSS, while elapsed-only source reads varied considerably.
For H2W, eigendecomposition dimension halves and shared-Gram construction
has fewer columns, but the new candidate adds one right-hand side and
preserved anchor replay/source I/O remain. Do not extrapolate a promised
speedup. Root chooses a finite remaining-panel worker/time/memory bound only
from the completed new pilot's resources and available allocation, preserves
that pilot, and publishes no partial scientific panel. No automatic retry,
larger lag schedule, solver change, new session or source interval is selected
by this proposal. Numerical launch still requires reviewed implementation,
focused compute checks, the declared pilot and sufficient actual resources.

Root resource selection,09:18UTC: the fixed pilot completed in job46472346
at09:16:53.285116–09:17:08.647683 in15.362559 seconds, source10.840313,
candidate/control fit and scoring3.690497, peak389,400KiB. All five exact
original state/array/map/score replays passed on1,038 trials. Pilot scientific
scores remain closed. Preserve this pilot and run fifteen fresh sessions
sequentially with one thread/4GiB,600-second internal/630-second external
aggregate bounds and unchanged270/300 children. Fifteen pilot equivalents
take230.438389 seconds; a complete300-second child reserve leaves69.561611
seconds for summary and variation. Require660seconds actual allocation time
remaining, including cleanup. The current allocation has insufficient time;
no remainder is launched or deadline shortened. Fresh summary and parent
paths are `primary_history2_repeatability_summary_attempt001` and
`primary_history2_repeatability_panel_attempt001`. Include the prospectively
declared complete original blend summary as a fixed anchor. No original
recipe, scientific choice, budget ceiling or stopping rule changes.
