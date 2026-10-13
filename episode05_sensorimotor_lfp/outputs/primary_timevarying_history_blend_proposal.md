# Does a fixed linear time schedule improve the saved W/HW blend?

**Adopted by root after independent scientific review, 2026-10-04 09:39 UTC.**
Isolated implementation and focused fixtures are authorized, followed by the
specified bounded saved-artifact pilot after passing checks. No provider
source read or neural refit is authorized. It was specified while
intersection I/R outcomes were closed; root subsequently released their
complete panel at comparison37, patience4, with the incumbent unchanged.
I/R outcomes were not consulted to choose this family or its grid. The
scientific motivation here uses the completed original W, HW, constant blend
and H2W records. No per-bin outcome was inspected to choose a slope, direction,
knot or time interval. Incoming count is 37, patience 4, alternatives 19 and
literal nulls 13 among 31 post-initial comparisons; the incumbent primary is
0.0026788544310999856. Root selects the remaining-panel bound only after the
pilot's resource evidence. No mixture score has yet been computed.

## Question and fixed scope

The [assessment](primary_post_blend_next_step_assessment.md) motivates a small
prediction-level alternative to a new temporal tensor fit. The complete
constant W/HW blend has primary increment0.0026788544310999856 and identity
increment0.004559394718773168, choosing lambda1 and alpha0.5 in every outer
fold. Both practical margins still fail. Complete H2W has primary
0.0024652592046145306 and loses to the blend in both animal primary means and
under every within-animal omission. It improves over current-only W, while
its comparison to HW retains an animal tradeoff. These are reasons to ask
whether the two preserved prediction maps should receive different weights
across the fixed reach window, not evidence that such a time pattern exists.

Use exactly two previously evaluated components, original W and HW. H2W and
the original blend are frozen comparators, not additional mixture components.
All sixteen original M1 sessions remain exposed exploratory development;
Chewie's two implants remain one animal. Preserve original trials, physical
support, target0:+14, B and joint native-XY scoring. No source values, new
neural fit, new target, lag set, event alignment or time-window selection is
part of this candidate. It is one prospectively finite two-component ensemble
under the existing GOAL/policy menu, not missing initial-model coverage.

Proposed recipe: `lmp_current_history_repeatability_linear_time_blend_v1`.

## Fixed cells, arithmetic and outer-session selection

For the original fifteen target bins t=0,...,14, compute in float64

`u(t)=2*t/14-1`,

`alpha_s(t)=0.5+s*u(t)`,

with fixed slopes `s=[-0.5,-0.25,0,0.25,0.5]` and original shared penalties
`lambda=[0.001,0.01,0.1,1,10]`. Define full native XY predictions

`P(lambda,s,t)=(1-alpha_s(t))*P_W(lambda,t)+alpha_s(t)*P_HW(lambda,t)`.

The same schedule applies to both coordinates and every direction/trial in
the evaluated session. Weights remain between0 and1. Their mathematical
bin average is0.5; compute the literal formula without numerical recentering,
clipping, renormalization or a data-derived adjustment. This fixes the average
mixing coefficient, not the average historical prediction contribution:
`u(t)*(P_HW(t)-P_W(t))` can have a nonzero mean and uneven amplitude. Neither
mean alpha nor shared lambda establishes equal effective model capacity.

For s=0, invoke the original constant-blend alpha0.5 arithmetic verbatim,
including its operation order, and require exact score replay against that
half-blend cell at all five penalties. Do not obtain zero slope through a new
residual recombination or alternative arithmetic path. At individual bins
where alpha is exactly0 or1, assign the literal corresponding W or HW values;
do not multiply an unused component by zero. Interior bins use the expression
above in its stated order. Both components already contain the same B, which
appears once algebraically; never add B again. Score the combined prediction
itself, retaining its error cross-terms, rather than averaging component SSEs.

Create 25 cells in lambda-major, slope-minor order. For each held outer
session, choose a single lambda/slope using only the other fifteen sessions'
already held-direction predictions. The objective is their original primary
increment, averaged equally over sessions within each animal and then equally
over the two animals. Maximize the exact tuple

`(training_primary_score, lambda, -abs(s), -s)`.

Exact score ties therefore favor stronger regularization, then smaller slope
magnitude, then negative slope. No tolerance or near-tie rule is introduced.
The held session's scores, identity contrast, omission behavior and slope
reversal result cannot choose either parameter. This is one global selection
recipe applied in every fold, not an animal-specific or session-specific
choice among different models. Do not add independent component penalties,
an intercept-alpha grid, knots, extra slopes, polynomial degrees or components.

## Exact saved-artifact support and original selection replay

All paths below are relative to `outputs/runs/adaptive_search/`. Require the
complete session and summary statuses, their original replay flags and all
sixteen sessions before publishing any selected panel.

| Fixed family | Session banks | Complete summary |
| --- | --- | --- |
| W | `primary_repeatability_weight_control_attempt001/` | `primary_repeatability_weight_summary_attempt001/primary_repeatability_weight.json` |
| HW | `primary_history_repeatability_weight_control_attempt002/` | `primary_history_repeatability_weight_summary_attempt002/primary_history_repeatability_weight.json` |
| Constant blend | `primary_history_blend_attempt001/` | `primary_history_blend_summary_attempt001/primary_history_blend.json` |
| H2W | `primary_history2_repeatability_attempt001/` | `primary_history2_repeatability_summary_attempt001/primary_history2_repeatability_weight.json` |

W/HW use their literal `<recipe>_predictions.npz` five-penalty candidate
arrays. Reuse the qualified original-component loader and score replay:
native trial IDs/order, direction labels, physical LMP IDs, target arrays
and B must agree exactly. The different input histories are preserved as
metadata: W0:+14, HW−3:+14, H2W effective−1:+14 with its qualified reader
−3:+14; all scored targets remain0:+14. Do not demand equal neural windows
or change metadata to satisfy a current-only helper.

Restore saved Y/B values to the qualified original time-fast trajectory
layout, `stack([trajectory.T.copy(order='C').T ...])`, before score replay.
Preserve literal saved C-order prediction arrays. This restores the original
reduction convention without altering values; no fallback layout, epsilon,
allclose or numerical-tolerance replay is allowed. Require finite float64
predictions and exact original full-support B/SSE/denominator/direction counts.
Replay all five W/HW candidate and wrong-pair score rows.

For the constant blend, reuse its completed 25-cell banks and literal selected
prediction paths recorded in its summary. Independently replay its original
other-fifteen-session `(score,lambda,-alpha)` selector from those banks.
Reconstruct alpha0.5 at every lambda with its original helper and require exact
agreement with the corresponding saved half-blend score cell. At the original
selected lambda/alpha, require exact saved-prediction and full score replay.
The new zero-slope cells must reproduce those half-blend scores exactly.

For H2W, load its saved
`lmp_history2_repeatability_weight_ridge_v1_predictions.npz` arrays and complete
candidate score banks. Require the same original trial IDs, directions,
physical support, Y and B, replay all five candidate/wrong-pair rows, and
independently recompute its original other-fifteen-session lambda selector.
Use exactly that original selected H2W prediction and score as the anchor.
Do not refit H2W, replace its lambda with the new lambda, consult its surrogate
to tune this model, or substitute H2W into the W/HW mixture.

The W and HW original selectors must also independently replay from their
complete banks. An original support, array, score or selection mismatch stops
the attempt; it does not authorize refitting, relaxed equality or replacement
with a nearby report. All existing artifacts and failed attempts are preserved.

## Scores and prospectively fixed descriptive comparisons

Use original `D=sum((Y-mean_coordinate(Y))²)` over all complete native XY
session trajectories. Primary is `(SSE_B-SSE_prediction)/D`; identity is
`(SSE_wrong-SSE_correct)/D`. The wrong-pair score remains the exact mean over
within-direction whole-trial evaluation derangements with predictions and
every fitted transform held fixed. Do not shuffle bins, coordinates or the
schedule within a trajectory; do not turn negative R² values into zero.

Report the selected policy's full sixteen-session rows, lambda/slope choices,
selected alpha at all fifteen bins, slope/sign counts, both animal means,
positive-session counts and every fixed-prediction within-animal omission.
Apply the unchanged primary0.01/identity0.005 margins and original
directional/majority/omission requirements. These checks cannot be replaced
by an improvement over a weaker comparator.

Report direct common-D full-prediction SSE and identity differences against:

1. The four fixed originally selected W, HW, constant-blend and H2W panels.
2. Original W and HW at the new policy's selected shared lambda.
3. The original constant alpha0.5 cell at that same new lambda.

Each difference uses `(SSE_anchor-SSE_new)/D` and
`identity_new-identity_anchor`. The shared-lambda half-blend isolates the
change in mixing schedule at that fixed pair of saved component predictions;
it does not match effective degrees of freedom or eliminate selection effects.
Keep the fixed originally selected blend contrast separate from this
shared-lambda half-blend contrast. These are descriptive anchors, with no
added threshold, noninferiority rule, fitted model or scientific count.

For each held session, also use the already computed cell at the *same*
selected lambda and reversed slope−s. Report its full prediction/identity
scores and the direct selected-minus-reversed contrasts, without retuning.
At s=0 the reversal is a literal duplicate with zero contrast. Reversal
preserves the schedule's mean, range and linear form while exchanging its
early/late ordering. Report every selected slope sign alongside the contrast:
mixed signs support at most the learned allocation policy, not one stable
early-history/late-history pattern. This finite internal contrast is neither
a training-pairing null nor phase randomization, and has zero scientific,
literal-null and alternative-check count credit.

No selected slope measures physiological latency, neural phase, causal timing,
spike contamination, population dynamics or online capability. The existing
HW/H2W capacity controls and old temporal tensor controls do not automatically
satisfy all required finalist falsifiers for a new mixture. Any such retest
must follow the existing finalist requirements and a separate authorized
specification; this proposal adds no automatic coverage obligation.

## Exact duplication, patience and bounded execution if adopted

After complete outer selection, compare the entire selected prediction panel
literally with each of the four original selected W/HW/blend/H2W panels.
If all sixteen sessions exactly reproduce any one fixed anchor, charge zero
and name every matched anchor. Similar scores, numerical closeness, a shared
lambda, zero slope alone or matching different anchors in different sessions
does not establish a global duplicate. Do not force a nonzero slope to obtain
new trial depth. Save exact per-session/per-anchor equality evidence and the
selected prediction arrays for replay.

Otherwise this is at most one distinct comparable original-support candidate.
The 25 cells, four original anchors, shared-lambda diagnostics and slope
reversal contribute no additional trials. Literal-null increment is0 and
alternative-explanation increment is0. A distinct candidate may reset
patience only through a strictly higher complete animal-balanced primary
score than the incumbent actually in force at completion. Identity gain or
a matched-lambda contrast cannot reset it. Root supplies the actual incoming
counts and ordering at adoption/completion; none is inferred here.

Retain the original28–64 scientific-depth limits, adopted conditional
twelve-trial patience, >=40% post-initial alternative allocation, all required
coverage/adaptive/challenger obligations, 2,000 core-hours, 120-hour clock and
other ceilings. This optional proposal neither creates a new requirement
nor permits work beyond a valid stopping condition. No development result
authorizes promotion or supplies the absent sealed future sessions.

If adopted, use isolated new tools and stream complete saved artifacts one
session at a time, with no provider source read and no neural refit. Preserve
old arrays, banks, selectors and code. Focused independent fixtures should
check exact zero-slope/endpoint-bin arithmetic, full-prediction cross-terms,
native target/B layout replay, all four original selections/anchor arrays,
held-session exclusion, all three exact tie levels, opposite-sign reversal,
whole-trial wrong pairing and independent global duplicate accounting.

Propose a fixed complete Chewie_20150313 resource pilot on1CPU/2GiB with one
numerical thread,60seconds internal/90seconds external. It must include all
25 cells, all four saved-anchor replays and fixed descriptive comparisons;
its scientific scores remain closed until all sixteen sessions complete.
The largest known1038-trial session needs about2.49MB for the two float64
five-penalty component prediction arrays; additional anchors, JSON metadata
and transient mixtures contribute separately. Stream one mixture at a time
rather than retain25 temporary predictions. This is an array-size estimate,
not a guaranteed peak RSS or runtime.

Root chooses a finite full-panel bound only from the completed pilot's
resources and available allocation, preserving that pilot and requiring
fresh remaining/summary paths. No full bound, retry, solver change, extra
grid cell, new allocation or executable command is selected by this
document. Root retains sole compute/scheduler ownership.

Root resource selection, 09:46 UTC: the fixed pilot completed in job 46473740
at 09:45:38.010208–09:45:43.664111 UTC, 5.653899 seconds, exit 0; saved-artifact
read/replay 5.598201 seconds, mixture scoring 0.054703 seconds, peak 466,612 KiB.
All six original replay flags passed on 1,038 trials and 86 electrodes with
zero source reads/refits. Scientific pilot scores remain closed. Preserve
the pilot and run fifteen fresh sessions sequentially with one thread/2 GiB,
300-second internal and 330-second external aggregate bounds, unchanged
60/90-second children. Fifteen pilot equivalents are 84.808483 seconds;
a full 90-second child reserve plus sixteen pilot equivalents for summary
replays (90.462382 seconds) leave 34.729136 seconds for overhead/variation.
Require 360 seconds of actual allocation time including cleanup. Fresh
roots are `primary_timevarying_history_blend_summary_attempt001` and
`primary_timevarying_history_blend_panel_attempt001` under adaptive_search.
Incoming scientific counters remain 37/patience 4, with no concurrent search
writer. No extension, retry or new scientific choice is selected.
