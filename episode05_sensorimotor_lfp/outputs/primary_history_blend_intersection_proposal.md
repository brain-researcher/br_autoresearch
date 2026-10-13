# Does the actual constant blend retain its benefit on the fixed guide intersection?

**Adopted by root after independent scientific review, 2026-10-04 09:54 UTC.**
Isolated implementation and focused fixtures are authorized, followed by the
specified bounded source pilot after passing checks in a new verified task
allocation. No full-panel bound or allocation has yet been selected. Evidence ends at
complete comparison38, patience5: the time-varying blend has primary
0.002646676360074717 and identity0.004602381426280034, below the constant
blend's primary0.0026788544310999856. The incumbent therefore remains the
original constant W/HW blend, identity0.004559394718773168. Both incumbent
practical margins fail. Incoming alternatives19 and literal nulls13 are among
32 post-initial comparisons. Root owns actual completion ordering and budgets.

## One unresolved comparison, with no new coverage obligation

The [post37 assessment](primary_post37_blend_support_vs_pls_assessment.md)
identifies a specific limit of the completed HW intersection comparison:
it did not fit current-only W on those supports. It therefore cannot establish
whether W and HW retain their complementary prediction errors there. For
full native-XY errors eW=Y-PW and eHW=Y-PHW,

`SSE_blend=(1-alpha)^2*SSE_W+alpha^2*SSE_HW+2*alpha*(1-alpha)*sum(eW*eHW)`.

The missing reduced-W errors and cross-term distinguish this calculation
from relabeling the earlier HW support test. Use the **actual original
constant-blend policy**, with its original finite lambda/alpha grid and
outer selector. The completed time-varying family does not become a component
or change the grid. Do not fix alpha to the observed full-support winner0.5;
that would test a different policy from the original selected blend recipe.

This is optional support robustness on exposed development data. It creates
no new mandatory coverage, practical margin, finalist gate or reason to extend
patience. It does not identify or remove spike contamination, demonstrate an
online mechanism, or supply the absent future-session confirmation. No PLS
family, additional history, band, population target or trial reduction is
included.

Freeze order and proposed recipe names:

1. `lmp_W_HW_blend_unit_intersection_v1` (I-blend).
2. `lmp_W_HW_blend_unit_intersection_random_support8_v1` (R-blend): mean of
   eight support-specific mixture losses at I-blend's selected parameters.

Both are nonpromotable changed-support stress comparisons. Neither can
replace the original-support incumbent or reset patience.

## Frozen metadata, complete prior banks and source boundary

Reuse the complete immutable metadata
`runs/source_support/primary_intersection_metadata_attempt002/primary_intersection_metadata.json`.
All paths here are relative to `outputs/` unless stated otherwise. Before
new numerical values, reconcile all sixteen session/trial descriptions with
the original trial table, original scalar/HW source identities and the complete
I/R HW session banks using the existing qualification/validation functions.
No new guide pass, join, draw, source requalification or session selection is
proposed. Require the already frozen exact nine support entries per session:
I, then ordinary draws r=0,...,7, with physical IDs/positions, full queues,
seed literals, source trial indices, directions and preserved unit labels.

The unchanged ordinary support seed literal is
`ep05-HW-unit-intersection-random-support-v1|20261004|<session>|<replicate>`.
Validate its existing Python Random version2 shuffle/take/order convention
against the saved queues; never redraw or condition on support overlap or
scores. The released metadata contains strict subsets in all sixteen
sessions. Exact repeated physical sets, if present in the frozen records,
must reuse one W fit and retain all requested draw records.

Use these complete artifacts:

| Artifact | Preserved path |
| --- | --- |
| Original trials | `runs/source_support/attempt002/trial_identity.csv` |
| Original scalar support | `runs/adaptive_search/primary_reference_attempt001/` |
| Reduced HW I/R banks and arrays | `runs/adaptive_search/primary_intersection_sensitivity_attempt001/<session>/score_banks.json` and `new_fitted_predictions.npz` |
| Original selected HW I/R report | `runs/adaptive_search/primary_intersection_sensitivity_summary_attempt001/primary_intersection.json` |
| Original full-support W arrays | `runs/adaptive_search/primary_repeatability_weight_control_attempt001/` |
| Original full-support HW arrays | `runs/adaptive_search/primary_history_repeatability_weight_control_attempt002/` |
| Original constant-blend cells | `runs/adaptive_search/primary_history_blend_attempt001/` |
| Original selected constant blend | `runs/adaptive_search/primary_history_blend_summary_attempt001/primary_history_blend.json` and its literal selected prediction paths |

An unavailable or incomplete dependency stops this specification, not a new
fit or a substitute report. Preserve all earlier files and failed attempts.

The original read-only provider root is the exact source path bound by
metadata002. If adopted, the existing qualified `load_lmp_history_session`
may read native LMP−3:+14 and native XY0:+14 once per session. Its existing
eighteen-bin reader preserves the qualified layout; new W uses only the
current fifteen-bin view. No spike values, power-band values, alternate
window, trial exclusion or target transform are allowed. Guide membership
remains metadata, not a signal-quality or contamination label.

## Refit only reduced current W; replay every reduced HW view

For each distinct reduced physical support, perform precisely
`selected_x=x[:,:,positions]`, then `current=selected_x[:,3:18,:]`, with
positions in original physical order. Retain this advanced-channel-indexing
then basic-current-slice layout. Do not force C order, use the trial-subsetting
helper, reorder trial/bin/channel axes, or copy Y through advanced indexing.
Pass the original native Y, labels and trial IDs unchanged to all views.

Fit only the true-pair arm of original
`lmp_repeatability_weight_ridge_v1` on that current view. Its eight direction
folds retain the following exact arithmetic and fitting boundaries:

- Fit the original equal-direction first-harmonic direction/time template
  B0 from the seven fitting directions. Compute
  `residual=(Y_train-B0_train).reshape(-1,2)`, its pooled coordinate intercept
  c, `U=residual-c`, and held-direction full baseline `B=B0_test+c`.
- Use unchanged `primary_pca.prepare_split` population mean/SD over fitting
  trials and fifteen bins for each retained physical channel. Zero SD uses
  divisor1. Compute the existing direction-balanced T/P/r statistics from
  these standardized fitting trajectories only: each of the seven directions
  contributes equally, `r=clip(P/T,0,1)` with the original zero-energy rule,
  and `a=0.1+r`. No outcome or held-direction value sets the metric.
- For flattened fitting Z, `G=Z.T@Z/n`; apply original current-W
  `trace_weights(a,G)`, namely `sqrt(a*trace(G)/(a dot diag(G)))`, including
  its exact zero-Gram and uniform-active-weight identity branches. Recompute
  the reduced-support trace normalization; do not subset full-W weights,
  borrow HW history normalization, or add history-column centering.
- Use the existing current-W ridge helper and all five penalties
  `[0.001,0.01,0.1,1,10]`: solve the native two-coordinate residual problem
  with `Phi.T@Phi+n*lambda*I`, add the prediction to B, and preserve the
  original solver/operation order. No post-weight standardization or
  centering, weight-label surrogate, new head or hyperparameter is fitted.

The old reduced-HW `new_fit_metadata[fit_key].metadata` already stores each
fold's current-only `physical_mean`, `physical_sd` and complete
`repeatability` dictionary. Require the newly computed W values to match
these records exactly for every view/fold. This checks the intended current
input and fitting boundary while keeping the different W and HW decoder
normalizations separate. A discrepancy is a technical failure; no tolerance,
alternate layout or borrowed metadata is a repair under this proposal.

Reuse each view's five saved reduced-HW predictions through its immutable
`fit_key` routing and `fitNNN_candidate_xy_by_lambda` array. Never rerun an
HW fit. Require complete original I/R statuses, qualified support identities,
exact physical IDs/positions, trial IDs/directions/order, native target Y and
B. Replay all five correct/wrong-pair score rows of all nine views, their
baseline SSE/R2, original D and direction counts against the old session
banks; independently replay the old eight-SSE mean bank. Require finite
float64 C-order prediction arrays. When replaying saved Y/B, restore the
qualified original time-fast layout with the existing helper; do not alter
values or select a fallback reduction convention. Newly computed B must
exactly match the original full-support and every reduced-HW B.

For an exactly full physical set, if such a record were present, reuse the
original W/HW arrays after exact score/support replay instead of refitting.
No full-W or full-HW refit is authorized. This branch does not manufacture a
full-support view absent from the frozen metadata.

## Original constant blend, controlled R and exact scores

For each of the nine saved supports and each original shared lambda, form
`P=(1-alpha)*P_W+alpha*P_HW`, using exactly the original five constant weights
`alpha=[0,0.25,0.5,0.75,1]`. Reuse the unchanged original blend helper:
alpha0/1 returns the literal component array, and interior arithmetic keeps
the original operation order. B is already present once in each component;
do not add B again. Score full mixture predictions, retaining their cross
terms. Do not average endpoint SSEs or fit a mixture of coefficients.

Across complete I-blend banks, each held outer session chooses one of25 cells
using only the other fifteen sessions' primary increments, equal session
weights within each animal and then equal animal weights. Use the original
exact `(training_primary_score,lambda,-alpha)` maximum: stronger lambda,
then smaller historical weight on exact ties. No near-tie tolerance, identity
tuning, per-support winner, new shared-lambda choice or held-session choice.

For R-blend, evaluate all eight support-specific mixtures at that session's
**same I-blend-selected lambda and alpha**. Average the eight correct SSEs
and eight exact wrong-pair SSEs, then compute primary/identity with original
B and D. Never average predictions, tune an R selector, select a seed, or
average per-seed winning cells. Retain all eight ordinary draws, including
literal repeats, fixed support overlaps and unfavorable signs.

Use `D=sum((Y-mean_coordinate(Y))^2)` on the original complete native XY
trajectories. Primary is `(SSE_B-SSE_P)/D`; identity is
`(SSE_wrong-SSE_correct)/D`, with the original exact within-direction
whole-trial derangement mean and every fitted transform held fixed. Retain
negative values. Both animals remain distinct biological units, and Chewie's
two implants remain one animal. Views, seeds and directions do not add animals.

## Necessary fixed anchors and descriptive comparisons

Independently replay the full-support original constant-blend selector from
its complete25-cell banks, including original other15 membership and exact
tie rule. Replay the corresponding literal selected full-support arrays,
their selected lambda/alpha, native support and full scores using the saved
original W/HW components and original helper. Do not retune the full blend.

Also independently replay the previous I-HW other15 lambda selector from its
complete five-penalty I banks, stronger-lambda exact ties, and confirm that
the previous R-HW report used that same lambda for all eight supports. Replay
the old selected I/R rows and per-draw arrays/losses before using them as
fixed anchors or duplicate evidence. This old selector is separate from the
new I-blend lambda/alpha selector.

Report the following direct common-D full-prediction contrasts, with identity
differences on the same trial pairs:

1. I-blend minus mean R-blend, and I-blend minus each ordinary R-blend draw.
2. I-blend and mean R-blend versus the original selected full-support blend.
3. I-blend versus intersection HW at the **new** selected lambda; each
   R-blend draw versus its own HW at that same new lambda, and their mean.
4. I-blend versus the previously selected I-HW; mean R-blend versus the
   previously selected R-HW and their corresponding per-draw differences.

The matched-lambda contrasts isolate adding the current component within a
fixed pair of predictions. The original I/R-HW contrasts additionally include
any change in selected lambda. Keep them distinct. No independent full-W,
full-HW, H2 or time-varying performance anchor is added. All performance
contrasts use `(SSE_anchor-SSE_new)/D`; none defines a new pass threshold.

Retain selected lambda/alpha, actual physical supports/overlaps, all sixteen
session signs, both animal means/positive counts, fixed-prediction omissions,
every R draw and finite draw ranges. Report each reduced support's selected
component SSEs and `sum((Y-PW)*(Y-PHW))/D` as a descriptive error cross-term.
The score is always computed directly from P; a rearranged cross-term formula
does not replace the exact scoring/replay path. No per-bin analysis chooses
or changes any parameter. State current-W and saved-HW selected covariance
trace/df separately; equal support count or lambda does not match capacity.

Apply the unchanged primary0.01 and identity0.005 reference screens with
original animal/majority/omission descriptions. Changed-support success
cannot rescue a failed original primary margin. I-minus-R is association
with recorded guide membership at fixed channel count, not a direct estimate
of spike contamination, anatomical matching or signal-free performance.

## Independent duplicate counts, patience and resource-only pilot

After complete outer selection, evaluate exact full-panel identities before
charging either recipe. I-blend is zero charge if all sixteen literal selected
prediction arrays equal the previously selected I-HW panel; alpha1 alone is
insufficient when its selected lambda differs. Also check exact equality to
the preserved full-support blend panel. R-blend is independently zero charge
if every one of its ordered eight selected prediction arrays in every session
equals the corresponding **previously selected R-HW** array, at the previous
I-HW-selected lambda. Mean score equality cannot establish this identity.

Also collapse R-blend if all eight predictions in every session literally
equal the current I-blend prediction panel, or all equal the same already
evaluated deterministic I-HW or full-blend panel. Conversely, I-blend may
collapse to the prior R-HW ensemble only if all eight prior draw predictions
in every session literally equal its deterministic prediction. These fixed
checks prevent ensemble averaging or matching different comparators by
session from manufacturing a global duplicate. Save all per-session/draw
equality evidence and selected arrays. No epsilon, score closeness,
allclose, small-energy heuristic or unproved algebraic equivalence is allowed.

A reduced-W alpha0 endpoint is a new reduced-support prediction unless one
of these exact complete-panel checks proves prior reuse. It is **not**
automatically full-W, a previously evaluated HW endpoint, or an old control.
Do not force alpha away from endpoints, redraw supports or alter ties to earn
depth. Repeated physical sets reduce fit work; they do not automatically
make a selected recipe globally duplicate when the selection rule differs.

Process I-blend first and mean R-blend second. Each distinct complete recipe
adds one valid stress/alternative comparison, zero literal-null controls and
one patience step, with no incumbent reset. At most two are charged. If both
are distinct with unchanged incoming state, the conditional transitions are
39/patience6 then40/patience7, alternatives21/literal nulls13 among34
post-initial comparisons. These are prospective arithmetic, not completions.
Technical failures, partial panels, reused arrays and duplicate recipes add
zero. Root applies actual live counts. Retain original28–64 depth,
conditional twelve-trial patience, >=40% post-initial alternative allocation,
all adaptive/challenger obligations,2,000 core-hours and120-hour ceilings.
This optional proposal grants no extension or new positive-result claim.

If adopted, use isolated tools and a fixed **Chewie_20150313** resource pilot
selected by metadata002, without new score-based selection:1038 original
trials,45 of86 physical electrodes, nine distinct frozen supports. Request
one CPU/6GiB and one numerical thread,270seconds internal/300seconds
external. Include all nine reduced-W fits, all reused-HW views/five penalties,
all25 blend cells per support, the original selected-blend and old I/R-HW
replays, and fixed contrasts. Keep the pilot's scientific scores closed until
all sixteen sessions are complete.

For this pilot, at most9*8*5=360 current-W ridge heads are needed, of width45;
no HW head is new. Original x occupies about12.9MB of float64 array payload;
one five-penalty native-XY prediction bank about1.25MB and all nine reused-HW
banks about11.2MB, before metadata, source-library buffers and fit workspaces.
Stream support mixtures instead of retaining225 prediction arrays. These are
metadata-based payload estimates, not a runtime or RSS guarantee. The pilot
must measure source, replay, reduced-W fit and mixture-score times separately.

Root's resource choice at adoption uses 6 GiB for the attended pilot/test
allocation because the preceding mixed allocation reported step MaxRSS
5,929,052 KiB, despite substantially smaller individual runner peaks. This
is conservative allocation sizing, not an attribution of that peak to any
one recipe or a change to the scientific method. Actual resource observations
will determine the remaining panel's bound and memory request.

Root selects a finite sequential full-panel/summary bound from the completed
pilot resources and actual allocation. No remaining bound, automatic retry,
concurrency escalation, renewal or future allocation is selected here.
Proposed fresh roots under `runs/adaptive_search/` are
`primary_history_blend_intersection_attempt001`,
`primary_history_blend_intersection_summary_attempt001` and
`primary_history_blend_intersection_panel_attempt001`.

Independent focused fixtures should cover channel/current layout and exact
saved prior-state replay, held-direction exclusion, unchanged B/Y/IDs/D,
literal endpoints and actual error cross-terms, every reused HW view, original
versus new outer selectors, mean-eight-SSE at I lambda/alpha, and independent
I/R zero/one/two accounting including alpha1 with a changed lambda and a
new reduced-W endpoint. Only root executes tests, source reads and jobs.

## Selected remainder resources, 2026-10-04 10:07 UTC

The fixed pilot completed in job 46476174 with all six replay guards true:
65.504663287 seconds total, 4.330372777 global metadata/selector replay,
0.651189520 artifact replay, 56.843372672 source reading, 2.692883464 reduced-W
fitting and 0.612290813 mixture scoring; maximum RSS 263,348 KiB. All nine
reduced supports and 360 ridge heads completed. Scientific scores remain
closed. Ten independent fixtures and ASTRA validation passed beforehand.

Root selects one sequential worker in the same one-CPU/6-GiB allocation,
**1440 seconds internal / 1470 seconds external** for the fifteen remaining
sessions plus final summary. The unchanged child bounds are 270/300 seconds;
the dispatcher retains its full 300-second launch reserve and drain policy.
Fifteen pilot durations total 982.569949 seconds. Adding the 300-second
child reserve and a conservative summary estimate of sixteen times the
pilot's global/artifact/mixture durations (89.501650 seconds) gives
1372.071599 seconds, leaving 67.928401 seconds inside the aggregate bound.
This resource extrapolation is not a guarantee and changes no scientific
choice. Require at least 1500 seconds before the actual 10:34:02 UTC expiry
when dispatching. Do not launch if that requirement is no longer met.

Preserve the completed pilot and require fifteen fresh session paths and
fresh parent/summary outputs. Incoming counters remain 38/patience5,
alternatives19/nulls13. No retry, automatic renewal, source concurrency,
partial-panel release or budget extension follows from this selection.
