# Does the strongest fixed LMP model survive matched trial and electrode budgets?

Prepared 2026-10-04 after the complete all-band result; **adopted by root at
08:19 UTC after root and independent written review**, including the
metadata-selected Chewie_20161014 resource pilot. Implementation and focused
fixtures are being prepared; this paragraph makes no execution claim.
Root owns adoption, execution, resource decisions and the scientific log.
Only released complete results, existing code and bounded support metadata
were used. Incomplete elastic-net scores remain closed and uncharged.

## Scientific question and fixed pipeline

All-band A completed with primary increment 0.0006719900156145613 and identity
increment 0.0013802590100496538; both practical margins and all five declared
complementarity screens failed. HW, the fixed four-bin history plus current
repeatability prior, remains the largest complete primary increment,
0.0023412544731769847. Its identity increment passes 0.005 but its primary
increment fails 0.01. Resource matching cannot rescue that original failure.

The required next question is whether HW's small development result depends
on unequal numbers of physical electrodes or trials across recordings.
[GOAL.md](../GOAL.md) requires matched trial/channel analysis and stress tests
of the strongest models. Freeze HW's existing scientific recipe from
[its adopted proposal](primary_history_repeatability_weight_proposal.md).
There is no new lag, repeatability floor, frequency block, response, bandwidth
or model family. This is a resource sensitivity, not an alternative winner.

Use a two-axis design with the original full-support HW result as the reused
reference. Three reduced conditions distinguish channel count, trial count
and their joint change. A joint condition alone could not separate the axes.

| Order | Diagnostic recipe | Physical electrodes | Trials in each direction |
| --- | --- | --- | --- |
| C | `lmp_HW_matched_channels8_v1` | 86 | Original session counts |
| T | `lmp_HW_matched_trials8_v1` | Original session electrodes | 12 |
| CT | `lmp_HW_matched_channels_trials8_v1` | 86 | 12 |

Each condition aggregates eight fixed subset views, numbered 0–7. Views are
internal sensitivity replicates, not independent animals, separate recipes
or an ensemble. Do not select a favorable view or add views after scoring.
All three are nonpromotable diagnostics and cannot update the same-support
candidate incumbent, even when C happens to retain all evaluation trials.

## Outcome-blind budgets and original identities

The released durable metadata record is
`outputs/runs/source_support/matched_budget_metadata_attempt001/matched_budget_metadata.json`.
It contains the original sixteen session identities, frozen physical LMP
electrode IDs and eight direction counts. It contains no neural values.

Freeze `E*=min_s E_s=86`: Chewie_20150313 and Chewie_20150319 have 86 original
electrodes, and the other fourteen sessions have 96. All original electrode
lists are ascending IDs 1..E within their own session. These IDs do not mean
that equal numbers identify the same tissue across animals or implants.
Freeze `n*=min_(s,d) n_sd=12`, attained at Chewie_20150629 direction 3.
T and CT therefore retain 96 trials/session, with 84 fitting trials and 12
evaluation trials in each held-direction split. No smaller convenience
budget, channel quality cutoff or outcome-dependent subset is introduced.

Use the original session-qualified trial IDs from
`outputs/runs/source_support/attempt002/trial_identity.csv`. For each session
and view, initialize separate Python `random.Random` instances with
`seed(literal, version=2)`:

- Electrode queue: `ep05-HW-budget-electrode-v1|20261004|<session>|<view>`.
  Start from its ascending original physical IDs, perform one ordinary
  `shuffle`, and take the first 86 IDs.
- Trial queue for direction d:
  `ep05-HW-budget-trial-v1|20261004|<session>|<view>|<direction>`.
  Sort original IDs by their validated source-trial number, perform one
  ordinary `shuffle`, and take the first 12 IDs.

Restore selected electrodes to ascending physical order and selected trials
to their original source order before fitting. The queue is for selection,
not numerical reordering. C and CT share each view's electrode selection;
T and CT share its trial selection. Selections are fixed across all held
directions, penalties and fits, using identities only. Accept chance repeated
subsets and full-support subsets; never redraw or require views to differ.
Keep the exact queues, selected IDs and original counts in the output.

Before any new source values or fits, the coordinated implementation must
reconcile all sixteen metadata records with the original scalar-source and
trial identity records, verify these two global minima and construct every
queue. Any mismatch stops, rather than recomputing a new minimum from a
smaller set. All sixteen sessions, both animals and all eight directions
remain required. There is no per-session or per-animal winner selection.

## Source and fitting paths

Reuse the qualified native LMP history reader: neural offsets -3:+14,
18 bins; native XY offsets 0:+14, 15 bins at 30 ms. Load the original complete
session once and use only declared ID subsets for each fit. No power or
spike values, new interval, smoothing or source transformation is requested.
Qualify original physical guide/trial support before values, as in HW.

Use the original time-fast reader layout. Define subsetting as original-order
trial indexing followed by original-order physical-channel indexing. Preserve
the raw current view `x[:,3:18,:]` for W normalization; do not force it to
C order. Source-like float32/float64 fixtures must check the declared layout.
Do not claim bitwise original-state replay after a genuine subset change.

For every condition/view and held direction, fit on only its retained trials
from the other seven directions. Refit the equal-direction first-harmonic
direction/time template B0, then the common native XY residual intercept c;
both candidate and behavioral comparator use B=B0+c. Refit all input states
on those same fitting observations: W's current physical mean/population SD
and equal-direction T/P/r/a statistics with the fixed 0.1 floor; H's separate
lag/electrode means/SDs and fitting-column center; and HW's single physical
weight shared across four lags, normalized with the whole reduced history
Gram trace. Keep zero-SD→1, zero-Gram/identity branches, lags `[0,-1,-2,-3]`,
native unscaled XY, no post-weight standardization or centering, and the
preserved full-span ridge solver exactly as adopted for HW.

Fit each view at all five original normalized-loss penalties
`[0.001,0.01,0.1,1,10]`. The changed number of fitting rows changes the
`n_fit*lambda` ridge term by the original objective; it does not introduce a
new penalty definition. No full-support physical mean, target baseline,
repeatability estimate or weight is reused for a genuinely reduced fit.

## Scoring, subset aggregation and outer selection

For each session/view, score every selected evaluation trial in every held
direction, retaining all fifteen bins and both XY coordinates. Compute its
own original pooled joint-XY denominator D from precisely those retained
targets. D must be finite and positive; a failure stops the complete panel,
without replacing a view or borrowing the full-support denominator.

Let b, p and q be respectively the baseline SSE, candidate correct-pair SSE,
and exact wrong-pair SSE for that view at a penalty. Wrong pairing uses the
same existing exact mean over all within-direction whole-trial derangements
on the **retained IDs**: each recipient is paired with every other retained
trial in its direction, with equal donor weight. No discarded trial or
cross-direction/binwise/coordinate shuffle enters this null. The same finite
recipient/donor sets apply to the full-HW reference predictions below.

Average each view's normalized losses equally:

`delta = mean_view((b-p)/D)` and
`identity = mean_view((q-p)/D)`.

Likewise report `baseline_R2=1-mean(b/D)` and `candidate_R2=1-mean(p/D)`.
Retain each view's actual D and raw losses. T/CT denominators can differ
between views; **do not divide mean raw SSE by mean D**, use the original
full-session denominator, or combine predictions on overlapping trials.

Each of C, T and CT independently selects one lambda for an outer session
from the mean-view primary banks of the other fifteen sessions, using the
original equal-session then equal-animal selector and stronger-lambda exact
tie rule. Its selected lambda is shared across its eight views in that
session. No view, evaluated session, held direction or animal chooses its
own tuning result. Collapse views before sessions and sessions before animals.

## Full-HW reference on identical evaluation support

Reuse only complete original HW artifacts:
`outputs/runs/adaptive_search/primary_history_repeatability_weight_control_attempt002`
and
`outputs/runs/adaptive_search/primary_history_repeatability_weight_summary_attempt002/primary_history_repeatability_weight.json`.
The original candidate ID is `lmp_history4_repeatability_weight_ridge_v1`.
Original prediction arrays contain all five penalties, true XY, B, trial IDs
and directions. Do not refit full HW or choose a new full-HW lambda.

After reduced fitting is fixed, match arrays by validated original IDs and
evaluate the saved full-HW predictions on precisely each condition/view's
retained target set. Require exact native Y, ID, direction and source-window
replay. Full-HW predictions remain trained on the original seven-direction
support; this is the intended unrestricted-resource comparator. They contain
no fitting access to their scored held-direction trials.

Report both the original independently selected full-HW lambda (from its
other-fifteen-session selection) and full HW at the reduced condition's
selected lambda. For each reference selection and view, use the same D and
retained-ID null. The direct resource contrast is

`prediction_gain = (SSE_fullHW - SSE_reduced)/D`,

not the difference of two gains that use different behavioral baselines.
Its identity contrast is `identity_reduced - identity_fullHW` on the same
retained pairs. Average these normalized contrasts across views, then apply
the original session/animal aggregation.

Also report full HW against the reduced B and its own original B, both on
the retained target set, and `(SSE_Breduced-SSE_Bfull)/D`. This explicitly
shows the contribution of refitting the behavioral template. A comparison of
each model's own-baseline delta combines that baseline contribution with the
direct prediction contrast and must not be mislabeled a pure resource effect.
These saved-reference evaluations and contrasts add no scientific charge.

## Diagnostics, interpretation and exact reuse

Report all sixteen session results; both animal means; primary-positive
counts; fixed-prediction session omissions; selected penalties; view ranges;
and original/retained trials, physical IDs, fitting rows, Gram trace and
selected effective df. View ranges and omissions are descriptive, not
confidence intervals or new biological replication.

For each condition, report the unchanged 0.01/0.005 practical screens and
original both-animal/majority/omission conditions on its own reduced-support
primary and identity readouts. Report positive or negative direct differences
from full HW without introducing a noninferiority tolerance or requiring a
reduced-budget model to outperform the unrestricted one. A positive reduced
increment supports persistence under that specific resource constraint;
failure bounds robustness. None rescues HW's original failed primary margin
or proves matched anatomical coverage, recording quality or equal model df.

If a condition/view's full physical and trial ID sets equal original HW,
reuse its original five-penalty predictions and baseline after exact target
replay; do not refit it. This applies to C's views in the two E=86 sessions.
If CT and T have the same IDs in a session/view, fit once and reuse the
within-attempt result. The same applies to any exact C/CT support identity.
Reuse is based on complete fitting/evaluation identity, not similar scores.

At recipe level, compare ordered eight-view support signatures across all
sixteen sessions. A condition identical everywhere to original full HW adds
zero depth. A later condition identical everywhere to an earlier condition
also adds zero; catch both C=CT and T=CT when an axis is globally unchanged.
Repeated views within a distinct recipe stay allowed but do not add depth.
Any additional algebraic-duplicate claim needs an exact documented proof,
never a score or small-variance tolerance. Do not permute weight labels or
add random features merely to inflate this resource check's recipe count.

## Order, bounded pilot and search accounting

Process scientific recipes in the fixed order **C, T, CT**. Each distinct
complete sixteen-session, eight-view diagnostic is one valid alternative-
explanation/stress comparison. At most three charges advance 27→30 valid
comparisons and 21→24 post-initial comparisons. The eleven existing literal
null-control recipes remain eleven; the three stress diagnostics increase
the conservative alternative-check allocation from11 to14 among24
post-initial comparisons. Pilots, views, penalties, folds, reference reuse and technical
repairs add zero. Partial or failed panels do not charge an unfinished recipe.

Follow the adopted
[prospective patience interpretation](primary_patience_interpretation.md).
No diagnostic updates the true-candidate incumbent. If all three are distinct,
C is comparison28 with patience not started, T is29 with counter1, and CT is30
with counter2. Root records actual transitions only after complete evidence;
duplicate charges are skipped. The original 28–64 depth, 40% alternatives,
12-trial patience, adaptive/challenger obligations, 2,000 core-hours, zero GPU,
120-hour clock and all other limits remain unchanged.

Propose one complete fixed **Chewie_20161014** pilot, all three conditions/eight
views/five penalties, one CPU with 6GiB, 270 seconds internal and 300 external.
Choose it from support metadata only: at 96 electrodes and 740 original trials
it is the largest session requiring genuinely reduced channel-only fits, and
it exercises distinct C/T/CT paths. The earlier usual Chewie_20150313 pilot
would reuse C and collapse T=CT, so would underrepresent this new workload.
Its full history is 740×18×96, about10.2MB float64. Channel matching yields
344 history columns; trial-only matching retains the original384. The largest
original source history elsewhere is1038×18×86, about12.9MB. These are shape
facts, not measured runtime or peak RSS. Evaluate conditions/views sequentially
within a session. Choose any remaining-session concurrency and aggregate bound
only from complete pilot resources; no automatic retry or resource escalation.

Fresh proposed roots are `primary_matched_budget_attempt001`,
`primary_matched_budget_summary_attempt001`, and
`primary_matched_budget_panel_attempt001` under `outputs/runs/adaptive_search`.
Preserve all earlier tools, bank arrays and failed attempts. New focused
fixtures must independently check outcome-blind queues/global minima, exact
ID nesting and duplicate reuse, reduced fitting-only B/H/W states, no removed
trial leakage, unequal-view denominator aggregation, retained-ID wrong pairs,
direct full-HW loss contrasts and baseline decomposition, outer exclusion and
the C/T/CT charge order. Root and independent review precede implementation.
