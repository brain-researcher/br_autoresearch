# Experiment log

## 2026-10-11 00:42 Pacific — primary_result_figure — rendered and visually inspected

The completed-report [PNG](figures/primary_evaluation/ep07_primary_result.png)
and [PDF](figures/primary_evaluation/ep07_primary_result.pdf) were rendered
successfully in parent47352162. Root inspected the PNG: all five days, both
comparators and aggregate values are readable; the 0.01 line is restricted
to the aggregate row and the failed support conditions are explicit. No
model, neural outcome or bootstrap calculation was repeated. Changed ASTRA
validation passed. The figure is a display of the closed primary result.

The parent completed after 49 seconds with one allocated CPU. Its figure
interval was five seconds, charged **0.0013888889 allocated CPU-hours to EP07**;
the remaining 44 seconds are charged once to EP06. Release the shared
reservation. No related job remains live and the completed scientific result
is unchanged: neither region meets the full prespecified support rule.

## 2026-10-11 00:39 Pacific — primary_result_figure — companion submitted

The completed-report figure is included in EP06 parent **47352162**, requested
1 CPU/7 GiB/one hour. Its separate at-most 95-second interval is included in
the shared at-most 2 CPU-hour reservation and will be charged to EP07 once.
Only the figure script and changed ASTRA are consumed in this scope; no EP07
model, control, primary opening or inference is repeated. The negative primary
result and completed-job accounting remain unchanged.

## 2026-10-11 00:36 Pacific — primary_result_figure — completed-report plot selected

The new `tools/plot_primary_result.py` has root source review. It reads only
the completed opening and primary report, displays both existing comparator
differences for every day and the reported aggregate, and distinguishes day
sign zero from the aggregate-only 0.01 margin. It calculates no uncertainty,
fits no model and opens no neural payload. PNG/PDF destinations are exclusive;
any failed partial rendering is preserved. Visual inspection follows rendering.

Select this bounded plot as a separately scoped companion in EP06's forthcoming
first-twelve allocation. Its timeout is 90 seconds plus five seconds of kill
grace, with one numerical thread. Record its start/end epochs and charge the
actual interval to EP07 once, as a portion of the shared at-most 2 CPU-hour
parent reservation. No new scientific comparison or primary opening follows.
No plotting job or figure has run at this entry.

## 2026-10-11 00:31 Pacific — primary_evaluation — terminal accounting and independent interpretation

Job **47351138 completed, exit 0:0**, with 278 elapsed seconds, one allocated
CPU, TotalCPU 3:06.953 and batch MaxRSS 3475652K. Its EP06 preparation interval
was 143 seconds (epochs 1791703669–1791703812), so charge **135/3600 = 0.0375
allocated CPU-hours to EP07** and **143/3600 = 0.0397222222 to EP06**. The
parent total is 0.0772222222 hours, charged once. Release the full shared
reservation. The longer extern-step cleanup is not a second parent charge.
All consumed source is now unfrozen; immutable attempts/results remain intact.

Independent Codex report/source review found no inconsistency in the negative
primary conclusion. Both regions have available inference and all 9999 draws;
M1 fails the required day-frequency and omission criteria, while PMd fails the
0.01 margin. The earlier BR directive `ext_review_dir_12d86d49bdd4` supplied
generic advisory criteria only. It did not inspect this report or issue a
verdict; this is not canonical BR acceptance. The primary is closed without a
supported region. Its conditional explanatory work cannot rescue this result.
A source-backed descriptive figure is being prepared from the completed report;
it will not refit, resample or reopen outcomes.

## 2026-10-11 00:30 Pacific — primary_evaluation — completed; neither region meets the full support rule

The single primary evaluation completed in job47351138. Analysis choices
froze at07:25:44.955802UTC; the spike-outcome opening ran from07:27:42.080333
to07:27:49.155259UTC. All freeze/fit/predict/score exits are zero and the
[opening record](runs/primary_evaluation/attempt001/opening.json) is complete.
The [primary report](runs/primary_evaluation/attempt001/primary_report.json)
contains all five fixed days, both regions, both comparators and9999 valid
paired whole-trial bootstrap draws. No missing draw or day was dropped.

Neither region meets every prespecified support condition:

| Region | History minus today-only R² | History minus calibration mean R² | Full-rule failure |
| --- | ---: | ---: | --- |
| M1 | +0.0172749073 | +0.0885562519 | Only1/3 Chewie days improves against both comparators; omitting October7 leaves Chewie history-minus-today negative. |
| PMd | +0.0077719354 | +0.1563031879 | History-minus-today is below the fixed0.01 margin. |

Differences weight days equally within animal, then animals equally. M1's
margin test passes (regional p0.0002, two-region Holm-adjusted p0.0004), but
this cannot override the failed stability criteria. Its Chewie today-only
differences are +0.0233082,−0.0108081,−0.00553789 across October7/14/21;
omitting October7 gives−0.00817301 in Chewie. PMd improves against both
comparators on all five days and passes the sign/omission requirements, but
its regional and adjusted margin p are0.9923. The report correctly lists
`supported_regions: []` and `prespecified_support_rule_not_met` for both.

This is a completed real-data evaluation with limited, heterogeneous gains,
not evidence that history never helps. Inference is conditional on these five
days, two animals and frozen fits; public-release/prior-campaign exposure and
same-account procedural separation remain disclosed. No coupling, mean-only
mechanism, independent replication or canonical BR acceptance is established.
Residual/pairing explanatory analyses were not performed and cannot rescue
the failed primary rule. Do not select another finalist or change a threshold
using these opened outcomes.

At07:29:10UTC the parent remains RUNNING on sh03-06n11 with actual1 CPU/7GiB;
reduce the shared reservation to1 allocated CPU-hour. Conditional EP06
preparation started at epoch1791703669, after primary completion, and has a
nested at-most0.25-hour share. Final accounting waits for the terminal parent
and splits that measured interval once between episodes.

## 2026-10-11 00:24 Pacific — primary_evaluation — unstarted primary submitted

Submitted **Slurm job47351138** with independently reviewed
`tools/finish_primary.sbatch`,1 CPU/7GiB/zero GPU/one hour. The predecessor is
scheduler-confirmed terminal and its failure is diagnosed. The new script
fixes only the empty-argument-array launch issue and consumes all completed
qualification/control evidence without repetition. Primary attempt001 and
conditional EP06 development001 are still unstarted at submission. Preserve
the shared at-most2 allocated CPU-hour reservation until actual allocation is
observed, with no separate additional EP06 reservation. Consumed source remains
frozen until this allocation ends. Standalone Codex/Slurm launch; no canonical
BR campaign or acceptance action.

## 2026-10-11 00:24 Pacific — primary_evaluation — Bash launch defect isolated; unstarted stages selected

Job47350382 ended FAILED1:0 after103seconds with1 allocated CPU, TotalCPU
1:27.333 and batch MaxRSS391332K. Charge **0.0286111111 allocated CPU-hours once
to EP07** and release the1-hour shared reservation. EP06 preparation never
started, so no portion is charged to EP06. All10 new tests, both ASTRA
validations and the complete four-case recovery result remain passing.

The script failed before invoking the first primary mode: older Bash under
`set -u` rejects expansion of the deliberately empty `ep07_args` array in
freeze mode (`slurm-47350382.err`, line117). No freeze log, analysis freeze,
fitted-state attempt, held feature access or outcome opening was created.
This is an ordinary script compatibility defect, not a system failure or
scientific failure. Preserve the original batch and all completed evidence.

Select `tools/finish_primary.sbatch` for only the four unstarted primary modes
and the unchanged conditional EP06 preparation. Every mode now builds a
nonempty argument array containing mode/output/deadline before adding any
role-specific arguments. Reuse the actual passing predecessor checks and
strict completed recovery validator; do not repeat tests, simulations,
empirical search or controls. Request1 CPU/7GiB/zero GPU/one hour, allow1--2
actual CPUs only for memory rounding, all numerical threads1. Reserve at most2
allocated CPU-hours total, including the at-most0.5 conditional EP06 portion;
split actual stage time once on completion. Existing800CPU-hour/96-hour limits
and current storage estimates persist. The only ASTRA change is its selected
primary batch command; no scientific implementation or inference changes.
Submit after focused independent source review. No new user decision is needed.

## 2026-10-11 00:21 Pacific — known_answer_cases — all four fixed recoveries passed

The numerical continuation completed all remaining30 fits in77.0722537041
seconds, preserving the first passed cell. The complete report has16 passing
case/region/stream cells,32 unique completed member tasks and33 cumulative
attempts including the original failed numerical attempt. Actual cumulative
active wall time is90.8443455696 seconds; conservative charged time91.0722537041
seconds, inside600. Cumulative process CPU is84.187011535 seconds and continuation
peak RSS507375616 bytes. The exact continuation validator passed; no additional
retry, case, seed, threshold change or successful-cell repeat occurred.

Evidence: [completed continuation](runs/alternative_explanations/qualification_repair001/known_answer_report.json).
These are bounded method checks, not biological findings. The unchanged fixed
finalists may now proceed through the selected full freeze, native final fit,
paired prediction publication and single primary evaluation in parent47350382.
At this observation only `known_answer_exit=0` is recorded; no completed held-out
result or EP06 neural access is claimed yet.

## 2026-10-11 00:20 Pacific — primary_evaluation — running; continuation and EP06 component checks passed

Job47350382 is RUNNING on sh04-04n22 with actual1 CPU/7GiB. Reduce the shared
parent reservation to1 allocated CPU-hour, including at most0.25 hours for
conditional EP06 preparation. Both new continuation checks passed (0.051s),
all eight separately scoped EP06 checks passed (9.360s), and both changed ASTRA
programs validated. The remaining fixed qualification fits are running.
Evidence: `qualification_repair001/continuation_checks_47350382.txt` and
`qualification_repair001/status_47350382.txt` under alternative_explanations.
No completed qualification or primary opening is claimed at this observation.

## 2026-10-11 00:19 Pacific — primary_evaluation — bounded continuation submitted

Submitted **Slurm job47350382** using `tools/resume_primary.sbatch`, requesting
1 CPU/7GiB/zero GPU/one hour. Preserve the shared at-most2 allocated CPU-hour
reservation until actual resources are observed. The original report and all
39 paired empirical evaluations per stream remain intact. The selected new
continuation writes only qualification_repair001; primary evaluation remains
the existing unstarted attempt001. EP06's conditional development001 preparation
has its separate scope and test/access gates. All consumed source is frozen.
This is standalone Codex-to-Slurm execution, not a canonical BR harness launch.

## 2026-10-11 00:19 Pacific — primary_evaluation — continuation batch selected for submission

Select [resume_primary.sbatch](tools/resume_primary.sbatch), requesting1 CPU,
7GiB,zero GPU and one hour on the already discovered russpold partition.
Allow1--2 actual CPUs solely for memory rounding, with all numerical thread
counts fixed at1. Reserve at most2 allocated CPU-hours for the entire parent;
at most0.5 of those hours cover the separately scoped conditional EP06 export,
not an additional reservation. After completion split the parent's actual
elapsed allocation cost using the recorded EP06 preparation interval; never
charge the same interval to both episodes. Original episode clocks/costs remain.

The batch aggregates the remaining30 fixed recovery fits, the already selected
eligible final fit/predict/score sequence and, only after completed EP07 primary
opening, EP06's seven-session development export. Expected useful work is
approximately10--25minutes with below3GiB additional combined storage; no
padding or repeated empirical control/search work. Recovery has581 seconds
until termination plus at most5 seconds of kill grace, inside the586-second
remaining allowance. EP06 export has at most900 seconds including kill grace.

Independent source review passed for continuation/validator/two new tests,
the primary eligibility change and batch, and EP06's coordinator/exporter/
reader/classifier plus eight focused checks. All ten new checks run on compute
before their dependent stages. Passing existing regression, control, final
transport and inference checks are reused. No actual qualification repair,
primary mode or EP06 export has run. The latest own-user queue is empty;
there is no duplicate pending job. Quota inspection shows scratch5.5/100TB
and Oak171.9/250TB; no environment install or intensive HOME I/O is selected.

EP07's existing opening status now projects only its verified freeze/check
completion booleans and freeze timestamp. EP06 consumes that score-free
completed status plus closed trial-role metadata; it never reads the mixed
analysis-freeze document, scientific report, fitted state or predictions.
All consumed source is frozen from submission until the parent ends.

## 2026-10-11 00:13 Pacific — alternative_explanations — bounded numerical continuation selected

Adopt the [explicit numerical exception](paper_plan.md#october-11-bounded-numerical-continuation)
under existing scientist-delegated repair and increased-compute authority,
consistent with the episode's two-attempt limit. The original local32-fit/
no-repeat bound changes to33 cumulative attempts for32 unique member tasks.
Reuse the one passed stable/M1/history cell and its two completed members;
perform30 remaining fits, including one retry of the failed starting09 task.
Only starting09's today-only elastic-net iteration ceiling changes to100000;
all scientific settings, convergence tolerance and recovery gates stay fixed.
Any new failure stops, without further retry or finalist replacement.

The prior active13.7720918655 seconds are charged as14; continuation is capped
at586 seconds, preserving the600-second cumulative limit. The original failed
report and all ten passed empirical controls remain intact. Implement a fresh
qualification_repair001 report, its exact eligibility binding and a bounded
batch that proceeds to the already selected primary evaluation only if every
required case passes. No allocation or additional budget reservation occurs
until source review and concrete batch selection. No new user decision is
pending for this engineering repair; EP06 neural access still waits for the
actual completed EP07 freeze and primary opening.

## 2026-10-11 00:12 Pacific — alternative_explanations — all ten controls complete; numerical qualification incomplete

Job47346999 is scheduler-confirmed FAILED1:0 after767 seconds with1 allocated
CPU, TotalCPU11:26.788 and batch MaxRSS1370420K. Charge **0.2130555556 allocated
CPU-hours once to EP07** and release its1-hour reservation. No related job is
currently live. Every empirical control has a complete paired feedback record
and passed its intervention-integrity criteria. Together with29 search pairs,
this gives **39 completed evaluations per stream**, the original maximum78
joint evaluations. No search or control requires repetition.

The fixed identity control changed neither stream. Removing earlier-day
information reduced history development R2 by0.0806372 in M1 and0.0311817 in
PMd. Earlier-day trial mismatch reduced it by0.00148424 and0.00445054; calibration
mismatch reduced it by0.00448374 and0.00845174. These are developmental
sensitivity measurements, not final support or proof of trial coupling.
All ten direct records are retained in
`runs/alternative_explanations/attempt001/control_01_feedback.json` through
`control_10_feedback.json`.

The four-case qualification stopped after13.7720918655 seconds and3 attempted
member fits, of which2 completed. The stable/M1/history ensemble passed its
fixed oracle criteria (NRMSE0.0011962451; slope7.9940242 versus8). The next
member, today-only starting09, exhausted1000 elastic-net iterations with
fresh-residual KKT error0.0026085344 above unchanged tolerance1e-5. This is
`RegressionNotConverged`, not a measured recovery failure. Evidence:
[original qualification report](runs/alternative_explanations/attempt001/known_answer_report.json).
Preserve that report and completed cell. Finalists remain history adaptive07
and today adaptive03; there is no runner-up substitution. Final held-out data
remain closed, so EP06's neural-access prerequisite is not yet met.

The existing episode policy permits two numerical attempts, and the scientist
has authorized routine repairs and increased compute. A narrowly scoped
continuation is being prepared: retain the successful cell, repair only the
elastic-net iteration allowance, and finish the remaining fixed cells. The
local32-fit/no-repeat bound requires an explicit recorded exception for the
one already failed numerical attempt; do not label33 actual attempts as32.
No continuation is submitted or active at this entry. Keep the original
four cases, scientific tolerances, seeds, models,600-second cumulative recovery
limit,800CPU-hour budget and96-hour search clock while specifying that repair.

## 2026-10-10 23:56 Pacific — alternative_explanations — running; repaired and final-transport tests pass

Job47346999 is RUNNING on sh03-06n11 with actual1 CPU/7GiB. Reduce its
reservation to1 allocated CPU-hour. All4 corrected primary-control tests and
all4 new final-fit/scorer tests passed, as did changed ASTRA validation.
Evidence: [repair checks](runs/alternative_explanations/attempt001/repaired_checks_47346999.txt)
and [final transport checks](runs/primary_evaluation/attempt001/transport_checks_47346999.txt).
Reuse the prior29 shared,8 unaffected control/recovery and6 feature/publication
checks; no additional qualification campaign is selected. Ten empirical
controls have started. Their outcomes and the four required recoveries are
not yet established; final held-out outcomes remain closed at this observation.

## 2026-10-10 23:55 Pacific — alternative_explanations / primary_evaluation — continuation submitted

Submitted **Slurm job47346999** with
[complete_controls.sbatch](tools/complete_controls.sbatch), requesting1 CPU,
7GiB,zero GPU and one hour. Independent source review passed for the corrected
test fixture, final fitted-state/scorer cores, their four new tests, primary
CLI and batch compatibility. The tests remain unrun at submission. Preserve
the at-most2 allocated CPU-hour reservation until actual allocation is known.
The previous job is scheduler-confirmed terminal; this submission reuses its
29 complete comparisons and starts only the unstarted controls/recoveries,
then the primary evaluation if every required gate passes. All consumed source
is frozen while the allocation is live. The original clock and all prior costs
remain intact. This is a standalone Codex/Slurm execution, not a BR campaign
or canonical acceptance action.

## 2026-10-10 23:54 Pacific — alternative_explanations / primary_evaluation — remaining work selected

Select [complete_controls.sbatch](tools/complete_controls.sbatch):1 CPU,7GiB,
zero GPU,at most1 hour on russpold; allow1--2 actual CPUs solely for memory
rounding and retain one numerical thread. Reserve at most2 allocated CPU-hours
until actual allocation is observed, then reduce accordingly. Expected useful
work is10--30minutes for ten paired empirical controls, four already-selected
bounded recoveries and, only if eligible, full fitted-state persistence and the
single primary evaluation. Estimate below2GiB additional fitted/prediction
storage. No padding, duplicate adaptive fits, new search or clock reset.

All adaptive work is complete. Job47345111 ended FAILED1:0 because the control
test gate failed, after successfully closing all seven remaining real-data
pairs. Scheduler elapsed248seconds,1 allocated CPU, TotalCPU3:41.297 and batch
MaxRSS1332460K. Charge **0.0688888889 allocated CPU-hours once to EP07** and
release its1-hour reservation. The final controller has15 complete starting
pairs plus14 valid adaptive pairs,29 completed evaluations per stream, zero
partial contacts, and stop_reason `maximum_attempted_pairs`. Starting08 remains
excluded/unscored after its two earlier numerical attempts. The ten mandatory
controls will bring the completed total to39 per stream if all are available.

| Last adaptive pair | History global development R2 | Today global development R2 |
| --- | ---: | ---: |
|07|0.0745198291|0.0583845427|
|08|0.0560242370|0.0520504114|
|09|0.0598694317|0.0401429784|
|10|0.0718427363|0.0278867763|
|11|-0.0493825625|-0.2077707588|
|12|0.0688690783|0.0061408720|
|13|0.0697242964|0.0157253071|

The independently selected incumbents are **history adaptive07**, global
development R2=0.0745198291, and **today adaptive03**, R2=0.0595689167. They are
fixed before any control/recovery outcome. These are developmental predictions,
not final support or a coupling finding. The controller's source evidence is
[adaptive002/progress.json](runs/matched_search/adaptive002/progress.json).

After scheduler-confirmed termination, corrected the one test fixture by
editing ordinary role metadata before constructing its poison-array wrapper.
Independent review confirms the array guard and intended role refusal remain.
Only its4-test module is rerun; reuse the other passed shared/control/recovery/
feature/publication checks. No production numerical source changed.

The selected allocation first finishes the unstarted controls and the four
fixed recoveries. New final-fit/scorer transports each have two focused tests,
run on fabricated data within this substantive job. Complete source review
before submission; no actual held access is authorized by merely writing code.
If any necessary test/control/recovery fails or remains unavailable, stop
without opening held data and without substituting a runner-up.

If all prerequisites pass, the selected primary runner verifies the existing
passed data-separation demonstration and complete search/control/recovery
records, then freezes unchanged model/inference choices, fits only original
training/calibration packs, saves both complete fitted panels, and predicts
from held features in a separate process. A separate scorer validates the full
paired prediction artifact and completed predecessor record before durably
claiming the single outcome opening. Failures after that claim stay opened and
incomplete. Undefined inference is reported distinctly from a clean failure.
The600second fixed-recovery bound and original96h/800CPU-hour limits persist.

The conditional residual/bridge/trial-identity branch is not performed in this
primary evaluation. Its absence licenses neither a coupling nor mean-only
claim; later choices on exposed outcomes are exploratory. This follows the
existing primary/conditional contract, independently reviewed. EP06 remains
closed pending a completed full freeze and primary opening, and supplies no
input to this job. No outstanding scientist decision blocks this selected work.

## 2026-10-10 23:48 Pacific — matched_search — running; one control-test fixture defect isolated

Job47345111 is RUNNING on sh04-04n22 with actual1 CPU/7GiB; reduce its
reservation to1 allocated CPU-hour. All29 affected shared checks, all6 final
feature/publication checks and ASTRA validation passed. The separately scoped
EP06 companion passed all3 bandwise checks and its ASTRA validation.

Eleven of12 control/recovery tests passed. The remaining failure is confined
to `test_primary_control.py`: `dataclasses.replace` was applied to a poison
array-access fixture, so Python read guarded fields while constructing the
test argument, before the production function was called. The minimal repair
is to change ordinary metadata first and then wrap the poison fixture. No
production calculation or scientific criterion is changed. Keep live consumed
source frozen until the allocation ends. The batch may finish its unaffected
real adaptive comparisons, but its control stage is correctly gated off.
Preserve completed feedback and resume only the unstarted controls after the
focused fixture repair and compute check; no repeated model search.

## 2026-10-10 23:46 Pacific — matched_search / alternative_explanations — completion batch submitted

Submitted **Slurm job47345111** using the selected
[complete_development.sbatch](tools/complete_development.sbatch), after independent
source review of the control runner and four fixed recovery constructions.
The immediately preceding own-user queue was empty; no duplicate or unrelated
job was submitted or changed. Requested resources are1 CPU,7 GiB,zero GPU and
one hour. Retain the at-most2 allocated CPU-hour reservation until actual
allocation is observed. This is a standalone Slurm launch by Codex, not a
canonical BR harness launch.

New affected checks remain unrun at submission. Both empirical streams and
all ten controls retain paired completion before development target contact.
Control and known-answer progress updates now preserve their previous complete
snapshot when a later write is interrupted. The fixed known-answer bound is
600seconds, at most32 member fits, with no repeats or seed search. Held-out
features/counts remain closed. The job's permitted operational records are in
[adaptive002](runs/matched_search/adaptive002/) and
[control attempt001](runs/alternative_explanations/attempt001/).

## 2026-10-10 23:40 Pacific — matched_search / alternative_explanations / known_answer_cases — bounded completion batch selected

The two scientist-approved amendments are operative. Select one combined
development/control allocation using [complete_development.sbatch](tools/complete_development.sbatch):
**1 CPU,7 GiB,zero GPU,at most1 hour**, on the previously discovered russpold
partition. Permit1--2 actual CPUs only for scheduler memory rounding; numerical
threads remain1. Reserve at most2 allocated CPU-hours until actual allocation
is observed, then reduce accordingly. Charge the parent once to EP07, including
its separately scoped EP06 array-only companion. Preserve all earlier charges
and the original96-hour search clock. Expected useful runtime is15--35minutes
for adaptive fitting plus ten finalist controls; this estimate is uncertain.
No runtime padding or checking-only job is selected. New prediction storage
is estimated below2GiB, within the unchanged500GB limit.

The earlier plan to reassess after adaptive07--09 is superseded prospectively
before any of those outcomes: declare the remaining four useful comparisons
as well, allowing their serial execution and required finalist work to share
one allocation. Independent source/scientific review found no duplicate
paired recipes. The original patience/cap still governs: starting at patience3,
four valid non-improving pairs would stop after adaptive10 and leave11--13
unrun. Batch duration never overrides that rule, and a technically incomplete
contact prevents further scoring or selection.

| Pair | Additional prospectively selected recipe |
| --- | --- |
|10|Moments/quartiles, robust scaling, rank4, explicit task predictors, ridge0.0003; history orthogonal alignment and calibration-similarity weights with source total1/calibration1; today no alignment.|
|11|Same moments/rank4/explicit representation, ReLU32 network penalty0.003,300 epochs, learning rate0.01, seed700713; history orthogonal alignment and equal source total1/calibration1; today no alignment.|
|12|Histogram4/rank8/explicit ridge0.0003; history orthogonal alignment/equal source total1/calibration1; today no alignment.|
|13|Moments/quartiles/rank8/explicit ridge0.003; history orthogonal alignment/equal source total1/calibration1; today no alignment.|

All are predictive comparisons using one global recipe per stream; simultaneous
mass/alignment/penalty changes cannot isolate a component's causal contribution.
Indices07--13 use fresh `runs/matched_search/adaptive002`, replaying all earlier
feedback and preserving every opportunity/contact/incumbent. Neither the
excluded starting08 nor an already evaluated comparison is retried.

Once the adopted starting and adaptive stages are complete, select the two
independent global incumbents once, then run all ten fixed paired controls in
`runs/alternative_explanations/attempt001`. The
[concrete conventions](paper_plan.md#concrete-ten-control-conventions-selected-before-execution)
now specify nominal-label identity, fixed whole-trial mismatches, complete
history/source removal refits, exact calibration template, equal source mass
and the global optional-component priority. No score-loss veto is invented.
The new source-control, mismatch and composition code is independently reviewed;
its focused tests remain unrun. The selected finalist never changes in response
to a control or recovery failure.

Also select exactly the
[four fixed identified-signal recoveries](paper_plan.md#fixed-four-case-construction-and-recovery-criteria)
with actual finalist recipes, centered conditional-mean NRMSE<=0.20 and the
declared native-count slope tolerance. They retain meaningful stable, pooling
drift, null-coupling and physical latent-amplitude oracles. Each construction
is fixed before generation; total bound600seconds and at most32 member fits.
No synthetic cohorts, seed search, easier substitute model or tolerance
adjustment is allowed. These are method qualification, not biological results.
Their implementation and independent review must finish before submission.

Run only affected shared implementation checks before new empirical fits.
Control/recovery checks gate their own stage. The six dormant held-feature/
paired-publication tests use fabricated arrays and files, not held-out data;
ordinary candidate interfaces remain closed. Changed ASTRA validation and
EP06's three new bandwise-array checks are bundled. All newly affected checks
are currently unrun. This selected job does not open empirical held-out
features or spike outcomes even if every control/recovery passes. EP06 neural
fitting still awaits EP07's full freeze and single primary opening.

## 2026-10-10 23:27 Pacific — matched_search — adaptive batch complete and both amendments adopted

The scientist explicitly approved both pending EP07 decisions: retain the15
complete starting pairs plus at least3 valid adaptive pairs, and adopt the
prospective control interpretation. GOAL, SEARCH_POLICY and the living paper
plan now implement those decisions. Preserve starting08 as an excluded,
unscored two-attempt failure; no third attempt, imputation or replacement.
The maximum remains14 attempted adaptive pairs, followed by all10 paired
finalist controls, with28--39 completed evaluations per stream and unchanged
scientific thresholds. No unanswered decision remains for these amendments.

Job **47340578 COMPLETED, exit0:0**, ending23:19:35 Pacific after394seconds
with one allocated CPU. Charge **0.1094444444 allocated CPU-hours once to EP07**
and release the1 CPU-hour reservation. TotalCPU06:16.268; batchMaxRSS1413116K.
Retain all prior charges and the original96-hour clock. All seven selected
adaptive pairs completed, yielding22 actual completed evaluations per stream
and23 attempted opportunities (16 starts plus7 adaptive). The existing
[progress file](runs/matched_search/adaptive001/progress.json) was written
before adoption and is preserved unchanged; its starting08 deficit no longer
prevents the adopted starting-stage completion. Its reported contacts and
scores remain valid. The amended accounting helper was independently reviewed
without findings; its three changed tests are unrun after this amendment.

| Adaptive pair | History global development R2 | Today-only global development R2 |
| --- | ---: | ---: |
|00|0.0625318751|0.0297378020|
|01|0.0654637122|-0.0175771960|
|02|0.0629638970|0.0395417845|
|03|0.0721450151|0.0595689167|
|04|0.0483539983|0.0374727286|
|05|0.0338539662|0.0109754887|
|06|0.0541833992|-0.0250519869|

Both incumbents are adaptive03: fixed half-weight native-prediction ensembles,
history starting13+00 and today starting09+00. Their respective M1 development
R2 are0.1238168231 and0.1061002145; PMd0.0204732070 and0.0130376189. History's
development advantage is therefore smaller than its advantage against weaker
today-only starts. These are selection feedback, not final support, animal/day
consistency or mechanism evidence. Patience is3, below the required7 consecutive
valid pairs without qualifying improvement; the14-attempt cap is also unreached.

All nine adaptive dependency checks, four dormant inference arithmetic checks
and then-current ASTRA validation passed in this job. Changes to ASTRA after
completion remain unvalidated. No live allocation remains. The actual batch
was shorter than its10--20minute estimate; aggregate upcoming short work with
substantive fitting, without padding runtime or submitting checking-only jobs.

Select the next three paired adaptive comparisons prospectively: adaptive07
uses flat half-weight ensembles history(starting13,adaptive02) and
today(starting09,adaptive02); adaptive08 uses moments/quartiles, robust scaling,
population rank4, explicit task predictors, no alignment and reduced-rank ridge
rank2/penalty0.001; adaptive09 uses the same representation with ridge0.005 in
both streams and history hierarchical source weights (strength120, floor0.1,
power1, total source mass1 and calibration mass1). For08 history uses mass1 per
source and calibration mass1. These are new global comparisons, not retries
of the excluded starting08. Reassess permitted feedback after these three;
do not preselect all remaining opportunities or treat a batch end as a stop.

Required controls are being implemented with strict control-only source
omission, complete affected refitting, and fixed within-direction whole-trial
mismatches. All ten remain mandatory; no outcome-dependent degradation veto
is added. Four bounded known-answer recoveries still require concrete
prospective generators/oracles/tolerances before execution. No final model is
selected, no final feature/target access is opened, and EP06 empirical fitting
still follows EP07's full freeze and single primary opening.

## 2026-10-10 23:10 Pacific — matched_search — repair closed and new adaptive development selected

Job **47338419 FAILED, exit1:0**, ending23:07:07 Pacific after713seconds
with two allocated CPUs. Charge **0.3961111111 allocated CPU-hours** once to
EP07, including the EP06 companion, and release the four-hour reservation.
TotalCPU was18:13.758; batch MaxRSS1018116K. Preserve all earlier charges,
including0.2844444444 for47336768 and0.0633333333 for47334643, and the original
96-hour clock started at2026-10-11T05:41:59.230511Z.

The [repair feedback](runs/matched_search/attempt002/development_feedback.json)
adds one complete pair: starting10 history global developmentR2=0.0161037001,
today=0.0219791488. There are now **15 complete paired starting evaluations**.
Starting08 exhausted its second allowed attempt in both streams at
Chewie-L20161014/M1 after100000 sweeps: fresh KKT errors0.0033807105118240882
(history) and0.0074493560945301067 (today), versus the unchanged1e-5 tolerance.
Both partial artifacts remain unscored; no third attempt, score imputation or
renamed replacement is selected. Source review found no concrete objective,
coordinate-update or design defect. Agreement with Gram KKT does not establish
a floating-point floor or show that more iterations would succeed. This is
numerical incompleteness, not evidence against transfer.

All two affected elastic-net and four source-weight checks passed. The EP06
companion's14 affected nomination/stability/H checks and both ASTRA programs
also passed. These are implementation evidence, not biological findings.

The policy permits further genuinely new paired development; it does not
explicitly require every starting configuration to converge before a
successor. Remove that added sequencing restriction from the dormant
adaptive accounting helper. Preserve all16 attempted starting identities,
the missing pair, actual contacts, two independent global incumbents and
the existing patience/attempt limits. Partial or unavailable contacted panels
remain explicit technical incompleteness. The helper never establishes full
search completion or held-out authority. Its three revised tests are unrun.

Select seven new paired opportunities in
`runs/matched_search/adaptive001`, as declared in
[adaptive_recipes.py](tools/adaptive_recipes.py). Selection uses only allowed
complete development feedback; starting10 does not replace either incumbent.
The original15 comparisons are not rescored or refitted. All configurations
are global across the ten settings; each stream fits its own permitted data.

| Pair | Comparison selected from the development evidence |
| --- | --- |
| adaptive00 | Histogram4/rank8/explicit ridge0.01, orthogonal history alignment; examines the successful histogram/network representation with a linear map. |
| adaptive01 | Same representation,16-unit ReLU network with penalty0.03; tests reduced capacity and stronger regularization than starting13. |
| adaptive02 | Moments/rank4/explicit ridge0.003, history calibration-similarity weights with total source mass1 and target mass1. |
| adaptive03 | Fixed half-weight native-prediction ensembles: history13+00 and today09+00; members were already evaluated, no R2 averaging. |
| adaptive04 | Moments/rank4/explicit ridge0.01, history mean-trajectory reliability weights with total source mass3 and target mass1. |
| adaptive05 | Histogram4/rank8/explicit32-unit ReLU, penalty0.1; a separate regularization comparison. |
| adaptive06 | Moments/rank4/explicit16-unit tanh, penalty0.01; tests tenfold stronger regularization than the small moments network in starting12. |

The new [batch](tools/adaptive_models.sbatch) requests **1 CPU,7 GiB,zero GPU,
at most one hour** on the previously discovered russpold partition. One
numerical process runs at a time, with one BLAS/Torch thread and a separate
scorer after each completed paired panel. Admission permits1–2 allocated CPUs
for memory rounding; reserve at most **2 allocated CPU-hours**, reducing to
actual allocation size when observed. Mixed native-data fitting across seven
pairs is estimated at10–20minutes; no runtime is padded. Estimated new
prediction storage is below512MiB. Retain all existing budget ceilings and
the original clock. This is a distinct development stage, not a retry of08.

Nine focused weighted-wrapper/ensemble/accounting checks precede their use.
Four dormant primary-inference arithmetic checks are bundled but do not gate
development. Changed ASTRA validation is also bundled. No checking-only job
is submitted. Stop on an incomplete scorer contact, deadlines or the existing
search stopping rule; record ordinary unscored fit failures against attempts.
Finishing this selected batch need not finish adaptive search and cannot
choose finalists or open held-out data.

Two [unadopted proposals](paper_plan.md#october-10-proposed-starting-count-amendment-unadopted-conditional)
are awaiting scientist decisions: fifteen completed starts plus at least three
valid successors, and prospective interpretation of the ten finalist controls.
These resolve full-search/final-evaluation requirements; they do not block
otherwise permitted new development. Neither proposal is silently activated.
Four bounded known-answer recoveries remain required before final evaluation,
with their precise construction and recovery criteria still to be selected
prospectively. EP06 neural fitting continues to respect EP07's freeze/opening
precedence. There is no new biological conclusion or held-out result.

Independent source review cleared the worker, seven declarations, scorer and
batch after adding a pre-contact binding between each prior feedback filename
and its candidate identity. **Slurm accepted job47340578 at23:11 Pacific**.
All source, test and ASTRA files consumed by this job are now preserved until
the job terminates. Submission alone does not establish running or passing
checks; the initial conservative reservation is2 allocated CPU-hours.

At23:13 Pacific, Slurm confirms **RUNNING** on sh03-06n11 with one allocated
CPU and7 GiB. Reduce the reservation to **1 allocated CPU-hour**. Operational
status records exit0 for all nine adaptive dependency checks and four dormant
inference arithmetic checks. ASTRA and empirical batch completion are not yet
observed. No final evaluation is opened by these checks.

## 2026-10-10 22:58 Pacific — primary_evaluation — prospective inference arithmetic selected

Before any held-out opening, select the following precise implementation of
the already required paired stratified bootstrap. Root and an independent
reviewer checked the tail direction, intersection–union construction and fixed
two-region Holm family. These conventions resolve previously unspecified
numerical details; they change no endpoint, margin, resampling unit, family,
sample size or outcome-access rule. No final model is selected or evaluated.

Use local NumPy `Generator(PCG64(7007999))` for exactly 9,999 draws. Iterate
draw, then Mihili/Chewie-L in chronological new-day order, then directions0–7.
Within each stratum order whole-trial IDs by their numeric `trial_` suffix and
draw its original count of positions with replacement using int64
`Generator.integers`. The same selected identities and multiplicities apply to
both regions and all three frozen predictions, retaining all native bins and
neurons. Recompute each draw's per-neuron-centered joint SST; do not freeze its
denominator. Day differences average within animal, then animals equally.

For each region/comparator, let delta be the observed history-minus-comparator
animal-balanced R2 difference and delta_star its bootstrap values. Use the
centered one-sided margin p-value
`(1 + count(delta_star - delta >= delta - 0.01)) / 10000`, including ties.
Require observed delta strictly greater than0.01 separately. The regional
p-value is the maximum of its two comparator p-values, implementing the
intersection–union requirement that both comparisons pass. Apply Holm across
the fixed M1/PMd family at0.05. This is approximate bootstrap inference, not an
exact randomization test or a population-level animal/day conclusion.

Keep the existing positive animal means, same-day 2/2 Mihili and at least2/3
Chewie frequencies, and five frozen-prediction omission conditions separate.
Each omission reweights remaining days within animal, preserving animal1/2;
both remaining animal means and aggregate must be positive for each comparator.
No omission refit, new interval, margin or repeated frequency test is added.
Equality fails every strict effect/sign requirement.

Original nonpositive/nonfinite joint SST makes the region unestimable. Any
undefined bootstrap contrast makes that regional inference unavailable, with
no discarded/redrawn samples, clipping or denominator epsilon. It stays
non-rejectable in the two-region Holm family via internal p=1; no valid reported
p-value is invented. Successful arithmetic still requires the independently
declared finalist, separation and scientific checks before a support claim.

Implement an array-only function in new `tools/primary_inference.py`, with no
file reader, fitter, selection or opening authority. Efficient per-trial SSE
and centered sums/squares may reproduce whole-trial resampling in batches of
at most256 draws; independent duplicated-trial arithmetic will check that
optimization. Focused tests remain unrun until substantive compute. Current
retry imports and ASTRA remain unchanged while job47338419 is live.


## 2026-10-10 22:53 Pacific — matched_search — final numerical repair selected

Select `runs/matched_search/attempt002` under existing continuation authority
for the second and final attempt of starting08 and starting10 in both streams.
Source review found the weighted elastic-net objective and cyclic updates
consistent with the declared model. Correlated, unequally scaled columns can
converge slowly; the previous exceptions did not distinguish slow convergence
from a numerical floor. The focused change verifies the same absolute KKT
tolerance using freshly formed training residuals before success or cap failure,
and preserves iterations, residual KKT and Gram KKT diagnostics on failure.
The iteration cap increases from 1,000 to 100,000 within hard job/search
deadlines; penalty, L1 ratio, scaling, rank, adjustment and scientific rules
stay fixed. There is no further automatic attempt or tolerance relaxation.

Root and independent source review found no remaining concrete defect in the
repair, affected arithmetic checks, worker or separate scorer. The worker
requires the recorded prior convergence failures and absence of outcome contact.
Only the two repaired pairs can receive new development feedback; the 14
complete first-attempt pairs and their scores are retained without repetition.
Count these four stream-model retries conservatively against the 12-retry
ceiling as well as the two-attempt-per-model limit. No adaptive successor,
finalist choice, scientific qualification or held-out opening is included.

The [bounded batch](tools/repair_starting_models.sbatch) requests **2 CPUs,
14 GiB, zero GPU and at most two hours** on the previously discovered russpold
partition. Two one-thread workers each own one failed pair. Admission allows
2–3 allocated CPUs for memory rounding; reserve at most **6 allocated CPU-hours**
and charge actual parent allocation once. Keep the original 96-hour clock and
all prior costs. Estimated new prediction output is below 256 MiB. The larger
iteration allowance supplies substantive real fitting; no runtime is padded.
The latest coordinator scheduler observation showed no pending/running job,
so this is not a duplicate.

Two affected elastic-net checks precede fitting. Four new dormant source-weight
checks run separately and cannot block this equal-weight repair. The parent
also carries the 14 reviewed EP06 scalar nomination/stability/H checks and
changed ASTRA validation for each named episode, with no EP06 empirical reader,
outcome access or sibling fitted input. No separate checking job is submitted.
Any unrelated companion failure remains explicit without gating EP07 fitting.
Operational state and the returned job ID will be recorded below.

Slurm accepted **job47338419**. The selected attempt's `status.txt`,
`regression_checks.txt`, `source_weighting_checks.txt`, two worker records and
`development_feedback.json` separate numerical verification, fitting and
development contact. Submission does not establish running or passing work.
Source consumed by this job is preserved until it terminates. The two dormant
weighted-fit/ensemble extensions under preparation are new files outside its
imports and cannot affect this attempt.

At22:55 Pacific, the recorded allocation and worker files establish **RUNNING**
on sh03-06n05 with two allocated CPUs. Reduce the reservation to **4 allocated
CPU-hours**; final accounting remains pending. Both affected elastic-net checks
passed in 0.245 seconds, all four source-weight checks passed in 0.083 seconds,
and EP07 ASTRA schema/semantic validation passed. Both workers have begun real
fits. The EP06 companion independently passed all 14 revised scalar checks and
ASTRA validation. None of these checks establishes repair convergence, a new
development score or a scientific result.


## 2026-10-10 22:50 Pacific — matched_search — fourteen starting pairs scored; two convergence failures

Job **47336768** ended **FAILED, exit1:0**, at22:45:52 Pacific after256seconds
with four allocated CPUs. Charge **0.2844444444 allocated CPU-hours** once to
EP07 and release the eight-hour reservation; retain the earlier preparation and
qualification charges. Batch MaxRSS was3079088K. This is an incomplete model
batch, not a scheduler fault. All36 focused implementation tests passed in4.763s,
and ASTRA schema/semantic validation passed before fitting.

The separate [development feedback](runs/matched_search/attempt001/development_feedback.json)
records **14 complete paired contacts**, or28 stream evaluations, over all five
new days and both regions. Starting08 and10 failed the elastic-net1000-iteration
limit at the specified absolute KKT tolerance1e-5 in both streams. Their partial
prediction files remain preserved and unscored. Four failed fits use one attempt
each; successful scored candidates will not be refitted or rescored for repair.
The numerical exception lacked final-KKT diagnostics, so source diagnosis is
selected before the second and final permitted attempt of each failed model.
No scientific threshold, penalty, L1 ratio, feature scaling, data role or model
opportunity changes are implied by a solver repair.

Among the14 completed comparisons, the largest global development selection
score is0.0657872387 for history starting13; the largest today-only score is
0.0343199661 for starting09. These are different provisional leaders from an
unfinished search, averaged over the two regions for selection only. They are
not held-out evidence, a scientific margin test, final model selection or proof
of trial-specific coupling. All held-out data remain closed. The sixteen-model
starting stage, adaptive successors, ten finalist checks and final qualification
remain incomplete.

The96-hour clock began at2026-10-11T05:41:59.230511+00:00 in
`runs/matched_search/search_clock.json`; repairs retain that clock and all costs.
The first aggregate batch finished sooner than forecast. Subsequent short checks
will be combined with substantive real fitting rather than submitted alone.


## 2026-10-10 22:39 Pacific — matched_search — first sixteen real-data pairs selected

Under the existing resume authority and adopted eleven-session decision, select
`runs/matched_search/attempt001` for the sixteen prescribed starting pairs.
The [fixed recipes](tools/starting_recipes.py) provide four mapping families
(ridge, elastic net, reduced-rank ridge, shallow network), ranks 4/8/12/16,
moment/quantile or histogram features, the three primary direction/time modes,
and explicit history alignment choices. Each stream receives sixteen distinct
configurations with the same feature/rank/mapping opportunities. Earlier days
have equal predictive mass, each equal to target calibration mass in these
starting seeds; this does not exclude learned source weighting, set encoders,
ensembles or other registered options from adaptive successors. No model is
chosen separately by day, animal or region, and no final architecture is fixed.

Before any development score, record the precise metric convention:
`R2 = 1 - sum((y-p)^2) / sum((y-mean_trial,bin(y_neuron))^2)` over the complete
native trial/bin/neuron tensor. Constant neurons stay in SSE; zero total SST
is unavailable, with no epsilon or dropped columns. Days average within animal
(two Mihili, three Chewie), then animals equally. The common search objective
is the equal mean of the two regional development R2 values; M1/PMd final
inference remains separate. This is also equivalent to choosing by aggregate
improvement over the fixed calibration comparator on the same complete panel.
Independent scientific review found these conventions consistent with the
current contract. The scorer returns regional aggregates, not trial/neuron
errors or raw targets to fitting code.

The complete fitting, transport, design, regression, publication and scorer
interfaces have received root and independent source review. The drop-first
cell design now rejects a missing intercept; interrupted scorer writes preserve
the last complete contact record. Thirty-six focused implementation tests remain
unrun and are bundled before real fitting in the substantive parent. They are
not a simulated scientific study. The four required known-answer cases qualify
the selected finalists before held-out opening, as GOAL specifies; ASTRA now
expresses that ordering rather than adding them as a starting-search gate.
They remain required, with no waiver or runner-up substitution after scientific
failure. Their finalist-specific expectations/tolerances remain to be fixed
before those checks; arbitrary zero-ridge recovery does not qualify every model.

The prepared parent requests **4 CPUs, 28 GiB, zero GPU, at most two hours** on
russpold, discovered with `sh_part`; quota inspection shows available Scratch
and Oak capacity. Four numerical workers use one thread each. Admission accepts
4–6 allocated CPUs to accommodate memory-driven scheduler rounding, so reserve
at most **12 allocated CPU-hours**, charged once to EP07 at actual allocation
size/duration. Retain all prior preparation charges, including 0.0633333333
CPU-hours for job47334643. Estimated new prediction storage is below 2 GiB;
the existing 800 CPU-hour, 96-hour, 32-core and 500-GB ceilings are unchanged.

The parent starts the persistent 96-hour search clock only after component
checks, immediately before its first real fitting workers. No earlier matched
search clock or fit is recorded; the selected clock/attempt paths were absent
at22:39 Pacific. Each worker owns four candidate pairs and fits all ten fixed
day/region settings; source-derived fitted state cannot enter today-only.
Complete prediction files are published exclusively. A separate scorer process
contacts only the prepared development-target file, only after both streams'
complete panels validate, and records contact-start before opening targets.
Unscored failed pairs are attempted opportunities, not completed evaluations;
their costs and retry limits remain. Missing required evaluations leave search
incomplete. No automatic retry, adaptive successor, model selection, finalist
check or held-out opening is included. The scheduler showed only existing EP21
job47329203, still RUNNING, so no EP07 duplicate is pending.

Final bounded-batch/driver source review found no remaining concrete blocker.
The allocation's recorded NumCPUs must also fit the 4–6 CPU reservation before
testing. Its accepted ID and actual observations will be appended; preparation
is not a running fit or a development result.

Slurm accepted **job47336768** for this selected attempt. Its operational files
are `runs/matched_search/attempt001/status.txt`, `allocation.txt`,
`component_checks.txt`, `worker_0.json` through `worker_3.json`, and the separate
`development_feedback.json` when scoring begins. Candidate HDF5 files contain
development predictions and the calibration comparator, never held-out data.
Submission alone establishes neither passing tests nor a fitted/scored model.
All code and recipes consumed by this live attempt are now preserved until it
finishes; no second EP07 job or retry was submitted.

At **22:41:43 Pacific**, Slurm reports **RUNNING** on sh03-06n05 with four
allocated CPUs and nine seconds elapsed. Reduce the conservative reservation
from twelve to **eight allocated CPU-hours** for the observed four-CPU parent;
final charging still uses terminal accounting. Component checks and real-data
fitting have not yet been verified from the operational files.

## 2026-10-10 22:31 Pacific — matched_search / frozen_bridge — fitting foundations implemented; primary assembly selected

The native trial adapter/calibration comparator, invariant moment/quantile and
robust-histogram feature families, and frozen population bridge are implemented
in `tools/native_trials.py`, `tools/lfp_features.py` and
`tools/population_bridge.py`. Root and an independent reviewer found no remaining
concrete defects after fixing calibration/development trial overlap in the
feature transform. Twelve focused arithmetic/role tests are prepared but unrun;
they await a substantive allocation. These are implementation checks, not a
simulation study, empirical model fit, bridge qualification or scientific result.

Features retain all nine bands, trial identities and native bins. Feature-fit
quantiles weight days equally; they do not determine predictive source-day
weights. Optional robust scaling retains zero-IQR features with scale one.
The bridge fits centered, unscaled PCA before residualization, uses the exact
QR source-link support ratio, and requires every registered source link at the
largest passing rank. Regularized correlation maps use ridge-whitened anchor
covariance, a canonical rotation and target unwhitening. This is an explicit
candidate convention. All fitted state remains bound to its permitted stream.
The changed ASTRA entries have not yet received schema/semantic validation;
the earlier passing validation does not cover these additions.

Select primary fitting assembly next: native population transport at each
candidate's explicit rank, direction/time design, weighted regression families,
and role-bound prediction. The primary search chooses among ranks 4/8/12/16;
the conditional residual explanation's largest-supported-rank and bridge gates
are not additional primary selection rules. Today-only parameters are fitted
from its own calibration data, with no history-derived rank or fitted object.
One worker owns regression primitives, one owns direction/time design, a third
reviews this translation, and root owns transport and integration. No new
allocation, outcome feedback, search-clock start or empirical fit is selected
by source implementation alone. Required four-case scientific checks remain
bounded and unexpanded.

## 2026-10-10 22:24 Pacific — source_support / data_separation — eleven-session views published

Job **47334643 COMPLETED, exit0:0**, on sh03-06n11 with one CPU/7GiB.
Scheduler accounting reports228seconds and batch MaxRSS6521040K. Charge
228/3600 = **0.0633333333 allocated CPU-hours once to EP07**, including its
bundled EP21 checks; release the one-hour reservation and preserve prior
preparation costs. The allocation has ended and cannot be reused for testing.

`runs/source_support/attempt002/source_support_report.json` reports complete
publication of all **eleven retained sessions / 2873 eligible whole trials**,
with the excluded March6 absent. The unchanged source validity predicates pass
for every retained trial. Role preparation took166.774seconds; four HDF5 views
occupy711179648bytes in the episode Scratch path. They separate fitting,
development features, scorer-only development targets and sealed held-out data.
Original positions, joint M1/PMd identities and EP06 session roles are recorded
in `trial_roles.csv`. Source files and failed attempt001 remain untouched.

The data-separation report passes, with3164 fitting,1276 development-feature,
1276 development-target and1306 held-out region/trial records. These counts
include both regions; they are not independent trial or animal counts. The
candidate reader rejects closed roles. Separation remains procedural under
the same account, not OS isolation. No neural magnitude summary, fitted model,
bridge, development score or held-out scientific result was produced.

The focused roster check passed. All33 EP21 component checks also passed in
the same parent; their evidence belongs to EP21's own companion directory.
EP07 ASTRA schema/semantic validation passed with12 optional future-format
warnings. This covers the program at this execution, before the newly prepared
fitting-component entries. The matched fitting pipeline and bridge/scoring
implementation are next; the new array-component checks remain unrun. No
simulation expansion, new reservation or subsequent empirical launch follows
automatically from preparation success.

## 2026-10-10 22:19 Pacific — matched_search — native fitting inputs and invariant LFP features selected

The adopted source preparation is now live and has passed its roster check.
Select the missing model-fitting foundation: convert the permitted candidate
reader records into day-specific whole-trial/native-bin tensors, implement the
required target-calibration direction/time mean comparator, and supply the
registered moments/quantiles and robust-histogram LFP feature alternatives.
Do not pool raw neuron columns across days, match channel numbers, reduce native
time bins or let development features/targets fit transformations. The
today-only stream has no earlier-day data or fitted state. The history stream
may fit transformations using its declared earlier training and target
calibration only. These alternatives seed the broad grammar; they neither
select finalists nor remove the registered trainable set-encoder family.

One worker owns `tools/native_trials.py` and its focused tests, one owns
`tools/lfp_features.py` and its tests, and one reviews scientific and API
consistency. Root owns integration and the two live jobs. Source edits and
array-only checks are distinct from executing model fits; no new empirical
opening, development feedback, search-clock start, simulation experiment,
resource reservation or launch is selected. The active preparation batch and
its selected checks remain unchanged. New checks await substantive compute.

The same implementation pass also selects `tools/population_bridge.py` under
the existing `frozen_bridge` analysis: centered, unscaled PCA of unresidualized
fitting counts, equal weight per native observation row, direction/time anchors
formed before residualization, and the exact registered rank/conditioning/QR
source-link gates. Mapping remains an explicit none/orthogonal/regularized
correlation candidate option. No empirical bridge or selected alignment is
claimed. A history bridge and its source-dependent chosen rank remain
history-only fitted state. The separate target-calibration basis accepts an
independently specified today-only candidate rank. Final capacity matching
and predictor assembly must resolve their interface without passing source-
derived state to today-only. Root owns this module, an independent worker owns
its focused tests, and the reviewer checks the contract translation. No tests
or new compute were executed by preparing these functions.

## 2026-10-10 22:15 Pacific — source_support — attempt002 selected for execution

Proceed under the adopted eleven-session amendment and the scientist's existing
resume instruction. The prepared single parent requests 1 CPU, 7 GiB, zero GPU
and at most one hour on the discovered russpold partition. Reserve at most one
allocated CPU-hour once to EP07; retain all previous source-preparation charges.
This is source preparation, not a matched-search start or a reset of any clock.
The log contains no recorded candidate fit or started 96-hour search window.

The role export receives 1800 seconds after its one focused roster check. It
preserves attempt001 and uses fresh attempt002 output and view paths, verified
absent before selection. The immutable v5 inputs are read only by the steward;
it emits eligibility/role metadata and separate role files without scores.
Estimated new role storage is below a few GiB, inside the unchanged 500-GB
temporary ceiling. Source qualification is reused; no repeated source survey,
source correction or expanded simulation program is selected.

Thirty-three newly prepared EP21 component checks are aggregated in this same
allocation to avoid a separate short checking job. They use fabricated fixtures
only, with output in EP21 and no sibling scientific inputs. Each suite is
bounded, thread counts stay at one, and total explicit stage limits plus grace
are 2475 seconds. EP07 ASTRA validation is included. Failures remain recorded
without automatic retry, extension or another model launch. At22:15Pacific the
scheduler showed only existing EP21 measurement47329203, still running; no
EP07 job or duplicate pending attempt was present. Accepted ID follows below.

Submission accepted: **47334643**. Root and independent review found no concrete
contract, roster, reader or batch defect before submission. Its Slurm logs are
`runs/source_support/attempt002/slurm-47334643.out` and `.err`; operational
progress is `status.txt` in that directory. Submission is not proof of running,
passing checks or a published role pack. No model fitting or scoring is launched.

At22:16Pacific Slurm confirms **PENDING**, with no preparation/check outputs yet.
Existing EP21 measurement47329203 remains RUNNING with1:12:35 elapsed. Preserve
both accepted jobs without duplicate submission.

At22:17Pacific Slurm confirms **RUNNING** on sh03-06n11, one CPU, elapsed41s.
The batch passed its actual-host/allocation guard and its focused roster test;
the first retained Mihili session is prepared. Full eleven-session publication,
data-separation evidence and companion checks remain pending. This is real-data
role preparation, with no fitted bridge, model or scientific score.

## 2026-10-10 Pacific — source_support — eleven-session amendment adopted

The scientist approved the reviewed joint EP06/EP07 eleven-session plan.
Applied the [adopted rules](paper_plan.md#october-10-eleven-session-amendment-adopted)
to GOAL, DATASETS, SEARCH_POLICY, the paper plan and ASTRA. Mihili March 6
is excluded in both regions solely for the already established native-count
failure. The retained five new days use weights 1/4 (Mihili) and 1/6
(Chewie-L), including every bootstrap draw. Both Mihili days and at least
two Chewie-L days must be positive against both comparators on the same day;
each animal mean must be positive. Five frozen-prediction omission checks
require positive remaining animal and aggregate means for each comparator,
without refitting, new uncertainty tests or reapplying full-panel frequency
rules. The 9999-draw trial bootstrap, margins, bridge and source-link gates,
search and resource ceilings remain unchanged; benefit forecasting has five
outer held-day folds. No consumed resources or started clocks are reset.

The source steward keeps the historical twelve-session `SESSIONS` catalog and
uses a separate eleven-session `PREPARATION_SESSIONS` roster for export. The
candidate reader rejects excluded March 6 before opening a file. Original
within-animal positions are retained. The existing `trial_roles.csv` will
record EP07 roles, original source indices, joint M1/PMd identity and fixed
EP06 reserved/development session roles. The report records
`preparation_only_no_scientific_scores`. This documents shared-corpus exposure;
it does not establish compatible cross-episode trial splits or independent
confirmation. EP06 scoring still follows EP07's full freeze and single primary
opening; sibling outcomes and fitted state remain unavailable to EP07 fitting.

Next selected analysis is fresh `source_support/attempt002`, preserving every
failed attempt001 artifact and partial file. The coordinator-owned
`outputs/tools/prepare_roles.sbatch` provides one 1-CPU, 7-GiB, at-most-1-hour
allocation with a 1800-second role-export limit and bounded required checks.
It uses the existing v5 source and unchanged trial split seed, exporting to
`$SCRATCH/br_autoresearch/episode07_lfp_session_transfer/source_support_attempt002`
and compact metadata under `outputs/runs/source_support/attempt002`. The one
focused `outputs/tools/test_eleven_session_roles.py` method checks exact
exclusion/rosters, unchanged historical positions and refusal before file I/O;
it is written but unrun. Submission and results belong in a subsequent record.

This amendment was applied through bounded login-node documentation/source
editing only: no source payload, neural outcome, project import, test, role
export, job or scheduler query was run by the implementation worker. The
preparation code is executable; matched model search, bridge qualification,
primary/explanatory inference and benefit forecasting still need engineering.
Historical qualification remains reusable; no additional simulation program
or corrected March 6 export is needed to begin this retained-panel preparation.

## 2026-10-10 Pacific — source_support — eleven-session amendment prepared

Completed the [joint decision proposal](paper_plan.md#october-10-eleven-session-amendment-for-decision-unadopted)
before any model scores. The five-new-day panel retains all six historical
days, weights animals equally, requires both Mihili days and at least two
Chewie days positive against both comparators, and uses five point-positive
omission checks without refitting or new significance tests. The existing
within-day/direction trial bootstrap, margins, search and support gates stay
fixed. The EP06 companion specifies its three/four development-session
consequences and preserves exact reserved dates and access precedence.

The recommendation is to adopt the joint exclusion now; waiting for corrected
counts remains the alternative. The October 6 request was not sent and its
proposed response window never started. Two read-only episode reviewers
informed the proposal. BR directive `ext_review_dir_12d86d49bdd4` supplied
generic review criteria only; it did not inspect artifacts, deliver a
tailored verdict or approve execution. Root applied relevant criteria locally.
No source payload, outcome, test, role export or job occurred in this review.
The active twelve-session contracts and implementation remain unchanged
until the scientist adopts the joint amendment.

## 2026-10-06 Pacific — bounded public correction search recorded

Added the verified October 6 public findings and exact URLs to the
[existing paper plan](paper_plan.md#public-source-check--october-6-2026-pacific).
The official Dryad record shows the listed 2022 release without a visible
March 6 correction notice. Author code applies downstream square-root and
smoothing operations; it does not authenticate the released array's transform
or a lossless inverse. The zero-open-issues listing and failed README/history
fetches do not prove absence of a correction. No authenticated replacement was
found, so the existing request for both original 30-ms count arrays and their
unaltered trial/guide/event/LFP correspondence remains useful.

The edit reuses the completed lookup without another search. No MAT payload,
protected outcome, test, import, scheduler action, provider contact or job was
performed. Historical diagnosis and attempts remain preserved. Active roles,
thresholds, budgets and access are unchanged; the joint exclusion proposal is
not adopted and EP06's dependency is not bypassed by this documentation.

## 2026-10-06 Pacific — correction-first and joint fallback proposal

Updated DATASETS and the existing paper plan to distinguish the completed
twelve-session eligibility check from unresolved March 6 count representation.
The original source request remains unsent. Proposed a 14–21-day window from
actual sending, followed by explicit joint EP06/EP07 amendment if necessary.
The draft addresses two remaining Mihili new days, EP06's fixed reserved
dates, mandatory sample-size review and a shared exposure ledger. No exclusion,
role change, threshold, budget, access opening or job is activated. No payload,
test, scheduler or external contact occurred.

## 2026-10-03 — source_support — completed inventory and source handoff

The remaining-seven qualification completed in allocation **46424502, step 8**,
exit 0, using the same fixed-session eligibility predicates and no role export.
Evidence: `runs/source_support/attempt001/remaining_source_validity.json`.
Together with the preserved attempt-001 report and fixed-file diagnosis, this
qualifies all twelve dates: eleven pass; only **Mihili 2014-03-06** fails the
native-count requirement, in both M1 and PMd.

| Animal | Fixed date | Complete trials | Minimum complete trials per direction | Source eligibility |
| --- | --- | ---: | ---: | --- |
| Mihili | 2014-02-17 | 208 | 22 | Pass |
| Mihili | 2014-02-18 | 225 | 25 | Pass |
| Mihili | 2014-03-03 | 208 | 21 | Pass |
| Mihili | 2014-03-04 | 203 | 22 | Pass |
| Mihili | 2014-03-06 | 0 | 0 | All 217 trials fail native-count validity |
| Mihili | 2014-03-07 | 216 | 24 | Pass |
| Chewie-L | 2016-09-29 | 208 | 23 | Pass |
| Chewie-L | 2016-10-05 | 202 | 23 | Pass |
| Chewie-L | 2016-10-06 | 209 | 23 | Pass |
| Chewie-L | 2016-10-07 | 168 | 17 | Pass |
| Chewie-L | 2016-10-14 | 740 | 85 | Pass |
| Chewie-L | 2016-10-21 | 286 | 33 | Pass |

Exact source request, prepared for the scientist to review/send: for
`Mihili_CO_20140306.mat` in Dryad DOI `10.5061/dryad.xd2547dkt`, local selected
release `dryad-xd2547dkt-v5-payload`, provide an authenticated native 30-ms
nonnegative-integer spike-count export for **both `M1_spikes` (43 columns) and
`PMd_spikes` (86 columns)** across the original 217 trial identities, with the
corresponding unit guides, event indices, native timing and LFP correspondence
preserved. Alternatively, provide a documented, lossless representation
correction specific to those released fields. The required windows are M1
150–450 ms after movement onset and PMd 450 ms before the go cue through the
go cue. Both regions are finite/nonnegative, but every required window contains
nonintegers beyond eight ULPs; a 0.03-second rate-to-count conversion does not
recover integer windows. We have not rounded, inverted an assumed transform,
removed units, changed days, or contacted a provider.

The unchanged contract cannot run on an eleven-session substitute. Preparation
is blocked pending this source remedy or an explicit scientific amendment.
The original source, failed attempt and unpublished partial role files are
preserved. No role pack was published, no bridge/model was fitted, and no
development score or held-out neural summary was computed. Source eligibility
does not establish the later training/calibration bridge-rank requirements.
EP06 scoring still waits for EP07's full freeze and one primary opening.

The coordinator also reported the official ASTRA 0.0.14 schema and semantic
validation **passed**, with only optional future-format warnings; evidence is
`outputs/astra/v0.0.14/validation_20261003.json`. Stable analysis IDs and
dependencies are preserved. The execution block is data validity, not ASTRA.

## 2026-10-03 — source_support — substantive native-count source block

The fixed-file diagnosis completed in allocation **46424502, step 6**, exit 0.
An earlier step 5 stopped during imports because SciPy was missing from that
wrapper; the coordinator loaded the available SciPy environment before retry,
and the failed import read no payload. Evidence:
`runs/source_support/attempt001/count_validity_diagnosis.json`.

On **Mihili 2014-03-06**, all 217 required windows in each of M1 and PMd are
finite and nonnegative, but contain nonintegers beyond eight floating-point
ULPs. None passes the near-integer predicate after multiplication by the
native 0.03-second bin duration. Thus neither float-equality noise nor a
simple rates-to-counts conversion explains this failure. Array metadata are
float64 with 43 M1 and 86 PMd neuron columns. No amplitudes, moments, neural
summary or score were disclosed. The native-count gate remains unchanged;
rounding, rescaling, dropping units or substituting days is not authorized.

The selected next bounded action is `tools/qualify_remaining_sources.py`,
restricted to the seven fixed sessions not reached by attempt 001: Mihili
2014-03-07 and the six Chewie dates. It applies the same identity, eight-
direction, native-window, guide and count predicates, emitting validity and
support counts only. It creates no roles or views and preserves the original
partial attempt. The coordinator will run it with 1 CPU/4 GiB/10 minutes.
This completes the concrete source-correction request without cherry-picking
sessions. A read-only provider-documentation check may establish a documented
representation correction; otherwise the required remedy is an authenticated
native-count export for every affected fixed session, or an explicit
scientific-contract amendment before preparation or scores can proceed.

The bounded provider lookup found no authenticated representation correction
in the [Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.xd2547dkt)
or its linked analysis examples. [Main.m](https://raw.githubusercontent.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/master/Main.m)
loads the trial structure; [fCCA.m](https://raw.githubusercontent.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/master/MainAnalysis/fCCA.m)
applies square-root transformation and smoothing downstream. Those examples
do not identify the affected file as already transformed and do not establish
an invertible correction for its released values. No inverse transform is
inferred. The selected local source remains the v5 payload; no alternate
version, replacement file, or new source was downloaded.

## 2026-10-03 — source_support — count-validity failure requires diagnosis

Steward step 3 exited 2 after 45.95 seconds. The first four Mihili dates passed
all declared eligibility checks and were written only to unpublished partial
role files. On the fixed new day 2014-03-06, all 217 trials failed
`not_native_nonnegative_integer_counts`; the resulting zero per-direction
support is a consequence of that predicate, not evidence of absent reach
directions. Evidence: `runs/source_support/attempt001/source_support_report.json`.
No role pack or candidate/scorer interface was published, and no model fit,
score, held-out neural summary, session replacement or unit exclusion occurred.

The next bounded diagnostic is `tools/diagnose_count_representation.py` on
**Mihili_CO_20140306.mat only**. It reports per-region counts of windows passing
finite/nonnegative/exact-integer predicates, distinguishes deviations confined
to eight floating-point ULPs, and checks the rate-versus-count dimensional
signature without printing any neural values or moments. These diagnostics
authorize no rounding, scaling or neuron removal. They distinguish a numerical
adapter issue from genuinely non-count source values before any scientific
decision. Attempt 001 and all partial files are preserved unchanged.

## 2026-10-03 — source_support — steward preparation selected

The scientist requested resuming available work toward running the questions.
The selected first EP07 action is source support and data separation for the
twelve fixed sessions, not model fitting, scoring, replacement days or a new
search design. The source is
`$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source/`.
All twelve named files and provider README are present. No sibling outputs or
fitted state are used. EP06 scoring retains its existing EP07-first order.

Prepared compute command: `tools/source_support.py inspect` reads MAT variable
metadata for the twelve fixed files and the field layout, timing/identity
metadata and channel/band guide metadata on two fixed earlier-day examples,
one per animal. It emits no neural values/statistics, prediction or score.
For MATLAB v5, deserializing a trial struct may load bundled neural fields
into steward memory; those arrays are not inspected for this structural
report and cannot enter its explicit output whitelist.

The coordinator owns compute allocation 46424502 and will run this bounded
inspection, then use its field evidence to finalize the eligibility/export
adapter. Required eligibility later inspects only finite availability and
native nonnegative-integer validity, with no magnitude, variability or
model-score criterion. No source payload or test ran in this editing shell.

The source remains readable to the same user account. The intended separate
files and restricted runner interfaces are procedural/runner-enforced
boundaries, not an OS-level blinded evaluator; that limitation will be stated
explicitly. Development targets are scorer-only and held-out features/targets
remain closed to candidate code. No trial role has yet been instantiated.

Structural-inspection repair: the first compute invocation stopped immediately
because SciPy's MAT reader reported MATLAB v7.3/HDF5. It emitted no report and
read no neural values. The inspector now uses h5py for that observed format:
root/field/reference metadata, whitelisted scalar identity/event data and
guide tables only. Neural datasets are queried for shape/dtype without loading
their elements. No scientific choice or access role changes. The coordinator
will retry the same bounded structural inspection in its allocation using the
available h5py 3.10 runtime.

The repaired inspection passed in allocation 46424502, step 1, bounded to
1 CPU/4 GiB/10 minutes. Structural report:
`runs/source_support/attempt001/structural_inspection.json`. The two earlier
day examples have native 0.03-second bins and both M1/PMd fields in the same
trial struct, 864 LFP columns with the same nine guide classes, and native
spike-array/channel-guide shapes. No neural values or statistics were emitted.

The next selected command is `tools/prepare_role_views.py` on exactly these
twelve files. It validates joint trial eligibility using the existing rules,
retains within-day neuron identities by guide intersection, and records one
Python `random.Random("ep07-target-role-split-v1")` shuffle stream in fixed
session/direction/native-trial order. Native MATLAB indices are converted by
subtracting one; M1 uses offsets 5:15 and PMd -15:0, both half-open, without
rebinning, smoothing, transforms or amplitude-based filtering. A failed fixed
day stops preparation; no replacement session or reduced primary corpus.

The coordinator launched that steward in allocation **46424502, step 3**,
bounded to **1 CPU/4 GiB/30 minutes**. The first Mihili 2014-02-17 role export
completed; overall completion is pending. This is preparation only. Root owns
the process and scheduler observation; no duplicate writer is running.

Prepared role files are separate `candidate/fit.h5`,
`candidate/development_features.h5`, `scorer/development_targets.h5`, and
`sealed/heldout.h5` under episode-specific Scratch. Fitting files contain
earlier training and later calibration only. `tools/view_access.py` restricts
candidate reads to one animal/target-day/region, excludes earlier-day state
from today-only, and rejects scorer/held-out roles. The required separation
check verifies role and whole-trial disjointness and negative access through
the actual reader; it emits metadata only. Partial role files stay unpublished
if source support fails. No candidate fit, bridge, neural summary or score is
part of this stewardship command.

Guide classes observed in both regions are (0,0), (0.5,4), (4,8), (8,12),
(12,25), (25,50), (50,100), (100,200), (200,400) Hz. The (0,0) class is
interpreted as LMP from the unique non-power class and the paper's nine-class
list, not from an explicit string in the MAT guide. The exact numeric guides
are preserved. Source fields label Chewie and M1/PMd, without a hemisphere
field; Chewie-L remains the contract's assignment for the six named 2016
sessions, not independently established by this metadata inspection.

Source-method lookup used the provider-linked
[analysis code](https://raw.githubusercontent.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/master/MainAnalysis/fCCA.m)
and [Dryad record](https://datadryad.org/dataset/doi:10.5061/dryad.xd2547dkt).
The [publication's measurement description](https://elifesciences.org/articles/73155.pdf)
lists LMP and the eight bands; broader publication access encountered a web
challenge, and no unpublished or EP06 result was accessed. Source code's
downstream smoothing/square-root preprocessing is not adopted by EP07's
native-count contract. These existing public materials are exposure context,
not fresh held-out evidence.

The coordinator subsequently authenticated the LMP mapping from the provider's
[Generate_all_figures.m](https://raw.githubusercontent.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/master/Generate_all_figures.m)
(lines 73–78): the zero-lower-frequency guide case takes a moving average of
the signal, while power bands take a moving average of its square. The
[paper's Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC9470163/) identifies
the moving-average signal as LMP. Combined with the observed unique (0,0)
guide class, this resolves its semantic identity. It supersedes the earlier
inference-only wording and does not alter any released value or feature choice.

No experiment has started. The current documents define the question, data
roles, comparisons, search budget, held-out decision, and follow-up analyses.

## 2026-09-30 — integrated paper design, not execution

- EP07 owns the history-and-population-components paper; EP06 supplies its
  separately specified recording-domain component. Exact primary roles,
  calibration, score, margins, budgets, and decision rules are unchanged.
- Read only design contracts, human-readable local notes, the existing concept
  image, and primary literature. No input payload, protected outcome, job,
  sibling result, model fit, or held-out score was accessed.
- The local ASTRA IDs `source_support`, `data_separation`, `known_answer_cases`,
  `matched_search`, `alternative_explanations`, and `primary_evaluation` are
  planned, not launched. `frozen_bridge`, `residual_prediction`,
  `trial_identity`, `signal_boundary`, `benefit_forecast`, and `new_day_test`
  are planned explanations/forward work, not primary replacements.
- Runtime: no attempts or scheduler/process IDs; no completed scientific
  artifacts. The source view and field-level correspondence remain unprepared;
  no new recording-day source is selected. Next authorized analysis, if later
  requested, is `source_support` on the named released sessions before scores.
- Scientific access order: EP06 scoring on this corpus follows EP07's full
  freeze and one primary opening, unless a scientist approves compatible joint
  roles before relevant score access. Manuscript integration is not data pooling.

## 2026-09-30 — scientist clarification: keep development search open

- Clarified broad adaptive exploration within the existing model grammar;
  starting models are seeds, not a final architecture shortlist. Primary
  success governs final explanation, not proposal of residual candidates.
- Removed the planned forecast's arbitrary two-indicator exploration cap.
  Feature/transform selection, scaling and tuning must be cross-fit by day;
  six target days still do not establish a precise transferable boundary.
- Clarified bridge freezing per candidate/fitting fold, separate from freezing
  the final comparison before applicable held-out/new-day outcomes.
- Updated `matched_search`, `frozen_bridge`, `residual_prediction` and
  `benefit_forecast` descriptions only; no analyses or reference edges added.
- Preserved operative model structural limits, paired-stream opportunity,
  56–80 evaluation budget, 800 CPU-hour ceiling, data roles and final margins.
- Local documentation amendment only: no data, score, outcome, job or Git
  mutation. No active analysis/attempt/job; the future first empirical stage
  remains `source_support` if separately requested.

## 2026-10-02 local scope review

Scope/novelty review completed from permitted design records and primary literature; see `paper_plan.md`. ASTRA `primary_evaluation`, `residual_prediction`, and `trial_identity` remain planned, not run. Repaired interpretation only: failed or imprecise residual/pairing increments do not identify mean-only transfer or geometry. No outcome access, new test, numerical change, or search restriction.

## 2026-10-02 figure review

Personally viewed the former 1619 × 971 PNG: its mean/geometry-only outcome contradicted the current interpretation and residual drawings blurred means with deviations. Regenerated with built-in imagegen and inspected final 1673 × 940 v2 after rejecting mislabelled pairing drafts; accepted diagram explicitly shows A→A/B→B versus A→B/B→A, zero references and the incremental history-minus-today-only quantity. Independent visual review accepted it. Exact prompt set saved; GOAL/plan/output-index captions preserve broad search and unresolved mechanism. Old assets retained. No data, policy, runtime or Git change.
