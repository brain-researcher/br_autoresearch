# Experiment log

## 2026-10-06 — `fixed_prediction_components`: complete saved-result comparison selected

The scientist requested further work on current episodes. Select comparative
interpretation and plotting of the completed HW, blend and calibrated-HW
decompositions. Read only their six declared completed JSON/CSV artifacts,
retain all16 sessions/all15 bins/both animals, and show the total, mean-correction
and trial-varying components together. No new source, model fit, selection,
bootstrap, future session or spike-population follow-up. Original45-comparison
stop and failed practical margins remain unchanged.

Implementation and the minimal ASTRA entry are prepared; independent source
review precedes execution. The one-thread120-second analysis and changed ASTRA
validation are aggregated into the separately scoped substantive EP14 parent.
All parent cost is charged once to EP14, with no extra EP05 charge. No new
figure/report execution exists yet.

**Completed in parent46735127, October6:** independent static review passed
after labeling0.01 correctly as the animal-balanced aggregate margin, not an
individual-animal requirement. Analysis and changed ASTRA validation exited0
on `sh04-18n31.int`. The full eight-panel figure passed visual review; PNG/PDF,
48 session rows,720 bin rows,32 paired contrasts and source execution metadata
are under `figures/fixed_prediction_components_attempt001/`.

The blend's +0.002678854 total gain includes +0.002542631 net trial-varying
correction. Calibrated HW slightly improves the total to+0.002732680 but its
trial-varying term drops to+0.002326927 while mean correction rises from
0.000136223 to0.000405753. The lower trial-varying term holds in both animal
averages; individual session effects remain fully displayed. Thus higher total
prediction does not imply more trial-specific information. Original practical
failures remain; this does not establish a motor-population mechanism.
All parent cost is recorded once in EP14; no additional EP05 charge.

Closeout: parent46735127 completed0:0 and the compute PTY closed October6
00:29:30Pacific. All1712allocatedCPU-seconds are recorded once in EP14's
`bilateral_regularization` closeout, with zero additional EP05 charge.

## 2026-10-05 — fixed_hw_correction_calibration — shared allocation closed

Shared Slurm46578298 completed normally, exit0:0,05:30:06–05:53:43UTC.
The1417 allocated seconds on four CPUs equal1.5744444444 allocated core-hours
for EP05 and EP14 together, counted once in the
[EP14 execution record](../../episode14_openbhb_roi_site_generalization/outputs/experiment_log.md#2026-10-04--discarded_direction_diagnostics--completed).
The coordinator exited the compute shell; no task-owned job remains. The
16-row component aggregation also completed in this session and is preserved
as `decomposition_aggregate.json`; no additional model was fitted. Pinned ASTRA
schema/semantic validation remains open because the validator was unavailable
on the discovered command path; no validation success is claimed.

## 2026-10-05 05:34 UTC — `fixed_hw_correction_calibration`: one successor complete; shrinkage reproduces aggregate primary gain with lower identity

Root reports all five focused fixtures passed. The adopted one-thread runner
completed in verified shared Slurm46578298 on`sh04-18n32.int`,exit0,
05:32:18.240017–05:32:21.710750UTC. Complete `calibration.json` reports
3.4707283989991993seconds process elapsed,peakRSS211932KiB,exact original
reference-score replay and passed decomposition/identity checks. Evidence and
the bounded interpretation are under
`runs/fixed_hw_correction_calibration/attempt001/`. No source-neural values,
new decoder fit or future session were accessed. Root owns final shared
allocation accounting in EP14; process wall time is not a core-hour charge.

All sixteen other-fifteen-session folds select amplitude0.75. Calibrated
primary0.002732679718402166 and identity0.004210787975507177 both fail the
unchanged0.01/0.005 practical requirements. Animal primary means are
Mihili0.001608141562026399/Chewie0.0038572178747779324; identity means are
0.0034602261097943818/0.004961349841219973. Primary is positive in6/6 and9/10
sessions; Chewie_20150630 remains negative. Every session's identity decreases
relative to both original references.

Calibrated-minus-HW primary is+0.0003914252452251812 and identity
-0.0014035959918357217. Both animal primary means and every within-animal
fixed-prediction omission remain positive, although only4/10 Chewie sessions
improve. Calibrated-minus-blend primary is+0.00005382528730217956 and identity
-0.0003486067432659905. Primary improves in4/6 Mihili and5/10 Chewie sessions;
Chewie's small mean contrast reverses on omitting20161007 or20150629.
This establishes a similar/slightly higher aggregate primary point estimate
without adding current-bin predictions, not equivalent predictions or a
robust new winner. The lower identity benefit preserves an important difference
between the policies; no physiological denoising or population mechanism follows.

The full source corpus and choice of fixed HW representation were previously
exposed. Leave-one-session calibration does not create fresh confirmation.
One successor comparison is complete, with no literal global duplicate;
original scientific count stays45 and its stop/margins remain unchanged.
There is no automatic successor search, promotion, penalty amendment or
future-session opening. Complete per-session/all-bin decompositions are saved.

The coordinator subsequently aggregated the existing sixteen decomposition
rows in compute, without another fit or selection; evidence is
`runs/fixed_hw_correction_calibration/attempt001/decomposition_aggregate.json`.
Calibrated animal-balanced mean-correction gain is0.0004057530593416252 and
net trial-varying gain0.002326926659060539, reconstructing primary0.002732680.
The original blend has mean gain0.00013622326480497934 and net trial gain
0.002542631166295005; original HW has0.000009315987409198488 and
0.002331938485767783. Thus calibration reaches a similar/slightly higher total
through a larger mean correction while giving a smaller trial-varying benefit
than the blend. This ordering holds in both animal means. Relative to HW,
the average trial-varying gain is slightly lower, so the total improvement is
from mean correction. Scalar shrinkage does not reproduce or explain away
the blend's distinct trial-varying benefit; equivalent information or a common
biological mechanism is not established. The interpretation now states this
distinction explicitly. Original margins, exposure and stopping claims remain
unchanged; shared allocation accounting stays with root.

## 2026-10-05 05:30 UTC — `fixed_hw_correction_calibration`: bounded successor adopted and implemented

Under the scientist's explicit continued-work confirmation, root adopted one
separate development-only HW correction calibration. The prospective
[specification](paper_plan.md#fixed-hw-correction-calibration) and matching
SEARCH_POLICY section freeze original HWlambda1, amplitudes0/0.25/0.5/0.75/1,
other-fifteen-session equal-animal primary selection, exact ties nearest1,
original whole-trial derangement identity, and fixed HW/half-blend comparators.
The original45-comparison stop and0.01/0.005 margins remain unchanged. This
does not adopt the independent-penalty amendment or activate a new search,
conditional spike-population explanation or future-session opening.

Implemented `tools/calibrate_fixed_hw_correction.py` and five focused fixtures
in `tools/test_fixed_hw_correction_calibration.py`. The runner reads only saved
episode-local prediction banks and reference summaries, replays original
support/scores exactly, and learns a scalar from the other fifteen sessions;
there is no source-neural read or decoder refit. It preserves all five cells,
selected amplitudes/training lists, all session/animal effects, fixed-prediction
omissions, frozen-reference contrasts and all direction/bin decomposition.
The scalar is new statistical calibration, not a descriptive-only score replay.
One complete comparison belongs to this successor; original count stays45.

New source code and the minimal ASTRA analysis/dependencies are prepared for
coordinator and independent review. Tests, ASTRA validation and scientific
execution have not been run by the implementation agent. The coordinator owns
the compute session and all scheduler observation; no allocation share is
charged here. After passing review/checks in verified compute, use the existing
qualified Python3.12.1/NumPy/SciPy/h5py environment and one numerical thread:

```bash
PYTHONPATH="$PWD/outputs/tools${PYTHONPATH:+:$PYTHONPATH}" python3.12 -B -m unittest discover -s outputs/tools -p test_fixed_hw_correction_calibration.py -v
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 timeout 210s python3.12 -B outputs/tools/calibrate_fixed_hw_correction.py --out outputs/runs/fixed_hw_correction_calibration/attempt001 --max-seconds 180
```

One CPU,2GiB and180seconds internal/210seconds external are the full adopted
calibration bound. The earlier decomposition's11.01seconds is only a resource
reference. The new result will be `calibration.json` plus selected prediction
arrays and five CSVs in the fresh attempt directory. No scalar outcome,
calibration score, comparison completion or automatic next experiment is
claimed at this preparation point.

## 2026-10-05 04:50 UTC — fixed-prediction descriptive decomposition complete

The coordinator executed the prepared script in verified shared Slurm46574496
on `sh04-18n32.int`; exit0,04:49:07.971618–04:49:18.985454UTC,
11.013831364922225seconds process elapsed,peakRSS230816KiB. All original
selected W/HW/blend scores replayed exactly, and direction/bin additive gain
plus finite-group derangement identities passed. Complete output is
`runs/descriptive_secondary/fixed_prediction_decomposition_attempt001/`;
`interpretation.md` explains `decomposition.json` and the five complete CSVs,
including720 bin rows and5760 direction/bin rows. No provider source was read,
no neural model fit or new selection occurred, and scientific count remains45.
The shared allocation completed normally with exit0:0 after177seconds on one
CPU (0.0491666667 allocated core-hours total for EP05 and EP14). Its cost is
recorded once in the [shared accounting](../../episode14_openbhb_roi_site_generalization/outputs/experiment_log.md#2026-10-04--cached_projection_diagnostics--completed),
not charged again here. The coordinator exited the compute shell normally.
Process wall time is not allocation billing.

Animal-balanced W/HW/blend mean-correction gains are respectively
0.000018262915089,0.000009315987409,0.000136223264805. Their net trial-varying
gains are0.002156587912800,0.002331938485768,0.002542631166295, reconstructing
the original primary increments. Thus about95% of the blend's small primary
gain concerns deviations between trials rather than direction/time mean
correction. Blend-minus-HW reduces covariance benefit by0.00101625 but lowers
the subtracted trial prediction energy by0.00122695, for net trial gain
0.00021069; mean correction adds0.00012691. The known total0.00033760 advantage
therefore reflects both terms, chiefly the net trial-varying improvement.

Net trial-varying blend-minus-HW gain is positive in both animal means:
Mihili+0.00026104,Chewie+0.00016035. Mean correction improves in Mihili
(+0.00025819) and slightly worsens in Chewie(-0.00000437). These are descriptive
fixed-prediction quantities. Reduced correction energy is not an estimate of
training-sample variance or proof of physiological denoising. Original
session influence, failed0.01/0.005 margins, absent future-session data and
unactivated spike-population explanation remain in force. No candidate,
promotion, threshold amendment, search restart or biological mechanism follows.

## 2026-10-05 04:48 UTC — fixed-prediction descriptive error decomposition prepared

The scientist asked to continue investigating the completed findings. The
coordinator selected a descriptive secondary analysis of the existing W, HW
and half-blend predictions only. This does not reopen the completed 45-comparison
search, change its stopping decision or 0.01/0.005 margins, adopt the separate
penalty amendment, or activate the conditional spike-population follow-up.
No new source reads, fits, candidate selection or held-session access are part
of this analysis. No new scientific score has been computed at preparation.

Prepared `tools/decompose_fixed_lfp_predictions.py` reuses the exact original
W/HW prediction loader and score replay from `run_primary_history_blend.py`,
the two complete component summaries, and the saved selected blend arrays in
`runs/adaptive_search/primary_history_blend_summary_attempt001/selected_predictions/`.
It preserves all sixteen sessions, their already selected lambda1 policies,
the blend's already selected alpha0.5, native targets/B, direction and trial
identities, and original whole-session joint-XY denominator. No arrays are
regenerated by fitting. The first original session's W, HW and blend saved
files were located through their documented paths; only file existence and code were
inspected while preparing this script.

For each original direction and all fifteen native bins, the fixed prediction
increment is decomposed into direction/time mean correction, twice the
within-direction target/prediction covariance, and the negative squared
trial-varying prediction energy. Evaluation-trial means are descriptive
arithmetic only; no evaluation-derived centered predictor is created. The
existing exact derangement identity readout has the separate finite-group
factor n/(n-1) multiplying each direction's covariance term. It is not equated
to the unrestricted-permutation covariance term. All original selected scores
must replay exactly; new decomposition identities allow only numerical
roundoff (rtol1e-11/atol1e-12), without changing any original replay tolerance
or scientific threshold. Native direction counts remain in the detailed CSV.

Session terms retain the original equal-session-within-animal then equal-animal
aggregation. The report preserves every bin and direction, including negative
values, plus fixed HW-minus-W/blend-minus-W/blend-minus-HW contrasts. This can
distinguish template correction from covariance benefit and prediction-energy
cost; it does not identify a physiological mechanism or a causal timing lag.

Prepared output is a fresh
`runs/descriptive_secondary/fixed_prediction_decomposition_attempt001/`
containing `decomposition.json`, `sessions.csv`, `animal_summaries.csv`,
`session_contrasts.csv`, `bins.csv` and `direction_bins.csv`. The existing
sixteen-session blend artifact run took about23seconds including its pilot;
that is a cost reference, not a measurement of this new analysis. Proposed
bound is one numerical thread, up to2GiB,180seconds internal/210seconds external,
inside the coordinator's verified attended Slurm allocation and existing
Python3.12.1/numerical module environment. The script includes reconstruction
checks; no new test suite, execution, job submission or result is claimed here.
The coordinator owns review, execution and subsequent interpretation.

## 2026-10-04 22:27 UTC — `adaptive_search`: final allocation share closed and accounted

Slurm46521343 completed exit0 at22:27:18UTC after1342seconds on one CPU:
0.3727777778allocated core-hours total. Split the shared EP05/EP17 preparation
cost equally,671seconds/0.1863888889core-hours each, counted once. These shares
are bookkeeping, not per-process CPU attribution; TotalCPU174.479seconds and
stepMaxRSS6265608K belong to the shared allocation step. PTY36735 is closed.

Replace the earlier full2103second reservation with the actual EP05 share.
The prior inclusive documented closed bound19.062037037core-hours plus this
share is19.248425926, conservatively rounded upward to19.249core-hours. This
inclusive bound still contains historical reproduction/shared work and is not
an exact search-only process bill. No unresolved active EP05 allocation remains.
The22:17 scientific stop and final report are unchanged; no further EP05 work
is scheduled. EP17's separate successor46523446 is charged to EP17 only.

Earlier execution observations follow.

## 2026-10-04 22:20 UTC — `adaptive_search`: final scientific report and program complete

`primary_development_report.md` is complete and root-reviewed, with both final
sourced figures, signed conditional contrasts, the small positive incumbent,
failed practical criteria, coverage/counting rationale and all development-only
claim limits. `primary_post_basis_stopping_review.md` now records the actual
22:17 stop adoption. Changed ASTRA closeout validation passed schema and
semantics in `astra/v0.0.14/validation_20261004_development_closeout.json`.
No eligible selected-policy artifact is produced; future-session and conditional
population stages remain inactive. Current46521343 is now shared sequentially
with EP17 tests/pilot; its actual terminal budget share remains to be recorded.

Earlier execution observations follow.

## 2026-10-04 22:17 UTC — `adaptive_search`: ordinary patience stop adopted

Root adopts termination of the exposed-data primary development search at
45 valid comparisons and12 consecutive counted nonprogress comparisons after
the last comparable incumbent gain at33. Independent stopping/coverage review
and complete compact-only interpretation agree. The minimum28, maximum64,
two adaptive rounds/leader challenges and minimum-family obligations are
reconciled in `primary_post_basis_stopping_review.md`; alternatives24/39 exceed
40%. Positive-finalist controls do not authorize promotion of an ineligible
candidate and need not be duplicated merely to report this negative closeout.
The result is **development-only, with no candidate meeting the registered
practical requirements**. It is not absence of LFP information or failure of
every possible decoder. The best comparable W/HW primary increment remains
0.0026788544310999856 (margin0.01) and identity0.004559394718773168 (margin0.005).
No finalist, future-session opening or conditional spike-population follow-up
is selected. The independent-penalty amendment remains unadopted.

Resource reconciliation preserves the original clock starting October3
22:23:02.824UTC and ending October8 at that time. Historical inclusive bound
66621core-seconds is corrected by replacing46473740's1803s reservation with
1667s terminal time:66485s. Later distinct additions are46476174=547s,
46515379 EP05 share1690/3s,46516505=221s,46518769 EP05 share1962/3s and
46519201=153s. Closed documented upper bound19.062037037core-hours is inclusive
of earlier reproduction/shared costs. Reserving the entire46521343 timestamp
envelope2103seconds gives19.646203704core-hours, conservatively rounded upward
to19.647, far below2000. This is a conservative capacity bound, not exact
consumed CPU or a new charge; shared totals are not added to their shares.
Current allocation remains open for sequential EP17 preparation; actual terminal
cost will replace its reservation once, without reopening scientific search.

Both changed ASTRA programs validated on compute. EP05 figure validation
`astra/v0.0.14/validation_20261004_history_pls_basis_figures.json` passed schema
and semantic checks with five preexisting optional-format warnings. The final
scientific report is being written from released evidence; its documentation
does not authorize further experiments or canonical BR transitions.

Earlier execution observations follow.

## 2026-10-04 22:16 UTC — `adaptive_search`: complete basis diagnostic released;45 comparisons and patience12

The fifteen fresh sessions and final summary completed in46521343 in
210.955795seconds; final summary29.329176seconds, peakRSS894012KiB. Together
with the preserved pilot, all16 original sessions, eight draws, stored HW/PLS
states/heads/scores and original selectors replayed. The complete wrapper is
`runs/adaptive_search/primary_history_pls_basis_remainder_attempt001/remaining_panel.json`;
full report and380316-byte scalar-only export are under
`runs/adaptive_search/primary_history_pls_basis_summary_attempt001/`.
Root extracted scores only after complete success. All global single-panel
and complete-ensemble exact duplicate lists are empty. The one mean basis-only
policy adds one alternative, zero candidates and zero literal nulls. Counts
44→45, patience11→12, post-initial alternatives24/39 and literal nulls15/39;
the W/HW incumbent remains unchanged. This is not yet a recorded stop decision.

Basis-only B gives primary−0.000320968039427831 and identity+0.0002158722477637581;
animal primary values Mihili−0.00023763092295630002/Chewie−0.00040430515589936193,
positive sessions1/6 and4/10. P−B gives primary+0.0006813046425830627 and
identity+0.00118974942105306. Both animal endpoints, strict primary session
majorities6/6 and10/10, and every within-animal omission retain positive signs;
all8 matched draw animal means are positive for both endpoints. B−F gives
primary−0.00000825425862852567/identity−0.0000663796039679276, opposing animal
signs and no passed directional screen; all8 mean identity contrasts are
negative. Restoring true heads on shuffled projections does not show consistent
recovery here. Paired projection correspondence matters conditionally on true
heads, but P/B capacities differ; this is not causal or physiological attribution.
The original PLS/HW/blend results and practical gates are unchanged. Two animals
and exposed development sessions do not establish population generalization.

Independent compact-only interpretation agrees. Existing BR advisory criteria
from ext_review_dir_d2180656a77d inform nested fitting, pseudoreplication and
mechanism limits; no new BR verdict, launch or canonical authority is claimed.
Both figures under `figures/primary_history_pls_basis/` rendered successfully
as PNG/SVG with30-row absolute and12-row contrast source tables. Root visually
checked labels, original margin lines, conditional zero references, animal
symbols and captions. Final coverage/stopping review and resource reconciliation
remain before adopting ordinary patience termination. Current allocation cost
is pending; its full35-minute one-CPU reservation bounds additional cost by
0.5833333333core-hours. No renewed search or unadopted penalty amendment follows.

Earlier execution observations follow.

## 2026-10-04 22:10 UTC — selected fifteen-session basis remainder launched

Changed ASTRA validation and selected shell syntax passed in46521343. Root
launched the exact1500s-internal/1560s-external one-worker remainder in PTY36735,
with more than the required26m10s remaining before22:39:59. The successful
pilot is preserved; no source job or second allocation was submitted. Count
44/patience11 remains until all16 sessions and final replays complete.


## 2026-10-04 22:08 UTC — successful stored-projection pilot; remaining panel selected

Pilot22:07:30.219508–22:07:53.087878UTC completed22.868360seconds; child
21.582976seconds, source10.502404s, stored-projection/head reconstruction
9.993668s, old heads0.023717s, new heads0.019348s, HW replay0.795278s,
prediction writes0.071167s. Peak childRSS482212KiB,1038trials/86electrodes.
All original HW and PLS head/score replays passed; zero new projections fit.
Only resource/replay facts were opened. No new scientific count or effect is
released. Evidence: runs/adaptive_search/primary_history_pls_basis_pilot_attempt001/pilot_resources.json.

Select fifteen fresh sessions preserving the pilot, one single-thread worker
inside current1CPU6GiB allocation46521343. Fifteen measured wrapper durations
plus the required900s final-child reserve total1243.025395s;1500s internal
allows256.974605s additional variability/summary.1560s external plus10s kill
reserve fits the allocation's remaining envelope when launched by22:13UTC
(conservative before22:39:59 expiry). The exact selected shell entrypoint is
outputs/tools/run_primary_history_pls_basis_remainder.sh; syntax and changed
ASTRA validation precede execution. Same source/banks, full16replay and no
partial effects; no extra allocation, retry, selector or scientific amendment.


## 2026-10-04 22:06 UTC — compute placement and ten basis fixtures pass

Allocation46521343 is RUNNING, Requeue0, on sh03-01n68; root verified
SLURM_JOB_ID46521343/SLURM_CPUS_PER_TASK1 inside PTY36735. Slurm start22:04:56UTC,
end22:39:59UTC,1CPU6GiB35minutes/noGPU. The explicit module Python3.12.1 and
unchanged numeric modules are loaded with all four thread caps1. Ten focused
basis-only fixtures passed0.623seconds, including independent head solves,
literal old-state/head replay, fixed selectors, exact whole-panel duplicate
counting, compact export and post-summary deadline refusal. No source effects
were evaluated by fixtures. Changed ASTRA validation precedes the selected
fixed resource-only pilot, using900s outer timeout plus10s termination grace.


## 2026-10-04 22:03 UTC — new allocation46521343 requested once

Slurm returned46521343 for the selected attended1CPU/6GiB/35minute normal
CPU_SKU:7502 allocation, initially pending resources. Requeue0 was applied.
Root owns PTY36735; compute hostname/job-ID verification must precede tests.
Independent final static review now passes all five implementation files and
ten focused fixtures, including post-summary deadline/failure handling.
No test or source job has yet run in this allocation; no duplicate is submitted.


## 2026-10-04 22:03 UTC — bounded attended basis-diagnostic compute selected

The five new code/fixture files are saved. Root and independent static review
have passed core/session/orchestration methods; final extractor/fixture review
is being completed. Select one new attended1CPU/6GiB/35minute/noGPU Slurm
allocation for focused fixtures, changed ASTRA validation, and the already
adopted600s-internal/900s-external fixed resource-only pilot after final review.
The previous allocation is closed. Fresh sh_part/sh_node_feat discovery confirms
normal's7day maximum and CPU_SKU:7502; reuse that hardware for literal numeric
replay. Set all four numerical-thread variables to1 and Requeue0. No remaining-
panel execution or expanded budget is selected; that depends on pilot facts.
One coordinator owns the PTY, scheduler and all compute execution.


## 2026-10-04 22:00 UTC — shared allocation46518769 closed and accounted

Root exited the attended compute shell after the selected preparation finished.
Slurm46518769 completed exit0 at22:00:53UTC, start21:28:11, elapsed1962seconds
on one CPU: **0.545 allocated core-hours total** across EP05/EP15/EP17.
Charge this episode an equal bookkeeping share0.1816666666666667 core-hours;
these three shares total the one allocation and are not measured process CPU.
Do not add the shared total again. Slurm TotalCPU61.855seconds and stepMaxRSS
1093668K are separate measurements. PTY34288 is closed; no compute session
remains and no tests may fall back to a login node. Prior closed jobs and
separate EP05 batch46519201 (153seconds,0.0425coreh) retain their own costs.
Further prepared work needs a new bounded attended allocation; no automatic
renewal, supervisor, keepalive or duplicate job exists.


## 2026-10-04 21:52 UTC — `adaptive_search`: basis-only PLS diagnostic adopted

Root reviewed the complete [post-PLS coverage recommendation](primary_post_pls_coverage_review.md)
and adopts its exact conditional diagnostic for implementation, focused checks
and the fixed resource-only pilot after code review. Prior coverage options
are complete through44 comparisons; this is an optional unresolved explanatory
contrast, not a missing family or a filler comparison to meet patience.

The completed PLS/full-target controls change projection and head training
labels together. Reuse each stored shuffled-target PLS transform verbatim,
restore only its ridge head to original native U, and keep the already selected
candidate penalty. This separates P-minus-meanB projection correspondence
from meanB-minus-meanF head correspondence conditional on the same shuffled
projection. Match each B/F draw exactly; disclose that P/B effective capacity
still differs. Preserve all eight draw contrasts and the original directional
screens. No new rank/grid/selector, normalization or physiological claim.

Replay original source/HW support/state/B/targets, stored candidate and full-
target heads at all five penalties before new release. No projection refit
or tolerance relaxation may repair a mismatch. Require all16 sessions and
conservative literal full-panel/ensemble duplicate accounting before release.
A distinct complete mean B policy adds one alternative comparison, no literal-
null credit, no candidate promotion and no incumbent reset. Incoming44/p11,
alternatives23/38 and nulls15/38 remain until then. Stopping still needs its
separate complete coverage/accounting review.

Selected pilot is Chewie_20150313,1CPU6GiB/one numerical thread/noGPU,
600s internal/900s external, fresh primary_history_pls_basis_attempt001 bank
and primary_history_pls_basis_pilot_attempt001 resource report. Actual
remaining-panel resources must follow measured pilot facts; no repeated
source job is selected yet. Existing core/wall budgets, previous failures
and unadopted independent-penalty amendment are unchanged. Reused BR review
criteria (ext_review_dir_d2180656a77d) concern nested training, pseudoreplication
and fit-as-mechanism limits; they are advisory, not a BR verdict or launch.


## 2026-10-04 21:48 UTC — `adaptive_search`: complete PLS panel; 44 comparisons, patience 11

Slurm46519201 completed with exit0 at21:40:35UTC after153seconds on one
CPU (0.0425 allocated core-hours). TotalCPU127.262s and batchMaxRSS1965528K
are separate process measurements. The remainder wrapper completed141.412s,
including18.218s full-panel replay/summary. The preserved resource pilot plus
fifteen fresh sessions complete all16. Compact evidence was released only
from the completed summary and passed its512KiB bound (469031bytes):
[runs/adaptive_search/primary_history_pls_summary_attempt001/compact_evidence_attempt001.json](runs/adaptive_search/primary_history_pls_summary_attempt001/compact_evidence_attempt001.json).

PLS primary increment0.00036033660315523165 and identity0.001405621668816818
miss the original0.01/0.005 margins. Primary is0.00043220384738284584 in
Mihili and0.0002884693589276175 in Chewie, with5/6 and8/10 positive sessions.
Both animal identity means and every within-animal omission remain positive.
The full target-label-refit mean control is−0.0003127137807993053 primary,
0.0002822518517316857 identity. Candidate-minus-control is
+0.000673050383954537/+0.0011233698170851326; all eight individual draws
are worse in both animal summaries. This supports fitted target correspondence
under the fixed PLS recipe, not biological mechanism or matched effective
capacity. Every128 candidate and1024 control fold fits retained two components;
all16 outer-session selectors chose the existing strongest lambda10. No grid
or component-count extension follows from this result.

PLS loses both independently selected HW and the W/HW blend: primary contrasts
−0.0019809178700217526 and−0.0023185178279447544. The blend incumbent stays
0.0026788544310999856. Even lambda-matched HW is better on primary and identity.
Literal full-panel duplicate checks find neither new policy duplicates a
previous anchor. Candidate then mean control add two distinct comparisons:
42→43→44, patience9→10→11/12, post-initial36→38, alternative credit22→23,
literal-null14→15. Search remains open under its original stopping policy;
no promotion, mechanistic follow-up, unseen-session access or penalty amendment.

Independent compact-only interpretation agrees. The scalar figure renderer
initially lacked matplotlib in the active module environment; module discovery
found installed py-matplotlib/3.8.3_py312. Loading that module repairs plotting
only, with no installation or scientific rerun. The figure rendered successfully with24 plotted scalar rows and passed visual
inspection; PNG/SVG/CSV are under figures/primary_history_pls. Shared allocation46518769
preparation cost remains to be finalized separately and charged only once.


## 2026-10-04 21:35 UTC — `adaptive_search`: remaining PLS panel submitted

Root's compute shell passed final batch syntax and changed ASTRA schema/
semantics (`validation_20261004_history_pls_remainder.json`). Submitted the
recorded batch once; Slurm returned **46519201**. Its1CPU/6GiB/35minute
resources and1980/2040s aggregate bounds remain as selected from the pilot.
Root owns monitoring. No score/count is released until all16 sessions replay.


## 2026-10-04 21:34 UTC — `adaptive_search`: successful PLS pilot and remaining panel selected

The fixed pilot completed21:30:30.859304–21:31:25.411957UTC in54.552643s.
Its child took52.199735s: source41.703751s, fit/scoring10.107919s,
projection6.976308s, heads0.025070s, original-HW replay0.162230s and
prediction writes0.029122s. Peak childRSS434480KiB;1038complete trials and
86physical electrodes. Every original HW replay passed. Candidate and all
eight controls retained two components in every held direction. Root opened
only the permitted resource report; scientific effects and counts remain closed.

Root and independent code review also passed the prepared remainder branch.
Select exactly fifteen fresh original sessions, preserving the pilot, with
one single-thread worker on1CPU/6GiB/35minutes/noGPU/no-requeue. Fifteen
pilot-wrapper durations plus the mandatory900s final-child reserve total
1718.29s; the1980s internal bound leaves261.71s additional allowance,
including summary/replay. External envelope2040s plus10s termination reserve
fits the2100s batch. The existing shared shell has insufficient remaining
time for this complete envelope, so submit a separate substantive batch while
using that shell for already selected EP15/EP17 preparation.

Fresh wrapper is `primary_history_pls_remainder_attempt001`; complete summary
is `primary_history_pls_summary_attempt001`. The batch is
`tools/run_primary_history_pls_remainder.sbatch`; logs go to the fresh
`primary_history_pls_submission_attempt001` directory. No changed recipe,
partial-panel score release, duplicate pending work, automatic retry or
reset budget. Final shell syntax and changed ASTRA validation precede submission.


## 2026-10-04 21:30 UTC — `adaptive_search`: fixed PLS resource pilot executing

Root and independent scientific/code reviews passed the stable core,
session runner, pilot branch and compact extractor; ten compute fixtures and
changed ASTRA validation passed as recorded below. Select and execute the
already adopted fixed Chewie_20150313 pilot in verified46518769, fresh bank
`runs/adaptive_search/primary_history_pls_attempt001/Chewie_20150313` and
wrapper `runs/adaptive_search/primary_history_pls_pilot_attempt001`.
One CPU/one numerical thread,6GiB allocation,600s internal and900s outer
envelope, no GPU/retry. Root opens only `pilot_resources.json` until the
full sixteen-session release; no scientific charge or score follows from
the pilot alone. The prepared generic remainder mode has no selected resources.


## 2026-10-04 21:29 UTC — `adaptive_search`: ten PLS fixtures pass on compute

Root verified46518769/sh03-01n68.int with1CPU; its35-minute limit runs
21:28:11–22:03:11UTC and Requeue=0. Module Python3.12.1/NumPy1.26.3 and
one numerical thread were explicitly bound. All ten independent source-free
PLS fixtures passed in0.698s. The changed ASTRA program passed schema and
semantic validation in `astra/v0.0.14/validation_20261004_history_pls.json`.
Root's static core/session/pilot/extractor review passed. Final independent
runner review precedes source execution; pilot effects remain closed.


## 2026-10-04 21:25 UTC — `adaptive_search`: bounded shared compute selected

Core PLS implementation and ten independent fixtures are prepared; root's
initial static review found no scientific blocker. Request one attended
normal/CPU_SKU:7502 allocation with1CPU/6GiB/35minutes/noGPU/no-requeue.
This aggregates EP05 focused checks and its already selected resource pilot,
EP15's selected exact role assignment, and EP17's bounded metadata stage if
its implementation and review finish in time. Run substantive commands
sequentially with all numerical threads fixed to one. Verify the actual shell
host/job before any test or scientific command. Pilot source execution still
requires final runner review and passing fixtures. There is no pending duplicate,
automatic renewal or successor source-score release in this request.

The first submission command was rejected locally by `srun` because
`--no-requeue` is not an srun option; exit255, no job ID or allocation was
created. This is a diagnosed command-line error, not a scheduler failure.
Correct the launcher after checking its supported options; preserve the
same resource request and do not create a duplicate allocation.

The corrected supported srun request returned job46518769, initially queued,
PTY34288. Root explicitly sets Requeue=0 on this owned job; the unsupported
flag is not retried. No test or source command is sent until the actual
compute host and allocation are verified.


## 2026-10-04 21:17 UTC — `adaptive_search`: fixed two-component deflated PLS adopted

Root and the independent scientific reviewer passed
`primary_history_pls_proposal.md`. Adopt its exact regression-deflation
transform and full target-label controls within the existing PLS/ridge grammar.
The second effective input direction can leave the initial cross-covariance
span; this tests a distinct compact representation on the original HW inputs.
The earlier stationary covariance-SVD result is contextual, not an isolated
empirical deflation contrast. Original native XY, support, five penalties,
outer-session selection, practical gates, budgets and counters42/patience9
remain unchanged. Controls relearn both transform and head and are not claimed
to have exactly matched realized capacity.

Root verified the proposal's method distinction against the official
[cross-decomposition documentation](https://scikit-learn.org/stable/modules/cross_decomposition.html#plsregression).
The explicit equations define this recipe; package defaults are not used.
Reuse the relevant BR advisory criteria from `ext_review_dir_3d22b4743ac1`
and `ext_review_dir_d2180656a77d`: biological-unit inference, grouped fitting,
full negative-control disclosure and separating prediction from mechanism.
Those criteria are advisory, not a BR verdict or canonical harness launch;
the independent scientific review and root's decision supply the local review.

Authorize uniquely named implementation, focused fixtures and the fixed
Chewie_20150313 resource-only pilot:1CPU/6GiB/noGPU, one numerical thread,
600s internal/900s external. Scores stay closed until full-panel release.
No active compute shell exists at adoption. Root will request a bounded
attended allocation after code is ready, verify its actual host/job and choose
the remaining-panel resources only from the pilot's measured operational cost.
No rank, penalty or solver extension, automatic retry or budget reset is selected.


## 2026-10-04 21:09 UTC — `adaptive_search`: shared preparation allocation closed

Root exited PTY45495 at21:09:26UTC; it returned exit0. Slurm46515379 is
COMPLETED,20:41:18–21:09:28UTC,1690seconds on one CPU,TotalCPU94.957s,
stepMaxRSS6265580K. Shared allocation cost is0.469444444 core-hours once,
bookkept as equal0.156481481-core-hour shares for EP05/15/17 (including
shared coordination), not measured per-episode CPU. EP05 separately adds
46516505's221core-seconds. See root README for the single shared accounting
entry; do not add its total again to the three shares. The compute shell is
closed and cannot be used for successor tests. No renewal was requested.

## 2026-10-04 21:06 UTC — `adaptive_search`: HW105 full panel completed; no incumbent improvement

Slurm46516505 COMPLETED,exit0:0,20:58:44–21:02:25UTC,221seconds on one CPU
(0.061388889 allocated core-hours); TotalCPU120.647s, scheduler batch
MaxRSS6030420K. These scheduler measurements are distinct from child-process
RSS. The remainder wrapper completed all15 fresh sessions in205.093s and
the complete-panel summarizer took11.818s. The preserved fixed pilot supplies
the sixteenth original session. Source reading was substantially faster in
the remaining sessions than the pilot projection; no runtime was padded.
All original HW state/arrays, HW/K/blend selectors and new scores replayed.

Full evidence is
`runs/adaptive_search/primary_history_rbf_summary_attempt001/primary_history_rbf.json`.
The complete-only extractor succeeded but its initial output included nested
anchor/session records and was unnecessarily large. A display-only repair
now projects explicit scalar, session and omission fields with a512KiB bound;
it succeeded and saved `compact_evidence_attempt001.json` alongside the
unchanged full report. No fitting, selection or scientific result was rerun.

| Policy | Animal-balanced primary increment | Identity increment |
| --- | ---: | ---: |
| HW105 nonlinear weighted history | 0.0014729045689071778 | 0.003169133294973225 |
| Mean of eight HW105 training-pairing controls | -0.00036720936232167897 | 0.0004991694518412006 |
| Original separately selected linear HW | 0.0023412544731769847 | 0.0056143839673429 |
| Original separately selected current-bin K | 0.0015620797368965428 | 0.004254813102499833 |
| Original constant current/history ridge blend | 0.0026788544310999856 | 0.004559394718773168 |

All16 outer selectors choose lambda0.001, the existing lower grid boundary;
this does not authorize extending the grid. HW105 fails both original0.01/
0.005 margins. Its primary means are0.0007544257207686869 (Mihili) and
0.0021913834170456687 (Chewie), with5/6 and9/10 positive sessions; both
metrics remain positive under every within-animal session omission. It loses
both metrics to selected HW and the blend in both animals and every such
omission; only one session in each animal has a positive primary contrast
against either anchor. Relative to current K, primary gain is positive in
Mihili and negative in Chewie, while identity loses in both animals. Matched
nominal-lambda contrasts are descriptive; they do not replace separately
selected anchors or establish equal capacity.

Against the capacity-matched mean pairing control, HW105 gains
0.0018401139312288566 primary and0.0026699638431320243 identity. Those contrasts
are positive in both animals, all16 primary session contrasts and all within-
animal omissions. Each of the eight draw contrasts is also positive in both
animals for both metrics. This supports predictive training-pair correspondence
conditional on the frozen kernel map. It does not identify a nonlinear
mechanism, specifically cross-lag interactions or a population-level effect;
the eight controls are not independent biological replicates.

Both new recipes are exact-duplicate-free, counted candidate then mean control.
Valid comparisons rise40→42; patience7→8→9 of12 because the primary incumbent
remains0.0026788544310999856. Post-initial comparisons34→36, alternative-
explanation credit21→22 and literal-null credit13→14. The40percent allocation
rule applies to alternative-explanation credit. No promotion, stop, independent-
penalty amendment, spike follow-up or new data access is authorized. The
original2000-core-hour/120-hour clock and all consumed work persist.

Root requested BR interpretation criteria at21:04UTC with permitted scalar
results and design details, no source path or canonical binding. Directive
`ext_review_dir_d2180656a77d` returned the generic rubric and no tailored checks;
it did not review OAK artifacts or issue an independent scientific verdict.
Its use is advisory only. Independent local interpretation passed the above
reading; any further comparison needs a scientifically motivated proposal
within the unchanged grammar rather than a new penalty grid.

The source-backed two-panel figure and30-row table are
`figures/primary_history_rbf/primary_history_rbf_comparison.{png,svg,csv}`.
Rendering used the existing module `viz py-matplotlib/3.8.3_py312` in46515379.
An initial reader-size refusal occurred before figure-directory creation;
the renderer now reads the compact projection. Root corrected the blend
caption to current/history ridge and repaired label spacing after visual
inspection. Final rendering passed and was visually reviewed. These are
two-animal development summaries, not confidence intervals. HW exceeds only
the identity margin; all five displayed primary means miss0.01.

A distinct genuinely deflated PLS compression proposal is being prepared
from the previously identified covariance-compression question. It remains
unadopted and unexecuted; the completed nonlinear result does not justify
extending its grid or manufacturing comparisons to reach patience.

## 2026-10-04 20:55 UTC — `adaptive_search`: HW105 remainder submitted

Compute-side shell syntax and pinned ASTRA schema/semantic validation both
passed (exit0); the latter is
`astra/v0.0.14/validation_20261004_history_rbf.json`. Root submitted the
recorded exact batch once and Slurm returned **46516505**. The job uses the
selected1CPU/6GiB/35minute bound, with no requeue, GPU or automatic retry.
Root observed it PENDING at20:56UTC; submission is not proof of execution.
Root owns subsequent status, logs and complete-result release. Independent
EP15/EP17 source work continues in attended job46515379.

## 2026-10-04 20:54 UTC — `adaptive_search`: successful HW105 pilot; full remainder selected

The fixed pilot completed in job46515379 with exit0,20:50:07–20:51:08UTC.
The wrapper elapsed61.118s; child60.239s includes54.246s source reading,
5.358s fitting and0.247s original-HW replay. PeakRSS541404KiB; all1038 trials,
86 physical electrodes and every original-HW replay passed. See
`runs/adaptive_search/primary_history_rbf_pilot_attempt001/pilot_resources.json`.
No scientific effects or comparison counts were released.

Root selects the preserved pilot plus exactly15 fresh sessions in one bounded
batch, using the reviewed `tools/run_primary_history_rbf_remainder.sbatch`:
normal/CPU_SKU:7502,1CPU,6GiB,35minutes,noGPU,no-requeue,one worker and one
numerical thread. At pilot cost the remainder projects about15minutes of
useful work. The internal1980s and external2040s bounds retain the600/900s
child limits and full900s dispatch reserve. The current attended session
cannot hold that complete envelope; this separate substantive allocation is
therefore required. Original budgets/deadline and all scientific rules persist.

The fresh wrapper attempt is `primary_history_rbf_remainder_attempt001`,
full summary `primary_history_rbf_summary_attempt001`, and all16 banks stay
in `primary_history_rbf_attempt001`, under `runs/adaptive_search/`.
Slurm output/error go to `primary_history_rbf_submission_attempt001` under
the same parent. Root reviews shell syntax and changed ASTRA on compute,
then submits this exact batch once. Returned job and result follow here;
no queued/running/completed state is inferred from this selection.

## 2026-10-04 20:50 UTC — `adaptive_search`: nine HW105 fixtures pass; fixed pilot selected

All nine `test_primary_history_rbf` fixtures passed in0.416s in verified
compute job46515379, using explicit module Python3.12.1 and one numerical
thread. Select the prospectively fixed Chewie_20150313 resource pilot now,
with600s internal/900s outer child limits and at least900s remaining before
the allocation ends21:11:22UTC. The command is
`python3.12 outputs/tools/run_primary_history_rbf_panel.py pilot` with
`--source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source"`,
`--trial-table outputs/runs/source_support/attempt002/trial_identity.csv`,
`--scalar-bank-root outputs/runs/adaptive_search/primary_reference_attempt001`,
`--hw-bank-root outputs/runs/adaptive_search/primary_history_repeatability_weight_control_attempt002`,
`--bank-root outputs/runs/adaptive_search/primary_history_rbf_attempt001`, and
`--out outputs/runs/adaptive_search/primary_history_rbf_pilot_attempt001`.
Only resource/count/replay evidence is released from this pilot. Full-panel
resource selection follows its measured result; comparisons remain40 and
patience7/12 until a complete valid comparison is released.

## 2026-10-04 20:44 UTC — `adaptive_search`: implementation reviewed; compute verified

Root and the independent reviewer passed the complete new HW105 core,
session/remainder runners, complete-result extractor and nine focused
fixtures. At20:41:47 UTC root verified compute `sh03-01n68.int`/46515379 in
PTY45495; it ends21:11:22 UTC and Requeue=0. Use the module's explicit
Python3.12.1 executable; an EP15 fixture import already diagnosed bare
`python` as inherited Miniforge3.13, incompatible with the module's NumPy.
The selected EP05 check is `python3.12 -m unittest -v test_primary_history_rbf`
from `outputs/tools`; the fixed resource pilot remains conditional on passing
it and fitting the full900second envelope in this attended allocation.

## 2026-10-04 20:37 UTC — `adaptive_search`: shared bounded allocation queued

The previously recorded attended request returned Slurm **46515379**, queued
through coordinator PTY45495: normal/CPU_SKU:7502, one CPU,6 GiB,30 minutes.
The root will verify its actual compute shell before testing. EP05 code and
independent review are in progress; no new session bank or score exists.
Use this allocation only for reviewed checks/pilots that fit its remaining
time, sequentially with one numerical thread. EP15 preparation may run first;
EP17 preparation may also use it. Final elapsed core-seconds are charged once
across the actual stages, not once per episode.

## 2026-10-04 20:36 UTC — `adaptive_search`: nonlinear historical-state comparison selected

The scientist requested additional useful compute after observing only the
EP12 dependency chain in the queue. A fresh independent review identified an
authorized successor that does not require the pending blend-penalty amendment:
apply the existing 105-landmark Nyström RBF operator to the fixed HW history
state, with eight capacity-matched training-pairing controls. Root reviewed and
prospectively adopted `primary_history_rbf_proposal.md` at 20:35 UTC. The prior
coverage audit did not close this allowed scientific question. Waiting for the
optional penalty amendment should not have blocked this independent work.

The question is nonlinear prediction on the same weighted historical state,
not proof of cross-lag interactions or a physiological mechanism. The original
supports, native targets, folds, landmark seeds, bandwidth rule, numerical
regularizer, five penalties, gates and resource clocks remain fixed. Controls
permute complete training feature trajectories after kernel construction and
share its exact Gram; this matches their conditional head capacity, not the
capacity of linear HW. At most one candidate and one mean-control recipe may
be counted, candidate first, only after a complete distinct sixteen-session
panel. Incoming state remains 40 valid, patience 7, alternatives 21/34 and
literal nulls 13/34. New implementation and tests are still in preparation.

Read-only BR request `ext_review_dir_3d22b4743ac1` at 20:35:49 UTC returned the
generic 125-criterion external-review directive with `tailored_checks=[]`.
It did not read the proposal or provide a scientific verdict. Its leakage,
exchangeability, pseudoreplication and mechanism cautions are already covered
by the proposed comparison; it adds no launch approval or canonical action.

Root will request one attended normal/CPU_SKU:7502 allocation with one CPU,
6 GiB and 30 minutes for the new EP05, EP15 and EP17 focused checks and any
reviewed pilot that fits the remaining time. Each child uses one numerical
thread and runs sequentially. Record the returned allocation before use and
charge its elapsed core-seconds once across these episodes. The fixed EP05
Chewie_20150313 pilot has a 600-second internal/900-second outer envelope;
its effects remain closed. No remainder resource plan or execution is implied
until that representative pilot completes and its cost is reviewed.

## 2026-10-04 10:17 UTC — `adaptive_search`: reviewed single-exception amendment awaits human decision

Root and independent scientific reviewers passed
`primary_independent_penalty_blend_amendment.md` as an **UNADOPTED decision
packet only**. Root requested an explicit human decision to release the
original prohibition on independent component penalties for one separate
exploratory challenger. No reply or authorization has arrived. Broad resume
authorization does not erase the adopted after-outcomes prohibition.

The proposed challenger keeps the existing full-support W/HW arrays,
five penalties per component and original five alpha values. Its 125 nominal
triples reduce to 85 canonical labels; all labels remain in the declared
selector even if equal predictions share computation. Exact ties prefer
the old diagonal policy and preserve its original penalty/alpha ordering.
It changes one interpretable constraint, increases adaptive selection
flexibility, requires no new source or neural fit and promises no gain.
All original margins, stopping/resource limits and future-data boundaries
remain binding. At most one distinct candidate and zero control/alternative
credit would follow from a complete run; frozen W/HW/blend/H2/time-blend
selected panels provide independent literal duplicate checks.

This is supported as a prospective scientific question because component
covariance spectra and error cross-terms differ. It is not an implementation
repair or a way to fill patience. Implementation, score computation, ASTRA
activation and resource planning for it await the explicit decision. The
operative policy and original proposal are unchanged; completed state remains
40 valid, patience7, alternatives21/34 and literal nulls13/34. No EP05
compute allocation remains open.

## 2026-10-04 10:14 UTC — `adaptive_search`: compute session closed and accounted

Root exited the finished PTY at 10:13 UTC, tool exit 0. Final accounting for
job 46476174 is COMPLETED, 9:07 elapsed, one allocated CPU, exit0:0,
TotalCPU3:18.333. The compute step reports 8:59 elapsed and MaxRSS2193M.
Charge the allocation's 547 core-seconds (0.151944 core-hours) once to EP05;
individual child durations are not additional allocation charges. No live
EP05 compute session remains, and no further allocation was requested.

The completed forty comparisons retain all frozen scientific and data-role
boundaries. Independent review finds a scientifically distinct possible
follow-up in component-specific blend regularization, but the original
adopted blend proposal expressly prohibited adding independent penalties
after outcomes. That proposal remains unmodified. A separate unadopted
amendment is being prepared for an explicit prospective decision; neither
broad resume authorization nor a new recipe name overrides that restriction.
No implementation or new scores follow from preparing the decision packet.
The original twelve-comparison patience condition remains unmet at seven.

## 2026-10-04 10:12 UTC — `adaptive_search`: actual-blend I/R complete; 40 valid, patience 7

The fifteen-session remainder and summary completed in job 46476174,
10:07:50.500375–10:10:43.485655 UTC, in 172.985281 seconds. Summary took
15.090206 seconds; maximum child RSS was 341,208 KiB. All sixteen banks,
original selectors/supports, every reduced-HW prediction, newly fitted W
state, saved mixture score/cross-term and literal duplicate flag replayed
exactly. The COMPLETE-only extractor succeeded. Evidence is
`runs/adaptive_search/primary_history_blend_intersection_summary_attempt001/primary_history_blend_intersection.json`
and `compact_evidence.json`; parent is the corresponding `_panel_attempt001`.

| Complete panel | Primary increment | Identity increment | Primary Mihili / Chewie | Positive primary sessions |
| --- | ---: | ---: | --- | --- |
| Guide-intersection blend I | 0.0019535755758133084 | 0.003575156889698564 | 0.000978262164 / 0.002928888988 | 4/6, 9/10 |
| Mean eight random-support blends R | 0.0014891879928673733 | 0.002917398780867341 | 0.000757943773 / 0.002220432213 | 5/6, 8/10 |

Both original practical margins fail for both panels. Both metrics remain
positive for each animal and every within-animal omission, with identity
positive in all sixteen sessions. Fifteen sessions select lambda1/alpha0.75;
Chewie_20161007 selects lambda0.1/alpha0.5 using the other fifteen sessions.
Every R draw uses that session's I-selected parameters. This selector change
must remain distinct from comparing mixtures to the formerly selected HW.

| Direct common-denominator contrast | Primary difference | Identity difference |
| --- | ---: | ---: |
| I minus mean R | +0.0004643875829459353 | +0.0006577581088312224 |
| I minus full original blend | -0.0007252788552866772 | -0.0009842378290746043 |
| R minus full original blend | -0.0011896664382326126 | -0.0016419959379058268 |
| I minus own HW at new I-blend lambda | +0.00013277687530689817 | -0.00043218776113087263 |
| R minus own HW at new I-blend lambda | +0.00013067230345789417 | -0.0003987931937339108 |
| I minus originally selected I-HW | -0.00006033525348511018 | -0.00015622161736400416 |
| R minus originally selected R-HW | -0.00006136401794078848 | -0.00018642907779957207 |

I-minus-mean-R is positive for both animals and all their omissions, with
primary 5/6 and 8/10 positive sessions; identity 4/6 and 8/10. All eight
overall draw contrasts are positive for both metrics, but draw4 has negative
Mihili primary (-0.00022287140749327968), retaining that sign under every
Mihili omission. Draw2's small positive Mihili primary reverses on omitting
20140218, 20140303 or 20140306. Draw4's positive Mihili identity reverses on
omitting 20140217. These overlapping draws are not biological replicates,
confidence intervals or p-values; guide membership does not identify
contamination.

Both I and R lose to the full original blend in both animal means and every
animal omission for both metrics. At the newly selected lambda, mixture
primary improves over own HW in both means, but I's Chewie effect reverses
when 20161007 is omitted; R's Chewie minimum remains only +0.000002296635.
Identity decreases in all sixteen sessions for both matched-lambda contrasts.
Against previously selected HW, both primary contrasts are negative overall:
Mihili improves, Chewie worsens. I's Chewie primary is negative under every
omission; R becomes just positive only when 20161007 is omitted. Thus a
conditional mixing benefit is not an improvement over the prior selected
reduced-HW policy. All per-draw contrasts and signed omissions remain in
the compact evidence.

Selected W/HW capacities remain separate: I W df7.487957–33.870614 versus
HW19.822966–83.432660; R W7.125151–35.433416 versus HW18.693691–82.162823.
Physical counts are 21–51, W ranks21–51 and HW84–204. The recorded full
native-error cross-products are descriptive and do not isolate a neural
mechanism or establish equal capacity.

Neither selected panel is a literal global duplicate. I then R therefore
advance 39/patience6 and 40/patience7, adding two nonpromotable stress and
alternative comparisons, zero null controls and no incumbent reset.
Alternatives are 21/34 and literal nulls 13/34 post-initial comparisons.
The original full-support constant blend remains best at
0.0026788544310999856; no primary-eligible candidate exists. Search is not
complete: the original twelve-comparison patience condition is unmet.

The root- and independently reviewed outcome-blind plotting script rendered
successfully. Root visually checked `figures/primary_history_blend_intersection_attempt001.png`;
PDF and 2,310-row source CSV accompany it. All seven direct comparisons,
sixteen selections and eight draw exceptions remain visible. Independent
result interpretation and exact accounting passed at 10:12 UTC. The
reviewer additionally checked that Chewie_20161007 gains +0.0015966678
against HW at its new lambda0.1 but loses -0.0022655748 against its formerly
selected HW lambda1, explaining why those two anchors must remain separate.
No next candidate is adopted.

## 2026-10-04 10:07 UTC — `adaptive_search`: blend-support remainder launched

The fixed pilot completed 10:05:09.359619–10:06:14.864292 UTC in
65.504663287 seconds, maximum RSS 263,348 KiB. Global metadata/selector
replay took 4.330373 seconds, saved-artifact replay 0.651190, source reading
56.843373, reduced-W fitting 2.692883 and mixture scoring 0.612291.
All nine supports/360 heads and six exact replay flags passed. Only these
resource/validity fields were opened; pilot scientific scores remain closed.

Root and independent reviewer passed the finite resource projection in
`primary_history_blend_intersection_proposal.md`. Root selected one worker,
one thread, same one-CPU/6-GiB allocation and 1440/1470-second aggregate
bounds, including the final summary. Unchanged 270/300-second children
retain a full 300-second dispatch reserve and drain-before-stop policy.
The conservative projection is 1372.071599 seconds, leaving 67.928401
seconds inside the bound. At dispatch preparation, the actual compute clock
showed 1588.174260 seconds before fixed expiry, above the 1500-second
requirement. Requeue=0 and Restarts=0 were verified at 10:06 UTC.

ASTRA schema/semantic validation passed with the five existing warnings;
see `astra/v0.0.14/validation_20261004_primary_history_blend_intersection_remaining.json`.
The reviewed wrapper now runs fifteen fresh sessions, preserving the pilot,
under `runs/adaptive_search/primary_history_blend_intersection_attempt001`.
Parent and summary roots are the corresponding `_panel_attempt001` and
`_summary_attempt001`. Frozen incoming counts are 38/patience 5,
alternatives 19/nulls 13. Scientific results and counts await complete
selection, saved-array replay and independent duplicate accounting.

## 2026-10-04 10:05 UTC — `adaptive_search`: ten checks passed; fixed blend-support pilot launched

Root verified the actual compute shell as sh03-01n68.int, Slurm 46476174,
one CPU/6 GiB, fixed expiry 10:34:02 UTC. Requeue was disabled and all
numerical thread counts set to one. All ten independently authored and
peer-reviewed `test_primary_history_blend_intersection` fixtures passed in
15.009 seconds. The final core, runners, wrapper and extractor have root
and independent static PASS. ASTRA schema and semantic validation passed;
the five existing optional-format warnings remain unchanged. Evidence is
`astra/v0.0.14/validation_20261004_primary_history_blend_intersection_pilot.json`.

The exact adopted Chewie_20150313 pilot is running with 270/300-second
bounds in fresh `runs/adaptive_search/primary_history_blend_intersection_attempt001/`.
Only resource and validity fields will be inspected before the full panel.
Scientific count remains 38/patience 5. Full-panel resources remain unselected.

## 2026-10-04 10:02 UTC — `adaptive_search`: reduced-W blend implementation reviewed; allocation queued

The adopted blend-support core, session runner and sequential panel wrapper
have root and independent static review. The independent fixture suite has
eight fitting/scoring/accounting cases and two saved-packet runner cases.
These cover the original W arithmetic, exact channel/current layout and
saved HW state, all nine reused HW banks, genuine mixture cross-terms,
I-only parameter selection, mean-eight losses and independent duplicate
counts. No tests or new source fits have run yet.

Root submitted attended job **46476174**, one CPU/6 GiB, normal partition,
verified CPU_SKU:7502, explicit 30-minute limit, for tests, the fixed
270/300-second pilot and a prospective resource-qualified remainder.
It is queued; no duplicate or other allocation was submitted. Root owns
PTY 79568 and will verify its compute hostname/job before commands.
Full-panel bound remains unselected and pilot scores remain closed.
Incoming scientific state stays 38 valid/patience 5, alternatives 19 and
literal nulls 13. No automatic renewal or retry is authorized.

## 2026-10-04 09:54 UTC — `adaptive_search`: actual-blend support stress adopted

Root and independent scientific reviewer passed
`primary_history_blend_intersection_proposal.md`, then root adopted it before
any reduced-W fit or new blend score. The unchanged incumbent is the original
constant blend. Its current/history error cross-term on the fixed guide
supports remains unmeasured by the earlier HW-only I/R comparison. This
specific question motivates the optional support stress; it is not a new
required family or a positive-result claim.

Use exactly metadata002's I plus eight ordinary random supports, unchanged
original trials/Y/B/D, and the original 25-cell constant-blend selector.
Refit reduced current-W only, retaining exact channel-then-current array
layout and replaying mean/SD/T/P/a against saved reduced-HW metadata. All
reduced-HW predictions are reused with complete score/selector replay. R
uses I-selected lambda/alpha for every draw and averages losses, not
predictions. Full original blend, matched-lambda reduced HW and originally
selected I/R-HW remain separate descriptive anchors. Guide membership still
does not identify contamination.

Each complete distinct I-then-R recipe contributes one nonpromotable stress
and alternative comparison, zero nulls, with no incumbent reset. Independent
literal global checks can reduce either count to zero; a newly fitted
reduced-W endpoint is not automatically a prior full-W duplicate. Incoming
38/patience 5, alternatives 19/nulls 13 among 32 are frozen. Existing depth,
patience, coverage and resource ceilings remain unchanged.

Isolated implementation and independent fixtures are assigned next. Root
selects the fixed metadata pilot Chewie_20150313 with one CPU/6 GiB and
270/300-second bounds after checks. Six GiB reflects the preceding mixed
allocation's 5,929,052-KiB step peak without attributing that peak to this
new recipe. No new allocation or remaining-panel bound is selected yet.
All real-data work awaits the next verified compute shell; the previous
shell is closed. Pilot scores will stay closed until all sixteen complete.

## 2026-10-04 09:52 UTC — `adaptive_search`: timing figure complete; compute shell closed

The root- and independently reviewed `tools/plot_timevarying_history_blend.py`
rendered successfully in job 46473740. Root visually checked the PNG; the
PNG/PDF and 776-row source CSV are saved under
`figures/primary_timevarying_history_blend_attempt001`. Every selected slope,
nine exact zero contrasts, both animal patterns and the algebraic dependence
of the reversal identity contrast remain explicit. No additional model,
outcome opening or comparison count follows from plotting.

Root exited the completed compute PTY at 09:52:30 UTC, exit 0, before its
fixed 09:54:47 expiry. No renewal or next allocation is selected. Current
count remains 38/patience 5. The separate actual-blend support proposal is
still being drafted and reviewed; no source fit is adopted or launched.

Final scheduler accounting at 09:52:46 UTC: job 46473740 COMPLETED in
27m47s with one CPU, 1,667 allocated core-seconds = 0.463056 core-hours,
charged once to EP05. Total CPU was 6m02.124s; the compute step lasted
27m42s, MaxRSS 5,929,052 KiB, exit 0. That step peak covers the entire mixed
allocation and is distinct from the individual Python runner peaks; no
per-stage attribution is inferred. EP12 job 45736217 remained RUNNING at
5d00h23m20s and 45736218 Pending Dependency. The stopped log routes remain
closed.

## 2026-10-04 09:49 UTC — `adaptive_search`: time-varying blend complete; no primary improvement

Independent completed-result/count review subsequently passed. The review
also notes that the selected-minus-reversed identity contrast is algebraically
twice the selected-minus-half-blend identity contrast here: wrong-minus-correct
SSE cancels prediction-quadratic terms and is affine in the prediction.
Those two identity contrasts therefore are not independent corroboration.

The fifteen fresh artifact sessions and final summary completed in job
46473740 at 09:47:01.259806–09:48:38.016762 UTC, 96.756950 seconds, exit 0;
summary 7.837965 seconds, maximum child RSS 695,944 KiB. All four original
selectors/supports/scores, every selected/reversed prediction and exact
duplicate flag replayed. Evidence is
`runs/adaptive_search/primary_timevarying_history_blend_panel_attempt001/remaining_panel.json`
and `primary_timevarying_history_blend_summary_attempt001/primary_timevarying_history_blend.json`
with its complete-only `compact_evidence.json` and selected arrays. No source
or neural refit was used, and partial/pilot scores were not opened.

The candidate's primary is 0.002646676360074717 and identity
0.004602381426280034: both original margins fail. Mihili/Chewie primary
means are 0.001555847610/0.003737505110, identity
0.003713451805/0.005491311047. Primary signs are 5/6 and 9/10; identity is
positive in all 16, and both metrics remain positive under every animal
session omission. All outer folds select lambda 1. Nine select zero slope
and literally reproduce their original half-blend predictions; seven choose
slope +0.25, ramping history weight from 0.25 to 0.75. These seven are four
Mihili sessions (0303/0304/0306/0307) and three Chewie sessions
(20161005/20161007/20150629). The complete selected panel is distinct from
each of the four original anchors; the nine unchanged rows do not make it
a global duplicate.

Against the constant blend, primary falls 0.000032178071 while identity
rises 0.000042986708. Primary animal differences are +0.000016781184 Mihili
and −0.000081137327 Chewie, with those signs preserved under every animal
omission. Only 3/6 and 0/10 session contrasts improve primary; all three
active Chewie ramps lose. Nine zero differences are preserved. Identity
animal differences are +0.000079617574/+0.000006355841, but Chewie reverses
to −0.000008576713 when 20150629 is omitted. Independent and shared-lambda
half-blend contrasts coincide exactly, since all penalties are 1.

Selected minus same-lambda reversed schedule has primary −0.000022491703
and identity +0.000085973415. Mihili primary improves, Chewie primary loses
under every omission; Chewie's small identity advantage again reverses when
20150629 is omitted. Reversal and its nine literal zero-slope duplicates
add no scientific/null/alternative count. The learned early/late allocation
does not improve the primary endpoint; it identifies neither physiological
latency nor a stable beneficial timing mechanism.

Against W, the schedule gains +0.000471825532 primary/+0.001097975956
identity, both animals/all omissions positive. Against HW it gains
+0.000305421887 primary but loses −0.001012002541 identity in every session;
Chewie primary reverses under omissions of 20161007 or 20150630. Against
H2W it gains +0.000181417155 primary in both animals/all omissions; identity
is +0.000022573071 overall but negative in Chewie. These reference advantages
do not rescue the practical thresholds or loss to the incumbent.

This is distinct candidate 38, advancing patience 4→5 with no reset.
Comparable best remains the constant blend's 0.0026788544310999856.
Literal nulls remain 13/32 and alternative checks 19/32 post-initial.
Independent result review and figure preparation are assigned. The next
unadopted support proposal concerns the actual constant blend's W/HW error
complementarity on the already fixed intersection/random supports; the HW-only
support panel did not measure that cross-term. No new source run is adopted
or launched by this result, and the conditional spike follow-up stays closed.

## 2026-10-04 09:47 UTC — `adaptive_search`: timing pilot complete; bounded remainder launched

The fixed saved-artifact pilot completed in job 46473740 at
09:45:38.010208–09:45:43.664111 UTC, 5.653899 seconds, exit 0;
artifact read/replay 5.598201 seconds, mixture scoring 0.054703 seconds,
peak RSS 466,612 KiB. All six replay flags passed on 1,038 original trials
and 86 electrodes. There were zero provider-source reads or neural refits.
Only resource/support/replay fields were opened; its scientific scores remain
closed until the full panel completes.

Root selected sequential 300/330-second remaining bounds with one thread
and 2 GiB, unchanged 60/90-second children. Fifteen pilot equivalents
84.808483 seconds, a full 90-second child reserve, and sixteen pilot
equivalents 90.462382 seconds for final replay leave 34.729136 seconds for
overhead/variation. Proposal and ASTRA now record this selection. ASTRA
validation passed with the same five optional-format warnings in
`astra/v0.0.14/validation_20261004_primary_timevarying_history_blend.json`.
Program preparation completed after the authorized pilot and validation
preceded the remaining-panel launch; no scientific definition changed.

At 09:46:37 UTC, scheduler state confirmed the same one-CPU/6 GiB allocation,
Requeue 0/Restarts 0, with fixed expiry 09:54:47 UTC. Root launched the
fifteen fresh sessions and complete summary around 09:47 UTC with more than
the required 360 seconds remaining. The completed pilot is preserved. No
other source/scoring workload overlaps this run. Count 37/patience 4 remains
until complete release; no partial outcome contributes a scientific trial.

## 2026-10-04 09:45 UTC — `adaptive_search`: linear-time blend checks pass

Root and independent peers passed the new core, saved-artifact session and
summary runner, sequential dispatcher, and complete-only extractor. Eight
independent core fixtures passed in 6.548 seconds and two actual saved-anchor
runner fixtures passed in 1.725 seconds in job 46473740, both exit 0. A fixture
deliberately casting extreme values to forbidden float32 emitted an expected
overflow warning before rejection; no real-data or algorithm failure occurred.
The tests establish literal zero-slope/endpoint arithmetic, native joint-XY
loss and whole-trial wrong pairing, nested animal-balanced selection and all
tie levels, four-anchor exact duplicate counting, selected-lambda reversal,
fixed target layout, and rejection of stale original selections or array drift.
Passing checks are not repeated. ASTRA preparation and the fixed 60/90-second
saved-artifact pilot are next; no new scientific score has been released.

## 2026-10-04 09:41 UTC — `adaptive_search`: intersection figure rendered

Root reviewed `tools/plot_primary_intersection.py`, rendered it successfully
in the verified job 46473740 shell, and visually checked the PNG. Files
`figures/primary_intersection_attempt001.{png,pdf,csv}` preserve 1,608 source
rows including every individual-draw session and omission. The figure keeps
positive own gains, losses to full HW, mean count-matched support advantage,
and the negative Mihili draw-4 primary contrast separate. Its annotations
retain the omission exceptions and unresolved contamination limitation.
Scientific counts remain 37/patience 4; implementation of the adopted timing
recipe and its independent fixtures continues.

## 2026-10-04 09:39 UTC — `adaptive_search`: fixed linear-time blend adopted

Root and independent scientific reviewer passed
`primary_timevarying_history_blend_proposal.md`; root adopted it before any
new prediction combination or score. It asks whether the two useful saved
W/HW maps benefit from a constrained early/late allocation, motivated by
the constant blend and H2W comparisons. The grid was specified while I/R
outcomes were closed. No new source, neural fit or target interval is used.

The one recipe uses shared original lambda and five fixed linear slopes
−0.5, −0.25, 0, 0.25, 0.5 around mean history weight 0.5. Other-fifteen-session
equal-animal primary selection uses exact ties favoring stronger lambda,
smaller absolute slope, then negative sign. Original zero-slope arithmetic
and endpoint-bin values are reused literally. All four originally selected
W/HW/blend/H2W anchors must replay; comparisons with fixed and shared-lambda
anchors remain separate. Reversal is descriptive and uncharged. Exact
global reproduction of any original anchor adds zero depth; otherwise at
most one candidate adds no null or alternative credit. Mean coefficient is
not matched contribution or effective capacity; a slope is not physiology.

Incoming count 37, patience 4, alternatives 19 and nulls 13 among 31
post-initial comparisons are frozen. Best comparable primary remains
0.0026788544310999856. Isolated implementation and independent fixtures are
assigned to separate agents. Root will run passing checks and the fixed
Chewie_20150313 saved-artifact pilot with one CPU/2 GiB, 60/90-second bounds.
Pilot scientific scores remain closed until a complete cohort. The full
bound is deliberately unselected until resource-only pilot review; the
current allocation still expires at 09:54:47 UTC without extension.

## 2026-10-04 09:38 UTC — `adaptive_search`: coverage review and resources reconciled

Independent I/R interpretation and counting review passed. The
`primary_post_intersection_coverage_audit.md` concludes that all four concrete
08:08 generic coverage gaps have now been performed, while retaining failed
promotion margins and the unidentified-contamination limitation. This does
not transfer HW's controls to another recipe. More than two adaptive rounds
and leader/challenger comparisons are documented; patience remains 4, so
no terminal search claim follows. Root agrees that new work must answer a
scientific question rather than manufacture additional counts.

Scheduler accounting for every 46-series allocation referenced in this
episode's execution log, including expanded reference-array tasks, totaled
66,621 allocated core-seconds when the active job was conservatively counted
for its full 1,803-second scheduler limit: 18.505833 core-hours. This is an
inclusive resource upper bound, not a new charge: it deliberately includes
pre-search reproduction and shared portfolio/EP08/EP19 allocations. Existing
per-job accounting remains authoritative and shared jobs are counted once.
It is well below 2,000 core-hours. The search clock from
2026-10-03 22:23:02.824 UTC was 11.254119 hours at the check, below 120 hours.
No extension, duplicate allocation or additional capacity is selected.

The root- and independently reviewed H2W figure rendered successfully in
job 46473740 and was visually checked and shared. Files
`figures/primary_history2_repeatability_attempt001.{png,pdf,csv}` contain
the figure and 852 source rows, retaining signed sessions, animal omissions,
finite-draw ranges and the small fragile identity contrast to the blend.
The prospective linear-time blend proposal is frozen but still unadopted
pending independent scientific review; its family/grid were specified before
I/R outcomes were released.

## 2026-10-04 09:37 UTC — `adaptive_search`: intersection sensitivities complete; neither margin passes

The adopted I/R remainder completed in job 46473740 at
09:33:05.597589–09:35:23.542192 UTC: 137.944611 seconds, exit 0,
summary 1.708227 seconds, maximum child RSS 296,428 KiB. All fifteen fresh
sessions and the preserved pilot passed complete support, original HW
selector/score and saved-array checks. Evidence is
`runs/adaptive_search/primary_intersection_sensitivity_panel_attempt001/remaining_panel.json`
and `primary_intersection_sensitivity_summary_attempt001/primary_intersection.json`
with its complete-only `compact_evidence.json`. All 144 reduced physical
supports are distinct; original full HW was not refitted.

Intersection I has primary 0.002013910829298419 and identity
0.0037313785070625686; mean matched-size random support R has
0.0015505520108081618 and 0.0031038278586669136. Both fail the original
0.01/0.005 margins. I primary animal means are 0.000853450565 Mihili and
0.003174371094 Chewie, with 4/6 and 9/10 positive sessions; R means are
0.000673086082 and 0.002428017939, with 5/6 and 9/10 positive sessions.
Both recipes have positive identity effects in all 16 sessions, and both
metrics remain positive in both animals under every session omission.
All outer folds select lambda 1; R retains I's penalty without retuning.

I minus R is +0.000463358818 primary and +0.000627550648 identity.
Both animals and every animal omission are positive; session signs are
5/6 and 8/10 for primary, 4/6 and 8/10 for identity. This average advantage
does not hold uniformly over all eight random draws: primary contrast to
draw 4 is negative in Mihili and under every Mihili omission; omitting
Chewie_20161005 reverses that draw's equal-animal primary contrast. Draw 2
has a small positive Mihili primary mean that reverses under three omissions.
Draw 4 identity contrast in Mihili reverses when omitting Mihili_20140217.
All eight equal-animal primary and identity contrasts remain positive before
omissions. These finite support ranges are descriptive, not confidence
intervals, new biological replication or a capacity-matched null.

I minus original full HW is −0.000327343644 primary and −0.001883005460
identity. Both animal means are negative; only 3/6 and 3/10 primary session
contrasts are positive, and identity is negative in all 16. Mihili primary
changes sign only when omitting Mihili_20140307; Chewie primary and both
identity animal means remain negative under every omission. Independent
and I-lambda-matched references agree exactly. The smaller guide-defined
support preserves a modest positive prediction increment but loses to full
HW. Unit-guide membership cannot identify spike contamination; that
scientific limitation remains unresolved.

Distinct stress diagnostic I is comparison 36 (patience 2→3), then R is
37 (3→4). Both are nonpromotable support sensitivities: no incumbent reset,
literal null or automatic finalist-control transfer. Literal controls remain
13/31 post-initial comparisons; alternative checks become 19/31. Best
comparable primary remains the blend's 0.0026788544310999856. Independent
coverage/interpretation review and the unadopted next-model proposal continue.
This panel completes the concrete guide-intersection computation; it does
not complete the twelve-trial patience requirement or authorize the
conditional spike-population follow-up.

Independent H2W result/count review also passed. The reviewed H2W plotter
is being rendered in the same verified allocation; no original fit changes.

## 2026-10-04 09:33 UTC — `adaptive_search`: H2W complete; intersection remainder launched

The H2W remainder completed in job46473740 at09:25:34.032722–09:27:24.815263UTC,
110.782534seconds,exit0; summary6.149307seconds,maximum childRSS589,588KiB.
All sixteen original/state/score/selector and new saved-score replays passed.
Evidence is `runs/adaptive_search/primary_history2_repeatability_panel_attempt001/remaining_panel.json`
and `primary_history2_repeatability_summary_attempt001/primary_history2_repeatability_weight.json`
with its complete-only `compact_evidence.json`. Failed/earlier attempts and
the complete pilot were preserved. Both recipes are distinct.

H2W primary0.0024652592046145306 and identity0.004579808354812508 fail the
original0.01/0.005 margins. Mihili/Chewie primary means are
0.001390284026/0.003540234383 and identity0.003636534373/0.005523082337.
Primary signs5/6 and9/10; identity all16; both metrics remain positive in
both animals under every fixed-prediction session omission. All sixteen
outer folds select lambda1. The matched training-surrogate mean has
primary−0.000647263210 and identity0.000600252990. H2W minus that control
is+0.003112522414 primary and+0.003979555365 identity, positive in every
session and every animal omission; each of the eight fixed draw contrasts
also passes these sign checks. These finite draws are not confidence
intervals or p-values. All128 fitting Grams are shared exactly across the
true and1024 surrogate folds; no surrogate-specific Gram or observed
map/design/cross-product identity occurred. Effective df29.127052–43.515519.

Against W, H2W gains+0.000290408377 primary/+0.001075402885 identity,
both animals and all omissions positive. Against HW it gains
+0.000124004731 primary but loses−0.001034575613 identity: the primary
gain is positive in Mihili but negative in Chewie, and identity declines
in every session. Independent and matched-lambda references coincide.
Against the frozen blend, primary falls−0.000213595226, negative in both
animals and every omission; only2/6 and2/10 sessions improve. Identity
rises only+0.000020413636, with Mihili's mean reversing under three session
omissions. The shorter-history challenger therefore supplies no incumbent
improvement. Candidate34 advances patience0→1 and control35 advances1→2;
best comparable primary stays0.0026788544310999856. Literal controls13/29
post-initial comparisons; conservative alternative checks17/29. No primary
eligibility, search completion or conditional spike-follow-up authorization.

After complete H2W release, root froze actual incoming counts35/patience2,
alternatives17/nulls13 in the I/R command. Changed ASTRA validation passed
with the same five optional-format warnings, recorded in
`astra/v0.0.14/validation_20261004_primary_intersection_remaining.json`.
The verified sh03-01n68/job46473740 shell launched the adopted I/R remainder
at09:33:01UTC: one thread,720/750seconds,unchanged270/300 children,
preserved metadata-qualified pilot and fifteen fresh session paths.
Over21minutes remained before09:54:47UTC, exceeding the780-second reserve.
I/R scores remain closed until the entire sixteen-session panel completes.
Scheduler09:33UTC also showed EP12 job45736217 RUNNING at5d00h03m34s of7d,
and45736218 Pending Dependency. Stopped log routes were not reopened.

## 2026-10-04 09:25 UTC — `adaptive_search`: H2W remainder launched

Job46473740 was granted; root verified sh03-01n68,SLURM_JOB_ID46473740,
1CPU/6GiB and the fixed30-minute expiry09:54:47UTC. Requeue was explicitly
disabled, with Restarts0. Root launched the already reviewed and validated
H2W remainder at600/630seconds with one thread and all fifteen fresh paths,
preserving the complete pilot and using the fixed complete blend anchor.
The allocation had over29minutes left, exceeding the660-second launch
reserve. No source workload overlaps it. Scientific scores remain closed
until its whole sixteen-session panel completes; count33/patience0 remains.
The I/R remainder is selected at720/750seconds but waits for H2W completion
and actual incoming counter values, without assuming an outcome or reset.

## 2026-10-04 09:22 UTC — `adaptive_search`: intersection pilot complete; two finite remainders prepared

Final accounting observed09:23UTC: job46472346 COMPLETED in14m35s,
1CPU,875allocated core-seconds=0.243056core-hours charged once toEP05;
stepCPU1m47.446s,peak1,412,356KiB,exit0. New finite job46473740 is pending
for1CPU6GiB/30minutes; no duplicate request was submitted.

All eight distinct I/R fixtures passed in root's allocation: six core checks
in0.516seconds and two actual-runner/stale-original-selector checks in0.243.
Root and independent peers passed the core, runner, sequential dispatcher and
complete-only extractor. Review added exact independent replay of the saved
full-HW outer selector. It also removed an overstrict generic patience ceiling:
the adopted stop requires patience AND completed applicable coverage, while
I/R is itself required coverage. No new scientific stopping rule was introduced.
ASTRA passed with the same five optional-format warnings in
`astra/v0.0.14/validation_20261004_primary_intersection_pilot_and_history2_remaining.json`.

The metadata-selected Chewie_20150313 pilot completed in job46472346 at
09:20:49.345353–09:21:13.268025UTC,23.922664seconds,exit0. Global metadata
validation took1.641233seconds, source10.067647, reduced fits11.756837;
peak323,424KiB. All nine frozen physical supports required unique fits.
Exact original target/B/trial/full-HW score replay passed, with no original-HW
refit. Scientific pilot scores stay closed; count33/patience0 is unchanged.

Root selected a sequential720/750-second remainder,one thread/6GiB,
unchanged270/300 children. Fifteen pilot equivalents358.839957seconds plus
a complete300-second child reserve leave61.160043seconds for summary and
variation. Require780seconds left including cleanup. Incoming scientific
counts remain unselected until actual launch after H2W completion. This
changes resources only; it neither assumes scientific outcomes nor shortens
either bound to fit the current allocation.

Root exited the completed compute PTY about09:22UTC. Final scheduler
accounting had not settled at the first query and will be recorded once
available. A new bounded attended1CPU6GiB/30-minute allocation is requested
for the declared H2W600/630 remainder followed by I/R720/750, with validation,
complete-result extraction and figure work. Source workloads stay serialized,
both complete pilots preserved, and no automatic renewal or requeue selected.

## 2026-10-04 09:18 UTC — `adaptive_search`: H2W pilot complete; remaining bounds selected

Root and independent peer static reviews passed the fixed two-lag core and
session/summary runner. Nine independent core fixtures passed in1.539seconds;
the new actual-runner boundary fixture and strengthened nonlinear omitted-lag
fixture passed in0.217seconds. No other passing tests were repeated. Changed
ASTRA validation passed with the same five optional-format warnings in
`astra/v0.0.14/validation_20261004_primary_history2_repeatability.json`.

The fresh Chewie_20150313 pilot completed in job46472346 at
09:16:53.285116–09:17:08.647683UTC,15.362559seconds,exit0. Source reading
took10.840313seconds, candidate/eight-control fit and score3.690497seconds,
peak389,400KiB. All five original HW/W neural-state, saved component arrays
and scores, capacity maps and capacity-array/score replays passed exactly.
Original1,038 trials and distinct reader−3:+14/model−1:+14/target0:+14
intervals were preserved; no original neural recipe was refitted.
Pilot scientific scores remain closed and add zero depth.

Root reviewed the sequential wrapper and complete-only extractor, then
selected600/630seconds for the fifteen-session remainder with one thread
and4GiB, unchanged270/300 children. Fifteen pilot equivalents230.438389s
plus a full300s child reserve leave69.561611s for summary/variation. A launch
requires660seconds left in an actual allocation; the current allocation
cannot fit it, so no remainder has launched or been shortened. The complete
original blend remains a fixed, prospectively declared anchor. I/R numerical
implementation and independent fixtures continue; count33/patience0 remains.

Scheduler at09:17UTC: EP12 job45736217 stillRUNNING,CPU4d23h43m51s,
RSS151,776KiB;45736218 stillPending Dependency. No stopped log route reopened.

## 2026-10-04 09:14 UTC — `adaptive_search`: all sixteen guide joins complete; result figures rendered

The cached metadata retry completed in job 46472346 at
09:09:14.192090–09:13:19.540138 UTC: 245.348033 seconds elapsed,
33.453381 CPU seconds, 151,840 KiB peak RSS, exit 0. The complete immutable
artifact is `runs/source_support/primary_intersection_metadata_attempt002/primary_intersection_metadata.json`.
All sixteen original guide/event/trial checks passed; spike, LFP and XY value
read flags and score-computation flags remain false. Failed attempt001 is
preserved. This completes metadata qualification only, adding zero depth.

Every intersection is a strict, nonempty subset. In original session order,
Mihili intersection counts are 33,31,40,30,31,21 of96; Chewie counts are
44,51,35,45,48,50 of96,45/86,38/86,34/96,30/96. All nine prescribed supports
per session are distinct and genuinely require fitting. The unchanged
metadata workload proxy selects Chewie_20150313: 1,038 trials,86 original
electrodes,45 intersection electrodes,9 supports,proxy32,201,388,000.
Root released these complete joins for the already adopted numerical I/R
implementation and its 270/300-second pilot; no numerical pilot has launched.
Guide membership still cannot identify the absence or amount of contamination.

Root and independent peers passed both completed-result plotters. Rendering
in the verified compute session exited0; root visually inspected both PNGs.
`figures/primary_capacity_matched_surrogate_attempt001` supplies PNG/PDF and
450 source CSV rows; `figures/primary_history_blend_attempt001` supplies
PNG/PDF and670 rows. Signed session effects, equal-animal means, descriptive
omissions/draw ranges and original margin failures are retained. The blend
figure separates its own positive animal-level gains from the fragile Chewie
advantage over HW and lower identity gain. Both figures were shared with the
user. Count remains33,patience0; no eligibility or promotion follows.

## 2026-10-04 09:09 UTC — `adaptive_search`: blend improves primary incumbent; neither margin passes

All fifteen fresh saved-artifact sessions and the summary completed in
job46472346 at09:08:13.394264–09:08:35.219161UTC,21.824880seconds,exit0;
summary6.411261seconds,maximum childRSS428,704KiB. Evidence is
`runs/adaptive_search/primary_history_blend_panel_attempt001/remaining_panel.json`,
all sixteen banks under `primary_history_blend_attempt001`, and the complete
`primary_history_blend_summary_attempt001/primary_history_blend.json` plus
its `compact_evidence.json` and selected prediction arrays. Original endpoint
supports/scores, independently recomputed selectors and all selected arrays
replayed exactly. No provider source or neural fit was used.

Every outer fold selected lambda1 and history weight0.5 using its other
fifteen sessions. The selected arrays are distinct from both original
components. This is valid candidate33, adding no null or alternative check.
Primary0.0026788544310999856 strictly exceeds HW0.0023412544731769847;
the prospectively adopted counter therefore resets from4 to0 and the blend
becomes the comparable primary incumbent. This is not scientific promotion:
primary remains below0.01 and identity0.004559394718773168 is below0.005.
Literal controls remain12 of27 post-initial comparisons, alternatives16 of27.

Mihili/Chewie primary means are0.001539066426/0.003818642437; identity
0.003633834232/0.005484955206. Primary signs are5/6 and9/10, identity
positive in all16; both metrics remain positive in each animal under every
session omission. Blend minus W is+0.000504003603 primary and
+0.001054989249 identity, both animals/all their omissions positive,
primary6/6 and9/10. Blend minus HW is+0.000337599958 primary but
−0.001054989249 identity. Primary animal contrasts are
+0.000519224023/+0.000155975893, with6/6 and only3/10 sessions positive;
omitting Chewie_20150630 reverses Chewie's mean contrast to−0.000003049014.
Identity contrast is negative in both animals and all their omissions.
Independent and matched-lambda anchor contrasts agree because all selected
penalties are1. The modest primary gain has a real identity tradeoff and a
fragile Chewie advantage; no causal, future-session or practical-effect
success follows from the new best score.

After the complete panel, root launched the adopted fresh cached guide-only
metadata002 at300/330seconds,one thread, in the same verified allocation.
All sixteen joins are still required before numerical intersection fitting;
no spike/LFP/XY values are authorized by this metadata pass.

## 2026-10-04 09:08 UTC — `adaptive_search`: artifact blend remainder running; fixed H2W challenger adopted

Job46472346 was granted and root verified sh03-01n68,1CPU/6GiB,
20-minute allocation ending09:27:25UTC. Default requeue was explicitly
disabled with a successful job update; no automatic renewal/successor is
selected. Changed blend-remainder ASTRA validation passed with the same five
optional-format warnings. Root launched the preserved-pilot/fifteen-fresh
artifact-only remainder at180/210 seconds,one worker, with over19minutes
remaining; no provider source workload overlaps it.

Root and an independent scientific reviewer passed
`primary_next_history_proposal.md`, now adopted before any blend or
intersection outcome release. It selects one new fixed two-lag LMP recipe
(current and immediately preceding bin) plus the existing eight-draw
conditional-capacity training surrogate, on unchanged original support and
baseline. Retained four-lag preprocessing columns/current W priors replay
exactly; only two-lag trace normalization and the new head change. One shared
Gram serves candidate and controls, controls use candidate-selected lambda
and mean SSE, and existing literal maps are reused. Zero-energy conditions
trigger an exact duplicate-proof requirement, never a rounding allowance.
The completed blend, if available before the new summary, is a prospectively
declared fixed anchor without retuning. Root has authorized isolated code and
independent fixtures, not a lag sweep or automatic source launch. Current
count32/patience4 leaves room for this pair even after at most three selected
blend/intersection comparisons; all live stopping and resource limits apply.

## 2026-10-04 09:05 UTC — `adaptive_search`: blend pilot complete; next finite bundle selected

The saved-prediction blend pilot completed09:04:27.835798–09:04:29.037626
UTC in1.201828 seconds, exit0; artifact read/replay1.156490, scoring0.045011,
peak67,584KiB,1,038 original trials/86 electrodes. Both endpoint/support
replays passed; zero provider-source reads or neural refits. Mixture scores
remain closed. Root selected one-worker180/210-second remainder bounds,
unchanged60/90-second children and fresh summary/parent paths; the proposal
and ASTRA contain the pilot-based resource arithmetic.

Root exited the finished compute PTY. Job46469298 completed in28m31s,
4allocated CPUs,6,844allocated core-seconds=1.901111core-hours charged once
to EP05. Step CPU9m34.745s,peak3,240,116KiB,exit0. No renewal was requested.
The next bounded attended bundle is1CPU6GiB/20minutes for the adopted
300/330-second metadata qualification,180/210-second saved-artifact blend
panel, reviewed checks/figure rendering and, only after complete guide
qualification, the already adopted270/300-second numerical intersection
pilot. Source workloads remain serialized. Initial srun syntax included
unsupported --no-requeue and was rejected before job creation; removing that
sbatch-only option preserves the request, with no submitted duplicate.

## 2026-10-04 09:04 UTC — `adaptive_search`: cached metadata checks passed; blend pilot launched

Root and independent peer passed the isolated cached-reference qualifier.
All six focused metadata fixtures passed in0.163 seconds in verified
job46469298/sh03-01n25. The object-reference dtype guard rejects numerical,
untyped object and region-reference tables before conversion; unchanged
qualification, selected-trial dereferencing and partial-failure gating remain
covered. Fresh metadata002 has not launched: its330-second external bound
plus cleanup cannot fit the allocation's fixed09:06:34 expiry.

The first eleven blend fixtures had one synthetic-expectation failure:
at displacement1e-8, subtracting tiny SSE from a much larger baseline SSE
can tie multiple interior primary scores exactly. The implementation correctly
applied the prospective smaller-alpha tie rule. Only the fixture changed:
unit displacement still requires alpha0.5; the tiny case requires an interior,
exactly distinct selected array and count1 despite numerical allclose to both
endpoints. All eleven passed in2.723 seconds after this correction, with no
core, selector or scientific-policy change. Root and peer static reviews of
the core, runner, sequential remainder and complete-only extractor passed.

Changed ASTRA validation passed with the same five optional-format warnings:
`astra/v0.0.14/validation_20261004_primary_history_blend_and_intersection_cached.json`.
Root launched the fixed Chewie_20150313 artifact-only blend pilot about
09:04:24 UTC at60/90 seconds, one thread, fresh
`runs/adaptive_search/primary_history_blend_attempt001/Chewie_20150313`.
No provider source or neural refit is involved. Pilot scientific scores remain
closed; remainder resources remain unselected and count stays32/patience4.

At09:03UTC, EP12 job45736217 remained RUNNING with CPU4d23h29m27s,
RSS151,776 KiB;45736218 remained pending on Dependency. No stopped log
route was reopened. Independent capacity-result interpretation also passed;
the completed comparison32 interpretation above is unchanged.

## 2026-10-04 08:55 UTC — `adaptive_search`: explicit capacity-matched surrogate complete

All fifteen remaining sessions completed in job46469298 at
08:53:50.153223–08:54:47.238076 UTC,57.084843 seconds,exit0; summary3.955635
seconds,maximum childRSS423,576 KiB. The original complete pilot is preserved.
Evidence is `runs/adaptive_search/primary_capacity_matched_surrogate_panel_attempt001/remaining_panel.json`,
all sixteen banks under `primary_capacity_matched_surrogate_attempt001`, and
`primary_capacity_matched_surrogate_summary_attempt001/primary_capacity_matched_surrogate.json`.
Root reviewed and ran the complete-only extractor; its `compact_evidence.json`
contains session/animal/omission/seed effects and shared-capacity records.
Every original neural-state, saved-array/scoring and weighted-covariance
replay passed; the original HW candidate was not refitted.

The new mean-eight surrogate is distinct: comparison32, patience4, no reset.
Literal null controls are now12 of26 post-initial comparisons; conservative
alternative checks are16 of26. The shared conditional-capacity check is
completed, without a new threshold or automatic promotion. HW remains the
comparable incumbent, with primary0.0023412544731769847 below0.01.

The mean surrogate has primary−0.0013666008543514398 and identity
0.0007611256572597324. Mihili/Chewie primary means are
−0.002026472276/−0.000706729432 and remain negative under every within-animal
omission; primary session signs are0/6 and6/10. Its identity means remain
positive in both animals and all animal omissions. A positive surrogate
identity score is compatible with preserved direction/time structure and
correctly paired evaluation features; this is not a signal-free null.

Original HW minus mean surrogate is+0.003707855327528424 primary and
+0.004853258310083167 identity, positive in every session and both animals
under every omission. Mihili/Chewie primary contrasts are
0.003046314679/0.004369395976, identity0.004157565815/0.005548950805.
Every one of eight fixed seed contrasts is also positive for both metrics
in all sixteen sessions. Equal-animal seed primary range is
0.003526351014–0.003871195470; identity0.004732294133–0.005015016608.
These finite draws are not independent biological replication or a tail test.

All original selected lambdas are1. Across128 direction folds/1,024 surrogate
direction fits, the shared design has344–384 columns, trace344–384,
numerical rank340–380 and selected effective df46.124229–73.893893.
There are128 shared Gram eigendecompositions and zero surrogate-specific
ones. No complete fold/draw map, design or response cross-product exactly
duplicates the true-pair arm. This supports the benefit of correct training
assignments at exactly matched conditional ridge capacity; it does not cure
the original practical-effect failure or identify a biological mechanism.

## 2026-10-04 08:54 UTC — `adaptive_search`: capacity remainder launched; bounded metadata repair adopted

Capacity-surrogate pilot completed 08:52:09.238911–08:52:26.312094 UTC in
17.073161 seconds: source10.170935, fitting/scoring5.934139, peak437,260 KiB,
1,038 original trials/86 electrodes, all three exact HW replays passed and
no original-HW refit. Scores remained closed. Root selected four single-thread
workers at420/450 seconds, unchanged270/300 children. Four pilot waves take
68.292644 seconds; a full300-second child reserve leaves51.707356 seconds
of variation allowance. Four measured peaks total about1.668 GiB within4CPU8GiB.
Root and peer passed the wrapper's unchanged-helper routing/count guards;
updated ASTRA validation passed with its same five optional-format warnings.
The fifteen-session remainder launched about08:53:40 UTC in job46469298,
with more than12 minutes remaining before fixed expiry; no overlapping
source workload. Scientific count remains31 until full completion.

Separately, root reviewed and adopted
`primary_intersection_metadata_repair_proposal.md`. Static code inspection
establishes repeated per-trial reference-table reads and duplicate shape-handle
resolutions; it does not prove these solely caused the timeout. The isolated
repair caches only fifteen small object-reference tables per session and
reuses duplicate per-trial shape handles, preserving Fortran indexing,
retained-trial-only dereferencing and every original guide/event/support
check. It adds per-session progress metadata and preserves failed001.
After reviewed code and focused compute checks, a fresh002 may run at
300/330 seconds, one thread/2GiB. This is an explicit finite engineering
resource choice, not an automatic extension or scientific-policy change.
No source retry has started at adoption, and no broken-filesystem diagnosis
or numerical intersection run is inferred.

## 2026-10-04 08:52 UTC — `adaptive_search`: capacity-control pilot launched; guide qualification timed out

All ten independent capacity-surrogate fixtures passed in 0.594 seconds in
job 46469298. Root and peer static reviews passed core/session runner,
including actual fitter-to-validator IDs/maps, independent permuted-design
ridge and smoother-conjugacy oracles, held-data exclusion, unchanged HW
state/add-back, mean-SSE scoring, fixed original lambda and null-only counts.
Changed ASTRA validation passed with the same five optional-format warnings
(`validation_20261004_primary_capacity_matched_surrogate.json`). Root launched
the fixed Chewie_20150313 pilot at 270/300 seconds, one numerical thread,
fresh `primary_capacity_matched_surrogate_attempt001/Chewie_20150313`.
Pilot scientific scores remain closed; remaining bounds are unselected.

The separate guide-only qualification stopped at its 60-second bound:
08:49:09.281194–08:50:09.303298 UTC, 60.022092 elapsed, 29.298867 CPU seconds,
93,704 KiB peak RSS, exit 1. Its traceback was in HDF5 reference dereferencing
for an M1_spikes dataset handle used only for shape access. The failed report
preserves `spike_values_read=false`, `LFP_XY_values_read=false`, and
`scores_computed=false`; no complete join artifact was published and no
scientific count was charged. This is a bounded workload failure, not evidence
that the server or filesystem is broken. Root has selected no retry, cap
increase or numerical intersection run. The owner is statically reviewing
the workload and any evidence-supported engineering revision before further
source access; failed attempt001 remains preserved.

## 2026-10-04 08:51 UTC — `adaptive_search`: prospective current/history convex blend adopted

Root and an independent scientific reviewer adopt
`primary_history_blend_proposal.md`. The released HW-minus-W primary contrast
has opposite animal signs, motivating one finite temporal mixture of those
two already evaluated pipelines. Use only their complete original five-lambda
prediction banks, with shared lambda and alpha in 0, 0.25, 0.5, 0.75, 1.
Select each held session's one parameter pair from the other fifteen sessions
using the original equal-animal primary criterion; exact ties favor stronger
lambda then less history. Reuse literal endpoints and score interior combined
predictions intentionally, rather than averaging component losses.

Original B/Y/ID/support and component scoring must replay before combination.
No source values or new neural fit are requested. A global exact duplicate
of original selected W or HW charges zero; no interior solution is forced.
At most one distinct complete original-support candidate can update the
incumbent/reset patience under the existing rule. It adds neither a literal
null control nor an automatic alternative-check count. No gate is changed.
After reviewed implementation and focused compute checks, a fixed
Chewie_20150313 artifact pilot may run at one thread, 2 GiB, 60/90 seconds;
remaining bounds await complete pilot resource facts.

## 2026-10-04 08:49 UTC — `source_support`: guide-only intersection qualification launched

Root's independent static review passed `tools/qualify_primary_intersection.py`
and its six focused fixtures. All six passed in 0.070 seconds in verified
job 46469298. Shape-only source stand-ins reject any spike/LFP/XY array read
or slice while exercising the actual guide adapter; additional fixtures cover
all-sixteen qualification, metadata identity changes, finite unit labels,
ordinary subsets, full-support reuse and fixed metadata pilot selection.

Root selected and launched the one-thread guide-only qualification at
60 seconds internal/90 external, with fresh output
`runs/source_support/primary_intersection_metadata_attempt001/`.
Only source identities, event/guide metadata and dataset shapes are accessed;
no neural/XY values, fits or scores. No concurrent science process runs in
the allocation. Qualification adds zero scientific comparisons; numeric
intersection implementation waits for a complete metadata artifact.

## 2026-10-04 08:48 UTC — `adaptive_search`: elapsed-only figure rendered

Root and an independent scientific reviewer passed `tools/plot_elapsed_only.py`.
Root rendered and visually checked `figures/primary_elapsed_only_attempt001.png`
and `.pdf` in job 46469298; the accompanying `.csv` contains 938 rows.
The four panels separate original-B-relative prediction accuracy from trial
identity and display both independently selected and matched-penalty direct
ET-HW-minus-HW contrasts. A prominent decomposition attributes the large
HW-minus-T increment to B-minus-T plus the original small HW-minus-B gain.
All signed session values, animal means and omission ranges are retained;
the latter are not confidence intervals. Count remains 31, patience 3.

## 2026-10-04 08:46 UTC — `adaptive_search`: intersection sensitivity adopted; matched figure rendered

Root and independent review adopt `primary_intersection_sensitivity_proposal.md`,
with an explicit channel-only indexing handoff: preserve original y, labels
and IDs unchanged, use original-order `x[:, :, positions]` without a forced
C-order copy, and require exact original B/Y/D replay. First qualify all
sixteen physical LMP/unit-guide intersections using guide metadata and shape
access only; do not read spike values or apply a unit-quality filter.
Then compare fixed HW on each intersection with eight ordinary physical
subsets of the same count, using I's outer-selected penalty and exact
full/support reuse. Both are nonpromotable support diagnostics, charged
I then mean R if distinct, with zero literal pairing-null increment and no
incumbent reset. The numerical pilot is selected by the prespecified
metadata-only workload proxy, before new values; its bounds are 270/300
seconds, one thread and at least 6 GiB. The metadata implementation and
focused fixtures must pass review before its bounded compute qualification.
Guide membership cannot identify contamination; that limitation remains.

Root and peer passed `tools/plot_matched_budget.py`. Root rendered and
visually checked `figures/primary_matched_budget_attempt001.png` and `.pdf`
in job 46469298; the source `.csv` has 804 rows. Four signed contrast panels
retain every session, animal/equal-animal mean and fixed-prediction omission
range under independent and matched-lambda full-HW anchors. Shared column
scales, retained-target denominators, changed baselines and unmatched
effective capacity are explicit. These are descriptive development results,
not confidence intervals or evidence of practical equivalence. Count is 31.

## 2026-10-04 08:45 UTC — `adaptive_search`: explicit HW capacity-matched surrogate adopted

Root and independent scientific review passed
`primary_capacity_matched_surrogate_proposal.md`. Adopt one fixed eight-draw
whole-trial within-direction training-feature surrogate of original HW.
Freeze true HW preprocessing/weights before the ordinary donor permutations;
share the original fitting Gram/eigensystem and originally selected outer
lambda. Only the feature/target training correspondence changes, giving
exact conditional ridge-spectrum/df matching. Evaluation features remain
correctly paired. This retains direction/time structure and is not a
signal-free synthetic-neural or raw-voltage phase null. Earlier T/S8 remains
unchanged and is not relabeled or recounted.

The distinct complete mean-eight recipe adds at most one literal null
control/valid comparison, without incumbent or patience reset. Existing HW
is an immutable replayed anchor and adds no charge. No new success gate is
created. Implement isolated tools and focused fixtures; after review and
passing compute checks, run the fixed Chewie_20150313 pilot at one numerical
thread, 270/300 seconds in at least 4 GiB. Remaining execution bounds are
unselected until complete pilot resource review. Root retains compute and
scheduler ownership; no source fit has begun at adoption.

## 2026-10-04 08:44 UTC — `adaptive_search`: elapsed-only panel complete; direction/time remains essential

The elapsed-only remainder completed 08:42:30.301057–08:43:09.503441 UTC
in 39.202382 seconds, exit 0, maximum child RSS 414,016 KiB; summary took
2.360664 seconds. Matched remainder resources, now finalized, were
08:40:30.410340–08:42:10.132102 UTC, maximum child RSS 297,496 KiB.
Both ran sequentially in job 46469298. Evidence: the two parent
`remaining_panel.json` files, and
`runs/adaptive_search/primary_elapsed_only_summary_attempt001/primary_elapsed_only.json`
with root's independently reviewed complete-only `compact_evidence.json`.
All sixteen original neural-state and saved-array/scoring replays passed;
original HW was not refitted. Neither exact duplicate condition holds.

One elapsed-only stress recipe is comparison 31; patience advances to 3.
It cannot update the primary incumbent. Literal null controls remain 11,
with 15 conservative alternative checks among 25 post-initial comparisons.
All eight elapsed-only directional/omission checks pass across the two
prespecified screen sets; this does not alter the original practical gates.

ET-HW versus its elapsed-only T baseline has primary gain
0.9725834340168558 and identity gain 0.055726354690169864, choosing lambda
0.01 in all sixteen sessions. Mihili/Chewie primary means are
0.958129423171/0.987037444863, identity 0.051672079589/0.059780629792.
Both metrics are positive in all sixteen sessions and under every
within-animal omission. Saved original HW versus T gains 1.2251724705354385,
but the direction/time behavioral baseline B alone versus T gains
1.2228312160622614. Their difference is exactly the original small HW
primary increment, 0.0023412544731769847; B has zero trial identity.
An R-squared difference may exceed one when the comparator has negative
R-squared. These large T-relative numbers do not isolate neural information
beyond direction and do not replace the registered B-relative endpoint.

ET-HW versus original B is −0.2502477820454057, negative in every session,
both animals and every within-animal omission. Direct ET-HW minus saved HW
prediction gain is −0.25258903651858267 at independently selected penalties
and −0.24547583950704915 at ET-HW's matched penalty, again negative in all
sixteen sessions and every animal omission. Identity differences are positive
0.050111970722826964/0.044205288014104255 in all sixteen sessions and every
animal omission. Thus stronger trial-pairing sensitivity alone does not
establish better full-trajectory prediction. The elapsed-only check passes
its narrow screens while the original primary result remains ineligible.
Baseline decompositions agree within 4.45e-16. No new margin or finalist is
introduced; intersection and explicit capacity-control proposals remain open.

## 2026-10-04 08:43 UTC — `adaptive_search`: three matched-budget diagnostics complete; elapsed-only running

The matched remainder completed all fifteen sessions in 99.721774 seconds
in job 46469298, exit 0; full summary took 6.600196 seconds. The preserved
pilot remains unchanged. Evidence: `runs/adaptive_search/primary_matched_budget_panel_attempt001/remaining_panel.json`,
all sixteen session banks in `primary_matched_budget_attempt001`, and
`primary_matched_budget_summary_attempt001/primary_matched_budget.json`.
Root used the independently reviewed complete-only extractor in compute;
`compact_evidence.json` records all contrasts, omissions, views and reuse.

All three recipes are distinct. C is comparison 28 with patience not started,
T is 29/counter 1, CT is 30/counter 2. No incumbent reset is allowed for
these nonpromotable support diagnostics. Literal null controls remain 11;
conservative alternative checks are 14 of 24 post-initial comparisons.

C (86 electrodes, original trials) has primary/identity increments
0.002213015995635249/0.005271332816995098. T (12 trials per direction,
original electrodes) gives 0.001030769434869120/0.001632066482847893.
CT gives 0.000931989725487741/0.001498317906988750. Every primary margin
fails 0.01; only C passes identity 0.005. All three have positive primary
and identity means in both animals, positive primary signs in 5/6 Mihili
and 9/10 Chewie sessions, and positive identity in all sixteen. Every
within-animal omission and every fixed subset-view animal mean is positive.
C selects lambda 1 throughout; T and CT select 10 throughout. These are
positive small resource sensitivities, not primary eligibility.

Direct reduced-minus-full prediction gains on identical retained targets are
C −0.000128238478, T −0.002254753878, CT −0.002353533588 under independently
selected HW. At condition-matched lambda they are −0.000128238478,
−0.001519080967 and −0.001617860676. Every animal primary/identity mean
and every within-animal omission is negative for all six anchor contrasts.
T/CT worsen their behavioral baseline by 0.001190359409 on average;
Mihili/Chewie contributions are 0.000954896014/0.001425822804. This baseline
change is reported separately and is not an improvement in neural prediction.
C baseline differences are floating-point roundoff, at most 1.73e-17.

There are 384 condition views: 352 unique new fits, 16 full-HW view reuses
and 16 cross-condition duplicate-view reuses. C selected ridge effective df
ranges 41.759–68.626, T 11.449–18.144 and CT 10.600–16.758; equal electrode
count does not establish equal model capacity. Eight fixed views are finite
support sensitivities, not confidence intervals or biological replications.
No noninferiority margin, finalist or primary-gate relaxation is introduced.

At 08:42:30 UTC root launched the elapsed-only remainder sequentially after
the matched panel: four single-thread workers, 720/750-second aggregate and
270/300-second child bounds, preserving the complete pilot. More than
23 minutes remained before the same allocation's fixed expiry. Scientific
elapsed-only scores remain closed until all sixteen sessions complete.

## 2026-10-04 08:40 UTC — `adaptive_search`: matched remainder running in verified allocation

Job 46469298 granted at 08:36:31 UTC; root verified compute shell
`sh03-01n25.int`, four CPUs, 8,192 MiB, Requeue=0/Restarts=0 and fixed
09:06:34 UTC expiry. Five new elapsed-only dispatcher fixtures passed in
0.033 seconds. Updated ASTRA validation passed with the same five existing
optional-format warnings (`validation_20261004_primary_elapsed_only_remaining.json`).

Root launched the matched C/T/CT remainder at the frozen 510/540-second
aggregate bounds, four single-thread workers and unchanged 270/300-second
children. The completed Chewie_20161014 pilot is reused; all fifteen other
session directories and the parent/summary outputs were fresh. No other
scientific workload overlaps. Scientific scores remain closed pending full
completion, so the valid count remains 27. Independent agents prepare the
remaining intersection and explicit capacity-control proposals while root
owns compute execution and scheduler monitoring.

## 2026-10-04 08:34 UTC — `adaptive_search`: both pilots preserved; new bounded allocation queued

ET-HW pilot completed 08:30:46.494475–08:31:46.128530 UTC in 59.634049 seconds:
source 56.241556 seconds, fitting/scoring 3.211058, maximum RSS 428,472 KiB,
1,038 original trials and 86 physical electrodes. Both exact original HW
neural-state and saved-array/scoring replays passed; original HW was not
refitted. Scientific scores remain closed and the pilot adds no charge.

Root selects four single-thread remaining workers at 720 seconds internal/
750 external, unchanged 270/300-second children. Four pilot-duration waves
cost 238.536198 seconds; a full 300-second reserve leaves 181.463802 seconds
of extra variation allowance. Four measured peaks total about 1.635 GiB,
within the selected 4-CPU/8-GiB allocation. The wrapper now reuses the
unchanged qualified fixed-wave dispatcher, with actual CPU/thread guards,
stop-only aggregate signal and child drainage. Its focused wiring checks
and changed ASTRA validation precede execution; scientific code is unchanged.

Root closed idle PTY 8760/job 46466271 at 08:33 UTC. Scheduler closeout at
08:34 reports COMPLETED, 25:24 with four CPUs = 1.693333 allocated core-hours
charged once to EP05, total CPU 12:15.867; compute step 25:13, peak RSS
6,849,984 KiB, exit 0:0. No science process was left behind.

Fresh bounded job 46469298 is queued: normal CPU_SKU:7502, four CPUs, 8 GiB,
30 minutes, attended PTY 16948. Run the preserved-pilot matched remainder
first at its frozen 510/540-second limit, then the elapsed-only remainder
at 720/750 if actual time remains sufficient. Panels execute sequentially;
each panel may use four single-thread workers. Actual compute-shell
verification precedes tests/source. No duplicate pending request or automatic
renewal exists. Valid comparison count remains 27.

## 2026-10-04 08:30 UTC — `adaptive_search`: elapsed-only checks pass; fixed pilot launched

Root and independent static reviews passed the elapsed-only core and runner;
all nine independent fixtures passed in 0.218 seconds in job 46466271.
The checks cover equal-direction elapsed fitting under unequal counts, held
data exclusion, unchanged HW neural state, an independent augmented-ridge
prediction oracle, original whole-trial nulls, direct common-denominator
contrasts, outer exclusion, exact duplicates and saved-anchor replay using
original B rather than the deliberately changed T.

Changed ASTRA validation passed with the same five existing optional-format
warnings (`astra/v0.0.14/validation_20261004_primary_elapsed_only.json`).
The adopted Chewie_20150313 pilot launched about 08:30:44 UTC, one numerical
thread, unchanged 270/300-second bounds. Fresh output:
`runs/adaptive_search/primary_elapsed_only_attempt001/Chewie_20150313`.
No other scientific workload overlaps it. Scientific scores remain closed
pending all-sixteen-session completion. The sequential remainder's bound
remains unselected until complete resource/support/replay facts are inspected.

## 2026-10-04 08:29 UTC — `adaptive_search`: matched pilot complete; finite remainder selected

The matched-budget pilot completed 08:26:25.209260–08:27:03.385141 UTC in
38.175872 seconds, maximum RSS 397,992 KiB: global metadata 0.697641 seconds,
source 8.462756, reduced fitting 27.034700, fitting/scoring 27.518156.
It exercised all 24 distinct condition/view fits, with zero full-HW view
reuses, 740 original trials and 96 physical electrodes. Both global support
qualification and exact saved-reference replay passed. Scientific scores
remain closed and the pilot adds no charge.

From these resource facts, root selects four single-thread workers with
510 seconds internal/540 external, unchanged 270/300-second child bounds.
Four pilot-duration waves total 152.703487 seconds; retaining a full
300-second child reserve leaves 57.296513 seconds of additional variation
allowance. Four measured peak RSS values total about 1.52 GiB, within a
4-CPU/8-GiB allocation. The ASTRA remainder command was frozen and validated
(`validation_20261004_primary_matched_budget_remaining.json`), with the same
five existing optional-format warnings.

The actual shell-time guard at 08:28:42 declined launch because fewer than
570 seconds remained before this allocation's fixed end, including the
30-second cleanup reserve. No remainder process or output attempt started;
this is not a failed scientific trial or an engineering retry. Root will
finish ready elapsed-only checks/pilot here, close the shell, and request a
fresh bounded allocation for the frozen remaining work. There is no automatic
renewal, bound extension or duplicate pending request.

## 2026-10-04 08:26 UTC — `adaptive_search`: matched-budget checks pass; resource pilot launched

All seven independent matched-budget fixtures passed in 0.815 seconds in
verified job 46466271. Root and independent static reviews passed core,
session/summary runner and the wrapper reusing the unchanged qualified
fixed-wave dispatcher. Before testing, root identified the actual released
metadata's string-keyed direction-count mapping; the implementation now
strictly validates and normalizes that format, preserving the artifact.
The fixture covers this format and rejects malformed keys/counts/minima.

Changed ASTRA validation passed with its five existing optional-format
warnings (`astra/v0.0.14/validation_20261004_primary_matched_budget.json`).
The adopted Chewie_20161014 pilot launched about 08:26:24 UTC, one numerical
thread, unchanged 270/300-second bounds, in the existing 4-CPU/8-GiB shell.
No other scientific process overlaps it. Its fresh output is
`runs/adaptive_search/primary_matched_budget_attempt001/Chewie_20161014`.
Only complete resource/support facts may inform the remaining execution
choice; pilot scientific scores remain closed. No new comparison is charged.

## 2026-10-04 08:25 UTC — `adaptive_search`: elapsed-only sensitivity adopted; matched checks running

Root and a separate independent reviewer passed
`primary_elapsed_only_proposal.md`. Adopt one nonpromotable ET-HW sensitivity:
replace the original direction/time behavioral template with its equal-
direction, intercept-only per-bin counterpart, retain the existing common
fitting-residual intercept, and change both the neural residual response and
the behavioral term added back. Original HW feature normalization, history,
repeatability weights, five penalties, full trial/electrode support and target
remain fixed. Saved original HW predictions supply immutable independent-
lambda and same-lambda anchors; direct common-denominator prediction-SSE
contrasts separate model change from baseline change. Standalone elapsed-only
and saved-anchor comparisons add no extra depth. At most one distinct
complete stress comparison is charged; literal null-control count remains
eleven, and this diagnostic cannot update the primary incumbent or reset
patience. Its eventual ordinal depends on actual completion order.

The pilot is Chewie_20150313, one CPU/4 GiB, 270/300 seconds; the remaining
bound is not selected until pilot resource review. Only implementation and
focused checks are authorized now. Matched C/T/CT implementation independently
passed review, including the repaired actual metadata mapping format, and its
seven focused fixtures are running in root's verified allocation 46466271.
Elapsed-only and matched tools have separate owners; shared ASTRA edits are
serialized. Root retains exclusive source execution and scheduler control.

## 2026-10-04 08:19 UTC — `adaptive_search`: matched trial/electrode budget stress adopted

Root and independent scientific review passed
`primary_matched_budget_proposal.md`. Adopt the fixed C, T, CT diagnostics
at 86 physical electrodes and 12 trials per direction, with eight paired
metadata-only subset views and the unchanged HW recipe. C changes only
electrode count, T only trial count, and CT both. Refit every genuine reduced
B/H/W state solely on its retained seven-direction fitting data. Each view
uses its own target denominator and retained-ID exact wrong-pair mean;
average normalized losses before the original outer session selection.

Preserved full-HW predictions are evaluated on exactly the same retained
targets, at original and condition-matched lambda. Direct prediction-SSE
contrasts and separate baseline-SSE changes prevent a changed behavioral
template from masquerading as a pure resource effect. Exact full-support and
cross-condition reuse is based on complete ID support, not close scores.
The three diagnostics cannot promote a model or update the candidate
incumbent. If all are distinct, process C as comparison 28/patience not
started, T as 29/counter 1, CT as 30/counter 2. Eleven existing literal
null-control recipes remain eleven; three stress diagnostics would make
fourteen conservative alternative checks among twenty-four post-initial
comparisons. Actual transitions await complete scientific evidence.

The metadata-selected resource pilot is Chewie_20161014, the largest
96-electrode session (740 trials), which exercises all three distinct fitting
paths. This prospectively replaces the initially drafted 86-electrode pilot,
whose full-support C reuse and T=CT identity would under-measure workload.
Limits remain one CPU, 6 GiB, 270 seconds internal/300 external. The current
4-CPU, 8 GiB allocation may run this single-thread pilot after focused
independent fixtures and changed ASTRA validation pass. Implementation is
authorized; no new source fit has launched at adoption. Root selects any
remaining finite execution bound from complete pilot resource facts before
opening pilot scientific scores. All source, budget and claim limits remain.

## 2026-10-04 08:17 UTC — `adaptive_search`: all-band figure and matched-budget metadata

Root and peer static review passed `tools/plot_all_band.py`. Root rendered
and visually checked `figures/primary_all_band_attempt001.png` and `.pdf`
in job 46466271; the accompanying `.csv` has 938 rows. The four panels
show candidate/control practical margins plus all five primary and identity
contrasts, all session effects, animal/equal-animal means and fixed-prediction
omissions. Signed axes and the positive matched-lambda LMP overall contrast
are preserved. Omission ranges are not confidence intervals, and equal
width/trace/lambda is not labeled equal effective capacity.

Root also extracted only existing support metadata in compute, validating
the original sixteen completed scalar records and frozen trial table. The
durable result is `runs/source_support/matched_budget_metadata_attempt001/matched_budget_metadata.json`.
All sessions support 86 physical LMP electrodes and at least 12 trials in
each direction. Chewie_20150313/20150319 set the electrode minimum; all
other sessions have 96. Chewie_20150629 direction 3 sets the trial minimum.
No neural/spike source values or new scientific scores were read. These
metadata support the prospective separate channel-only, trial-only and joint
budget sensitivity; the proposal remains under independent review.

## 2026-10-04 08:15 UTC — `adaptive_search`: all-band panel complete; no added-band improvement

The fifteen remaining sessions and full summary completed in job 46466271
from 08:08:36.995408 to 08:13:26.407556 UTC, 289.412163 seconds, exit 0.
Maximum child RSS was 854,568 KiB; the scheduler reported a step peak of
5,985,036 KiB during parallel execution. The preserved pilot took 84.233666
seconds separately; summary generation took 3.283270 seconds. Evidence:
`runs/adaptive_search/primary_all_band_panel_attempt001/remaining_panel.json`,
all sixteen banks under `primary_all_band_control_attempt001`, and
`primary_all_band_summary_attempt001/primary_all_band.json`. Root opened
scientific outcomes only after the complete panel, using the independently
reviewed `tools/extract_primary_all_band.py` in compute; its saved
`compact_evidence.json` contains the full contrasts, seeds and omissions.

Both recipes are distinct: 27 valid comparisons now include 11 controls
among 21 post-initial comparisons. All-band A has primary increment
0.0006719900156145613 and identity increment 0.0013802590100496538. Both
practical margins fail. Mihili/Chewie primary means are
0.00001844126312027549/0.0013255387681088472, identity
0.0008694670497996835/0.001891050970299624; primary signs are positive in
5/6 and 9/10 sessions. Every outer selector chooses lambda 10. Overall
primary omissions remain positive, but Mihili reverses when omitting
20140218, 20140303, 20140306 or 20140307. Its omission range is
−0.000096238830 to +0.000348147819. Both animal identity omission ranges
remain positive. No primary-eligible candidate is established.

Mean added-block training-label control C has primary
0.0005859210813388123 and identity 0.0012666933465901062. A−C is
+0.000086068934275749 primary and +0.000113565663459548 identity.
Animal primary contrasts are Mihili +0.000092076865 and Chewie
+0.000080061003, but only 2/6 Mihili and 7/10 Chewie session contrasts are
positive. The prospective alignment screen fails. Omitting Mihili_20140306
reverses Mihili to −0.000020990045; overall omissions stay positive.
Identity contrasts and their animal omission ranges are positive. Seed 7
reverses the overall primary contrast; seeds 3 and 7 reverse Mihili, seed 2
reverses Chewie. Mihili identity also reverses for seed 3. These are finite
control spreads, not independent replication or uncertainty intervals.

A−J primary is −0.001581433784 under independently selected penalties and
−0.000477384136 at A's matched lambda. Both animal primary means and every
animal primary omission remain negative for both anchors. Independent
identity is −0.003826521860; matched identity is +0.000045202298, with
Chewie reversing after omitting 20161005 or 20161006. A−LMP independent
primary/identity are −0.001265394730/−0.002016065536, both animals negative
under every omission. Matched-lambda A−LMP is +0.000042530223 primary and
+0.000605906804 identity, but primary is negative in Mihili
(−0.000315528230) and positive in Chewie (+0.000400588675), with those
animal signs unchanged under every omission. Overall matched-LMP primary
reverses after omitting Mihili_20140306 or Chewie_20161005/20161006.
All four anchor screens fail; all five complementarity screens fail.

Across 128 candidate and 1,024 control fits, added fitting energy spans
516–576, with no zero-energy or structural duplicate fit. Candidate/control
Gram traces are identically 774–864 and numerical ranks 773–863; effective
degrees of freedom are 41.214529–66.948601 for A and 41.601815–67.204754 for
controls. Control/candidate df ratios are 1.001999–1.010917. Equal width,
trace and penalty do not equalize effective capacity. The comparison does
not establish that individual frequency bands are biologically useless.

Process candidate comparison 26, then control 27 under the adopted
prospective patience rule. HW remains the best comparable primary score,
0.0023412544731769847; patience is NOT STARTED through comparison 27.
Required development work continues. The next proposal addresses matched
trial and physical-electrode budgets; no changed recipe or source fit is
authorized merely by this result. A completed-result figure is being prepared.

## 2026-10-04 08:08 UTC — `adaptive_search`: all-band remainder launched; remaining coverage reconciled

Scheduler closeout observed at 08:09 UTC: the previous shared job 46464685
completed in 18:48 with one allocated CPU, 0.313333 allocated core-hours
counted once across EP05/EP08. Total CPU was 6:20.307; its compute step ran
18:41, peak RSS 1,670,284 KiB, exit 0:0. This cost includes both episodes'
completed work and validation; EP08 must not charge the allocation again.

Job 46466271 was granted on sh03-01n25.int. Root verified actual compute
hostname, SLURM_JOB_ID, four CPUs and 8192 MiB in PTY 8760, with Requeue=0
and Restarts=0. The frozen fifteen-session remainder launched at about
08:08:31 UTC, four single-thread workers, unchanged 1260/1290-second aggregate
and 270/300-second child bounds. No other compute workload overlaps its
workers. The preserved pilot is reused; outcomes remain closed until the
full sixteen-session summary completes.

The bounded static coverage audit identifies four substantive open tasks:
elapsed-time-only behavioral comparison (explicitly left open in
`primary_reference_proposal.md`); matched trial and physical electrode counts;
primary same-electrode/unit-intersection sensitivity, distinct from any
spike-population latent analysis; and an explicitly permitted capacity-matched
random-feature or surrogate comparison. Common feature width or penalty does
not certify matched effective capacity. Existing whole-trial pairing controls
are not retroactively relabeled as the separate random/surrogate check.

Original registered LMP, grouped low-frequency and grouped high-frequency
blocks are complete; all-band coverage awaits the current complete panel.
The eight stored bands do not automatically impose eight single-band recipes.
Further band attribution would need a corresponding finalist claim. GOAL
232–239 keeps applicable development coverage required; passing positive-result
gates is required for promotion, not for reporting a valid negative. The
conditional spike-population explanation remains unactivated. No successor
recipe is adopted before the complete current result is available.

## 2026-10-04 08:06 UTC — `adaptive_search`: four-worker checks pass; bounded allocation queued

Root and independent static review passed the all-band fixed-wave dispatcher
and its allocation/failure wiring. All nine focused dispatch fixtures passed
in 0.020 seconds in verified job 46464685. The scientific implementation and
eight previously passing scientific fixtures are unchanged. Changed ASTRA
validation passed, retaining only its five existing optional-format warnings:
`astra/v0.0.14/validation_20261004_primary_all_band_remaining.json`.

Root closed attended PTY 40459/job 46464685 after its completed EP08 report,
EP05 pilot and checks. A new bounded four-CPU, 8 GiB, 30-minute allocation,
job 46466271, is queued on normal with the previously discovered
CPU_SKU:7502 feature. No duplicate request or automatic renewal exists.
The remaining all-band command is frozen at four single-thread workers,
1260/1290 seconds aggregate and 270/300 seconds per child. Actual compute
hostname/allocation verification precedes launch. Pilot scientific scores
remain closed; complete count remains 25 until the entire panel succeeds.

## 2026-10-04 07:58 UTC — `adaptive_search`: all-band pilot complete; four-worker remainder selected from resource facts

Pilot001 completed07:56:15.419051–07:57:39.652726UTC in46464685:
84.233666seconds total,19.524004source and64.229375fit/score,
maximum RSS847024KiB. All1038 original trials,86 physical electrodes and
all nine guide pairs are qualified before values; selected logical source
volume is96,658,560bytes. Original LMP target/baseline/trial replay and J
support/zero-SD/target replay both pass. Scientific banks remain unopened.

Root selects the remaining fifteen sessions with four single-thread workers
in a fresh4CPU/8GiB/30minute normal7502 allocation. Internal/external bounds
are1260/1290seconds, preserving270/300seconds per child and a full300second
reserve before each launch. Four pilot-length waves take336.934663seconds;
one full child reserve leaves623.065337seconds variation/summary headroom.
Four pilot memory peaks total about3.23GiB, with additional parent/variation
headroom within8GiB. No penalty, representation, tolerance or scientific
support changes. The existing one-CPU allocation cannot run four workers.

Owner will adapt the existing sequential wrapper to fixed bounded parallel
dispatch, with independent review and focused checks in the current compute
session. Stop new launches on failure, drain/reap active children, preserve
completed banks and publish no partial scientific summary. Reuse the pilot;
the other fifteen paths, parent and summary attempt001 stay fresh. Root will
close the current idle session before requesting the new allocation; no
duplicate or automatic renewal. Remaining source work has not launched.

## 2026-10-04 07:56 UTC — `adaptive_search`: conservative prospective patience rule; all-band pilot selected

Root and independent review adopted
`primary_patience_interpretation.md` before any all-band source fit. The
inherited contract does not specify a material-gain number or counter start.
The documented execution choice preserves28–64 and all ceilings, starts
counting only after28, and recognizes every strict same-support candidate
primary improvement as progress. It is not a retrospective counter or an
assertion that the original text required40. Candidate precedes mean control
in paired count order. The full-precision current best is HW
0.0023412544731769847; patience is not started at25. Required stress tests,
including matched trial/electrode support, remain separate obligations.

All eight all-band fixtures passed in0.444seconds in46464685. Changed ASTRA
validation passed with the same five existing optional-format warnings.
Root selects the complete fixed Chewie_20150313 source pilot, one CPU/6GiB,
270seconds internal/300external, fresh
`runs/adaptive_search/primary_all_band_control_attempt001/Chewie_20150313`.
The parent ends08:17:21, leaving the full pilot reserve. Only resource/support
and replay fields will be inspected before choosing a finite remainder bound.
No scientific score, extra comparison or remaining-run bound is claimed yet.

## 2026-10-04 07:51 UTC — `adaptive_search`: all-band implementation frozen; checks selected

Root and independent static reviews passed the new reader, core, session
runner and sequential wrapper. Owner peer review passed all eight independent
fixtures, covering exact nine-band support/layout, qualification before value
reads, fitting-only normalization, J-prefix nesting, direct ridge agreement,
all72 own-Gram true-target arms, unchanged evaluation, mean-SSE controls,
outer-session exclusion, anchor contrasts and exact duplicate accounting.
ASTRA now records the fixed270/300second one-CPU/6GiB pilot; the remainder
remains unselected. Root selects these fixtures and changed ASTRA validation
in46464685 after the active EP08 panel returns. Only then may the fixed fresh
Chewie_20150313 pilot start. No source run or count increment is claimed yet.

## 2026-10-04 07:47 UTC — `adaptive_search`: next shared allocation verified

46464685 started07:47:16UTC on sh03-01n68 and ends08:17:21UTC,
normal CPU_SKU:7502, one CPU/6GiB, no GPU. PTY40459 independently reports
sh03-01n68.int and SLURM_JOB_ID46464685. Requeue=0 and Restarts=0 are
verified. Root loaded the existing qualified Python/numerical modules and
set all numerical threads to one. EP08's frozen robustness panel executes
first; EP05 all-band source work remains pending its implementation checks.

## 2026-10-04 07:45 UTC — `adaptive_search`: shared allocation accounted; next request queued

Scheduler confirms46463321 COMPLETED,24:44 elapsed, one CPU,0.412222
allocated core-hours counted once across the shared EP05/EP08 work.
Total CPU was6:00.311; step elapsed24:34, peakRSS2130796KiB, exit0:0.
This parent success does not overwrite any prior failed scientific attempt.
New bounded allocation46464685/PTY40459 is queued with the selected
1CPU/6GiB/30minute normal7502 request. No scientific process has started in
it. The07:45scheduler check also shows EP12's existing45736217 still running
with advancing CPU and its45736218 successor dependency-pending; prohibited
scratch-log reads were not retried.

## 2026-10-04 07:45 UTC — `adaptive_search`: full-target figure complete; shared session closed

The report-only full-target PNG/PDF/402-row CSV rendered in46463321.
Root visually inspected the full signed scales and all session/animal markers;
the primary failure and positive identity readout are explicit. No refits or
source reads were added. Root exited idle PTY27650 at07:45UTC after EP08
metadata-role preparation and this render. Scheduler accounting follows at
the next eligible check; no job completion is inferred solely from shell exit.

Select a fresh attended normal CPU_SKU:7502 allocation, one CPU,6GiB,30minutes,
for EP08's adopted600/630second full robustness panel and aggregated EP05
all-band validation/resource pilot when implementation is ready. Resource
discovery at07:42 confirmed the partition and feature. Current completed
work is preserved; no duplicate pending request or automatic renewal is used.

## 2026-10-04 07:44 UTC — `adaptive_search`: all-band complementarity adopted; full-target figure selected

Root and independent written scientific reviews passed
`primary_all_band_complementarity_proposal.md` after correcting its J summary
path to the actual preserved `primary_lmp_high.json`. Root adopts the fixed
LMP/high prefix plus six0.5–100Hz bands, with eight fitting-only whole-added-
block label controls, unchanged evaluation inputs and each control's own full
Gram. Original physical support cannot shrink. The fixed Chewie_20150313
pilot is270/300seconds, one CPU within6GiB; remaining bounds await resource
facts. This fills missing all-band coverage without claiming passed finalist
controls or changing the28–64 stopping contract. Implementation/independent
fixtures are delegated; no new source fit has launched.

Root also reviewed `tools/plot_full_target_sensitivity.py` and selects its
report-only render in46463321 after EP08 preparation. It compares preserved
R with completed F and C on full signed linear scales, all16 sessions, both
animals, equal-animal means and fixed-prediction omission ranges. Fresh PNG,
PDF and402-row CSV use `figures/primary_full_target_sensitivity_attempt001`.
No source values or models are added.

## 2026-10-04 07:41 UTC — `adaptive_search`: full-target sensitivity fails baseline robustness;25 valid comparisons

The remaining panel completed07:39:26.428446–07:40:44.398839UTC,
77.970395seconds, maximum childRSS187632KiB. Pilot plus remainder consumed
90.039432seconds. Complete evidence is
`runs/adaptive_search/primary_full_target_sensitivity_summary_attempt001/primary_full_target_sensitivity.json`;
parent resources are in `primary_full_target_sensitivity_panel_attempt001/remaining_panel.json`.
Both recipes are distinct:25 valid comparisons,10 controls among19
post-initial comparisons. There are no constant-B or equal-control duplicates.

Direct full-target F has primary increment−0.322120966841168 and identity
increment0.050607777468613. Primary means are Mihili−0.367114890536330 and
Chewie−0.277127043146007, with0/6 and0/10 positive sessions; every animal
omission remains negative. Identity means0.043234392382100/0.057981162555125
remain positive under every omission. All16 sessions select lambda0.001.
The mean-eight training-label control scores−0.347056778372580 primary and
0.042565551876062 identity. Both sensitivity recipes fail the primary margin.

F-minus-control is+0.024935811531411 primary/+0.008042225592550 identity:
Mihili+0.022255648696515/+0.006568292072208 and
Chewie+0.027615974366308/+0.009516159112892. All16 primary contrasts,
both animal metrics under every omission, and both animal metrics under all
eight seeds are positive. The training-pairing directional screen passes.
Seed overall primary contrasts span0.023140393651733–0.027613273874991,
identity0.007618937839035–0.008212860621963; this is finite control spread,
not biological replication or a confidence interval.

F-minus-original residual-LMP is−0.324058351586389 primary/
+0.047211452922814 identity under independent penalties, and
−0.319886571506963/+0.041507434877237 at F's matched lambda.
Both primary animal contrasts and all omissions remain negative, with no
positive primary session under either anchor comparison. Both anchor screens
fail. Matching lambda and the X Gram does not match the information supplied
by the separate direction/time baseline. Correct training correspondence helps
the direct model, but it does not establish robustness beyond residual fitting.
The identity gain cannot rescue poor full-trajectory prediction or the failed
primary objective. These sensitivities cannot replace the registered target.

Diagnostics cover128 folds: shared candidate/control trace86–96, numerical
rank85–95, selected ridge df84.423812–94.588428. Fitting native XY means range
−3.278520–2.749682 and−4.252647–2.986793. No fold has B=mu or any control's
cross-product exactly equal to the candidate's. No primary-eligible model is
established; required development work continues. A separate all-band proposal
is being prepared under the existing representation menu, without activation.

## 2026-10-04 07:41 UTC — `adaptive_search`: full-target panel complete; report extraction selected

All fifteen remaining sessions and the complete summary returned successfully
in46463321. The remaining panel consumed77.970395seconds, with2.712771seconds
for summary construction. Both distinct complete sensitivity recipes add two
valid comparisons,23→25. Root reviewed the summary-only extractor and selects
its compute-side read of both panels, all controls/anchors, omission summaries
and capacity diagnostics. This completion does not promote a full-target
sensitivity to the primary objective; scientific interpretation follows.

## 2026-10-04 07:39 UTC — `adaptive_search`: remaining full-target panel dispatched

Changed ASTRA validation passed with the existing five optional-format warnings.
Root dispatched the selected600/630second sequential remainder in46463321
at07:39UTC, with the full reserve before the07:50:17allocation end.
The completed pilot is retained; all fifteen other paths and the summary are
fresh. No source or scoring recipe changed after pilot resource inspection.

## 2026-10-04 07:38 UTC — `adaptive_search`: full-target pilot complete; remaining bound selected before scores

Pilot001 completed07:37:45.405158–07:37:57.474201UTC in46463321:
12.069037seconds total,10.060996source,0.665461fit/score and172292KiB
maximum RSS. All1038 original complete trials and86 physical electrodes are
preserved, with exact original baseline/target/trial replay. Only these
resource/support fields were opened; pilot scientific banks remain unread.

Select all fifteen remaining sessions sequentially, retaining the complete
pilot, at600seconds internal/630external with unchanged270/300child limits.
Fifteen pilot runtimes predict181.035554seconds; adding one full300second
child reserve leaves118.964446seconds variation/summary headroom inside600.
The existing1CPU/6GiB parent ends07:50:17, leaving the full external reserve
plus cleanup at launch. Fresh panel and summary paths use
`primary_full_target_sensitivity_panel_attempt001` and
`primary_full_target_sensitivity_summary_attempt001`. Freeze/validate changed
ASTRA before dispatch. No resource limit or scientific recipe is relaxed.

## 2026-10-04 07:37 UTC — `adaptive_search`: full-target checks pass; fixed pilot selected

The first fixture run passed7/8 in0.533seconds. Its constant-baseline test
incorrectly expected trial-weighted least squares on symmetric nonzero targets
to produce literal zero coefficients. Independent diagnosis and owner review
agreed on a fixture-only repair: test literal B=mu with zero targets at the
preprocessing stage, and retain the defined nonzero-target prediction-equivalence
check at its original tolerance. No scientific implementation, duplicate rule
or tolerance changed. All eight checks then passed in0.537seconds.
ASTRA validation passed with the same five existing optional-format warnings.

Select and dispatch the fixed complete Chewie_20150313 pilot in46463321:
one CPU, existing6GiB parent,270second internal/300second external bound.
The new path is
`runs/adaptive_search/primary_full_target_sensitivity_control_attempt001/Chewie_20150313`.
Current allocation has more than twelve minutes left; full remaining-panel
reserve will be checked separately after observing resource-only pilot fields.
No scientific score or count increase is claimed by this launch.

## 2026-10-04 07:36 UTC — `adaptive_search`: full-target implementation checks selected

Root checked the direct full-target implementation against the original flat
LMP normalization and baseline code; independent static review passed the core,
session runner and sequential wrapper. The owner also peer-reviewed all eight
independent fixtures. Select those focused fixtures and the changed ASTRA
validation in the existing verified sh03-01n68 allocation46463321, one CPU
and numerical threads one. No source process starts until these checks pass.
The fixed Chewie_20150313 pilot remains270/300seconds at fresh
`primary_full_target_sensitivity_control_attempt001/Chewie_20150313`; only
resource/support/replay fields will inform the remaining-panel bound.

## 2026-10-04 07:29 UTC — `adaptive_search`: H/W/HW figure complete

Report-only plotting completed in46463321. Root visually inspected the
fresh `figures/primary_history_repeatability_weight_attempt002.png` and shared
it with the user; PDF and402-row source CSV are saved beside it. The figure
shows all sixteen sessions per recipe, both animal means, equal-animal means
and fixed-prediction omission ranges. Equal-animal primary and identity
thresholds are labeled correctly; the HW primary failure is prominent.
No source data, models or additional comparisons were added.

## 2026-10-04 07:28 UTC — `adaptive_search`: completed H/W/HW figure selected

Root statically reviewed `tools/plot_history_repeatability_weight.py` and
selects its report-only render in46463321. It reads only the complete H,W,HW
summaries, displaying all original sessions, animal/equal-animal means and
fixed-prediction omission ranges, with practical thresholds explicitly
applying to equal-animal means. Fresh output prefix is
`figures/primary_history_repeatability_weight_attempt002`; no source reads,
new fits, confidence intervals or promotion claims are added.

## 2026-10-04 07:27 UTC — `adaptive_search`: required current-LMP full-target sensitivity adopted

Root and independent written reviews passed
`primary_full_target_sensitivity_proposal.md`; its native-X wording was
clarified before adoption. Root adopts the exact current-LMP sensitivity:
original flat fitting X normalization, true fitting-global native XY mean mu,
direct `mu+Z beta` prediction without adding the direction/time baseline,
and eight within-fitting-direction whole15x2 target-label controls sharing
true mu and Gram but using their own cross-products. The original baseline
remains the scoring comparator. Residual-LMP anchors use independent and
candidate-matched lambda; no new noninferiority margin is invented.

This is the existing GOAL-required residualization sensitivity, not a
promotable replacement target. It does not automatically cover later
finalists with different features. Exact duplicate rules, native XY units,
original supports, other-session lambda selection and mean-SSE controls
remain fixed. Owner prepares new modules/ASTRA and an independent worker
prepares fixtures; no source process is launched yet. Select the fixed
complete Chewie_20150313 pilot270/300seconds after checks and actual reserve
verification. Count remains23; only complete distinct recipes can add up to
two. Root still owns execution/log/navigation and shared budgets.

## 2026-10-04 07:25 UTC — `adaptive_search`: H4+W complete;23 valid comparisons, primary threshold still fails

All fifteen remaining sessions completed in46463321, followed by the complete
summary:07:21:09.136376–07:24:00.973347UTC,171.836965seconds,
maximum childRSS426780KiB. The successful pilot plus remaining executions
sum195.177617seconds; the preserved failed pilot separately consumed
23.075923seconds. No completed anchor was refit. Evidence:
`runs/adaptive_search/primary_history_repeatability_weight_summary_attempt002/primary_history_repeatability_weight.json`
and `primary_history_repeatability_weight_panel_attempt002/remaining_panel.json`.
Root ran the reviewed complete-summary extractor and inspected both anchors,
all seed/session-omission summaries and fit diagnostics. Both recipes are
distinct: count21→23, controls9 of17 post-initial comparisons; no identity,
no-past or all-control-equals-candidate duplicate occurred.

HW primary increment is0.002341254473177, below0.01; identity increment
0.005614383967343 exceeds0.005. Primary means are Mihili0.001019842402803
and Chewie0.003662666543551, identity0.004613634813059/0.006615133121627;
5/6 and9/10 sessions have positive primary gain. Both metrics remain positive
in both animals under every fixed-prediction omission. All16 sessions select
lambda1. **No primary-eligible candidate is established.**

The mean-eight weight-label control has primary0.001996493175487 and
identity0.005421556165177. HW-minus-control is
0.000344761297690/0.000192827802166; both animal means and every primary
session are positive. Both metrics stay positive in each animal under all
omissions and all eight individual seeds. Seed overall primary contrasts
range0.000196541972666–0.000449959018960, identity
0.000107559359419–0.000253903044867. The weight-label screen passes.

HW-minus-H is0.000302472384485 primary/0.000115768229194 identity under
both independent and HW-matched lambda selection. Both animal means,
every animal omission and6/6 Mihili plus9/10 Chewie primary sessions are
positive; both H-anchor screens pass. HW-minus-W is
0.000166403645288/0.002109978497139 overall, but primary Mihili is
−0.000228909235730 (negative under every omission), versus Chewie
+0.000561716526307. Only2/6 Mihili and8/10 Chewie primary contrasts are
positive. Both W-anchor screens fail. Thus weighting helps the history
recipe, while adding history to W does not improve primary prediction in
both animals. Identity improvement alone cannot rescue that result.

Diagnostics cover128 candidate and1024 control folds,12128 repeated physical
fold/electrode entries. All four lags have positive fitted energy, total q≈4
per physical electrode; no current-zero/history-active case occurs in these
data. Current-only repeatability diagnostics exactly replay W:34 negative P,
no zero T, r0–0.814484784. Applied weights0.438263801–1.494654854,
trace344–384 with maximum discrepancy1.14e-13, numerical ranks340–380.
Selected-lambda df is46.124–73.894 for HW and47.163–75.940 for controls;
control/HW df ratio0.997478–1.064961. Matched trace or lambda does not mean
matched capacity. Source, temporal, native-target and exploratory claim
boundaries remain unchanged. The required full-target fitting sensitivity
is drafted separately and remains unadopted pending scientific review.

## 2026-10-04 07:20 UTC — `adaptive_search`: remaining allocation granted and verified

Slurm46463321 started07:20:12UTC and ends07:50:17UTC, normal
CPU_SKU:7502,1CPU/6GiB, no GPU. PTY27650 reports
`hostname=sh03-01n68.int` and `SLURM_JOB_ID=46463321`. Root explicitly
set and verified Requeue=0, Restarts=0. Qualified Python3.12.1,
NumPy1.26.3,scikit-learn1.5.1 and h5py3.10 modules are loaded with
all numerical threads1. Changed remaining ASTRA validation precedes the
frozen960/990second panel launch, with over29minutes of parent reserve.

Previous shared allocation46460577 is terminal COMPLETED, elapsed25:45,
1CPU (0.429167 allocated core-hours once for the shared session), actual CPU
4:50.771, stepRSS1410880KiB, exit0:0. This shell status does not change the
preserved failed first HW pilot; corrected pilot002 passed separately.

## 2026-10-04 07:18:48 UTC — `adaptive_search`: remaining-work allocation requested

Root exited PTY69705/job46460577, rediscovered available partition/hardware
limits with `sh_part`/`sh_node_feat`, and requested one attended
`srun --partition=normal --cpus-per-task=1 --mem=6G --time=00:30:00
--constraint=CPU_SKU:7502 --job-name=ep05-ep08-remaining --pty bash --noprofile --norc`.
Slurm returned46463321 queued; PTY27650 owns the pending request. No duplicate
request or new scientific process was launched. ASTRA now records the frozen
960/990second remaining-panel command selecting pilot/banks002, fresh
`primary_history_repeatability_weight_summary_attempt002` and
`primary_history_repeatability_weight_panel_attempt002`. Root will verify the
actual shell and allocation on grant, disable requeue, validate the changed
ASTRA object and launch only with the complete selected reserve.

## 2026-10-04 07:18 UTC — `adaptive_search`: corrected H4+W pilot complete; remaining resources selected

Pilot002 completed07:16:56.468621–07:17:19.809280UTC in23.340652seconds,
source10.116058seconds, fit/score12.911586seconds, RSS428072KiB,
original1038 trials/86 electrodes. All three exact source/baseline/current-W
state replay checks passed. The earlier dispatch timestamp described the
command dispatch; these are the observed process start/finish times.
No partial scientific scores were opened and pilot charge remains zero.

Before outcome release, root selects the remaining fifteen original sessions
sequentially,960seconds internal/990external, unchanged270/300second child
limits and full300second reserve before each child. Resource rationale:
15 times the23.341second complete pilot is350.11seconds, plus300seconds
complete-child reserve and309.89seconds variation/final-report allowance.
Use completed pilot002 once, new remaining banks under the same attempt002,
fresh summary002 and parent002; preserve failed001.

The current46460577 allocation ends07:22:53UTC and cannot provide990seconds
reserve. Root closes its now-idle attended shell and selects a new bounded
1CPU/6GiB/30minute normal CPU allocation for this remaining panel and the
reviewed EP08 raw-initializer comparison (shared sequential execution).
This is a manual request for specified work, with no renewal or supervisor.
ASTRA will record the selected full command before launch. Count remains21.

## 2026-10-04 07:16:49 UTC — `adaptive_search`: corrected H4+W pilot dispatched

All ten fixtures passed in0.526seconds, including both independent
reader-layout regressions. Changed ASTRA schema and semantics passed, with
the same five existing optional-format warnings; validation is saved as
`validation_20261004_primary_history_repeatability_weight_layout_repair.json`.
Root dispatched fresh pilot attempt002 in46460577, retaining the full
300second external reserve before the allocation ends07:22:53UTC.
Exact replay and270second internal limits remain unchanged. Resource-only
inspection follows completion; count stays21 until a complete full panel.

## 2026-10-04 07:16 UTC — `adaptive_search`: exact-layout repair diagnosed and adopted

In46460577, the independent source-free reader-layout diagnostic passed
in0.038seconds for float32/float64 source-like arrays: the unforced current
slice exactly reproduces the original W mean, SD and all repeatability
statistics; forcing C-order changes the mean. The actual new fitter's
regression test failed as expected in0.055seconds (5/7 adversarial means
differ, maximum relative difference8.63e-16). Root adopts removal of that
C-order coercion only, preserving the reader-layout view, original H
decoder state and every exact replay condition. This restores the adopted
numerical path; no tolerance, model or resource limit changes.

After all ten focused fixtures and changed ASTRA pass, select one corrected
fixed Chewie_20150313 pilot under fresh
`primary_history_repeatability_weight_control_attempt002`, still270/300seconds.
Attempt001 remains immutable and uncharged. No full-panel bound has been
selected, and partial scientific scores remain closed.

## 2026-10-04 07:14 UTC — `adaptive_search`: H4+W pilot stopped at exact current-state replay

Attempt001 failed after fitting at
`check_weight_reference_after_fit`: original W `physical_mean` did not replay
exactly. Start07:13:35.342117UTC, finish07:13:58.418045UTC,
elapsed23.075923seconds, source10.179931seconds, fit/score12.848738seconds,
RSS442264KiB, original1038 trials/86 electrodes. The failure JSON is preserved;
no banks or predictions were published, no scientific scores were opened and
no comparison was charged (complete count21).

Root, owner and independent reviewer identify the explicit C-order copy of
the current history slice as a likely numerical-path defect. Both qualified
readers stack transposed per-trial arrays with time-fast inner storage; the
new copy changes reduction traversal in the original W mean/SD/statistics.
This is not yet an empirically established diagnosis. A source-free fixture
will compare both original reader layouts and the proposed unchanged-layout
slice inside the current allocation. No replay tolerance change, source retry
or repair is selected before that diagnostic result.

## 2026-10-04 07:13:35 UTC — `adaptive_search`: H4+W pilot dispatched

All eight focused fixtures passed in0.381seconds. Changed ASTRA schema and
semantics passed (`validation_20261004_primary_history_repeatability_weight.json`),
with five existing optional-format warnings. Root dispatched the frozen
Chewie_20150313 command from the ASTRA pilot recipe to verified
46460577/sh03-01n68, fresh
`runs/adaptive_search/primary_history_repeatability_weight_control_attempt001/Chewie_20150313`.
The270/300second limits and scientific-score withholding remain in force.
No complete panel, comparison charge or remaining launch is inferred.

## 2026-10-04 07:13 UTC — `adaptive_search`: H4+W pilot checks selected

Root and independent static reviews passed the core, session runner and
sequential panel wrapper. Root selects eight focused scientific fixtures in
verified46460577/sh03-01n68, followed by changed ASTRA validation and the
fixed complete Chewie_20150313 pilot (270seconds internal/300external).
The existing attended allocation has1CPU/6GiB and ends07:22:53UTC;
threads remain1. Pilot scientific scores stay withheld for resource selection.
No remaining-panel bound or launch is selected yet. Exact pilot paths and
command are recorded in the ASTRA program before execution.

## 2026-10-04 07:10 UTC — `adaptive_search`: complete repeatability result figure

Root rendered `tools/plot_repeatability_weight.py` successfully in verified
46460577/sh03-01n68 and visually checked the PNG. Fresh artifacts
`figures/primary_repeatability_weight_attempt001.{png,pdf,csv}` contain all
sixteen session contrasts against full LMP and mean-eight weight controls,
both metrics, animal/equal-animal means and fixed-prediction omission ranges
(not confidence intervals); CSV has268 rows. Both failed practical gates are
prominent. No source reads or fits were added. Figure shared with the user.

## 2026-10-04 07:08 UTC — `adaptive_search`: fixed history plus repeatability weighting adopted

Root and the independent reviewer passed the written
`primary_history_repeatability_weight_proposal.md`; root adopts that exact
candidate/control pair. The current-bin repeatability state remains W's
original state, while the four-lag decoder preserves H's original separate
normalization and centering. One physical weight is repeated across lags and
normalized by the full history Gram trace; all eight original W maps use
their own history Grams. Independent and candidate-matched H/W anchors,
literal structural duplication accounting, original thresholds and source
windows remain fixed. Implementation and focused checks are pending; no
new scientific fit or comparison charge has occurred. Root owns Slurm and
execution records; the EP05 worker owns new implementation and ASTRA, with
independent fixtures/review from the review worker.

The completed W plotting code was statically reviewed. Root selects its
report-only render in the existing 46460577 allocation after the already
running EP08 resource pilot; no model refits or additional source reads.

## 2026-10-04 06:32 UTC — `adaptive_search`: reviewed native sweep repair adopted; unchanged bounded failed-session pilot selected

Root and independent review passed the proposed one-sweep C99 kernel and
root adopts this engineering repair. The C function performs only the original
ascending physical-electrode sweep with independent X/Y updates; Python still
checks time before every sweep and computes the original-statistics KKT
certificate after every sweep. Warm starts, common scale, objective, five
penalties and10,000-sweep cap remain fixed. GCC12.4.0 was discovered through
the site's module system. Compilation is selected only inside46458807 with
`-O3 -std=c99 -fPIC -shared -ffp-contract=off`, no fast-math. New code/library
paths preserve the Python implementation and every earlier bank. Independent
native-versus-Python and original objective fixtures are required before use.

The resource pilot is specifically the failed Mihili_20140307 session, fresh
`primary_elastic_net_control_attempt002/Mihili_20140307`, with the unchanged
270/300second bounds. Only resource/convergence/support facts will select the
remaining-seven execution bound. Reuse exactly these complete original banks:
Chewie_20150313; Mihili_20140217/20140218/20140303/20140304/20140306;
Chewie_20160929/20161005. The other seven Chewie sessions are unlaunched.
New summary inputs will explicitly identify each reused or new bank, without
copying sources, refitting completed models or falling back to failed paths.
No complete scientific-panel charge is earned by this repair or pilot.

## 2026-10-04 06:31 UTC — `adaptive_search`: local optimizer deadline stopped the partial panel cleanly

The four-worker panel ran06:22:59.776586–06:30:13.702453UTC in46458807,
433.925877seconds, exit1. Seven remaining session banks completed:
Mihili_20140217/20140218/20140303/20140304/20140306 and
Chewie_20160929/20161005. Together with the original Chewie_20150313 pilot,
eight complete session banks are preserved in primary_elastic_net_control_attempt001.
Mihili_20140307 ran06:25:43.634354–06:30:13.634473UTC and hit its270second
internal deadline during the Python per-electrode coefficient update. Source
reading had completed in3.379398seconds; peak childRSS100,480KiB. This is an
observed local optimization timeout, not an observed source/filesystem failure,
KKT nonconvergence-cap failure or nonfinite fit. Its failed report is preserved.
The parent settled the active wave and launched none of the last seven
Chewie sessions. All partial scientific scores remain withheld; count19 persists.

Root requested a prospective engineering repair: compile only one canonical
cyclic coordinate sweep, preserving the original native objective, common
scale, physical order, five penalties, NumPy KKT check after every sweep,
Python deadline checks and10,000-sweep cap. Old Python tools and all banks stay
intact. The repair must pass independent review and coefficient/prediction
equivalence checks; no bitwise identity is presumed across reduction kernels.
The failed session is the proposed unchanged270/300s resource pilot before
any remaining-seven bound. No retry or larger deadline has been selected here.

## 2026-10-04 06:23 UTC — `adaptive_search`: four-worker elastic-net remainder launching after passed dispatch checks

All six focused dispatch fixtures passed in0.002seconds in46458807.
Changed ASTRA schema/semantic validation passed in
`astra/v0.0.14/validation_20261004_primary_elastic_net_remaining.json`, with
only the same five legacy optional-format warnings. Independent review and
root review passed the fixed-wave implementation. Root now launches all15
fresh remaining sessions, four single-thread workers, with the preselected
1,260/1,290second aggregate and270/300second per-child limits. The completed
pilot is reused unchanged. Fresh execution output is
`runs/adaptive_search/primary_elastic_net_panel_attempt001/remaining_panel.json`;
the complete scientific summary will be
`runs/adaptive_search/primary_elastic_net_summary_attempt001/primary_elastic_net.json`.
The current allocation has sufficient full aggregate reserve before its
06:51:41UTC end. Count19 and withheld pilot scientific scores remain until
the required full summary completes; no success is inferred from launch.

## 2026-10-04 06:22 UTC — `adaptive_search`: four-CPU allocation granted and actual shell verified

Job46458807 started06:21:37UTC on sh03-01n05.int, with scheduled end
06:51:41UTC. Root verified the actual shell hostname and SLURM_JOB_ID plus
scontrol: normal, four CPUs,12GiB, CPU_SKU:7502, Requeue=0, zero restarts and
no GPUs. PTY98868 is the sole active attended compute shell. Root loaded the
existing Python3.12.1/NumPy1.26.3/scikit-learn1.5.1/h5py3.10 modules and fixed
OpenMP/OpenBLAS/MKL/NumExpr to one thread. Six dispatch fixtures and changed
ASTRA validation are now executing there, followed by the independent EP08
high-power checks before any four-worker source launch. Both implementations
and fixture sets passed separate static reviews. No remaining EP05 session
or high-power scientific run has started at this entry.

## 2026-10-04 06:18 UTC — `adaptive_search`: completed one-CPU session closed; four-CPU execution allocation selected

Root exited attended compute PTY65476/job46457171 at06:18:07UTC, observed
exit0, after the pilot, EP08 conditional panel and result figure finished.
The new fixed-wave elastic-net wrapper passed root static review. Its focused
concurrency fixtures are being authored and will run in the four-CPU session,
so even mock worker counts fit the actual allocation. The unchanged eight
scientific model fixtures need no rerun. Fresh partition/hardware discovery
confirmed normal and CPU_SKU:7502. Root now selects one bounded attended
`srun --partition=normal --cpus-per-task=4 --mem=12G --time=00:30:00
--constraint=CPU_SKU:7502 --job-name=ep05-elastic-net --pty bash --noprofile --norc`
allocation for concrete parallel checks and the preselected four-worker panel.
Actual hostname/job ID and allocation limits will be verified inside its shell.
No source execution starts before the concurrency checks and review pass;
no duplicate request, automatic renewal or unrelated allocation is used.
The request returned job46458807, pending resources, attached to PTY98868.
Root set Requeue=0 successfully. The actual thirty-minute window starts only
when Slurm grants the allocation; a running node has not yet been observed.
At06:20UTC scheduler accounting confirms old46457171 COMPLETED/exit0,
24:52 allocated wall, one CPU, total CPU7:38.704 and step peakRSS2,766,012KiB.
The new46458807 is still pending; no duplicate is submitted. EP12's existing
worker remains running with advancing CPU and its successor stays dependency
pending; its stopped scratch-log route is not retried.

## 2026-10-04 06:15 UTC — `adaptive_search`: elastic-net pilot complete; four-worker remainder selected from resource facts

The complete Chewie_20150313 pilot ran06:12:20.156853–06:15:01.560127UTC
in46457171, exit0:161.403263seconds, source10.064492seconds and
fit/score151.120390seconds, peakRSS183,152KiB. Original1,038-trial/86-electrode
support replay is exact. All360 two-response fits converged; sweeps range
1–1,728, maximum KKT-residual/tolerance ratio0.999457. The pilot report is
`runs/adaptive_search/primary_elastic_net_control_attempt001/Chewie_20150313/score_banks.json`.
Only these resource/support/convergence fields were released; no scientific
pilot scores, sparsity, selected penalties or effect estimates were opened.

Root selects all remaining15 sessions with **four parallel single-thread
workers**, preserving each270second internal/300second external bound.
At the observed pilot cost this suggests roughly ten minutes of fit/read work;
four waves at the full child bound plus a60second summary reserve define a
**1,260second internal/1,290second external** aggregate bound. A concrete
four-CPU/12GiB/30minute attended allocation will be requested after the wrapper
change and focused checks are ready. This is within the existing shared budget;
no scientific threshold, solver cap or per-session deadline is increased.
The current one-CPU allocation cannot execute four simultaneous workers.

The unexecuted wrapper is being changed to enforce allocation-sized worker
count, original session identities and deterministic report ordering, complete
child settlement and no new queued launches after failure or insufficient
time reserve. Exact completed pilot reuse, fresh remainder directories and
complete16-session summaries remain required. The scientific count stays19
until both complete panels succeed. Root owns submission and monitoring.

## 2026-10-04 06:12 UTC — `adaptive_search`: elastic-net checks passed; selected pilot launching

Root and independent review passed the new core/session/panel implementation.
All eight independent scientific fixtures passed in0.633seconds in46457171,
including active-sign enumeration of the unique convex optimum, KKT error
bound, native qualified IDs, whole-trial controls, held-target exclusion and
outer-session selection. ASTRA schema/semantic validation passed in
`astra/v0.0.14/validation_20261004_primary_elastic_net.json`, with only the
five previously recorded legacy output-format warnings. Root selects immediate
execution of the predeclared270/300s Chewie_20150313 pilot into fresh
`runs/adaptive_search/primary_elastic_net_control_attempt001/Chewie_20150313`.
All scientific pilot scores stay closed pending a resource-only remaining
bound; count19 remains until complete cohort summaries. Root owns the launch
in the verified one-CPU/6GiB allocation46457171, ending06:23:20UTC.

## 2026-10-04 06:08 UTC — `adaptive_search`: fixed elastic-net pair adopted; resource pilot selected

Root adopts [the fixed elastic-net recipe](primary_elastic_net_proposal.md)
after root and independent scientific review. All physical LMP coordinates
remain available; the original native-XY mean-loss ridge objective gains the
fixed coordinatewise L1 term lambda times a single fitting-only pooled target
RMS. The original five penalties, baseline, support, outer-session selection,
wrong-trial scoring and practical margins remain unchanged. Eight whole-trial
within-direction training-input permutations preserve the exact input Gram
and are scored at the candidate-selected penalty. Realized sparsity need not
match. Full-LMP ridge banks supply independent and matched-penalty anchors.

The prescribed cyclic solver requires the original-statistics KKT certificate
at tolerance 1e-8 times max(1,maxabsC), at most10,000 sweeps per penalty/arm;
failure is unresolved, with no tolerance relaxation or omitted fit. Native
session-qualified trial IDs remain intact, with numeric ordering supplied by
the previously qualified representation helper. Root selected the complete
Chewie_20150313 resource pilot,270seconds internal/300seconds external,
one CPU in the current6GiB allocation46457171; scientific scores remain closed
until a resource-only remaining-cohort bound is selected. The branch expects
under4GiB. Implementation/independent fixtures are underway; no new scientific
fit or comparison is claimed. Complete candidate/control panels would add
two evaluations, from19 to21. This is delegated exploration, not a new margin,
data role, budget extension or canonical acceptance.

## 2026-10-04 06:01 UTC — `adaptive_search`: nineteen comparisons complete; supervised compression loses to full LMP

The remaining15 sessions completed05:58:42.021278–06:00:05.369424UTC in
46457171, exit0,83.348147s, maximum childRSS158,684KiB, plus the preserved
11.355112s repaired pilot. All16 original sessions and eight basis-control
seeds completed with original support replay; neither structural duplicate
condition holds. See [execution](runs/adaptive_search/primary_covariance_svd_panel_attempt001/remaining_panel.json)
and [complete result](runs/adaptive_search/primary_covariance_svd_summary_attempt001/primary_covariance_svd.json).
There are19 valid comparisons, including7 controls among13 post-initial
comparisons. No candidate satisfies the primary practical criteria.

Covariance-SVD P gives primary0.000290014223 and identity0.000602383366,
missing0.01/0.005. Mihili/Chewie primary means are0.000061180244/0.000518848203,
identity0.000391660985/0.000813105747, with4/6+8/10 primary-positive sessions.
Every outer session selects lambda10. Overall fixed omissions remain positive,
but omitting Mihili20140307 flips its primary mean to−0.000039183287; both
identity animal means remain positive under every omission.

Mean-eight basis-label control C gives primary−0.000063037717 and
identity0.000045966959. P−C is0.000353051940/0.000556416407, with both animals
positive and5/6+9/10 primary-positive sessions. Its declared directional screen
passes. All eight individual seed contrasts and all fixed omissions retain
positive primary/identity means in both animals. Overall omission ranges are
0.000299378496–0.000423627094 primary and0.000505836946–0.000581846357 identity.
This supports information in the learned basis under this comparison, not a
practically sufficient decoder or a causal mechanism.

P loses to independently selected full LMP by0.001647370522 primary and
0.002793941180 identity; both animals lose and only1/6+1/10 sessions improve
primary. At the candidate's matched lambda10, losses remain0.000339445570 and
0.000171968841, both animal means negative and0/16 primary-positive sessions.
Every fixed omission preserves both animal losses on both metrics for both
anchor selections. Both improvement-over-anchor screens fail.

All128 candidate folds and1024 control folds retain numerical rank2. Candidate
projected trace spans4.440633–24.015962, control3.006551–41.268611; control/candidate
trace ratios0.268381–7.241716. Effective ridge df spans0.347773–1.043025 versus
0.260878–1.227585; projector overlap0.010600–1.332482. These head diagnostics
do not count the supervised basis-estimation degrees of freedom or establish
equal capacity. No rank/energy diagnostic changes selection. The next worker
is preparing an unadopted fixed elastic-net challenger under the existing
grammar; further search remains required and future-session access stays closed.

## 2026-10-04 05:55 UTC — `adaptive_search`: repaired pilot complete; full remainder selected from resource facts

The fresh attempt002 resource pilot completed05:54:34.599768–05:54:45.954891UTC
in46457171, exit0,11.355112s (source10.198228s, two-recipe fit/score0.968338s),
RSS154,524KiB. All1038 trials/86 physical LMP channels replay the original
baseline, target and trial identities exactly. The failed attempt001 is
preserved separately. Only support, timing, memory and replay fields were
released; scientific scores and learned ranks were not inspected.

Root selects all fifteen remaining original sessions with600s internal/630s
external aggregate bounds, each original270/300s child retaining the300s
start reserve. The measured11.36s pilot and earlier complete current-bin
remainder runtimes66.88/131.43s support this bound with substantial allowance;
this is a resource decision, not a guarantee of completion. Reuse the complete
pilot once, write fresh remaining sessions into `primary_covariance_svd_control_attempt002`,
the execution record into `primary_covariance_svd_panel_attempt001`, and the
complete summary into `primary_covariance_svd_summary_attempt001`.
Root will execute after the ready EP08 comparison in the same attended
allocation ending06:23:20UTC, with enough room for both full bounds and cleanup.
All17 valid comparisons remain unchanged until both full panels complete.

The selected remaining-cohort ASTRA recipe passed schema/semantic validation
with the same five optional-format warnings
(`validation_20261004_primary_covariance_svd_remaining.json`). Actual launch
was05:58:41UTC in46457171/PTY65476, unchanged600/630s bounds. Root monitors
this process; no partial scientific score is released.

## 2026-10-04 05:50 UTC — `adaptive_search`: pilot ID-type defect repaired; fresh bounded pilot selected

The first covariance-SVD pilot failed before fitting at05:48:27.366444UTC,
after42.343398s (source42.343334s), RSS110,012KiB, exit1. Its original complete
source view had1038 trials and86 LMP channels. The preserved
`primary_covariance_svd_control_attempt001/Chewie_20150313/score_banks.json`
records `ValueError: Complete original current-bin LMP/XY/trial support required`.
The diagnosed cause is a new fitter integer-ID guard: the qualified source
adapter uses session-qualified IDs such as `Chewie_20150313_trial000001`.
No covariance basis, fitted model or scientific score was produced.

Root repaired only ID representation handling: preserve every full original
token in maps/predictions and parse its canonical six-digit trial suffix solely
for numeric ordering. Session prefixes and malformed suffixes are rejected.
This retains the fixed trial identities, seed literals and scientific recipe;
no source support, gate, normalization or model changed. Independent static
review passed. Ten scientific fixtures now pass in1.116s in46456142, including
exact numeric-versus-native-token permutation and prediction equality.

The failed attempt remains intact. Root selects a fresh Chewie resource pilot
under `primary_covariance_svd_control_attempt002`, same270/300s bounds, to be
run in a new attended one-CPU4GiB30-minute normal allocation using the discovered
CPU_SKU:7502 hardware. The current allocation had insufficient safe headroom
for that full bound and was closed normally at05:50:33UTC (PTYexit0); it is not
renewed. The next finite allocation bundles the repaired pilot, any remaining
cohort selected from its resource facts, and ready authorized EP08 work.
No new comparison is charged, and all pilot scientific scores remain closed.

At05:52 the new allocation46457171 remained pending (one task, one CPU,
30minutes); no duplicate was submitted. Accounting confirms46456142 COMPLETED,
25:05 allocated elapsed (0.418056 core-hours), TotalCPU4:45.626, stepRSS4,151,796KiB,
exit0:0. This observed aggregate peak approaches its4GiB limit even though the
scientific subprocess reports were smaller. Root therefore updates only the
pending new job's memory request to6144MiB (6GiB) for headroom; scientific
threads and runtime bounds remain fixed. An initial `MinMemoryNode=6G` command
was rejected as invalid syntax and made no change; integer6144 is the corrected
Slurm argument. This is within both episodes' existing memory budgets.

Allocation46457171 started05:53:16UTC and ends06:23:20UTC. The actual
execution shell is verified `sh03-01n68.int`, job46457171, one CPU,
SLURM_MEM_PER_NODE6144; scheduler allocation agrees6GiB. Numerical threads
are one; no GPU. Root disabled the scheduler's default requeue flag on this
own job. The corrected attempt002 ASTRA paths validated successfully in this
shell (`validation_20261004_primary_covariance_svd_attempt002.json`). Root
launches the selected fresh repaired pilot here at05:54UTC, retaining the
270/300s bounds and all17 prior valid comparisons.

## 2026-10-04 05:47 UTC — `adaptive_search`: covariance-SVD checks passed; resource pilot launching

Root and independent implementation review passed for the new core, session
runner and remaining-session wrapper. All nine scientific fixtures passed
in0.938s in the verified46456142 compute shell (`sh03-01n68.int`, one CPU,
4GiB, explicit one-thread numerical libraries). They cover direct native
cross-covariance, rank0/1 handling, orthogonal projection invariance, fitting
scaling, whole-trial original-ID maps, true-target heads, eight-seed SSE,
held-direction exclusion, full-span replay, outer lambda and duplicate counts.
[ASTRA validation](astra/v0.0.14/validation_20261004_primary_covariance_svd.json)
passed schema and semantics with the same five pre-existing optional-format
warnings. No scientific comparison is charged by these checks.

Before the process side effect, root selects the adopted complete
Chewie_20150313 resource pilot in46456142,270s internal/300s external, fresh
`runs/adaptive_search/primary_covariance_svd_control_attempt001/Chewie_20150313`.
The allocation ends05:55:37UTC; remaining-cohort execution will be selected
only after the pilot's support/resource facts and remaining walltime are known.
No pilot scores will be released for that resource decision.

## 2026-10-04 05:41 UTC — `adaptive_search`: covariance-SVD candidate/control adopted

After root and independent scientific reviews passed, root adopts the exact
[covariance-SVD proposal](primary_covariance_svd_proposal.md) under the existing
train-only PLS/ridge grammar. Every outer seven-direction fitting set supplies
its physical scaling and native-XY cross-covariance; retain all numerical-nonnull
left singular vectors (at most two), then fit the original ridge head. Eight
ordinary within-direction whole-trial target maps learn separate control bases;
all heads fit true targets. Zero/one-rank cases remain valid without padding,
and controls retain their own rank and covariance. Candidate-selected outer
lambda and mean-eight SSE, both anchor selections, all original gates and
budgets remain fixed. This advances no comparison until both full panels finish.

The implementation worker owns new core/runner/wrapper and ASTRA entries;
a separate worker supplies scientific fixtures. Root owns compute and records.
The selected first execution is the predetermined Chewie_20150313 resource
pilot, one CPU within4GiB,270s internal/300s external, after focused code and
fixture review. Its numerical comparison scores stay closed until root chooses
the remaining-panel bound from resource/support facts. No source execution has
started for this recipe. The adopted proposal supplies exact permutation seeds,
SVD tolerance, claim limits and prospective comparison screens.

Optional read-only BR retrieval at05:42 used `kg_search_nodes`, query
`Gallego Carracedo local field potentials`, Publication limit3, to look for
direct M1 LFP methodological context. It returned PMID21122819,19362152 and
26019312 (Parkinson LFP/MEG, emotional-stimulus epilepsy LFP, accumbens reward),
not relevant evidence for this motor-LFP transform. No method fields were
inferred, no BR endorsement obtained and no scientific decision changed.

Root also rendered and visually reviewed the completed top16 evidence in
46456142: [PNG](figures/primary_relevance_attempt001.png),
[PDF](figures/primary_relevance_attempt001.pdf), and
[54-row data](figures/primary_relevance_attempt001.csv). Only the complete
scientific summary was read; all16 sessions and both animal means are shown
against the preserved full-LMP anchor at candidate-matched lambda. No new fit,
source read or scientific comparison is charged.

## 2026-10-04 05:35 UTC — `adaptive_search`: seventeen comparisons complete; top16 relevance fails its screens

The remaining15 sessions completed05:32:13.821781–05:34:25.249503UTC in
46456142, exit0,**131.427721s**, maximum childRSS182,164KiB, plus the preserved
50.381017s pilot. All original sessions/eight control seeds completed; the
pair is not a full-retention duplicate. See the
[remaining execution](runs/adaptive_search/primary_relevance_panel_attempt001/remaining_panel.json)
and [complete scientific result](runs/adaptive_search/primary_relevance_summary_attempt001/primary_relevance.json).
The total is17 valid comparisons, including6 controls among11 post-initial
comparisons. All practical and adopted relevance screens fail.

R gives primary **0.000565574957** and identity **0.001844181862**, below
0.01/0.005. Mihili/Chewie primary means are−0.000155254565/+0.001286404480;
identity means are0.000939664729/0.002748698996. Primary is positive in3/6
and9/10 sessions, insufficient for the Mihili majority. Lambda is1 in15
sessions and0.1 inone; all controls share the selected R lambda.

The mean-eight rank-label control gives primary0.000447292860 and
identity0.000856185950. R minus control is primary+0.000118282098 and
identity+0.000987995912. Mihili primary is−0.000321925912 while Chewie is
+0.000558490108; only1/6 and8/10 primary session contrasts are positive.
Both identity animal contrasts are positive. Every fixed omission preserves
the negative Mihili and positive Chewie primary contrast; the overall primary
range remains+0.000030046096 to+0.000381286659. All eight individual seeds
also beat R on Mihili primary, while R beats every seed on Chewie primary and
on both animal identity means. Seed0 beats R on overall primary. These are
finite control spreads, not eight independent replicates of the animals.

Against independently selected full LMP, R loses **0.001371809788 primary
and0.001552142683 identity**. At R's matched lambda it loses
0.001387522226/0.001753783496. Both animal means and every fixed omission
retain both losses. R therefore does not improve the full representation;
training-only single-electrode relevance does not establish jointly useful
retention under this fixed16-channel recipe.

All128 folds retain16 channels and have16–73 positive inner-ranking gains;
none has all scores tied. One fold has a two-channel exact cutoff tie handled
by original physical ID. Candidate/control covariance traces are16 to floating
precision; numerical ranks range15–16 and effective degrees of freedom still
differ (R4.530–12.886; control4.585–12.809). Control overlap ranges0–7 retained
channels. No rank deficiency, tie, negative score or unfavorable session was
removed. Marginal relevance can miss jointly useful coordinates; this result
does not reject all supervised compression, LFP information or sparse models.
No final selection, spike follow-up or future-session opening is activated.

## 2026-10-04 05:30 UTC — `adaptive_search`: top16 pilot complete; full remaining panel selected

The complete Chewie_20150313 pilot ran05:27:47.390898–05:28:37.771921UTC
in46456142, exit0: **50.381017s**, with47.809722s source reading,
2.117207s paired/control fitting and scoring, and188,080KiB peakRSS.
All1,038 original trials,86 physical LMP channels and0:+14 native bins were
preserved with exact original baseline/target/trial replay. Only these
resource/support fields were released; no pilot scientific score was consulted.

Select the complete remaining15 with **1200s internal/1230s external**,
unchanged270/300 per child and300s reserve before each new child. Fifteen
pilot-sized sessions plus the reserve project1056s, fitting within1200s.
This is a resource-only choice, not an outcome-adaptive reduction of work.
The unchanged pilot is reused; all other bank paths and the summary/panel
paths are fresh. The wrapper requires every original session/seed, preserves
failures, never retries, and charges zero only for complete structural
full-retention duplication. Otherwise the complete pair advances15to17.

Run the exact selected `relevance_remaining_banks` ASTRA command sequentially
after EP08, only if at least1230seconds remain before the verified allocation
end05:55:37UTC. The revised ASTRA recipe still awaits compute validation;
no remaining-panel launch is claimed in this selection entry.

At05:31UTC the revised ASTRA schema/semantics passed, exit0, with the same
five optional format warnings (`validation_20261004_primary_relevance_remaining.json`).
The preceding EP08 control completed normally. Root now launches the selected
1200/1230 remaining-panel command in46456142 with over23minutes remaining,
more than its20m30s external bound. All fifteen source sessions and the complete
summary remain required; count stays15 until completion.

## 2026-10-04 05:28 UTC — `adaptive_search`: top16 checks passed; complete-session pilot launch

Root verified the actual shell assh03-01n68.int,job46456142,1CPU/4096MiB.
The eight new fixtures first gave6passes/2failures in0.604s: their zero-tolerance
expectations rejected3.55e-15 centering roundoff and a retained−3.05e-16 signed
zero-signal SSE gain. Only those fixture expectations were corrected to1e-13
absolute tolerance; no score clipping, numerical recipe or scientific gate
changed. All8 then passed in0.534s. Root and independent code review passed.
Changed ASTRA schema/semantics passed, exit0, with the same five pre-existing
optional format warnings (`astra/v0.0.14/validation_20261004_primary_relevance.json`).

Select and launch the already adopted complete Chewie_20150313 pilot below;
all directions, eight controls, original trials/channels and five lambdas are
required. Release only resource/support facts before choosing the remaining
fifteen-session execution bound. No source read occurs on the login node.

```bash
timeout 300s python3.12 outputs/tools/run_primary_relevance.py session --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" --trial-table outputs/runs/source_support/attempt002/trial_identity.csv --scalar-bank-root outputs/runs/adaptive_search/primary_reference_attempt001 --reference-bank-root outputs/runs/adaptive_search/primary_multichannel_attempt001 --session Chewie_20150313 --out outputs/runs/adaptive_search/primary_relevance_control_attempt001/Chewie_20150313 --max-seconds 270
```

## 2026-10-04 05:23 UTC — `adaptive_search`: attended execution resumed after interruption

The earlier compute handle77696 is gone. Slurm records46441736 FAILED,
15:40 elapsed, exit0:9, step peakRSS1,014,704KiB and totalCPU2:33.754. At the
last observed state the shell was idle after the completed PCA panel and plot;
the top16 fixtures and pilot had not run. Saved completed reports remain
valid. The terminal interruption is not classified as a scientific failure
or evidence of source corruption.

Root authored eight independent top16 fixtures and reviewed the new core,
single-session runner and sequential remaining-session wrapper; a separate
worker's static review passed. Tests and changed ASTRA validation still
require compute. Root requested a new bounded attended allocation for this
selected pilot/panel and the separately scoped EP08 control: returned job
**46456142**, normal/CPU_SKU7502,1CPU/4GiB/30minutes, currently queued.
No source reads or tests ran on the login node, and no old attempt is retried.

## 2026-10-04 00:39 UTC — `adaptive_search`: nested predictive-relevance top16 pair adopted

Root and independent proposal review passed. Root adopts
[the complete fixed top16 specification](primary_relevance_top16_proposal.md)
within the current training-only top-k grammar. Its singleton k16, nested
six-direction baseline/intercept/scaling and fixed alpha1 single-channel
native-XY relevance ranking were drafted before PCA scores; the released PCA
failure now supplies additional motivation without altering the rule.
Every final outer model retains the original seven fitting directions and
all original trials, native XY, bins and scoring. Eight fixed ordinary rank-
label permutations supply one same-width random-retention control with own
Grams and the candidate's outer lambda. Retain both original full-LMP anchor
comparisons, all negative/tied ranking scores and covariance/capacity caveats.
This estimates supervised predictive relevance, not intrinsic reliability or
conditional complementarity. No margin or stopping rule changes.

Implementation and independent scientific fixtures precede a selected complete
Chewie_20150313 resource pilot on1CPU/4GiB,270s internal/300s external, only
with sufficient time in the attended allocation. Scores remain unreleased
until a complete fixed sixteen-session panel. New code and ASTRA belong to
the EP05 worker; root owns execution and these records. No empirical launch
is asserted here.

## 2026-10-04 00:38 UTC — `adaptive_search`: fifteen comparisons complete; PCA denoising does not improve full LMP

The remaining fifteen sessions completed in46441736 from00:36:01.556271 to
00:37:08.425307UTC, exit0,66.877635s, preserving the11.581939s pilot.
All sixteen original sessions and eight loading-label seeds completed without
retry. The [execution report](runs/adaptive_search/primary_pca_panel_attempt001/panel_execution.json)
and [scientific summary](runs/adaptive_search/primary_pca_control_summary_attempt001/primary_pca.json)
record two valid recipes, bringing the total to15, with5controls among9
post-initial comparisons. This is not a full-span structural duplicate.

PCA90 gives primary increment **0.001377968512** and identity increment
**0.002837072838**: both0.01/0.005 margins fail. Mihili/Chewie primary means
are0.000597586030/0.002158350994 and identity means
0.002066755501/0.003607390175. Primary is positive in5/6 and9/10 sessions;
identity is positive in all16. Every session selects lambda1.

The mean-eight loading-label control gives primary0.000924334377 and
identity0.001691368973. PCA minus control is primary+0.000453634135 and
identity+0.001145703865, with both animals positive,4/6 and9/10 positive
primary sessions, and all16 positive identity contrasts: its directional
screen passes. Fixed-session omissions retain positive overall contrasts
(primary0.000329486302–0.000588911021; identity
0.001075259638–0.001199552490). Individual seed control means range
0.000778347124–0.001074902426 primary and
0.001562189243–0.001839181051 identity, all below the candidate overall.

PCA minus full LMP is **−0.000559416233 primary and−0.000559251708 identity**.
The independently selected and PCA-lambda-matched references coincide at1.
Both animal primary and identity contrasts are negative; primary gains occur
in0/6 Mihili and2/10 Chewie sessions, identity gains in0/16. All fixed
omissions preserve both overall losses. Consequently both adopted denoising
screens fail; the loading-label result does not establish a denoising benefit.

Across128 held-direction folds, retained K ranges24–44. Every control retains
numerical rank K, but its projected covariance trace is only25.02–57.31% of
the paired projection and its ridge effective degrees of freedom are lower.
These diagnostic differences were expected and do not trigger a redraw or
retuning. The control supports a predictive role for the learned subspace
relative to permuted loadings, with covariance/energy confounding preserved;
it is not a pure physical-channel identity intervention. No spike follow-up,
future-session evaluation, new-animal claim or final selection is activated.

Root rendered `tools/plot_primary_pca.py` in46441736 and visually reviewed
`figures/primary_pca_attempt001.{png,pdf,csv}`:54 plotted rows cover all16
sessions and both animal means for full LMP, PCA and mean-eight control.
The source is the complete summary and matched-lambda anchor; no neural reads,
refits, exclusions or confidence intervals are added.

## 2026-10-04 00:35 UTC — `adaptive_search`: PCA remaining-panel validation and execution handoff

The selected remaining-fifteen ASTRA recipe passed schema and semantic
validation in46441736, exit0; the same five pre-existing optional format
warnings remain (`astra/v0.0.14/validation_20261004_primary_pca_remaining.json`).
After the EP08 comparison completed, the coordinator launched the exact600/630
command below at approximately00:36UTC in this one-CPU allocation. Its verified
end is00:57:04UTC. This records launch, not a completed panel or a release of
pilot scientific scores.

## 2026-10-04 00:30 UTC — `adaptive_search`: PCA pilot complete; remaining fifteen selected

The complete Chewie20150313 pilot ran in46441736 from00:28:22.948702 to
00:28:34.530648UTC, exit0: **11.581939s**, including9.952324s source reads and
1.408708s paired/control fit and score, with163,384KiB peakRSS. All1,038
original trials,86 matched LMP electrodes and0:+14 bins are retained, with
exact original target/baseline/trial replay. Only timing/support fields were
released before selecting the remaining workload; no pilot score selected
scope or a model. The completed bank stays at
`runs/adaptive_search/primary_pca_control_attempt001/Chewie_20150313`.

Select all remaining fifteen sessions in the same1CPU/4GiB allocation with
**600s internal/630s external aggregate**, unchanged270s internal/300s external
per child, and the wrapper's300s reserve before each new child. Fifteen
pilot-sized sessions project174s; the600s bound accommodates source variation,
the final child reserve and summary. Root reviewed the finite
`tools/run_pca_panel.py`: complete pilot reuse, fresh remaining paths, no retries,
all sixteen before summary, and correct zero-versus-two structural-duplicate
accounting. The exact original scientific fixtures remain applicable.

```bash
timeout 630s python3.12 outputs/tools/run_pca_panel.py --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" --trial-table outputs/runs/source_support/attempt002/trial_identity.csv --scalar-bank-root outputs/runs/adaptive_search/primary_reference_attempt001 --reference-bank-root outputs/runs/adaptive_search/primary_multichannel_attempt001 --reference-summary outputs/runs/adaptive_search/primary_multichannel_summary_attempt001/primary_multichannel.json --bank-root outputs/runs/adaptive_search/primary_pca_control_attempt001 --summary-out outputs/runs/adaptive_search/primary_pca_control_summary_attempt001 --out outputs/runs/adaptive_search/primary_pca_panel_attempt001 --max-seconds 600
```

## 2026-10-04 00:19 UTC — `adaptive_search`: PCA denoising and frozen loading-label control adopted

Root adopted the complete [PCA90 specification](primary_pca_denoising_proposal.md)
after independent review, including descriptive effective-rank/variance
diagnostics, before new source values or fits. This is the existing permitted
training-only PCA branch, not an activated direction-rotated target/model.
Keep current LMP/XY0:+14, all sixteen sessions, original electrodes/trials,
held directions, baseline/intercept and practical margins unchanged.

P retains the smallest training covariance span reaching90% positive variance,
including the complete exact cutoff-eigenvalue tie. It does not whiten or
rescale components. The fixed PSD roundoff convention is64*eps_float64*E*
max(1,max_abs_eigenvalue). Eight control maps permute physical rows of the
already learned V, keeping K fixed; maps remain constant across each session's
folds, trials, times and lambdas. Do not refit PCA on relabeled coordinates.
Each arm fits its own projected Gram. Controls retain the original physical
normalization and change covariance/retained variance; nominal K is not an
equal-effective-information claim.

Every arm/fold retains covariance eigenvalues, trace, numerical rank under
the corresponding fixed K-scaled tolerance, and ridge effective df at each
original lambda. These do not select models, change rank or exclude a control.
Rank-deficient and zero-energy controls remain evaluable. Candidate-selected
outer lambda is shared by all controls and the reused matched-lambda LMP
reference; the independently selected anchor is also retained. Average seed
correct/wrong SSE, never predictions. Full-span folds are reported as
uninformative label controls; if all128folds retain the full span, the duplicate
pair adds no scientific depth. Otherwise two complete recipes advance13to15,
including one control (five among nine post-initial comparisons).

New implementation is `tools/primary_pca.py` and `tools/run_primary_pca.py`,
reusing qualified LMP reads, ridge solve and post-fit exact reference replay.
Root's full core/runner static review passed. In allocation46440428 the first
nine-fixture run had eight passes and one report-key KeyError in0.181s: root
tested before the requested `joint_lambda_matched` to `pca_lambda_matched`
rename handoff. After the reporting-key repair, all nine independent scientific
fixtures passed in0.418s; no numerical equation changed. Changed ASTRA schema
and semantic validation passed with the same five pre-existing optional format
warnings (`astra/v0.0.14/validation_20261004_primary_pca.json`). No worker
executed tests or opened source values.

Root selected complete Chewie_20150313, both arms/all seeds,1CPU/4GiB,
270s internal/300s external, only if at least five minutes remain in the
verified parent allocation. Current46440428 ends00:28:57UTC. A later bounded
allocation is required if review finishes too late; no deadline reduction
or login-node fallback. Proposed bank/summary paths are
`primary_pca_control_attempt001` and `primary_pca_control_summary_attempt001`
under `runs/adaptive_search`. Remaining-panel sizing awaits pilot cost.

At00:24UTC fewer than five minutes remained, so no PCA pilot was launched in
46440428. Root exited that attended shell and selected a new explicit bounded
30-minute normal/CPU_SKU:7502,1CPU/4GiB attended allocation for the ready PCA
pilot, measured remaining-panel execution if it fits, and the adopted EP08
reverse-transfer work after its own checks. Current `sh_part`/`sh_node_feat`
confirm those public CPU resources. No renewal, background supervisor or
duplicate pending allocation is used. The next allocation ID and actual
execution-shell verification will be recorded before source execution.
Root's single new request returned **job46441736**, pending resources in
attended PTY77696 at00:25UTC; the previous PTY71811 exited cleanly with code0.
Pending state is not an execution failure or permission to submit a duplicate.
At00:28:02UTC the actual new shell reported `sh03-06n50.int`,
`SLURM_JOB_ID=46441736`,1CPU and4096MiB; scheduler independently showed it
RUNNING with the requested30-minute limit. The qualified module stack and
single-thread settings are being loaded, then root starts the exact logged
Chewie20150313 pilot with270s internal/300s external. Previous46440428 is
independently COMPLETED, exit0:0,25:55 wall,3:48.294 TotalCPU and2,527,016KiB
step peakRSS; its completed EP05/EP08 artifacts and checks remain reusable.

The optional BR PLS-title lookup did not provide directly relevant LFP
evidence or method fields. One quoted Boolean request failed with a parser
ParseException; a single corrected plain-language query succeeded, returning
four fMRI/gray-matter/visuomotor PLS titles. This was not a KG outage, branch
validation or scientific endorsement. The proposal documents the exact DOIs;
there was no selection change based on this limited advisory retrieval.

```bash
PYTHONPATH="$PWD/outputs/tools${PYTHONPATH:+:$PYTHONPATH}" python3.12 -m unittest discover -s outputs/tools -p test_primary_pca.py -v
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261004_primary_pca.json
timeout 300s python3.12 outputs/tools/run_primary_pca.py session \
  --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
  --scalar-bank-root outputs/runs/adaptive_search/primary_reference_attempt001 \
  --reference-bank-root outputs/runs/adaptive_search/primary_multichannel_attempt001 \
  --session Chewie_20150313 \
  --out outputs/runs/adaptive_search/primary_pca_control_attempt001/Chewie_20150313 \
  --max-seconds 270
```

## 2026-10-04 00:10 UTC — `adaptive_search`: complete fixed-decoder timing control passes its directional screen

The [complete timing-control report](runs/adaptive_search/primary_temporal_offset_attempt001/primary_temporal_offset.json)
ran 00:10:36.619–00:10:37.333 UTC in allocation46440428, **0.714216 seconds**
with 66,160 KiB peak RSS. It retains all sixteen original sessions and fourteen
nonzero offsets, with selected anchor scores and original trial support replayed.
No model was fitted and no provider/source payload was read. Root released the
whole result after completion. Charge **one** temporal alternative-explanation
comparison: **thirteen total**, four controls among seven post-initial
comparisons (57.1%). Offsets, exact wrong-pair readouts and the reused aligned
anchor do not add further comparisons.

| Comparison | Primary increment | Identity increment | Positive primary sessions (Mihili/Chewie) |
| --- | ---: | ---: | --- |
| Frozen aligned full-channel LMP A | +0.001937385 | +0.003396325 | 5/6; 10/10 |
| Mean fourteen neural-correction offsets O | −0.001746353 | +0.000392453 | 0/6; 1/10 |
| A minus O | +0.003683738 | +0.003003871 | 6/6; 10/10 |

O's primary means are −0.001542628 Mihili / −0.001950079 Chewie, with small
positive identity means +0.000373052 / +0.000411855. A-minus-O primary means
are +0.002557370 / +0.004810107 and identity means +0.002162016 /
+0.003845727. Both primary and identity contrasts are positive in every
session; the adopted directional timing screen passes. Fixed-prediction
session-omission contrasts remain positive: primary +0.003550127 to
+0.003812866, identity +0.002888475 to +0.003128235. No offset or lambda was
selected from these results, and the fourteen per-offset scores remain
descriptive components of this one finite control.

This supports within-trial temporal alignment for the fixed stationary LMP
decoder. Offset trajectories retain the original trial's correction values
and can preserve slow or trial-constant information, consistent with the
small positive control identity score. The cyclic wrap and nonstationary
movement-phase limitations remain: this is not a uniform physical lead/lag,
online test or causal result. Keeping the direction/time baseline and shared
intercept at the original target time distinguishes this from shifting the
behavior prediction itself.

The aligned anchor is unchanged and still fails **0.01 primary / 0.005 identity**;
passing this control does not rescue those failures. A later different model
cannot inherit this fixed-anchor timing screen automatically. Remaining
coverage, 28–64 depth, patience and promotion conditions stay incomplete.
Root has requested a prospective within-grammar training-only denoising
candidate/control design; no new source or model is activated by that request.

Root rendered and visually reviewed the [timing-control figure](figures/primary_temporal_offset_attempt001.png)
and [PDF](figures/primary_temporal_offset_attempt001.pdf) in46440428, then shared
them with the user. The [288-row plotted table](figures/primary_temporal_offset_attempt001.csv)
retains every session, both animal means, all fourteen offsets and the separate
aligned/mean-control summaries. There are no confidence intervals or extra fits.

## 2026-10-04 00:10 UTC — `adaptive_search`: timing-control report launch selected

Root's full static review of the fixed temporal-offset core, finite runner
and seven focused fixtures passed. All seven tests passed in **0.040 seconds**
inside allocation46440428. Changed ASTRA schema and semantic validation passed
with the same five optional-format warnings; evidence is
`astra/v0.0.14/validation_20261004_primary_temporal_offset.json`.
The actual compute shell was reverified as sh03-06n50.int / SLURM_JOB_ID46440428.

Root starts the exact previously logged `run_primary_temporal_offset.py`
command after this handoff, with **150s internal / 180s external**, 1 CPU /
4 GiB and numerical threads one. Output remains
`runs/adaptive_search/primary_temporal_offset_attempt001/primary_temporal_offset.json`.
All fourteen nonzero offsets, original selected anchor lambdas, fixed behavior
baseline/intercept and all sixteen sessions remain unchanged. This run reads
only completed anchor predictions/score records and trial-identity metadata;
there is no provider/source payload access or new fit. No result is reported
or extra comparison charged before complete execution. Root owns the live
log through timing completion.

## 2026-10-04 00:03 UTC — `adaptive_search`: joint high-band trial alignment passes; complementarity fails

The [complete joint/control summary](runs/adaptive_search/primary_lmp_high_control_summary_attempt001/primary_lmp_high.json)
finished **00:03:52.151 UTC** in allocation46440428. The
[remaining-panel wrapper](runs/adaptive_search/primary_lmp_high_control_execution_attempt001/panel_execution.json)
took **142.204311 seconds**, separately from the 23.735810-second complete
pilot in46438625. All original sixteen sessions, eight high-block trial
derangements and exact reference replays completed. The summary's 0.072122
seconds / 61,576 KiB describe aggregation, not source/fitting panel cost.
Root released the scientific panel only after completion. Charge two
comparisons: **twelve total**, three controls among six post-initial
comparisons (50%).

| Comparison | Primary increment | Identity increment | Positive primary sessions (Mihili/Chewie) |
| --- | ---: | ---: | --- |
| Joint current-bin LMP + high power J | +0.002253424 | +0.005206781 | 5/6; 9/10 |
| Eight-seed high-trial derangement D | +0.000947359 | +0.003569632 | 3/6; 9/10 |
| J minus D | +0.001306064 | +0.001637149 | 6/6; 10/10 |

Every outer selector chose lambda1 for J, all control refits and matched-lambda
LMP. J passes the **0.005 identity** margin and both-animal sign/majority gates,
but fails **0.01 primary**. J's primary animal means are +0.000548255 Mihili /
+0.003958592 Chewie; identity means are +0.003489070 / +0.006924492.

Correct high-block trial alignment helps the joint decoder: J-minus-D primary
means are +0.000894608 / +0.001717521, and identity means are
+0.000924082 / +0.002350217. All sixteen primary contrasts are positive, so
the adopted alignment screen passes. D retains genuine recipient LMP input;
its positive identity mean is expected to remain possible and is not a failed
control construction. This test changes high-to-target pairing and LMP/high
cross-covariance together; it does not isolate causal cross-block synergy.

The stronger claim of added value beyond LMP **fails the adopted complementarity
screen**. J-minus-LMP primary is +0.000316039 overall, but **Mihili −0.000466486 /
Chewie +0.001098564**, with only 1/6 Mihili and 8/10 Chewie primary contrasts
positive. Identity increases +0.001810456 overall (+0.000954002 /
+0.002666911), which cannot rescue the primary loss in Mihili. Both the
independently selected and same-lambda LMP reference screens fail; their
contrasts coincide because all selected anchor/joint lambdas are1. No
per-animal feature choice, anchor substitution, high-only mechanism or
promotion is justified. High-band spike-contamination sensitivity remains
open.

The next required within-trial timing control was specified and adopted
before this joint outcome was opened, and continues to use the preserved
stationary LMP anchor. The 28–64 minimum/maximum, patience, remaining controls
and full-search requirements remain unchanged and incomplete.

## 2026-10-04 00:00 UTC — `adaptive_search`: exact within-trial timing control adopted

Root adopted the entire [temporal-offset specification](primary_temporal_offset_proposal.md)
before any offset evaluation. Keep the original stationary full-channel LMP
anchor, all sixteen sessions and the frozen 0:+14 endpoint. Exactly fourteen
nonzero cyclic offsets move only the held-out neural correction `P-B` within
each original trial; B, including the common residual intercept, stays at
target time. Stored independently selected anchor lambdas and original model
fits are unchanged. No source payload, new fit, lag search or additional
candidate is introduced. The pending joint result cannot change this choice.

The new core/finite runner are `tools/primary_temporal_offset.py` and
`tools/run_primary_temporal_offset.py`. Only complete preserved LMP predictions,
score banks, original panel selection and trial-identity metadata are inputs.
Selected unshifted correct/wrong SSE, denominator, baseline and row identities
must replay before offset scoring. Correct/wrong SSEs are computed separately
per offset and then averaged; predictions are not ensembled. Report every
offset descriptively and the adopted aligned-minus-offset both-animal/majority
directional screen. One complete panel charges one temporal control, not
fourteen comparisons. The diagnostic has explicit cyclic wrap and movement-
phase limitations and cannot rescue the frozen primary margins or establish
a physiological lead, online prediction or causality.

Seven focused fixtures are prepared in `tools/test_primary_temporal_offset.py`:
exact within-trial XY time maps/zero identity; baseline-only and constant-neural
invariants; equivalence to a frozen affine feature map; frozen lambda/reference
row checks; selected unshifted score replay; direct correct/wrong mean-SSE
versus ensemble; and equal-animal full-panel aggregation/one-control count.
Workers have not executed tests, read prediction arrays or run the control.
Root review, compute checks and changed ASTRA validation precede execution.

Root selected 1 CPU/4 GiB, 150s internal/180s external inside its verified
allocation46440428, after the full joint LMP/high panel. Preserve the fresh
`runs/adaptive_search/primary_temporal_offset_attempt001` directory and any
partial result; no automatic retry or score-based scope change.

```bash
PYTHONPATH="$PWD/outputs/tools${PYTHONPATH:+:$PYTHONPATH}" python3.12 -m unittest discover -s outputs/tools -p test_primary_temporal_offset.py -v
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261004_primary_temporal_offset.json
timeout 180s python3.12 outputs/tools/run_primary_temporal_offset.py \
  --anchor-bank-root outputs/runs/adaptive_search/primary_multichannel_attempt001 \
  --anchor-summary outputs/runs/adaptive_search/primary_multichannel_summary_attempt001/primary_multichannel.json \
  --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
  --out outputs/runs/adaptive_search/primary_temporal_offset_attempt001 \
  --max-seconds 150
```

## 2026-10-04 00:01 UTC — `adaptive_search`: remaining joint panel starting in verified allocation

Allocation46440428 started23:58:54UTC and ends00:28:57UTC. The actual attended
shell reports sh03-06n50.int and SLURM_JOB_ID46440428; scontrol confirms
1CPU/4GiB,30minutes. The qualified Python3.12.1/NumPy/scikit-learn/h5py module
stack and one-thread limits are loaded. Start the already selected remaining
panel now with900s internal/930s external, reusing the completed pilot at its
actual `primary_lmp_high_control_attempt001` location. No scientific choices
or child deadlines changed. Previous allocation46438625 is independently
COMPLETED, exit0:0,34:04 wall and10:54.364 TotalCPU; its failed EP04 subtask
remains a separate preserved failure despite the allocation's clean exit.

## 2026-10-03 23:55 UTC — `adaptive_search`: joint pilot complete; remaining panel selected

The complete pilot finished23:54:13.995250UTC in23.735810s, with source
16.562735s, fit/score7.082174s and388,068KiB peakRSS. All1,038original trials,
both recipes and eight controls completed with exact baseline/target/trial
replay; the qualified reader's low block was discarded without fitting.
Only completion, support and resource fields were inspected before scaling.

Actual bank root is `runs/adaptive_search/primary_lmp_high_control_attempt001`.
The pilot was written there rather than the earlier prepared command's
`primary_lmp_high_attempt001`; no data or code changed. ASTRA paths now match
the actual output, and fresh schema/semantic validation passed with the same
five optional warnings (`validation_20261003_primary_lmp_high_actual_paths.json`).
The finite `tools/run_lmp_high_panel.py` passed root static review and compute
AST parsing. All existing scientific tests remain applicable.

Select the remaining fifteen sessions with900s aggregate internal/930s
external limits and unchanged270s internal/300s external per child. Fifteen
largest-session costs project356s before overhead; the selected ceiling also
preserves the wrapper's300s reserve before each child. Reuse the complete
pilot regardless of its score; stop on any failure without retry or deletion.
Summary/execution locations are `primary_lmp_high_control_summary_attempt001`
and `primary_lmp_high_control_execution_attempt001` under `runs/adaptive_search`.

The attended46438625 shell exited cleanly at23:55UTC. The single next request
was accepted as46440428 and is queued as of23:55:59UTC (PTY71811). Its request
is normal/CPU_SKU:7502,1CPU,4GiB,30minutes. No science has run in that pending
allocation; the actual shell will be verified after Slurm starts it.

The current allocation has less than the selected ceiling remaining. Close
its finished attended shell, then request one bounded30-minute1CPU/4GiB
attended allocation for the complete panel and substantive result closeouts.
Root retains sole scheduler ownership, one numerical thread and no duplicate
pending request, automatic renewal or detached process. Verify the new shell
and actual allocation before scientific execution; no login-node fallback.

## 2026-10-03 23:53 UTC — `adaptive_search`: joint/control pilot launch

Root and independent full scientific reviews passed for the adopted proposal,
core and runner. All nine independent fixtures passed in0.394s in46438625.
Changed ASTRA schema/semantic validation passed with the same five optional
format warnings (`astra/v0.0.14/validation_20261003_primary_lmp_high.json`).
Launch the selected complete Chewie_20150313 pilot in1CPU/4GiB allocation
46438625 on sh03-06n50 with270s internal/300s external limits, one numerical
thread, all eight controls and unchanged science. More than five minutes
remain before the parent's00:01:29UTC deadline. Preserve all prior artifacts;
size the remaining panel from completed timing/resource fields only.

## 2026-10-03 23:48 UTC — `adaptive_search`: complete history result; joint LMP/high pair adopted

The [complete history/control result](runs/adaptive_search/primary_history_control_summary_attempt001/primary_history_control.json)
finished 23:46:21.119 UTC in allocation 46438625. The
[remaining-panel wrapper](runs/adaptive_search/primary_history_control_execution_attempt001/panel_execution.json)
took **130.533912 seconds**, separately from the 22.868114-second complete
Chewie pilot. All sixteen sessions, eight label-control seeds, original
trial/channel maps and explicit history/target windows completed and passed
reference replay. The summary's 0.867316 seconds is aggregation time, not
panel compute cost. Root released the scientific result after completion.
Charge two comparisons: **ten total**, with two controls among four
post-initial comparisons (50%).

| Comparison | Primary increment | Identity increment | Positive primary sessions (Mihili/Chewie) |
| --- | ---: | ---: | --- |
| Fixed four-bin history H | +0.002038782 | +0.005498616 | 5/6; 8/10 |
| Eight-seed channel-label control L | −0.004775122 | −0.000063844 | 0/6; 0/10 |
| H minus L | +0.006813904 | +0.005562460 | 6/6; 10/10 |

Every outer selector chose lambda 1, shared by all control refits. H passes
the **0.005 identity margin**, both-animal signs and primary majorities, but
fails the **0.01 primary margin**. Its primary animal means are
+0.000730601 Mihili / +0.003346963 Chewie; identity means are
+0.004479957 / +0.006517275. No promotion follows from the stronger identity
readout. H-minus-L passes the adopted directional label screen: primary means
+0.005402434 / +0.008225375 and identity means +0.004567850 / +0.006557069.
This supports stable labels after physical normalization for this fixed
history model; the control also changes covariance and retains normalization
information, so it is not a pure causal-channel test.

H's primary gain over stationary current-bin LMP is only +0.000101397 overall,
with **Mihili −0.000284140 / Chewie +0.000486935** and just 2/6 Mihili contrasts
positive. Removing Chewie_20150629 in the fixed-prediction influence diagnostic
changes that aggregate difference to −0.000059428. Identity improves by
+0.002102291 overall, but this is not robust primary superiority over the
anchor. Do not select different histories/models for the animals.

Before this complete result was opened, the next proposal fixed the stationary
LMP anchor plus one matched physical-electrode 100–200/200–400 Hz block and
an eight-seed within-direction whole-high-block trial derangement. Root
**adopted the full [joint/control proposal](primary_lmp_high_control_proposal.md)
at 23:48 UTC before any new fits**, including shared candidate-selected lambda,
same-lambda preserved LMP contrasts, both directional screens and the disclosed
unchanged reader's discarded low-band values. The history result does not
change that selected anchor or recipe. Implementation/review precedes the
complete Chewie_20150313 pilot: 1 CPU/4 GiB, 270 internal/300 external seconds,
only if at least five minutes remain in root's allocation. Remaining-panel
sizing awaits pilot cost; no worker executes source/tests. This pair tests
aligned extra high content, not isolated cross-block synergy. The 28–64 depth,
patience, remaining controls and promotion requirements remain open.

Root rendered and visually reviewed the [ten-comparison figure](figures/primary_ten_attempt001.png)
and [PDF](figures/primary_ten_attempt001.pdf) in allocation 46438625 at 23:49 UTC.
It shows both animal means, their equal-weight mean and the frozen margins;
plotting adds no fits or scientific comparisons.

The new joint implementation is `tools/primary_lmp_high.py` with runner
`tools/run_primary_lmp_high.py`. Root and independent core reviews pass.
The independent implementation colleague authored nine focused fixtures in
`tools/test_primary_lmp_high.py`: exact physical band order; fitting-only
normalization and nested LMP metric; canonical derangements; complete high
trajectories with fixed LMP; unchanged block/changed cross Gram and independent
solve; constant-feature baseline/target immutability; mean correct/wrong SSE;
exact seed/rejection draw accounting; and joint-only outer lambda with both
preserved anchor contrasts. No worker executed them. ASTRA now declares the
new pilot/full-panel artifacts. Root runs these checks and validation before
the selected pilot; no aggregate bound is yet selected.

```bash
PYTHONPATH="$PWD/outputs/tools${PYTHONPATH:+:$PYTHONPATH}" python3.12 -m unittest discover -s outputs/tools -p test_primary_lmp_high.py -v
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261003_primary_lmp_high.json
timeout 300s python3.12 outputs/tools/run_primary_lmp_high.py session \
  --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
  --scalar-bank-root outputs/runs/adaptive_search/primary_reference_attempt001 \
  --reference-bank-root outputs/runs/adaptive_search/primary_multichannel_attempt001 \
  --session Chewie_20150313 \
  --out outputs/runs/adaptive_search/primary_lmp_high_attempt001/Chewie_20150313 \
  --max-seconds 270
```

## 2026-10-03 23:44 UTC — `adaptive_search`: history pilot complete; remaining panel selected

The complete Chewie_20150313 pilot finished23:43:31.948117UTC after22.868114s
(source9.884705s, fit/score12.600741s), peakRSS385,796KiB, all1,038original
trials, both recipes and all eight control seeds. The wider LMP source and
unchanged target windows were verified; original baseline/target/trial replay
passed. Only support, completion, timing and memory fields informed scaling;
no partial scientific score was opened. Evidence:
`runs/adaptive_search/primary_history_control_attempt001/Chewie_20150313/score_banks.json`.

Root reviewed `tools/run_history_panel.py` and compute-side AST parsing passed.
Select the remaining15sessions sequentially in the same1CPU/4GiB allocation,
900s aggregate internal/930s external, unchanged270s internal/300s external
per session. Fifteen measured largest-session costs project343s before
overhead;900s supplies additional source/cost variation and the wrapper's
required300s reserve before each child. The parent has over17minutes remaining.
Reuse the completed pilot regardless of its score; preserve any partial
directory and stop on failure, with no automatic retries. Only the complete
sixteen-session summary will expose the new scientific comparison.

Bank root is `runs/adaptive_search/primary_history_control_attempt001`;
summary and execution directories are respectively
`primary_history_control_summary_attempt001` and
`primary_history_control_execution_attempt001` under `runs/adaptive_search`.

## 2026-10-03 23:43 UTC — `adaptive_search`: history/control pilot launch

Root's full static review and the second worker's independent scientific
review passed. Nine focused tests passed in0.294s in allocation46438625 on
sh03-06n50; ASTRA schema/semantic validation passed with the same five optional
format warnings. Start the previously selected complete Chewie_20150313 pilot
now in this verified1CPU/4GiB allocation, using270s internal/300s external
limits and one numerical thread. Its00:01:29UTC allocation deadline leaves
more than the required five minutes. Preserve the fresh attempt and inspect
completion/runtime/resource fields before selecting the remaining-panel bound.

## 2026-10-03 23:41 UTC — `adaptive_search`: fixed history/channel-label pair adopted and implemented

Before any new source values or fits, root adopted
[the fixed history/channel-label specification](primary_history_control_proposal.md),
including its directional screen. Candidate H uses stationary ridge on lags
[0,−1,−2,−3] of the same LMP electrodes; the evaluated XY endpoint stays
0:+14. Only the LMP source slice expands to −3:+14, inside the already
qualified −4:+14 support. Trial eligibility, folds, animals, baseline, five
penalties, primary/identity margins and the original current-bin anchor remain
unchanged. No animal receives a separately chosen history.

Control L uses eight fixed per-trial physical-channel permutations, each
shared across all four lags and fifteen evaluated bins, in training and
evaluation features. Physical lag-channel means/SDs come from fitting
directions before permutation; each arm then centers its own fitting columns
without rescaling. The control retains physical normalization information
and changes global covariance. It tests stable labels after that normalization,
not pure causal channel identity or absence of neural trial information.
Every arm fits its own Gram eigensystem; candidate outer-training selection
provides the common lambda. Seed SSEs are averaged, never predictions.

The new files are `tools/history_lmp_reader.py`, `tools/primary_history.py`,
`tools/run_primary_history.py` and `tools/test_primary_history.py`. Earlier
source/science code and results remain unchanged. The new reader reports
distinct native LMP and target windows. Its post-fit replay helper explicitly
checks those windows before calling the unchanged reference identity/target/
baseline checker; it does not relabel the larger feature window as 0:+14.
Every original trial's eight destination-to-source electrode-position maps
are retained alongside the frozen source channel list.

Nine focused fixtures cover native slice/lag endpoints, held-direction
exclusion, stationary embedding, whole-electrode time/lag maps and chance
identity permutations, the changed Gram's independent solve, constant-history
baseline and unchanged targets, mean-SSE aggregation, shared outer lambda and
truthful distinct-window reference replay. Root's full static review passed,
and all nine fixtures passed in 0.294 seconds on sh03-06n50.int in the
coordinator's allocation 46438625. The changed ASTRA program passed schema
and semantic validation, retaining only the five existing optional-format
warnings; evidence is `astra/v0.0.14/validation_20261003_primary_history_control.json`.
Independent colleague review is pending. **No worker ran tests or source
code.** The scientific implementation has not changed since those checks.

Root selected the complete Chewie_20150313 pilot, both arms/all eight seeds,
1 CPU / 4 GiB / five minutes, numerical threads one, 270-second internal and
300-second external limits. Use allocation 46438625 only if at least five
minutes remain after review/tests; its recorded end is 00:01:29 UTC. The
finite full-panel wrapper `tools/run_history_panel.py` was prepared after core freeze, with no
aggregate deadline selected until pilot runtime is observed. It preserves
complete banks, rejects existing partial attempts, requires all eight maps
per original trial and all fixed seed/lag identities, and needs at least
300 seconds before each unchanged 270-second/300-second child. No partial score
may choose scope or stop unfavorable science. Complete pair count would be
eight to ten, two controls among four post-initial comparisons (50%).

Compute-only selected pilot handoff after root review:

```bash
PYTHONPATH="$PWD/outputs/tools${PYTHONPATH:+:$PYTHONPATH}" python3.12 -m unittest discover -s outputs/tools -p test_primary_history.py -v
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261003_primary_history_control.json
timeout 300s python3.12 outputs/tools/run_primary_history.py session \
  --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
  --scalar-bank-root outputs/runs/adaptive_search/primary_reference_attempt001 \
  --reference-bank-root outputs/runs/adaptive_search/primary_multichannel_attempt001 \
  --session Chewie_20150313 \
  --out outputs/runs/adaptive_search/primary_history_control_attempt001/Chewie_20150313 \
  --max-seconds 270
```

## 2026-10-03 23:32 UTC — `adaptive_search`: temporal/control pair complete; primary remains unsupported

The [complete temporal/control summary](runs/adaptive_search/primary_temporal_control_summary_attempt001/primary_temporal_control.json)
finished at 23:30:49.816 UTC in allocation 46438625 on sh03-06n50.int.
The [remaining-panel wrapper](runs/adaptive_search/primary_temporal_control_execution_attempt001/panel_execution.json)
ran 23:29:39.320–23:30:49.908 UTC in **70.588471 seconds**, reusing the
separate 13.517179-second pilot. Every session, held direction and all eight
control seeds completed with the fixed original-ID maps and exact reference
replay. Root released the whole result only after completion. The summary's
0.068172 seconds and 85,664 KiB RSS describe summary generation, not the full
source/fitting panel. Charge two valid comparisons, **eight total**: one
candidate and one alternative-explanation control after the six-recipe
initial coverage (one control of two post-initial comparisons, 50%).

| Comparison | Primary increment | Identity increment | Positive primary sessions (Mihili/Chewie) |
| --- | ---: | ---: | --- |
| Quadratic time-varying LMP ridge | +0.001177898 | +0.003951206 | 1/6; 9/10 |
| Eight-seed training-derangement control | −0.001658523 | +0.000218910 | 0/6; 5/10 |
| Paired minus deranged training | +0.002836421 | +0.003732296 | 6/6; 10/10 |

Every outer fold selected lambda 1 from the other fifteen paired banks and
used that same lambda for all eight control refits. The temporal candidate
misses both **0.01/0.005 practical margins**; its Mihili primary mean is
−0.000424661 versus +0.002780457 in Chewie, so both-animal primary positivity
and the within-animal session-majority requirements also fail. Its identity
means remain positive in both animals (0.002695461/0.005206950).

Relative to the preserved stationary full-channel ridge, the temporal family
loses primary performance in both animals (−0.001439402 Mihili and
−0.000079571 Chewie; balanced difference −0.000759487), while its identity
increment increases by 0.000554881 overall. Thus a stronger pairing readout
does not translate into a better full-velocity prediction here. Do not select
separate models for the two animals or tune polynomial degree from this result.
Stationary full-channel ridge remains the provisional aggregate primary anchor.

The adopted paired-training directional screen passes: paired-minus-control
primary means are +0.001735776/+0.003937066 and identity means are
+0.002764658/+0.004699933 in Mihili/Chewie, with positive primary contrasts
in all sixteen sessions. This supports the usefulness of correctly paired
training supervision for this particular tensor model. It does not rescue
its failed primary criteria, establish biological significance or identify
a shared motor-population mechanism. The control preserves direction/time
mean structure; its score need not be exactly zero. All results remain exposed
exploratory development, with no future-session source or promotion.

The next suggested scientific branch is one prospectively fixed short-history
LMP representation with a matched physical-channel-label control. It would
retain the evaluated 0:+14 endpoint and use only the already qualified wider
source support. This recommendation is **not selected, specified for execution,
implemented or scored**. Root will decide after reviewing its exact history,
normalization and control definition. The 28–64 trial budget, patience and
remaining coverage/control requirements stay open.

## 2026-10-03 23:29 UTC — `adaptive_search`: temporal pilot complete; remaining fixed panel selected

Root and independent static reviews passed. All seven focused temporal/control
fixtures passed in0.334s in allocation46438625; the changed ASTRA program
passed schema/semantic validation, retaining the five pre-existing optional
format warnings. Evidence: `astra/v0.0.14/validation_20261003_primary_temporal_control.json`.
The complete Chewie_20150313 pilot finished23:28:18.375994UTC in13.517179s,
with11.702805s source reading,1.693534s fitting/scoring and273,140KiB peak RSS.
Its exact original baseline/target/trial replay passed. Both recipes and all
eight fixed control seeds completed. No scientific trial is charged for
this single-session bank and its outcome did not select further work.

Select the unchanged remaining fifteen sessions and all-sixteen summary in
`tools/run_temporal_panel.py`, reusing the pilot. Root reviewed the finite
wrapper and its original-ID/direction/seed checks. Use the same1CPU/4GiB
allocation with600s aggregate/620s external bounds; each session keeps270s
internal/300s child bounds, with at least300s remaining before another child.
The13.5s largest-session pilot and preceding109.9s capacity panel support this
bounded scaling. No partial directory is retried or deleted. The fresh
execution record is `runs/adaptive_search/primary_temporal_control_execution_attempt001`;
banks remain under `runs/adaptive_search/primary_temporal_control_attempt001`
and the fresh complete summary goes to
`runs/adaptive_search/primary_temporal_control_summary_attempt001`.
Only a complete comparison adds two valid trials, bringing six to eight.

## 2026-10-03 23:20 UTC — `adaptive_search`: temporal/control pair adopted and implemented; complete pilot selected

Before any new fit, root read and adopted
`primary_temporal_control_proposal.md`, including its paired-training
directional screen. The next two comparisons use a fixed quadratic
elapsed-time tensor of current-bin LMP and its eight-seed within-direction
whole-trial training-derangement control. The 0:+14 evaluated window, native
XY target, baseline, sixteen sessions, trial identities and 0.01/0.005 margins
remain unchanged. No source outcomes informed the new basis or seed scheme.

The implementation is `tools/primary_temporal.py` and
`tools/run_primary_temporal.py`. It reuses the unchanged LMP-only source
reader and exact post-fit baseline/target/trial reference replay. Product
features are centered from permitted fitting trials after the original
channel standardization, without another scaling. One Gram eigendecomposition
per held direction is shared by the paired and eight deranged refits; their
target cross-products remain separate. Every original target/source trial-ID
map is retained. Training and evaluation XY stay unchanged, evaluation
features stay correctly paired, and each outer fold's paired-selected lambda
is shared by all null refits. Eight seed SSEs are averaged; predictions are
never averaged into an ensemble. Existing scientific code and results were
not modified.

Seven focused fixtures cover discrete basis/centering and stationary
embedding, the shared eigensolve against independent refits, complete-trial
derangement invariants, held-direction exclusion, mean-score aggregation,
unchanged inputs/exact constant-feature baseline, and the common lambda's
outer-session exclusion. They are **not yet run**. The changed ASTRA program
contains the adopted pilot and complete-panel recipes and awaits root's
compute validation. Root owns review, testing, source execution and scheduler
actions; the implementation worker stayed on the login node for editing only.

Root selected the complete Chewie_20150313 representative with both arms and
all eight control seeds, 1 CPU / 4 GiB / 5 minutes, one numerical thread,
270-second internal and 300-second external bounds. Run it in root's next
verified bounded substantive allocation alongside the separately scoped
EP08 work, with at least five minutes available before launch. Use measured
cost to select the remaining fifteen-session deadline without examining
pilot scientific scores. No pilot alone adds a valid comparison. The complete
pair would move six to eight valid comparisons, with one of two post-initial
comparisons being a control; full search and other required controls remain.

Exact compute-only handoff after root review:

```bash
PYTHONPATH="$PWD/outputs/tools${PYTHONPATH:+:$PYTHONPATH}" python3.12 -m unittest discover -s outputs/tools -p test_primary_temporal.py -v
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261003_primary_temporal.json
timeout 300s python3.12 outputs/tools/run_primary_temporal.py session \
  --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
  --scalar-bank-root outputs/runs/adaptive_search/primary_reference_attempt001 \
  --reference-bank-root outputs/runs/adaptive_search/primary_multichannel_attempt001 \
  --session Chewie_20150313 \
  --out outputs/runs/adaptive_search/primary_temporal_control_attempt001/Chewie_20150313 \
  --max-seconds 270
```

The complete-summary command is in ASTRA and the runner. It requires both
banks for all sixteen sessions and reuses the preserved stationary LMP
summary as the global anchor. The source-backed
[capacity comparison figure](figures/primary_capacity_attempt001.png)
and [PDF](figures/primary_capacity_attempt001.pdf) were generated separately
by root from the complete capacity summary; no new fits are implied.

## 2026-10-03 23:15 UTC — `adaptive_search`: six complete comparisons; temporal/control pair proposed

Root released the complete capacity panel after observing completion at
23:13:56 UTC. The [summary](runs/adaptive_search/primary_capacity_summary_attempt001/primary_capacity.json)
finished at 23:13:31.145 UTC in allocation 46436630 on sh03-06n50.int.
The [remaining-panel execution record](runs/adaptive_search/primary_capacity_execution_attempt001/panel_execution.json)
ran 23:11:41.364–23:13:31.223 UTC, **109.858901 seconds**, reusing the
separately recorded 43.172851-second complete pilot. Both recipes and all
sixteen sessions completed. The summary's 0.024888 seconds and 51,156 KiB
maximum RSS describe summary generation only. No additional allocated-core
accounting is inferred from these process elapsed times; root owns actual
allocation accounting. Six focused capacity fixtures and ASTRA validation
had passed before the pilot. No partial score selected or abandoned a recipe.
Charge **two new valid comparisons, six total**; pilots, five penalties and
infrastructure tests do not count separately.

| Recipe | Primary increment | Identity increment | Positive primary sessions (Mihili/Chewie) |
| --- | ---: | ---: | --- |
| Rank-one LMP ridge | +0.000987916 | +0.002051556 | 5/6; 9/10 |
| 105-landmark Nyström RBF | +0.001562080 | +0.004254813 | 4/6; 9/10 |

Both miss the adopted **0.01 primary and 0.005 identity margins**. Both
animal primary/identity means are positive, and every session has a positive
identity increment. Rank-one animal primary means are 0.000686365 and
0.001289467; RBF means are 0.000078989 and 0.003045171. All outer folds choose
lambda 1 for rank-one and 0.001 for RBF. Every pooled fixed-prediction
leave-one-session omission remains positive for both readouts; this is
descriptive influence, not completed refitted robustness.

Full-channel stationary ridge remains the largest animal-balanced primary
increment. Rank-one restriction loses primary and identity performance;
the particular finite RBF reference improves identity but loses aggregate
primary performance and leaves almost no average primary gain in Mihili.
These findings motivate a different temporal representation, not a claim
that nonlinear models are universally ineffective or permission to tune
the RBF grid from the observed endpoint. The minimum listed model families
are now covered, but 28–64-trial search, patience, controls and promotion
requirements remain incomplete. No population follow-up or future-session
test is opened; the source remains exposed exploratory development.

The coordinator requested [an exact prospective two-comparison proposal](primary_temporal_control_proposal.md):
one fixed quadratic LMP-by-elapsed-time ridge family and its matched
eight-seed, within-direction whole-trial training-derangement refit control.
Its evaluated 0:+14 window, endpoint, baseline, trials and margins remain
unchanged. The control keeps XY and correct evaluation pairing unchanged,
uses the paired arm's outer-training-selected lambda, and collapses eight
seed-level scores to one comparison. The nested stationary anchor is reused.
The proposal is **not yet adopted, implemented or run**; root will review
its equations, control interpretation and directional screen before adoption.

One coordinator-owned read-only BR `kg_verify_hypothesis` request asked
whether a stationary-versus-time-varying LFP map and trial-pairing control
were scientifically supported. It returned no evidence or verdict because
the Neo4j/Lucene query exceeded `TooManyClauses` (maximum 1024). No retry,
canonical mutation, scientific endorsement or gate follows. Decision impact
is limited to the absence of optional retrieved assistance; the proposed
comparison is motivated by the declared grammar and complete local panel.

## 2026-10-03 23:12 UTC — `adaptive_search`: capacity pilot completed; remaining panel selected

The selected complete Chewie_20150313 two-recipe pilot completed in
43.172851 seconds in attended allocation 46436630 on sh03-06n50.int,
after all six focused capacity fixtures passed (0.172 seconds) and the
pinned ASTRA validator passed. Evidence:
`runs/adaptive_search/primary_capacity_attempt001/Chewie_20150313/score_banks.json`
and `astra/v0.0.14/validation_20261003_primary_capacity.json`.
This is one session bank, not a complete scientific comparison; no new valid
trial is charged yet. Its outcomes did not select any recipe or session.

Select the remaining fixed fifteen sessions and complete-panel summary using
`tools/run_capacity_panel.py`, reusing this pilot and preserving both recipes,
all trials, the five lambdas, and the existing full-LMP anchor. The previous
three-band remaining panel cost 157 seconds; this largest-session capacity
pilot cost 43 seconds. Use one CPU and the existing 4 GiB allocation with a
600-second aggregate bound (620-second external timeout), each new session
retaining its 270-second internal/300-second child bound. The allocation ends
23:23:10 UTC, so this selected finite panel fits the remaining reservation.
The wrapper requires at least 300 seconds remaining before another session,
never retries a failed/partial directory, and charges two comparisons only
after all sixteen banks and the summary complete. Output banks are
`runs/adaptive_search/primary_capacity_attempt001`, the fresh execution record
is `runs/adaptive_search/primary_capacity_execution_attempt001`, and the fresh
summary is `runs/adaptive_search/primary_capacity_summary_attempt001`.
Root reviewed the finite wrapper; no science implementation changed.

## 2026-10-03 22:59 UTC — two capacity recipes adopted; implementation and pilot frozen

Root prospectively adopted `primary_capacity_proposal.md` after interpreting
the completed four-trial panel: one rank-one regularized XY response map and
one105-landmark Nyström-RBF reference on the same global full-channel LMP
anchor. This is explicit development adaptation, not new held evidence. The
same five penalties, outer15 selector, native XY target, baseline, trials,
window, margins and claim boundaries remain unchanged. The fixed1e−8 landmark
regularizer is part of the declared approximation, not an adaptive numerical
repair. Two complete sixteen-session comparisons would add2 valid trials,
bringing4→6; no pilot or test adds a scientific trial.

Before implementation, root clarified the RBF-only training-constant rule:
coordinates whose training SD is zero are set identically zero in **both**
training and evaluation kernel inputs. Their identities/counts are retained.
A held-direction value may vary even if its fitting values were constant, so
the linear zero-SD→1 rule alone would not implement zero kernel distance.
Linear recipes keep their original rule. D_active counts positive training-SD
columns; if zero, the centered kernel is exactly zero and predicts the same
behavior baseline. No threshold or outcome-selected feature deletion is added.

New frozen code is `primary_capacity.py`, `capacity_lmp_reader.py` and
`run_primary_capacity.py`. Six focused fixtures in `test_primary_capacity.py`
cover the augmented-design rank-one optimum, regularized Nyström orientation
with duplicate landmarks, held-direction exclusion from kernel parameters,
constant-column evaluation distance, exact all-constant baseline and outer-
session tuning exclusion. They have not yet run. Root independently reviewed
all four files and six fixtures and reported PASS, with no concrete blocker:
the augmented rank-one metric, Nyström factor orientation, constant-column
kernel handling, LMP-only source boundary and exact post-fit reference replay
were checked. Agent A is no longer live, and no review by that agent is claimed.
The earlier linear/multichannel files remain unchanged.

One guide-qualified LMP/XY source read supplies both new recipes. It does not
reread unused power-band values or create a separate source cache. After all
new session models are fixed, the runner verifies exact baseline, actual XY,
direction and original trial-ID equality against the preserved full-LMP
reference, without giving its scientific scores to the fitter. Complete-panel
summary likewise requires all sixteen complete two-recipe banks.

After that review, root selected the unchanged complete Chewie_20150313
representative at 1 CPU / 4 GiB / 5 minutes, numeric threads 1, a 270-second
internal and 300-second external limit. Root owns execution after the focused
tests and ASTRA validation, in a verified allocation with at least five minutes
remaining. Preserve
the pilot and measure cost before selecting the other fifteen sessions; do
not choose or abandon either recipe from pilot scores. Exact commands:

```bash
PYTHONPATH="$PWD/outputs/tools${PYTHONPATH:+:$PYTHONPATH}" python3.12 -m unittest discover -s outputs/tools -p test_primary_capacity.py -v
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261003_primary_capacity.json
timeout 300s python3.12 outputs/tools/run_primary_capacity.py session \
  --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
  --scalar-bank-root outputs/runs/adaptive_search/primary_reference_attempt001 \
  --reference-bank-root outputs/runs/adaptive_search/primary_multichannel_attempt001 \
  --session Chewie_20150313 \
  --out outputs/runs/adaptive_search/primary_capacity_attempt001/Chewie_20150313 \
  --max-seconds 270
```

The changed ASTRA program is frozen for validation. No capacity test, new
source value or fit was run/read by the worker. Full-summary command, only
after both banks complete for all sixteen fixed sessions:

```bash
python3.12 outputs/tools/run_primary_capacity.py summarize \
  --bank-root outputs/runs/adaptive_search/primary_capacity_attempt001 \
  --reference-summary outputs/runs/adaptive_search/primary_multichannel_summary_attempt001/primary_multichannel.json \
  --out outputs/runs/adaptive_search/primary_capacity_summary_attempt001
```

## 2026-10-03 22:52 UTC — complete multichannel panel: small LMP gain, no margin pass

The remaining-panel wrapper completed in46436630.0 from22:46:21.284 to
22:48:58.389UTC in157.104652seconds. This excludes the separately recorded
46.888856-second pilot, which was reused without refitting. All sixteen
sessions and three recipes completed, with zero automatic retries. The
[complete comparison](runs/adaptive_search/primary_multichannel_summary_attempt001/primary_multichannel.json)
finished22:48:58.303UTC; its0.016801seconds is summary-only cost. The
[execution record](runs/adaptive_search/primary_multichannel_execution_attempt001/panel_execution.json)
records every completed bank and the pilot reuse. Charge3 new valid scientific
trials, bringing the primary-search total to **4**. Root authorized full-result
inspection only after this complete panel; neither pilot scores nor partial
session results selected its scope.

| Fixed feature recipe | Animal-balanced primary increment | Trial-identity increment | Positive primary sessions (Mihili/Chewie) |
| --- | ---: | ---: | --- |
| Full-channel LMP | +0.001937385 | +0.003396325 | 5/6;10/10 |
|0.5–12Hz block | −0.000375444 | +0.000072259 | 2/6;4/10 |
|100–400Hz block | +0.000401027 | +0.000697846 | 6/6;8/10 |

**None reaches the adopted0.01 primary and0.005 identity margins.** The LMP
animal primary means are+0.001014741(Mihili) and+0.002860028(Chewie), and
identity means are+0.002535068/+0.004257581. All sixteen identity increments
are positive. Every fixed-prediction leave-one-session aggregate remains
positive for both readouts, so the small pooled LMP result is not created by
one favorable session. These are descriptive influence omissions, not a
completed refitted robustness program or biological significance test.
All LMP outer folds chose lambda1. The low block chose lambda10 everywhere
and has negative primary means in both animals. High-band primary/identity
means are positive in both animals, but much smaller than the practical
margins; its last two Chewie-R sessions have negative primary increments
and select lambda1, while the other fourteen choose lambda10. No band or
session is removed from reporting.

The selected source diagnostic now establishes that the retained LMP spatial
mean has extremely small varying amplitude: its population-SD ratio to the
root-mean electrode variance ranges7.58238e−17–5.79963e−16 across the sixteen
sessions. Every stored LFP dtype is float64; every spatial mean is nonzero and
nonconstant under exact equality. RMS ratios also vary, reaching3.87390e−13
where a tiny nearly constant offset dominates. Therefore do not conflate the
RMS ratio with the varying component that survives training centering. These
measurements are consistent with spatial cancellation after common-average
referencing. They do not identify the exact sequence of upstream rounding
operations or justify relabeling the scalar trial as invalid. The scalar
pipeline scaled this tiny residual by its nonzero training SD as prescribed;
the failed mean representation remains the first charged result.

**Scientific interpretation:** retaining spatial LMP contrasts yields small
development increments across sessions and a trial-pairing signal, while the
spatial mean is almost cancelled. Neither the LMP nor the tested power blocks
establishes a practically adequate candidate. This is not fresh confirmation,
proof of a motor-population mechanism, raw-LFP/online validity, or completed
search. Keep full-channel LMP as the global provisional development anchor
for the required reduced-rank/nonlinear coverage proposal. Selecting that
anchor after seeing the complete development panel is explicit adaptation;
it does not make its reported score an unbiased estimate of an adaptively
selected final family. No new recipe or score is activated by interpretation.

The existing validated ASTRA `adaptive_search.multichannel_report` describes
this completed output. Its program is unchanged; a new unadopted capacity
proposal is not inserted as an active run. Scalar, all multichannel recipes,
banks and negative results remain preserved. Root separately owns the figure
renderer/output and will supply its completed visual evidence.

## 2026-10-03 22:46 UTC — multichannel pilot complete; remaining fifteen selected

Root reports the first actual source run completed successfully in46436630.0:
22:44:08.847–22:44:55.736UTC,46.888856seconds total,45.214094seconds source
read and392,992KiB maximum RSS. All1,038 frozen trials and86 original LMP
electrodes were retained; selected logical source bytes were64,522,080. The
three fit/score banks took0.221563s(LMP),0.862810s(low) and0.504166s(high).
All three completed banks remain at
`runs/adaptive_search/primary_multichannel_attempt001/Chewie_20150313`.
No pilot scientific scores were examined to choose, abandon or modify a
recipe. Charge remains1 completed scientific trial until the full new panel.

Root selects **one CPU,4GiB,20minutes**,1200-second external and1140-second
internal aggregate bounds, for the remaining fifteen unchanged sessions and
complete fresh summary. Fifteen times the observed large-session cost is
approximately703seconds, leaving time for source-I/O variation and the
wrapper's unchanged300-second remaining-time floor before each new child.
The per-session internal limit remains270seconds. The wrapper passed root's
static review. It reuses the completed pilot, rejects any preexisting partial
session directory without retry, retains each new completed bank and does not
release a comparison summary until all sixteen sessions and three recipes
are complete. No new hardware, scientific window, feature or threshold is
selected by this scaling decision.

Exact selected command in the verified attended allocation, numeric threads1;
selected before launch to follow root's EP08 checks. Root subsequently reports
the exact command started22:46:21UTC in46436630.0. No scientific output was
read at launch; scalar and multichannel implementations remain frozen:

```bash
timeout 1200s python3.12 outputs/tools/run_multichannel_panel.py \
  --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
  --scalar-bank-root outputs/runs/adaptive_search/primary_reference_attempt001 \
  --bank-root outputs/runs/adaptive_search/primary_multichannel_attempt001 \
  --summary-out outputs/runs/adaptive_search/primary_multichannel_summary_attempt001 \
  --out outputs/runs/adaptive_search/primary_multichannel_execution_attempt001 \
  --max-seconds 1140
```

Source, orchestration and ASTRA are frozen. Passing multichannel tests and
the already successful ASTRA validation remain applicable; no scientific
implementation changed after those checks. Session/runtime metadata alone
informed scaling. Full-result interpretation waits for coordinator completion.

## 2026-10-03 22:44:08 UTC — multichannel representative started in fresh allocation

Root verified allocation46436630 on `sh03-06n50.int`, one CPU and4GiB, and
started the unchanged selected Chewie_20150313 three-bank pilot directly in
that attended shell with300-second external/270-second internal bounds.
An initial nested exclusive `srun` step could not start because the attended
shell already held the sole CPU. Root cancelled only that pending step using
Ctrl-C; it created no result path and read no source data. This is execution
routing before the first source run, not a scientific retry or new allocation.
Production source remains frozen until root reports completion.

## 2026-10-03 22:41 UTC — multichannel focused checks and ASTRA passed

Root reports all four multichannel fixtures PASS in allocation46433944 in
0.235s. The frozen program's
[validation](astra/v0.0.14/validation_20261003_primary_multichannel.json)
passed schema and semantics. Five optional future-format warnings concern
unfinished downstream artifacts; they are not execution or scientific failures.
The attended allocation exited cleanly before its limit. No multichannel source
pilot was run there; the selected three-bank pilot awaits root's next bounded
allocation. The four reviewed scientific source/fixture files and ASTRA remain
unchanged after these checks; the execution wrapper below is newly prepared.

`tools/run_multichannel_panel.py` now prepares sequential execution after the
pilot: require the completed selected representative, reuse valid complete
session banks, reject existing failed/partial directories before launching any
missing session, preserve each success, and stop on any failure without retry.
The full three-recipe summary waits for all sixteen sessions. The exact
aggregate deadline/resource selection will be recorded after measured pilot
cost; this wrapper does not select a new scientific or hardware budget.

## 2026-10-03 22:36 UTC — adaptive_search / multichannel recipes adopted; pilot selected

Before any multichannel source read or score, root adopted all three recipes in
`primary_multichannel_proposal.md`: full-channel LMP,0.5–12Hz and100–400Hz, using
the existing LMP physical-channel sets and unchanged primary folds, baseline,
window, five penalties, scoring and margins. Root explicitly selected the
direct standardized-feature ridge solve equivalent to the full unwhitened PCA
span. No numerical PCA is computed, no postrotation scaling or rank cutoff
is introduced, and scalar code/banks/count1 remain unchanged. Independent
proposal and four-file implementation reviews by agentA found no concrete
defect. The implementation review confirmed six-band gather/undo/block order,
guide-before-value access, original channel authentication, full-span ridge,
common intercept and exact baseline equality, and each recipe's outer15
selection. No reviewer tests, source values or new scores were accessed.
Literal `(0.5,4.0)` guide support is documented in the completed
source-support metadata; exact per-channel/per-trial identities remain checked
before values rather than substituting a lower-edge alias.

Root also selected the descriptive stored-dtype and spatial-mean/electrode
RMS/SD ratios during the same authorized feature read. These outputs do not
set channel eligibility, fit preprocessing, a numerical-rank threshold or a
behavior association. No independent diagnostic replay is required.

The separate implementation consists of `primary_multichannel.py`,
`multichannel_reader.py`, `run_primary_multichannel.py` and four focused
fixtures in `test_primary_multichannel.py`. All selected bands across all
original trials are guide-qualified before feature values. The source reader
opens only selected LFP and XY windows, no spike values. All three banks must
complete for the pilot; only the complete16×3 panel can charge3 new scientific
trials. Preserve any failed/partial attempt and report technical cause without
shrinking its scientific scope.

Select the following full Chewie_20150313 pilot with1CPU/4GiB/5minutes,
numeric threads1 and270-second internal bound. Root executes in a suitable
verified allocation; **not yet launched**. If the current allocation cannot
accommodate this bound, keep it prepared for the next substantive allocation.
Use measured pilot cost before selecting the other fifteen sessions; do not
inspect pilot scores to choose or abandon recipes.

```bash
PYTHONPATH="$PWD/outputs/tools${PYTHONPATH:+:$PYTHONPATH}" python3.12 -m unittest discover -s outputs/tools -p test_primary_multichannel.py -v
timeout 300s python3.12 outputs/tools/run_primary_multichannel.py session \
  --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
  --scalar-bank-root outputs/runs/adaptive_search/primary_reference_attempt001 \
  --session Chewie_20150313 \
  --out outputs/runs/adaptive_search/primary_multichannel_attempt001/Chewie_20150313 \
  --max-seconds 270
env -u PYTHONPATH "$EP_AST_BIN" validate outputs/astra/v0.0.14/astra.yaml --skip-evidence --json > outputs/astra/v0.0.14/validation_20261003_primary_multichannel.json
```

The changed ASTRA program is frozen for coordinator validation. No tests,
source values or scores were run/read by the implementation worker. Future
summary command, only after all three banks complete in every fixed session:

```bash
python3.12 outputs/tools/run_primary_multichannel.py summarize \
  --bank-root outputs/runs/adaptive_search/primary_multichannel_attempt001 \
  --out outputs/runs/adaptive_search/primary_multichannel_summary_attempt001
```

## 2026-10-03 22:31 UTC — adaptive_search / first complete primary comparison is negative

Root reports the remaining fifteen session banks and complete summary finished
successfully in `46433944.6`. The preserved
[sixteen-session result](runs/adaptive_search/primary_reference_summary_attempt001/primary_reference.json)
finished at22:26:17.474UTC; summary-only elapsed time is0.012954s and maximum
RSS34,500KiB, not the cost of source processing or all session fits. Reuse the
completed pilot and all sixteen banks under `primary_reference_attempt001`.
Charge **one valid scientific trial** for `ep05_primary_lmp_mean_v1`. The
120-hour primary clock remains anchored at22:23:02.824UTC. No result is dropped
or relabeled as an engineering failure.

Every outer fold selected the strongest already fixed penalty,lambda10. The
animal-balanced primary increment is−8.767985415e−7 R2 and the trial-identity
increment is−2.778050134e−7. Primary means are−1.146734414e−6 for Mihili and
−6.068626688e−7 for Chewie; identity means are−8.316687892e−7 and+2.760587624e−7.
Only1/6 Mihili and5/10 Chewie sessions have positive primary increments. All
five first-reference margin screens fail. Every fixed-prediction leave-one-
session primary and identity summary remains negative; these omissions do not
refit policy selection. The behavior baseline R2 spans0.88030–0.95481.

**Interpretation:** the fixed all-electrode LMP mean does not improve the
adopted out-of-direction residual prediction, and it supplies no aggregate
trial-identity increment. This is a failed first feature summary, not evidence
that all multichannel LFP representations fail. No candidate is promoted. The
minimum model set, remaining28–64-trial program, required falsifiers and future-
session limitation remain open. Retain this negative recipe as the comparison
anchor; do not grow its penalty grid after viewing this result.

The primary paper's [LFP-processing Methods](https://elifesciences.org/articles/73155)
describe common-average de-referencing, session temporal centering and notch
filtering before LMP moving averaging. With the same reference electrode set
and identical linear temporal operations, an across-electrode arithmetic mean
is zero algebraically. This is a plausible limitation of the chosen mean
summary. It does not prove that the released retained-electrode set, including
the86-electrode pilot, sums to zero or consists only of rounding error. Stored
feature magnitudes have not been inspected and current banks report only
exact-zero SD flags. No source-value diagnostic is selected by this entry.
The next [multichannel proposal](primary_multichannel_proposal.md) retains
spatial contrasts and registers low/high band comparisons prospectively;
implementation or scoring awaits the coordinator's selection of that recipe.

## 2026-10-03 22:24 UTC — first primary bank complete; full fixed panel selected

ASTRA `adaptive_search`: the predetermined Chewie_20150313 pilot ran in
`46433944.5` from 22:23:02.824 UTC to 22:23:11.771 UTC and returned zero.
All 1,038 trials and 86 common LMP electrodes were retained. Source preparation
took 8.908443 seconds; the complete session bank took 8.947201 seconds and
reported 77,092 KiB maximum RSS. No pilot score is used to choose or abandon
the recipe. Preserve and reuse this bank. The first primary execution starts
the existing 120-hour search clock at 22:23:02.824 UTC; a complete scientific
comparison has not yet been charged.

Select the other fifteen fixed sessions sequentially in the same allocation,
one CPU, 4 GiB and a ten-minute aggregate step. The predetermined large-session
pilot supports approximately 2.3 minutes for sixteen sessions at its observed
rate; the larger step bound allows source-I/O variation. Each session keeps
its existing 270-second internal limit. Stop on a technical failure and retain
completed banks, then summarize all sixteen only after every bank completes.
No failed session is dropped, no hyperparameter grid grows, and no partial
panel counts as a valid comparison. The benchmark PNG/PDF were also rendered
and visually checked at `figures/publication_comparison_attempt001.*`.

## 2026-10-03T22:20:21Z — publication_reproduction / scoped benchmark accepted

Root and the independent implementation reviewer read the complete frozen
[comparison](runs/publication_reproduction/comparison_attempt001/benchmark_comparison.json),
which completed in allocation46433944 in6.226584s. All16 sessions contribute:
606 electrode/session CCA values per band,1600 split-coordinate decoding
values per representation and144 session-band association points, exactly
matching the scoped publication support. No incomplete or unfavorable session
was omitted; Chewie_20150313 uses the unchanged recovered attempt004.

The association has positive slope1.615416919, R2=0.778990153 and descriptive
F-test p=2.146046206e-48. Its difference from the printed0.78 is−0.001009847;
it falls in the prospectively specified[0.775,0.785) rounding interval and
satisfies the published p<0.001 statement. This F-test reproduces a pooled
publication statistic;144 points are not144 independent animals.

The complete frequency profiles reproduce the scoped broad low/high-frequency
versus middle-band pattern in both animals and both Chewie implants. For
example, Mihili mean decoding is0.69355 for LMP and0.62847 for200–400Hz versus
0.04697/0.02284/0.04093 for8–12/12–25/25–50Hz; Chewie is0.77206/0.72336 versus
0.17738/0.04363/0.09673. These features are visible across session profiles,
not solely in a favorable recording. Retain Chewie's4–8Hz shoulder and raised
8–12Hz CCA relative to its12–50Hz trough; the preregistered descriptive target
does not require a monotone curve or every pairwise band inequality. LMP and
high-frequency decoding are not identical to latent decoding. No numerical
equivalence or “nearly equal” cutoff was invented. Both pooled caption medians
and figure-code means are reported without choosing whichever agrees better.

The already selected10000-draw conditional split-randomness diagnostic has
unshifted difference quantiles[−0.006568443,0.006757642] and shifted interval
[0.772421710,0.785747795], overlapping the publication rounding interval. It is
descriptive compatibility of split randomness, not a biological confidence
interval, equivalence proof, or the basis for the benchmark decision.

**Decision:** complete faithful execution and the scoped M1 frequency/decoding
benchmark are sufficiently established to begin the already adopted primary
recipe. Root explicitly accepted this result; the independent reviewer found
no concrete material mismatch under the frozen comparison. This is not whole-
article reproduction, a primary LFP efficacy result, fresh confirmation or a
candidate promotion. Original paper split draws are unarchived, the historical
TrialData compatibility revision is not authenticated as the original runtime,
and the caption-median/code-mean discrepancy remains explicit. All-trial spike
PCA, reference correlation-squared scoring, and the pinned final-coordinate-only
cubic mapping are preserved publication semantics; the primary uses its own
training-only residual/SSE recipe and does not inherit these transformations.

## 2026-10-03T22:20:21Z — adaptive_search / first primary pilot selected, not launched

The benchmark gate is now satisfied by the decision above. Root reports all
three focused primary checks PASS in allocation46433944 in0.081s. Select the
previously prepared Chewie_20150313 full-session score-bank pilot,1CPU/4GiB,
all numeric threads1,270-second internal deadline and a5-minute bounded step,
after the currently running EP08 work if the attended allocation has enough
time. No launch or primary score is claimed by this entry; root owns the
execution and will provide its job-step ID and actual outcome.

```bash
timeout 300s python3.12 outputs/tools/run_primary_reference.py session \
  --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
  --session Chewie_20150313 \
  --out outputs/runs/adaptive_search/primary_reference_attempt001/Chewie_20150313 \
  --max-seconds 270
```

Reuse a completed pilot bank in the fixed sixteen-session comparison; do not
select a winner, abandon an unfavorable recipe or charge a complete valid
trial from this one session. Actual first-fit timing will start the existing
primary wall-clock accounting. Remaining-session cost is sized from observed
pilot time, not an assumed reproduction runtime. The ASTRA program is unchanged
during root's current validation; its existing `benchmark_report` is the direct
scientific evidence artifact, with this log recording the contract decision.
Root subsequently reported schema/semantic validation PASS(exit0) in46433944:
`astra/v0.0.14/validation_20261003_primary_reference.json`. No ASTRA edit
followed that validation. The attended allocation ends22:40:51UTC; if the
pilot's complete bounded step no longer fits after EP08, leave it selected
for later coordinated compute rather than starting a truncated attempt.

## 2026-10-03 22:19 UTC — all sixteen reference sessions completed

ASTRA `publication_reproduction`: recovery step `46433944.1` returned zero.
Chewie_20150313 attempt004 completed at 22:18:28 UTC after 939.163313 seconds,
retaining the original 1,038 trials and seed 20261015. This restores the full
sixteen-session panel; numerical benchmark results have not yet been inspected.
Select the frozen summarizer now in the same attended allocation, one CPU,
2 GiB and at most ten minutes, with the exact single-session recovery override
and the already selected `--split-bootstrap` diagnostic. Output is
`runs/publication_reproduction/comparison_attempt001/benchmark_comparison.json`.
Interpretation of this complete result, rather than process completion, still
determines whether primary execution can begin.

## 2026-10-03T22:14:53Z — adaptive_search / fixed reference implementation prepared, no source launch

Prepared `tools/primary_reference.py`, `tools/run_primary_reference.py` and
`tools/test_primary_reference.py` for the adopted recipe. Reproduction source
and running driver are untouched. The session reader first verifies the exact
qualified trial identities/events and determines common LMP electrodes from
guide metadata. It then decodes only those LMP channels and velocity in0:+14,
without spike values or other LFP bands. Every session retains all of its
qualified whole trials; support/finite failures stop rather than shrink it.
HDF5 can internally decompress chunks containing other elements, but only
selected values are returned to Python.

Each bank computes eight shared baselines and forty ridge fits, saves all
complete trial predictions, and prints only completion/cost metadata. The
sixteen-bank summary requires every fixed session and uses only the fifteen
other session rows to choose each outer lambda. The baseline includes the
same residual intercept as the candidate. Exact wrong-trial SSE, unchanged
native-coordinate denominator, animal balance and adopted margin screens are
implemented. Complete-panel aggregation charges one valid trial and always
reports promotion/search incomplete; no pilot chooses a winner.

Three generated fixtures cover outer-row exclusion, zero-LMP equality to the
fair comparator under unequal direction counts, and exact within-direction
whole-trajectory wrong-pair SSE. Selected generated-only checks, for root's
next available compute step, require no benchmark/source outcomes:

```bash
PYTHONPATH="$PWD/outputs/tools${PYTHONPATH:+:$PYTHONPATH}" python3.12 -m unittest discover \
  -s outputs/tools -p test_primary_reference.py -v
```

The following substantive pilot is prepared **only after root explicitly
accepts the complete publication benchmark**. It is not selected for immediate
execution by implementation readiness or the synthetic tests:

```bash
timeout 300s python3.12 outputs/tools/run_primary_reference.py session \
  --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
  --session Chewie_20150313 \
  --out outputs/runs/adaptive_search/primary_reference_attempt001/Chewie_20150313 \
  --max-seconds 270
```

Use1CPU/4GiB, NumPy/h5py under Python3.12, all numeric thread variables1,
and a5-minute bounded step inside a substantive allocation. This predetermined
1038-trial session measures source-I/O and forty univariate-fit cost; it is
part of the same fixed comparison. Preserve/reuse its complete bank, then size
the remaining fifteen from the observed cost. Memory is expected below4GiB
because one session's LMP mean, velocity and five prediction arrays are held,
with no full native-band tensor or source copy. Source latency is unmeasured;
neither the estimate nor an unfavorable score authorizes dropping a session.
No source/test code has run in the implementation worker's login shell.

AgentA independently reviewed the three frozen files without tests, source
values or benchmark scores and found no concrete launch-blocking defect. It
confirmed the zero-based event slice, guide-first LMP intersection, absence
of spike-value access, common intercept, normalized lambda, complete-panel
outer exclusion/animal weights, exact whole-trial pairing and finite
derangement-support disclosure. This technical review does not open the
benchmark gate. The unittest command preserves Sherlock's existing PYTHONPATH
while prepending the episode's tool directory.

Once all sixteen exact session banks exist under that same root, the prepared
summary command is:

```bash
python3.12 outputs/tools/run_primary_reference.py summarize \
  --bank-root outputs/runs/adaptive_search/primary_reference_attempt001 \
  --out outputs/runs/adaptive_search/primary_reference_summary001
```

## 2026-10-03T22:12:25Z — adaptive_search / first recipe adopted; benchmark gate remains closed

The coordinator prospectively adopted the complete
[primary reference recipe](primary_reference_proposal.md) under existing
operational authority: fixed0:+14 window, first-harmonic canonical-angle
template, common residual intercept in both comparator and candidate,
all-common-LMP mean, normalized five-lambda ridge, sixteen outer singleton
session folds, exact whole-trial wrong-pair mean, pooled X/Y SSE and equal
session-then-animal aggregation. Aggregate primary increment must be >=0.01R2
and identity increment >=0.005R2, with strictly positive summaries in both
animals and strict primary-positive session majorities(4/6 Mihili,6/10 Chewie).
All other required falsifiers/influence definitions and results remain necessary
for promotion. This single recipe, once complete, counts as one valid scientific
trial; it does not complete the minimum model set or search.

The benchmark must first complete and the coordinator must explicitly record
its acceptance before **any** primary source execution or scientific score.
Implementation and generated-data checks may be prepared now. No benchmark
numerical scores informed this adoption, and the implementation worker has
not inspected them. Existing source/reproduction code remains unchanged.

Independent agentA reviewed the equations/nesting and found no concrete
conflict. Root also requested read-only BR `pipeline_plan_review(use_kg=true)`
with the full method summary(window, common intercept, mean-loss ridge,
outer-session exclusion, derangement mean, animal weights and margins). It
returned approve/low/no findings, but `kg_rules_consulted=[]` and its checklist
only noted four steps and absent registration. This generic advisory supplied
no substantive LFP/nesting evidence, no canonical approval and no added gate;
the tailored independent equations review is the useful technical review.

## 2026-10-03T22:12:25Z — publication_reproduction / frozen split-randomness diagnostic selected

Before numerical benchmark inspection, root selected the existing optional
10,000 paired-bank split diagnostic exactly as defined in
[publication_comparison.md](publication_comparison.md):PCG64 seed20261003,
within-session/band split resampling with X/Y kept together, fixed CCA values,
and the specified linear-quantile difference interval. It is descriptive
conditional Monte Carlo compatibility, not a biological interval, equivalence
test, benchmark pass or primary authorization. The summary recipe now includes
`--split-bootstrap`; full sixteen-session completeness is still required.

```bash
python3 outputs/tools/summarize_publication_reproduction.py \
  --first-session outputs/runs/publication_reproduction/Mihili_20140217_attempt002 \
  --remaining-root outputs/runs/publication_reproduction/remaining_sessions_attempt003 \
  --chewie-20150313-recovery outputs/runs/publication_reproduction/remaining_sessions_attempt004/Chewie_20150313 \
  --out outputs/runs/publication_reproduction/comparison_attempt001 \
  --split-bootstrap
```

## 2026-10-03 21:59 UTC — publication_reproduction / attended recovery selected

Final task5 accounting is COMPLETED at21:56:08UTC,20minutes, exit0. Its
three sessions are saved, leaving15 of16 complete and only Chewie_20150313
unfinished. Independent static review of attempt004 and the exact-session
summary override found no blocker.

To group this substantive recovery with ready EP04 embedding preparation and
EP08 checks, the coordinator selects one attended normal allocation with one
CPU,8GiB and40minutes. This supersedes standalone sbatch submission: do not
submit a duplicate recovery job. After focused checks, execute the unchanged
attempt004 shell wrapper with bash in an explicit one-CPU Slurm step, preserving
its fixed source/seed/driver and1500second MATLAB bound. The scheduler comments
in that file do not create another job when invoked with bash. Each short
episode stage retains its own output paths and bound, with no simultaneous CPU
work. No EP19 or EP03 stopped source route is retried in this allocation.

Slurm returned allocation46433944; the attended shell was verified on
sh03-02n16.int at22:01UTC with SLURM_JOB_ID46433944, one task CPU and8192MiB.
The requested40minute allocation is finite. Run the recovery before the shorter
prepared stages so its25minute MATLAB bound fits comfortably; no duplicate
standalone submission is made.

Shell syntax and both summary fixtures passed (0.001s). The updated EP05
ASTRA validated successfully. At22:02UTC the coordinator launched attempt004
in allocation46433944, step1, with one CPU,8GiB and a30minute step bound;
the unchanged MATLAB subprocess remains limited to1500seconds. The attended
shell is idle during this sole numerical worker. Logs are
`publication_reproduction_attempt004_46433944.out` and `.err`.

## 2026-10-03 21:55 UTC observation — publication_reproduction / attempt004 recovery selected, unrun

The coordinator's accounting reports task46431719_4 TIMEOUT at30:22 with
batch signal15. Its Chewie_20161014 and Chewie_20161021 reports were completed
and saved; only Chewie_20150313 was unfinished. Task5 was still processing
Chewie_20150630 at the last coordinator observation, with its preceding
Chewie_20150319 and Chewie_20150629 reports complete. Root owns further
monitoring; the worker queried no scheduler and opened no result MAT arrays.
The timeout is a grouped-allocation limit, not a scientific disagreement or
evidence that the individual1500second MATLAB bound needs revision.

Root selected the single-session recovery wrapper
`tools/publication_reproduction_attempt004.sbatch`: one requested CPU,
8 GiB,40minutes on the existing normal partition, no array, no requeue and
no retry loop. It runs only Chewie_20150313, preserving every1038 qualified
trial in the original table. The unchanged runner derives seed20261015 from
fixed session index12. Analysis revision
`cbda8e2e6106f5eb5ff98e18a689c595179ac5db`, TrialData compatibility revision
`d5e5eeb1af592cf88c03599df7433878f9d11bbe`, MATLABR2022b, driver,
19-bin paper window,50 splits per representation and1500second MATLAB
subprocess timeout are unchanged. No scientific trial or seed search occurs.
The individual job receives the whole40minute envelope instead of the time
left after two earlier sessions in a30minute grouped allocation; the40minute
limit does not promise that the unchanged MATLAB process will finish.

Fresh work directory:
`$SCRATCH/br_autoresearch/episode05_sensorimotor_lfp/remaining_sessions_attempt004/Chewie_20150313`.
Fresh durable output:
`runs/publication_reproduction/remaining_sessions_attempt004/Chewie_20150313`.
Both must be absent at launch. All attempt003 artifacts, including partial
Chewie_20150313 output, are preserved. No input/result MAT array was needed
for this preparation. No worker-side test, build, source execution or
submission ran. The coordinator reviews and submits the exact wrapper.

The summary reader adds only `--chewie-20150313-recovery <directory>`.
Only that exact session uses the override; completed first-session and
remaining attempt003 directories retain their existing mapping. Every
session still passes the unchanged completion, identity, seed, revision,
fixture, band, coefficient and split checks. The output records each actual
input directory. A focused fixture checks that exactly contract index12 can
change under the override. The benchmark still requires all16 sessions and
cannot become a successful subset. Prepared eventual command:

```bash
python3 outputs/tools/summarize_publication_reproduction.py \
  --first-session outputs/runs/publication_reproduction/Mihili_20140217_attempt002 \
  --remaining-root outputs/runs/publication_reproduction/remaining_sessions_attempt003 \
  --chewie-20150313-recovery outputs/runs/publication_reproduction/remaining_sessions_attempt004/Chewie_20150313 \
  --out outputs/runs/publication_reproduction/comparison_attempt001
```

The updated ASTRA recipes await coordinator validation in a suitable compute
allocation; prior validation does not cover this recovery edit. The fixed
publication comparison and primary-search boundary are unchanged.

## 2026-10-03 21:21 UTC — remaining reference sessions submitted

After compute-node shell syntax and ASTRA validation passed in allocation
46430982, root submitted `tools/publication_reproduction_attempt003.sbatch`
as array **46431719**, tasks 1–5 with concurrency 4. Each task processes its
three fixed remaining sessions sequentially, with 8 GiB, 30 minutes, one
numerical thread and no requeue. The completed Mihili 2014-02-17 result is
reused; no duplicate session was submitted. The comparison in
`publication_comparison.md` was frozen before inspecting numerical reference
outcomes. Validation: `astra/v0.0.14/validation_20261003_remaining_array.json`.
The array is execution in progress, not a publication-reproduction verdict
or authorization for the primary adaptive search.

## 2026-10-03 — source_support — attempt 002 completed with both windows

The coordinator completed the separately selected attempt 002 in allocation
**46425841**. The actual report explicitly distinguishes the paper/figure
movement-onset offsets **-4 through +14 inclusive (19 native samples)** from
the Main-example offsets **0 through +15 inclusive (16 native samples)**.
All **sixteen M1 sessions** have full source-trial support under both windows,
with no support failures. Both windows support 208, 225, 208, 203, 217 and 216
trials for the six Mihili dates; 208, 202, 209, 168, 740 and 286 for the six
Chewie-L dates; and 1,038, 1,026, 179 and 178 for the four Chewie-R dates, in
the contract's date order. This includes Mihili 2014-03-06 without imposing
the separate native-integer-spike gate on the kinematic endpoint.

Evidence: `runs/source_support/attempt002/source_support.json`,
`session_table.csv` and `trial_identity.csv`. The same 25 source objects and
9,571,571,204 bytes were observed, with no missing or unexpected filenames.
The original source and attempt 001 remain unchanged; attempt 001's historical
`publication_*` labels still mean the Main-example window only.

Official ASTRA schema and semantic revalidation passed for the attempt-002
recipe; `astra/v0.0.14/validation_20261003_source_attempt002.json` retains six
optional-format warnings. The unchanged method-component tests were not
repeated. No empirical model was fitted or scored, no trial groups/session
folds were frozen, and neither window was selected as the primary scientific
window. The next substantive stage remains the publication-reproduction and
prospective method specifications described below; this source-support result
does not establish publication fidelity or activate adaptive search.

## 2026-10-03 — source_support / publication_reproduction — attempt 001 completed

Coordinator allocation **46425841** completed the source adapter and all seven
method-component checks. Attempt 001 loaded the **original adapter**: its
`publication_window` string and all `publication_*` column names refer to the
**Main-example 0:+15-tick window**, not the paper figure's -4:+14-tick window.
Those historical artifacts are preserved unchanged; their names must not be
used to claim paper-window qualification. The later source-code edit did not
affect the code already loaded by that process.

The source contains exactly **25 objects** (24 session MAT files plus README),
**9,571,571,204 bytes**, with no missing or unexpected filenames. All sixteen
M1 sessions passed the Main-example structural/finite-window support check,
including Mihili 2014-03-06. Evidence:
`runs/source_support/attempt001/source_support.json`, `session_table.csv`, and
`trial_identity.csv`. These are exposed-corpus identity/support records, not
primary eligibility/fold freezing or model-comparison results.

The synthetic method checks passed: cyclic held-direction maximum error
1.33e-15; linear trial-specific known-answer R2_SSE 1.0; fixed wrong-trial
pairing -0.48156; evaluated-direction fitting/scaling isolation, negative
R-squared retention and noncontiguous electrode joining all passed. Evidence:
`runs/source_support/attempt001/method_component_checks.json`. These values
are synthetic fixtures. No empirical model was fitted or scored, and
publication fidelity remains unestablished.

The official ASTRA schema/semantic validator passed in the same allocation,
with six optional-format warnings; report
`astra/v0.0.14/validation_20261003_job46425841.json`. A later recipe-only edit
points the next source attempt to attempt 002; that changed object still needs
its ordinary validation, which is not a source-work gate.

**Selected next command**, with both windows explicitly described by the
updated adapter: `python3 outputs/tools/source_adapter.py --source
"$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source"
--out outputs/runs/source_support/attempt002`. Run once in coordinator compute,
1 CPU/4 GiB/10 minutes. It qualifies the paper -4:+14 window and separately
reports Main-example 0:+15 support. The original attempt and source are
preserved. No repeat of the passing method-component checks is needed because
those method files have not changed. Primary scientific choices remain
unactivated; no search begins from this source check.

## 2026-10-03 — source_support — original-source adapter selected

The scientist authorized resuming available work toward completing the episode
questions. The next EP05 stage is source qualification and fixed publication
reproduction, before adaptive model scoring. The entire historical release is
already outcome-exposed exploratory development evidence. No future-session
source is available. No historical or sibling scientific artifact is a run
input; this action reads the original provider release directly from
`$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source`.

`tools/source_adapter.py` implements the actual MATLAB-v7.3/HDF5 field layout,
stable original trial IDs, animal/session grouping and guide-based channel/unit
joins. The first command inspects the expected 24 session files plus README
by filename/size and qualifies all sixteen M1 sessions using successful reaches,
identity/events, common LFP/velocity native time axes, guide consistency and
finite publication-window availability. It emits only source/support metadata
and readable trial/session tables; it does not fit a model or emit scores.
The output does not freeze the primary window, eligibility threshold or folds.

Prepared command: `python3 outputs/tools/source_adapter.py --source
"$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source"
--out outputs/runs/source_support/attempt001`. The coordinator runs it in a
bounded compute step, proposed 1 CPU/4 GiB/10 minutes with NumPy and h5py.
The editing shell is login sh04-ln04 without SLURM_JOB_ID; no source payload
or test ran there. No code was submitted by this worker.

The adapter intentionally does not import EP07's native-integer-spike gate.
EP05's primary target is X/Y velocity, so a noninteger spike representation is
not a reason to drop a kinematically supported session. The reported original-
source issue on Mihili 2014-03-06 is relevant to interpreting the separate
spike benchmark/follow-up and must remain documented there; no silent rounding
or inverse transform is introduced. Chewie-L and Chewie-R stay nested within
one animal; implant assignments currently come from the contract's fixed dates,
not an assumed hemisphere field.

## 2026-10-03 — publication_reproduction — pinned method boundary

The provider analysis repository is pinned to
[`cbda8e2e6106f5eb5ff98e18a689c595179ac5db`](https://github.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/commit/cbda8e2e6106f5eb5ff98e18a689c595179ac5db),
dated July 18, 2022. A subsequent direct Methods/figure-script check established
two different windows at this same revision. `Main.m` calls `trim_data_v2.m`,
movement onset through +15 native ticks (16 samples). The paper Methods and
`Generate_all_figures.m` instead use `trim_data.m`, offsets **-4 through +14**
(-120 through +420 ms, 19 samples). The adapter now qualifies the paper window
and separately reports Main-example support; it never silently adopts either
as the EP05 primary window. `ComputeVelModel.m` uses 90/10 whole-trial train/test splits, a three-
past-bin input branch, a linear Wiener decoder with cubic output mapping,
and publication correlation-squared/VAF readouts. These do not replace EP05's
full-velocity SSE-based R-squared objective. The LFP/population example applies
square-root and 50-ms smoothing before spike PCA and CCA.

Sources: [paper Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC9470163/),
[`trim_data.m`](https://raw.githubusercontent.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/cbda8e2e6106f5eb5ff98e18a689c595179ac5db/MainAnalysis/trim_data.m),
and [`Generate_all_figures.m`](https://raw.githubusercontent.com/BeNeuroLab/Relationship_between_latent_dynamics_and_LFP/cbda8e2e6106f5eb5ff98e18a689c595179ac5db/Generate_all_figures.m).
The figure script uses 50 velocity splits per session, whereas the Main example
uses five. Figure-level reproduction must use the former; the three past-bin
shift reduces its 19-sample velocity window to 16 usable samples. This source
distinction was resolved before any benchmark score was computed.

The linked TrialData dependency does not specify an author-used commit. A
historical compatibility candidate is
`d5e5eeb1af592cf88c03599df7433878f9d11bbe` (April 13, 2022), the latest commit
listed before the analysis-source date; this is not established paper-runtime
provenance. GitHub reports no repository-wide license for either repository;
the analysis repository's nested TME optimizer license does not establish a
license for the whole analysis. Provider functions have not been copied into
this episode or rewritten as an authenticated reproduction. Compatibility,
dependency semantics, reproduction targets/tolerances and permitted runtime
must be settled before any benchmark claim or adaptive candidate score.

The main unresolved scientific specifications for later search are the primary
time window, cyclic behavioral basis/support and interpolation rule, within-
animal aggregation, minimum practical gain, session-majority rule, pairing-null
count/center/margin and invalid-cell rule, finite feature/capacity schedules and
the nested session policy-selection procedure. Source qualification proceeds
independently; these choices cannot be set from new model scores.

An inspected dependency detail must remain explicit: the historical
[`getModel.m`](https://raw.githubusercontent.com/NeuralAnalysis/TrialData/d5e5eeb1af592cf88c03599df7433878f9d11bbe/Analysis/getModel.m)
fits polynomial maps for both velocity coordinates, but its prediction code
applies `polyval` after the `iVar` loop, to only the final coordinate. This
is a source-semantic observation, not an empirical result. A corrected two-
coordinate implementation would be a separately labelled compatibility
sensitivity; it cannot silently replace an exact pinned-code benchmark.

The episode's own `tools/method_components.py` is now an executable building
block: cyclic direction-by-native-time baseline, training-only standardization
and ridge residual prediction, complete-trial leave-one-direction-out folds,
joint X/Y SSE-based R-squared, and physical-guide electrode joins. It does
not import publication code or any sibling artifact. Harmonic order, ridge
penalty and minimum direction support are explicit caller inputs; these
components neither choose a primary recipe nor replace the required broader
model grammar/whole-session outer search. The implemented baseline weighting
is equal across training directions and remains a proposed method choice.

Selected component check: `python3 outputs/tools/qualify_method_components.py
--out outputs/runs/source_support/attempt001/method_component_checks.json`,
bounded to 1 CPU/4 GiB/5 minutes in coordinator compute. Its synthetic cases
recover a cyclic held-direction known answer, recover trial-varying residual
prediction, verify no fitting/scaling from the evaluated direction, demonstrate
fixed whole-trial mismatching sensitivity, retain negative R-squared and verify
noncontiguous electrode joins. Their synthetic settings are fixtures, not the
selected scientific margins/support or a publication-fidelity result. No real
source is read by this command and it does not count as a scientific trial.

## 2026-09-26 — science-first design revision

The episode was rewritten around one concrete question: whether the LFP from
the correct trial predicts how that reach differs from the movement expected
from its target direction and timing.

The revision adds a schematic of the scientific logic, makes the wrong-trial
pairing comparison explicit, and links a positive kinematic result to the next
test involving simultaneously recorded spike-population activity. It also
states what negative, direction-level, high-frequency-only, and
session-specific results would mean.

A previous engineering-only prototype was removed from the active episode
because it did not answer this scientific question. No Dryad object was opened,
no publication analysis was run, and no real-data score was computed during
this revision.

## 2026-09-30 — imagegen concept figure

Added `ep05_question_imagegen.png` and its exact generation/correction prompts.
Visual review corrected the independent baseline/LFP branches and crossed the
wrong-trial predictions to the other behavioral trials. The earlier SVG is
preserved. This changed presentation only: no model, policy, source access,
job, or outcome computation was added.

## 2026-10-02 — Scope and novelty retained

Checked source-study, stable-LFP-control, and population-stability overlap;
retained the trial-specific kinematic question and conditional population chain.
Added [scope review (consolidated)](paper_plan.md) and corrected the paper
plan's model count from four to five without adding a model. No policy, data
role, threshold, or next scientific action changed. No neural payload or score
was read; the ASTRA program is absent and was not made a new gate. Existing
status notes remain accurate and unchanged.

## 2026-10-03 — publication_reproduction: direct reference session selected

The next substantive stage is the unchanged publication `fCCA` and
`vel_decoder` code at analysis revision
`cbda8e2e6106f5eb5ff98e18a689c595179ac5db`, with the explicitly historical
TrialData compatibility revision `d5e5eeb1af592cf88c03599df7433878f9d11bbe`.
This dependency revision is not established author-runtime provenance.
The existing licensing observation (no repository-wide license identified)
remains explicit; no broader redistribution grant is claimed. Private source
acquisition and reference execution are part of the authorized reproduction.
MATLAB module `math matlab/R2022b` resolves to the site installation; only
module discovery was performed on the login node. [Sherlock MATLAB guidance](https://www.sherlock.stanford.edu/docs/software/using/matlab/)
shows ordinary Slurm execution and no separate license-resource token.

**Selected scope before any new empirical score:** all 208 qualified successful
whole trials in Mihili_20140217, as already recorded in source-support attempt
002. DATASETS assigns all eligible release sessions to exposed infrastructure
reproduction/development; its only future-session role is absent. This does
not consume an EP05 future holdout or import EP07/EP08 artifacts. Only M1 LFP,
M1 spikes and XY velocity from the 19-bin paper window, plus their existing
identities/guides, are read. A minimal transient MAT is necessary to deliver
these selected windows to the original MATLAB functions; the original source
is read-only, and no complete source copy or other-array payload is made.
No integer-count requirement or reverse transformation is imposed on the
released spike representation. Negative/nonfinite spike values would instead
produce a technical failure before the reference square root; no trial drops.

The runner preserves paper offsets -4:+14, sqrt-spike transformation, 50-ms
Gaussian smoothing, all-trial rank-10 PCA, scalar LFP/population CCA, and all
50 independent 90/10 whole-trial splits for each of nine bands and the latent
benchmark (500 fits). Three past bins leave 16 predicted samples. The original
last-coordinate-only cubic prediction is retained. CCA has no TME surrogate
branch, matching the pooled figure entrypoint. The source's channel-ID-as-band-
position indexing must agree with actual physical identities, or the run fails
rather than silently correcting it. These all-trial transforms and correlation-
squared scores are publication semantics, not leakage-safe EP05 primary scores.

[TrialData smoothing](https://raw.githubusercontent.com/NeuralAnalysis/TrialData/d5e5eeb1af592cf88c03599df7433878f9d11bbe/util/smooth_data.m)
uses a finite Gaussian kernel with half width `ceil(3*SD/dt)` and divides the
zero-padded convolution by the available kernel mass. The MATLAB fixtures
independently verify this boundary on an asymmetric impulse, three past-bin
alignment, the Y-only polynomial quirk, and correlation-squared semantics.
Their fixed absolute numerical tolerance is `1e-10`; it is a compatibility
check, not a tolerance for judging empirical agreement with the paper. The
seven earlier EP05 component checks and complete source-window checks are
reused. No new generic qualification sequence is added.

The first run sets MATLAB Twister seed 20261003 and retains all 500 source-trial
permutations, initial/final RNG states, CCA coefficients, XY predictions and
scores. Original paper random draws were not archived; byte-identical original
splits are not claimed. The replayed RNG state must equal the state following
the unchanged reference decoder; an unaccounted random draw fails the attempt.
Subsequent predefined sessions use 20261003 plus their index in the existing
sixteen-session contract order. No seed is selected from outcomes.

**Compute handoff, not yet launched:** 1 CPU, 8 GiB, 30 minutes, zero GPU;
MATLAB `-singleCompThread` and BLAS/OpenMP thread limits one, with a 1500-second
MATLAB subprocess bound. Source transfer consists of two small pinned code
archives in the attempt's Scratch directory, followed by selected-window reads.
Root owns the allocation, launch, observed cost and any subsequent scaling.

```bash
module load math devel matlab/R2022b
module load py-scikit-learn/1.5.1_py312 py-h5py/3.10.0_py312
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 outputs/tools/run_publication_reproduction.py \
  --source "$OAK/data/br_autoresearch_data/sensorimotor_lfp/dryad-xd2547dkt-v5-payload/source" \
  --trial-table outputs/runs/source_support/attempt002/trial_identity.csv \
  --session Mihili_20140217 \
  --work "$SCRATCH/br_autoresearch/episode05_sensorimotor_lfp/reference_Mihili_20140217_attempt001" \
  --out outputs/runs/publication_reproduction/Mihili_20140217_attempt001 \
  --matlab /share/software/user/restricted/matlab/R2022b/bin/matlab
```

This representative is a complete one-session reference analysis, not a reduced
split pilot and not a full-paper conclusion. A completed execution records
`reference_session_completed`; full sixteen-session comparison and a declared
reproduced/material-discrepancy/technical-inability conclusion still precede
any adaptive candidate score. The published pooled CCA/decoding profile is the
benchmark target, including the reported band/session association, not a new
primary endpoint. The [paper's original methods and figure descriptions](https://pmc.ncbi.nlm.nih.gov/articles/PMC9470163/)
remain the external comparison. No empirical acceptance margin is chosen from
this first session, and no completed session alone opens adaptive search.
The later primary cyclic basis/window, practical improvement and pairing-null
margins, nested folds and finite model schedules remain prospective choices;
they do not prevent the fixed reproduction above. ASTRA now contains this
recipe; its changed object awaits root's next compute validation.

The independent source review found no launch-blocking defect in the selected
reference route. It checked the selective MAT export, exact pinned helper
semantics, all 500 fit/permutation positions and XY summary orientation.
The matched-channel guard now explicitly requires finite positive one-based
integers, preventing Python negative indexing from accepting channel zero.
The unchanged original functions remain untouched. The bounded coordinator
submission wrapper is `tools/publication_reproduction_attempt001.sbatch`:
normal, one CPU, 8 GiB, 30 minutes, with fixed attempt-001 paths and
`outputs/publication_reproduction_attempt001_%j.{out,err}` logs. It prints
host/job/allocation, rejects login execution or existing attempt directories,
and keeps caches, MATLAB preferences and temporary files in job-local/Scratch
storage. No submission or shell test was performed by the implementation agent.

### Reference attempt 001 submitted

The coordinator submitted the exact bounded wrapper at **2026-10-03 20:55:37
UTC** and received Slurm job **46430433**. This is the selected
`publication_reproduction` Mihili_20140217 attempt 001, not a completion or
an adaptive trial. Root owns scheduler monitoring and any diagnosis. Durable
results will be under `runs/publication_reproduction/Mihili_20140217_attempt001`;
Slurm logs are `publication_reproduction_attempt001_46430433.out` and `.err`.
No empirical result has yet been reported to this agent.
## 2026-10-03 20:59 UTC — publication reference wrapper repair selected

Job `46430433` failed after 10 seconds on `sh02-01n45.int` with exit 2.
Slurm supplied `SLURM_CPUS_PER_TASK=2` despite the one-CPU request; the wrapper's
exactly-one-CPU guard stopped before source acquisition, attempt directories,
MATLAB or scientific data access. Charge the actual two allocated CPUs:
20 core-seconds. The original script and stdout/stderr are preserved.

Selected engineering retry before submission:
`tools/publication_reproduction_attempt002.sbatch`, same one-CPU request,
8 GiB and 30-minute bound, now accepts any positive allocated CPU count while
all BLAS/OpenMP limits and MATLAB `-singleCompThread` remain one. Actual Slurm
allocation will be charged. Only the wrapper guard and fresh attempt002 paths
change; the complete 208-trial / 500-fit scientific recipe is unchanged.
This is no additional scientific evaluation and opens no primary search.

Submission returned job `46430580`; pending/running status and terminal
execution remain to be observed by the coordinator.

## 2026-10-03 — publication_reproduction: scoped comparison frozen; remaining sessions selected

Before the implementation agent inspected any numerical run result, the
coordinator reported that job **46430580** completed the full first-session
reference: 208 trials, all fixed MATLAB fixtures passed, CCA 1.437 seconds,
decoder 149.794 seconds, runner total 239.611 seconds. Slurm elapsed time was
254 seconds with **two actual allocated CPUs**, or 508 core-seconds;
batch MaxRSS was 2,016,100 KiB. These are technical/resource observations only.
The numerical CCA, predictions and scores remain unopened by this agent.

The coordinator adopted [the fixed M1 comparison](publication_comparison.md)
before numerical inspection. It preserves exact published support checks of
606 CCA values per band, 1,600 XY/split values per representation, and 144
session/band association points. Both caption medians and literal-code means
will be reported: this ambiguity was found in the published source, not from
our results. The Figure 4D slope, F-test p < .001 and printed R-squared
rounding interval [.775,.785) are explicit checks. The publication does not
uniquely imply per-band inequalities or a meaningful “near latent” margin,
so the complete frequency profile is compared descriptively, without invented
cutoffs. The optional fixed 10,000 paired-XY split bootstrap is descriptive
Monte Carlo compatibility only; a wide interval cannot qualify reproduction.
The final report must state scoped pattern replication, numerical differences,
materiality uncertainty, and the actual contract decision. Reference exit
success alone does not open primary search. Other regions, TME, whole-article
reproduction and bitwise equality are not added gates.

**Selected before submission:**
`tools/publication_reproduction_attempt003.sbatch` runs the fifteen remaining
contract sessions, indices 1–15, once each. The first-session attempt002 is
reused. The site's short-job aggregation requirement is met by **five finite
array tasks of three sequential sessions**, `--array=1-5%4`, each requesting
normal/one CPU/8 GiB/30 minutes. The coordinator explicitly accepted this
grouping. Numerical threads remain one, MATLAB uses `-singleCompThread`, and
any positive allocated CPU count is accepted and printed. At the observed
two CPUs/task, concurrency is at most eight allocated CPUs, expected cost is
about 2.1 allocated core-hours, and the five-task hard reservation is five
core-hours. Actual allocation counts determine charging and concurrency
adjustment; no one-CPU assumption is made. Larger sessions retain the 8-GiB
headroom. This grouping is finite, with no retries, requeue, polling or renewal.

Each source session uses the unchanged driver and seed `20261003 + contract
index`. Unique durable paths are
`runs/publication_reproduction/remaining_sessions_attempt003/<session>/`;
work paths are
`$SCRATCH/br_autoresearch/episode05_sensorimotor_lfp/remaining_sessions_attempt003/<session>/`.
Existing directories cause a stop, and any failed session stops its task.
Already completed sessions and all partial artifacts remain intact. Logs use
`publication_reproduction_attempt003_%A_%a.{out,err}`. Root owns submission,
monitoring, actual resource accounting and any later diagnosed repair.

Compute-only shell check for root:
`bash -n outputs/tools/publication_reproduction_attempt003.sbatch`.
Then the selected submission is
`sbatch --parsable outputs/tools/publication_reproduction_attempt003.sbatch`.
Neither command was run by this agent. No source payload, scored output,
additional scientific test, or primary contract was opened or changed.

### Post-array comparison implementation prepared, not executed

`tools/summarize_publication_reproduction.py` applies the frozen summaries to
the sixteen named completed reference directories, without provider-payload
access or model refits. It checks source/seed/function identities and fixed
support, reconstructs paired X/Y scores by each saved representation label,
and reports both pooled means/medians, all session profiles, animal/implant
summaries, and the 144-point ordinary regression. The MATLAB velocity result
struct includes saved predictions, so reading that selected output artifact
also deserializes its predictions; this is not a raw-source metadata claim.
The optional `--split-bootstrap` implements the predeclared descriptive
Monte Carlo diagnostic; it is not selected automatically. Numerical assembly
never sets a primary-search pass. The actual evidence interpretation and
contract decision belong in the completion record.

One focused hand-calculated regression fixture has been added for the new
summary formula: slope 2, SSE 4, SST 84, R-squared 20/21 and an analytical
F(1,2) tail probability. Root can execute it within coordinated compute:
`python3 -m unittest discover -s outputs/tools -p test_publication_summary.py -v`.
It has not been run by this agent. No existing method checks are repeated.

After all sixteen references complete, root's substantive comparison command
(one CPU, 2 GiB, up to ten minutes, bundled with authorized compute) is:

```bash
python3 outputs/tools/summarize_publication_reproduction.py \
  --first-session outputs/runs/publication_reproduction/Mihili_20140217_attempt002 \
  --remaining-root outputs/runs/publication_reproduction/remaining_sessions_attempt003 \
  --out outputs/runs/publication_reproduction/comparison_attempt001
```

The optional diagnostic must be explicitly selected before this aggregate
attempt by adding `--split-bootstrap`; its broad interval can never create a
pass. The ASTRA benchmark artifact now points to this real aggregation recipe,
and adaptive search depends on that scoped benchmark rather than one session's
reference completion. The code and changed program await root's review and
compute validation. No numerical reference output has been opened here.

### 2026-10-04 06:41 UTC — `adaptive_search`: native elastic-net pilot selected

In verified task-owned Slurm46458807/sh03-01n05 (4CPU/12GiB; ends06:51:41UTC), gcc12.4.0 built the separately preserved C99 single-sweep backend with `-O3 -std=c99 -fPIC -shared -ffp-contract=off`. Seven independent native equivalence/ABI/deadline/KKT fixtures passed0.610s, three recovery routing fixtures passed0.002s, and the original eight scientific objective/control fixtures passed against the native backend0.554s. Root and independent static reviews passed. Select Mihili_20140307 native pilot in fresh primary_elastic_net_control_attempt002 with unchanged270s internal/300s external limits; preserve old failed bank and all eight old completed banks. Only resource/support/convergence fields will be inspected before the seven-session recovery bound is chosen. No scores or additional scientific charges are released yet.

Native pilot completed06:41:35.149343–06:42:11.841864UTC in36.692516s: source2.566939s,fit33.992009s,peakRSS101080KiB,216complete trials/96physical electrodes,exact original LMP/baseline/target/trial replay. All360fits converged in1–5792sweeps;maximum original KKT ratio0.9999201162. Only resources/support/convergence were opened. Before scientific scores, select seven remaining sessions in fresh primary_elastic_net_control_attempt002, four single-thread workers,480s internal/510s external aggregate bound, unchanged270/300s child bounds and full300s remaining reserve before either wave. Two waves are expected around74s from this36.7s pilot; the shorter aggregate bound stops dispatch if a complete child reserve is unavailable. Summary path primary_elastic_net_summary_attempt002, recovery record primary_elastic_net_recovery_panel_attempt002; exact8-old/8-native mapping only. Parent remains46458807,4CPU12GiB; no other scientific workload overlaps its four workers.

### 2026-10-04 06:44 UTC — `adaptive_search`: native elastic-net recovery reaches its fixed convergence stop

Changed ASTRA validation passed (`validation_20261004_primary_elastic_net_recovery.json`). Recovery in46458807 started06:43:12.744090UTC and ended06:44:05.628608 after52.884528s,exit1,maximum childRSS185596KiB. All workers drained before the parent returned. Five more sessions completed: Chewie_20161006/1007/1014/1021/20150319 in28.324037/22.356643/25.617687/26.753795/23.500917s. Fourteen of the sixteen selected session banks are now complete, including eight preserved Python banks and six native banks. They remain preserved and their scientific scores unreleased.

Chewie_20150629 and20150630 failed at the fixed lambda0.001 KKT certificate after10000sweeps. Their elapsed times were6.980875/4.990231s, source reads4.260369/3.975141s, peakRSS91432/87436KiB. These were solver-cap failures, not the former Python runtime deadline or a filesystem symptom. Both concrete error records are under primary_elastic_net_control_attempt002; the parent record is `runs/adaptive_search/primary_elastic_net_recovery_panel_attempt002/remaining_panel.json`. No complete summary or selected-bank map was published. No tolerance relaxation, extra sweeps, changed lambda grid, subset scoring or automatic retry is selected. This prospective pair remains technically incomplete and uncharged: nineteen valid scientific comparisons remain. Resource costs from both preserved attempts remain consumed. An independently reviewed, distinct permitted repeatability-weighting comparison is being proposed; it is not yet adopted or executed.

At06:47UTC the coordinator closed the idle attended compute shell46458807 after the native recovery had drained and EP08 report-only rendering completed. No scientific process was left behind. Further prepared work will use a separately bounded attended allocation; there is no automatic renewal or duplicate pending request.

### 2026-10-04 06:48 UTC — `adaptive_search`: repeatability-weighted ridge adopted

Root and independent scientific review passed `primary_repeatability_weight_proposal.md`; adopt that exact prospective pair within the existing training-only reliability-weighting/ridge grammar. It keeps every original physical LMP coordinate and uses only seven-direction fitting input trajectories to form a clipped cross-trial agreement prior with fixed0.1 floor, then normalizes weighted Gram trace. Eight frozen physical weight-label controls use their own normalization/Gram and true targets. The original native-XY baseline, support, five penalties, candidate-only outer-session lambda, primary/identity gates and both full-LMP anchors remain unchanged. Per-recipe structural identity determines0/1/2 new comparison charges; seeds/folds do not add depth. No partial elastic-net scores informed this choice.

New implementation and focused independent fixtures are authorized. The first source measurement will be the fixed complete Chewie_20150313 session, one numerical thread within4GiB,270s internal/300s external, after passing checks in a verified task-owned allocation. Root will choose any finite remaining-panel bound from resource/support evidence before pilot scientific scores are opened. No source launch or additional valid comparison is claimed at adoption; nineteen remain complete. The previous4CPU shell is closed; a smaller bounded attended allocation will aggregate substantive prepared EP05/EP08 work when ready.

Scheduler closeout06:48:49UTC: parent46458807 COMPLETED25:43 with4allocated CPUs (1.714444 allocated core-hours), actual totalCPU25:04.112; step46458807.0 COMPLETED25:37,MaxRSS5003324KiB,exit0:0. Parent-shell success does not change the elastic-net scientific child failure recorded above. This allocation also served EP08's already separately recorded HF execution/finalization/figure.

### 2026-10-04 06:50 UTC — `adaptive_search`: next attended allocation selected

Following explicit closure of46458807 and fresh `sh_part`/`sh_node_feat` discovery at06:47, select one attended normal-partition CPU_SKU:7502 allocation:1task/1CPU/6GiB/30minutes, no GPU. It aggregates newly prepared EP08 logdet checks/full-cohort execution with adopted EP05 repeatability-weight checks/pilot and any resource-qualified remainder fitting the same allocation. Each recipe's own bounds remain unchanged; all numerical threads1. Root owns the PTY, submissions and scheduler checks. No allocation renewal, duplicate pending request, stopped-source workaround or unrelated job reuse is selected. Actual allocation ID and execution-host verification follow scheduler grant.

`srun` returned job46460577 queued/waiting at06:50UTC; root-owned PTY69705 is waiting for this one request. Source execution has not begun. EP08's three new logdet modules passed root static review; independent checks and ASTRA preparation are proceeding while the allocation waits.

### 2026-10-04 06:55 UTC — `adaptive_search`: repeatability-weight pilot selected after passing checks

In verified46460577/sh03-01n68,1CPU6GiB/30minutes through07:22:53UTC (Requeue disabled), all eight focused scientific fixtures passed0.378s and pinned ASTRA validation passed `validation_20261004_primary_repeatability_weight.json`. Root, independent reviewer and owner fixture review passed. The exact control-equals-candidate accounting edge is adopted: control charge is zero when every effective active-column weight equals the candidate in every required arm/fold/session, with no tolerance or score-based equality.

Select the fixed complete Chewie_20150313 pilot now under270s internal/300s external limits, fresh `runs/adaptive_search/primary_repeatability_weight_control_attempt001/Chewie_20150313`. Root will inspect only resource/support/replay fields before choosing the remaining aggregate bound. Count19 remains until a complete16-session result establishes any nonduplicate charges. Prior partial elastic-net scores stay unreleased.

### 2026-10-04 06:57 UTC — `adaptive_search`: repeatability-weight remainder selected from pilot resources

The fixed Chewie_20150313 pilot completed06:55:34.412003–06:55:47.999026UTC in13.587018s (source10.069161s,fit/score3.422184s,peakRSS187384KiB), with1038 original complete trials,86 physical electrodes and exact original LMP/baseline/target/trial replay. Only resource/support/replay fields were inspected; pilot scientific scores remain closed.

Before those scores, select all fifteen remaining sessions sequentially on the existing1CPU6GiB allocation46460577:720s internal aggregate deadline/750s external timeout, each child unchanged270s/300s and a complete300s reserve required before dispatch. Fifteen pilot times imply about204seconds; the selected envelope includes the300second terminal-child reserve and roughly216seconds additional allowance. Original pilot is reused. Fresh complete summary path `primary_repeatability_weight_summary_attempt001`, remaining execution record `primary_repeatability_weight_panel_attempt001`; source banks remain `primary_repeatability_weight_control_attempt001`. Changed ASTRA validation immediately precedes execution. No subset summary, automatic retry or scientific charge before complete sixteen-session evaluation.

### 2026-10-04 06:59 UTC — `adaptive_search`: repeatability-weighted ridge completes; small robust increment, practical gates still fail

Changed ASTRA validation passed in `validation_20261004_primary_repeatability_weight_remaining.json`. All fifteen remaining sessions and summary completed in46460577 at06:57:17.898175–06:58:41.906295UTC,84.008103s,exit0,maximum childRSS187076KiB. Pilot13.587018s is retained separately. Evidence: `runs/adaptive_search/primary_repeatability_weight_panel_attempt001/remaining_panel.json`, all16banks under `primary_repeatability_weight_control_attempt001`, and complete `primary_repeatability_weight_summary_attempt001/primary_repeatability_weight.json`. Root opened scientific scores only after complete-cohort success and used `tools/extract_primary_repeatability_weight.py` in compute. Both recipes are nonduplicate;21valid comparisons now include8controls among15post-initial comparisons.

Repeatability-weighted W gives primary deltaR2=0.002174850827889 and identity=0.003504405470203. Mihili/Chewie primary means are0.001248751638533/0.003100950017244; identity0.002654033650503/0.004354777289904. Primary is positive in5/6 Mihili and10/10 Chewie sessions. All sixteen outer selectors choose lambda1. Both practical margins0.01/0.005 fail; no primary-eligible candidate or promotion follows.

Mean weight-label control C gives primary0.001861584307623/identity0.003345396767572. W-minus-C is +0.000313266520265/+0.000159008702632: primary Mihili+0.000289026268136/Chewie+0.000337506772394, identity+0.000182482082511/+0.000135535322753. All16 primary session contrasts are positive. Both animal contrasts stay positive for both metrics under every fixed-prediction session omission and against each of the8control seeds; seed primary differences span0.000192794192335–0.000413862350127 overall and identity0.000085783515193–0.000224669779454.

W-minus-original fullLMP is +0.000237466082668 primary/+0.000108080924405 identity under both independent and matched-lambda anchors, which coincide atlambda1. Animal primary differences are Mihili+0.000234010257226/Chewie+0.000240921908110, identity+0.000118965792534/+0.000097196056276. Primary improves6/6 Mihili and9/10 Chewie sessions; every animal primary and identity contrast remains positive under every session omission. All three prospective weighting directional screens pass. This is a small input-prior improvement and does not relax the practical gates or establish intrinsic electrode reliability.

Across128candidate/1024control direction fits, no arm is structurally identity and no control equals its candidate on active weights. All12128 repeated fold/electrode T values are positive;34P values are negative (26Mihili/8Chewie), retained in diagnostics and assigned the fixed lower floor. Applied r spans0–0.814484784 and w0.438263801–1.494654854. Gram traces are86–96 with maximum numerical discrepancy4.27e−14. Selected ridge df spans18.793–26.853 forW and19.949–28.820 forC; control/candidate df ratios1.01449–1.10904 show that trace matching did not equalize effective capacity. These repeated diagnostic rows do not add independent animals. Required development search continues under its original bounds.
