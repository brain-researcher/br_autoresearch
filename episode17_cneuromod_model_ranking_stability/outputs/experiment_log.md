# Experiment log

## 2026-10-12 — prediction_qualification — EP21 attempt002 companion selected before fit002

Before submission, add one direct serial EP21 discriminator child to the
selected4CPU/31GiB fit002 allocation, before its four-worker fit. EP21's
attempt001 stopped during NumPy import before data access because its earlier
parent wrapper unset the module package path after module loading. Its
separately selected attempt002 retains the same scientific calculation and
uses the corrected unset-before-load order in an isolated module subshell.

The EP21 child truthfully admits this actual4CPU/31GiB parent, using one
numerical thread,300seconds internal/310seconds external+10seconds kill grace,
6.5GiB observed processRSS and1MiB categorical output. There is no srun child,
claim of a1CPU/7GiB hard memory boundary, additional allocation or EP21 charge.
The full32CPUh maximum reservation remains charged once to EP17. Require650
seconds remaining at job entry for checks and this companion, and at least350
seconds within the existing effective deadline before launching the child.
Record its independent exit in`status.txt`; the EP17 fit starts afterward and
is not gated by this unrelated companion's result. No overlap, automatic retry
or extension of epoch1791916149 occurs. Earlier fit002 scientific checks,
policy adoption, resources, paths and full-panel scope remain unchanged.

## 2026-10-12 — prediction_qualification — scientist adopts narrow block rule and fresh fit002 selected

The scientist answered question`call6o13giUJ`, "采用仅针对小块方差的修订（推荐）".
Adopt exactly the [narrow proposal](proposed_score_specific_validity_followup.md):
remove the individual uncertainty-block centered-response-variance veto only
for primary R². Preserve finite inputs and computed moments, strictly positive
finite SST for every required column in every individual block around its
matching frozen training mean, all training/validation/calibration/whole-audit
variance checks, unchanged supports and family001, secondary correlation/CKA
validity, all10000-draw/seed1729/family/undefined-draw rules, and the original
predictor/relation freeze before the single audit opening. No new audit access
or model result follows from this decision. GOAL, DATASETS, SEARCH_POLICY and
the operative qualification specification record the same adopted rule.

Source implementation is limited to a new finite-input/moment helper in
`tools/nested_prediction.py`, its individual block admission and the analogous
block admission in`tools/audit_prediction.py`. Shared variance qualification
and every SST calculation remain. The three focused methods in
`tests/test_primary_block_variance_amendment.py` use existing invented rosters
to check constant-block acceptance, unchanged training/validation and whole-
audit refusal, individual-block zero-SST/nonfinite/moment-overflow refusal and
unchanged secondary invalid-vector handling. The one affected original audit
test now retains its block refusal through zero frozen-mean SST. Source is
prepared only; no test execution is claimed in this selection entry.

Select fresh`tools/run_complete_nested_fit002.sbatch` for4CPU/31GiB/8hours,
four model threads with one BLAS thread each, no GPU/requeue/retry/successor.
Its maximum reservation remains32 allocatedCPUh. The actual bound is the
minimum of process start+8hours−60seconds, allocation EndTime−60seconds and
the unchanged episode deadline**1791916149** (October13 18:29:09UTC) minus
30seconds reserved for the external timeout offset, kill grace and dispatch.
The process-RSS guard is28GiB; Slurm's31GiB limit governs aggregate allocation
memory. An observed process-RSS guard does not establish an aggregate limit.
The coordinator's fit001 accounting reports MaxRSS16776672KiB before inner
scoring, so the previous16GiB allocation has inadequate demonstrated headroom.
Static array estimates omit native workspaces and retained allocator memory;
31GiB is a finite admission choice, not a guarantee of completion. Before
submission the coordinator reduced the unsubmitted32GiB proposal to31GiB
(31744MiB), below the reported four-core memory boundary, preserving the
4CPU/32CPUh reservation. No32GiB attempt was launched. The coordinator verifies
the accepted CPU/memory allocation rather than assuming requested resources.

Reuse qualified full001 features, actual cost002, family001, dry001, passed
component/concurrency evidence and original registry/exposure records. Run
the three new plus one affected methods (120seconds), changed ASTRA (60seconds),
adopted-policy parse/binding (20seconds) and the existing pre-score record checks (30seconds) in this substantive job
before fitting. Scientific checks, pre-score records and adopted-policy binding
are mandatory. ASTRA tooling/schema exit status is recorded without becoming
an execution gate. ASTRA and policy parsing use the existing qualified
Python3.12.1 module chain in a separate subshell with its refreshed library
path; the scientific encoding_model parent runtime is unchanged. No automatic
retry is selected.
The complete pipeline recomputes all352 PCAs, retains every84-setting/model/
group/stage/outer/final fit and publishes only a complete private checkpoint.
`runs/nested_prediction/fit002` and`cache/private/nested_prediction/fit002`
must both be absent before entry and are created exclusively. Existing CLI
attempt syntax already supports fit002; no path guard is weakened.
Preserve fit001 and both categorical diagnoses. The unchanged4GiB checkpoint,
64MiB metadata,256MiB feature,512MiB response and128KiB report caps remain.
There is no submission/completion claim in this prospective record.

## 2026-10-12 16:19 Pacific — prediction_qualification — narrow amendment presented; no retry selected

The complete `proposed_score_specific_validity_followup.md` is narrowed to
only the individual uncertainty-block centered-variance veto for primary R².
Root reviewed the concrete development and audit source locations. All finite
checks, training/validation variance, individual-block positive SST, calibration
support, secondary-statistic validity and uncertainty rules remain unchanged.
One focused scientist decision has been requested; it remains unanswered and
UNADOPTED. The earlier broad pooled-SST relaxation is not proposed. No new fit,
support revision or audit access is selected. The attended parent is closed,
with its entire0.594722222CPUh charge recorded once by EP15.


## 2026-10-12 16:15 Pacific — prediction_qualification — complete census isolates constant uncertainty blocks

The complete census finished at23:14:12.869380UTC,367.383036seconds elapsed,
356.906973loading,321.120903processCPU seconds,1309784KiB peakRSS. All1248
variance partitions and1632 training-baseline SST partitions completed.
There were18 invalid partition-column occurrences, all exact constant zero
variance, confined to sub-02/right at each B/C/D stage in blocks04/08/07/11
(1/1/3/1 columns respectively). Each stage had3unique affected columns;
identical cross-stage column membership is not inferred from these counts.
All training/validation variance partitions passed. No nonfinite values,
moment overflow, underflow or positive-below-relative-threshold failure occurred.
All1745832 tested SST column occurrences were finite and strictly positive,
including every individual uncertainty block. Evidence:
`runs/nested_prediction/fit001_diagnostic002/diagnostic_report.json`.

No model prediction, candidate score, support deletion or audit access occurred.
The proposed amendment is being narrowed to the per-uncertainty-block
centered-response-variance veto only. Retain all training/validation variance
rules and individual-block positive finite SST gates, original calibrated
support, secondary-statistic validity and undefined-draw handling. The rule
remains unchanged pending the scientist's decision; no replacement fit selected.

Changed ASTRA passed after replacing shell brace expansion that the command
template parser rejected with its equivalent plain variable reference. The
initial semantic failure and successful attempt002 are preserved under
`runs/nested_fit001_job_47564947/astra_response_quality_validation_47565811*`.
All companion work is charged only within EP15 parent47565811.


## 2026-10-12 16:07 Pacific — prediction_qualification — complete response-quality census selected

Select `tools/diagnose_fit001_response_quality_census.py`, independently
source-reviewed, in the existing attended parent47565811 on sh04-18n32.int.
One CPU/7GiB with a6GiB RSS guard,870-second internal and885-second external
limits, also bounded by the allocation and retained episode deadline. It
classifies all1248 fixed variance partitions and1632 training-baseline SST
partitions using the released development responses, retaining every calibrated
column and original partition. Counts/identities only, report at most1MiB;
no PCA, fitted predictions, candidate scores, endpoint or audit. Fresh output:
`runs/nested_prediction/fit001_diagnostic002/diagnostic_report.json`.
This diagnoses the rule failure; the score-specific-validity proposal remains
unadopted. No retry or changed validity rule is authorized by this diagnostic.
The whole attended parent remains charged once to EP15.

The initial invocation exited before imports or data access with a missing
closing parenthesis at source line113. Its execution log is preserved. Corrected
that syntax only and selected the same untouched output for attempt002, with
the same bounds; no empirical attempt had started or written that output.

## 2026-10-12 16:01 Pacific — prediction_qualification — variance refusal reproduced on actual block04

The exact response-only diagnostic completed at22:59:18.603586UTC after
375.870896seconds (371.095965loading), peak1309160KiB. It reproduced
`nested_prediction.NestedPredictionError: response_partition_variance_invalid`
for sub-02/right, stageB, uncertainty block04, shape101×942. No PCA, fit, inner
score, endpoint or audit was computed. Evidence:
`runs/nested_prediction/fit001_diagnostic001/diagnostic_report.json`.

The source rule requires finite inputs/moments, positive population variance
and variance>1e−12×mean-square for every frozen voxel in every training,
validation and uncertainty partition. The categorical refusal alone does not
distinguish zero variance from the relative-threshold or moment failure. The
current contract explicitly retains this stronger rule, independently of the
positive training-baseline SST needed for R². A complete count-only variance/SST
diagnosis and a clearly unadopted score-specific-validity proposal are being
prepared for a concrete judgement. No variance relaxation, support removal,
full-fit retry or protocol amendment is selected.

## 2026-10-12 15:51 Pacific — prediction_qualification — exact response-preparation diagnosis selected

Select `tools/diagnose_fit001_response_variance.py` after the EP15 mapping
diagnosis in attended parent47565811,1CPU/7GiB,6GiB observedRSS,870-second
internal/885-second external bounds, also capped by the allocation and retained
episode deadline. Existing released full001 development inputs are authorized
for the exact `_row_indices`/`_prepare_responses` path only: no PCA, model fit,
inner error, endpoint or audit computation. Fresh output is
`runs/nested_prediction/fit001_diagnostic001/diagnostic_report.json`; preserve
fit001. The report records existing categorical error and operation/group/stage/
partition identity, with no response magnitudes. If the documented variance
qualification fails, this does not authorize deleting columns/blocks or relaxing
the rule. The whole parent remains charged once to EP15.

Changed endpoint ASTRA passed with the existing EP18 Python3.12 dependency
path. The first invocation's missing-YAML import is preserved in its log;
no installation or ASTRA program change was needed. Evidence:
`runs/nested_fit001_job_47564947/astra_endpoint_caller_validation_47565811_attempt002.txt`.

## 2026-10-12 15:50 Pacific — initial_coverage — endpoint caller checks passed; failed-fit accounting closed

Both endpoint caller composition/binding tests passed in1.241seconds in
EP15 parent47565811, actual shellsh04-18n32.int, one numerical thread. Evidence:
`runs/nested_fit001_job_47564947/endpoint_caller_checks_in_47565811.txt`.
This qualifies only the new caller; the failed fit published no input checkpoint
and empirical endpoint execution remains unselected.

Slurm closed47564947 FAILED1:0 after494seconds with4allocatedCPUs, or
0.548888889allocatedCPUh charged once to EP17. The batch MaxRSS16776672KiB
and process-report peak3422352KiB measure different quantities; no OOM state
is reported. Preserve both observations and diagnose the response check before
deciding any successor. Step elapsed is not added to the parent charge.

## 2026-10-12 15:49 Pacific — prediction_qualification — fit001 stopped before scoring; source diagnosis and endpoint caller checks selected

The actual `runs/nested_prediction/fit001/fit_report.json` records all352PCA
fits completed at22:39:53.159029UTC, then a stop at22:39:58.498649UTC with
last operation `qualify_response_variance`. Pipeline elapsed460.509099seconds,
processCPU662.908471seconds, peakRSS3422352KiB; no private inner scoring was
attempted and no complete checkpoint was published. `runs/nested_fit001_job_47564947/status.txt`
records full_fit_exit1. The generic catch discarded the cause; a source-only
diagnosis is assigned before any retry or checkpoint read.

Separately, the next endpoint caller `tools/run_development_endpoint_panel.py`
has independent source review with no concrete defect. Select its two focused
composition/binding tests as a120-second optional companion in the idle,
attended EP15 parent47565811 (oneCPU,7GiB, all numerical threads1; no empirical
input or scheduler submission). Evidence will be
`runs/nested_fit001_job_47564947/endpoint_caller_checks_in_47565811.txt`.
This preserves all four pairs and the two10,000-draw interval families per pair;
it does not authorize empirical endpoint execution or claim initial coverage.

## 2026-10-12 15:34 Pacific — prediction_qualification — full development process running on real responses

Slurm confirms47564947 RUNNING onsh04-04n24. The actual batch shell records
4CPU/16GiB and four single-thread model workers. Selected pre-score records
passed; the actual multiworker overlap method passed in17.354s without a skip.
The full fitting process began22:32:17.990UTC and recorded its first real
development-response read at22:32:31.181UTC. Its current operational report
shows response_mean_bytes497664816, status started, no published checkpoint,
no inner-scoring attempt yet, no endpoint evaluation and no audit access.
This is confirmed real-data process execution, not a completed model result.
Evidence: `runs/nested_fit001_job_47564947/status.txt`,
`runs/nested_fit001_job_47564947/concurrency_check.txt` and
`runs/nested_prediction/fit001/fit_report.json`.


## 2026-10-12 15:30 Pacific — prediction_qualification — previous allocation closed; full fit pending

Slurm reports parent47553427 COMPLETED0:0,4190seconds at one allocatedCPU,
ending15:29:02Pacific:1.163888889allocatedCPUh charged once to EP17, including
its EP15/EP18 companions. Step0 MaxRSS4160012KiB; extern/step elapsed are not
additional charges. The attended PTY has closed. In the same scheduler
observation,47564947 is PENDING with no start time; no fit execution is claimed.


## 2026-10-12 15:29 Pacific — prediction_qualification — complete fit001 submitted as47564947

The duplicate-name queue check returned no job; one sbatch submission returned
job47564947. It owns the admitted4CPU/16GiB/8h full development fit001, with
all limits and prerequisite evidence in the admission below. Submission is
confirmed; running state and completion are not yet observed. Operational
evidence will be under `runs/nested_fit001_job_47564947/` and
`runs/nested_prediction/fit001/fit_report.json`. No automatic retry is selected.
The attended companion parent47553427 was exited at22:29:00UTC after its
selected work; closed accounting remains to be read once from Slurm.


## 2026-10-12 15:29 Pacific — prediction_qualification — repaired publication verified; complete development fit001 admitted

The affected real derive_rank_pca checkpoint regression passed in0.426s;
the previous five checkpoint passes remain valid. The full-fit wrapper passed
bash syntax checking and changed ASTRA passed schema/semantic validation.
Evidence is in `runs/stimulus_features/pilot_operations_20261012_001/`:
`checkpoint_loading_view_regression_47553427.txt`,
`full_fit_wrapper_syntax_47553427.txt` and
`astra_full_fit_validation_47553427.json`. Failed fixture evidence is preserved.

Select complete development fit001 from completed full001 image features,
cost002, family001 and dry001 through the reviewed existing full-data bridge.
Release its literal twelve development response HDF files, frozen native-column
maps, design identities and existing derived calibration operators to the
procedural development reader. The complete bridge preserves the original
response arithmetic, all2423 images,8558 retained columns, four models, four
layers, all84 settings, six outer fits and each final fit, and352-PCA feasibility
barrier. Private inner validation objectives are now authorized for fitting;
no audit inputs or public endpoint scores are released. The complete prediction
checkpoint remains private. Existing registry, known-answer, deliberate-leakage,
cross-episode exposure and numerical evidence referenced below is reused.

Select one non-requeued Slurm job:4CPU,16GiB,8h,noGPU; four model workers with
one numerical thread each. The measured width/count/training-row projection is
about10.66 serialCPUh before full-copy I/O and concurrency overhead;32allocated
CPUh is the reservation ceiling, not expected use. Retained arrays total about
4.39GB plus native/runtime space and less than40MiB PCA publication copies.
Use12GiB observed-RSS guard,4GiB checkpoint cap,64MiB metadata,256MiB features,
512MiB response means and128KiB report caps. Node-local temporary/cache storage
and established Oak artifacts preserve the admitted layout and existing budget.
The absolute deadline is the minimum of actual Slurm end minus60s, start plus
8h minus60s and1791916149 (October13 18:29:09UTC, existing execution reference).
No budget reset, automatic retry, reduced panel or audit opening is selected.

The job first runs the one actual multiworker overlap check previously skipped
in the1CPU parent; a skip/failure prevents fitting. Verify no existing duplicate
before submitting `outputs/tools/run_complete_nested_fit001.sbatch` once.


## 2026-10-12 15:27 Pacific — prediction_qualification — checkpoint production checks passed; new fixture mapping repaired

The selected six-check module ran in1.081s: all five existing methods passed;
the new regression saved and loaded actual strided PCA views successfully,
then failed its strict exact-state comparison because its replacement fixture
used dicts while production and the reader use immutable mapping proxies.
Evidence: `runs/stimulus_features/pilot_operations_20261012_001/checkpoint_loading_layout_checks_47553427.txt`.
Only the new fixture now uses the producer's recursive immutable-map helper.
Production code and exact-state assertions are unchanged. Select only that
affected method for rerun in the same120s bounded parent slot; reuse the five
passes. Independent source review found no production defect.

## 2026-10-12 15:26 Pacific — prediction_qualification — endpoint family and evaluator complete; checkpoint-layout repair selected

Family001 completed at22:21:50.662UTC in8.384770s, peakRSS400016KiB.
Its seven ordered primary components are B_Aall_raw, C_Aall_raw, D_Aall_raw,
D_AN_raw, D_RN_raw, D_VNC_raw and D_VNC_normalized, with six raw_R2 scales
and one ceiling_normalized_R2 scale. All five directed primary edges remain.
H_BC and H_CD are inapplicable because sub-02/right has undefined calibration
gain; no participant/ROI group is removed. The eight retained widths sum8558.
The exact family is frozen in
`runs/prediction_qualification/endpoint_family/family001/family_report.json`.
No candidate scores or raw calibration responses were read by this binding.

Dry001 completed the actual frozen prediction, registered10000-draw bounds and
primary adjudication in8.594044s, peakRSS263088KiB, using the existing invented
606-image/120-concept/12-block fixture. No study payloads or scientific audit
were opened. Evidence:
`runs/prediction_qualification/evaluator_dry_run/dry001/dry_run_report.json`.

Source review found that actual derive_rank_pca loading slices are strided,
whereas the complete checkpoint writer required C-order for every array.
The narrow repair canonicalizes only these small PCA loadings, after existing
resource/shape/dtype checks, and retains the no-copy requirement for large
response/prediction banks. Select the existing five checkpoint methods plus
one actual derive_rank_pca save/load/forward-equivalence regression in parent
47553427: one thread,120s plus5s grace, existing4GiB allocation and node-local
temporary storage. Revalidate changed ASTRA and syntax-check the prepared full
fit wrapper. Full fitting remains unsubmitted until the repair passes.

Existing completed prerequisite evidence is reused: stable/reversal/null
recovery under job46531931 (October4 23:50UTC log entry), and the six actual
audit-prediction checks in
`runs/prediction_qualification/cuda_roster005_companion001/audit_prediction.txt`
(1.697s), including deliberate composite/leaky input rejection. Registry and
cross-episode exposure were adopted October4; comparisons remain correlated
with EP18 and checkpoint target-image exposure remains unknown. No clean
exposure or independent replication claim is authorized.

## 2026-10-12 15:19 Pacific — prediction_qualification — fixed real cost complete; family binding and required evaluator composition selected

Cost002 completed at22:17:19.984UTC in76.756063s overall, with54.533017s input
loading and22.158905s for the fixed numerical probe. All16 SVDs,8064 ridge fits
and8064 predictions completed. Respective operation times were10.328012,
8.189025 and0.757161seconds; peakRSS1346932KiB. It read only the admitted
training responses; validation/fold01 responses and scores were not returned.
Response headers total497664816bytes, fixing the summed A_all width at8558.
This measures a complete timing probe, not full fitting or a neural result.

The new derived-endpoint binder and outcome-free evaluator caller have root
source review. Select three focused binder checks<=60s plus5s grace, then one
registered full606-image/120-concept/12-block unmocked evaluator dry001:300s
internal,360s external plus10s KILL grace,2GiB observedRSS and64KiB report.
It uses the existing invented fixture,10000 registered crossed draws and actual
primary adjudicator; no real audit, smaller draw count or new recovery program.
Run serially in parent47553427 with qualified EP17 runtime/addon, one numerical
thread and existing node-local temporary storage. Revalidate changed ASTRA.
The true>=2CPU overlap check belongs in the later substantive fitting allocation.

After passing binder checks, select a<=60s derived-calibration-only release:
existing operator_report.json, applicability.json and48–80 named vectors in
operators.npz, with32MiB metadata/array caps and2GiB observedRSS. Bind and record
the exact existing EndpointFamily, ordered components, scales, edges and
inapplicability reasons. No raw calibration responses or candidate scores are
released. Full development fitting still requires this frozen family, completed
evaluator dry run and a concrete allocation informed by measured cost.

## 2026-10-12 15:16 Pacific — prediction_qualification — input-mode repair passed; fixed cost002 admitted

Five cost-input methods passed in0.732s and three complete-bridge methods
passed in0.024s in parent47553427. Evidence is in the two mode-repair transcripts
under `runs/stimulus_features/pilot_operations_20261012_001/`. The immutable
full001 image features, earlier numerical checks and unchanged ASTRA are reused.

Select fresh cost002 using exactly the15:03 procedural input release and fixed
16-SVD/8064-fit/8064-prediction probe. Its absolute deadline is22:45UTC
(1791845100), with external TERM after remaining time plus10s and10s KILL grace.
This new repaired attempt preserves cost001's failed12.417s; it does not reset
consumed compute, the240-hour reference or parent23:18:13UTC work deadline.
The same1CPU/4GiB parent, one numerical thread,3GiB observed-RSS guard,
64MiB metadata,256MiB feature,512MiB response-header and64KiB report caps apply.
No full fitting, validation response, scoring or automatic retry is selected.

## 2026-10-12 15:13 Pacific — prediction_qualification — inherited setgid input mode diagnosed and repaired

The categorical trace locates cost001's refusal in the builder-only directory
mode assertion, before reading the row map. Bounded stat confirms role-copy
attempt002 has inherited mode2700; its development directory is700 and map is
regular600 with one link. Operator directory/file are700/600. This is not a
sealed-input or row-map-format failure. No permission change is needed.

The cost and complete-response bridges now use a read-only check accepting
exactly700/2700 directories and unchanged600 single-link regular files under
the admitted root. Symlink, path, group/other-access and size checks remain.
Source review found no concrete defect. Select the five cost-input methods
and three complete-bridge methods, each module<=60seconds plus5s grace, in
parent47553427 after its active EP15 diagnostic exits. Use the qualified EP17
runtime/addon and one numerical thread. These checks use invented inputs;
failed cost001 and complete full001 remain preserved. A new cost attempt is
not started until checks pass and its deadline/resource admission is recorded.

## 2026-10-12 15:08 Pacific — prediction_qualification — cost001 input binding refused before neural values

Cost001 stopped at22:06:36.508UTC after12.417212s with exit1, peakRSS878380KiB,
loading_cost_inputs_failed and last operation bind_frozen_development_row_map.
No neural training read was attempted, no response header allocation was
planned and no cost grid executed. Preserve its report and the complete full001
features; this is an input-adapter failure, not measured full-fitting cost.

Select one<=60s metadata-binding diagnosis in the same1CPU/4GiB parent and
3GiB observed-RSS limit. Traverse only the admitted cost input path, stop before
any read_training_column event, and emit exception categories and source
file/function/line frames (including causal frames), never exception text,
feature/neural values or source lines. Source review of the row-map producer
and consumer proceeds in parallel. A retry requires the diagnosed repair and
affected checks; no resource, role or numerical condition is relaxed.

## 2026-10-12 15:07 Pacific — prediction_qualification — real cost probe started; historical clock evidence recovered

The corrected parallel test module completed in7.337s: both serial methods
passed and the actual >=2CPU overlap method remains explicitly skipped.
Preserve its earlier failed transcript. The cost-input and complete-bridge
passes and ASTRA below remain valid; no production fitting rule changed.

Actual cost001 started at22:06:24.091UTC in parent47553427. Its fresh
`runs/prediction_cost/cost001/cost_report.json` records started, not completed;
no timing projection or full-fit admission follows until the measured report
is complete. The22:35UTC cost deadline and parent23:18:13UTC work deadline
remain unchanged. Only the previously selected development cost scope is open.

The240-hour first-development-round limit has no separately fixed start in
GOAL or policy, as the October9 record already noted. The prior October3
midnight reference was explicitly conservative, not an observed timestamp.
At22:06UTC, coordinator `TZ=UTC sacct` recovered the actual starts of the two
earliest documented substantive jobs:46425120, ep-assets-prepare, started
October3 at18:29:09UTC and completed at18:31:34;46425841, ep-portfolio-7502,
started18:42:25 and completed19:38:11. Both belong to zijiao. Earlier permitted
episode records describe planning, not execution. This supplies a more precise
earliest-documented-execution reference for future admission:240hours from
18:29:09 ends October13 at18:29:09UTC. It is not a newly reset240-hour budget
or proof of a retrospectively registered timer. Preserve consumed compute;
no existing attempt, selected process deadline or resource ceiling is extended.

## 2026-10-12 15:03 Pacific — stimulus_features — complete real development bank published privately; cost probe admitted

Full001 completed at21:58:59.041UTC with process exit0, after1173.165825seconds
(1172.139504CPU seconds). Its operational report records all2423 preprocessing
operations,9692 image/model inferences, four loaded backbones and
full_development_features_complete=true. The150218142-byte feature container
was published, partial cleanup succeeded, and peakRSS was955124KiB. Evidence:
`runs/stimulus_features/full001/feature_report.json` and parent47553427's
attended shell. No neural, calibration/audit pixel or candidate-score access
occurred; the report correctly remains completed_private_staging.

The coordinator verified sh03-17n15.int, job47553427 and one allocated CPU
inside the same shell. Cost-input checks passed3/3 in0.708s; complete fitting
bridge/publication checks passed3/3 in0.023s; changed ASTRA passed. The parallel
module had one pass, one fixture failure and the expected >=2CPU overlap skip
in9.872s. Its fixture incorrectly required the new parallel mapping wrapper
during the original serial reference call. Only that assertion scope was
corrected; all parallel immutability checks remain and production is unchanged.
Select the same module<=60s plus5s once, preserving the failed transcript.
This separate fitting test does not invalidate the completed features or
passing fixed-cost path. No full fitting has been admitted.

Select cost001 with the following procedural builder handoff: full001's complete
development feature HDF, existing role-copy attempt002 development row map
and exactly twelve development/{participant}/{B,C,D}.h5 files, plus the eight
already frozen A_all operator axes and role/event metadata. No mixed source,
audit, raw calibration responses or scoring view is released. Neural logical
reads are only folds03–06 and the16 smallest original columns per group;
HDF5 physical chunks may span other development cells. Cost-only NaN response
headers do not authorize full response fitting or a search-worker score.

Admit this unchanged16-SVD/8064-fit/8064-prediction probe in the same1CPU/4GiB
parent, one numerical thread,3GiB observed-RSS guard and absolute deadline
1791844500 (22:35UTC), with external TERM/KILL grace. Set metadata64MiB,
features256MiB, response headers512MiB and report64KiB caps. The documented
structural width upper bound implies497664816response bytes plus148869120
feature bytes; exact header bytes are checked before allocation/neural values.
Only one HDF is open at a time with1MiB cache. Largest input/PCA workspaces and
runtime overhead are admitted within the remaining memory headroom, with the
hard4GiB Slurm limit; this is a resource estimate, not a proven native-memory
bound. No new allocation, full fitting or automatic retry is selected. First
finish the already selected short EP15/EP18 companions, then launch cost001
while at least20minutes remain to its absolute deadline.

## 2026-10-12 14:57 Pacific — prediction_qualification — complete parallel fitting path implemented; focused checks selected

New `tools/complete_development_inputs.py`, `parallel_nested_prediction.py`
and `run_complete_development_prediction.py` connect the existing twelve
development HDF copies to the unchanged complete fitting kernels and private
checkpoint publisher. One through four model workers share immutable inputs;
all352 PCA fits across22 training partitions finish before response
qualification and selection. Every model retains all84 settings, eight groups,
B/C/D, six outer fits and the final all-development fit. Resource callbacks
are synchronized and any failure stops the whole run. Active image extraction
and the existing cost-probe modules were not changed. Root source review found
no concrete defect; empirical execution remains unselected.

Select `tests/test_complete_development_prediction.py` (three methods) and
`tests/test_parallel_nested_prediction.py` (two serial methods plus one
explicit >=2-CPU overlap method), each at most60seconds plus5seconds grace,
after full001 exits in parent47553427. Use the existing qualified EP17 runtime,
addon and one numerical thread. The overlap method must skip in this1CPU
parent and later run in an admitted multi-CPU fitting allocation; do not create
a test-only job. These checks open invented inputs only. Revalidate changed
ASTRA with the already selected cost-input checks.

Memory admission must include original responses, immutable snapshots, shared
baseline, four OOF banks, the complete PCA bank and concurrent native workspaces.
For N=2423 and total calibrated group width W, the retained response contribution
can reach7*(3*N*W*8) bytes before workspaces; the input and float64 feature
snapshot contribution is3*148869120 bytes. The measured cost001 report must
inform actual resources and walltime; the current4GiB parent is not admitted
for full fitting. Full features, the fixed cost probe, the required outcome-free
evaluator boundary and remaining pre-score qualifications still precede launch.

## 2026-10-12 14:45 Pacific — prediction_qualification — existing cost probe connected to development-only artifacts

While full001 runs, the new `tools/released_development_inputs.py` connects the
already completed role-copy attempt002, development row map, eight frozen
calibration-derived A_all axes and selected complete feature bank to the
existing cost probe. No new materialization or calibration is needed. It
logically reads only folds03–06 training repeats and the16 smallest original
A_all columns per group. Full response headers contain NaN for unobserved
cells and are explicitly cost-only, not complete response panels for fitting.
HDF5 physical caching may span unselected cells; only admitted logical values
are returned to Python. Response-header memory is explicitly admitted before
allocation. No audit, mixed source or raw calibration response is opened.

`tools/run_development_prediction_cost.py` adds an explicit bounded entrypoint
with no automatic handoff or execution. It preserves the original16 SVDs,
8064 ridge fits and8064 predictions, with no validation response, error,
objective, selected model or numerical output. Required caps cover the entire
pipeline, including loading/imports; an exclusive attempt report preserves
stopped work. Root source review found no concrete defect. Three focused
methods in `tests/test_released_development_inputs.py` remain **unrun**.

Select those three methods≤60seconds plus5seconds termination in this parent
only after its active full-image process finishes, with the same qualified
runtime/addon and one numerical thread. Revalidate the changed ASTRA then.
Full001 completion, passing checks, explicit procedural builder-handoff
selection and whole-pipeline resource admission still precede actual cost001.
No neural access or fixed cost probe has yet been launched. The following full
fitter already performs22-partition feature feasibility and response-variance
qualification internally before selection; do not create another qualification
program or treat already completed calibration as pending.


## 2026-10-12 14:39 Pacific — stimulus_features — real pilot completed; complete development extraction admitted

The three affected launcher checks passed in0.082seconds and four full-I/O
checks in0.872seconds. Changed ASTRA passed schema and semantic validation.
Actual **pilot002 completed** at21:37:08UTC after33.921792seconds: all32 fixed
development images, four fixed models and four layers, peakRSS670128KiB.
The existing seed17 checkpoint was reused. Neural and audit access stayed false.
Evidence: `runs/stimulus_features/pilot002/pilot_report.json` and the preserved
operational outputs in `pilot_operations_20261012_001`. This is real-image
feature preparation, not neural scoring or a model-ranking finding.

The completed pilot measures6.0230275fixed seconds and27.8986919variable
seconds including conservatively scaled residual work. The existing full-route
formula2×(fixed+2423/32×variable)+60 projects4296.9542seconds (71.62minutes).
Independent source review confirms one image at a time, four resident backbones
(376128512parameter bytes),148869120feature payload bytes, and at most19849216
additional bytes during bank freezing after model release. Runtime/checkpoint
workspaces and unusual image geometry remain outside that static sum. The
pilot's sequential-model peak is not represented as a full-memory measurement.

Select **full001** for every2423 frozen development image, all four models and
all four layers, in existing parent47553427:1CPU/4GiB/noGPU, one numerical
thread,3GiB observed-process-RSS guard,256MiB private HDF allowance and64KiB
operational report cap. The explicit deadline is1791846300 (**23:05UTC**),
before the parent23:18:13work deadline and the episode's conservative calendar
limit. At admission21:37:53UTC,5226seconds remained to this full-run deadline,
exceeding the buffered projection; startup overhead is included in that margin.
The external timeout uses the remaining absolute-deadline seconds plus10seconds
termination grace, and Slurm supplies the hard4GiB limit. Charge no new parent:
all serial work remains inside the existing≤2CPUh reservation. No automatic
retry follows a failure or timeout.

Explicitly bind full001 to the successful pilot002 report. Preserve the failed
pilot001, both completed pilot files and the original source/archive/seed17.
The full bank remains private staging. No calibration/audit pixels, neural
responses, candidate scores or automatic role-handoff release are selected.

Changed ASTRA passed. Actual full001 began at21:39:25.875UTC; the operational
report records first image access21:39:29.199UTC and first feature computation
21:39:29.376UTC in parent47553427. Status is started, not completed. Evidence:
`runs/stimulus_features/full001/feature_report.json`; final completion requires
both the complete status and full_development_features_complete flag.



## 2026-10-12 14:37 Pacific — stimulus_features — wrapped field diagnosed; explicit pilot002 repair selected

Correction to the provisional14:25 interpretation: the notice does not contain
a space-bearing passphrase. Its password label lists ZIP filenames across two
lines; the final filename and colon are still part of that label. The parser
now extracts the single explicit field and rejects ambiguous alternatives.
No credential guessing, alternate source or notice text disclosure occurred.
The affected format check passed. The same authorized first member then opened
successfully, without reading image bytes, as recorded in
`runs/stimulus_features/pilot_operations_20261012_001/wrapped_label_member_open_result.txt`.
This diagnoses pilot001's first-member failure as a parse-boundary defect;
preserve that failed report, partial directory and persistent seed17 checkpoint.

The independently reviewed repair adds explicit named attempts rather than
overwriting or automatically advancing them. Select the three affected launcher
methods and four full-feature I/O methods in attended1CPU/4GiB/noGPU parent
47553427, serially with one numerical thread and30/90second external caps.
Their fixtures do not access acquired images or neural data. After these
checks pass, select **pilot002**, the unchanged fixed32 development images,
four models and four layers, with600seconds internal and660seconds external
plus10seconds termination grace. Reuse the existing seed17 checkpoint. The
parent ends23:19:13UTC; work deadline23:18:13UTC remains earlier than the
conservative episode deadline. Use fresh operational output `private_pilot002.txt`;
no full2423-image extraction is admitted until the pilot's completed cost report
has been inspected. Candidate scoring, neural access, audit access and
role-handoff release remain closed.


## 2026-10-12 14:29 Pacific — stimulus_features — archive verified but first member opening refused

The repaired delivery reached the actual pilot and created its first report.
Attemptpilot001 stopped at21:27:54UTC after11.246seconds in `image_decode`.
The archive's exact content identity and selected metadata passed; the seed17
checkpoint was created and must be reused. The report records one member
opening attempted, zero preprocessed images, image_pixels_read=false, no model
inference and no neural/audit access. Preserve the private partial directory,
report and checkpoint; pilot001 cannot be overwritten.

Select a bounded30second diagnosis of the same first member's opening with
the existing parsed value, emitting only a fixed exception category and whether
ZipFile reports a bad-password error. Do not read image bytes or print exception
text. In the same trusted diagnostic, classify whether the notice's final line
has an explicit archive-named field before its colon; no private text/value
is emitted. This distinguishes a parse-boundary error from a decoder failure
without trying candidate credentials. No new pilot is selected until diagnosed.
## 2026-10-12 14:25 Pacific — stimulus_features — provider passphrase format repair selected

The private format-only diagnostics establish no inline colon/equal password
field, an unpunctuated penultimate heading starting with Password, and one
final unquoted printable-ASCII passphrase containing spaces. No notice text,
credential value or credential length was exposed. The pinned provider README
identifies this file as the archive-password source; its already accepted
terms remain unchanged. The existing parser rejected this valid document
format before any image access.

The narrow launcher repair accepts one such heading followed by one terminal
line, preserving internal spaces exactly. Ambiguous headings, quoted values,
multiline values and existing ambiguous inline fields refuse. No candidate
password search, alternate source or weakened archive authentication is used.
The existing parser test adds an invented passphrase and these refusals; select
the three affected launcher methods≤30seconds in parent47553427. Once passing
and source-reviewed, select the same fixed32-image pilot≤660seconds plus10s
termination, under the unchanged parent/calendar deadlines. Its pilot001
output path is still unused; use a fresh operational stdout file named
`private_pilot_repair001.txt`, preserving the earlier delivery refusal.

Separately, the first full-development I/O run had3passes/1failure in0.895s:
Pillow's context closed the source stream but retained decoded image memory.
The loader now explicitly closes the image in finally, preserving the original
post-context assertion and adding exceptional-exit coverage. All four methods
then passed in **0.781seconds**. Preserve both check outputs; no further repeat
is selected. Complete source review and tests support this software interface,
not empirical feature completion or reader release.

The three launcher methods passed after the format repair. Independent review
then identified mixed inline/heading ambiguity; reject that combination before
selecting either field. Its regression was added to the existing parser method,
which passed in the same allocation; the other two passing methods are reused.
Run the selected fixed pilot now with its fresh operational stdout. No private
value was printed during diagnosis or repair.

The first repaired launch still refused before private delivery. A bounded
grammar diagnosis found an internal period in the explicit archive heading;
the initial heading grammar had excluded filename periods. Admit internal
periods while excluding sentence-final prose, and replace the invented valid
case with an archive-filename heading. Preserve `private_pilot_repair001.txt`.
Verify the affected parser method and the actual notice's parse-success flag
only before selecting the fresh `private_pilot_repair002.txt` launch. Pilot001
scientific outputs remain unused. No credential guesses or pixel retries occurred.

The format probe additionally identified an archive-list comma in the heading.
The final narrow grammar includes internal filename periods and commas while
rejecting sentence-final prose and multiple credential fields. The existing
invented test now represents that heading shape and passes. Crucially, the
actual authorized notice now also returns **parse passed** without exposing
its contents (`notice_parse_final_result.txt`). Select the fresh operational
repair002 launch below using that verified parser; preserve all earlier
format probes and refusal outputs.

## 2026-10-12 14:21 Pacific — stimulus_features — startup restored and private-delivery refusal under diagnosis

In the actual4GiB attended parent47553427, the companion passed all startup
checks, its three new launcher tests (**0.013seconds**) and ASTRA validation.
At21:20:01UTC the actual launcher stopped with categorical phase
`private_delivery`; no pilot001 report was created. Preserve all outputs in
`pilot_operations_20261012_001`. The inherited-memory startup problem is
resolved for this actual allocation; this is a separate delivery refusal.

Select one30second diagnostic in the same compute PTY: import the existing
callable, apply its frozen metadata selector, check original pilot paths are
unused, then read and parse only the existing authorized bounded private
notice. Emit fixed phase/success and exception-class categories only, never
notice text, tokens or exception text. No exec, pixel decode, model load or
pilot retry follows the probe automatically. This identifies the failed
operation while preserving the private-delivery boundary.

The new full-development I/O and CLI modules have independent source review.
The one post-publication interruption/success-reporting defect was repaired;
four focused methods cover complete2423-row/16-matrix persistence, exact
metadata binding, member limits, private delivery and closeout. Select those
four checks≤90seconds in this existing parent, one thread and job-local
temporary fixtures (~143MiB). These tests read no acquired assets and do not
admit full empirical extraction. Pilot costs are still required for that
decision. All use remains inside the selected2CPUh parent reservation.

The diagnostic passed imports, the exact32/2423 metadata selection, unused
output checks and private notice reading. It failed specifically in
`private_notice_parse` with DeliveryRefusal. Select one further bounded
format-only classification of that same notice, emitting only fixed format
categories/counts, no private text, tokens, token lengths or exception text.
This will support a narrow parser repair without guessing a credential.

## 2026-10-12 14:19 Pacific — stimulus_features — environment refusal reproduced and attended image work selected

The bounded diagnostic completed successfully in EP21 step47375558.1 on
sh02-15n09. It reports valid job/step IDs, one CPU, an accessible EP17 directory,
and memory environment value7168 despite the explicit128MiB step request.
This reproduces the inherited-field mechanism that makes the original
`SLURM_MEM_PER_NODE == 4096` assertion fail. It does not recover the original
step's unlogged environment. The issue precedes the private launcher and is
not evidence of a credential or image failure. Diagnostic output is preserved
in `runs/stimulus_features/startup_environment_diagnosis_001/`; no scientific
work or new standalone allocation occurred in that probe.

Select one attended **1CPU/4GiB/2-hour/zeroGPU** allocation on the freshly
discovered dev partition, with at most120seconds of allocation wait. The
unchanged pilot companion will run directly in this4GiB allocation rather
than a child of a7GiB parent, bounded externally to900seconds plus10seconds
termination grace. Verify the actual node, job and allocation first. The three
new launcher checks and ASTRA precede the fixed32-image/four-model pilot.
The prior attempt never created its operational directory or pilot001, so
these original first-pilot output paths remain unused; preserve its failed
Slurm record and EP18-captured output. No submitted EP18/EP21 source changes.

This substantive allocation can then host the new complete2423-image feature
producer only after its source review, focused checks and a resource decision
based on the actual pilot report. Full extraction is not admitted merely by
obtaining the allocation. Reuse all unaffected passing evidence. The separate
EP18 continuous-EEG bridge's three focused source-reviewed checks may run
serially, with no raw EEG or image access, inside this parent. All numerical
threads remain one; no simultaneous workers are selected.

Reserve at most2CPUh once to EP17, including its serial companions; preserve
all prior charges and the existing conservative October3 calendar reference.
Every stage must end before the earlier of actual parent end minus60seconds
and October13 00:00UTC. Require sufficient time for each selected stage and
close the attended shell when work finishes. No keepalive, renewal, detached
service, automatic retry, audit exposure or neural scoring is selected.

Allocation **47553427** started at21:19:12UTC on **sh03-17n15.int**.
The actual attended shell reports step0, one CPU and4096MiB, with scheduler
end23:19:13UTC. Its stage deadline is23:18:13UTC, earlier than the retained
calendar bound. The original companion's memory assertion is appropriate to
this actual4GiB parent; no memory variable is rewritten to manufacture a pass.
Run the selected companion now, retaining its original unused pilot paths.

## 2026-10-12 14:17 Pacific — stimulus_features — bounded startup-environment diagnosis selected

Source review narrows the silent exit1 to the companion's pre-log shell
assertions but does not identify which assertion failed. Select one read-only
diagnostic step in the task-owned, freshly verified RUNNING EP21 parent
47375558: one CPU, 128 MiB, zero GPUs, at most60seconds, with a90second outer
submission/execution limit. The parent's unchanged training uses one numerical
thread and a15GiB RSS guard within2CPU/16GiB; this tiny shell-only companion
fits the remaining resources and is charged only within the existing parent.
No private notice, pixels, scientific Python, retry or training edit is selected.

The diagnostic explicitly supplies the original failed caller's observed
SLURM_MEM_PER_NODE=7168 as an srun input while requesting128MiB on its command
line. This is a declared reproduction of the environment-inheritance condition,
not a claim that the EP21 parent or new step has a7GiB allocation. Print only
hostname, job/step IDs, CPU and memory environment fields and the accessibility
of this episode's known directory. The explicit command-line request and step
accounting, not an inherited environment field, establish the selected grant.
Official srun documentation states command-line memory options take precedence
over inherited memory inputs: https://slurm.schedmd.com/srun.html#OPT_mem .
This probe may establish the refusal mechanism; it cannot recover the original
step's unlogged environment. No real-image rerun is selected by this entry.

## 2026-10-12 14:13 Pacific — stimulus_features — companion stopped before operational logging

The containing EP18 builder47551249 started on sh03-06n09 and passed its
configuration and ASTRA checks. The selected EP17 Slurm step47551249.0 then
exited1 after one second, with sampled MaxRSS1120K. Its captured stderr only
reports the locale warning and Slurm task exit/step termination. The expected
`pilot_operations_20261012_001` directory and `pilot001/pilot_report.json`
were not created. The companion script creates that operational directory
before launcher tests, ASTRA or any private notice/pixel access; none of those
stages is observed to have run.

The pre-log shell checks are under diagnosis. An inherited parent memory
environment value is a candidate explanation, not an established cause.
Preserve the submitted script and attempt. Do not automatically rerun the
pilot or claim its three new checks passed. The independent EP18 pixel
builder continues as prospectively selected; this one-second step remains
included once in the EP18 parent charge. The next action is to establish the
failed startup condition before choosing a narrow repair and fresh attempt.

## 2026-10-12 14:05 Pacific — stimulus_features — private real-image pilot selected

Under the adopted procedural same-account boundary and standing execution
authority, select the existing fixed32-development-image/four-model pilot.
The exact EP17 acquisition's `archives/images_fmri.zip` and
`terms/password_images.txt` were confirmed by metadata only: ordinary files,
917286854 and397bytes, mode0440. Neither credential contents nor source pixels
were read in preparation. Existing terms acceptance applies. No new credential
or permission request is needed. EP18's different archive, password and
feature outputs are not substitutes or inputs to this EP17 attempt.

The new trusted launcher `outputs/tools/run_private_stimulus_feature_pilot.py`
first applies the existing fixed development selector, then privately reads
one bounded explicit password field and passes only its value through an
inherited pipe. It execs in the same PID; no secret enters arguments,
environment, reports or terminal. Failure emits a categorical status. The
existing callable, four models, four layers,32 images, fixed seed17 checkpoint
and pilot001 paths are unchanged. Prior outputs cannot be overwritten.

Select `outputs/tools/run_private_pilot_companion_001.sh` inside the EP18
substantive builder parent: an actual **1CPU/4GiB/15-minute Slurm step**, noGPU,
one numerical thread. Run three new invented-notice/descriptor tests≤30s and
changed ASTRA≤60s, then the real pilot≤600s internally/660s externally with10s
termination grace. Reuse all unaffected passing evidence. The source has no
process-RSS guard; actual Slurm step memory provides the4GiB allocation.
The existing4MiB feature-cache,128MiB checkpoint and256KiB report limits remain.
No full feature extraction, neural fit, scoring release or automatic retry.

All parent/step use is charged once to EP18; EP17 records its measured child
time as included. Preserve prior EP17 charges. As no earlier calendar origin
was fixed, conservatively use October3 00:00UTC, the earliest documented
execution day, for this admission and require the entire15-minute step to
finish by **October13 00:00UTC**, within240h. This does not reset a timer from
today. The scientific-design entries of September27/30 and October2 explicitly
record no execution attempt. Parent and child original deadlines both apply.

Independent source review accepted the launcher, tests and companion. Actual
compute checks, decryption, image inference and completion remain pending.
The operational report will establish those facts. No audit pixels/features
or neural responses are admitted; all reader-release/scoring flags stay false.

The containing EP18 builder **47551249** was submitted and observed PENDING
at21:07UTC. EP17 will run only in its separately admitted4GiB step after parent
startup and EP18's configuration/program checks. The outer step wait/execution
is additionally capped at900s plus10s termination grace. No EP17 startup,
private notice read, decryption or feature extraction has yet been observed.

## 2026-10-12 13:10 Pacific — source_roles — scientist adopts procedural same-account separation

At about 13:10 Pacific the scientist (zijiao) answered the portfolio review's
evaluation-separation item for EP02, EP15, EP17 and EP18 with "follow your
recommendation": procedural separation with a disclosed weaker claim, rather
than sending the SRCC draft at EP18 log:11. That request is not sent. The
review cited the October 3 sandbox failure (namespace error, ENOSPC, EP18 job
46427768), which is EP18's route, not an EP17 attempt. EP17's basis is not
public re-download: its neural source and password-protected stimulus pixels
are held under accepted access terms. The operative point is that this
account already holds the restricted source and the October 9 private role
copies (job 47087633), including audit-role bytes, so same-UID permissions
cannot enforce separation.

The [October 6 and October 9 handoffs](source_roles_proposal.md#october-9-current-reader-and-delivery-handoff)
named a disclosed procedural boundary as an unadopted alternative but drafted
no terms. Their request for a supported Slurm route or controlled evaluator
identity, and that unadopted status, are superseded; the file is unedited.
The new short terms follow EP07's procedural wording and are recorded in
[GOAL](../GOAL.md#data-readiness-and-access-boundary), SEARCH_POLICY
`access_boundary.separation_amendment_2026_10_12` and
[DATASETS](../DATASETS.md#what-must-exist-before-neural-model-comparison).
Development reads only the released development handoff. Audit, mixed-source,
raw-calibration and sealed outcome paths are not opened, listed or searched.
Frozen choices are recorded before the single recorded audit opening. Every
audit/sealed access is logged here; unplanned access is disclosed and
affected results are marked. Every evaluation result discloses procedural
same-account separation and is never called enforced or blinded.

Unchanged: the 720-concept universe, 480/120/120 roles, N=256/q=32 rule,
margins, the 6,000 CPU-hour/256 GPU-hour/240 wall-clock-hour/4 TiB limits,
the one-opening limit and every other gate. Separation is now satisfied
procedurally; physical isolation stays unverified, and no reader-release,
isolation or scoring flag is set true. Next dependency: private stimulus
delivery through the documented `password_images.txt` interface still blocks
the 32-image stimulus pilot; resource admission and the outcome-free
evaluator dry run follow. Enforced-boundary wording in subsidiary
specifications and ASTRA is superseded; refresh and validate ASTRA at the
next compute session. Documentation edits only, on sh04-ln04 without an
allocation; no secret, payload, audit file, job or scheduler action.

## 2026-10-11 09:21 Pacific — prediction_qualification — image-lifetime repair passed and compute session closed

The selected repair explicitly closes the converted RGB object in `finally`,
including preprocessing exceptions. The source image remains owned by its
loader. An independent source review also confirmed that the named remaining
submitted EP21 stages do not invoke these EP17 files after the already-closed
roster005 companion. No EP21 wrapper, grant or training source was changed.

Attended job47390492 ran onsh04-18n32.int with oneCPU/4GiB/10minutes,
noGPU and one numerical thread, using the original companion's
encoding_model Python3.11 and EP17 h5py target. All five affected methods
passed in5.871seconds. The formerly failing full-roster method now reaches
all original numeric-value and immutable-array/mapping assertions; the
expanded existing error method also verifies both image objects close and
no inference occurs after preprocessing failure. No new method was added.
Evidence: [tests](runs/prediction_qualification/rgb_close_repair001/tests.txt),
[process measurement](runs/prediction_qualification/rgb_close_repair001/time.txt),
and [closed status](runs/prediction_qualification/rgb_close_repair001/status.txt).
Independent inspection verified the five passing methods and zero exit.

The test process took24.92wall seconds,10.99user plus0.63system CPU seconds,
and621024KiB peak RSS. The local fixture directory was empty at exit
(6bytes); the small durable receipt directory was6304bytes before its final
close lines. The shell exited at16:21:16UTC. At16:21:35UTC, Slurm confirmed
COMPLETED0:0,67allocated seconds on oneCPU and11.673totalCPU seconds;
the step's sampled MaxRSS was1607660KiB. Charge0.018611allocatedCPUh
once to this separate EP17 repair, within its1/6CPUh ceiling. The original
EP21 companion remains counted only inside its original parent.

All49 selected methods now have passing evidence across the original
44 unaffected methods and the repaired five-method module. Preserve the
original49-method attempt's48pass/1fail result and development_features_exit=1;
do not overwrite it or claim that attempt passed. Its unchanged ASTRA
validation remains reusable. This closes the observed software defect only.
Private stimulus delivery, enforced reader separation, empirical cost/resource
admission and the original model qualification remain unresolved. No private
payload, real feature bank, scientific audit or ranking result was opened.

## 2026-10-11 09:20 Pacific — prediction_qualification — completed companion and selected image-lifetime repair

The [original companion](runs/prediction_qualification/cuda_roster005_companion001/status.txt)
in EP21 parent47375558 closed after all49 selected methods executed:48
passed and one failed. All other groups and ASTRA schema/semantic validation
passed. The failure in [development_features.txt](runs/prediction_qualification/cuda_roster005_companion001/development_features.txt)
is at test_development_features.py:139: the converted RGB image remains
readable after preprocessing, although the producer promises to close it
before inference. The later numeric-value and feature-array immutability
assertions were not reached. Do not infer an array-mutability failure or
describe the original companion as fully passing. Its failed receipts remain
unchanged, with development_features_exit=1 and status=closed.

Root and two independent source reviews diagnose the reliance on
`with image.convert("RGB")` as insufficient to close the in-memory RGB
object in this runtime. Select an explicit `rgb.close()` in `finally`,
preserving the caller's source-image loader context, preprocessing, all2423
images, four models,16 layer banks and bytes-backed immutable arrays. Extend
the existing error-case method to verify source and RGB closure plus zero
model calls when preprocessing raises; add no test method or scientific
simulation. No private images, checkpoint or neural values are opened.

Select only the affected existing five-method module for an attended Slurm
check, using the same encoding_model Python3.11 and EP17 h5py target as the
failed companion, with GPU visibility empty and every numeric thread set1.
The previous module took10.441seconds; the new exception subcase visits one
fixture image. Request oneCPU,4GiB,10minutes in the discovered dev partition,
with a115-second test timeout plus5-second termination grace and a64MiB
local fixture allowance. Maximum additional charge is1/6allocatedCPUh to
EP17, separate from the original once-only EP21 parent charge. No suitable
CPU training allocation remains; the live GPU fits retain their16GiB host
memory and are not used for this repair check. This is an attended test
session, not a new empirical worker. Record its job ID and actual result
below; no retry, install, broader test suite or new empirical access is
selected. Reuse the44 passing methods outside this affected module and
unchanged ASTRA; the other four original passes remain historical evidence.

## 2026-10-11 05:44 Pacific — prediction_qualification — original audit response delivery implemented and checks selected

The new `tools/audit_responses.py` fills the missing role-copy-to-audit-panel
interface specified by DATASETS.md's equal retained-repeat endpoint and
`audit_prediction.AuditResponsePanel`. It consumes caller-supplied authenticated
metadata and already-open audit role handles, preserving606 images,120
concepts,twelve blocks,eight original A_all axes and all B/C/D stages.
Every row mapping and all twelve headers bind before values; each physical
512-column stripe is reused across both ROIs, with the original chronological
float64 equal-repeat means and immutable output. It neither opens a source
nor qualifies audit variance, fits a model, scores a candidate or releases data.
Reader separation and the completed one-shot model/relation/forecast lock
remain external requirements. Independent source review passed with no findings.

Exactly three new `test_audit_responses` methods remain **unrun**. Select them
and the existing46 unique pending methods as an independent serialized
software companion in EP21's newly selected original CUDA pair5 parent.
The nine older modules are `test_development_endpoints`(4),
`test_development_forecasts`(6),`test_development_secondary`(5),
`test_audit_prediction`(6),`test_audit_secondary`(5),
`test_predictor_checkpoint`(5),`test_development_responses`(5),
`test_development_prediction_cost`(5),`test_development_features`(5).
The audit-secondary recipe's repeated primary method is already included
in the full audit-prediction suite and is not counted or executed twice.
The20 checks passed in47329203 are reused.

Use existing qualified module Python3.12 for numeric groups, and the existing
encoding_model Python3.11 plus EP17 h5py target for response/feature delivery
and the new audit adapter. GPU visibility is empty and numerical threads one.
Retain existing115/175-second group limits: old limits plus kill grace total
1020seconds, new adapter adds120seconds; ASTRA and setup are additional.
The existing checkpoint fixture requires512MiB local scratch, so the new
EP21 roster parents explicitly allow1GiB before submission. No install occurs.

Separate records use `runs/prediction_qualification/cuda_roster005_companion001/`.
Ordinary EP17 failures do not gate EP21; SIGKILL or the shared parent's
169200-second StartTime deadline (also EndTime minus60) stops the parent.
EP21 accounts for the parent once, with no separate EP17 compute charge.
The parent was submitted as **47375558** at05:46:56Pacific. All49 checks and
changed ASTRA remain unrun pending actual allocation.
Private stimulus delivery, reader isolation and empirical qualification remain
unresolved; this implementation does not open an audit or establish a result.

## 2026-10-10 22:44 Pacific — prediction_qualification — twenty composition checks passed

The completed EP21 parent **47329203** ran `test_endpoint_evaluator`,
`test_audit_adjudicator` and `test_nested_prediction`: all **20 methods passed
in 59.289 seconds**. Evidence is in
`runs/scoring_composition_checks_job_47329203/tests.txt` and its closed
`status.txt`; current ASTRA schema/semantic validation also passed. Reuse these
checks while their dependencies remain unchanged. Later source components
outside this named suite still have unrun checks.

The parent completed with exit 0:0 and is charged once to EP21, including this
tail; add no EP17 reservation or charge. These fabricated fixtures establish
endpoint, audit and nested-fitting behavior only. No private asset, empirical
predictor, model-ranking result or audit opening occurred. Encrypted access,
permitted isolated reader delivery and full empirical admission remain separate
requirements. No new EP17 job or automatic successor is selected.

## 2026-10-09 — reader handoff clarified before empirical pilots

The current [source-role handoff](source_roles_proposal.md#october-9-current-reader-and-delivery-handoff)
now separates actual reader/delivery dependencies from completed preparation.
[Full role copying](#2026-10-09-0134-pacific--role_materialization-complete-attended-calibration-selected),
[calibration maps/supports](#2026-10-09-0141-pacific--calibration_support-full-empirical-maps-and-supports-completed)
and [calibration operators](#2026-10-09-0837-pacific--calibration_support-operators-completed-stimulus-adapter-checks-failed-at-import)
are complete and are not retry targets. The stopped namespace route named in
the materialization specification belongs to EP18; no reviewed EP17 record
establishes an EP17 namespace/ENOSPC attempt. The ambiguous stimulus-specification
reference now names EP18 explicitly.

An effective supported reader boundary and private inherited-descriptor
stimulus delivery remain unestablished. The next empirical dependency is the
existing 32-image stimulus pilot after those prerequisites, followed by
resource admission for development prediction cost. No credential, payload,
system configuration, namespace probe, fixture, job or new scientific outcome
was accessed or executed in this clarification. The 66 pending checks and
unchanged EP21 companion selection remain as previously recorded.

## 2026-10-09 — prediction_qualification selects required prediction-cost implementation

The full feature/response/nested-fit/checkpoint interfaces already compose.
Two independent source inspections instead found the required outcome-blind
prediction cost pilot in prediction_qualification_specification.md still lacks
an implementation. Select that missing caller, not another glue wrapper and
not its execution. The existing frozen pilot holds fold01 completely aside,
trains on all images in the 320 concepts of fold03 through fold06 and predicts
the 80 concepts of fold02. The assignment writer's exact fold01...fold06
labels determine those roles, never first image occurrence or a caller-selected
ordering. Use the first 16 ascending original A_all column IDs per group,
all four registered models, four layers, three PCA ranks, seven penalties,
eight participant/ROI groups and every B/C/D target.

Consume caller-admitted FeaturePanels and ResponsePanels with their complete
frozen DevelopmentDesign and original column bindings. Inspect metadata before
values; do not reuse the full fitter's all-row snapshot routine. Never access
fold01 values or validation neural responses. Reuse the production full SVD
at rank128 and rank-specific scale derivation: 16 full SVDs, 48 rank-specific
training/validation transform pairs, 8,064 ridge fits and 8,064 predictions.
No selector objective, validation score, chosen setting or model comparison
is computed. Release only dimensions, operation counts, finite/success flags,
timings and peak RSS; discard private fitted and predicted arrays. A stop
must remain explicitly incomplete, with no smaller grid or successful-subset
result. This is the already specified cost probe, not a reduced experiment.

One worker owns tools/development_prediction_cost.py, a second its focused
checks, and a third reviews the source. Root owns records and compute. Require
the caller's resource callback before data copies and numerical operations;
reporting a cost never grants a resource budget or reader access. Existing
one CPU / 6 GiB / 600-second internal, 660-second external plus 10-second grace
pilot limits remain a proposal, pending measured source-feature dimensions
and actual admitted reader delivery. No private payload, model fit, import,
test or allocation is selected this turn. New checks remain outside the
unchanged EP21 companion; passing numerical-core evidence is reused.

The callable [development_prediction_cost.py](tools/development_prediction_cost.py)
is now implemented. Root and independent source reviews found no concrete
defect. Fixed fold names are independent of the design's first-occurrence
order. Header checks precede values; feature conversion/finite checks follow
allowed row slicing, and target slicing joins only training rows to the 16
smallest original column IDs. No validation-response read, full-panel snapshot,
score or selector call occurs. Completed prediction counts certify their
finite shape checks, not the validity of unvisited operations or empirical
source qualification. The returned immutable report contains only operational
dimensions, attempted/completed counts, operation/total wall and process CPU
time, process-lifetime peak RSS and fixed completion/failure codes.

Caught failures retain elapsed time spent in the failed operation; an after-
operation resource refusal preserves the completed numerical count while
leaving the whole pilot stopped. Private exception objects/text and numeric
arrays are absent from the report. An external timeout remains necessary for
a blocked numerical kernel, and a hard kill can prevent a report entirely.
No cost has actually been measured. The one-partition/16-column result will
not by itself certify the full 22-partition/all-column workload, and the
callable neither extrapolates resources nor activates a full fit.

Five methods in [test_development_prediction_cost.py](tools/test_development_prediction_cost.py)
cover exact full-grid dispatch and original-column joins, guarded/poisoned
forbidden rows, absence of scoring and private report values, metadata refusal
before reads, and honest interruption/failure counts. Root reviewed their
source. Expensive numerical kernels are instrumented; their existing passing
arithmetic checks are reused. All five methods and changed ASTRA validation
remain **unrun**. The future check recipe uses one CPU / 2 GiB, 115 seconds
plus five seconds termination grace, inside a substantive allocation. Its
estimated live fixture arrays stay below roughly 100 MiB before runtime and
allocator overhead; actual duration and RSS remain unmeasured. No standalone
test job, resource extension or change to the EP21 companion is selected.

The full caller-resident feature/response panels can occupy 646,533,936 bytes,
before admitted feature slices, SVD matrices/workspace, scores, the current
16-column target/fit/prediction and runtime. These are source bounds, not a
measured resource admission. EP17 now has **66 new methods unrun**: the
original 20 remain in the unchanged EP21 proposal, and the latest 46 remain
outside it. Actual admitted image delivery, the fixed stimulus pilot and
persisted random checkpoint, reader separation and selected empirical cost
execution still precede full fitting. No private payload, fit, scientific
score, new finding or Slurm job was produced this turn.

## 2026-10-09 — prediction_qualification selects full development feature delivery

The existing stimulus pilot covers exactly 32 images; the complete nested
fitter requires all 2423 eligible development images and all four fixed models'
four registered layers. Select the missing full image-to-FeaturePanel delivery
under the existing checkpoint, preprocessing, float32 and spatial-mean rules.
Bind the full independent eligible image manifest to DevelopmentDesign before
pixels, preserve its original image/concept/fold/block order, and retain all
sixteen model/layer banks without centering, PCA, model-specific exclusions or
neural outcome access. Reuse the existing exact backbone loader and require
the already persisted canonical seed17 artifact; do not create or reroll it.

The caller supplies an admitted exact-image loader yielding decoded originals.
This routine discovers no archive or credential, grants no reader release and
cannot authenticate pixels merely from matching IDs. It converts the original
JPEG to RGB without EXIF rotation and uses unchanged model_features preprocessing.
Keep one image/tensor at a time, infer sequentially through four CPU backbones
at batch size one, and retain full float32 banks. Require a caller resource
callback before model loads, output allocations, image access, preprocessing,
inference and immutable output copies. A failure aborts the entire return.

One worker owns tools/development_features.py and focused tests, with independent
scientific/source review and root integration. Final feature arrays contain
148,869,120 bytes; four resident backbone parameter banks add about 376 MB,
plus buffers, activations, image decoding, immutable-copy transients and runtime.
Actual pilot measurements and full resource admission remain required before
execution. No source pixel, checkpoint, model forward, test or new allocation
is selected here. The 32-image pilot, effective reader separation and full-cache
resource selection are not replaced by this implementation. New checks remain
outside the unchanged pending EP21 companion proposal.

`tools/development_features.py` and five focused methods in
`tools/test_development_features.py` are complete. Independent reviewer and
root read the implementation and tests without finding a concrete defect.
The public API returns four immutable FeaturePanels, with the same complete
DevelopmentDesign and all sixteen native-float32 layer banks. It requires the
canonical existing random checkpoint before any model or pixel access, then
uses the unchanged fixed loaders. The caller's exact-image loader retains
the existing 16-MiB member and single-JPEG / 16-million-pixel bounds before
eager decode; a post-load dimension check alone cannot bound decoder memory.
The producer closes its RGB copy and caller source context before inference.
Failures expose fixed operational codes and cannot return partial panels.

All five methods remain **unrun**. They retain the complete 2423-row design,
four models and real feature widths while instrumenting only source/model
I/O and the already-qualified preprocessing/inference kernels. An independent
row/model/layer/column oracle checks feature placement. Additional checks cover
metadata and random-artifact refusal before reads, whole-call failure for a
bad model/layer, source cleanup, private-error suppression and resource refusal
through the full-bank freeze boundary. Existing passing backbone/preprocessing
checks are reused, not reselected.

The unselected recipe uses the existing model Python and installed EP17 h5py
target, one CPU / 2 GiB and 115 seconds plus5-second termination grace inside
a future substantive allocation. No installation or separate short job is
selected. Fixture outputs use 148,869,120 bytes, with at most one additional
19,849,216-byte layer during freezing or one similarly sized oracle, plus
test metadata and Python/Torch runtime; actual RSS/time remain unmeasured.
Production additionally retains 376,128,512 bytes of model parameters plus
buffers, checkpoint-loading transients, decoder/activation memory and allocator
retention. Models are released before immutable output copying. This source
bound does not qualify production RSS or reuse the pilot's smaller residency.

There are now **61 new EP17 methods unrun**: the original20 remain in the
pending EP21 companion and the latest41 remain outside its unchanged tail.
Changed ASTRA validation is also unrun. Full image-to-array production logic
is implemented; actual admitted loader wiring, the measured32-image pilot,
canonical random-artifact creation/reuse, physical reader separation, full
resource admission and empirical feature/rank qualification remain required.
No source pixel, model parameter, actual feature, neural response, fit, test,
import, scheduler inspection or new job was accessed or executed this turn.

## 2026-10-09 — prediction_qualification selects development response delivery

The nested fitter requires complete image-mean ResponsePanels, but the existing
role-copy reducer only handles calibration. Select the missing development
adapter under DATASETS' equal-repetition response rule. It accepts already
admitted private role-copy handles, their full row maps, the independently
frozen eligible image manifest and A_all original columns; it discovers no
paths and grants no reader access. Bind all 2423 images, 480 concepts, existing
fold/block labels, participant repeat counts and acquisition chronology before
reading values. Check all twelve B/C/D headers first, then read each necessary
physical 512-column stripe once per participant/stage across both ROIs.

Use the unchanged float64 equal-repeat mean in calibration_core.center and
retain all selected images and original columns in the caller's frozen order.
Return eight immutable ResponsePanels for the existing fitter. Nonfinite
selected responses fail the call without dropping observations; reuse the
fitter's existing variance/SST checks rather than duplicate them. Full feature
delivery, empirical execution, persistence and actual reader separation are
outside this implementation. Require the caller's resource callback before
allocation and I/O; no production memory or runtime grant is selected.

One worker owns the adapter and focused tests; root independently reviews and
owns integration and records. The retained numeric output is at most 497,664,816
bytes at 8558 total columns, plus bounded stripe, aggregation and immutable
snapshot buffers and caller-owned HDF5 caches. Actual physical reads and runtime
must be measured under a later selected allocation. No payload, test, import,
fit, new allocation or change to the pending EP21 proposal is authorized here.

`tools/development_responses.py` is implemented. It binds the full independent
image manifest to the caller's frozen design, authentic per-person retained
counts and complete copied local-row axis. Image/concept/fold/block identities
are not string-coerced. Eligible repeats follow the retained acquisition order;
counts, ordinal ranks and alternating half metadata must agree. The existing
copy reader checks all twelve native-shape, float32, ordinary HDF5 headers
before any response slice. Distinct 512-column stripes are joined by original
column IDs across both ROIs and visited once per participant/stage. Only
frozen eligible-image repeats and A_all columns enter the means. Invalid
selected values refuse the complete return without observation deletion.

The adapter returns all eight ResponsePanels with immutable float64 means and
original column axes. It reuses calibration_core.center, including the exact
constant-column branch, while preserving provider float32 values and scaling.
It does not reuse calibration-only eligibility thresholds or the smaller
calibration stripe-size limit. The existing fitter retains responsibility for
partition/block variance and SST. Caller source identity, effective physical
reader separation, full feature inputs and actual job admission remain required;
passing metadata checks cannot establish those external facts.

Author and independent root source review found no remaining concrete defect.
Five methods in `tools/test_development_responses.py` cover the complete row,
group and stage rosters with independent ordered scalar means, unequal repeat
counts, cancellation-sensitive values, original column order and shared stripe
reuse. They also cover metadata refusal before headers, the twelfth-header
failure before any values, nonfinite selected responses and resource refusal
before metadata, I/O or averaging. The fixture replaces only the existing
copied-header helper with instrumented sources; no actual HDF5 files, source
arrays or model fits are involved. All five methods remain **unrun**.

The new unselected recipe uses the existing model Python and already installed
EP17-only h5py target, one CPU, 2 GiB and 115 seconds plus5-second termination
grace. It requires no install and no separate short job. The fixture visits
36 source stripes sequentially, each at most 12,408,832 bytes, with 1,628,256
bytes of final means, plus metadata/runtime and reference buffers. Runtime
and peak RSS remain unmeasured. In production, completed panels plus current
mutable ROI means total at most 497,664,816 bytes for C<=8558 columns. Peak
also includes either a physical stripe (at most 26,173,440 bytes using the
published row ceiling) and repeat-reduction buffers, or one group/stage's
immutable copy and finite mask, plus metadata and caller-owned HDF5 caches.
No fixed production RSS or runtime is established by this source bound.

There are now **56 new EP17 methods unrun**. The original20 remain in the
pending EP21 proposal and the latest36 remain outside its unchanged625-second
tail. Changed ASTRA validation remains unrun. No test, import, actual response
read, scientific fit, scheduler inspection or new job occurred this turn.
Physical reader isolation and full feature delivery still prevent empirical
model fitting; this implementation closes the response-aggregation gap only.

## 2026-10-09 — prediction_qualification selects complete prediction-bank persistence

The nested implementation currently returns an in-memory result only. Select
its missing save/load interface so downstream jobs can reuse a completed fit
without recomputing models or choosing new settings. Preserve all four model
OOF banks, the common per-image baselines, original column axes and typed
image/concept/fold/block design, every outer selection and rank64 transform,
and all final development predictors with their complete84-setting objectives,
training rosters, PCA/ridge parameters and fit metadata. This container is a
complete-result checkpoint, not a resume point inside an interrupted nested
fit. The separately frozen block order, calibration supports/operators/banks
and authentic role-safe responses/features remain external dependencies.

Use one private ZIP_STORED container with numeric NPY members and compact JSON
metadata, without pickle, public score sidecars, receipt chains or refitting.
Require an explicit caller byte cap and resource callback for both writing
and reading. The caller supplies an admitted existing private directory;
the implementation creates no directory, selects no data source and grants
no access. Exclusive0600 partial output is published only after completion
with no overwrite of an existing destination. Failed partial output remains
separate from a complete checkpoint. Restore immutable numerical snapshots
and exact identities/settings; serialization does not establish scientific
qualification, method-freeze provenance or reader isolation.

One worker owns the implementation, one independently reviews fidelity and
storage boundaries, and one owns bounded independent tests. Root owns records
and integration. No actual checkpoint, private payload, fit, test or new job
is selected here. Runtime checks require coordinated compute and remain
outside the pending EP21 proposal; its resource question is still unanswered.

`tools/predictor_checkpoint.py` now implements the complete save/load boundary.
All 512 numeric members and the private metadata remain in one uncompressed
container; parameter-selection objectives are not exposed in a public sidecar.
The writer validates complete state before creating its exclusive partial,
checks the serialized lower bound before scanning each array, and enforces the
actual byte cap during writing. Numeric headers are checked against expected
shape, dtype and payload size before loading; restored arrays use immutable
byte backing. This is a format for the producer's completed artifacts, not a
general archive authentication or provenance system.

Publication closes and synchronizes the completed file before an atomic
no-overwrite hard link. A publication race preserves the destination and the
writer's partial. Reserved partial names cannot be loaded, even if their ZIP
contents are complete. Once publication succeeds, there are no further
resource callbacks; failure to remove the extra partial link returns the
committed path with `partial_cleanup_complete=False`, rather than falsely
reporting publication failure. Source review also corrected a writer/reader
layout mismatch by requiring C-contiguous producer arrays before any write.
No conversion, repair fitting or parameter selection occurs during reload.

Independent implementation review, author/test cross-review and root source
review found no remaining concrete issue. Five methods in
`tools/test_predictor_checkpoint.py` cover byte-exact full-state roundtrip,
typed identities and exact tie selections, immutable restoration and a direct
stored-predictor calculation; incomplete state and malformed containers;
admission before file/header/payload work; and failed or racing publication
with separate post-publication cleanup status. These are bounded software
fixtures, not empirical or recovery evidence. All five remain **unrun**.

The unselected ASTRA recipe requests one CPU, 2 GiB and 115 seconds plus
5-second termination grace within a future substantive allocation. Its
full-row fixture has only 2–5 response columns per group: approximately
25 MiB per result/container, estimated numerical arrays below 256 MiB, and
approximately 200 MiB disposable node-local files across refusal cases.
Reserve 512 MiB local scratch. All estimates are source-based and unmeasured.
For production, at 2423 images and 8558 total columns, choosing the largest
2048-wide layer and rank128 throughout gives a numerical payload bound of
2,633,264,896 bytes (about 2.45 GiB), plus JSON and NPY/ZIP overhead. The actual
identity lengths determine metadata size. Writing avoids a complete result
copy; reading retains one full numeric bank, with caller state and temporary
memory additional. This estimate does not authorize or size a production job.

There are now **51 new EP17 methods unrun**. The original20 remain in the
pending EP21 companion proposal; the latest31 remain outside its unchanged
625-second tail. Changed ASTRA validation remains unrun. No actual checkpoint,
test/import, private payload access, empirical fit or job submission occurred.
The coordinator's single scheduler inspection at 2026-10-10 01:18:10 UTC
(October9 18:18 Pacific) showed an empty user queue. Qualified empirical
features/responses, enforced reader separation, full freeze/decision
integration and measured production resources remain unresolved. Completed-fit
reuse is implemented; incremental recovery inside an interrupted fit is not.

## 2026-10-09 — prediction_qualification selects audit secondary score composition

Select the missing audit counterpart of the adopted Fisher-z correlation and
rank64 CKA calculations. The caller supplies the existing audit application
result, its selected primary pair and at most two prelocked secondary check
descriptors. Compute only the named scales, retaining each scale's entire
calibration-eligible six-component family for its separate simultaneous bounds.
No additional pair, scale, selected check or scientific adjudication is selected.

The audit uses one all-development rank64 transform over all606 images and
the retained12-block order, without development fold mixing. Reuse the existing
centered weighted kernels and actual32-bank CKA averaging. Whole-block integer
multiplicities represent literal row replication; zero-mass people contribute
zero, and any positive-mass undefined component fails its complete family.
Reuse the existing10000-draw audit uncertainty engine without changing draws,
margins or failure rules. Actual bank membership is required only for selected
CKA; correlation uses already frozen inclusion weights.

Retain the primary application's existing immutable response snapshots in its
result so the secondary caller uses exactly that observation bank without a
replacement response input or matrix recopy. One worker owns the adapter and
this minimal producer change, another reviews audit semantics, and a third owns
focused new tests. Root owns integration and records. No private data, scoring
execution, audit opening, reader release, persistence or job submission is
selected. Checks remain unrun and outside the pending EP21 companion proposal;
that resource question remains unanswered.

`tools/audit_secondary.py` is implemented. Only scales named by the retained
lock produce statistics, each with the complete frozen eligible component
family and the separately identified selected checks. The caller cannot add a
new pair or replacement response input through this interface. Original block
coordinates, per-voxel Fisher transformation, single-transform whole-audit CKA,
all32 anatomical subset CKAs, same D bank and equal ROI/person aggregation are
preserved. An empty secondary lock returns no statistic; Fisher-only binding
needs no unused CKA bank. Numerical failure never shrinks the selected family.

The primary application's result now retains the immutable response matrices
already used for its SSE/SST. The secondary adapter reuses those exact matrix
objects, along with predictions and transformed features, while freezing small
mapping/coordinate structures. This extends the lifetime of one124467552-byte
response bank at8558 total columns (about118.70MiB), without a new matrix copy
or higher peak from this producer change. Statistical kernels still allocate
temporary arrays, and full10000-draw production runtime remains unmeasured.
Caller provenance, fully validated lock and actual one-shot reader enforcement
remain prerequisites; array checks do not establish those facts.

Independent implementation review, author/test cross-review and root source
review found no remaining concrete issue. Five new methods in
`tools/test_audit_secondary.py` cover complete locked scale families, actual
Fisher composition against literal row replication, instrumented single-bank
CKA/all32 subsets, original bindings, undefined results and resource refusal.
The existing application method now also checks response snapshot equality,
source-mutation independence and immutable backing. A test-only memory issue
was corrected: callback mocks no longer retain every temporary606-by256 subset
in their argument history, and the class fixture is released after its suite.

The unselected ASTRA recipe combines these five methods and that one affected
application method under one CPU,2GiB,115 seconds plus5-second termination grace
inside a future substantive allocation. Its main three-bank fixture is
178716672bytes (about170.4MiB), with errors/features adding about8MiB. The
combined sequence's numerical-array estimate is conservatively below500MiB,
plus interpreter/runtime overhead; peak memory and timing are **unmeasured**.
There are now **46 new EP17 methods unrun**, including the modified existing
method counted once. The original20 remain in the pending EP21 companion
proposal; the latest26 remain outside its unchanged625-second tail. Changed
ASTRA validation also remains unrun. No scheduler call, test, import, private
payload access or scientific score occurred this turn.

Predictor persistence, complete freeze/decision integration, qualified empirical
features/responses, enforced reader separation and the actual scientific
search/audit remain unfinished. These implementations supply required scoring
components, not evidence of a model-ranking effect or audit readiness.

## 2026-10-09 — prediction_qualification selects frozen audit prediction application

Select the missing array-only application of already fitted development
predictors to a caller-supplied audit panel. Bind the existing frozen audit
relation, its selected P1/P2/P3 pair and calibration family before application.
Require the complete606-image/120-concept audit design, explicit12-block order,
the source metadata's `sealed_audit` role and disjoint development identities.
Wrap each final predictor with its original response-column bindings; apply
stored PCA/ridge parameters and the shared all-development response means
without fitting, retuning, support changes or audit-mean substitution. Check
response variance and scoring denominators separately within the evaluator.

One worker owns the implementation, another reviews contract/numerical bindings,
and a third owns focused independent tests. Root owns integration and records.
Preserve the selected model's frozen rank64 features and at most two supplied
prelocked secondary check descriptors; their separate complete-family scoring
is not implemented by this primary application. No persistence, physical reader
release, audit-opening authorization, empirical application or automatic final
adjudication is selected. Actual one-shot access enforcement remains a caller
requirement. Tests require coordinated compute and remain outside the unchanged
pending EP21 proposal; no new allocation is authorized by this implementation.

`tools/audit_prediction.py` is implemented and independently source-reviewed.
It snapshots the two frozen predictors, matches all original coefficient and
response columns, verifies equal stored development means, checks the entire
606-image response and every block for centered numerical variance, and then
applies the stored PCA/ridge parameters. Whole-block SSE/SST bind to the existing
audit endpoint family; the single D bank is shared. It retains one final rank64
feature bank per selected model and the supplied secondary descriptors without
computing secondary scores or a scientific verdict. The caller must validate
the complete frozen relation's directions, thresholds and forecast before use;
this interface checks pair/family/inference-roster correspondence only.

Six methods in `tools/test_audit_prediction.py` are source-reviewed and
**unrun**. They compare the actual frozen forward calculation against an
independent sparse closed form, block errors and all seven primary endpoint
aggregates, and exercise consequential role, identity, coefficient, shared-mean,
variance and resource failures. Fitting, secondary kernels and adjudication are
explicitly prohibited in the numerical case. The suspected existing development
block-variance omission was a false alarm: that check already exists and the
shared nested implementation was not changed.

The unselected ASTRA recipe requests one CPU,2GiB and115 seconds plus5-second
termination grace within a future substantive allocation. The fixture's six
main banks total201056256bytes (about191.74MiB), with feature/coefficient arrays,
temporaries and interpreter overhead additional. At8558 total original columns,
production's three internal response/prediction banks total373402656bytes
(about356.10MiB); the two models' copied rank128 coefficient banks add at most
52580352bytes. Caller inputs, possibly retained development banks, feature/PCA
arrays, block sums and temporaries are additional. These are source size
estimates, not measured peak memory, runtime or empirical resource admission.

All **41 new EP17 methods remain unrun**. The original20 remain selected in
the pending EP21 companion proposal; the latest21 remain outside its unchanged
625-second tail. Changed ASTRA validation is also unrun. Root's single Slurm
inspection at2026-10-10 00:57:08 UTC (October9 17:57 Pacific) showed an empty
user queue; no jobs were submitted. No tests, imports or private payloads were
executed/accessed this turn. Predictor persistence, audit secondary composition,
qualified empirical inputs, enforced reader separation, complete method freeze
and the actual scientific search/audit remain unfinished. This is application
implementation, not new evidence of a model-ranking effect.

## 2026-10-09 — prediction_qualification selects development secondary score composition

The primary endpoint/forecast compositions are implemented, but the adopted
six-support Fisher-z correlation sensitivity and rank64 linear-CKA falsifier
still lack their development caller. Select their complete array-only
composition under the existing specifications: all four registered models,
P1/P2/P3/F1 signs, four participants, two ROIs, B/C/D stages and the complete
calibration-eligible secondary family. Correlation is voxelwise before support
averaging. CKA uses each outer fold's encoding-selected layer and stored
training-only rank64 transform, then unique-image-weighted fold averaging.
The anatomical endpoint averages all32 individual subset CKAs; its inclusion
weights cannot substitute for those nonlinear calculations. There is no
ceiling-normalized duplicate or cross-scale pooling.

One worker owns the implementation, a second reviews resampling and numerical
semantics, and a third owns focused independent tests. Root owns prospective
implementation choices, records and later compute selection. Reuse the fixed
10000-draw crossed-bootstrap engine without changing its draws, family, margins
or failure rules. Bind separately supplied frozen block order and original
anatomical bank membership before any statistic. No empirical source, pixel,
feature, neural outcome, audit opening or model refit is selected. Tests remain
unrun and outside the pending EP21 companion tail; its resource question still
has no answer. This work creates no new job or scientific finding.

Root and independent method review adopt the explicit secondary bootstrap
composition now recorded in the prediction specification: integer whole-block
image weights, centered weighted Pearson/Fisher on pooled OOF images, and
foldwise CKA under the unchanged training transform, weighted by resampled
image mass. A metadata-zero-mass fold contributes zero; a positive-mass
undefined statistic invalidates the whole draw with no redraw or deletion.
This is the original estimator applied to replicated blocks; no candidate
outcome or fitted feature transform chooses the convention.

The same self-contained proposal was sent to BR's read-only scientific-review
router. At00:45:47 UTC it returned `ext_review_dir_dbe2dbf2ed91`: general KG
criteria for within-fold transforms, resampling units, grouping and avoiding
pseudoreplication, with no substantive verdict or local artifact inspection.
Those relevant criteria inform the independent source review. No formal
verdict submission, canonical binding, launch approval or additional generic
validation campaign follows from the directive.

`tools/development_secondary.py` is complete. It binds all required development
rows, original columns, model/fold selections and actual32-bank membership;
applies each stored selected-layer rank64 transform only to its held-out rows;
and returns separate Fisher-z and CKA statistics for all four fixed signed
pairs. The caller retains the frozen eligible six-support family and delegates
each scale/pair's bounds to the unchanged10000-draw engine. No normalized CKA
duplicate, response-column whitening, support-specific model fit or score-
driven family deletion is introduced. Resource callbacks precede snapshots,
transforms, subset copies and numerical kernels.

Independent mathematical source review and root review found no remaining
concrete issue. Five methods in `tools/test_development_secondary.py` are
source-reviewed and **unrun**. Small numerical cases compare with literal
replicated rows; an analytic32-bank contrast has equal inclusion counts but
different mean CKAs, detecting substitution of average weights. Full-roster
routing instruments expensive kernels and checks all scales/pairs, selected
layers, resampled image mass, absent people/folds, fixed bank bindings,
retained-data failures and resource refusal. It does not execute a new10000-
draw qualification campaign or establish empirical feature rank/performance.

The unselected ASTRA recipe is one CPU,2GiB,175 seconds plus5-second termination
grace inside a future substantive allocation. Its broadcast-backed fixture
retains1190952960bytes (about1.11GiB) in five numerical snapshots, before small
transforms and temporaries. At the8558 structural-column ceiling, production's
five snapshots alone total2488324080bytes (about2.32GiB), alongside caller/
nested banks, features, coefficients and kernel temporaries. These are source-
based size calculations, not measured RSS or runtime; the full repeated CKA
cost remains to be measured before resource selection.

All **35 newly added EP17 methods remain unrun**: the original20 are in the
pending EP21 companion proposal; the10 preceding composition checks and these
five secondary checks remain outside its unchanged625-second tail. Changed
ASTRA validation remains unrun. No test/import, scheduler call, private
payload or actual secondary score was executed this turn. Predictor persistence,
audit application, empirical feature/response qualification, enforced reader
separation and the scientific search/audit remain unfinished.

## 2026-10-09 — prediction_qualification selects prediction-to-endpoint and forecast composition

The complete nested predictor and primary endpoint evaluator still require a
caller that converts the common OOF predictions into their whole-block error
sums. Select that missing development-only array composition: all2423 supplied
images, all four registered models, all eight participant/ROI groups and all
B/C/D stages, with one common per-image outer-training baseline. Compute each
model's block SSE and shared SST before forming the fixed P1/P2/P3/F1 signed
pairs; preserve original columns, frozen block identity, all images and the
single D prediction bank. Bind only to an already frozen endpoint family.
Do not average fold R2 or give repetitions additional weight.

In parallel, select the existing contract's development-only explanatory
composition where fully specified: beta12/12-block cross-fitting and24-block
refitting, and pooled-development support/ceiling forecasts from the fixed
operators. Keep beta's equal-block forecast target distinct from the primary
pooled-image score. Do not invent a size-control explanatory forecast, change
eligibility, or infer a final audit verdict. CKA still needs its actual32-bank
membership and separate complete computation; mean inclusion weights cannot
replace the bank for this nonlinear statistic.

One worker owns each new implementation; root owns integration and records.
All work uses source and supplied-array interfaces only. No private payload,
feature extraction, empirical fitting/scoring, reader release or audit access
is selected. New focused checks require coordinated compute and are not added
to the pending625-second EP21 companion tail. The existing EP21 resource
question remains unanswered; no job submission follows from this continuation.

`tools/development_endpoints.py` and its four independent tests are complete.
The bridge requires the separately frozen metadata block order, preserving
integer/string identities instead of inferring order from image appearance.
It verifies every stored per-image baseline against the corresponding
other-five-fold response mean, computes each model's block SSE once and one
shared SST, then binds all four signed pairs to the existing endpoint family.
Group/stage snapshots and resource admission precede array copies and numerical
work; failed admission or malformed inputs return no partial comparison.
The endpoint module's block-label interface now preserves existing integer as
well as string labels while refusing Boolean aliases.

`tools/development_forecasts.py` and six focused tests are also complete.
The beta path checks exact reference-block order and original-column, spatial-
bin and gain-stratum bindings before the existing12/12 cross-fit and24-block
OLS refit. It combines the two ROI predictions/observations before taking
absolute errors, retaining4-by24 participant/block diagnostics. Support and
ceiling forecasts use pooled development advantages under the fixed operators;
beta keeps its distinct equal-block target. No cross-fit pass threshold or
forecast direction is invented. Recorded calibration inapplicability and the
unadopted size-control explanation remain explicit without zero-valued rescue.

Root found that the endpoint family previously validated but discarded its
spatial-bin vector, permitting a supplied gain design with different bin
intercepts to pass the available bindings. The family now retains that immutable
vector, and the forecast caller requires exact matching one-hot intercepts.
The targeted test moves columns between bins while preserving bin counts,
gain and strata. Independent final source reviews of both implementations and
tests found no remaining concrete issue after this correction.

All **10 new methods are unrun**, as are the earlier20 methods. The new ASTRA
recipe proposes a combined115-second test cap plus5-second termination grace,
one CPU and2GiB, inside a subsequently selected substantive allocation. The
bridge fixture's six principal banks total803893248bytes (about767MiB); actual
RSS/runtime are unmeasured. These10 methods remain outside the unactivated
EP21 proposal's existing625-second companion tail. Changed ASTRA validation
also remains unrun; no import, test, scheduler call or private payload access
occurred in this implementation turn.

Qualified features/responses, encrypted stimulus delivery, enforced reader
separation and measured empirical resource admission remain unresolved.
Predictor persistence/application, the separate correlation/CKA compositions,
scientific search and one-shot audit are not completed by these interfaces.
The original full episode objective and completed empirical calibration remain
intact; this is pipeline implementation, not a model-ranking finding.

## 2026-10-09 — prediction_qualification selects nested prediction-bank orchestration

The completed PCA/ridge/error primitives and endpoint/audit compositions do not
yet perform the adopted nested fitting workflow. Select the missing array-only
orchestration under the unchanged common-linking specification: all six outer
concept folds; the five remaining folds as inner validation; every registered
layer/rank/penalty setting; training-only feature transforms and response means;
pooled inner error sums before equal voxel/ROI/participant/stage aggregation;
the exact common-setting tie rule per model; and the all-development predictor
freeze. Each D-stage prediction bank is reused across support/ceiling endpoints.
The full common feature-grid feasibility requirement precedes neural selection.
No reduced production roster or grid, response-driven support or audit refit
is authorized by this implementation task.

One worker owns `tools/nested_prediction.py`; a second owns focused tests and
independent contract review. Root owns integration, records and any later
compute selection. Supplied arrays and metadata are an implementation interface,
not authenticated reader release. Source/feature qualification, actual physical
reader separation, resource admission and empirical selection remain separate.
No payload, model fit, source fetch, audit access or allocation is selected.
Existing numerical-core checks remain valid and will not be rerun unchanged.
New code checks are unrun and are not yet included in the pending EP21 batch.

The selected interface binds the already established2423 development images,
480 concepts, six80-concept folds and24 twenty-concept blocks; a matching count
does not authenticate their identities or roles. Check feature feasibility on
all22 distinct required training partitions and all four registered models and
layers before neural selection. Reuse one deterministic full-SVD rank128 fit
per partition/model/layer and derive ranks32/64 from its same loadings with
each exact-degeneracy cutoff checked and each training-score scale recomputed.
This removes redundant SVDs without whitening, changing the grid or using held-
out rows to fit a transform. Preserve supplied original-column calibration
bindings, explicit resource callbacks and incomplete status on any refusal.

BR was given this self-contained nested-fitting plan for focused leakage and
measurement-isolation critique. At00:27:55 UTC the read-only router returned
directive `ext_review_dir_66a7b35c3a59`, with general KG criteria for inner/outer
partition coverage, held-out conflicts, within-fold feature transforms and
subject/concept grouping. It did not read local artifacts or return a
substantive verdict. The independent reviewer applies those relevant criteria
to the planned code checks; no additional manifest scheme, synthetic campaign,
canonical review submission or launch approval follows from the directive.

`tools/nested_prediction.py` and six methods in
`tools/test_nested_prediction.py` are now complete. Two independent source
reviews found no remaining material issue after root identified and the author
fixed eager row-copy allocation before resource callbacks. Every allocating
row slice now requests admission first. The author independently reviewed the
test implementation. The complete procedure retains per-model OOF predictions,
one common baseline bank, all outer selections/CKA transforms and all-development
predictors; support-specific D refits and audit inputs are absent.

The source-implied full production schedule is352 full SVDs,290304 inner ridge
fits and672 refits. At the8558 structural-column ceiling with float64 caller
inputs retained, selected major resident arrays total approximately4.12GiB,
excluding coefficient banks, temporary arrays and runtime overhead. These are
conditional source calculations, not measured runtime or peak memory. The
existing empirical cost pilot, qualified features/responses and actual reader
separation remain necessary. Prediction persistence, bank-to-endpoint assembly,
explanatory forecast fitting and audit application are still separate work.

Select the six new tests inside the unactivated EP21 proposal's remaining
companion budget: combine them with the14 endpoint/adjudicator methods under
one235+5-second EP17 cap, retaining the proposed parent's1CPU/16GiB/two-hour
maximum. The new fixture's planned envelope is one CPU/up to7GiB, unmeasured.
The entire configured tail becomes625seconds within the existing630-second
allowance, with actual remaining-time admission and explicit deferral if short.
All **20 EP17 methods and changed ASTRA validation remain unrun**. Instrumented
full-roster tests check composition; they are not empirical rank qualification
or another research simulation campaign. No job or scientific execution is
authorized until the separate EP21 resource decision arrives.

## 2026-10-09 17:07 Pacific — prediction_qualification selects missing evaluator composition

The completed prediction, measurement and uncertainty primitives still lack
their contract-defined endpoint-statistic caller. Select an array-only
composition from the adopted measurement and prediction specifications:
seven primary endpoints and eligible H components, original-column support
and operator alignment, pooled image-level SSE/SST before R2, equal weighting
of all four participants and two ROIs, participant/block multiplicities and
one shared D prediction bank. Integrate with the existing crossed-bounds
interface without inventing margins, changing the family or retuning models.
Prepare focused checks and a separate frozen-audit adjudicator only where the
existing specifications fully determine its inputs and decision rules.

One worker owns the new evaluator module and checks; root owns records and
later compute selection. No reader, payload, credential, stimulus extraction,
feature execution, outcome opening, isolation claim or allocation is selected.
Private delivery, effective reader separation, response/feature qualification
and nested model-bank orchestration remain distinct dependencies. Existing
h5py/adapter checks stay valid and are not repeated. This source work cannot
be reported as an empirical ranking or audit result.

The worker completed `tools/endpoint_evaluator.py` and six focused methods in
`tools/test_endpoint_evaluator.py`; independent source/test review found no
remaining material issue. Review strengthened nonuniform inclusion and ceiling
fixtures and added explicit fixed signed model-pair identity. The statistic
now joins seven primary endpoints and eligible H components to crossed-bounds
recomputation while preserving original columns, pooled error sums, the common
D bank and equal four-person/two-ROI weighting. The beta block-mean forecast
target remains separate from the pooled-image endpoint difference.

Tests and changed ASTRA validation are **unrun**; the unactivated EP21 resource
measurement proposal contains their115+5-second companion recipe. No data or
allocation has been opened. Full nested prediction-bank orchestration and the
final audit adjudicator are still missing; this completed composition is not
the whole evaluator or empirical qualification. Existing core tests are reused.

While the separately proposed EP21 measurement awaits its resource decision,
continue the already selected source-only audit-adjudicator component in
`tools/audit_adjudicator.py` and its focused tests, restricted to rules fully
determined by the adopted specification. It must use supplied frozen relation,
signed components, forecasts and endpoint evidence, retain one common
at-least-three-person passing intersection and the leave-one-person/block and
adequacy gates, and create no audit access or fitted threshold. Underdetermined
rules must remain explicit rather than receiving a default. This component
is not yet complete or included in the EP21 companion recipe.

The selected primary adjudicator and eight focused methods are now complete in
`tools/audit_adjudicator.py` and `tools/test_audit_adjudicator.py`. Independent
source/test review found no remaining material issue after correcting width
adequacy to the frozen relation's required component set J. Every eligible H
still contributes to joint max-t inference and width diagnostics; unselected H
does not become an extra primary adequacy gate. Width is a negative-precision
criterion, not an additional positive-pass condition. The same-person
intersection, leave-out directions, signed bounds and frozen forecasts remain
separate, with beta block-mean and pooled primary targets kept distinct.

This caller applies only fully determined primary rules. A failed participant
or leave-out gate is not an operational test of stable heterogeneity, which
remains unspecified. An adequate equivalence/opposite pattern still needs
precise development qualification before a nonreplication claim. The anatomical
size-control edge remains in robustness inference but has no registered
explanatory forecast for promotion. These limits are explicit returns/refusals,
not invented scientific defaults. No actual audit or source data was opened.

Select the combined six endpoint and eight adjudicator methods for the same
proposed EP21 companion, within its unchanged115+5-second cap and existing
630-second total tail. The profile remains unactivated pending the separate
resource decision. All **14 methods and changed ASTRA validation are unrun**;
runtime and resource sufficiency remain unmeasured. Nested prediction-bank
orchestration, empirical qualification and effective reader delivery/separation
remain unfinished. This source completion does not establish a complete
empirical evaluator, model-ranking result or candidate readiness.

## 2026-10-09 16:51 Pacific — stimulus_features runtime repaired and all five adapter checks passed

The selected companion executed in the reverified sh04-14n31.int shell of
parent47144602, with993 seconds remaining at admission. Installation of the
single pinned h5py3.10.0 binary wheel into the fresh EP17-only package target
succeeded. Retained CPython3.11.15, NumPy1.26.4, Torch2.4.1+cu121,
Torchvision0.19.1+cu121 and Pillow12.2.0 version assertions passed. No existing
environment package was replaced. All **five source-free adapter tests passed
in0.412 seconds**, including actual HDF5 write/read behavior. Changed ASTRA
schema and semantic validation passed, retaining seven existing optional
output-format warnings. Installation, tests, validation and companion exits
were all zero.

Evidence is [the companion directory](runs/stimulus_runtime_repair_job_47144602/):
`install.json`, `install.txt`, `stimulus_adapter_tests.txt`,
`astra_validation.json`, `status.txt` and `resources.txt`. The complete
companion took17.20 seconds with peak74,452KiB. Parent accounting remains
outstanding once to EP20; no separate EP17 charge is added. The original import
failure is preserved; the five test bodies are now verified rather than unrun.
No private source pixels, credential, checkpoint, neural values or feature
pilot were accessed. Actual stimulus delivery, enforced reader release,
remaining response qualification and evaluator completion remain separate
empirical dependencies. This repaired component needs no unchanged rerun.

The parent subsequently closed COMPLETED,0:0 at16:56:21 Pacific. Observed
terminal accounting charges6571 allocated CPU-seconds once to EP20, replacing
its reservation. EP17 receives no duplicate parent or child charge; the compute
PTY is closed.

## 2026-10-09 16:13 Pacific — stimulus_features selects bounded runtime repair in existing allocation

The five source-free adapter checks remain unrun after the documented missing
`h5py` import failure. Root and an independent worker reviewed the existing
[pinned-wheel repair](stimulus_feature_adapter_specification.md#october-9-runtime-repair-proposal--no-installation-or-retry-selected).
Select that repair and those same five checks as a sequential companion in
the already owned EP20 parent **47144602**, one CPU, 7 GiB, zero GPUs. Dispatch
only after its foreground preflight terminates, the actual compute shell and
allocation are reverified, and at least **400 seconds** remain. No additional
allocation or CPU-hour reservation is selected. The parent remains charged
once to EP20; record this child without adding its CPU time again.

Run from this EP17 directory inside a subshell. Keep the retained
`encoding_model` CPython3.11.15 environment unchanged; install only the official
hash-pinned h5py3.10.0 wheel with `--no-deps`, binary-only, no cache or bytecode,
into the fresh EP17-only `$GROUP_HOME/users/zijiao/ep17_h5py_3_10_0_cp311` target.
Keep installation temporaries under `$L_SCRATCH_JOB`, retain partial evidence
on failure, and refuse an existing target. Installation is capped at180+5
seconds; unchanged checks and version assertions at120+5; changed ASTRA
validation at30+5. An inclusive350+5-second companion bound and400-second
admission leave closeout time. Do not retry automatically or change runtimes.

Retain installation/report, test output, validation and separate exit status
under `outputs/runs/stimulus_runtime_repair_job_47144602/`, creating it only on
actual dispatch. Confirm that five test bodies execute; loader success alone
is insufficient. The existing fixtures exercise real NumPy/Pillow/HDF5 on
generated JPEGs and fake feature backbones, with no acquired pixels,
credentials, neural payloads or model inference. Existing operator checks and
real calibration outputs are reused. Actual stimulus execution and reader
release remain unselected. Source review supports the resource envelope;
actual timing and memory remain unmeasured until execution.

The existing resource limits and consumed accounting remain unchanged. The
earliest logged compute origin, October3 00:00 UTC, is less than169 hours before
this parent's end, within the240-hour limit; no clock reset is introduced.
The ASTRA check recipe now binds the additive package target; its changed
validation is pending in the selected companion, not passing evidence.

## 2026-10-09 08:43 Pacific — stimulus_features missing-h5py repair recipe prepared, not executed

Read-only runtime metadata confirms the retained `encoding_model` environment
is Python3.11.15/NumPy1.26.4/Torch2.4.1+cu121/Torchvision0.19.1+cu121/
Pillow12.2.0/pip26.1.2 and contains no h5py distribution metadata. Sherlock's
`ml spider py-h5py` catalogue has Python2.7/3.6/3.9/3.12 builds, with no matching
Python3.11 module. No scientific import or environment modification occurred.

The [concrete repair recipe](stimulus_feature_adapter_specification.md#october-9-runtime-repair-proposal--no-installation-or-retry-selected)
proposes only the official h5py3.10.0 CPython3.11 Linux wheel in a fresh
EP17-only `$GROUP_HOME` target, then the unchanged interpreter with that single
explicit import path. Official package metadata supports the retained Python
and NumPy versions. The pinned wheel hash, `--no-deps`, binary-only installation
and exact existing-version checks preserve the model runtime. The proposed
compute-only envelopes are180+5seconds for installation and120+5seconds for the
same five source-free adapter checks, inside a future substantive allocation.
No package was downloaded or installed, no retry/allocation was selected and
no active job or recipe was edited. Actual image/feature access remains
unselected; private delivery and enforced reader separation remain unresolved.

## 2026-10-09 08:37 Pacific — calibration_support operators completed; stimulus adapter checks failed at import

The closed [companion status](runs/private_operator_checks_job_47101165/status.txt)
records `tests_exit=0`, `operator_exit=0`, `stimulus_adapter_tests_exit=1`,
`astra_exit=0` and `status=closed`. The [four operator checks](runs/private_operator_checks_job_47101165/tests.txt)
passed in **0.299 seconds**. The actual
[operator run](runs/private_operator_checks_job_47101165/operator.txt) and
[allowed operational report](runs/calibration_operators/attempt001/operator_report.json)
record `operator_artifacts_complete=true`, no errors, **8 participant/ROI
groups** and **16 beta group-edges** processed from completed private
calibration artifacts. Elapsed time was **0.380813 seconds**, peak RSS
**55,500 KiB**, and private output size **2,455,772 bytes**, within the declared
2-GiB/120-second/32-MiB limits. The output remains at the fixed private
`outputs/cache/private/calibration_operators/attempt001` path. This closeout
read only operational metadata, not those private arrays or applicability values.
Artifact completion does not assert that every explanatory edge is applicable.

The [stimulus fixture attempt](runs/private_operator_checks_job_47101165/stimulus_adapter_tests.txt)
failed before any of its five test bodies ran: importing
`tests/test_private_stimulus_feature_pilot.py:22` raised
`ModuleNotFoundError: No module named 'h5py'` in the selected `encoding_model`
Python3.11 environment. The unittest loader reports one failed import, not five
executed adapter checks. This is a concrete missing runtime dependency; no
adapter behavior was tested and no source fix or retry was made here. Correct
the selected runtime binding before rerunning the five source-free checks.
The completed operator work is not a retry target.

[ASTRA validation](runs/private_operator_checks_job_47101165/astra_validation.json)
passed schema and semantic checks, retaining seven existing optional
output-format warnings. The coordinator's **15:37 UTC** terminal observation
records shared parent **47101165 COMPLETED, 0:0**, one CPU, **11,875 elapsed
seconds**, ending **08:22:41 Pacific**. Those **11,875 CPU-seconds are charged
once to EP21**; EP17 receives no duplicate charge. Primary parent success does
not erase the separate stimulus-test import failure.

The operator report explicitly records no new raw neural or stimulus values,
candidate scores or development forecast fitting, no reader handoff or candidate
scoring authorization, and scientific audit count zero. The actual 32-image
pixel/feature pilot remains **unselected** pending authorized private delivery
and enforced reader separation. Existing source acceptance still needs no
repeat. No credential, private applicability value or scientific outcome was
opened during this closeout; no tests, imports, scheduler calls, submissions or
source changes were performed. Earlier pending operator/ASTRA descriptions are
historical; the five stimulus test bodies remain unrun.

## 2026-10-09 04:33 Pacific — stimulus_features private callable implemented; source-free checks selected

The [callable](tools/private_stimulus_feature_pilot.py) implements the existing
fixed 32-development-image pilot, using the completed exact image/role joins,
same acquired archive and four registered backbones. It verifies that archive's
declared content identity before selected-member decoding, receives a secret
only through an explicitly inherited descriptor, and keeps JPEG decoding and
the transformed input bank in memory. The seed-17 random backbone is persisted
once or reused before image access. Four private HDF5 files share the unchanged
ordered index; no pixel copies, feature summaries or secret-bearing exception
text enter the operational report. Partial outputs remain incomplete, preserved
and unreleased. There is no CLI, automatic retry, namespace route or reader
release. Independent source review is complete after two concrete corrections:
only newly created owned directories have inherited setgid normalized to 0700,
and the cost projection includes all measured preprojection work, assigning
unclassified selection/index/container/flush overhead conservatively to the
per-image term. Existing directories are untouched. The final operational-report
write is explicitly outside that timing coverage; the projection is an estimate.

The [five focused fixtures](tests/test_private_stimulus_feature_pilot.py) cover
role/order refusal before source access, bounded descriptor ownership without
echo and new-directory-only setgid handling, exact selected JPEG streams without substitution, sanitized failures and
interruptions with preserved partial output, and identical private row ordering
across all four models. They use generated fixture images and fake backbone
outputs, with no acquired data/checkpoints or real credentials. All five and
the changed ASTRA validation remain **unrun**. Existing passing archive and
feature-core checks are retained.

Root selects only these five checks in the existing optional EP21 post-assembly
tail, using the previously qualified `encoding_model` Python3.11 runtime:

```sh
env -u PYTHONPATH PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' timeout --signal=TERM --kill-after=5s 120s "$GROUP_HOME/zijiao/miniforge3/envs/encoding_model/bin/python" -B -m unittest discover -s outputs/tests -p test_private_stimulus_feature_pilot.py -v
```

The test log will be
`outputs/runs/private_operator_checks_job_<jobid>/stimulus_adapter_tests.txt`,
with separate `stimulus_adapter_tests_exit` status. The existing four operator
checks, conditional real operator computation and ASTRA retain their limits.
The combined child envelopes are now 1,050 seconds; the optional tail is capped
at 1,150 plus ten seconds and requires at least 1,250 verified actual seconds
remaining. The parent keeps its four-hour, one-CPU, 7-GiB reservation and
14,130-plus-ten-second primary assembly limit. All work is sequential; primary
assembly status remains authoritative and the parent is charged once to EP21.
No separate allocation or empirical stimulus dispatch is selected.

The actual pixel/feature pilot remains **unselected** pending authorized private
delivery and the enforced reader boundary. Existing source acceptance remains
valid; no repeated acceptance or credential disclosure is requested. The source
retains the prospective one-CPU/4-GiB, 600-second internal and 660-plus-ten-second
external envelope, but actual decryption/inference throughput and memory remain
unmeasured. The callable needs the existing Torch2.4.1/Torchvision0.19.1/
Pillow12.2.0 binding plus NumPy and h5py; the new fixtures require NumPy/h5py/
Pillow while mocking inference. This edit ran no tests, imports, invocation,
payload or credential access, scheduler action or isolation probe.

## 2026-10-09 02:52 Pacific — calibration_support operator run bound to optional post-assembly execution

03:00 amendment: EP21 added twelve affected converter checks. The selected
parent tail now requires at least 1,100 seconds remaining and is capped at
1,000 plus ten seconds; EP17's own 325-second envelope is unchanged.
The original selection below is historical; no work has been dispatched.

The source-reviewed, unsubmitted EP21 assembly wrapper now selects the four
operator checks, then the existing private operator derivation only if those
checks pass, and changed ASTRA validation. Limits are 90 plus ten seconds,
180 plus ten seconds, and 30 plus five seconds respectively. The producer keeps
its internal 120-second/2-GiB/32-MiB bounds and fixed attempt001 output paths.
Companion logs will be `outputs/runs/private_operator_checks_job_<jobid>/`;
no private operator values enter them.

This sequence runs only after successful assembly with sufficient actual time
remaining, inside a 900-second plus ten-second multi-episode tail. All checks
and operator computation remain unrun. The eventual parent is charged once to
EP21, without a separate job or duplicate EP17 charge. Completed calibration
artifacts are its only scientific inputs; it grants no worker release, image
access, candidate fit or score. No further calibration-only experiment is
selected after these operators; model comparison still requires features and
enforced reader separation.

## 2026-10-09 01:55 Pacific — calibration_support remaining operator adapter source-reviewed

`tools/derive_private_operators.py` and four focused fixtures now implement the
[private operator stage](prediction_qualification_specification.md#builder-private-calibration-derived-operators)
from completed calibration arrays only. Root source review found no blocker
after separating the canonical positive ceiling map from its edge-difference
weights. Undefined beta gains or frozen rank/variance failures preserve every
original voxel and primary endpoint while recording explanation/H applicability
privately. No model scores or raw neural/image values enter the adapter.

ASTRA retains `calibration_support` as the owning analysis and the existing
downstream `operator_maps` reference, adding separate full-calibration and
derived-operator reports. GOAL/DATASETS now reflect completed copying and
calibration without suggesting reader release. The four adapter checks,
changed ASTRA validation and actual operator run remain **unrun**. Select
1CPU/2GiB,120internal/180external seconds plus10second kill grace and32MiB
new outputs within the next substantive allocation; no small standalone job
or additional simulation program is selected.

## 2026-10-09 01:42 Pacific — calibration_support attended allocation closed and charged

The coordinator exited the compute PTY after successful full calibration and
all nineteen serial EP20 checks plus its ASTRA validation. The **08:42 UTC**
`sacct -X` observation records **47090250 COMPLETED, 0:0**, one CPU and
**471 elapsed seconds**, from01:34:35 to01:42:26 Pacific. Charge **471
CPU-seconds (0.130833 hours)** once to EP17, replacing its two-hour reservation.
EP20 receives no duplicate charge. All earlier job charges are preserved;
no interactive allocation remains open. Empirical calibration and support
generation are complete within the declared limits; model comparison remains
dependent on stimulus/features, enforced reader separation and evaluator work.

## 2026-10-09 01:41 Pacific — calibration_support full empirical maps and supports completed

The full child exited zero inside attended allocation47090250. Its closed
[attempt001 report](runs/calibration_support/attempt001/calibration_report.json)
records **calibration_support_complete=true**, verified private arrays and no
errors: all **8558 original columns**, **605 fixed eligible images**, twelve
participant/stage products and **3258/3258 physical stripes** completed.
Elapsed time was **227.665114 seconds**, with **456524 KiB peak RSS**;
reading used33.420189seconds and numerical work188.413452seconds. Full private
outputs occupy5222991bytes; including the preserved pilot, accounted output
is5931467bytes under the256MiB cap.

Calibration used only private support-calibration responses. Development,
mixed-source and audit values remained unread, scientific audit count iszero,
and no model reader was activated. The unchanged N=256/q=32 support construction
completed; no data-dependent relaxation was needed. This closes actual B/C/D
calibration maps and A_all/A_N/R_N/V_NC support generation, not gain-operator
applicability, model-feature qualification, physical reader isolation or the
complete evaluator. No model-ranking result is claimed.

The coordinator proceeds to the already selected serial EP20 checks in the
same allocation, then closes it and charges its actual parent runtime once
here. Next empirical model fitting still needs authorized stimulus extraction,
qualified features/readout and an enforced reader boundary.

## 2026-10-09 01:36 Pacific — calibration_support pilot completed; full calibration selected

In verified attended allocation **47090250**, all six adapter checks passed
in 0.195 seconds and changed ASTRA validation exited zero. The fixed real-data
[pilot001](runs/calibration_support/pilot001/calibration_report.json) completed
in **10.406207 seconds**, with **456528 KiB peak RSS**, all 605 eligible images,
128 original columns and three stages. Its 72 physical stripes decoded at most
452984832 bytes; private reading took 0.896619 seconds and numerical work
4.082399 seconds. Saved private arrays passed byte verification. No development,
mixed-source or audit response values were read.

The same compute allocation's metadata-only
[full cost plan](runs/calibration_support/compute_job_47090250/full_cost_plan.json)
contains **8558 original columns, twelve products, 3258 physical stripes and
18924699648 decoded bytes**. Reading scales by at most 45.25 times the pilot
stripe count; a conservative numerical estimate uses the larger 66.859375
column ratio. Their estimated sum is about 314 seconds plus setup/support
construction. Select **900 internal / 960 external seconds plus ten-second
kill grace**, one CPU and the same 2 GiB limit, within the existing attended
allocation. The derived-output cap remains 256 MiB including the saved pilot.

Run `tools/calibrate_private_roles.py --mode full --max-seconds 900` once.
Do not change image eligibility, numerical thresholds or the N=256/q=32
support quotas if the actual data cannot support them. Terminal output will
be `runs/calibration_support/attempt001/calibration_report.json`; private
maps and supports remain builder-only, and no model-fitting reader is released.

## 2026-10-09 01:34 Pacific — role_materialization complete; attended calibration selected

The coordinator's **08:32 UTC** `sacct -X` observation reports **47087633
COMPLETED, 0:0**, one CPU and **2090 elapsed seconds**, ending at 01:29:07
Pacific. Charge **2090 CPU-seconds (0.580556 hours)** once to EP17, replacing
the three-hour reservation and retaining all previous charges. The closed
[attempt002 report](runs/role_materialization/attempt002/materialization_report.json)
confirms all twelve participant/stage products complete and every routed
copy byte-equal. The child took 2071.274465 seconds with 456080 KiB peak RSS;
accounted output including preserved attempts is 155009890920 bytes, below
160 GiB. Scientific audit evaluations remain zero. Role-filtered copying does
not establish physical reader isolation or activate candidate scoring.

Select one attended **1 CPU, 2 GiB, at most two-hour** allocation for the
already specified fixed real-data calibration pilot and, only after reviewing
its measured cost and success, full calibration if it fits the remaining
allocation. The six adapter checks and changed ASTRA validation run in the
same allocation. The serial EP20 raw-time/array checks may share it after
their source is stable; charge this parent once to EP17. No standalone
checks-only job, padded runtime or automatic full escalation is selected.
The independently source-reviewed pilot wrapper is
`tools/run_private_calibration_pilot_001.sh`; the coordinator will choose
the full-mode deadline from actual pilot timing and decoded work. Preserve
private calibration outputs; no development or audit values are admitted.

The attended `srun --pty` request returned **47090250**, initially queued
waiting for resources. Its reservation is two CPU-hours; actual cost will be
reconciled once when the parent closes. The coordinator owns the PTY and will
verify its actual compute hostname and Slurm environment before execution.
Do not duplicate this allocation or run its children on the login node.
The actual PTY subsequently verified `sh04-04n21.int`, job47090250, one CPU
and 2048 MiB before starting any checks or calibration.

## 2026-10-09 01:20 Pacific — calibration_support private adapter source-reviewed; fixed pilot prepared

`tools/calibrate_private_roles.py` and six focused checks in
`tests/test_calibrate_private_roles.py` are implemented. Independent source
review found no blocker in full-copy report binding, exact row-map/chronology
reuse, complete eligible-image roster, physical stripe planning, numerical
core calls, support eligibility, private output limits or public error
redaction. No source response payload was opened by implementation/review.
The changed ASTRA objects and new checks remain unrun on compute.

The [selected cost-pilot specification](prediction_qualification_specification.md#builder-private-calibration-adapter-and-fixed-cost-pilot)
uses the first128 ascending sub01-left original-column IDs, all605 fixed
eligible calibration images and B/C/D. Calibration copies retain2129 total
rows per stage. One decoded stripe is bounded by three padded1024-row chunks
times512columns times4bytes =6,291,456bytes. The actual touched-stripe count
is frozen before values; the conservative pilot upper bound is384stage-stripes
and2.25GiB decoded work. Metadata joins previously used about445MiB RSS; the
new pilot keeps1CPU/2GiB,300internal/360external seconds plus10second grace,
and64MiB outputs. Full calibration needs measured cost and a separate duration;
no automatic escalation or job submission is selected here.

Public progress includes only operational counts/timing/RSS/status. Private
arrays preserve response-derived flags/reliability/ceilings and supports;
calibration completion does not establish gain-operator applicability, physical
isolation or evaluator readiness. OutputBudget reserves1MiB for the terminal
report; full's256MiB cap also includes preserved pilot artifacts.

At the coordinator's **08:14:38UTC** scheduler observation, copy47087633 was
RUNNING at1221elapsed seconds. Do not alter its sources, duplicate it or inspect
active private arrays. After successful full-copy closeout, reuse an appropriate
coordinated substantive allocation for the six new checks and fixed real-data
pilot; combine the thirteen prepared EP20 checks and changed ASTRA validations
as serial children, with parent cost charged once. Choose full calibration only
after reviewing pilot success, exact storage work and measured runtime/RSS.

## 2026-10-09 01:10 Pacific — calibration_support trusted-builder sequencing clarified; empirical adapter selected

The coordinator checked GOAL's trusted-builder permission and SEARCH_POLICY's
calibration-only reader/use clauses against the prediction specification.
The table's former “after reader release” wording unnecessarily coupled private
calibration to later search-worker release. The
[clarified specification](prediction_qualification_specification.md#october-9-clarification-builder-calibration-precedes-worker-release)
and ASTRA now connect calibration to successful private full copying and fixed
assignment/eligibility/chronology metadata. All adopted numerical conventions,
N=256/q=32 support quotas, margins, image rosters and access roles are unchanged.
Candidate fitting and scientific evaluation still require their actual reader
boundary; no physical isolation or reader release is inferred from permissions.

Implementation is selected for an empirical adapter to the existing tested
calibration/support cores, restricted to the completed attempt002 calibration
views and exact original-column structural bins. The current copy47087633 is
an active writer and is not opened by the coordinator or implementation agents.
Derived arrays will remain builder-private; public reports will contain only
operational progress/resource information. No new data access, compute
allocation or calibration result is selected by this implementation entry.
Compute tests and bounded calibration execution await completed source,
reviewed read scope and resource plan. Earlier 33 numerical checks are reused.

## 2026-10-09 00:53 Pacific — role_materialization full copy accepted as 47087633; attended preparation closed

`sbatch --parsable outputs/tools/run_neural_columns_full_001.sbatch`, submitted
from this episode's root, returned **47087633** with exit zero. The selected
one-CPU/2-GiB/three-hour reservation is now accepted. At **07:54:51 UTC**,
`squeue` and `sacct -X` show **RUNNING** on `sh04-04n24`, with start 00:54:17
Pacific and 34 elapsed seconds. The wrapper confirms one CPU and 2048 MiB;
full-copy completion remains unobserved. Do not duplicate this accepted job.
Its actual Slurm logs are
`runs/role_materialization/full_columns001-47087633.out/.err`, and operational
status/capacity/child output will be in
`runs/role_materialization/full_columns_job_47087633/`. The full terminal
report is `runs/role_materialization/attempt002/materialization_report.json`;
absence of that successful report is not completion or a released handoff.

Final independent wrapper review passed after making a failed virtual-memory
limit explicitly stop the child. Bash syntax and changed ASTRA schema/semantics
passed on compute; all six column checks and the actual pilot passed. No check
rerun is selected for the full job. Source and destination layouts, capacity,
pilot binding and finite output/deadline bounds remain checked by the copier.

The coordinator exited the attended shell. Fresh `sacct -X` reports
**47086577 COMPLETED, 0:0**, one CPU, **1143 elapsed seconds**, from 00:33:52
to 00:52:55 Pacific. Charge **1143 CPU-seconds (0.3175 CPU-hours)** once to
EP17, replacing its two-hour reservation and preserving all earlier charges.
This includes the serial EP19 scoring and EP20 contact checks, which receive
no duplicate charge. The current three-hour full-copy reservation is separate.
No interactive test allocation remains open. Candidate scoring, physical
reader separation and a scientific audit remain unopened.

## 2026-10-09 00:51 Pacific — role_materialization pilot003 complete; full column-copy attempt002 selected

All **six column-copy checks passed in 0.289 seconds** in
`runs/role_materialization/compute_job_47086577/column_tests.txt`.
Independent module and fixture review found no blocking defect. The
[pilot003 report](runs/role_materialization/pilot003/pilot_report.json) is
**completed**, with all copied bytes equal, 26,173,440 logical source bytes,
33,926,074 private output bytes, **5.270826 seconds** elapsed and
**456,100 KiB peak RSS**. Source reading took 0.506150 seconds and
route/write/readback took 0.228693 seconds; source-header inspection took
0.004008 seconds. Metadata/setup is not treated as per-stripe transfer time.

This trusted pilot **did read neural values and raw audit-response bytes**
within the selected sub-01/B all-trial, first-512-voxel scope. No value summary,
model feature, calibration score or scientific audit evaluation was released;
scientific audit count remains zero and reader handoffs remain inactive.
The failed pre-access pilot001 and pilot002 reports are preserved.

The coordinator now selects full **attempt002**, all twelve products with
5169 fixed column stripes and pilot003 reuse. Extrapolating the measured
0.734843 seconds per stripe gives about **3798 seconds**; this is a planning
estimate, not a throughput guarantee for other stages or filesystem conditions.
Select **9000 seconds internal**, **9060 seconds external plus ten-second
termination grace**, within a fresh **1 CPU, 2 GiB, three-hour** Slurm job.
This is more than twice the measured transfer estimate with setup allowance.
The pilot used approximately 445 MiB RSS and full stripes retain the same
maximum buffer; the 2 GiB virtual-address cap remains in force. The current
attended allocation has insufficient remaining time for that doubled estimate
and will close; it is not extended or automatically renewed.

Reserve three CPU-hours once to EP17 for
`tools/run_neural_columns_full_001.sbatch`. Its fresh quota observation reports
Oak 171.5/250 TB without a cached-data warning; exact-parent free space is
86,276,024,791,040 bytes. At **07:51:25 UTC**, retain a conservative 77 TB
quota-headroom bound and 560 GB for concurrent work. GROUP_SCRATCH remains
unavailable and is not used. Bind this assessment to the actual new job ID
at startup. The 160 GiB output cap, exact padded-storage accounting, original
roles/axes/bytes, source-chunk reuse, operational-only reporting and closed
reader handoffs remain unchanged. No successful full copy, new scheduler ID
or scientific result is claimed at selection. No automatic retry is selected.

## 2026-10-09 00:43 Pacific — role_materialization column-copy padded storage fits existing cap

The [metadata-only projection](runs/role_materialization/compute_job_47086577/column_storage_plan.json)
uses all frozen rows, including ineligible images, with participant/role counts
`sub-01: 8519/2129/2132`, `sub-02: 8523/2126/2131`,
`sub-03: 8518/2132/2130`, and `sub-06: 7762/1938/1940`
(development/calibration/audit). Across 36 full outputs, allocated chunk payload
is **154,958,561,280 bytes**, above the logical 132,332,156,640 bytes because
edge chunks are padded. Add 1 GiB for HDF5 metadata, full routing CSVs and
engineering slack, 256 MiB for the preserved pilot, and 1 MiB for reports:
**156,301,787,136 bytes**, below **171,798,691,840 bytes (160 GiB)**.
This allowance is a planning bound, not a mathematical guarantee of every HDF5
metadata byte; the hard aggregate file-size guard still applies at runtime.
No neural values were read by this calculation.

The changed EP17 ASTRA program passed schema and semantic validation in
`compute_job_47086577/astra_validation.json`; its eight legacy missing-format
notices are optional in the pinned version and are not execution failures.
The later column-pilot program change will need its own changed-object
validation. Parent47086577 remains active; no terminal cost is inferred.

## 2026-10-09 00:41 Pacific — role_materialization column traversal amendment and pilot003 selected

The header-only report for all twelve named source products confirms every
file uses `<f4`, gzip level 3, and chunks `(participant_trial_count,1,1,1)`.
Two independent workers reviewed the column-oriented plan. The coordinator
adopts this engineering correction before any neural values have been read:
read all source trials for consecutive **512-voxel stripes**, routing rows by
the same frozen metadata, without any change to eligibility, role, source
identity, values, dtype, axes, estimator or scientific question.

Select the fixed outcome-blind **pilot003: sub-01/B, all 12,780 source rows,
original voxel columns [0,512)**. This is 26,173,440 logical and decoded bytes,
512 complete source chunks. Retain the original one-worker, 2 GiB child,
120-second internal/180-second external plus ten-second grace, 256 MiB decoded
and total-pilot-output limits. The scope change is explicit: it replaces the
unworkable first-64-rows/all-voxels pilot and includes the original roles of
all these rows. Trusted copying may access raw audit responses; it releases
no values or scores to the controller and consumes no scientific audit.
Preserve pilot001 and pilot002 stopped reports and their unused private paths.

The new `materialize_neural_roles_columns.py` implementation and focused tests
are being prepared. Destination chunks align with the traversal:
`(min(1024,role_rows),1,1,min(512,voxel_width))`, uncompressed ordinary HDF5,
0700 directories and 0600 files. Every routed stripe must pass persisted
bitwise readback. Calculate padded output allocation from frozen role counts
before values; retain aggregate runtime file-size limits and a bounded metadata
allowance. The coordinator selects that metadata-only calculation in the
current allocation, plus the new tests and pilot003 when ready/reviewed.

The full column-copy **attempt002 remains unselected** until a successful
pilot supplies measured throughput, RSS, byte equality and storage overhead.
Its proposed scope remains all twelve products, 132,332,156,640 logical bytes,
5169 stripes, and 160 GiB total including preserved attempts and partials.
Reuse pilot003's first 512 sub-01/B columns through ordinary reads/writes,
without rereading those source chunks. The old full row-copy implementation
remains unselected; no cap increase, automatic full launch or reader release
is authorized by this amendment. Allocation47086577 remains the one attended
two-hour parent; no duplicate allocation is requested.

## 2026-10-09 00:37 Pacific — role_materialization nine checks pass; pilot002 stops at actual source chunk layout

The attended shell is verified on `sh04-04n24.int`, allocation **47086577**,
one CPU and 7168 MiB. All **nine new checks passed in 0.432 seconds** in
`runs/role_materialization/compute_job_47086577/tests.txt`. Independent source
review found no blocking copy/helper defect. The directory fix worked.

The [pilot002 report](runs/role_materialization/pilot002/pilot_report.json)
stopped with `source_layout_exceeds_decoded_bound`, before any neural read,
after 4.328955 seconds (peak RSS 454,496 KiB). Its metadata and capacity
checks passed. No full copy or further payload retry is selected.

A bounded, compute-node inspection of only the allowed source dataset's header
is recorded in `compute_job_47086577/source_layout_diagnostic.json`: sub-01/B
has shape `(12780,1,1,227214)`, dtype `<f4`, gzip level 3, and chunks
**`(12780,1,1,1)`**. Reading the proposed 64 rows across every voxel intersects
227,214 complete-trial chunks and entails **11,615,179,680 decoded bytes**,
above the unchanged 256 MiB bound. Raising that bound would not fix the repeated
decoding cost of the proposed row traversal.

Select a header-only inspection of the same twelve already named B/C/D products
to determine whether a column-oriented literal-copy plan can preserve all rows,
roles, axes and bytes with bounded work. This permits headers only, no numerical
dataset reads. A new resource pilot and traversal require an explicit reviewed
engineering amendment before execution. The prior full row-copy runner remains
unselected. Reader isolation, scoring and audit access remain closed.

Two independent prepared components may use the same allocation serially:
EP19's five scoring checks (60-second bound), EP20's six spatial-contact checks
(120-second bound), and changed EP17/19/20 ASTRA validation (30 seconds each).
Their scopes are recorded in their own logs. They introduce no source payload
or empirical scoring; parent cost remains charged once to EP17. No new job,
automatic successor or repeat of passing generation is selected.

## 2026-10-09 00:33 Pacific — role_materialization attended allocation 47086577 queued

The selected `srun --pty` request returned **47086577**, queued and waiting
for resources. The coordinator owns its attended terminal; a hostname and
resource check must execute inside the allocated shell before any test or
copy command. No startup, test or payload read is observed yet. Preserve this
single request rather than submit a duplicate. Its selected resource envelope
and conditional full-copy decision are recorded below.

## 2026-10-09 00:33 Pacific — role_materialization pilot002 and focused checks selected in an attended allocation

Select one attended **1 CPU, 7 GiB, two-hour** allocation on the freshly
discovered `russpold,normal` partitions with `CPU_GEN:SIE`. Reserve two CPU-hours
once to EP17; actual elapsed allocation time replaces this reservation when
terminal. The allocation is for the corrected private-copy pilot and, only if
its measurements and independent source review support a separately recorded
finite execution bound, the substantive twelve-product materialization.
There is no automatic full-copy dispatch. If the pilot fails, stop affected
work and preserve its report rather than filling the allocation with repeats.

First run the **nine new checks** for private-directory creation, explicit
pilot selection, literal full-copy routing, bit preservation, layout and storage
bounds, under a 120-second external limit. Reuse the old ten passing checks of
the unchanged pilot module. The fresh `materialize_neural_roles_pilot002.py`
binds only this process to pilot002 paths and the reviewed new-directory helper;
the original module and pilot001 report remain intact. Select the same sub-01/B
rows [0,64), 2 GiB child cap, 120-second internal and 180-second external deadline
plus ten-second grace. The 00:30 directory repair succeeded: exact mode is now
0700, still owned by zijiao, with no content or other path changed.

Before the pilot, bind a fresh capacity record to the actual allocation ID,
using the coordinator's 07:15:26 UTC capacity assessment (77 TB conservative
quota headroom, 86,288,309,837,824 filesystem bytes and 560 GB concurrent
reservations). Pilot002 has its separate 256 MiB cap. The possible later full
copy retains the 160 GiB cap and needs a new explicit selection after measured
pilot costs. No neural values, copy completion or scheduler acceptance is
claimed by this prospective record; no reader release or model score is opened.

## 2026-10-09 00:30 Pacific — role_materialization narrowly scoped directory repair selected

The coordinator and independent worker confirmed the task-created
`outputs/cache/private` directory is empty, owned by zijiao, and mode 2700.
Select removal of only its inherited setgid bit, retaining exact owner-only
0700 access. This grants no new read/write access and changes no source or
existing output. Future directories will be set to exact 0700 immediately
after creation, with unexpected existing modes still refused. Preserve the
original pilot001 program and failed report; implement a fixed pilot002
entrypoint for the correction and a focused setgid-inheritance regression.
The full-copy runner remains unselected until actual successful pilot costs.

## 2026-10-09 00:27 Pacific — role_materialization pilot stopped before neural access at inherited directory mode

The [terminal pilot report](runs/role_materialization/pilot001/pilot_report.json)
records `stopped`, `pilot_copy_complete=false`, and
`private_directory_permissions_invalid`. Metadata validation completed in
4.868890 seconds; elapsed pilot time was 4.893398 seconds and peak RSS
455,980 KiB. No neural read was attempted, no neural or raw audit values were
read, and no scientific audit was evaluated. The capacity observation passed.

The coordinator's bounded metadata inspection found `outputs/cache` mode
2770 and its newly created, user-owned `private` child mode **2700**. The
role-materialization and pilot001 private directories do not exist. The
parent's setgid bit was inherited by `mkdir(mode=0700)`; the helper requires
exact mode 0700 and therefore refused 2700 before opening the source. This is
a directory-creation defect, not a quota, source-data or payload-copy failure.
Preserve the failed public pilot001 report. Prepare a focused correction that
sets exact mode on newly created private directories, retaining refusal for
unexpected pre-existing permissions, and use a fresh pilot identity on retry.
 No retry has yet been selected or submitted.

Parent **47085635** ran one CPU for **43 seconds** and ended `FAILED`, `2:0`,
at 00:24:09 Pacific, observed at 07:26 UTC. The full parent cost is charged
once to EP20, which successfully generated its 84 worlds before this child;
prior EP17 charges remain unchanged. Full copying remains unselected until a
successful pilot supplies actual layout, copy timing and memory evidence.
Reader isolation, encrypted image delivery and model scoring remain unfinished.

## 2026-10-09 00:19 Pacific — role_materialization pilot accepted as the final child of 47085635

EP20 parent **47085635** was accepted at **07:19:42 UTC**. The previously
adopted private pilot runs only after successful EP20 generation; its limits,
source-row scope and operational-only report remain unchanged. Startup and
copy execution are not yet observed. The capacity file binds to this actual
Slurm ID at child startup. The public result remains
`runs/role_materialization/pilot001/pilot_report.json`; a parent completion
without that successful report is not a completed pilot. Any pilot failure
stops the parent, with no later scientific I/O. Parent cost is charged once
to EP20; previous EP17 costs remain intact. No full copy or reader handoff
is opened by this acceptance.

## 2026-10-09 00:17 Pacific — role_materialization unchanged private pilot selected for shared generation allocation

The coordinator selects the previously adopted sub-01/B rows [0,64) pilot as
a serial child of the forthcoming EP20 generation allocation. Fresh capacity
evidence and unchanged limits are recorded below. The parent requests one CPU,
7 GiB and two hours; this child retains its 2 GiB virtual-address cap,
120-second internal deadline and 180-second external limit plus ten-second
grace. Its existing ten passing checks and ASTRA validation are reused.
The capacity record will bind to the actual new Slurm ID before the child.
The parent cost is charged once to EP20, preserving previous EP17 charges.
No new job has yet been accepted and no neural copy has run. Full copying,
reader handoffs and model scoring remain unselected/closed.

## 2026-10-09 — `role_materialization`: current capacity observation resolves prior stop; pilot resume prepared

The coordinator supplied a fresh official `sh_quota` observation at
07:09 UTC: Oak reports 171.5 TB used of 250 TB, with no cached-data or timeout
warning. Its current `df` observation for this episode's exact
`outputs/cache` parent reports 86,288,309,837,824 available bytes of
274,877,906,944,000 bytes. The October 4 stop was the absence of current
quota information before any copy launch; it was not a failed source read,
failed write or evidence that the filesystem was full. These new observations
resolve that specific prerequisite without changing the copy method or
repeating the earlier failed quota query in this worker.

The existing October 4 22:30 UTC adoption selects only sub-01 stage B source
rows [0,64): 58,166,784 logical source bytes, a 268,435,456-byte decoded/output
cap, one CPU, 2 GiB, a 120-second internal deadline and 180-second external
deadline with 10-second termination grace. Full001 metadata assignments
completed on October 4. The unchanged runner's ten focused checks passed in
0.035 seconds in allocation 46524139; its ASTRA schema/semantics also passed.
These results are reused, with no new runner or test allocation. Source layout
and exact metadata prerequisites remain checked by the runner before copying.

The coordinator plans to select the existing pilot as a serial child of the
forthcoming substantive EP20 one-CPU, 7 GiB, two-hour world-generation job.
Impose the original 2 GiB child virtual-address limit with `ulimit -v 2097152`,
retain the four required numerical thread variables at one, and invoke the
existing command from this episode's directory:

```sh
(
  ulimit -v 2097152
  exec timeout --signal=TERM --kill-after=10s 180s python3.12 -B \
    outputs/tools/materialize_neural_roles.py --mode pilot \
    --capacity-observation outputs/runs/role_materialization/capacity_observation001.json
)
```

Before dispatch, the capacity JSON must bind its exact absolute `parent` to
this episode's `outputs/cache`, its `slurm_job_id` to the actual new allocation,
and `observed_at_utc` to the actual UTC observation timestamp. Use
`quota_checked: true`, the observed 86,288,309,837,824 filesystem-available
bytes, and a conservative 77,000,000,000,000-byte quota-headroom lower bound,
below the rounded approximately 78.5 TB reported headroom. Reserve
560,000,000,000 concurrent bytes for EP21's 500 GB cap plus EP20's 20 GB cap;
this also covers those caps if interpreted in GiB. The pilot's own 256 MiB
cap is checked separately. No capacity record is fabricated without the
actual job identity, and this preparation entry does not claim dispatch,
neural access, a copied file or a new scheduler allocation.

GOAL and SEARCH_POLICY retain the first development round's 6,000 CPU-hour,
256 GPU-hour, 240-wall-hour and 4 TiB limits. They record no separate fixed
wall-clock origin; no candidate search has begun. Even conservatively counting
from October 3 00:00 UTC, the earliest logged compute date, October 9 morning
is within 240 hours. This is not a clock reset or budget extension. Charge
the forthcoming shared allocation once to EP20, retain previous EP17 charges,
and disclose this EP17 child in its operational accounting.

The full twelve-product copy remains unselected pending pilot costs and all
layout bounds. Builder-private copying does not activate reader handoffs,
physical isolation, source-role completion, candidate scoring or an audit
evaluation. This worker made documentation/source inspections only on the
login node: no payload, secret, test, import, scheduler action or quota probe.

## 2026-10-06 — completed metadata and conventions reconciled into DATASETS

Propagated the existing 720-concept/3,634-qualified-image universe, fixed
480/120/120 roles, anatomical column mapping and N=256/q=32 quotas, adopted
calibration/margin/endpoint conventions, 33 passing numerical checks and exact
EP18 overlap into DATASETS, GOAL readiness and the paper-plan next action.
The provider password filename is documented without opening a secret.
Remaining delivery, capacity, physical readers and empirical calibration are
kept separate from those completed facts. The coordinator also authorized
synchronizing stale policy nulls to the exact October 4 23:24 adoption:
relative response variance formula, Fisher-z boundary tolerance, repeat-aware
ceiling formula/complete-roster eligibility, 0.10 stage floors, five practical
margins and four full audit-width thresholds. The prediction specification's
stale “proposed/unset” descriptions now reflect that adoption. No new values,
data role, access authority, payload read, job or test was selected; this
repairs the record of already-adopted decisions.

## 2026-10-06 Pacific — provider stimulus interface documented without secret access

At the user's request, added the [verified pinned source README](https://raw.githubusercontent.com/courtois-neuromod/things.stimuli/f8f5162f79fb2e9fb0d8cbb8bac610db11a54180/README.md)
to the existing source-role and stimulus-adapter handoffs. It explicitly names
`password_images.txt` for `images_fmri.zip`, resolving the generic question
of what interface the provider documents. Its presence in an authorized local
delivery path and successful bounded use remain unverified. No password/key
file, secret-bearing receipt field or pixel was inspected. Existing accepted
terms are preserved; no repeated acceptance or new margin approval is added.

The [source note](source_roles_proposal.md#october-6-public-source-update--documented-stimulus-password-interface)
keeps file-interface identity distinct from availability, access conditions,
reader isolation and empirical release. No policy, budget, claim, source role
or data-access boundary changed. Work on sh04-ln04 without allocation was
limited to documentation edits/readback; no install, import, test, scheduler
action, download archive, decryption, external contact or job occurred.

## 2026-10-06 10:57 Pacific (PDT) — supplied review reconciled; access recommendations unadopted

Revised the existing `source_roles_proposal.md` handoff from current permitted
documentation. The October 4 metadata roles/overlap are complete; empirical
access remains unresolved. The proposed secret interface requires the user or
designated owner to identify it; no receipt field is assumed to be a password.
The supplied 159.4/250-TB quota figure is recorded as a historical report, not
fresh capacity or authority to restart copying. Effective runtime isolation
remains possible in principle; same-UID file modes alone do not establish it.

Verified margin provenance in the October 4 23:24 entry and prediction
specification: the coordinator adopted the explicit values under delegated
pre-score specification authority. Personal ratification was not a contract
requirement. Preserve that status and offer prospective scientific review
without adding an approval gate. No margin, budget, data role or access
boundary changes. The new delivery/separation recommendations are unadopted.

Documentation-only work on sh04-ln04 with no Slurm allocation: no secret,
payload, closed outcome, test, import, scheduler action, transfer or external
contact. Historical attempts are preserved; no new execution is selected.

## 2026-10-04 23:55 UTC — allocation46531931 completed and accounted

Root exited the compute shell after all selected work and document checks.
Slurm records COMPLETED0:0,23:46:14–23:55:36UTC,562 seconds on one CPU:
**0.1561111111 allocated core-hours**, charged once to EP17. This includes
the short EP18 handoff/document validation; do not add a second EP18 charge.
TotalCPU50.260seconds and step peakRSS113,856K are separate observations.
At23:55:40 the completed job is still in queue teardown (`COMPLETING`);
no other job is listed. No compute shell is available for further tests.

The33 numerical checks, exact metadata overlap and changed EP17 ASTRA
validation passed. EP18's exposure JSON/result reference also passed. Root
added its current exposure record and immutable handoff as explicit feature
freeze dependencies. The initial EP18 ASTRA check rejected uppercase letters
in the new input ID; the preserved failed validation and repaired lowercase-ID
validation document that local schema repair. Repaired EP18 schema/semantics
pass, and scoped whitespace checks pass. No scientific rule or source result
changed during that repair.

This goal turn made verified progress. Remaining empirical dependencies are
the stopped quota/copy route, encrypted-image credential interface, physical
reader-separated handoffs and their subsequent calibration/feature/scoring
work. No quota retry, credential collection, audit opening, automatic renewal
or new bulk job is selected. The episode remains scientifically unfinished;
these standalone results do not establish canonical BR launch or acceptance.

Earlier execution observations follow.

## 2026-10-04 23:53 UTC — `cross_episode_exposure`: exact overlap completed

Attempt001 completed with exit0 in4.209763443seconds, peak RSS37,684KiB,
zero errors and a3,666,473-byte report. All720 EP17 concepts occur in EP18's
main experiment. All4,320 observed,3,840 common and3,634 repeat-qualified EP17
image identities occur there too. The qualified-image role counts remain
2,423 development /605 calibration /606 sealed audit; concept roles remain
480/120/120. There is zero overlap with the200 repeated-session image IDs,
but200 concept identities overlap (126 development /35 calibration /39 audit).
The exact lists and provider-index bridge are retained in
`runs/cross_episode_exposure/attempt001/exposure_report.json`.

Both designs/roles are retained and the correlated, design-exposed decision is
adopted. In particular, all120 EP17 audit concepts already belong to EP18's
main stimulus inventory. Neither episode can supply independent stimulus-family
confirmation of the other. No pixel-byte identity is inferred. The source
archives' previous size/CRC differences are preserved. DINO and supervised
ResNet50 share nominated official checkpoint identities, but EP18's DINO
interpolation differs and identical feature arrays are not established.
Known source-access history is copied from the reviewed EP18 metadata record;
EP17's checkpoint acquisition, structural and metadata access remain recorded
in this log. This is not an exhaustive historical-access audit. Image pixels,
GLMsingle response values, candidate features/scores and both sealed outcomes
remain unopened by this work; physical reader separation is not established.

Root supplies the unchanged completed report as an explicit read-only metadata
handoff to EP18 at `outputs/runs/cross_episode_exposure/ep17_handoff001/` in that
episode; its worker updates only exposure documentation. No live sibling output
becomes a run input. Changed EP17 ASTRA schema and semantics pass in
`astra/v0.0.14/validation_20261004_calibration_numerics.json`, with the same8
optional-format warnings. All selected scientific preparation in46531931 is
complete; root will close the compute shell after final documentation checks.

Earlier execution observations follow.

## 2026-10-04 23:52 UTC — `cross_episode_exposure`: reviewed metadata comparison dispatched

Root and independent static review pass the bounded reader. The reviewer
identified a hardcoded checkpoint-access assertion; root repaired it to emit
the exact boolean from the uniquely matched frozen handoff record. Root also
uses the site's OAK environment variable for the two metadata source paths.
The four coordinator-copied EP18 JSON records total345,218bytes and are marked
read-only under `runs/cross_episode_exposure/ep18_handoff001`; this is an explicit
metadata handoff, not a live sibling run input or an OS-isolation claim.

Root now dispatches fresh `cross_episode_exposure/attempt001` in owned
allocation46531931 with the selected60s-internal/90s-external+10s-termination
envelope, one numerical thread and no payload access. The allocation ends
00:01:18UTC, leaving more than the complete envelope. Existing attempts and
both episodes' assignments are unchanged. Changed ASTRA validation follows.

Earlier execution observations follow.

## 2026-10-04 23:50 UTC — numerical cores pass; exact endpoint conventions adopted

Root verified compute shell sh02-02n43.int, allocation46531931, one CPU,
and one-thread numeric environment. All selected suites pass: calibration10
tests/0.653s, support8/0.312s, measurement8/0.085s, uncertainty7/33.755s.
Exact terminal evidence is retained under
`runs/prediction_qualification/core_checks001/`. Initial output redirection
failed because root omitted the parent directory; no test ran in that command.
Creating the missing parent fixed that local preparation error. All four
subsequent suite commands returned zero. No source or empirical score was read.

Root adopts `measurement_contract_specification.md` after independent review:
seven primary endpoints/five edges; explanatory H components share max-t but
do not enter stability classification; per-pair families; six separate
Fisher-z and CKA secondary contracts; same-three-participant signed point
estimate margins; leave-one-out direction checks. Actual calibration eligibility,
reader isolation and the full evaluator remain unfinished. Existing BR criteria
are reused; these checks are not BR launch or scientific acceptance.

For the required EP17/EP18 exposure decision, root adopts the existing-contract
option of retaining both frozen designs and treating their evidence as
correlated/design-exposed. Neither provides independent stimulus-family
confirmation of the other. An agent prepares a bounded metadata comparison
using an explicit coordinator-frozen EP18 handoff (four named metadata records)
and EP17's frozen concept/image assignments. The exact provider one-based
THINGS image-index bridge is required; no filename guessing or image-byte
substitution. Root selects this metadata-only stage for the remaining owned
allocation if its 60s internal/90s external plus10s termination bound fits:
one CPU, at most1GiB planning memory, no new download, pixel, EEG, fMRI or
feature access. Actual execution waits for static review. The quota/copy and
EP18 isolation routes remain stopped; this is an independent metadata task.

Earlier execution observations follow.

## 2026-10-04 23:42 UTC — `prediction_qualification`: combined-check allocation queued

Job 46531931 is queued for the selected one-CPU / 4-GiB / 15-minute combined
checks; root owns PTY87007. Root requests Requeue=0 on this owned interactive
job and will verify it before tests. All four cores now have independent
static review; the final uncertainty review also caught a fixture-only
12-versus-24-block mismatch. Root corrected the null fixture's explicit audit
phase before its first run. The prediction/measurement implementations and
frozen statistical conventions were unchanged by that fixture correction.
The exact endpoint/audit-homolog table is being made concrete in parallel;
it does not activate a real-data score or audit.

Earlier execution observations follow.

## 2026-10-04 23:41 UTC — `calibration_support` / `prediction_qualification`: array-core implementation and combined checks selected

The preceding goal turn made verified progress: three data-preparation stages
completed, structural quotas froze, and the existing prediction core passed.
The stopped quota/copy route and unresolved credential interface are unchanged.
No source, neural, image or audit value is read in the following preparation.

Two agents implemented calibration_core.py and support_core.py under separate
file ownership; both passed independent static review. Calibration preserves
all images and original columns while explicitly flagging unusable voxels,
separates image-mean and half-mean validity, and uses the adopted unequal-repeat
ceiling and fixed Fisher boundary. Support construction keeps anatomy independent
of reliability and preserves q32/N256/32-bank quotas without rerolls.

Root implemented array-only measurement_core.py and uncertainty_core.py from
the frozen policy: common-bank support/ceiling reaggregation; separate beta
block-mean OLS forecast and pooled-image primary effects; fixed 10,000-draw
crossed max-t inference. Independent review prompted an exact constant-statistic
degeneracy check, an explicit pre-inference primary-endpoint subset within
joint primary/explanatory bounds, and a median-crossing tie fixture. No scoring
reader, candidate search or audit-opening path is introduced by these APIs.

Root selects one bounded attended 15-minute normal allocation, one CPU / 4 GiB,
no GPU and no automatic requeue, combining the four new focused suites and
changed ASTRA validation. Each calibration/support/measurement suite is capped
at 60 seconds externally plus 5 seconds termination; full-draw uncertainty
fixtures at 180 seconds plus 10. Required fixes may be retested only if their
complete bound fits. These are mathematical known-answer checks, not real-data
results or completion of the full reader/evaluator qualification. No login-node
test, quota retry, new synthetic research search or source-stage rerun is selected.

Earlier execution observations follow.

## 2026-10-04 23:30 UTC — `calibration_support`: structural quota complete; allocation closed

Structural support attempt001 completed in 13.590053111 seconds, peak RSS
25,864 KiB, exit zero. It streamed all 881,359 source-axis rows once and
preserved all 8,558 nominated ROI rows. The minimum of the 64 geometry bins
is 104, fixing q=32 and N=256 for every participant/ROI. The gzip metadata
table is 147,879 bytes. Both structural completion flags are true; calibration,
finite-neural eligibility, reliable subsets and anatomical banks remain
uncomputed. No source volume, neural value, image, feature or audit outcome
was opened by this step. The actual quota is now recorded in the adopted
prediction specification; later eligibility failure cannot lower it.

All selected compute work finished. Root exited PTY2622 normally. Scheduler
records job 46527997 COMPLETED 0:0, 23:20:06–23:30:23 UTC: 617 seconds on one
CPU = **0.1713888889 allocated core-hours**, charged once to EP17. TotalCPU
47.307 seconds; step peak RSS 526,724 KiB. The user queue is empty at the
23:30:44 observation. No compute shell, pending job or automatic renewal remains.
Changed ASTRA schema/semantics passed in
astra/v0.0.14/validation_20261004_prediction_support.json; scoped whitespace
checks passed. Both independent code reviews passed, and the 11 prediction
and 8 structural fixtures passed on the owned compute node.

The next empirical stages need the current quota/copy route resolved and
actual role-filtered handoffs, plus the encrypted archive's existing private
credential delivery interface. The latter was requested once as missing
information, without requesting a password value. The quota route is not
retried or relocated. Source roles and scientific scoring remain incomplete.
Existing BR advisory criteria are reused; these standalone Slurm results
are not a canonical BR dispatch or BR scientific verdict.

Earlier execution observations follow.

## 2026-10-04 23:29 UTC — `calibration_support`: structural-bin dispatch

Root and independent review pass the stable structural-support reader and
eight focused fixtures. All eight tests passed in 0.287 seconds, exit zero,
inside allocation 46527997. Root now dispatches the adopted fresh
structural_support/attempt001 against the completed public voxel-axis table,
with the unchanged 60-second internal / 90-second external plus 10-second
termination envelope. More than five minutes remain in the existing allocation.
No response value, calibration statistic or feature is an input to this step.

Earlier execution observations follow.

## 2026-10-04 23:28 UTC — `prediction_qualification`: shared linking core fixtures pass

The array-only prediction_core.py implementation passed root review and all
11 focused fixtures in 0.139 seconds, exit zero, inside verified allocation
46527997 with one numeric thread. Tests cover the three fixed PCA ranks,
training-only means/basis/scale, the n-times-lambda ridge system, unpenalized
intercepts, exact tie order, response-scale invariance, proper training-mean
R² denominators, pooled sums, nonfinite refusal and CKA scaling distinctions.
No source file reader, selector, empirical model score or calibration/audit
access is implemented by this core. Structural-support implementation remains
under final focused review before its source-table dispatch.

Earlier execution observations follow.

## 2026-10-04 23:24 UTC — `calibration_support` / `prediction_qualification`: numerical conventions adopted

The sub01 structural diagnostic matches NiBabel's entire decoded atlas and
affine exactly, in 2.858540549 seconds. Source left/right labels have 1,689 /
2,383 six-connected components, with largest-component fractions 0.850581 /
0.778422. The broad bounds therefore occur in the supplied segmentation, not
the mapper's storage order. Keep all nominated source-label components and
interpret coordinates in native T1w, not MNI; no anatomical cleanup is adopted.
The changed ASTRA validates successfully in
astra/v0.0.14/validation_20261004_geometry_archive.json (8 optional warnings).
The file contains a JSON-encoded text report; attempting to decode the text
again as an object failed only in root's display helper, not the validator.

After independent scientific review and three precise clarifications, root
adopts prediction_qualification_specification.md: 84 shared settings, fixed
train-only transforms and B/C/D selection rule, repeated-response calibration
estimator and floor, relative numerical variance rules, scale-specific
practical/forecast/adequacy margins and fixed crossed-bootstrap convention.
The existing policy requires prospective freezing rather than separate
scientist approval. No participant, role, comparison family, search budget or
audit rule changes. The private cost pilot remains unselected.

Root selects implementation of the shared array-level prediction core, and
a structural-only bin computation consuming completed attempt002 metadata.
The latter streams the public voxel-axis table once, retains labels1007/2007,
uses the adopted four y groups then two z halves, and computes shared
q=min(32,floor(minimum bin count/2)), N=8q, refusing q<8. No response-based
filter, support bank or reliable subset is constructed. Fresh
structural_support/attempt001 outputs are capped at 2 MiB, one CPU / 2 GiB,
60-second internal / 90-second external plus 10-second termination bound.
Implementation, independent review and focused compute checks precede this
structural source-table dispatch. Existing allocation is reused only if the
full envelope still fits; no automatic renewal. Root retains all execution
and shared-document ownership.

Earlier execution observations follow.

## 2026-10-04 23:22 UTC — `anatomical_axis` / `stimulus_features`: terminal execution and focused structural diagnosis

Attempt002 returned zero with both structural completion flags true: 881,359
original columns, 80,237,266 uncompressed table bytes, 14.258795875 seconds,
peak RSS 408,168 KiB. No neural values were read. Root nevertheless observed
unexpectedly broad fusiform world-coordinate ranges (sub01 left spans y
−70.18 to +87.82 and z −22.96 to +75.04 mm). Before adopting any support bins
or N, select one focused structural diagnostic comparing the exact already
selected sub01 atlas decode with NiBabel's independent file reader and reporting
label counts/bounds/connectivity. No new volume or response array is selected.
One CPU / existing 4 GiB allocation, 60-second internal / 90-second external
plus 10 seconds; save only a small operational diagnostic in
anatomical_axis/decode_diagnostic001.json. Do not infer anatomy validity from
the runner's terminal flag alone.

Archive metadata001 also returned zero: all 32 exact selected members match
once; 8,640 entries are encrypted with method 8, flags 9, no AES metadata in
the selected records. Bounded reads total 971,173 bytes, 0.383516322 seconds,
peak RSS 34,288 KiB. No member stream, password, image or feature was opened.
The approved secret-delivery interface remains unresolved; the feature pilot
is still unselected.

Earlier execution observations follow.

## 2026-10-04 23:20 UTC — `anatomical_axis` / `stimulus_features`: source dispatch

Allocation 46527997 started at 23:20:06 UTC on sh02-02n44, with one CPU,
4 GiB, Requeue=0 and observed end 23:35:09 UTC. Root verified hostname and
SLURM_JOB_ID in PTY2622 and loaded the existing module runtime with all numeric
thread counts set to one. The reviewed attempt002 mapping runs first, then
the independent archive_metadata001 inspection, under their adopted external
timeouts. No task is sent to the login shell. ASTRA validation follows.

Earlier execution observations follow.

## 2026-10-04 23:17 UTC — `source_roles`: provider-native response convention accepted

Root and independent review accept the already documented provider-native
convention for subsequent named-model comparisons: delivered GLMsingle B/C/D
estimates derived from runwise voxel-standardized BOLD, with provider scaling
preserved. Physiological percent-BOLD remains unverified. The current pinned
source evidence suffices without optional DESIGNINFO reads or perfect
historical binary identity. The anatomical specification records the exact
claim limits, scale-invariance argument for voxelwise R²/reliability, and
scaling sensitivity of absolute variance thresholds and cross-voxel CKA.
No neural value was read and no scoring or audit boundary changes.

Job 46527997 was observed pending with Requeue=1; root's update to Requeue=0
returned success. It will be verified with the next appropriately spaced
scheduler observation before source execution.

Earlier execution observations follow.

## 2026-10-04 23:16 UTC — `anatomical_axis` / `stimulus_features`: allocation submitted

The documented plain interactive request returned job **46527997**, queued
for the selected combined 15-minute / one-CPU / 4-GiB scope. Root owns PTY2622.
Actual compute-node identity, job limits and Requeue=0 will be verified before
either source process starts. There is one pending request, not duplicate jobs.

Earlier execution observations follow.

## 2026-10-04 23:15 UTC — `anatomical_axis` / `stimulus_features`: repaired mapping and metadata inspection selected

The units001 source witness completed in 1.625699538 seconds, peak RSS 55,180
KiB. Four requests read exactly 28,069 compressed bytes. All four source
headers explicitly use mm and agree with their saved union geometry at every
corner (maximum difference 0.0 mm). No structural voxel array was decoded by
this witness, and no neural or image value was read.

The focused mapping repair consumes this exact complete report only for
unknown unit headers, verifies unchanged current geometry, and preserves
observed unknown versus witness-derived mm in its output. Independent review
passed; all 23 mapping fixtures passed in 0.086 seconds. Separately, the bounded
archive metadata reader passed independent review and all 19 fixtures in
0.013 seconds. These source-free tests ran in verified allocation 46525781.

That shell exited normally. Scheduler accounting records COMPLETED 0:0,
22:55:27–23:13:57 UTC, 1,110 seconds on one CPU = 0.3083333333 allocated
core-hours, charged once to EP17. TotalCPU 41.887 seconds; step peak RSS
2,026,324 KiB includes the model checks. No allocation remains active.

Root selects one attended 15-minute normal-partition allocation, one CPU /
4 GiB, no GPU, no requeue, for fresh anatomical attempt002 (unchanged 300-second
internal / 360-second external plus 10-second termination bound), archive
metadata001 (60 / 90 plus 10 seconds), and the changed ASTRA validation.
Both reviewed source steps fit independently within the requested envelope;
they are combined rather than submitted as separate tiny jobs. Original
attempt001 is preserved. No quota/copy retry, neural scoring, image decoding,
credential access or feature pilot is selected. Root owns submission and
monitoring; recorded IDs and observed results follow after execution.

The first submission command exited 255 before allocating because this site's
`srun` does not accept `--no-requeue`. The corrected request uses `salloc
--no-requeue` with an attended `srun --pty` step, preserving the same explicit
resources and no-requeue requirement. No job or source process was created by
the rejected command. The attempted `salloc` spelling also rejects that flag
before allocation. Local `srun --help` confirms it is not an interactive
submission option. Use the documented plain `srun --pty` request and verify
the owned job's Requeue field before source dispatch (setting Requeue=0 on
that job if needed); neither rejected command created an allocation.

Earlier execution observations follow.

## 2026-10-04 23:09 UTC — `anatomical_axis`: unit-witness fixtures pass; units001 dispatch

The first focused suite had9 passes and one fixture expectation failure: a
manually modified JSON srow was not an exactly representable stored float32
NIfTI value, correctly triggering the earlier metadata-replay refusal. Root
changed only the fixture to derive its altered srow from a real constructed
NIfTI header. The unchanged production core and all10 focused fixtures then
passed in0.091seconds. Root static review passes the exact pointer identities,
bounded header decoding and independent inherited-affine comparison. Changed
ASTRA schema/semantics passed in
`astra/v0.0.14/validation_20261004_source_diagnostics.json`.

Root now dispatches the adopted units001 witness under180s external+10
with120s internal, using the verified46525781 module interpreter. More than
the full witness envelope remains before23:15:30UTC. No source request ran
before the passing fixtures. The original failed anatomy attempt001 remains
preserved; no geometry rerun is selected by this diagnostic dispatch.

Earlier execution observations follow.

## 2026-10-04 23:05 UTC — `model_pair_registry`:9 fixtures pass; two bounded metadata diagnoses selected

All9 source-free model-feature fixtures passed in5.188seconds, exit0, using
the exact existing encoding_model runtime recorded above. Root prospectively
pins its observed Pillow12.2.0 in the feature core before any cache exists.
No source image, acquired checkpoint, model score or saved feature was used
by these tests. The model feature representation is implementation-qualified;
source feature execution and the wider scientific prerequisites remain open.

For the observed anatomical-unit loss, pinned provider code and Nilearn0.9.2
show that intersect_masks preserves the first original run mask's numeric
affine but constructs a new image without copying its header. Root selects
four exact first lexical fMRIPrep5155ba... source-mask header witnesses,
28,069 compressed bytes total. Four sequential fixed-URL GETs,8192bytes cap
each,15s/request,no redirects/retries,120s internal/180s external+10,one
CPU≤1GiB. Verify each pinned compressed length/MD5, decode at most352 bytes
with zlib, require explicit mm/coded native T1w and shape/0.001mm corner
agreement with the original union. Full compressed structural transport is
disclosed; voxel arrays remain undecoded. Implementation/fixtures/review
precede root dispatch; no anatomy retry is yet selected.

Separately, acquisition metadata resolves the existing stimulus archive at
`$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/cneuromod_images_fmri_pinned_f8f5162_20260922/archives/images_fmri.zip`,
917286854bytes, with8640 recorded image entries. Root selects a bounded
central-directory/member-metadata inspection: classic ZIP EOCD tail≤65557B,
central directory≤16MiB,≤16384entries; no ZIP64/multidisk/automatic expansion.
Join the exact completed provider/role metadata and record deterministic32
development-image selection plus namespace/encryption/compression facts.
One CPU2GiB,60s internal/90s external+10,output≤256KiB. No member stream,
password, decoded pixel or feature is opened; no new full-archive digest is
needed for this bounded metadata diagnosis. Actual feature pilot remains
unselected. Reviewer02 cannot reactivate (tool: agent thread limit reached),
so the two active authors cross-review separate files and root integrates.

Earlier execution observations follow.

## 2026-10-04 23:02 UTC — `model_pair_registry`: existing exact runtime resolved; fixtures dispatch

Compute-side software metadata and actual imports identify the existing
`$GROUP_HOME/zijiao/miniforge3/envs/encoding_model/bin/python` as Python3.11.15,
Torch2.4.1+cu121,Torchvision0.19.1+cu121,Pillow12.2.0,one Torch thread.
Use `env -u PYTHONPATH PYTHONNOUSERSITE=1` for this executable so the unrelated
module builds do not shadow it. This preserves the adopted model-library
binding; no version check was relaxed and no environment was modified.
Root now dispatches all9 reviewed source-free feature fixtures under180s
external timeout plus10s grace in46525781. No acquired checkpoint or source
image is read by these fixtures; no random checkpoint or features are saved.

Earlier execution observations follow.

## 2026-10-04 22:58 UTC — `anatomical_axis`: attempt001 stops on unresolved units

The fixed source attempt001 returned2 after0.086713699seconds,peakRSS57520KiB.
Its terminal report is `runs/anatomical_axis/attempt001/anatomical_axis_report.json`.
The sub01 compressed atlas identity was verified and its original union mask
was read, but selected affine qualification stopped with
`spatial_units_unresolved`. No atlas mapping row was written; the115-byte
uncompressed table header and terminal failure report are preserved. Both
structural completion flags remain false. No neural, stimulus or model value
was opened. This is a source-metadata diagnosis, not a server failure or
scientific null. Provider-code diagnosis is delegated; no automatic retry,
millimeter assumption from voxel size, or geometry substitution is selected.

The independently reviewed model-core precision fix is complete. Existing
module imports unexpectedly report runtime Torch2.4.0a0/Torchvision0.19.1a0,
Pillow10.2.0.post0 and one Torch thread despite their2.4.1/0.19.1 module names.
These differ from the adopted2.4.1/0.19.1 runtime binding and earlier acquisition
record, so model fixtures have not been run under a silently relaxed version
check. Root is resolving the existing software binding on compute; no install
or source feature execution is selected.

Earlier execution observations follow.

## 2026-10-04 22:56 UTC — `anatomical_axis`: verified compute fixtures pass; attempt001 dispatch

Root verified sh03-01n68,SLURM_JOB_ID46525781,one CPU in PTY19268.
Scheduler records start22:55:27UTC and end23:15:30UTC,4GiB,no-requeue.
All17 anatomical fixtures passed in0.054seconds on the explicit Python3.12.1
environment. Root now dispatches the previously selected fixed eight-volume
attempt001 under360s external timeout plus10s kill grace, retaining its300s
internal deadline. No source result is inferred from the synthetic fixtures.
Model-core review requested a narrow caller-precision restoration fix; those
fixtures wait for the final implementation. No neural-copy or scoring launch.

Earlier execution observations follow.

## 2026-10-04 22:53 UTC — `anatomical_axis`: allocation46525781 submitted

The selected srun returned Slurm46525781, queued for resources in PTY19268.
Root disabled requeue. One CPU4GiB/20minutes remains fixed; no compute-node
placement, test result or structural result is yet inferred. No source
command is sent until hostname and SLURM_JOB_ID are verified in that shell.

Earlier execution observations follow.

## 2026-10-04 22:53 UTC — `anatomical_axis` / `model_pair_registry`: combined attended preparation selected

Root and independent static reviews passed the anatomical runner and17
meaningful source-free fixtures. Select one attended normal CPU_SKU:7502
allocation,1CPU4GiB/20minutes,noGPU,no-requeue. Aggregate those fixtures,
the already adopted300s-internal/360s-external+10 structural attempt001,
changed ASTRA validation and the model-core focused checks if reviewed/ready
within the same resource envelope. Source structural execution follows its
passing fixtures; a failure stops that route without automatic retry.
Model checks stay source-free and do not create a source feature cache.
Root owns the shell and will verify actual placement before computation.

Existing source headers did not record xyzt_units; unknown units remain an
explicit refusal, not millimeters inferred from voxel size. Exact meter/micron
conversions are now explicit in the adopted specification. No additional
source object, neural value, optional DESIGNINFO read, N or score is selected.
The separate quota/copy route remains stopped. Previous allocation46524139
is terminal; no task compute allocation or pending duplicate is being reused.

Earlier execution observations follow.

## 2026-10-04 22:51 UTC — `model_pair_registry`: prospective feature definition adopted

Root and independent review passed `model_pair_registry_specification.md`.
Adopt the three existing DINO/SwAV/supervised ResNet50 pair signs, the single
SUP-vs-random(seed17) falsifier, common bilinear256-to224 RGB preprocessing,
the documented normalization and four unweighted stage-GAP feature vectors.
Float32 inference disables autocast and both CUDA/cuDNN TF32 paths. Provider
recipe deviations are disclosed; dependent model pairs are not independent
replications. Exact target-image exposure remains unknown for every trained
asset, so no clean-exposure or causal training-recipe claim is authorized.
Known overlap would retain the policy's control-only restriction.

Implementation and focused meaningful fixtures are delegated in the tools/tests
scope, reusing already acquired and validated checkpoints. No new download,
source pixel, feature cache, neural/model score or random-checkpoint publication
is selected by this adoption. The actual Pillow version and executable binding
must be recorded before feature caching. The source-image adapter, physical
reader boundary, feature resource pilot, linking grids and scientific margins
remain unfinished. No extra candidate menu or seed search is opened.

Earlier execution observations follow.

## 2026-10-04 22:49 UTC — `role_materialization` / `anatomical_axis`: preparation allocation closed

Root exited PTY39790 normally after the prepared checks and observed final
Slurm accounting:46524139 completed exit0:0,22:41:29–22:48:30UTC,
421seconds on one CPU =0.1169444444allocated core-hours, charged to EP17 once.
TotalCPU10.014seconds and stepMaxRSS177036K are separate measurements. No
task compute shell remains. The copy pilot was not launched; the quota route
remains stopped. Existing module discovery/import checks found NiBabel5.4.0,
SciPy1.12.0 and NumPy1.26.3, with no installation. The adopted anatomical
program's changed ASTRA validation passed schema and semantics in
`astra/v0.0.14/validation_20261004_anatomical_axis.json` (9 optional format
warnings); source-free fixtures and structural execution await the implemented
runner and suitable4GiB allocation. Scoped document whitespace check passed.

Earlier execution observations follow.

## 2026-10-04 22:46 UTC — `role_materialization`: capacity route stopped before neural access

Owned allocation46524139 placed on sh03-01n68; root verified that hostname,
SLURM_JOB_ID and one CPU in PTY39790. Scheduler start22:41:29UTC and current
end23:01:34UTC were observed; request remains1CPU2GiB,20minutes,no-requeue.
All10 focused copy fixtures passed in0.035seconds, including independently
reopened persisted-byte checks and conservative failed-read exposure. Changed
ASTRA schema/semantics passed in
`astra/v0.0.14/validation_20261004_role_materialization_pilot.json` (9 existing
optional output-format warnings).

The bounded official `sh_quota` command returned exit0 but explicitly warned
`quota query timed out for OAK, showing cached data`. Its displayed156.4TB/
250.0TB is cached, not a current quota observation. A bounded df returned
102097851768832 filesystem-available bytes. That does not supply the missing
current quota headroom. No capacity-passing record was created, the copy pilot
was not launched, and no neural value or audit byte was read. Per Sherlock's
infrastructure stop rule, the affected quota/copy route is stopped without
retry, alternative quota probing, relocation or full-copy selection. The user
was informed of the warning and its effect. Small independent implementation
and structural qualification preparation continue; no filesystem outage is
inferred from this quota timeout alone.

## 2026-10-04 22:45 UTC — `anatomical_axis`: bounded structural stage adopted

Root read the final `anatomical_axis_specification.md` and the independent
scientific PASS, and adopts its exact four original union-mask plus four DK
aparcaseg structural-volume scope. One CPU4GiB,300s internal/360s external
plus10s grace, eight-volume/count/grid/table caps and no network fetch/retry.
Implementation and focused source-free fixtures are delegated; source
execution follows their passing review/checks in a suitable allocation.
The current2GiB allocation is unsuitable for the4GiB structural payload stage.
No optional DESIGNINFO scalar, neural values, finite-response support, N,
spatial quotas or scoring is selected. Source-authenticated shared T1w frame,
C-order original columns and nearest-neighbor integer atlas mapping remain
fixed. Structural mapping is independent of the stopped large-copy capacity
route; it does not release reader handoffs.

Earlier execution observations follow.

## 2026-10-04 22:38 UTC — `role_materialization`: preparation46524139 queued

The selected bounded srun returned46524139 and is waiting for resources in
PTY39790. Requeue was disabled by root. No compute-node shell has yet been
verified; no tests, capacity probe or source read is dispatched into this PTY
until allocation placement is established. One CPU2GiB/20minutes remains the
fixed attended request; no duplicate job was submitted.

Earlier execution observations follow.

## 2026-10-04 22:38 UTC — `role_materialization`: bounded attended preparation allocation selected

The pilot core is saved and initial root/independent review passed its exact
metadata joins, source indirection/layout caps, private routing and conservative
decoded-chunk audit exposure. Root requested that byte validation reopen the
closed destination rather than use its writer cache; the owner is implementing
that focused fix and fixtures. No source payload has been opened.

Select one attended preparation allocation: normal CPU_SKU:7502,1CPU2GiB,
20minutes,noGPU, for those fixtures, changed ASTRA validation, bounded quota/
capacity observation and the already adopted fixed pilot if all prerequisites
pass. The prior metadata job is terminal; no duplicate or overlapping test
allocation exists. Site partition/hardware discovery from this session remains
applicable. Root owns the PTY and will verify hostname/SLURM_JOB_ID there and
disable requeue. The scientific pilot remains120s internal/180s external+10;
allocation waiting does not authorize payload access before review/capacity.

Earlier execution observations follow.

## 2026-10-04 22:35 UTC — `concept_role_assignment`: terminal scheduler accounting observed

Slurm46523446 completed exit0:0,22:30:56–22:31:26UTC,30seconds on one CPU,
0.0083333333allocated core-hours charged to EP17 once. TotalCPU21.581seconds,
batchMaxRSS250584K; runner duration21.604258seconds/RSS227580KiB are separate
measurements. Independent operational-report review passed the exact decreasing
objectives, valid finite development-block termination and preserved counts.
No active task compute shell or batch remains. The adopted source-copy pilot
awaits stable implementation, compute fixtures, capacity and source-layout checks.

Earlier execution observations follow.

## 2026-10-04 22:33 UTC — `concept_role_assignment`: all four metadata assignments complete

The batch's verified shell was sh03-01n68,SLURM_JOB_ID46523446,Python3.12.1.
The terminal `runs/concept_role_assignment/full001/assignment_report.json`
reports completed/metadata_assignments_complete=true,21.604258seconds,
peakRSS227580KiB, no errors. Exactly480development/120support_calibration/
120sealed_audit concepts are assigned. All720concepts,4320observed images,
49980retained presentations and833runs are preserved in ordinary metadata
tables. Original ineligible images inherit their concept role and remain
ineligible. Development blocks did not receive outer-fold labels; exact raw
pilot features were reused without new source summaries or outcomes.

Roles and folds each terminate after6sweeps with a zero-swap final sweep;
audit blocks after4 with zero. Development blocks complete the fixed8sweeps
with one accepted move in the eighth, so their result is a valid fixed-budget
assignment, **not a claimed pairwise local or global optimum**. No additional
sweep, restart or role rescue is selected. Exact integer/cache checks and
capacities passed. Full signed metadata balance residuals are retained
in `balance_report.json`; they are not neural
scores and are not a new eligibility threshold.

Roles/folds/blocks flags are true; source_roles_complete, neural materialization,
physical isolation, neural/pixel/model/behavior outcome reads and audit opening
remain false. This satisfies the completed-map prerequisite for the adopted
builder-copy pilot, whose layout/capacity/review checks still precede payload
access. Scheduler terminal accounting remains to be observed at the next
permitted check; disappearance from squeue alone was not used as completion.

Earlier execution observations follow.

## 2026-10-04 22:30 UTC — `role_materialization`: source-preserving builder pilot adopted conditionally

Root and independent review passed `role_materialization_specification.md`,
including exact byte/resource arithmetic, truthful neural/audit-byte exposure,
ordinary private copies under ignored `outputs/cache/private/role_materialization`,
all original rows/dtype/columns, and prohibited external/virtual/link-backed
outputs. Root adopts implementation, focused source-free fixtures and the
fixed sub01/B source rows[0,64) pilot **only after full001 metadata assignment
completion, one bounded capacity observation and the exact layout cap pass**.
No neural payload has been read or copied by this decision.

The pilot uses1CPU2GiB,120s internal/180s external plus10s termination,
58,166,784logical source bytes,256MiB decoded-chunk/output ceiling and one
worker. Reopened destination bytes must exactly match source dtype/shape/bit
patterns, including NaNs/signed zero; no values, finite counts or payload
digests are exported. Original chronology, eligibility and role/index mapping
remain unchanged. The builder records first neural access and whether source
audit rows were copied; this is preparation, not a scientific audit evaluation.
Physical_role_isolation_verified, reader_handoffs_activated, source_roles_complete
and candidate_scoring_authorized all remain false. Same-UID file modes are
not isolation; the mixed source and private copies are not search-worker inputs.

The proposed twelve-product full payload is132,332,156,640bytes/123.243925GiB,
2,346blocks, with160GiB new-output ceiling. All12 allowed layout headers must
pass before full payload access. Full copying remains unselected until measured
pilot resource evidence supports a finite run within original budgets. Existing
BR advisory criteria from ext_review_dir_3f7a40471cbd continue to inform leakage
and pseudoreplication limits; no new BR verdict or launch authority is implied.
Separately, an agent is proposing the provider-supported voxel-axis/anatomical
mapping from documentation only; no anatomical payload stage is adopted yet.

Earlier execution observations follow.

## 2026-10-04 22:27 UTC — `concept_role_assignment`: batch46523446 submitted; shared preparation closed

Root submitted the reviewed full assignment as Slurm46523446 using the exact
selected batch script and fresh submission001 logs. This is an actual scheduler
submission; running/completion is not yet inferred. One CPU6GiB/35minutes,
no-requeue, original1800/1860s bounds. Root alone monitors this dependency.

Owned preparation46521343/PTY36735 closed normally, exit0, at22:27:18UTC.
Job elapsed1342seconds on one CPU gives0.3727777778allocated core-hours total.
The two authorized scopes EP05/EP17 each receive671seconds/0.1863888889core-hours
as bookkeeping shares, counted once; these are not measured per-process CPU.
Slurm TotalCPU174.479seconds and stepMaxRSS6265608K are separate shared-step
measurements. The EP17 pilot itself reported292064KiB; the shared peak is not
attributed solely to it. No compute shell remains and no login execution follows.

Earlier execution observations follow.

## 2026-10-04 22:26 UTC — `concept_role_assignment`: full runner ready for selected batch

All23 focused fixtures passed0.026seconds in46521343; root and independent
full-mode/code/publication reviews passed. Six new cases cover exact raw-feature
reuse, rational family losses, ineligible-image inheritance, changed anchors,
incomplete-fourth-assignment publication refusal and deadline resets. Batch
syntax and changed ASTRA schema/semantics passed
`astra/v0.0.14/validation_20261004_concept_role_full.json`.

The exact selected entrypoint is `tools/run_concept_role_assignment.sbatch`,
normal CPU_SKU:7502,1CPU6GiB/35minutes,no-requeue. Root will close the completed
shared preparation shell then submit this one job, with fresh
`runs/concept_role_assignment/submission001/slurm_%j.out` and `.err`.
Internal1800s/external1860s+10 and fresh full001 remain unchanged. Terminal
assignment_report.json must show all four maps and all inherited metadata
complete before any downstream use; physical handoffs/scoring remain closed.

Earlier execution observations follow.

## 2026-10-04 22:21 UTC — `concept_role_assignment`: pilot complete; full finite assignment selected

The selected pilot completed in46521343 in16.310645seconds, peakRSS292064KiB,
720concepts and237metadata fields. All4096 cached deltas matched independent
direct feature-sum calculations. Temporary cache updates restored exactly;
zero source swaps were accepted and no role/fold/block map was published.
Evidence: `runs/concept_role_assignment/pilot001/pilot_report.json`, with exact
Decimal-string raw features and field definitions beside it. Errors are empty;
pilot_complete=true, metadata_assignments_complete/source_roles_complete=false.

Measured input/raw preparation4.527763+4.029647s, parent scaling0.464329s,
role Gram5.468483s, pair order0.610766s. All-four-parent projections retain
3,967,200pair visits and2,380,665,600possible per-concept cache updates:
Gram10.485683s, pair ordering1.170140s, checks5.199683s, worst cache685.024821s,
initial/eight-sweep replays1.253047s, repeated input/features8.557410s and
parent scaling1.160823s. Twice their sum plus60seconds gives1485.703214s.
These are measured planning projections, not guaranteed bounds; no favorable
swap count or early convergence is assumed.

Root selects the unchanged finite full assignment for implementation and
execution after focused publication/reuse fixtures and independent review:
one CPU6GiB,1800s internal/1860s external plus10s termination, fresh full001
attempt. A35-minute allocation will cover the envelope. The current allocation
cannot fit the full31m10s remaining bound; complete preparation here, close it,
then submit one fresh bounded job once ready. No renewal or parallel duplicate.
The fixed four assignments, features, precision, capacities, one start and
eight sweeps are unchanged. Reuse the pilot's exact raw metadata features with
their missingness and weights; maps become complete only after all four
assignments and inherited image/presentation outputs are written. Physical
reader/handoff qualification remains separate, so source_roles_complete stays
false; no neural, behavioral, image or model values or audit outcomes are opened.

Earlier execution observations follow.

## 2026-10-04 22:20 UTC — `concept_role_assignment`: selected metadata timing pilot launched

All17 focused fixtures passed0.011seconds in verified46521343. Root and
independent final review passed all three stable implementation/fixture files,
including exact arbitrary-integer deltas, unequal capacities, lag semantics,
constant handling, child input separation and failed-publication success resets.
Changed ASTRA schema/semantic validation passed. Dispatch the already adopted
metadata-only pilot in root PTY36735 now: internal300s/external360s plus10s
termination,1CPU6GiB, one worker, fresh `runs/concept_role_assignment/pilot001`.
More than19minutes remain before22:39:59, covering the full envelope.
No role swap is accepted or map released in this pilot; root will inspect only
preparation, replay and cost fields to select or stop full assignment.

Earlier execution observations follow.

## 2026-10-04 22:18 UTC — `concept_role_assignment`: pilot implementation review and shared compute selected

Root has reviewed the fixed metadata loader and exact integer assignment core;
independent review agrees, with focused fixtures still being completed. The
selected executable exposes only `--mode pilot`; no full assignment launch is
enabled. New ASTRA node connects completed eligibility/chronology and all fixed
metadata projections to the pilot. Schema/semantic validation passed in
`astra/v0.0.14/validation_20261004_concept_role_pilot.json`, with nine preexisting
optional-format warnings. The pilot output directory is fresh
`runs/concept_role_assignment/pilot001`; it will contain only exact raw features,
field definitions and operational timing/replay evidence, no role assignments.

EP05's substantive work is complete. Root selects sequential reuse of owned
46521343/PTY36735 for EP17 focused fixtures and this already adopted pilot after
code/fixture review passes. Placement verified sh03-01n68,1CPU6GiB,Requeue0;
22:16:41UTC scheduler observation confirms end22:39:59. A complete360s+10s
external envelope is required at dispatch. One numeric worker and BLAS/OpenMP
thread caps1 remain. Shared allocation cost will be split equally between EP05
and EP17 once at terminal closeout; no simultaneous separate charge of the
shared total. No additional allocation or scheduler submission is made now.

Earlier execution observations follow.

## 2026-10-04 22:10 UTC — concept-role numerical specification and metadata pilot adopted

Root and independent scientific review passed the saved
[concept_role_assignment_specification.md](concept_role_assignment_specification.md).
Adopt exact features, participant/image weighting, distinct delay sentinels/
unknown coverage, parent-specific imputation, Decimal80 half-even scaling,
1e-4 standardized-unit quantization, exact integer objective/deltas and fixed
single initialization/eight sweeps for each of roles/folds/devblocks/auditblocks.
Hard capacities remain480/120/120,6×80,24×20 and12×10. Development blocks do
not consume fold labels. A fixed eight-sweep result is not a global/local
optimum unless its documented termination supports the narrower statement.
Canonical row accumulation and identical-value moment/standardization branches
prevent tiny Decimal rounding variance from amplifying constant features;
no variance floor, clipping or outcome-informed tie rule is introduced.

Select implementation, meaningful source-free fixtures and one metadata-only
preparation/timing pilot after code review:1CPU6GiB,300s internal/360s external
plus10s termination,32MiB/file, fixed completed metadata only, no network.
The6GiB choice follows an independent conservative structural peak estimate
below2GiB with nonredundant parsing; actual peak remains to be measured. The
pilot accepts no swaps, publishes no roles, and measures exact Gram/cache
checks on4096 fixed initially cross-group pairs. It may preserve reusable
exact features, but no completed assignment map. Full resource selection is
pending measured all-four-Gram costs,3,967,200 pair visits and the conservative
2,380,665,600 possible per-concept cache updates. No reduced sweep, extra
restart, altered feature, interrupted-incumbent publication or full assignment
is automatically authorized by pilot success. The current EP05 allocation
may be reused sequentially for this bounded preparation only if sufficient
remaining resources and time are verified; no concurrent numerical worker.


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


## 2026-10-04 21:59 UTC — `source_chronology`: complete retained order and eligible repeat halves

Root and independent final code review passed after correcting component-order
invalidation. All15 focused fixtures passed0.006s in46518769; changed ASTRA
schema/semantics passed validation_20261004_source_chronology.json. The bounded
attempt001 launched at21:58:57UTC and completed in4.375075seconds.

[chronology_report.json](runs/source_chronology/attempt001/chronology_report.json)
reports all49980 presentation rows and833 runs preserved; retained-half,
global-chronology, overall-completion and all-observed-ordinal flags true.
Each participant has one connected component, no invalid or ordinal-conflicted
component, and zero issues. Exact constraint edges are8452,8460,8460,7785 for
sub01/02/03/06. The documented sub01/session17 order1,2,3,5,6,4 is matched.
Twenty-three nonconsecutive source-ordinal pairs were skipped rather than
bridged with an inapplicable last-repeat gap. No elapsed-time interpretation
or complete discarded-exposure-history claim is made.

The companion presentation/run CSVs retain source coordinates and distinguish
component-relative positions, reconstructed full-history positions and retained
acquisition ranks. Only eligible images receive analysis repeat-half labels;
absence was not misclassified as an observed bad ordinal. This completes the
selected metadata chronology stage. Whole-concept roles, folds/blocks,
permission-separated handoffs, neural units, checkpoint qualification and
scientific scoring remain unfinished. The next proposal fixes finite metadata
balance features/precision/optimizer choices before any role assignment.
No new source access or roles are selected by this completion.


## 2026-10-04 21:49 UTC — `source_eligibility`: repaired attempt002 completes 720 concepts

Independent final repair review passed; all17 focused fixtures passed0.013s
in46518769. Changed ASTRA schema/semantics passed
validation_20261004_taxonomy_repair.json (nine pre-existing optional output-format
warnings do not fail v0.0.14). The selected bounded attempt002 completed in
4.714159seconds with eligibility_complete and eligible_720_concept_universe_verified
both true. All49,980 B/C/D row mappings and two pinned object identities passed.

There are4320 observed images,3840 common experimental images,3634 qualified
images and exactly720 qualified concepts. Original common five/six counts are
480/240; qualified exemplar counts are3:1,4:95,5:493,6:131. Taxonomy crosswalk
covers720 concepts, including128 explicitly unmapped concepts retained without
exclusion. Eligibility uses the rule adopted before counts; no rescue occurred.
[Report](runs/source_qualification/attempt002/qualification_report.json) and
its four metadata CSVs preserve every failure reason and source coordinate.
Attempt001 remains as failed diagnostic history. No roles, folds, repeat halves,
actual acquisition order, neural units/values, reader separation or scoring
are established by this result. The report's ordinal-availability list includes
image/participant pairs absent from the roster, so it is not by itself evidence
of bad ordinals among retained qualified presentations.

Root prospectively selects metadata-only chronology qualification for
implementation and reviewed bounded execution: exact positive unique source
repetition ordinals define within-image retained ranks; alternating retained
ranks define halves. Full retained acquisition order must be established from
integer run-offset constraints at consecutive full-history repetition ordinals:
start(cur)−start(prev)=trials_since_lastrep(cur)+source_order(prev)−source_order(cur).
Require cycle consistency, every retained run connected per participant,
nonoverlapping60-trial run intervals, every image order consistent and the
provider's sub01/session17 run-order exception. No numeric-run fallback,
private dates, neural similarity, dropped rows or fitted offset rescue.
Disconnected or inconsistent chronology remains unresolved. Absence is
separate from an invalid observed ordinal; qualified images must support all
four participant repeat halves. Only existing metadata and completed
attempt002 may be used, no new fetch;32MiB/file,120s internal/180s external
plus10s termination,1CPU4GiB. New code/fixtures and independent review precede
fresh runs/source_chronology/attempt001. Roles remain unadopted.


## 2026-10-04 21:47 UTC — source-grounded taxonomy repair selected before retry

The independently reviewed diagnostic002 completed in2.139663seconds in
46518769, with exactly the two pinned taxonomy objects and the bounded
presentation-identity CSV. All720 source CID/folder-name anchors are exact
uniqueID keys in both tables. Both CID-minus-one routes have zero mismatches
among these720; the eight global ordering permutations are outside this
study's selected concepts. Full1854-ID sets remain identical. No eligibility,
taxonomy memberships, roles or outcomes were evaluated by this diagnostic.
Evidence: [taxonomy_bridge_diagnostic002.json](runs/source_qualification/taxonomy_bridge_diagnostic002.json).

Root prospectively adopts a narrow implementation repair, independently
reviewed as contract-preserving: join the concepts table by exact uniqueID,
retain unique/nonempty equal1854-key sets and category53 one-based CID versus
exact source folder/name checks for every selected concept. Remove only the
unnecessary global rowwise equality assumption. No normalization, fuzzy
matching, offset fit, source-value or repeat-eligibility change. Pinned source
identities and the selected720 mappings remain unchanged.

After focused fixtures and independent implementation review, select fresh
source_qualification/attempt002 with the same120s internal/180s external plus
10s termination,1CPU4GiB,32MiB per existing CSV, two exact taxonomy GETs totaling
724963bytes,20s each/no retries. This is a diagnosed script repair; attempt001
is preserved. The revised ASTRA recipe must validate before execution.
Chronology, role optimization and assignments remain unadopted.


## 2026-10-04 21:41 UTC — `source_eligibility`: discrepancy is eight table-order permutations

The first bounded diagnostic completed1.552721s in46518769. Both1854-row
tables have unique IDs and identical exact ID sets. Eight rows differ:
zero-based801–804 permute hot-air_balloon/hot-water_bottle/hot_chocolate/
hot_tub;817–820 permute ice-cream_cone/ice_cream/ice_cube/ice_pack.
There is no whitespace or duplicate-ID explanation. Evidence is
`taxonomy_bridge_diagnostic001.json`; no eligibility was rerun.

Provider annotation code defines concept_name from the exact image parent
directory, but its numeric lookup assumes both table orders agree. Select a
second bounded metadata diagnosis against the completed attempt003 concept
ID/name anchors to identify the correct bridge. It will compare all720
exact source names to each table's ordinal and exact-ID routes, with no fuzzy
normalization or new scientific values. The prepared helper will use the same
two pinned small objects,120s internal/180s external in the existing allocation,
and a fresh bounded `taxonomy_bridge_diagnostic002.json`. Root review precedes
that command. This does not yet adopt reordered tables or retry eligibility.


## 2026-10-04 21:38 UTC — `source_eligibility`: taxonomy bridge mismatch; attempt preserved

Attempt001 stopped in3.852467s with
`taxonomy_uniqueID_bridge_mismatch` at zero-based row801. All49980
retained B/C/D presentation coordinates passed, and both taxonomy objects
matched their exact pinned lengths and annex digests. No eligibility count,
qualified universe, roles or scientific result was released. This is an
observed source-table identity discrepancy, not a transfer or cluster failure.

Select one bounded diagnosis in the current compute allocation,120s outer:
fetch those same two small public metadata objects once each using unchanged
length/digest checks and20s request bounds, inspect only uniqueID fields and
their whitespace/duplicate/order/set agreement, and release at most ten
discrepant ID pairs plus aggregate identity counts. Save a fresh
`runs/source_qualification/taxonomy_bridge_diagnostic001.json`; do not
persist raw table bodies or normative columns. This diagnostic is not a retry
of eligibility, count relaxation, fuzzy matching or adoption of a new crosswalk.
Any subsequent repair must be grounded in the observed discrepancy and source
semantics and recorded separately; preserve attempt001.


## 2026-10-04 21:36 UTC — `source_eligibility`: validation and BR advisory before execution

Changed ASTRA schema/semantics passed in46518769, record
`astra/v0.0.14/validation_20261004_source_eligibility.json`.
Root requested focused BR advisory review of prospective eligibility,
identity/taxonomy joins and interpretation limits using public design and
operational facts only. BR returned `ext_review_dir_3f7a40471cbd` at
21:36:05UTC with generic scientific review criteria and no tailored checks;
this is not a case-specific verdict, source read, scientific acceptance or
canonical launch. Local root/independent review remains the scientific
decision basis. Preserve the criteria's separation of mathematical/data
correctness from evidence quality and avoid treating eligible metadata as
independent biological evidence or proof of chronology.

Execute the selected180s envelope now in the verified shared shell.

## 2026-10-04 21:35 UTC — `source_eligibility`: bounded metadata stage selected

Root and independent static review passed the fixed eligibility/taxonomy
runner and fourteen meaningful fixtures; all14 passed in0.011s in verified
46518769/sh03-01n68.int. The fixed retained roster is all experimental rows,
as the completed attempt003 public condition counts already establish.
The code refuses a changed scope before taxonomy parsing. A reviewed deadline
race was fixed before execution: every caught failure clears both eligibility
completion flags and cannot return success.

Select fresh `runs/source_qualification/attempt001`,1CPU/4GiB,
120s internal/180s external plus10s termination, in the existing shared
allocation after changed ASTRA validation. Read only the existing completed
49980-row projection and named attempt003 metadata, then exactly two pinned
public taxonomy objects totaling724963 bytes. The fixed source identities
and provider-supported numeric/uniqueID bridge are in the specification.
Original repeat/image/concept rules were adopted21:20UTC before counts.
No behavioral response, neural, pixel or model-feature payload is selected.
The public concept metadata table has unrelated normative columns parsed
transiently; only its uniqueID bridge can reach selected output.

This selects eligibility and crosswalk qualification only. Roles, repeat
halves, actual global chronology, balance optimizer, handoffs and all scientific
scores remain closed. Any count other than720 or identity/transfer/processing
failure stops without a repaired universe, retry or partial success.
The runner's analysis_id is aligned to this new ASTRA node; that label-only
record change does not alter the fourteen already tested methods.


## 2026-10-04 21:20 UTC — `source_roles`: repeat eligibility fixed before counting

Root reviewed `source_roles_proposal.md` against the current data contract.
Adopt only its repeat/image/concept completeness interpretation at this step:
an image needs at least two structurally matched retained presentations in
each of the four fixed participants; retain all valid repetitions with equal
weight. A concept is eligible if at least one common image meets this rule.
Original five/six-exemplar status is a required design-balance covariate, not
a requirement to retain every original exemplar; report the surviving count
and all failed-image reasons separately. The contract explicitly applies
structural eligibility to each primary image and does not fix a primary image
count. These choices are frozen before calculating whether they yield720
concepts. Any other concept count stops this design before neural access;
there is no threshold relaxation, replacement or alternative-universe search.

This does not adopt the remaining taxonomy, chronology or balance optimizer
proposal and launches no new source-role process. The next useful work is to
resolve the pinned public chronology semantics and exact taxonomy objects,
then prepare the bounded metadata-only count/mapping stage. Outcomes, neural
values, image pixels and model features remain closed; role assignments and
handoffs remain incomplete. No blanket extra approval is inferred for
contract-preserving implementation choices under the scientist's resume
instruction. A changed design or frozen count still requires explicit amendment.


## 2026-10-04 21:13:24 UTC — `released_event_reconciliation`: all 833 retained runs completed

Root observed Slurm **46516829 COMPLETED, exit 0**, on `sh03-01n63.int`.
The [public reconciliation report](runs/released_event_reconciliation/retained001/reconciliation_report.json)
records `status=completed`, `retained_roster_event_reconciliation_complete=true`,
**49,980 projected presentation rows**, 830 newly matched runs and all three
pilot runs reused. Its error array is empty. The event stage took
436.048604 seconds. The preceding manifest completed all 833 entries with
835 requests in 142.618077 seconds, as recorded in the preceding entries.
Selected event bytes total 33,997,747, of which 33,869,972 were new; the
180 pilot projections were reused without redownloading. No automatic retry
was made. Evidence remains in the completed `retained001` attempt and the
existing `released_event_manifest/attempt001/annex_manifest.json`.

This completes reconciliation of the **selected retained roster**, not every
source event: `all_source_events_reconciled=false`. The trusted process read
mixed released event bodies in memory and exported only the declared metadata
projection. The report records no raw-event file writes, selected behavioral
columns, persisted or reported behavioral values, neural values, image values,
model features, candidate scoring or audit opening. No eligibility rule,
720-concept universe, roles or folds was established by this stage. Source
roles remain incomplete; the earlier unqualified 3,840-image/720-concept
metadata intersection is not promoted to an eligible universe.

Root's scheduler accounting gives start **2026-10-04 21:03:35 UTC**, end
**21:13:24 UTC**, elapsed **589 seconds at one allocated CPU**, or
**0.163611111 core-hours**. Reported process `TotalCPU` is 57.693 seconds and
`MaxRSS` is 117,528K. This is the substantive batch's separate allocation
charge; do not add the already recorded shared test-allocation charge again.
All 28 focused fixtures, batch syntax and changed ASTRA schema/semantics
passed before launch, as preserved in the preceding entries. No new test or
scheduler query was made for this documentation closeout.

The next stage is the pending prospective `source_roles` proposal: fix the
repeat-completeness rule, taxonomy release/crosswalk, unmapped and multilabel
handling, grouping, balance objective, tolerances and tie rule before applying
eligibility or assigning 480/120/120 whole-concept roles, folds and blocks.
Then establish the required role-filtered handoffs. The proposal's preparation
does not adopt those choices or open neural, calibration, image/model or audit
values. Later geometry, calibration and checkpoint/exposure requirements
remain open. No scientific model comparison or conclusion is claimed here.

## 2026-10-04 21:09 UTC — `released_event_reconciliation`: test session closed; substantive batch continues

Root exited shared PTY45495; Slurm46515379 completed normally after1690
one-CPU allocation seconds,0.469444444core-hours total. Equal shared
preparation/coordination bookkeeping assigns EP17 one third,
0.156481481core-hours, with the remainder assigned once to EP05/EP15.
These are allocation shares, not measured per-episode process CPU; root
README contains the single shared accounting entry. Do not add the total
again. The test shell is closed without renewal. Root's21:09:37UTC scheduler
observation finds substantive46516829 still RUNNING at sh03-01n63,6m02s
elapsed; its separate eventual allocation cost remains to be recorded.

## 2026-10-04 21:08 UTC — `released_event_reconciliation`: manifest completed; event stage running

Root's21:07 scheduler observation finds46516829 RUNNING at sh03-01n63.
Its own bounded log verifies `sh03-01n63.int`, SLURM_JOB_ID46516829 and
module Python3.12.1. The exact manifest completed833 runs using835 requests
in142.618077s. It reports33,997,747 total event bytes,33,869,972 new bytes,
maximum object48,005bytes and both selected transport bounds satisfied.
Evidence: `runs/released_event_manifest/attempt001/annex_manifest.json`.
The conditional event stage is now running; completion is not yet claimed.

The revised pilot/manifest/full ASTRA program passed schema and semantic
validation in job46515379, with source evidence reads skipped. Record:
`astra/v0.0.14/validation_20261004_retained_events.json`.

## 2026-10-04 21:01 UTC — `released_event_reconciliation`: retained-roster batch submitted

Final independent review passed the specification, both new runners, shared
projector,12 new fixtures and batch. All28 compute fixtures and final shell
syntax passed. Root submitted the recorded batch once with expanded episode
log paths; Slurm returned **46516829**. This is the selected1CPU/4GiB/40minute
conditional manifest/reconciliation sequence, not a claim of completed
metadata, source roles or scientific results. Root owns scheduler and log
monitoring; no worker submits duplicates or polls independently.

## 2026-10-04 21:00 UTC — `released_event_reconciliation`: full retained-roster preparation adopted

Root adopts `full_event_metadata_reconciliation_specification.md`, the two
bounded runners and combined batch after root review, independent initial
review and all28 source-free fixtures passing in0.015s on job46515379.
The generalized projector preserves the original16 tests;12 new fixtures
verify exact pinned symlink metadata, manifest scope, pilot reuse and stopped
projection semantics. No source payload was opened by tests. Shell syntax
passed; root then bound the batch explicitly to module Python3.12.1 and the
site profile, matching the known inherited-interpreter correction.

Select exactly the833 retained runs/49980 presentations from the completed
metadata:213 runs each for sub-01/02/03 and194 for sub-06. Reuse pilot001's
180 projections without redownloading. Manifest stage makes at most5 tree
and830 bounded symlink-text requests at the pinned source commit, with no
event body. The second stage is prospectively authorized only if the manifest
is complete and exact, each event is at most128KiB and all833 together at
most64MiB. Its code checks these conditions before any of830 new event GETs.
No manual size-dependent scientific choice is delegated to the manifest.

The recorded selection is1CPU/4GiB/40minutes on normal CPU_SKU:7502, no GPU,
one worker, no-requeue. Each sequential stage has1080s internal/1140s outer
and10s termination grace; combined2300s max fits2400s. This substantive
batch aggregates short metadata requests; no runtime padding, renewal or
automatic retry. The existing attended allocation cannot fit the full bound.

Fresh manifest is `runs/released_event_manifest/attempt001/annex_manifest.json`;
reconciliation is `runs/released_event_reconciliation/retained001`; scheduler
logs go to `runs/released_event_reconciliation/submission001/slurm_%j.out`
and `.err`. Root will submit `tools/run_retained_event_reconciliation.sbatch`
once with these expanded log paths after final checks. A partial projection
is diagnostic only; downstream use requires the terminal complete flag.
No eligibility, roles, folds, neural/pixel/model values or scores are activated.

## 2026-10-04 20:53 UTC — `released_event_reconciliation`: three-file pilot completed

All16 focused identity/projection/transfer fixtures passed in0.006s in
verified job46515379. The adopted source pilot completed with exit0 in
2.556249s. Its [public report](runs/released_event_reconciliation/pilot001/reconciliation_report.json)
records all three exact annex identities verified,127775 received bytes,
180 projected rows and zero mismatches/errors. The event image/order/onset
join agrees with the preserved metadata for these runs. Unknown first-repeat
delay fields remain unknown; they do not trigger exclusions or imputation.
No mixed source file, behavioral values, neural/pixel/model values, assigned
roles or scores were persisted/released. This establishes only the selected
pilot, not full source qualification.

The new ASTRA child `released_event_reconciliation` names this independently
runnable stage; earlier entries and runner reports use its parent phase
`source_metadata`. A full retained-roster specification and implementation
are now being prepared. The exact pinned annex manifest must come from
selected symlink metadata, without opening event bodies during manifest
preparation. Reuse the three completed pilot projections without redownloading.
The pilot projects about710s for833 sequential event requests, but exact
roster/byte totals and manifest-preparation cost still precede resource
adoption. No full run has yet been selected or launched. The updated ASTRA
program passed schema and semantic validation (exit0) in job46515379,
recorded in `astra/v0.0.14/validation_20261004_event_pilot.json`; source evidence
verification was skipped.

## 2026-10-04 20:44 UTC — `source_metadata`: released-event pilot adopted

The scientist requested additional useful compute. Independent source review
recovered the exact acquired-release gitlinks through fMRIPrep to official
`courtois-neuromod/things` commit b17d3697171824c198ebc6f15fe7ceee9a1f526f.
Its provider-advertised public CONP mirror serves the first pinned event
object (HEAD200,41689bytes). The missing local fMRIPrep checkout therefore
does not make released event metadata unavailable. No event body was read
during preparation and no Git mutation was performed.

Root adopts `event_metadata_reconciliation_specification.md` after root and
independent review of the specification, runner and16 proposed fixtures.
Exactly sub-01/ses-01 runs01–03 are selected:127775 expected event bytes,
one GET each,20seconds/request,540seconds internal and600seconds external.
Provider annex lengths/digests bind existing source identities. Mixed event
bytes remain in trusted-builder memory; only the explicit stimulus/identity/
onset/repetition/schedule projection is exported. Behavioral outcomes, private
scan logs, neural values, pixels, roles and model scores remain outside the
released handoff. Same-UID operation is not OS sealing.

Pinned provider schema/code establish TrialNumber1–60, order0–59 and the
exact `int((onset-3.0)/1.49)` conversion. The nested stimulus gitlink matches
the previously acquired f8f5162 table. No base/offset is fitted, no mismatch
causes row exclusion, and released repetition is not equated to retained-run
repetition. Redirects and HTTP errors close unread response bodies; telemetry
distinguishes attempted/received/identity-verified transfer states.

Root's shared allocation46515379 is verified at sh03-01n68.int, with one CPU,
6GiB,Requeue0 and end21:11:22 UTC. Use explicit module Python3.12.1 for focused
fixtures before the source pilot. Attempt is
`runs/released_event_reconciliation/pilot001`; no full-roster acquisition is
selected until measured pilot cost and reconciliation are reviewed.

## 2026-10-03 — source_metadata — attempt 003 completed; source roles remain incomplete

Coordinator allocation **46425841** completed attempt 003 with **exit 0**:
four participants processed, zero structural errors. The report retains four
optional missing `unionNaN` diagnostic-mask warnings, one per participant.
The available union/clean-mask headers and matching B/C/D axes support this
limited metadata-preparation result, not finite-voxel or neural qualification.

Evidence: `runs/source_metadata/attempt003/metadata_report.json`,
`presentation_identity.csv`, `image_repeat_summary.csv` and `run_metadata.csv`.
The unqualified intersection remains **3,840 experimental images across 720
concepts** (480 concepts with five common images and 240 with six). No image,
neural, mask-voxel or behavioral outcome values were read. No roles or folds
were assigned, no candidate was scored, and audit remains closed.

Official ASTRA schema and semantic validation passed:
`astra/v0.0.14/validation_20261003_metadata_attempt003.json`, with nine
optional-format warnings on previously planned outputs. Attempts 001 and 002
remain unchanged. The next source-role work requires raw-event identity and
schedule reconciliation plus prospective repeat-completeness/taxonomy/balance
choices before the 480/120/120 assignment. Source roles are **not complete**.

## 2026-10-03 — source_metadata — attempt 002 inspected all participants; optional-mask correction

The first launch of attempt 002 stopped in the coordinator's module setup
because `py-h5py/3.11.0` was unavailable. The script did not run or create
attempt output. After checking available modules, the coordinator used
`py-h5py/3.10.0_py312` and ran the same selected attempt in allocation
**46425841**. That actual metadata run processed all four participants and
returned **exit 2** with four errors; the sequential ASTRA validator again did
not run. Both attempt directories are preserved.

The four errors are the absent git-annex targets for the provider's
`unionNaN` diagnostic masks, one per participant. Required original-union and
clean-`unionNonNaN` mask headers were available. All twelve B/C/D `modelmd`
headers share their participant's shape and match its design presentation
count: 12,780 presentations for sub-01/02/03 and 11,640 for sub-06. Corresponding
voxel-axis sizes are 227,214, 236,701, 210,349 and 207,095. These are stored
dimensions only; finite values, response units and voxel identity remain
unverified. No neural or mask voxel values were read.

The pinned identity CSV contains 8,640 experimental, 100 catch and 100 test
images. All retained-run presentations are experimental. The four-person
intersection is 3,840 images spanning 720 provider concepts: 480 concepts
have five common images and 240 have six. These are **unqualified metadata
counts**, not the required eligibility or role split. Raw-event reconciliation
and prospective repeat-completeness/assignment specifications remain open.
Evidence: `runs/source_metadata/attempt002/metadata_report.json` and its
presentation, run and repeat CSVs.

The adapter incorrectly made the extra `unionNaN` diagnostic header fatal for
this metadata-only task. The contract does not require that complementary
mask for presentation reconstruction or the available union/clean header
description. The focused correction preserves its missing-file fact as an
optional-metadata warning; it does not synthesize a mask, waive later finite-
voxel checks or release source roles. Missing union/clean masks and B/C/D
axis mismatches remain structural errors.

Selected attempt 003 command, **1 CPU / 4 GiB / 10 minutes**, using the already
available NumPy and `py-h5py/3.10.0_py312` modules:

```bash
python3 outputs/tools/source_metadata.py \
  --source "$OAK/users/zijiao/br_autoresearch_data/restricted/cneuromod-things-1.0.1-restricted-raw" \
  --stimulus-metadata-csv "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/cneuromod_images_fmri_pinned_f8f5162_20260922/provider_git/image_paths_fmri.csv" \
  --out outputs/runs/source_metadata/attempt003
```

Only the coordinator executes this retry and the changed ASTRA validation.
No tests, source processing or scheduler queries ran in the login shell.

## 2026-10-03 — source_metadata — attempt 001 stopped; schema repair selected as attempt 002

Coordinator compute **46425841**, host **sh03-06n50**, ran attempt 001 and
returned **exit 2**, before processing any participant. The preserved
`runs/source_metadata/attempt001/metadata_report.json` records
`ValueError: unexpected provider image path`, zero processed participants,
zero raw event files and all prohibited-value access flags false. The
sequential ASTRA validation did not run after that nonzero exit.

The actual pinned CSV includes three source classes: experimental
`images_fmri/<concept>/<image>` rows, `images_catch_fmri/<image>` rows with
`condition=catch` and empty THINGS concept ID, and `images_test_fmri/<image>`
rows with `condition=test`. The original adapter incorrectly required the
experimental three-part path and nonempty concept ID on every source row.
The correction parses these three explicit schemas and retains every
runlist presentation. Catch rows keep their empty concept IDs; no artificial
concept is created. Counts are separated by provider condition, and common
experimental images/concepts remain explicitly unqualified. This is a parser
repair, not a trial exclusion, repeat rule, scientific eligibility decision or
role assignment. The next attempt's complete metadata report will establish
actual counts; no neural, behavioral or image-value access is needed.

Selected attempt 002 command, with unchanged **1 CPU / 4 GiB / 10 minutes**
and the existing NumPy/h5py runtime:

```bash
python3 outputs/tools/source_metadata.py \
  --source "$OAK/users/zijiao/br_autoresearch_data/restricted/cneuromod-things-1.0.1-restricted-raw" \
  --stimulus-metadata-csv "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/cneuromod_images_fmri_pinned_f8f5162_20260922/provider_git/image_paths_fmri.csv" \
  --out outputs/runs/source_metadata/attempt002
```

The attempt 001 directory and original sources remain unchanged. The
coordinator owns execution and ordinary ASTRA validation of the changed
program; no test or metadata processing ran in the login editing shell.

## 2026-10-03 — source_metadata — metadata preparation selected, attempt 001

The scientist's current resume instruction authorizes substantive source
preparation. Existing source-use acceptance is retained: the completed
`cneuromod_images_fmri_pinned_f8f5162_20260922/ACCEPTANCE_RECEIPT.txt`
records project academic/noncommercial scientific use on September 22. Its
`VERSION_CHAIN.txt` and central-directory reconstructability receipts bind the
exact CNeuroMod stimulus release; a full-THINGS archive is not a replacement.
No duplicate acceptance or credential inspection is required by this step.

Bounded directory inspection found that the acquired release's
`THINGS/fmriprep/` is empty. Its original GLMsingle derivatives retain the
per-participant `imgDesignNumbers.json`, `desc-sparse_design.h5` and shared
`task-things_runlist.h5`. The provider's completed local source files
`GLMsingle_makedesign.py` and `GLMsingle_run.m` document that the coordinates
encode onset TR and image index, and that model trial order follows the
runlist and within-run onset order. This permits useful identity preparation;
the required raw-event reconciliation remains incomplete.

Selected implementation: `tools/source_metadata.py`. It reads only those
named non-outcome metadata objects and the pinned `image_paths_fmri.csv`,
reports image/concept identity, sessions/runs, presentation order, repeat
counts and presentation-index gaps. It describes only the `modelmd` shape and
dtype in the B/C/D files; it never slices neural datasets. Mask inspection is
limited to each fixed 348-byte NIfTI-1 header, with no mask voxel values.
No behavioral annotation, recognition answer, reaction time, image pixel,
credential, checkpoint or public noise-ceiling value is read. Original
git-annex links may resolve only within the selected read-only source.

Selected root-compute command, **1 CPU / 4 GiB / 10 minutes**, with the existing
NumPy/h5py runtime and one BLAS/OpenMP thread:

```bash
python3 outputs/tools/source_metadata.py \
  --source "$OAK/users/zijiao/br_autoresearch_data/restricted/cneuromod-things-1.0.1-restricted-raw" \
  --stimulus-metadata-csv "$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/cneuromod_images_fmri_pinned_f8f5162_20260922/provider_git/image_paths_fmri.csv" \
  --out outputs/runs/source_metadata/attempt001
```

The attempt refuses to overwrite prior output. It emits `metadata_report.json`,
`presentation_identity.csv`, `image_repeat_summary.csv` and `run_metadata.csv`.
Any available four-person common counts are explicitly **unqualified**: no
repeat-completeness rule, eligibility decision, 480/120/120 assignment, folds,
blocks or reader release is selected from them. Raw-event absence and any
missing/misaligned structural metadata remain visible. The original source,
encrypted stimuli, completed model acquisitions and other writers are unchanged.

ASTRA adds the independently runnable `source_metadata` predecessor while
preserving `source_roles` and all later IDs. Its output is preparation, not
completion of source_roles or permission for neural scoring. The coordinator
will run the changed ASTRA object's ordinary official validation on compute;
validation availability is not a data-work gate. No test or source processing
was run in the editing shell (sh04-ln04, no SLURM_JOB_ID). Results and the
coordinator's actual allocation/step ID will be recorded after execution.

## 2026-10-03 - model_pair_registry - checkpoint acquisition, attempt 001

The scientist authorized obtaining missing nominated model assets, not neural
analysis or audit access. The selected files are official DINO ResNet-50
`dino_resnet50_pretrain.pth`, standard-width 800-epoch SwAV ResNet-50
`swav_800ep_pretrain.pth.tar`, and supervised Torchvision ResNet-50
`resnet50-0676ba61.pth` (`IMAGENET1K_V1`, not moving `DEFAULT`).

Before acquisition, the exact destination files and the two inspected default
Torch cache locations were absent. Other caches were not scanned. The durable destination is
`outputs/cache/model_checkpoints/`, already excluded from Git; `inputs/` must
not contain model caches. Official HEAD responses declare 94,345,885,
113,703,565, and 102,530,333 bytes respectively (310,579,783 bytes total).

One bounded task-owned CPU allocation will download the files and check safe
CPU-only deserialization and complete ResNet-50 backbone key/shape compatibility.
The DTN rejected the existing noninteractive SSH authentication, so this small
attended transfer will run on compute instead; no credentials or SSH settings
are changed. Planned limits: 1 CPU, 8 GiB, 30 minutes, no GPU. No images, neural
arrays, features, or outcomes are selected. Acquisition does not complete
exposure qualification or freeze the model-pair registry. Scheduler ID and
observed results are recorded below. The attended allocation request was
Slurm job `46425120`, initially pending in the normal partition.
The same pending request was reduced to 4 GiB after the checkpoint sizes and
CPU-only check were established; the 30-minute limit and one worker/thread
remain unchanged. No duplicate allocation was submitted.
Adding a second public CPU partition to this same pending request was rejected
by accounting/QOS policy; no partition, account, or QOS bypass was attempted.
After consulting Sherlock's service-job guidance, the same pending job was
successfully moved to `service` for the bounded transfer and lightweight
CPU-only checkpoint loading checks (1 CPU, 4 GiB, 30 minutes). No model
inference or training is included, and no duplicate job was submitted.

The job started on `sh03-01n70` with verified job ID `46425120`, one CPU and
4,096 MiB. All three official downloads completed at the expected byte sizes.
With Torch 2.4.1+cu121 and Torchvision 0.19.1+cu121, safe CPU-only loading,
finite values and strict 318-entry backbone loading passed for DINO and SwAV;
only SwAV's projection/prototype heads were excluded. The initial explicit
key-set check stopped on the older supervised checkpoint's 53 absent
`num_batches_tracked` buffers, not missing learned parameters. The next check
will use a fresh model and PyTorch's strict backward-compatible loading,
requiring all other keys/shapes and confirming zero-initialized counters.
No new download, allocation, inference or checkpoint rewrite is required.

Final verification passed: a fresh supervised ResNet-50 loaded all 267 stored
entries with `strict=True`, with no missing or unexpected keys. PyTorch's
backward compatibility supplied exactly the 53 legacy BatchNorm counters at
zero; learned weights and running statistics were present, finite and had
the expected shapes. The downloaded checkpoint was not modified. All three
assets are now available; no images, neural arrays, feature extraction or
model inference were used. The attended compute shell was exited afterward.
Slurm accounting confirms `46425120` completed with exit `0:0` in 2m25s
(step peak RSS 1,987,648 KiB). All three checkpoint paths are Git-ignored;
the scoped document whitespace check passed. No commit or push was requested.

| Asset | Official source | Local file | Bytes |
| --- | --- | --- | ---: |
| DINO ResNet-50 | [DINO download](https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth) | [Checkpoint](cache/model_checkpoints/dino_resnet50_pretrain.pth) | 94,345,885 |
| SwAV ResNet-50, 800 epochs | [SwAV download](https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar) | [Checkpoint](cache/model_checkpoints/swav_800ep_pretrain.pth.tar) | 113,703,565 |
| Supervised ResNet-50, IMAGENET1K_V1 | [Torchvision download](https://download.pytorch.org/models/resnet50-0676ba61.pth) | [Checkpoint](cache/model_checkpoints/resnet50-0676ba61.pth) | 102,530,333 |

This completes checkpoint acquisition only. The next scientific stage remains
`source_roles`; role-filtered handoffs, exposure qualification and the
model-pair registry are not completed by a successful download/load check.

2026-10-02 — Concept figure: generated/edited and visually inspected [ep17_conceptual_question-v3.png](ep17_conceptual_question-v3.png) with built-in image generation; exact prompts in [ep17_conceptual_question-v3-prompt.md](ep17_conceptual_question-v3-prompt.md). Replaced the support-only lead with all three measurement operations; verified equal support counts and genuinely different positive weights on the same voxels, without implying a novel explanatory forecast. Earlier assets retained; no analysis, outcome access or scientific-rule change.

## 2026-10-02 — scope/novelty repair; `model_pair_registry`, no attempt

- Replaced the single-training-property admission rule with claim-specific
  eligibility for fixed-checkpoint measurement comparisons; causal training
  attribution stays prohibited. Synchronized the ASTRA registry description.
- Broadened the title and corrected the forecast interpretation: fixed-weight
  support/ceiling forecasts equal the development edge carried to new concepts.
  Reproducibility is valuable evidence, not a new mechanistic law.
- Reviewed Conwell 2024, Prince 2022, official DINO/SwAV records and primary
  noise-ceiling methods; decision and prospective options are in
  `paper_plan.md`. BR hypothesis verification failed with
  Lucene `TooManyNestedClauses`; no valid evidence verdict or retry chain.
- All data roles, scoring/margins, budgets, model counts and audit access are
  unchanged. No data, job, checkpoint fetch, Git mutation or canonical action.
- The next execution stage remains `source_roles` when authorized; this
  amendment resolves a design inconsistency, not source or checkpoint readiness.
- Scoped `git diff --check` passed and direct field checks retained role counts,
  pair counts and audit limits. ASTRA changed only its registry description;
  no analysis IDs or artifact references changed. Official 0.0.14 validation
  remains open; it is not represented by these document checks.
- A second independent mathematical review found that beta OLS spatial-bin
  intercepts also reproduce the aggregate fitted target mean. Corrected its
  interpretation consistently: aggregate agreement is repeatability; high/low
  is descriptive association, not incremental gain-map evidence. No new gate.
  PyYAML/Brain Researcher and Node YAML parsers are unavailable; no install or
  official validation was attempted via an invented environment.
- Second reviewer verified the revised algebra and identified an aggregation
  ambiguity. Explicitly defined beta's forecast companion as the equal mean
  of twelve audit-block changes, matching its blockwise training/cross-fit
  target. Pooled-image primary beta/high-low endpoints are unchanged. Support
  and ceiling still target their aggregate edge. No new analysis or gate.

## 2026-09-30 — concrete paper design, no execution attempt

- Nominated official DINO, SwAV and supervised ResNet-50 backbones; nominated
  DK anatomical fusiform labels 1007/2007. These are design choices, not a
  qualified three-pair registry or atlas mapping.
- Focused the paper on the pure D-stage `A_N → R_N` membership edge and its
  development-only audit forecast. Preserved all five edges, three eligible
  pairs, 32–72 valid trials and the one-shot audit.
- Reviewed primary model-comparison/SSL/GLMsingle/inference literature; no
  global-priority or causal learning-objective claim is made.
- Generated and inspected `ep17_conceptual_question-v2.png`; corrected the
  schematic to give each support equal counts in every spatial bin and use
  an explicit anatomical-scope card. Exact prompts are saved; v1 retained.
- Replaced the old adapter projection with the nine-stage planned scientific
  worklist. Its analysis IDs organize future authorized work, not current
  attempts. No recipe or run was fabricated; official validation is open.
- Scoped document links, YAML parsing and internal artifact references were
  checked. No stimulus, neural response, score, checkpoint or audit was read.
- Active experiment/attempt/job: none. Next planned stage: `source_roles`,
  only after separate authorization for the required source/role work.

## 2026-09-27 — scientific design revision

- Reframed the episode from a model-ranking robustness grid to a prediction
  about operator–advantage alignment.
- Distinguished beta response change, voxel-membership change, and positive
  ceiling reweighting.
- Required common hyperparameters for beta edges and one common prediction
  bank for support edges.
- Restored the fixed five-edge endpoint family, participant-by-ROI indexing,
  separate raw and normalized margins, and relation-specific audit gates.
- Separated exact development attribution from a development-only prospective
  audit forecast and specified the beta reliability-gain cross-fit.
- Generated `ep17_conceptual_question.png` with the image-generation tool from
  a synthetic prompt; it contains no source stimulus or neural value.
- `astra-milestone plan` completed successfully with no validation warnings.
- No B/C/D neural array, candidate score, or sealed concept outcome was read.
