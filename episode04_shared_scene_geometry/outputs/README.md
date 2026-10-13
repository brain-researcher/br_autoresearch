# Outputs

[Public LAION-fMRI assets and documentation](../DATASETS.md#public-online-resources--checked-october-6-2026)
are now listed directly in DATASETS, including caption/mask coverage and release corrections.

The [source handoff](source_handoff.md#completed-development-evidence--october-6-2026)
and [DATASETS](../DATASETS.md) now incorporate the completed development
coverage and terminal decision; their older source-readiness snapshots are
explicitly historical.

**Scientific terminal result, October6,01:21UTC: `closed_no_candidate`.**
The [complete fixed-panel decision](runs/panel_decision/registered_panel_decision.json)
admits all required controls/diagnostics and selects no VLM. Mean native gains
over DINOv2 are CLIP-0.0348, PEcore+0.0010 and SigLIP2-0.0392 Fisher z; none
meets the registered+0.02 margin or required participant/influence conditions.
This is a five-participant development result, without a population or causal
claim. All25 control bundles and collection/reporting46690071 completed0:0.
No candidate rerun, configuration lock or OOD audit follows, and no EP04 job
or reservation remains. See the [terminal interpretation and accounting](experiment_log.md#2026-10-06-0121-utc--candidate_decision--closed_no_candidate-reporting-complete).

[View the complete result figure](figures/registered_panel_decision/20261005-full01/registered_panel_decision.png)
 · [SVG](figures/registered_panel_decision/20261005-full01/registered_panel_decision.svg)
 · [Caption and limits](figures/registered_panel_decision/20261005-full01/caption.md)

The following dated snapshots are historical and are superseded by this result.

**October6,01:16UTC:** all **25 tasks of46688490 completed successfully**.
The dependent complete-panel collection/decision job **46690071 is RUNNING**.
See [completion and once-only accounting](experiment_log.md#2026-10-06-0116-utc--controls_diagnostics--all25-bundles-complete-collection-running).
The array consumed19.7319444444CPU-hours; no candidate result is yet observed.

**October6,00:55UTC:** array46688490 has **18 completed tasks (exit0), seven
running and no reported failures**. All25 encoding unit reports already contain
their complete/evaluable400 rows with no errors; see the
[unit-completion record](experiment_log.md#2026-10-06-0048-utc--all-encoding-unit-reports-complete-first-eight-bundles-finished).
Full collection and the one candidate decision remain pending in46690071.
The [conditional selected-row rerun](experiment_log.md#2026-10-06-0055-utc--development_rerun-implementation-and-conditional-dispatch-prepared)
now has a concrete reviewed driver, wrapper and six unrun focused checks.
It will be needed only if the saved decision selects a candidate; no rerun
allocation or extra budget reservation has been made. Earlier snapshots below
are historical; they do not override this observed stage.

**Complete-panel successor submitted, October6,00:13UTC:** **46690071**
waits for all of array46688490 to succeed, then collects the complete null
and encoding panels, performs the one registered decision and renders its
saved effects/gates. At00:13:45UTC all25 array tasks were RUNNING, with no
failed/canceled/timed-out/out-of-memory entries reported. Successor resources
are1CPU/4GiB/40m, with no retry, lock or OOD
successor. See the [bounded selection and submission](experiment_log.md#2026-10-06-0013-utc--select-dependent-complete-panel-aggregation-and-reporting).
No complete collection or candidate decision has occurred yet.

**Controls advancing, October6,00:02UTC:** first parent46680856 completed
successfully, including all1600 evaluable sensitivity rows and22 passing
checks. Array46688490 has9 running and16 pending slots at00:01UTC.
See the [completion and accounting record](experiment_log.md#2026-10-06-0002-utc--first-controls-complete-remainder-running).
All required null/encoding collection and candidate screening remain pending.

**Remaining controls submitted, October5,23:59UTC:** array **46688490**
contains25 participant/fold bundles, up to25 concurrent, each1CPU/4GiB/90m.
The first encoding/null units completed successfully and all22 checks passed;
their measured costs determined this dispatch. Completed prefixes are retained.
See the [selection and submission record](experiment_log.md#2026-10-05-2358-utc--measured-controls-pass-select-25-bundled-remainder-tasks)
and [explicit manifest](code/remaining_controls_selected/20261005-full01/manifest.json).
Sensitivity job46680856 remains independent. Full collection and one candidate
screen await complete required inputs; the original120-hour clock continues.

**Observed panel complete, October5,23:46UTC:** job **46675158** finished
successfully. All **400 native/low-level partial MRI comparisons** across the
five participants and25folds are evaluable, with no errors or missing EVC
support. Independent review passed. The [completion record](experiment_log.md#2026-10-05-2346-utc--fixed_panel--all400-observed-comparisons-complete)
links the report, original120-hour clock and once-only0.2625CPU-hour charge.
Required sensitivities, encoding and nulls remain incomplete; no candidate
screen or lock has occurred. Control job **46680856** is now RUNNING on
`sh04-04n24`, with ASTRA and all four encoding checks passing. Reviewed remaining-task preparation
preserves the first-pass fits/draws; dispatch still waits for measured cost.

**Completed control / running panel, October5,23:31 UTC:** all **23,560**
approved development images now have complete finite3150-dimensional RGB
controls. Job **46674207** completed successfully in95minutes and consumed
1.5833333333 allocated CPU-hours once. Independent review passed; shared/OOD
images remain closed. See the [completion and resource record](experiment_log.md#2026-10-05-2331-utc--full-low-level-coverage-complete-observed-panel-running).
The fixed400-comparison observed MRI panel **46675158** is now RUNNING;
controls job **46680856** waits on its success. No primary-panel completion
or candidate decision is yet observed. This supersedes older statements below
that full image-control coverage was pending.

**Running, October 5, 21:53 UTC:** job **46674207** is extracting the exact
[RGB block-moment control](low_level_control_proposal.md) for all 23,560
unique development images, with 1 CPU / 2 GiB / 3h. The fixed 60-image sample
completed successfully in 21.98 seconds; measured cost determined the full
allocation. See the [execution log](experiment_log.md) for evidence and
once-only accounting. Complete finite coverage remains a prerequisite for
primary scoring; shared/OOD images remain closed. Observed-panel job
**46675158** is submitted with `afterok:46674207`; it independently checks
complete features before the fixed 400 real-MRI comparisons. No primary
score or candidate decision is yet complete.
The reviewed first controls pass is submitted as **46680856**, dependent on
the observed panel, with 2 CPUs / 8 GiB / 50 minutes. It includes all four
sensitivities and retained encoding/null cost units; full remainder dispatch
waits for measured cost.
Both complete-panel collectors and the single-decision consumer have passed
independent static review; their focused checks are included in that existing
allocation. Collection and candidate selection remain unexecuted. The null
collector requires all registered draws and all 25 cells before thresholds;
the decision also requires complete sensitivities, encoding and matched EVC
context. The separately scoped EP12 completed-table figure shares the allocation
after the sensitivity worker, with separate status and no duplicate charge.

**Approved, October 5:** the scientist approved the exact
[RGB block-moment control](low_level_control_proposal.md): a 60-image timing
sample followed by all 23,560 unique development images. The existing DUA and
passed analytic checks are reused. Root owns launch and measured full-run
resources; full coverage is not yet established. Shared/OOD images
remain closed. This supersedes the older pending-approval statements below.

**Completed, October 5, 20:22 UTC:** the matched training-context table now
contains **all five participants, 25 folds and 400 evaluable MRI–model
comparisons**, with no errors. Root and independent review confirmed exact
coverage and matched support. The [completion record](experiment_log.md#2026-10-05-2020-utc--all-five-participants-matched-training-table-complete)
links timings, reports and validation. Shared job **46660232 completed
successfully**, including all four EP04 workers and the separately scoped
EP03/EP19 source steps. Its whole allocation consumed **0.9644444444 core-hours**,
charged once under EP04. No successor is selected.

[View figure](figures/training_context_rsa/20261005-display-46660232/training_context_rsa.png)
· [Download SVG](figures/training_context_rsa/20261005-display-46660232/training_context_rsa.svg)
· [All 400 rows with support/reliability context](figures/training_context_rsa/20261005-display-46660232/training_context_rsa.csv)

These are descriptive training data. Held-out primary scoring and candidate
selection remain unstarted; the approved raw-image control still requires
complete extraction. OOD outcomes remain closed. Dated entries below preserve history.

**October 5, 20:05 UTC:** sub-01 completed all **80 evaluable matched
training-context RSA rows**, five folds, with no errors in 490.14 seconds.
The [completed report](runs/training_context_rsa/20261005-rsa-pilot-46658777/sub-01/training_context_rsa.json)
is preserved. Shared parent 46658777 failed only because its EP03 step timed
out; EP04 and EP19 completed. The [completion and selection record](experiment_log.md#2026-10-05-2004-utc--sub-01-rsa-complete-remaining-four-selected)
contains independent review, measured resources and once-only accounting.

Job **46660232** is submitted for the remaining four participants concurrently:
sub-03/05/06/07, **320 rows**, using the same frozen calculation. It then runs
separately scoped EP03 and EP19 source steps. Request: 4 CPUs, 28 GiB, 45 minutes.
The [new output directory](runs/training_context_rsa/20261005-rsa-remaining-46660232/)
will retain every action's status. These are descriptive training results;
primary scoring and candidate selection remain unstarted. Raw-image control
approval remains pending, and OOD outcomes remain closed.

The dated entries below are historical observations.

**Submitted, October 5, 19:52 UTC:** corrected job **46658777** runs a bounded sub-01 pilot for the
required matched training-context RSA table: all five folds, four ROIs and
four released models (**80 rows**), using completed training mean patterns
and identical saved image/repeat/voxel support. The [driver](code/run_training_context_rsa.py)
reuses the qualified native-RSA arithmetic and existing reliability/NC context.
The [prospective record](experiment_log.md#2026-10-05-1947-utc--matched-training-context-rsa-pilot-selected)
and [setup correction](experiment_log.md#2026-10-05-1952-utc--memory-unit-request-corrected-before-any-source-processing)
define the 1 CPU / 7 GiB / 40-minute shared parent and 1,500-second RSA cap.
The earlier CPU-guard failure opened no data. This is not a completed result. Remaining participants
are not automatically dispatched. The table is descriptive training context
and cannot select a candidate or establish held-out performance. Raw-image
approval requested at 19:38 UTC remains pending; low-level controls and the
primary panel are not activated by this pilot.

**Completed, October 5, 19:35 UTC:** training-only repeat reliability and
provider-formula noise-ceiling components are complete for **all five
participants: 25 folds and 100 ROI units**, with no errors or EVC support gaps.
The preserved sub-05 result is joined by sub-01 and the parallel sub-03/06/07
pack. Both new parent jobs completed successfully and used **0.1066666667
allocated core-hours** in total. The [completion record and per-participant
reports](experiment_log.md#2026-10-05-1935-utc--all-five-participants-training-context-complete)
contain timings, support and once-only accounting.

Original training-only calculations and completed preprocessing remain fixed.
The matched training-context RSA table, fixed model panel, low-level control
and OOD audit remain unfinished; this component is not a model-alignment
finding. No primary panel evaluation has started the 120-hour clock, and
raw-image extension approval remains outstanding. No retry or simulation
continuation is selected.

The [initialization prefix](runs/training_neural_context/20261005-initialization-diagnostic-46650095/initialization.jsonl)
completed in7.970102s with4,712 images/30sessions/12,833voxels and no beta values.
The subsequent [training-chunk diagnostic](runs/training_neural_context/20261005-chunk-diagnostic-46651952/chunk_diagnostic.json)
completed in13.682356s with30 training slices,3,859,456 values and one original
moment call; no scientific values were exported. Both shared parent costs
remain charged once to EP01. The measured260-chunk full workload suggests
roughly22minutes before cache/layout variation. These measurements support
the new finite resource plan but do not identify the earlier timeout's cause:
the old first checkpoint followed the first complete ROI, so missing output
never established a pre-fold stall.

Earlier execution, October3 at23:40UTC: **all150 source sessions and all25
participant-fold preprocessing units are complete**. The remaining bundle
46438525 finished successfully in6:54; see its
[completion report](runs/neural_preprocessing/20261003-remaining-folds01/bundle.json).
The [sub-05 training-only reliability/NC components](runs/training_neural_context/20261003-sub05-folds01/training_context.json)
also completed all five folds in56.56s, with no errors and no EVC support gaps.
Five focused checks and ASTRA validation passed. The remaining sequence stopped
on sub-01's external190s timeout, with no fold/report emitted; the other three
participants were not launched. The affected route is stopped pending diagnosis,
with all earlier completions preserved. Raw-image approval, the matched training-context RSA table,
model comparisons and the OOD audit remain separate unfinished stages.
The [latest log entries](experiment_log.md) supersede the earlier snapshots below.

October3,2026: [sub-05 neural preprocessing completed](runs/neural_preprocessing/20261003-sub05-folds01/preprocessing.json)
in all five folds,327.327s, with1,126,208KiB process peak RSS and no errors.
All4,712 images retain primary minimum-two support;4,707 meet complete-four.
Every fold retains all frozen ROI voxels, every source cell is usable, and no
primary evaluation image lacks EVC support. This establishes preprocessing
support, not model alignment or diagnostic completion. At that milestone no RDM, reliability/NC,
encoding or model score has been computed; the panel and OOD audit remain closed.
Exact fold counts and interpretation are in the [experiment log](experiment_log.md).

The [unchanged preprocessing implementation](code/preprocess_development_neural.py)
uses the [prospectively adopted recipe](neural_preprocessing_proposal.md):
training-only session statistics and voxel support, common-primary repeat
means, separate EVC validity, float64 and unchanged folds. All six focused
fixtures passed in0.020s in attended allocation46436630, and the
[ASTRA validation](astra/v0.0.14/validation_20261003_neural_preprocessing01.json)
passed schema and semantics. The recipe document's initial no-execution
statement is historical; the dated run and log record the completed pilot.

Root submitted the [remaining-participant bundle](code/neural_preprocessing_remaining01.sbatch)
once as job **46438525** at23:15UTC with dependency `afterok:46436056`.
It covers sub-01/03/06/07:2CPUs/8GiB/35minutes, two single-thread children,
900s external/840s internal per child, no retry. Root static review and
compute-side AST parsing/shell syntax checks passed. The
[finite runner](code/preprocess_development_neural_remaining01.py) preserves
partials and starts no second wave after failure. At23:20UTC root observed it
RUNNING at elapsed1:21 on`sh03-02n16`; completion remains unobserved.
Root owns monitoring and active code is frozen. The
[log](experiment_log.md) records exact commands and resource reasoning.
The source array46436056 is now **complete across all150 sessions**, with
94,216 selected development presentations and no reported participant errors.
The final [sub-07 handoff](runs/source_readiness/20261003-development_neural_full01/sub-07/participant_handoff.json)
finished23:17:04UTC with all30 sessions and18,842 selected presentations.

The [training-only reliability/NC module](code/training_neural_context.py) and
[five scientific fixtures](tests/test_training_neural_context.py) passed
independent static review and await root compute checks/execution. The selected first pilot is sub-05,
all five folds,1CPU/4GiB/15minutes. It reuses saved preprocessing parameters,
support and validity; the common primary/EVC training-ID intersection is
diagnostic only and cannot change primary coverage. It computes signed repeat
consistency and the adopted provider-formula NC, with exact repeat/voxel IDs
and explicit missingness. No RDM or RSA is included. Execution and changed
ASTRA validation remain pending; see the [selected commands](experiment_log.md).

Earlier completed preparation remains reusable:

- [Metadata and five folds](runs/source_readiness/20261003-metadata_folds04/source_metadata_report.json):
  all150 trial tables,23,560 development images,94,216 presentations and14,285
  groups; seven checks passed. [Frozen fold rules](config/development_metadata.json)
  preserve neighbour components and balance participant/source cells within
  one image. Metadata contain23,536 four-repeat and24 three-repeat images.
- [Array/ROI structural handoff](runs/source_readiness/20261003-array_metadata02/array_metadata.json):
  all beta-header/sidecar joins, four embedding declarations,15 primary masks
  and30 retinotopic components. The
  [grid diagnostic](runs/source_readiness/20261003-roi_grid_offsets01/roi_grid_offsets.json)
  supports the adopted [same-index mapping](config/roi_mapping.json), with
  maximum measured displacement2.8875e-05 beta voxels and no resampling.
- [Development embedding handoff](runs/source_readiness/20261003-development_embeddings01/embedding_handoff.json):
  all23,560 frozen IDs across four existing arrays, original float16 values,
  common row order and no errors. No normalization, RDM or fit was performed.
- [First-session neural source pilot](runs/source_readiness/20261003-development_neural_pilot01/neural_handoff.json):
  575 selected trials across12,833 union voxels in26.02s; zero estimates were
  preserved and excluded trials were never numerically decoded. Trusted row
  filtering is not an OS-enforced OOD seal.
- [Equation qualification](runs/implementation_qualification/job_46425841/qualification.json):
  ten mathematical checks passed for the [native-RSA module](code/native_rsa.py).
  This does not qualify the full empirical panel. The
  [ASTRA program](astra/v0.0.14/astra.yaml) records analysis dependencies.

The [low-level control proposal](low_level_control_proposal.md) remains
inactive pending the contract's separate raw-image extension approval.
The accepted DUA, exact RGB block-moment settings, reader and bounded plan are
reviewable; three generated-pixel analytic checks passed. No raw pixels have
been opened. The [source handoff](source_handoff.md) records remaining source
and pre-score requirements. Required matched training-image RSA/reliability/
provider-formula NC context and full panel execution remain future work.

The [conceptual figure](ep04_conceptual_question-v2.png) and its
[exact generation/correction prompts](ep04_conceptual_question-v2-prompt.md)
illustrate native RSA and the conditional within-category image-shift audit.
All depicted images, embeddings, RDMs and brain shading are illustrative;
no observed winner or causal language effect is shown. Earlier artifacts and
failed attempts remain preserved. Inputs are read-only and excluded neural
outcomes remain closed.
