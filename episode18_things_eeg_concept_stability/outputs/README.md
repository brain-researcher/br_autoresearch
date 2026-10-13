# EP18 outputs

**October12,16:55Pacific — all30 participants' real EEG preprocessing complete.**
The [production report](runs/eeg_preprocessing/development_20261012_001/operational.json)
confirms all2160 intervals complete,9.56GB durable output,646.08seconds
preparation and0.86GB process peakRSS. Batch exit0 is recorded. Image feature
extraction continues; no encoding-model score or audit opening is claimed.
See the [current execution record](experiment_log.md). Earlier states below
are historical.

**October12,16:41Pacific — real EEG preprocessing running; first participant complete.**
Job47572924 is running onsh02-08n53. Both cache tests and changed ASTRA passed;
sub-07's complete cache preserves all72 sequences and22248 main events, with
586832 full-grid samples and583232 scored samples. See the
[first participant completion](runs/eeg_preprocessing/development_20261012_001/sub-07/complete.json)
and [running-stage record](experiment_log.md#2026-10-12-1640-pacific--development_eeg_preprocessing--allocation-running-and-checks-passed).
The all30-person manifest is not yet complete. Image extraction continues;
the full fitting backend's exact capacity computation remains unresolved.
These are preparation results, with no EEG prediction score or scientific claim.

**October12,16:33Pacific — all30-person EEG preprocessing submitted.**
Job47572924 requests1CPU/7GiB/2h for the fixed preparation of all2160 main
sequences, preserving62 channels and the original common masks. Cache and
wrapper source reviews passed; two cache checks precede actual EEG reads.
See the [selection and submission](experiment_log.md#2026-10-12-1632-pacific--development_eeg_preprocessing--complete-development-eeg-preparation-selected).
This is preprocessing, with no model fit or scientific score. Image extraction
and the dependent false-group cost job47571099 continue separately.

**October12,16:18Pacific — first production false-group cost job submitted.**
Job47571099 requests13CPU/96GiB/4h and depends on the existing complete-image
assembler47551339. It prepares all30 participants' structural inputs, then
builds the full first-participant cache and measures the fixed eight-group-bank
search. It uses one numerical thread; CPU count satisfies the scheduler's
memory requirement. Earlier resource-request issues were corrected before
scientific execution. See the [execution record](experiment_log.md).
All2160 development sequences are bound; image extraction/assembly continues.
No EEG fit or accepted false-group bank exists. Earlier entries are historical.


**October12,16:08Pacific — development geometry complete; image features still running.**
All30 development participants,2160 sequences,667440 main events and the
original62-channel order passed exact metadata binding. The causal filter and
recorded-E3 span are now fixed; three affected reader checks and ASTRA passed.
See the [complete geometry evidence](runs/feature_preparation/development_main_geometry_20261012_001/operational.json)
and [execution log](experiment_log.md). A first complete-participant cost run
for false-group construction is being prepared behind image assembly. No EEG
fit or accepted false-group bank exists. Earlier entries below are historical.

**October12,15:48Pacific — all22448 images delivered;22 feature workers running.**
Builder47551249 completed successfully:702 pixel packs cover every original
image, including the200 repeated images. At the15:48Pacific scheduler check,
22 workers in the original extraction arrays were running; four fast tasks
waited for their concurrency limit and the assembler waited on dependencies.
This completes pixel delivery; the six-family features and final assembly
are still in progress. See the
[completion and accounting](experiment_log.md#2026-10-12-1548-pacific--feature_preparation--all-image-delivery-complete-22-feature-workers-running).

**October12,15:44Pacific — development metadata/FIR inspection complete.**
All30 development recordings have1000Hz timing,62 common recorded channels
and sufficient recorded sequence margins for the proposed scoring/filter
support. The inspection took19.93seconds and121.84MiB peak RSS in EP15
parent47565811; it read headers/markers and EEG file sizes, not EEG amplitudes.
The prospective129-tap causal filter's64ms delay and numerical frequency
response are recorded; the filter is **not adopted**. Actual E3-to-first-image
gaps vary, so the reader's fixed1000-sample assumption needs replacement with
the authentic per-sequence join before EEG preparation. Main-sequence identity
and frozen channel-order checks still belong to that integration. See the
[completion and remaining boundary](experiment_log.md#2026-10-12-1544-pacific--input_role_preparation--development-geometry-and-prospective-fir-inspection-completed).
No EEG comparison or scientific result is available.

**October12,15:29Pacific — false-partition search implementation checks passed.**
Both focused search checks and changed ASTRA passed; earlier reader/design
checks also passed. These are software results, with no accepted empirical
partition bank or EEG score. At15:25Pacific the active image builder had
prepared18080/22448 pixel-pack images (80.5%); the26 feature tasks and assembly
still depend on completed preparation. See the [execution log](experiment_log.md).
EEG preparation and full-scale exact capacity computation remain unfinished.


**October12,14:39Pacific — tuning composition verified; sequence source resolved.**
The continuous EEG preparation bridge passed three checks; nested tuning and
capacity matching passed all four focused checks plus changed ASTRA. These
are software results. The original paper resolves E3 as sequence start; the
new explicit BrainVision reader binds recorded units and development-only paths,
but its focused tests and empirical execution remain pending. Full-image
preparation and its26 dependent feature tasks continue. No EEG comparison or
scientific claim is available. See the [checks, source and disclosure](experiment_log.md#2026-10-12-1439-pacific--baseline_search--tuning-checks-complete-sequence-trigger-source-resolved).

**October 12, 14:13 Pacific — full-image preparation running.**
Builder **47551249** is running on `sh03-06n09`. Configuration admission and
ASTRA validation passed; the operational stream confirms real pixel-pack
preparation has begun. The 26 feature tasks and assembler remain queued on
their successful dependencies. EP17's independent companion exited before
its operational directory was created; this did not stop EP18 preparation.
See the [startup observation](experiment_log.md#2026-10-12-1413-pacific--feature_preparation--builder-running-and-independent-companion-stopped).

**October12,14:07Pacific — full real-image extraction submitted.**
The chain is builder47551249 → fast array47551335 plus BLIP array47551337 →
assembler47551339. It covers **22448 images × six feature families**, with
26 feature tasks and at most22 concurrent workers, bounded to91CPUh total.
The builder was pending at the last observation; dependent tasks wait for
successful prerequisites. EP17's separately scoped32-image pilot shares a
bounded step inside the builder parent. See the
[selection, resource accounting and submission](experiment_log.md#2026-10-12-1405-pacific--feature_preparation--full-real-image-extraction-selected).
The completed small pilot below is the cost evidence; full features, false
partitions and empirical EEG results remain unfinished.

**October 12, 13:49 Pacific — all six real-image pilot families complete.**
The fixed32-image pilot finished successfully in299.14seconds; all six
singleton-repeat checks agreed exactly. The measured encoding costs project
to about37CPU-hours for22448images, before loading/I/O and variability
allowances. See the [completion, costs and next work](experiment_log.md#2026-10-12-1349-pacific--feature_preparation--all-six-real-image-pilot-families-complete).
Full extraction code passed three focused checks and independent source
review, including a duplicate-invocation history-preservation repair. The
attended parent completed and closed at13:55Pacific, using0.252222allocatedCPUh.
Full extraction is not launched. Full image features,
eight false partitions and the complete EEG fitting/evaluation pipeline
remain unfinished; no empirical EEG result is claimed.

**October 12, 13:46 Pacific — real-image cost pilot running; fold bridge tested.**
Parent47547023 is running the fixed32-image/six-family CPU pilot after all
five image-runner checks, four full-roster fold checks and changed ASTRA passed.
A private-notice format mismatch stopped the first attempt before decoding;
its narrow parser repair passed, and the fresh repaired attempt retains the
same allocation/deadline. See the [diagnosis, checks and selected repaired run](experiment_log.md#2026-10-12-1343-pacific--feature_preparation--private-notice-parser-diagnosis-and-bounded-repair-selected).
The [fold bridge](code/continuous_fold_runner.py) composes all six outer and
five inner splits with shared masks and training-only whitening. It still
requires raw preprocessing, the complete tuning/EDF caller and final scoring
convention; it is not an empirical EEG result. Image extraction is real-data
preparation. Full extraction, eight false partitions and EEG comparison remain
unfinished. The procedural path-listing disclosure is retained in the
[implementation entry](experiment_log.md#2026-10-12-1334-pacific--feature_preparation-and-baseline_search--implementation-resumed-under-adopted-procedural-boundary).

**October 12, 13:23 Pacific — procedural separation adopted:** disclosed same-account procedural separation replaces enforced EEG/pixel-worker isolation, so the SRCC draft is not needed and isolation requirements below are superseded; remaining features, partitions and the EEG runner still gate fitting ([decision](experiment_log.md#2026-10-12-1323-pacific--scientist-decision-procedural-same-account-separation-adopted)).

**October 11, 23:44 Pacific — current isolation support handoff prepared:**
the [unsent SRCC draft](experiment_log.md#2026-10-11-2344-pacific--input_role_preparation-current-unsent-isolation-support-handoff)
asks for a supported isolated image/EEG evaluator route using the retained
October 3 namespace failure. Exact runtime/assets are already qualified;
no new fault observation, retry, weaker boundary or support contact occurred.

**October 7, 02:21 Pacific — public-label geometry complete:**
job **46881096** prepared all **1,854 labels**, representing all **27 requested
rank triples / 81 bandwidth mappings** with **six geometries and 18 kernels**.
The runner took **33.25 seconds**, reported **310.72 MiB peak RSS**, and retained
**507,723,286 cached array bytes** within its 600 MB allowance. Its cost is
included once in the EP21 parent. Image/EEG access and empirical fitting remain
unreleased. See the [completion record](experiment_log.md#2026-10-07-0221-pacific--feature_preparation-public-label-geometry-completes)
and [closed report](runs/feature_preparation/label_geometry_46881096/label_geometry_report.json).
The earlier unrun statements below are historical snapshots.

**October 7, 00:47 Pacific — public-label geometry production selected, unrun:**
the coordinator selected the existing PCA/RBF runner for fresh EP21 global
index **12 / work_0013**, as a separate serial EP18 child bounded to **1 CPU /
7 GiB / 1,800 seconds / zero GPUs**. It reuses the 27 passing checks and reads
only public-label features and permitted preparation records. The full
eight-hour parent reservation and actual cost remain charged once to EP21;
EP18 records its child time as included, with a planned **600,000,000-byte**
storage allowance and actual bytes to reconcile. No parent startup or cache
completion is claimed. See the [selected scope and accounting](experiment_log.md#2026-10-07-0047-pacific--feature_preparation-public-label-geometry-selected-for-shared-worker).

**October 6, 22:59 Pacific — 27 component checks passed:**
the training transforms, continuous sequence model, slot-capacity calculation
and label RBF checks all passed inside EP21 job **46864089**, with **0.041 s**
summed test-suite runtime. EP18 ASTRA schema and semantic validation also
passed. At that documentation update the coordinator reported the parent
active; its full cost is charged once to EP21, with no final charge then recorded.
Label-geometry preparation, image/EEG access, empirical fitting and scoring
did not run. See the [component execution record](experiment_log.md#2026-10-06-2259-pacific--baseline_search-component-checks-pass-in-ep21-job-46864089)
and [check receipts](runs/component_checks_job_46864089/status.txt).
The historical pending-check statements below are superseded by this result;
the separate preparation and access dependencies remain open.

**Historical October 6, 21:32 Pacific — nonlinear label component reviewed:**
the [RBF component](code/label_rbf.py) now supplies the registered nonlinear
label geometry. Coordinator and independent static reviews found no blocking
issue; four new checks remain unrun. Identical training-concept inventories
permit reuse of label transforms across folds; the
[finite preparation runner](code/prepare_label_geometry.py) is now ready and
statically reviewed, with execution still pending. No new compute reservation
or EEG access was selected at that time; the October 7 selection above
supersedes this historical allocation status. See the
[review and next execution scope](experiment_log.md#2026-10-06-2132-pacific--baseline_search-nonlinear-label-component-reviewed).

**Historical October 6, 21:21 Pacific — capacity matching implemented, checks pending:**
the [new component](code/slot_capacity.py) measures added-slot flexibility
under the joint fit and lists eligible pairs on the existing penalty grid.
Static review is complete; five checks and changed ASTRA validation remain
unrun. Dense resource qualification, RBF construction and empirical fitting
remain unfinished. See the [implementation record](experiment_log.md#2026-10-06-2121-pacific--baseline_search-exact-capacity-component-prepared).

**Historical October 6, 21:08 Pacific — training transforms reviewed, checks pending:**
the [new component](code/training_transforms.py) implements the registered
training-only feature reductions, fixed sensor projection and explicitly
specified covariance whitening. Static review is complete; eight new checks,
ten affected sequence checks and changed ASTRA validation await the next
suitable substantive allocation. See the [review and execution scope](experiment_log.md#2026-10-06-2108-pacific--baseline_search-training-transforms-reviewed-compute-pending).
No image/EEG reader release or empirical fit has occurred.

**October 6 Pacific — continuous fitting core checks passed:**
the [sequence-local numerical model](code/continuous_sequence_model.py) handles
overlapping events, a direct joint nuisance/feature ridge fit and fitted-event
ablation. All **ten focused checks passed in 0.011 s** in the coordinated
one-CPU / 7-GiB session **46845646** on **sh04-18n32**; changed ASTRA validation
also passed. This supplies no EEG reader release or empirical fit. See the
[execution evidence and remaining pipeline](experiment_log.md#2026-10-06-pacific--baseline_search-sequence-local-fitting-checks-pass).

**October 6 Pacific — all concept-name features complete:** allocation
**46843116** completed the first-64-name pilot and, after explicit coordinator
release, all **1,854** prepared names. The [full label-feature report](runs/feature_preparation/all_name_text_46843116/label_feature_report.json)
records a **1,854 by 512 float32** CLIP matrix in **169.455 s**, peak
**609,144 KiB**, with zero token truncations or provider-name fallbacks.
The [exact runtime and public assets](runs/feature_preparation/exact_cpu_provision_46842156/model_runtime/model_runtime_qualification.json)
had qualified in **46842156**. Enforced image-worker and EEG isolation,
remaining image/caption features and the EEG fitting/evaluation pipeline are
unfinished. The label process read no pixels, private image routing, events
or EEG and emitted no complete feature freeze or access seal. See the
[execution log](experiment_log.md) for attempts and validation results.

The attended allocation is **COMPLETED 0:0** and closed, with **366 allocated
CPU-seconds** charged once. Updated workflow validation passed; its original
dependency failure remains preserved in the log. Together with model
provisioning, these two jobs used 713 CPU-seconds, excluding earlier history.

## Historical execution snapshots

The dated snapshots below preserve what was known at each point; the current
readiness statement above supersedes their pending/unobserved descriptions.

**Historical October 6, 18:30 Pacific — model provisioning complete:** job **46842156**
finished successfully, using **347 allocated CPU-seconds**. All fixed public
model assets passed source-integrity checks and all six selected model routes
loaded offline with the exact stable runtime. No inference, image or EEG access
occurred. See the [completion and remaining boundaries](experiment_log.md#2026-10-06-1830-pacific--feature_preparation-exact-runtime-and-all-public-model-assets-qualified).
The existing first-64-name label-only cost pilot is selected in a separate
attended session; image-worker and EEG audit isolation remain unresolved.
The two changed ASTRA documents await a retry with the diagnosed dependency
path correction, preserving the original missing-YAML failure logs.

**Historical October 6, 18:19 Pacific:** provisioning job **46842156** is submitted with
one CPU / 7 GiB / two hours. It will install the exact stable CPU runtime,
acquire the fixed public model assets and verify/load them offline. Shared
EP09 identity and EP20 circuit checks use this allocation serially. Startup
is pending; no feature inference or EEG access is released. See the
[submission and selected scope](experiment_log.md#2026-10-06-1819-pacific--feature_preparation-provisioning-job-46842156-submitted).

Historical October 6 Pacific: a [single finite provisioning job](code/provision_exact_cpu_runtime_001.sbatch)
is prepared for the exact CPU-wheel runtime and the fixed public model assets:
one CPU, 7 GiB, at most two hours. It preserves the old environment, checks
source integrity before offline model loading, and performs no feature/text
inference or EEG access. The [current execution route](paper_plan.md#october-6-pacific-selected-compute-provisioning-correction)
corrects the impossible interactive-DTN handoff; the fixed public files do not
require account credentials. Runtime success and source delivery remain
unobserved, and both pixel-worker separation and the EEG audit seal remain
unverified. See the [execution log](experiment_log.md) for actual launches.

[Public online resources, checked October 6](../DATASETS.md#public-online-resources--checked-october-6-2026): fixed-release THINGS-EEG documentation, exact PyTorch distributions and the pinned CLIP model configuration.

Historical October 6 reconciliation: [DATASETS readiness](../DATASETS.md#what-must-exist-before-model-comparison)
and GOAL now include the completed event join, 30/16 roles, human-feature and
private-routing metadata, and EP17 overlap. Authenticated assets, the exact
qualified runtime, reader isolation and scientific fitting remain unfinished;
see the [record](experiment_log.md).

Historical October 6 public-source update: the
[official release and asset-metadata note](paper_plan.md#october-6-public-source-update--exact-release-recipes-exist-runtime-untested)
links PyTorch's exact 2.4.1/0.19.1 distribution recipes and the existing pinned
CLIP configuration. Public release availability is confirmed; Sherlock
compatibility, checkpoint transfer and reader isolation remain unverified.
No install or new runtime test occurred; see the [log](experiment_log.md#2026-10-06-pacific--official-runtime-recipes-and-pinned-asset-metadata-recorded).

Historical October 6, 10:57 Pacific (PDT): the [then-unadopted route handoff](paper_plan.md#october-6-execution-route-recommendations--unadopted)
separates completed metadata/exposure work from the remaining authenticated
transfer, exact-runtime and reader-isolation dependencies. It corrects the
unsupported equation of namespace `ENOSPC` with Codex's sandbox limitation;
effective isolation through a suitably controlled runtime remains possible.
No compute-node transfer or alpha-build substitution is selected, and another
episode's runtime acceptance does not amend EP18. Recommendations are
unadopted; see the [log](experiment_log.md).

**Historical October5,15:04UTC:** the exact model-asset map and bounded
[DTN transfer](code/acquire_model_assets.py) are prepared and independently
reviewed. Agent SSH to the DTN was denied before transfer; the user has the
concrete authenticated-session command. Assets remain unacquired by EP18.
Seven project dependencies installed in the selected GroupHome environment,
including the diagnosed missing regex distribution. Actual imports then
showed that the advertised Torch2.4.1/Torchvision0.19.1 modules report
2.4.0a0/0.19.1a0. The module route is stopped pending supported matching
builds or provider clarification; no model or feature inference ran. See the
[evidence and support draft](experiment_log.md#2026-10-05-1504-utc--feature_preparation-regex-repaired-module-runtime-route-stopped).
The [runtime loader](code/qualify_model_runtime.py) remains prepared but
unexecuted. These tasks establish no pixel-worker boundary or EEG audit seal.

## Completed metadata and retained preparation records

This directory contains small, reviewable episode records and planned paper
artifacts. As of 2026-10-04, the
[30/16 participant roles and complete event join are frozen](runs/input_role_preparation/20261003-freeze_metadata01/frozen_inputs_and_roles.json),
using outcome-blind source work covering all 46 participants:
[structural report](runs/input_role_preparation/20261003-metadata01/structural_input_report.json)
and [explicit sample-origin repair](runs/input_role_preparation/20261003-sample_origin01/sample_origin_report.json).
The reports establish 62 common channels; the working join retains 1,133,808 events. Source
samples are one-based; original values remain alongside normalized recording
indices. The [clock diagnosis](runs/input_role_preparation/20261003-clock01/clock_segment_report.json)
locates the large sub-36/sub-38 clock discontinuities at recording-segment and
physical-sequence boundaries; all events remain present. The
[exact 30/16 role solution](runs/input_role_preparation/20261003-role_proposal01/role_proposal.json)
satisfies the three-tier balance objective, verified against an exhaustive
fixture and independently reviewed. Durable disjoint
[development](runs/input_role_preparation/20261003-freeze_metadata01/development_manifest.json)
and [audit](runs/input_role_preparation/20261003-freeze_metadata01/audit_manifest.json)
metadata views retain **739,440** and **394,368** events respectively.
The [source-only EP17/EP18 exposure record](config/ep17_ep18_exposure_record.json)
now records [exact source-inventory overlap](runs/input_role_preparation/20261003-freeze_metadata01/source_overlap.json):
**8,640 main-image identities / 720 concepts** and **100 repeated-image
identities** are shared. All 200 repeated concepts occur in the source
experimental concept inventory. The CNeuroMod CSV's 100 unmapped catch images
remain separate. This historical source inventory is distinct from the now
completed [EP17 qualified-universe comparison](runs/cross_episode_exposure/ep17_handoff001/exposure_report.json):
all **4,320 observed**, **3,840 common** and **3,634 repeat-qualified** EP17
images match EP18 main-image identities, across **720 shared concepts**.
None of those EP17 images matches an EP18 repeated-session image; **200
concepts** overlap that session. The coordinator supplied this immutable
metadata handoff after successful execution on job **46531931** in **4.210 s**.
Both designs and all role assignments remain frozen under the adopted
**correlated, design-exposed** decision. New EEG participants do not provide
independent stimulus-family confirmation; matching identities do not imply
identical pixels, identical features or clean pretraining exposure. See the
[October 4 closeout](experiment_log.md#2026-10-04--cross_episode_exposure-qualified-universe-and-claim-decision).
The required [development runtime separation](../inputs/README.md) remains a
distinct unverified prerequisite for EEG-driven candidate comparisons.
Actual attempts and access limits are in the
[execution log](experiment_log.md#2026-10-03--input_role_preparation-outcome-blind-source-qualification).
The freeze completed on job **46427768** in **49.539 s**; two focused identity
checks and [official ASTRA validation](astra/v0.0.14/validation_20261003_freeze_metadata.json)
passed. EEG/image values remain closed, the **audit seal is false**, and no EEG
result is claimed. Frozen roles and file manifests are not access controls.

The [sentinel-only isolated-runtime check](runs/input_role_preparation/20261003-runtime_sentinel01/runtime_feasibility.json)
failed before child launch with the platform's namespace `ENOSPC` response.
That runtime route is stopped; an existing supported isolated evaluator/runtime
is still needed before EEG-driven comparisons. No configuration changes,
weakened isolation or retry on another node are selected. Metadata and feature
registry preparation can continue independently; exact available property and
category tables and the remaining model-record prerequisites are listed in the
execution log.

The [human-feature crosswalk](runs/feature_preparation/20261003-human_metadata01/human_metadata_report.json)
also completed: all **1,854 concepts**, **12 property/size dimensions** and
**53 broad-category indicators** join without missing values. The 406 concepts
with no membership in those 53 categories remain included; they are not
missing records. Raw values are preserved without imputation, scaling or PCA.
The [complete prospective feature recipe](config/feature_registry.json) now
selects all required image, caption and label families, including the extra
independently trained encoder for the expanded panel. Its source references,
layers, dimensions, transforms and training-only reduction rules are explicit.
These are choices made before EEG access. The [pixel-only adapters](code/feature_extractors.py)
are implemented; [private routing/name preparation](runs/feature_preparation/20261003-mapping01/feature_metadata_report.json)
completed on job **46429138**, retaining **22,448 image identities** and
**1,854 name prompts**, with no provider-name fallbacks. The private source
crosswalk is for the trusted builder, never an image worker. Two focused
low-level formula checks passed in **1.543 s**, and the
[then-current ASTRA validation](astra/v0.0.14/validation_20261003_feature_recipe.json)
passed. Model assets and the exact runtime subsequently qualified, and all
concept-name features completed, as linked above. Effective image-worker
delivery remains unverified; no source-image pixels or EEG have been opened
by EP18, and no complete `frozen_feature_record` is yet emitted. The
[execution log](experiment_log.md#2026-10-03--feature_preparation-prospective-complete-panel)
distinguishes those engineering tasks from the existing access boundary.

- [Current conceptual figure, v2](ep18_conceptual_question-v2.png)
- [Exact image-edit prompt](ep18_conceptual_question-v2-prompt.md)

The current image distinguishes 10 Hz presentation from EEG sampling and
shows the shared image baseline; its caption makes explicit that all three
added-slot comparisons also share the linear-label baseline. Its alternative patterns are hypothetical;
the earlier `ep18_conceptual_question.png` is retained as historical material.

Future scientific outputs must respect the participant roles and access
boundary in [../DATASETS.md](../DATASETS.md).
