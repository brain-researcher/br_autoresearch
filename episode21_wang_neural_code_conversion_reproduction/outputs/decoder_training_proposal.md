# Decoder training: adopted full continuation and preserved earlier proposals

**Adopted October 6, 2026 Pacific:** the scientist accepted the proposed
500 allocated CPU-hour aggregate ceiling including first-layer job 46801996,
at most eight concurrent one-CPU/7-GiB workers, zero GPUs and 500 GB new durable
output. The earlier prefix and channel-0 fits remain preserved. A graceful
checkpoint stop closed the serial layer attempt with 664,832 new saved units,
bringing retained coverage to 716,160; the exact unfinished units can now be
distributed through the prepared full-stage worker. Actual job costs and
dispatch evidence belong in the [execution log](experiment_log.md).

The first-layer job's six required checks and the 46 full-pipeline checks
already passed. The newly required closed-layer storage-accounting checks
run before state preparation. The storage calculation releases the closed
layer's unused allowance only against verified named artifacts; it does not
raise the 500 GB ceiling. The concrete authority is
`decoder_training_full_authorization.json`; immutable assignments and the
finite-batch accounting journal remain under `decoder_training/full_continuation_001/`.
The earlier **unadopted/unactivated** labels below preserve proposal history
and are superseded only by this exact decoder-only adoption. No scoring,
converter, reconstruction, external-site, tau or OOD access is included.

## October 6 manual worker refill — engineering revision within the adopted caps

October 7 scheduling selection: preserve the prepared work_0011, work_0012 and
work_0013 refills, then use subsequent free slots for the first unassigned item
from each remaining participant: **work_0016 (sub-03, index 15), work_0031
(sub-05, index 30), work_0046 (sub-06, index 45), work_0061 (sub-07, index 60)**.
Each already-fixed assignment contains 1,003,520 units, conv1_1 channels 0–19.
This broadens active training coverage and supplies runtime evidence across
participants without adding fits, changing their order within an item, or
opening scoring. It is a submission-order choice only. Admit each item using
fresh terminal/queue/budget evidence; selection reserves no allocation itself.
Retain the eight-worker ceiling and original manifest indices. After these
items are assigned, resume remaining manifest items without duplication.

The explicit `submit --manual-refill` option can admit fresh work after an
individual worker completes, while other registered workers continue. The
older whole-batch wait remains the default. This changes dispatch order under
the existing resource grant; it does not change the authority, frozen manifest,
logical assignments, scientific method or access boundary.

Admission requires completed participant states, at least one successfully
closed worker, and the ordinary exact result checks for every terminal job.
Only registered nonterminal workers may defer their result checks. All
accepted nonterminal allocations, including pending jobs or jobs missing from
the queue, plus the selected new tasks must total at most eight or the smaller
authorized cap. One expanded queue observation must exactly reconcile their
IDs, names and live states; a known terminal job still visible in the queue
also refuses, even if renamed. Failed, partial, missing or uncertain closeout
stops further dispatch, with only the already reviewed pre-admission failure
exception preserved.

Full outstanding CPU reservations remain in the 500-hour budget. New work
must be never assigned and use fresh shard paths, original global indices,
one CPU / 7 GiB / eight hours, and unchanged storage limits. The durable intent
records `manual_worker_refill: true` before submission and rechecks the global
envelope during replay. There is no automatic retry or scheduler loop.

The first rollout selects **work_0009 alone**, only after a terminal result and
its actual parent charge free a slot. Its
`decoder_training/full_workers_refill_001.sbatch` runs the changed dispatch,
array-identity and repair/accounting checks before this worker's payload
access. Seven new refill regression tests and three earlier CPU-constraint
tests were pending at preparation; all 37 required checks subsequently passed
in job 46864089. Static coordinator and independent reviews also passed.
The first-check condition for subsequent manual refills is therefore met;
ordinary closed-result, queue, freshness and budget checks remain required.

The same allocation bundles only the pending named EP18/EP19/EP20 component
fixtures and four changed ASTRA documents, serially in a separate child
runtime capped at 1,020 seconds, including the subsequently added joint DAQ
and finite array-to-readout executor checks. EP18, EP19 and EP20 have 27, 4
and 28 then-pending component checks respectively. Their failures are logged per episode. The
unchanged empirical worker restores its modules and clips its budget to the
allocation EndTime minus 300 seconds, including all check overhead. The entire
parent is charged once to EP21; these checks grant no additional source access.
The preparation created no allocation or journal event. Execution updates:
the first refill's 37 required checks passed in job 46864089. The sibling
component checks also passed, but an empty-array expansion on older Bash
skipped EP21's optional ASTRA validation. The explicit empty-test guard is
repaired; global index 9 runs only that missed ASTRA check in a separate
60-second child with a 30-second validator limit before normal fitting.
That missed schema and semantic validation subsequently passed in job
46864962, with both child and wrapper exits zero. Existing passing tests are
not repeated. The [execution log](experiment_log.md)
owns admission, actual results and budget charges.

The next fresh global index 10 also hosts a separately bounded 240-second
public-source check correcting the diagnosed converter sum-versus-mean loss
error. This preserves the existing parent budget and historical lock, and
does not activate the converter stage. The
[diagnosis and selected check scope](experiment_log.md#2026-10-06-2327-pacific--pipeline_conformance-tagged-converter-loss-correction-prepared)
record the four pending regressions, numerical fixture and changed ASTRA check.
The same index also includes EP19's ready development-header qualification in
a separate 75-second child, reusing its passed reader checks. It reads only
the already assigned 155 text headers and fixed metadata reports. Both children
remain inside the parent's unchanged eight-hour limit and are charged once;
their failures do not stop independent decoder fitting. EP19 retains its own
report and data-role boundaries.

Fresh global index 11 / work_0012 hosts the prepared converter-library and
changed decoder-reader checks in a separate **180-second** child, including
setup and kill grace. Twenty new tests use small numerical/source-code fixtures;
the existing decoder-assembly tests cover its extended channel selector. They
open no scientific payload and do not fit an empirical converter. Both Torch
thread pools and the inherited BLAS/OpenMP settings are limited to one. Failure
is recorded without stopping independent decoder fitting. No additional job,
parent time, resource grant or converter data access is added. See the
[implementation record](experiment_log.md#2026-10-07-0035-pacific--interindividual_converter_training-components-implemented-checks-pending).

Fresh global index 12 / work_0013 now includes independently authorized EP18
public-label PCA/RBF preparation as a separate child of at most **1,800
seconds**, including setup and kill grace. This is production of EP18 public
label transforms, not another component test or an EP21 input. EP18 owns the
fresh outputs and records a nested 1,800-CPU-second allowance and planned
0.6-GB cache allocation; the full parent is still charged once to EP21, with
no additional physical job charge. Both its eight-hour reservation and the
500-hour aggregate ceiling are unchanged. The child uses only completed
public embeddings, human features and permitted structural metadata. Failure
stays isolated; decoder fitting clips to the original allocation deadline.
No EEG/image/event access, frozen authority change or automatic retry is added.
See the [selected scope](experiment_log.md#2026-10-07-0048-pacific--laion_decoder_training-next-refill-bundles-authorized-ep18-label-preparation).

Job 46687670 retained all 1,216 requested sub-01 units using all 4,712 images
after the complete 24,681-image
feature publication. Its [final metadata](decoder_training/retained_prefix_001/prefix_status.json)
records no prediction, score, tau or OOD access and no successor authority.
The coordinator's terminal accounting is **599 allocated CPU-seconds**, not
the 2-hour reservation. Eleven focused checks passed in 0.874 seconds and
the optional ASTRA validation passed inside that allocation; their cost is
included in those 599 seconds. The existing experiment log owns closeout.

The scientist's October 5, 23:51 UTC approval and
[bounded authorization](decoder_training_prefix_authorization.json) applied
only to that completed 1 CPU / 7 GiB / 2-hour prefix. The historical scope
below is retained; the new [full-continuation section](#full-decoder-continuation--unactivated)
is a costed implementation proposal, not a modification of that approval.
The scientist separately approved one new 1 CPU / 7 GiB / 24-hour allocation
for sub01 conv1_1 channel 0 spatial-F [64,50176); see the amendment below and
`decoder_training_continuation_authorization.json`. No replay, score or
further successor is authorized. Job 46726793 subsequently completed all
50,112 additional units in 940 allocated CPU-seconds; together the two jobs
retain 51,328 units. The next proposal finishes the remaining 63 channels of
sub01 conv1_1, rather than repeating the completed tranche.

## Historical optional execution-order amendment — never adopted

This option was prepared while the original jobs were pending. It was not
adopted: the completed prefix's `feature_admission.mode` is
`approved_complete_regular_export`. Both jobs are now terminal. The former
dependency-removal option is obsolete and supplies no continuation authority.
The following paragraphs record its original scope, not current pending work.

The original participant-specific decoder contract requires sub01's own4712
training images and their complete feature/neural inputs. It does not require
the other participants' feature values before fitting these1216 units. The
current authorization above nevertheless explicitly requires complete global
feature export. It remains operative until the scientist adopts this amendment.

The proposed change was to let the **then-pending job46687670** begin from the
already completed, preserved sub01 partition while the unchanged GPU export
continues independently. Bind only these three existing feature files:

- `outputs/steward_provisioning/regular_vgg19_sealed.building.46424955/sub-01/features.h5`
- `outputs/steward_provisioning/regular_vgg19_sealed.building.46424955/sub-01/finite_packed.h5`
- `outputs/steward_provisioning/regular_vgg19_sealed.building.46424955/sub-01/image_index.tsv`

The old producer's recorded sealed completion and exact4712-row index/header
admission replace the global feature-value completion prerequisite for this
prefix alone. The authenticated global24681-image **role metadata**, lock008,
public conformance007/008, source recipe/checkpoint/preprocessing, all19 shapes,
float32/finite-bit conventions and original neural joins remain required.
Only existing selected-unit values are consumed by the eventual approved run;
this preparation opens none of them.

Source review of `continue_regular_vgg19.py:300–329` confirms that final export
hardlinks the preserved original files into a **different** new publication
tree and renames that new tree. The original sub01 partition is neither moved
nor deleted. The prefix therefore keeps one fixed original path; it does not
create a partial canonical directory, copy payloads, introduce a directory
fallback, delay the GPU job, or alter the exporter.

The proposed amendment preserves the single allocation,1CPU/7GiB/2h,
1216 fixed units/all4712 images, all numerical operations and failure rules,
2GB new durable/10GB scratch limits, original charges and complete eventual
five-participant/19-layer scope. The remaining10545 feature forwards were still
required at that point and have since completed. It added no prediction, score, tau/OOD, converter, reconstruction,
retry, extension or successor authority. It advances retained fits and cost
measurements; it does not claim the full question is complete sooner.

Activation would require `allow_completed_sub01_before_global_export: true`,
`complete_feature_export_required: false`, and the exact matching retained
feature whitelist in the existing authorization, followed by
removing only the existing pending job's dependency. No replacement or duplicate
prefix job would be submitted. The optional worker also binds this route to
job46687670; no new CLI argument is needed by its existing batch snapshot.
Missing/false opt-in retains the currently approved global-completion route.
The original authorization and scheduler dependency were not changed for this
option. The prefix subsequently ran after complete publication, with its
runtime checks inside the one substantive allocation before empirical access.

Implementation and two independent static reviews are complete as of
2026-10-06 00:36 UTC. Four focused admission tests supplement the seven
existing numerical checks: explicit/default authority, refusal to fall back
from global completion, preservation of the exact source during publication,
and rejection of incomplete/corrupt layer or row metadata. All eleven checks
subsequently passed in the approved allocation. This did not adopt the optional
ordering amendment or authorize a successor.

## Completed prefix scope — original approved work

Fit and retain a fixed prefix of the **sub-01** native VGG19 decoder, using all
**4,712 subject-unique training images** and the prescribed 500 selected voxels
for every fitted output unit. The prefix contains **64 output units from each
of all 19 locked layers: 1,216 fitted units in total**. It measures real loading,
normalization, voxel ranking, ridge solving and publication costs; the fitted
weights remain part of the eventual complete decoder rather than being discarded
as a benchmark. Sub-01 is the first participant in the locked order, not an
outcome-selected participant.

The convolutional prefix is channel 0, spatial Fortran indices 0 through 63:
for a layer of shape `(C,H,W)`, use `(channel=0, h=s % H, w=s // H)` for
`s=0..63`. Preserve these tensor coordinates and the corresponding full-layer
Fortran index `0 + C*(h + H*w)`. FC prefixes are unit indices 0 through 63.
This is a fixed prefix within the source's channel-chunk organization, not the
first 64 entries of a differently flattened full layer. Process layers in the
existing `feature_protocol.LAYERS` order. No easier, faster, more variable or
better-fitting unit may replace a failed or undefined prefix unit.

All five participants, all 19 layers and every required output unit remain the
eventual scope. This prefix cannot stand in for a complete layer, participant,
decoder, conformance result or Finding A. No converter, prediction, fit-quality
score, held-out evaluation, reconstruction, AlexNet extraction, or external-site
work is included. Both tau roles and the entire OOD pool remain closed to this
worker. Future target and within-source uses may reference the same completed
participant-native fit because their locked training inputs/settings coincide;
the distinct roles must remain explicit in the eventual complete manifest.

## Frozen numerical behavior

Use `outputs/protocol_lock.yaml` (lock008), its resolved
`feature_decoder_contract`, `outputs/protocol_implementation_bindings_2026-09-30.yaml`,
and the existing source receipts. These already specify:

- Assign repetitions to the authenticated subject-unique role, then average
  that image's preserved repetitions before fitting. Keep every one of the
  4,712 images and its exact identity; no complete-case image subset or imputation.
- Normalize neural columns and each target feature using their own training-axis
  mean and sample standard deviation (`ddof=1`), as recorded from the tagged
  caller in `outputs/source_qualification/attempt-019/public_method_semantics_audit.json`.
  Preserve means/scales for later application without refitting them on tau.
- Apply only the adopted training-only zero-variance input-voxel validity filter,
  retaining native voxel IDs and their fixed order. Fewer than 500 eligible
  voxels makes the affected decoder unavailable, not a smaller-voxel fit.
- Rank each output's eligible voxels by absolute training Pearson correlation,
  with ascending stable native voxel ID resolving ties. Preserve the source's
  population-input-variance-below-`1e-8` correlation-zero rule at its specified
  point in the float32 fitting calculation; it is not an added eligibility rule.
- Fit FastL2LiR 0.9's **sum-SSE ridge with alpha 100**, float32 fit inputs,
  retained ones-column bias **penalized by the same alpha**, and Fortran target
  reshaping/restoration. Use the source `numpy.linalg.solve` convention on the
  selected 501-by-501 system, not a mean-SSE rescaling, unpenalized intercept,
  alternate regularizer, approximate solve, or dimensionality reduction.
- Reject nonfinite required normalized training inputs before BdPy's default
  infinity-to-zero substitution. Thus a constant feature with zero training
  scale has no usable normalized target: record its exact fixed identity and
  failure, stop the affected fit, and preserve completed earlier units. Do not
  use an epsilon, constant decoder, zero filling, removal, or substitute feature.
  The same stop rule applies to nonfinite required data/solves. A stopped prefix
  does not authorize a scientific repair or replacement run.

The package's default descending `argsort` tie behavior and its fallback to all
available inputs when fewer than 500 exist are explicitly superseded by the
already-adopted native-ID and shortage rules. Calling its stock feature-selection
branch unchanged would therefore be incorrect.

With more than 500 eligible voxels, the source forms
`X_aug.T @ X_aug + 100*I` and `Y.T @ X_aug`, then selects the
500 voxel rows/columns plus bias separately for each output. With exactly 500,
preserve the source's no-selection branch: all voxels in native order,
`X_aug.T @ Y`, and the same full 501-dimensional penalized solve. Below 500
remains unavailable under the adopted override. A single unchanged
full-voxel Gram may be reused across the prefix's layers; this is identical
training input/alpha, not shared feature-specific selection. The prepared
adapter preserves the source correlation and float32 operation order.

## Exact input interfaces and prepared implementation

No payload was opened to prepare this proposal. The following interfaces are
established by the existing producer code and metadata:

| Input | Exact existing producer path / definition |
| --- | --- |
| Neural values | `outputs/steward_provisioning/regular_neural_sealed/sub-01/subject_unique_responses.npy` |
| Neural row roles/IDs | Same directory, `subject_unique_trials.tsv` |
| Neural finite flags | Same directory, `subject_unique_finite_packed.npy`, little bit order |
| Native IDs | Same directory, `native_voxel_indices.npy` |
| True features | `outputs/steward_provisioning/regular_vgg19_sealed/sub-01/features.h5`, named layer datasets |
| Feature row IDs | Same directory, `image_index.tsv` |
| Feature finite flags | Same directory, `finite_packed.h5` |
| Completion evidence | `outputs/steward_provisioning/regular_neural_structural_report.json` and the export's `regular_vgg19_structural_report.json` |

Activation must expose only these subject-unique files through the regular
scoring/feature role interface. Do not mount/traverse the whole sealed tree as
if it were a training role: it also contains tau partitions. The original files
remain read-only. Reuse the verified role and mask records, and bind the worker's
actual selected row IDs and feature tensor coordinates to them; do not re-export
the provider data or rerun the mask/feature/conformance jobs.

The current supporting code was checked through explicit existing bindings:

1. `outputs/source_pipeline_binding/fastl2lir_0_9/plaintext/fastl2lir/fastl2lir.py`
   supplies the pinned estimator (`fit`, `corrmat`, penalized-bias solve).
2. `outputs/source_pipeline_binding/bdpy_0_25/plaintext/bdpy/ml/learning.py`
   supplies channel chunking, caller normalization and logical sparse-W/dense-b
   storage semantics. It eagerly handles whole target chunks; blindly invoking
   it is not a bounded prefix reader.
3. `outputs/pipeline_conformance/tools/execution_adapter.py:320`
   still raises `E_EMPIRICAL_CONTEXT_UNAVAILABLE` in `bind_empirical_context`.
   It is a payload-free planning scaffold, not an existing empirical fit runner.
4. `outputs/pipeline_conformance/tools/feature_protocol.py` supplies the exact
   layer names/shapes and completed lock context, but is feature-only.
5. The ASTRA `laion_decoder_training` node specifies a decoder manifest but has
   no executable recipe. Existing `run_public_runtime_conformance.py` contains
   only the completed public/analytic check, not a neural-training loader.

**Completed implementation, used only within the bounded prefix authorization:**
`outputs/decoder_training/tools/train_retained_prefix.py` now composes the bounded
subject-unique loader, role-assigned float32 image averaging, adopted estimator
overrides, retained-unit publication and actual allocation/deadline checks. It
reuses the exact pinned source `corrmat` body and preserves the full float32 Gram
and source solve branches. The old planning scaffold and qualified sources are
unchanged. The driver subsequently passed its checks and completed the approved
prefix. Its historical executed CLI was:

```text
python outputs/decoder_training/tools/train_retained_prefix.py \
  --authorization outputs/decoder_training_prefix_authorization.json \
  --participant sub-01 --units-per-layer 64 --workers 1 \
  --max-seconds 6900
```

The entrypoint fixes the layer list, channel/spatial ordering, alpha,
voxel count and input paths; none is a search parameter. Its bounded
authorization is present and was consumed by the completed job. Its output location is fixed
to `outputs/decoder_training/retained_prefix_001/`; it must refuse an existing
attempt rather than overwrite/replay it. Seven focused checks for authorization
scope, tensor identities, repetition joins, zero-variance rules, source solve
branches/ties, HDF5 slicing and write-once unit storage are prepared at
`outputs/decoder_training/tests/test_train_retained_prefix.py`. Together with
the four later admission checks, they passed before the empirical prefix.
Reuse these checks and conformance007/008; do not rerun the closed prefix or
the public/converter fixtures merely to prepare continuation.
The historical check command was:

```text
python -m unittest discover -s outputs/decoder_training/tests -p test_train_retained_prefix.py
```

The prepared root-owned wrapper is
`outputs/decoder_training/tools/retained_prefix001.sbatch`. It loads existing
Python 3.12.1, NumPy 1.26.3 and h5py 3.10.0 modules, sets all six thread caps before
imports, and runs one worker in the verified one-CPU allocation. The runner checks
the used numerical package versions against lock008 and records the actual
interpreter, prefix and loaded modules. It does not require `threadpoolctl`,
PyTorch, Pillow or a new environment for these NumPy/HDF5 decoder fits. The pinned
`corrmat` function is loaded from authenticated source without importing its
unneeded package-level dependencies.

## Original prefix allocation and shape-based resource envelope

The request below describes the completed allocation, not an available
reservation or permission for new work.

Request **one CPU-only Slurm job, 1 CPU, 7 GiB, 2 hours maximum**, one worker,
all BLAS/OpenMP thread limits one. Root chooses an actually verified partition;
none is guessed here. The coordinator has already observed the applicable
8,000-MiB-per-CPU cap: requesting `--mem=8G` inflated a related job to two CPUs.
Use `--mem=7G` to keep this request within one CPU, and verify the actual allocated
CPU count before work/accounting. Internal wall clock is at most **6,900 seconds**
and is also bounded by the actual remaining Slurm time minus 300 seconds. The
parent's reserve includes the focused tests (60 seconds plus 10 seconds kill
grace), module startup and closure within the same two-hour allocation. The
wrapper additionally caps the driver process at **6,930 seconds**: at most 30
seconds after its 6,900-second internal stop, then 10 seconds kill grace. This
smaller process grace can end closure before the parent's remaining reserve;
the flushed unit markers and initial pending manifest survive such a stop.
Neither cap extends the allocation. This is a finite spending cap,
**not measured timing or a promise that all 1,216 fits finish**. Stop at the first
deadline, memory/storage limit, I/O fault or required-data/numerical failure.
No automatic retry, extension, next participant, full-layer expansion or
successor submission is included.

The retained prefix has a **2,000,000,000-byte new durable cap** and a
**10,000,000,000-byte transient cap**. These are separate proposed decoder-stage
caps, not permission to consume or enlarge the feature-export ceilings.
Existing 1.7-TB feature and sealed neural payloads remain preserved inputs.
Use `$SCRATCH/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/decoder-prefix-$SLURM_JOB_ID/`
for transient matrices; required retained artifacts must reach the episode's
durable output directory before success. Do not copy the 280-GB participant
feature bank merely to run a prefix.

Sizing facts and bounds from the authenticated structure/source:

| Object | Shape / consequence |
| --- | --- |
| Sub-01 unique neural measurements | 18,843 by 12,295 float32 trial rows; about 0.927 GB, streamed/memmapped |
| Image-averaged neural matrix | 4,712 by at most 12,295; about 0.232 GB at the locked float32 fit boundary; allow bounded normalization/workspace copies without changing the source arithmetic |
| Shared penalized Gram | At most 12,296 squared; about 0.605 GB float32; avoid retaining redundant Gram copies |
| Each selected solve | 501 squared float32 plus one 501-vector, about 1 MB before solver workspace |
| All requested targets | 4,712 by 1,216 float32, about 22.9 MB; layer-wise reading can use less |
| Retained selected coefficients | 1,216 by 500 weights plus native IDs and biases; under 10 MB with float32 weights and int64 IDs, before metadata |
| Optional retained shared sufficient state | Averaged X, normalization/eligibility and the Gram remain below 1 GB in float32; count every retained byte, with no duplicate dense full-decoder W |

Seven GiB allows bounded input/normalization copies, the source Gram, correlation
blocks, solver workspace and library overhead. It does not permit an eager
full-layer target matrix. HDF5 features are **participant matrix banks, not
per-image model files**, with float32 datasets of shape `(4712,C,H,W)` or
`(4712,U)`. The continuation creates contiguous datasets for new partitions;
sub-01 was preserved from the original exporter. The completed prefix now
reports all 19 sub-01 datasets as contiguous and uncompressed. No separate
layout qualification job is needed. Reading each image's channel-0
plane and selecting the fixed Fortran spatial prefix avoids 64 scattered reads
per image. The largest plane is 224-by-224 (200,704 bytes); all images for that
one plane are about 0.946 GB. Across the 16 convolutional planes and complete
three small FC vectors, the logical selected-plane transfer is about **2.85 GB**.
This is a logical selected-plane I/O plan, not measured physical filesystem
bytes or throughput; original chunk/compression geometry may amplify reads.
Stage only the selected target columns and required neural data/shared state.
Record actual logical bytes read and elapsed I/O separately from solve time.

The full frozen hierarchy contains **14,861,288 outputs per participant**, so
even 500 stored float32 weights alone imply about 29.7 GB per participant before
indices, scales and storage overhead. Five full decoders therefore cannot be
budgeted from the small prefix's output size. The five native voxel counts are
12,295 / 12,897 / 10,891 / 12,023 / 12,057; the largest full float32 Gram is about
0.666 GB. Report these shape facts with measured prefix costs, without claiming
one participant's timing predicts all five exactly.

## Retained outputs, stopping and cost report

Retain training image IDs/repetition counts, ordered eligible native voxel IDs,
normalization means/scales, completed per-unit selected IDs/weights/bias, precise
layer/tensor/unit coordinates, and explicit pending/failed identities for all
1,216 requested units. Preserve float32 coefficients and logical sparse W;
do not renumber fragments as complete BdPy channel files. A later complete
channel assembler can restore the source W/b axes from these exact identities.
No converter can consume this partial manifest as a complete decoder.

Use 19 fresh per-layer HDF5 containers rather than 1,216 tiny model files. Each
unit slot is written once: fsync fitted arrays first, then fsync its completion
marker; completed slots cannot be changed. The initial requested-unit manifest
and each container retain pending/in-progress/completed/failed/interrupted
states even if a hard allocation stop prevents a final summary. The runner also
retains normalized X, moments, eligibility and the full Gram in one shared file.
Publish completed unit fragments without overwrite, and mark a layer complete
for this prefix only after all 64 named units are durable. On failure/timeout,
preserve completed fragments and a partial status; do not call it a scientific
negative result, replace a failing unit, repeat solved units or restart. Required
outputs must not exist solely in purgeable Scratch. Partial data are never
evidence of full-model eligibility.

Record actual runtime/package versions, allocation resources, elapsed I/O,
aggregation/normalization/Gram costs, per-layer target-loading/ranking/solve/write
times, peak memory, durable/transient bytes and counts of completed/failed units.
Report coefficients/normalizations as fitted objects, not as scientific results;
do not inspect predictive performance or tau to decide what to launch next.

Charge this one original decoder-prefix allocation **once** from actual allocated
CPU-seconds (ceiling 2 CPU-hours, zero GPU-hours), including preparation/checks
inside it. Do not add child-step time again, reset prior export consumption,
borrow a completed allocation's unused allowance, or reclassify this as free
qualification. Preserve every historical feature/CPU/GPU charge. A timeout spends
the consumed budget and ends this attempt. Full dispatch requires a separately
reviewable costed plan using the retained fits and observed resource measurements;
the prefix is not standing approval for more allocations.

## Historical prefix review/activation boundary

Source/static review, the scientist's bounded approval, in-allocation checks
and the retained prefix are now complete. That sequence granted no full
five-participant dispatch or further allocation. Tau/OOD, scoring, converters,
reconstruction and external sites remain outside decoder continuation;
A/C/E comparability margins remain unavailable as already locked.

## Full-decoder continuation — UNACTIVATED

### October 6 review proposal — 500 CPU-hour cap (UNADOPTED)

The October 6 document revision proposes **at most 500 additional allocated
CPU-hours actually consumed, zero GPU-hours and 500,000,000,000 new durable
bytes on Oak**, starting only after explicit adoption. It is an aggregate
cap, not 500 hours per job or a guarantee of full-decoder completion. Retain
all 51,328 completed units (~0.069% of 74,306,440) and their original costs;
exclude them from new work. Complete feature export is distinct from complete
decoder training. Generalizing the worker and finite-batch dispatch are
implemented; 46 focused checks of the full pipeline remain unexecuted.

The full work-assignment reservation scenario is **608–648 CPU-hours** and
cannot be submitted all at once under this proposal. The prepared replacement
dispatcher keeps the full scientific assignment and admits finite subsets using:

`actual CPU-hours charged + full CPU-hour reservations of accepted unfinished jobs + proposed new reservations <= 500`.

Use full reservation costs until an accepted job's terminal charge is known;
uncertain submission/accounting stops new dispatch. State preparation, checks,
failed attempts, fitting and assembly all count. If the separate 24-hour
first-layer proposal is also adopted, its actual cost and unfinished
reservation count inside this same envelope, and its retained identities are
removed from the subsequent manifest. Do not authorize 500 hours plus another
24-hour job by implication. The replacement helper now keeps one append-only
batch/accounting journal, with an immutable submitted authority and full
manifest, explicit operator reconciliation, and worker-side admission.
Implementation and static review do not substitute for the pending
in-allocation checks. At the cap,
preserve completed fits and report partial coverage; any extension is a new
decision. No deadline-driven automatic successor is authorized.

Keep the existing 1 CPU / 7 GiB worker shape and proposed maximum concurrency
of eight, with explicit per-job walltime and at most 2 GB scratch per worker.
Validate representative state preparation/fitting within any adopted budget
before wider dispatch. The measured ~386.91 parent-CPU-hour scenario is not a
guarantee for other participants or assembly. Runtime failure or a numerical
defect requires diagnosis, not a larger throughput assumption.

The existing **int64 native-voxel-ID** representation needs
**449,182,429,800 bytes** of full-decoder arrays before state/container
metadata. Changing 500 saved IDs per unit to int32 could theoretically save
**148,612,880,000 bytes** (~148.61 GB), yielding 300,569,549,800 array bytes.
Those figures are a full-format counterfactual. Preserving the 51,328 completed
units as int64 and converting only the remaining units would instead save
**148,510,224,000 bytes**, yielding **300,672,205,800 total array bytes**.
This is an optional format change, **not applied or required by this budget**.
Readers/writers currently require int64; int32 adoption would require range
checks, compatibility/version handling and compute-node verification while
preserving already completed files unchanged. The 500 GB proposal uses the
current representation and includes states, fragment metadata and assembly
metadata without a duplicate serialized model copy. Check usable capacity
before execution; this document revision makes no current quota claim.

This proposal changes no numerical/scientific target or closed data role.
Tau, OOD, prediction, scoring, converters and reconstruction remain outside
its scope. No full-stage authorization or manifest is created by this text;
the first-layer authorization remains false. A decoder-only budget would
still not authorize or complete the broader reproduction.

### Current completion target, measured costs and concrete next job

The unchanged target is **five participants × 14,861,288 units = 74,306,440
units**, all 19 layers for sub-01/sub-03/sub-05/sub-06/sub-07. Completed
prefix 46687670 contributes 1,216 units; completed continuation 46726793
adds 50,112 distinct conv1_1/channel-0 units. **51,328 are retained and
74,255,112 remain**. The full spatial channel-0 interval [0,50176) is now
complete for sub01 conv1_1; each of the other 18 layers retains its original
64-unit channel-0/FC prefix. Every later assignment must subtract these
identities, not reinterpret them as a contiguous full-layer Fortran prefix.

The proposed **one next job** finishes sub01 conv1_1, fitting **channels
1–63, each spatial-F [0,50176): 3,161,088 new units**. It uses all 4,712
training images, immutable sub01 normalized X/Gram and unchanged source
float32 arithmetic, 500 selected native voxels, stable native-ID ties and
alpha 100 with penalized bias. No original neural reread, normalization
refit, prior-unit replay, tau/OOD, prediction, score, converter or
reconstruction is included. Success would yield the first complete required
layer (3,211,264 units), **3,212,416 retained units overall**, and **71,094,024
remaining** across the five-participant target. It would not finish the
sub01 decoder (11,648,872 units still remaining) or the reproduction.

Requested prospective ceiling: **one 1 CPU / 7 GiB / 24-hour job**, at most
24 allocated CPU-hours, zero GPU-hours; **25,000,000,000 new durable bytes**
and **2,000,000,000 peak simultaneously retained transient bytes**. These
are new proposed allowances, not leftover authority from either consumed
job. The 63 new channels require **19,108,776,960 bytes** for the current
6045-byte/unit arrays before HDF5/container metadata. One sequential
945,717,248-byte channel bank is deleted before the next. Successful staging
writes **59,580,186,624 cumulative temporary bytes**, while peak transient
occupancy stays below 2 GB; logical feature-plus-finite reads are about
61,442,067,456 bytes. Cumulative traffic is not the storage-cap definition,
physical transfer measurement, or a claim about filesystem throughput.

Completed channel-0 metadata reports 905.581 seconds fitting/publication
for 50,112 units (452.681 setup/correlation/RHS, 352.098 selection/solve,
100.030 publication), 9.131 seconds channel loading and 5.602 seconds shared
state loading. Parent allocation time is 940 seconds. Extrapolation to the
new 63-channel target gives **15.87 CPU-hours for fitting/publication**, or
**16.47 hours using the observed parent-time-per-unit ratio**. These are
planning scenarios, not bounds: repeated state loading in that second ratio
is conservative for this one worker, whereas different channels, contention,
undefined feature scales and failures can increase cost or stop completion.
The finite 24-hour cap is not a complete-layer promise.

Worker RSS for 46726793 was 2,271,324 KiB; its coordinator-recorded Slurm
MaxRSS was 4,972,976 KiB. The earlier prefix Slurm MaxRSS **7,314,336 KiB**
remains the largest observed value and cannot be erased by the newer smaller
measurement. Seven GiB is a resource cap, not proved headroom. The unchanged
process RSS guard, explicit large-array estimate, one bank and 64-target
blocks remain in force; no automatic memory enlargement or retry follows.

### Prepared next implementation and review boundary

The new files are:

- `outputs/decoder_training/tools/decoder_coverage.py`: exact five-participant
  channel/FC identities, subtraction of all 51,328 retained units,
  deterministic whole-channel task packing and a gap/overlap-rejecting
  coverage verifier. This is planning logic, not a completed fit manifest.
- `outputs/decoder_training/tools/train_layer_continuation.py`: fixed sub01
  conv1_1 channels 1–63 entrypoint. It reuses the unchanged completed
  continuation's allocation/shared-state guards and prefix mathematical
  helpers. The grouped writer retains one fresh HDF5 container with 63
  channel groups, one owner, float32 coefficients and native int64 IDs.
- `outputs/decoder_training/tests/test_train_layer_continuation.py`: six
  focused tests for separate authority, complete-target/exclusion coverage,
  nonzero-channel finite-bit/F-index layout, grouped publication failure,
  exact helper dispatch/short tail/partial stop, and finalization errors.
  **Not yet executed.**
- `outputs/decoder_training/continue_layer_001.sbatch`: one 24-hour parent,
  60-second test ceiling, 86,100-second internal and 86,130-second outer
  ceiling with 10-second kill grace. Actual allocation end minus 300 seconds
  further limits work. No requeue or successor.

The coordinator's wrapper also bundles EP19's sixteen source-free classical
and non-neural forecast-backend checks in their separately scoped working directory, with a
120-second ceiling and 30-second ASTRA validation. They read no empirical
EP19 payload, record their own logs, and do not gate EP21 fitting. Optional
checks of the prepared full EP21 pipeline run its 46 fixture cases under one
120-second ceiling plus five-second kill grace. They use no empirical payload,
have separate per-job logs, do not activate the full stage, and do not gate this
first-layer worker. Optional EP21 ASTRA validation has a 30-second ceiling.
These small preludes, the six
required EP21 checks, module startup and fitting share the **same 24-hour
parent cap and once-only CPU charge**; no additional check allocation is
proposed. The pending approval request describes the EP19 prelude; the later
addition of the full-pipeline checks was disclosed before approval and does
not change the empirical scope, parent limits or storage allowance.

The distinct `outputs/decoder_training_layer_authorization.json` is prepared
with **approved=false** and must be
explicitly approved with exactly the new scope/resources before the worker
loads numerical libraries or empirical data. The old authorization cannot
activate it. Tests belong inside this substantive job, not another login-node
or check-only run. The coordinator chooses the partition from current site
discovery and submits once after approval. Prepared execution interface:

```text
python outputs/decoder_training/tools/train_layer_continuation.py \
  --authorization outputs/decoder_training_layer_authorization.json \
  --max-seconds 86100
```

The fresh output `outputs/decoder_training/layer_continuation_001/` contains
`requested_units.json`, `channels.h5`, and ordinary-closeout
`continuation_status.json`. Solved rows are fsynced before completion markers;
ordinary deadlines flush solved partial blocks. Marker assignment/fsync
failure preserves the exact uncertain channel/interval and makes the reported
remaining range non-authoritative for repair. Hard kills can leave only the
request and fragment; neither missing closeout nor a present file establishes
completion. No existing result is reopened for writing.

Only successful exact coverage and clean fragment fsync/close/accounting
produce `complete_layer_manifest.json`,
referencing the original prefix, completed channel-0 continuation and new
groups, with `complete_decoder=false`. No prediction or converter is enabled
by that layer manifest. The eventual consumer must understand ranked selected
native-ID coefficients, source axes and shared normalization; no dense W,
zero-filled missing units or post-hoc selection is introduced. Each ordinary
cleanup failure is recorded individually; an earlier fitting error is
preserved and final status is still attempted. Scratch cleanup failure alone
does not invalidate already fsynced fitted rows. A storage error that prevents
final status publication remains explicitly unresolved rather than becoming
an inferred completion.

### Full-stage route and costed scheduling option

**October 6 preparation:** the missing full-target components below are now
written and have passed independent static integration review. Their focused
checks have **not run**. No full-stage authorization, concrete work manifest,
batch dispatch or complete decoder has been produced. The first-layer job
above retains its separate, unchanged scope and pending approval.

| Prepared component under `outputs/decoder_training/tools/` | Purpose and execution boundary |
| --- | --- |
| `participant_training_state.py`, `prepare_decoder_states.py` | Bind each participant's own 4,712 images and all repetitions; build sub03/sub05/sub06/sub07 normalized X and Gram sequentially within one shared parent budget. Reuse sub01 unchanged. Eligibility comes from that participant's authorized training data. |
| `plan_full_decoder_work.py`, `decoder_execution_contract.py` | Validate supplied terminal metadata, subtract every exact retained interval, and assign the full target without overlap. An approved or live first-layer attempt cannot be omitted. Exact retention alone does not authorize another fit of a scientifically invalid target. |
| `train_decoder_work_item.py` | Execute one separately authorized, fresh work item, preserving both source voxel-selection branches, float32 kernels, convolution/FC identities, genuine tails and durable partial commits. Never build missing state implicitly or submit another job. |
| `decoder_assembly.py` | Validate all 74,306,440 physical rows, native IDs, training moments and C/F indices before publishing a complete manifest. Reuse coefficient containers and expose bounded sparse W/b blocks through a caller-owned compute context. No dense full W, duplicate weight archive or prediction is created. |
| `decoder_batch_accounting.py`, `submit_full_decoder_jobs.py` and the three `full_*_001.sbatch` wrappers in the parent directory | Admit one explicitly selected phase/batch against actual charges plus full outstanding reservations. Previous phases/batches require terminal accounting and complete results. Immutable submitted assignments and an append-only journal prevent automatic replay; uncertain responses stop dispatch. Separate full-stage authority and a concrete manifest remain required. |

The prepared checks comprise ten state/parent cases, eight generalized-worker
cases, four planning cases, twelve assembler cases and twelve mocked scheduler/accounting
cases. These small arithmetic,
coverage and interface fixtures await an authorized substantive allocation;
they are not new physiological simulations or empirical validation results.
They are now selected as a separately logged, non-gating prelude in the proposed
first-layer job above, without empirical access or full-stage activation.
The full stage still needs passing runtime checks, a concrete current work
manifest, an approved aggregate budget and coordinator-owned finite submissions
through the prepared finite wrappers. Full-data state and assembly
runtimes remain unmeasured.

The prepared dispatcher validates every assignment and the full durable and
scratch envelopes, then admits **one operator-selected phase/batch per call**.
States must finish first. In the default route, each selected fit array must
receive terminal accounting and complete task results before another batch;
the explicit manual refill option above permits overlap after certified
individual completion. Assembly requires every assigned fit. The default
current-user queue check rejects live EP21 jobs; manual refill instead exactly
reconciles the registered ongoing workers.
The immutable global order `work_0001` through `work_NNNN` maps to indices
`0` through `N-1`; a subset uses those same indices, never renumbers them.
Each array has at most eight concurrent tasks and the smaller authorized
concurrency if applicable. State preparation/assembly retain four-hour caps,
each fit eight hours, all one CPU / 7 GiB / zero GPU. The coordinator supplies
currently discovered eligible nonpreemptible partitions. This replaces the
earlier three-call all-at-once `afterok` chain.

`decoder_batch_accounting.py` and `submit_full_decoder_jobs.py` reserve each
array element independently before a single bounded `sbatch --parsable` call.
Intent is fsynced before submission and acceptance afterward in
`full_continuation_001/submission.jsonl`. A compute entrypoint reads the same
registered job/assignment before payload access; its shared file lock waits
once, for at most 45 seconds, behind the coordinator's 30-second scheduler
call and acceptance publication. There is no polling loop. A missing,
malformed, failed or timed-out reply leaves the whole intent reserved and
stops new dispatch. An operator can record the verified accepted ID or
confirmed nonacceptance; nonacceptance still requires separate review and
does not reopen the same work for automatic retry.

Terminal records explicitly bind the scheduler parent/task ID, numeric
`SLURM_JOB_ID`, allocated CPUs, elapsed seconds, state, exit code and the
coordinator's observed evidence. Never charge `.batch`/`.extern` steps or CPU
utilization as additional parents. One physical allocation cannot be charged
twice. Every failure is charged; a failed, partial or missing result stops
later batches. The first-layer job is included by exact job ID with an initial
86,400 CPU-second / 25 GB reservation, replaced by its observed terminal CPU
charge. Its closed retained coverage must match the full manifest. The journal
reserves at most 64 MiB durable storage; this and any included 25 GB layer
allowance count inside the 500 GB envelope. No automatic successor, retry,
extension, supervisor or agent session is introduced.

After explicit adoption, creation of the exact full authority/manifest, and
passing checks inside a substantive allocation, the operator interface is:

```text
python3.12 -B outputs/decoder_training/tools/submit_full_decoder_jobs.py --authorization outputs/decoder_training_full_authorization.json initialize
python3.12 -B outputs/decoder_training/tools/submit_full_decoder_jobs.py --authorization outputs/decoder_training_full_authorization.json status
python3.12 -B outputs/decoder_training/tools/submit_full_decoder_jobs.py --authorization outputs/decoder_training_full_authorization.json submit --manifest outputs/decoder_training/full_continuation_001/work_manifest.json --phase states --partitions <discovered-partitions>
python3.12 -B outputs/decoder_training/tools/submit_full_decoder_jobs.py --authorization outputs/decoder_training_full_authorization.json terminal --job-id <parent-or-arrayparent_index> --allocation-job-id <numeric-SLURM_JOB_ID> --allocated-cpus <0-or-1> --elapsed-seconds <observed-seconds> --state <terminal-state> --exit-code <exit:signal> --observed-utc <UTC-time> --evidence <exact-parent-accounting-observation>
python3.12 -B outputs/decoder_training/tools/submit_full_decoder_jobs.py --authorization outputs/decoder_training_full_authorization.json submit --manifest outputs/decoder_training/full_continuation_001/work_manifest.json --phase workers --work-items <explicit-unsubmitted-work-IDs> --partitions <discovered-partitions>
```

These are prospective interface examples, not commands executed here.
`submit --phase assembly` uses the same fixed manifest and partition arguments,
only after all work is complete. `resolve-submission --batch-id <id> --job-id
<accepted-parent> --evidence <observation>` resolves an uncertain acceptance;
`--not-accepted` replaces `--job-id` only for verified nonacceptance. The
authority must explicitly select
`cpu_budget_mode: actual_allocated_with_full_outstanding_reservations` and
`included_layer_job` (the exact included reservation or null). `initialize`
may track an included first-layer reservation before the full manifest is
available; full authority and manifest freeze together at first submission.

The approved authority and concrete manifest are immutable once submission
begins. Do not edit assignment order/spans while any job is pending or running.
A partial attempt needs separate reconciliation and an authorized continuation,
not deletion of its journal or mutation of its submitted manifest. Existing
physical assembly checks remain required before a complete decoder is claimed.

For an ordinary deadline/resource stop, a future full authorization must
identify the closed attempt and unchanged continuation cause. Nonfinite data,
an undefined required feature scale, a failed solve or an unresolved contract
defect cannot be made runnable merely by assigning its remaining indices to
another worker. No required target is dropped. Uncertain completion markers
require reconciliation before new assignments.

At the measured sub01 channel rate the **74,255,112 currently remaining
units** imply **372.74 CPU-hours fitting/publication**, or **386.91 CPU-hours
using parent-time/unit**. The latter is still not a full-stage bound and
does not predict other participants' eligibility, undefined targets,
initialization/assembly, load contention or convergence/runtime failures.
Do not count these scenarios in addition to actual job charges.

One concrete future scheduling option packs up to **1,003,520 new units**
per item in fixed participant/layer/channel order, never mixing participant
states. Whole-channel packing yields approximately **75–80 tasks** for the
current remainder; exact generated assignments/counts are checked by the
focused coverage test before any full dispatch. Per-item **1 CPU / 7 GiB /
8 hours**, with **at most 8 concurrent tasks** (8 CPUs, 56 GiB, 16 GB peak
transient), gives a **600–640 CPU-hour reservation scenario**. Typical
max-size items project about 5.23 hours at the measured parent rate. The
prepared planner separately proposes one sequential four-state parent and
one assembly parent, each capped at 1 CPU / 7 GiB / 4 hours, adding at most
8 CPU-hours: **608–648 CPU-hours in total** before repacking from later
completed work. These are spending ceilings, not runtime guarantees. Repack only the
remaining identities after any approved tranche and reduce reservations
accordingly; never submit both that tranche and overlapping full-stage tasks.

Each work item additionally reserves 4,096 metadata bytes per channel span
and 65,536 fixed bytes beyond its coefficient/container estimate. This avoids
rejecting late-layer tasks whose request and final status contain thousands
of channels. Aggregate proposed new storage includes these metadata reserves,
8 GB across the four new training states and 1 GB for assembly metadata, and
must remain within the separate 500 GB envelope. Assembly can scan in bounded
256-unit blocks, independently of the unchanged 64-unit fitting arithmetic;
it opens each physical fragment/status once and reads approximately 449 GB
of saved arrays. This is unmeasured I/O, not an instantaneous final step.

Current-format arrays across all five full decoders need
**449,182,429,800 bytes**, plus participant state/container metadata. A
**500 GB new durable envelope** remains a reasonable storage proposal,
including all future fragments but excluding a duplicate serialized model
copy. Existing features, neural data and every completed artifact remain
preserved. Full-stage CPU/storage/concurrency authority remains **unset**;
no complete-stage budget is inferred from this scheduling option or the
scientist's consumed single-job approvals.

### Historical full-plan and channel-0 proposal — superseded by the current plan above

The following earlier cost estimates and channel-0 proposal record the
pre-continuation state. Its 1,216-only coverage and unexecuted/absent-authority
statements are historical, not current instructions. Channel-0 execution is
complete and charged once; no clause below activates the new layer job.

**October 5, 23:08 Pacific amendment:** the scientist has now approved exactly
one next retained tranche below at **1 CPU / 7 GiB / at most 24 hours**.
Its separate authority is `decoder_training_continuation_authorization.json`.
This supersedes the historical two-hour/unapproved wording below for that
single sub01 conv1_1 channel-0 [64,50176) tranche only; the full five-participant
dispatch remains unactivated. The prepared wrapper/worker now use 86,400s
allocation, 86,100s internal and 86,130s outer limits plus 10s kill grace, with
the same actual-end-minus-300s closure reserve. The new 24-hour guard check
brings the focused suite to five tests, run within the substantive allocation.
Original prefix code/outputs/authority and all numerical/data/storage limits
remain unchanged. Execution facts belong in the current log.

### Historical pre-channel-0 completion target and retained work

The target remains **sub-01, sub-03, sub-05, sub-06 and sub-07, each with all
19 layers and 14,861,288 output units**: 74,306,440 units altogether.
The completed sub-01 prefix contributes 1,216 of them, leaving **74,305,224
unfitted units**. It is a channel-0 spatial-Fortran prefix in each convolutional
layer and indices 0–63 in each FC layer, not the first 64 full-layer Fortran
indices. Exclude those exact identities from any new work manifest and retain
their existing files unchanged. A smaller layer set, participant subset, voxel
count or output representation is not completion of this target.

Every participant uses its own complete 4,712-image subject-unique training
role, original image/repetition identities, native voxel order and the frozen
float32/500-voxel/alpha-100 rules above. Tau-train, tau-test and OOD remain
closed. No decoder prediction, scientific score, converter or reconstruction
is part of the proposed continuation. Complete decoder training would still
not complete the broader EP21 reproduction.

### Historical prefix measurements — retained evidence, not current remaining-work counts

The final `prefix_status.json` reports 578.443 seconds inside the worker and
2,308,432 KiB process peak RSS (about 2.20 GiB). Separately, the coordinator's
terminal Slurm record reports **MaxRSS 7,314,336 K**, just below the 7-GiB
allocation. These memory observations disagree and must both be retained;
the smaller process number does not demonstrate spare allocation memory.
Their difference has not been diagnosed and is not assigned a speculative
cause here. The coordinator measured 599 allocated CPU-seconds for the parent
job; those are different clocks, not additive costs.
Neural image averaging/normalization took **303.304 seconds**, and the shared
full Gram took **8.399 seconds**. Sub-01 retained 12,294 eligible voxels from
12,295 native IDs. Do not assume the other participants have the same
eligibility, initialization cost or runtime.

The following measurements are **seconds for 64 units per layer**, not the
full layer. Loading read whole image/channel planes to select that prefix.
“Setup” combines feature normalization, the source correlation operation and
RHS construction. “Select/solve” includes exact ranking, full-Gram gathering
and the selected 501-dimensional solve. Publication measures `complete()`;
the preceding per-unit `start()` fsync is outside that timer.

| Layer | Full units per participant | Prefix load | Prefix setup | Prefix select/solve | Prefix publication |
| --- | ---: | ---: | ---: | ---: | ---: |
| conv1_1 | 3,211,264 | 8.452 | 0.654 | 0.409 | 5.322 |
| conv1_2 | 3,211,264 | 6.338 | 0.634 | 0.445 | 7.015 |
| conv2_1 | 1,605,632 | 2.248 | 0.585 | 0.425 | 7.005 |
| conv2_2 | 1,605,632 | 0.949 | 0.635 | 0.444 | 3.498 |
| conv3_1 | 802,816 | 1.720 | 0.653 | 0.420 | 7.080 |
| conv3_2 | 802,816 | 0.871 | 0.611 | 0.426 | 8.814 |
| conv3_3 | 802,816 | 0.902 | 0.618 | 0.413 | 3.020 |
| conv3_4 | 802,816 | 0.898 | 0.686 | 0.427 | 10.014 |
| conv4_1 | 401,408 | 1.550 | 0.667 | 0.434 | 10.013 |
| conv4_2 | 401,408 | 0.761 | 0.684 | 0.416 | 13.288 |
| conv4_3 | 401,408 | 0.735 | 0.634 | 0.433 | 13.101 |
| conv4_4 | 401,408 | 0.762 | 0.654 | 0.423 | 6.957 |
| conv5_1 | 100,352 | 2.355 | 0.660 | 0.424 | 7.375 |
| conv5_2 | 100,352 | 0.562 | 0.655 | 0.418 | 6.031 |
| conv5_3 | 100,352 | 0.578 | 0.658 | 0.446 | 8.007 |
| conv5_4 | 100,352 | 0.776 | 0.642 | 0.434 | 6.455 |
| fc6 | 4,096 | 0.495 | 0.643 | 0.442 | 20.625 |
| fc7 | 4,096 | 0.387 | 0.637 | 0.428 | 6.646 |
| fc8 | 1,000 | 0.360 | 0.633 | 0.428 | 6.439 |

The 19-layer totals are **31.699 seconds loading, 12.244 setup, 8.134
selection/solving and 156.707 publication**. These and the 311.703-second
neural/Gram preparation do not exhaust worker time: admission, shared-state
publication, unit-start fsyncs, container setup and closure also occur. The
status reports 3,806,797,391 logical input bytes, 845,949,837 durable bytes
before final status, and no worker-written Scratch bytes. Logical bytes are
not a measured physical storage transfer or cache-independent throughput.

There is no justified full-stage upper bound from this one participant's
channel-0 prefix. As a transparent planning scenario only, multiplying each
layer's **setup plus select/solve** time by its full-unit count divided by 64
gives about **68.75 CPU-hours per participant, or 343.73 CPU-hours for five
participants at identical rates**. This excludes loading, participant setup,
publication, task startup, failures and assembly; it is neither a guaranteed
total nor an authorization. Repeating the old per-unit publication pattern
at those layer-specific rates would itself contribute roughly **2,171 hours**
across five participants, before untimed start fsyncs. That is evidence to
replace the storage transaction pattern, not a prediction that optimized
publication will take that long or a claim of a measured speedup. Do not
multiply the 599-second allocation by the ratio of full to prefix unit counts.

### Historical general continuation design — reusable state and source-faithful arithmetic

Reuse sub-01's existing `retained_prefix_001/shared_training_state.npz` and
`training_row_binding.json`: normalized X, full penalized Gram, all/eligible
native IDs, eligibility mask, neural means/stds, image IDs and repetition
counts are already retained. Load once per worker; do not recompute the
303-second preparation or Gram for every layer/channel. The other four
participants each need one corresponding state built from their own exact
role files under `regular_neural_sealed/<participant>/`; publish it once and
reuse it. No sub-01 state is shared across participants. The selected feature
inputs are only `regular_vgg19_sealed/<participant>/{features.h5,
finite_packed.h5,image_index.tsv}`. Existing global role metadata remains
readable; global value files and tau partitions do not become worker inputs.

Retain the source operation order: float32 repetition means in stored order,
image-equal training mean/sample-SD normalization, normalized fit cast, full
augmented Gram with penalized bias, variance guard, source `corrmat`, exact
native-ID ties, rank-ordered selected submatrix and `numpy.linalg.solve`.
Do not replace the full Gram with a selected-only recomputation, Cholesky
approximation, float64 accumulation, normalized-X dot-product shortcut or
an RHS reconstructed from correlations. The exactly-500 native-order branch
and below-500 failure stay explicit for every participant.

Use the already exercised **64-target numerical block** first, with real
shorter tail blocks where a channel/FC length is not divisible by 64; do not
pad with invented targets or recompute completed units to fill a block.
Caching X's source-centered transpose, sample SD and low-variance mask is
possible because they are participant-only quantities. Such a refactor must
preserve `dot(centeredX, centeredY.T)/(n-1)` followed by division by the SD
outer product, the float32 dtype and array layout. Keep the original
`normalizedY.T @ X_aug` RHS for the selection branch. Changing BLAS target
block width can change rounding and voxel ranks; it is not automatically
bitwise equivalent and no near-tie tolerance is introduced. The first
implementation can reuse the existing unmodified `prepare_layer_math` and
`solve_selected` functions instead of requiring this additional cache refactor.

### Historical general continuation design — channel I/O and complete assembly

All 19 observed sub-01 feature datasets are contiguous and uncompressed.
The continuation exporter declares the same layout for its new partitions;
ordinary header admission in each substantive worker confirms its selected
files. An early-layer channel has 50,176 spatial positions: its entire
4,712-image target bank is **945,717,248 bytes**. Read bounded image batches
for that one channel into a disk-backed temporary channel bank, preserving each image's
spatial-Fortran order, then feed 64-column C-contiguous blocks to the existing
math. This avoids rereading the entire plane for every 64 units. It also
reduces HDF5 call overhead compared with one call per image/unit, without
changing feature values. Read only that channel's corresponding packed
finite flags using full C-indices; never confuse them with Fortran unit IDs.
Release the channel bank before the next channel. No complete layer, all-five
target matrix or whole participant source copy is needed.

For eventual scheduling, pack **whole channels** into deterministic work items
of approximately at most 131,072 target units: 2 conv1 channels, 10 conv2,
41 conv3, 167 conv4 or all 512 conv5 channels per item (the final group can be
smaller). Coalesce the small FC tasks with compatible substantive work; do not
launch a tiny allocation per FC layer. This preliminary grouping is around
690 work items for five participants before prior-fragment exclusions and
first-tranche packing. It is not a dispatch manifest or a fixed job count.
At 2 hours per item, blindly reserving every item would represent about
1,380 CPU-hours of caps, substantially different from the component-rate
scenario above. Neither number establishes a currently available budget.
Finite deadlines preserve partial progress and report remaining identities;
they never silently shorten the complete decoder target.

Use **one exclusively owned HDF5 fragment container per work item**, not one
file per output or per 64-unit numerical block. Batch durable transactions
after a small completed numerical block, and flush any solved partial block
on an ordinary unitwise deadline: write its exact IDs/coefficients/normalizers,
flush/fsync, then publish and fsync the completion markers. Only
those committed rows are complete; an interrupted uncommitted block remains
explicitly incomplete. Other workers and later continuations cannot rewrite
completed rows. The existing prefix's 19 per-layer HDF5 containers and
per-unit markers remain unchanged.
Any later repair of an incomplete fragment needs an explicitly assigned
remaining-work manifest, not automatic replay or an uncontrolled retry loop.

Represent coverage compactly by participant, layer, channel and spatial-F
interval, with exact prefix exclusions. For `(C,H,W)`, the full Fortran output
index is `channel + C*(h + H*w)` and its feature-mask C-index is
`channel*H*W + h*W + w`. The assembler must verify disjoint complete coverage
of all 14,861,288 units per participant, including original prefix references;
all 500 native voxel IDs/rank-ordered coefficients, bias and target moments
remain attached to their exact unit. It can publish a complete logical W/b
manifest referencing immutable fragments, restoring the source axes without
filling a missing decoder with zeros. It must never mark a partial layer or
participant complete. No selection by fit quality is available to the assembler.

Proposed layout, **not yet created**:

```text
outputs/decoder_training/full_continuation_001/
  work_manifest.json                         # exact nonoverlapping identities/caps
  shared/sub-03/shared_training_state.npz     # likewise sub-05/06/07, one each
  shared/sub-03/training_row_binding.json     # sub-01 references retained_prefix_001
  shards/<participant>/<fixed-work-item>.h5   # grouped units, immutable committed rows
  reports/<fixed-work-item>.json             # cost/coverage/failure only, no scores
  complete_decoder_manifest.json             # only after all five complete
```

The source BdPy writer requests logical sparse W and separate b. Preserve
those mathematical axes; do not create a globally dense W merely to imitate
a serialization path. Explicit fragment-consumer/assembly support is still
needed before a decoder can be presented as usable by later stages. If a
later consumer requires materialized BdPy channel files, that serializer and
its additional bytes must be costed explicitly; they are not hidden free
copies or permission to launch a converter.

### Historical initial resource estimates — superseded by measured channel-0 costs above

The largest declared participant has 12,897 native voxels: normalized X is
about 0.243 GB and the full float32 Gram about 0.665 GB. An early-layer channel
requires 0.946 GB of target storage; a 64-image loading batch adds about
12.85 MB. Use a disk-backed channel bank and bounded target-block copies,
not an additional eager resident 0.946-GB array justified by process RSS alone.
The separately recorded Slurm peak was already close to 7 GiB. Thus **1 CPU /
7 GiB / at most 2 hours is an initial spending/resource cap to assess**, not
demonstrated memory headroom or a guarantee that the new worker fits. Temporary
backing storage is not claimed as free allocation memory. Preserve partial
output if a memory limit terminates the job; do not silently enlarge resources.
Keep six thread caps at one and one worker; test only the new loader/publication
logic inside substantive work. Full-layer eager loading, redundant large-array
copies and per-worker copies of every participant are excluded.

Across all five complete decoders, 500 float32 coefficients per unit require
**148.613 GB**, and 500 int64 native IDs require **297.226 GB**: together
**445.839 GB**, before bias, means/stds, unit coordinates, state and container
metadata. Shared participant state adds less than about 4.55 GB by the largest
declared shapes, plus small bindings. A planning envelope near **500 GB of
new durable storage** is plausible for compact fragments and a reference
manifest; it is **not approved capacity** and does not include a second full
serialized copy. A dense full W would be roughly 3.58 TB across participants
and is unnecessary. Keep existing features, neural inputs and prefix intact.

The training feature elements alone total about **1.4005 TB logical float32
input** across the five 4,712-image roles, plus about 43.8 GB packed finite
flags if each is traversed once. Repeated shared-state loading, metadata calls,
shard-boundary effects, storage contention and cache behavior add costs; these
are not physical transfer measurements. Choosing whole-channel work items
avoids systematic plane rereads across ordinary shard boundaries. A worker
needs only its current channel/state and bounded output workspace; actual
per-job and aggregate transient caps must be explicit in any new authorization.

No verified remaining global EP21 CPU/storage ceiling is available in the
current handoff. The policy contains bounded prior attempts, not a standing
full-training allowance. Therefore this proposal leaves **full-stage CPU,
concurrency and storage authority unset**. Preserve all charges for the
original feature attempts including 46424955, publication 46673714, continuation
46673727, earlier provisioning/conformance and the **599-second prefix**.
Do not turn unspent reservation time or freed staging capacity into authority.
Any approved new parent allocation is charged once by measured allocated CPU
runtime, including its preparation/checks, without adding child steps again.

### Historical channel-0 tranche proposal — completed as 46726793, no longer a next job

**Everything in this section describes the earlier preparation stage.**
The scientist later approved its 24-hour revision, all five checks passed,
and 46726793 completed the 50,112 units. Its authority exists and is consumed.
The old unapproved/two-hour/absent-file/unexecuted statements below are
preserved history only. The current next job is channels 1–63 under the
separate, still-unapproved layer authorization described above.

The prepared next-stage implementation proposes **one bounded sub-01 worker for the remaining
conv1_1 channel-0 units**, using all 4,712 images and the already retained
participant state. Its fixed traversal is spatial-F indices **64–50,175**:
**50,112 new units** in ascending order. It reuses the existing first 64 in
that channel; all other completed sub-01 prefix units remain retained for
later assembly. This is useful retained decoder work that measures the new
channel I/O/publication path, not a disposable benchmark or a substitute for
the full five-participant target. The other participants remain unfitted and
their cost uncertainty remains explicit.

A concrete **unapproved** next-allocation ceiling is **one 1 CPU / 7 GiB job,
at most 2 hours: 2 allocated CPU-hours maximum, zero GPU-hours**, with the
existing 6,900-second internal/6,930-second outer-process convention and
closure reserve. Proposed new caps are **1,000,000,000 durable bytes and
2,000,000,000 transient bytes**, separate from all earlier allowances.
The 50,112 new rows need about **300.672 MB** for float32 weights and int64
native IDs, plus small bias/moment/identity/state records; existing shared
state is referenced, not republished. The one-channel bank uses about 0.946 GB
of bounded temporary backing storage with small in-memory blocks. No whole source copy or
new participant state is required in this next allocation.

The measured conv1_1 setup-plus-select/solve rate alone would imply about
834 seconds for the full 50,176-unit channel on the measured configuration;
loading, publication, source-state admission and closure remain uncertain.
**No complete-channel promise is made.** Check the deadline between output
units, preserve every committed unit, flush a solved partial batch during
ordinary closure, and explicitly record the first uncompleted identity and
all later pending identities. Stop at the first resource or required-value/
numerical failure without substituting an easier unit. A hard external stop
may leave the final uncommitted batch incomplete; it can never be labelled
fitted. No automatic successor, retry or expanded participant scope follows.

The next-worker portions of items 1–3 and the partial-coverage interface in
item 4 below are now implemented and statically reviewed. Their four focused
checks remain unexecuted and are bundled before empirical access in the
proposed substantive allocation. Generalization to the other participants and the full assembler are
required before their dispatch/full-model use, not new prerequisites for
the first retained sub-01 continuation:

1. A continuation worker with an explicit new authority,
   exact subject-unique file whitelist, immutable existing-prefix exclusion,
   retained shared-state reuse/build and one fixed work-item assignment. The
   current prefix driver hardcodes sub-01/64 units and refuses its existing
   output; it must not simply be rerun with its consumed authorization.
2. The channel-bank reader and 64-unit numerical blocks, including genuine
   shorter tails, exact C-mask/F-output indexing and the unchanged source
   math. Reuse qualified kernels; add only focused new indexing/cache checks,
   not another broad conformance or recovery campaign.
3. A bounded grouped publisher and compact coverage manifest that preserve
   committed rows through timeout/failure, reject duplicate ownership and
   never claim incomplete work as fitted. Test interrupted publication and
   reuse of the 1,216 existing identities before new values in the substantive
   allocation; preserve all historical artifacts.
4. An assembly/consumer adapter that maps selected native IDs to the exact
   logical W/b axes, includes original fragments, checks complete coverage
   and emits a complete manifest only when the full five-participant target
   is actually met. The next allocation remains partial regardless of its exit.

The following fixed-scope files are **prepared and have passed independent
static review; they have not been run**:

- `outputs/decoder_training/tools/train_channel_continuation.py`
- `outputs/decoder_training/tests/test_train_channel_continuation.py`
- `outputs/decoder_training/continue_channel_001.sbatch`

The separate `outputs/decoder_training_continuation_authorization.json`
remains absent. The consumed prefix authorization cannot activate this worker.
The wrapper fixes one CPU, 7 GiB, two hours and no requeue; the coordinator
must choose a discovered eligible partition only after activation. Its
optional ASTRA validation has a 30-second limit plus 5-second kill grace and
does not gate empirical work. The four new adapter checks have a 60-second
limit plus 10-second kill grace. The worker retains the 6,930-second external
limit plus 10-second kill grace and a 6,900-second internal limit, further
shortened to retain 300 seconds before the actual allocation end. Module
startup and all checks count within the same two-hour parent allocation;
there is no extension, separate check allocation or automatic successor.

Its exact prepared interface is:

```text
python outputs/decoder_training/tools/train_channel_continuation.py \
  --authorization outputs/decoder_training_continuation_authorization.json \
  --max-seconds 86100
```

The participant, layer, channel, remaining index interval and output are fixed,
not new CLI search parameters. The later full assembler must reference this
continuation's committed rows as well as the original prefix, without replay.

The **four unexecuted checks** cover authority/path refusal before empirical
access; all 783 fixed numerical blocks and C-mask/F-output identities without
prefix replay; unchanged source-helper dispatch with float32 contiguous
targets and retention of a solved partial block; and publication ordering,
fresh-file exclusivity, immutable completed rows and failures at each fsync.
They reuse the qualified source math rather than rerunning conformance007/008.

The worker will exclusively create `outputs/decoder_training/continuation_001/`
for `requested_units.json`, one `conv1_1_channel0.h5` fragment and, on ordinary
closure, `continuation_status.json`. No existing output can be reopened or
overwritten. The status is `complete_continuation_tranche_only` or
`partial_continuation_stopped`; `complete_decoder` and `successor_authorized`
remain false in both cases. It records requested/committed/remaining spatial-F
intervals, requested/completed counts and the retained-prefix/shared-state
references. A failed completion-marker assignment or fsync additionally
records `uncertain_publication` with its exact interval and phase, sets
`completed_units_is_lower_bound: true`, and labels the remaining interval
`not_authoritative_for_repair_due_to_uncertain_marker_publication`. Potentially
published markers are preserved, not cleared or retried. Any repair needs a
separate authorized assignment. A hard external stop may leave only the
request and durable fragment; absence of a closing status is not completion.

The report separately times shared-state loading/bias augmentation and channel
loading, and retains cumulative `block_timings` for normalization/correlation/
RHS, selection/solve and publication, including ordinary partial-stop paths.
It also records total fitting/publication and elapsed time, worker peak RSS,
logical channel bytes, durable bytes before final status and node-local bytes
written. These are future measurements, not established costs. Terminal Slurm
memory and allocated CPU runtime remain separately recorded by the coordinator;
process RSS is not a guarantee of allocation headroom, and the parent is
charged once without adding child steps.

Possible later generalized interfaces, **not existing executable commands**:

```text
python outputs/decoder_training/tools/train_decoder_shard.py \
  --authorization outputs/decoder_training_continuation_authorization.json \
  --manifest outputs/decoder_training/full_continuation_001/work_manifest.json \
  --task-index N --max-seconds 6900
python outputs/decoder_training/tools/assemble_decoder_manifest.py \
  --manifest outputs/decoder_training/full_continuation_001/work_manifest.json
```

The next retained channel would supply new publication/I/O costs, actual
memory and exact remaining coverage. It would not establish the other four
participants' runtime or a full-stage upper bound. Their own state construction
and remaining fits must appear explicitly in the later complete-target plan.
Full dispatch still requires its own finite aggregate budget/manifest decision;
one successful continuation allocation cannot authorize it. Planning
here read only terminal timing/status metadata and bound code/contracts; it
opened no neural/feature payload, ran no test, and submitted no job.
