# Converter training and Finding A preparation

**October11,02:13Pacific collection implementation:** the bounded original
Finding A adapter now retains all855 decoded/unavailable slots, full point
vectors and JSON report/manifest sidecars. Root and independent source review
passed; three focused tests and changed ASTRA validation remain unrun.
The [implementation record](experiment_log.md#2026-10-11-0213-pacific--finding_a_feature_decoding--collection-implemented-and-source-reviewed)
lists remaining access/model/storage/runtime work. No empirical evaluation
has been activated. Historical descriptions of a missing collection below
are superseded only for this source implementation.

**October11,02:08Pacific brain-pair outcome:** job47358620 failed with
`BrainLossNotEvaluable` before publishing a model. The exact numerical reason
was not retained. See the [failure evidence and accounting](experiment_log.md#2026-10-11-0208-pacific--interindividual_converter_training--first-brain-pair-closed-without-a-model).
No retry is selected; the independent content measurement and collection
implementation continue. The activation entry below records history.

**October11,02:01Pacific brain-comparator activation:** the existing first
sub-03 to sub-01 full brain-loss fit is now selected and submitted as47358620
under its [separate finite grant](brain_pair_proposal.json). Reuse all13 passing
component checks. This opens only the named897-image tau-train fit and final
model; other pairs, production content training and scientific evaluation
remain unactivated. The [selection record](experiment_log.md#2026-10-11-0159-pacific--interindividual_converter_training--first-full-brain-pair-selected)
supersedes earlier unactivated brain-pair descriptions below. No result yet.

**October11,01:56Pacific:** the bounded stripe reader's three equivalence/
lifecycle checks and seven admission methods passed in repaired parent47357865.
Reuse seventeen passing interface/loss/measurement methods from47357234,
whose own test file cap prevented two fixture writes before empirical execution.
The replacement four-update measurement is admitted and claimed; no terminal
timing report or speedup claim yet. See the [current execution record](experiment_log.md#2026-10-11-0156-pacific--content_resource_measurement--repaired-checks-pass-measurement-claimed).
Production remains unactivated; earlier selected/unrun descriptions below
are historical.

**October11 selection:** implement the bounded feature-read change below and
measure the same four disposable updates under the
[separate finite grant](converter_io_measurement_authorization.json).
The [selection record](experiment_log.md#2026-10-11-0143-pacific--content_resource_measurement--bounded-feature-io-repair-selected)
supersedes the earlier unactivated engineering descriptions only. Production,
brain-pair fitting and scientific evaluation remain unactivated. Equivalence
checks and the new measurement have not yet run.

**October10,22:44Pacific measurement result:** the separately authorized
four-update job47329203 completed successfully in5781.607measured seconds.
The later updates took1355–1554seconds each, implying roughly16–18days for
1024updates of this measured pair before snapshot/write costs. This estimate
does not cover the other19pairs or guarantee production performance. Read
traffic and the process/batch memory discrepancy need source-level review
before a concrete production resource proposal. See the
[report](runs/content_resource_measurement/attempt002/report.json) and
[accounting/limits](experiment_log.md#2026-10-10-2244-pacific--content_resource_measurement--four-full-updates-completed).
No successor grant, production fit or scientific evaluation is activated.

### Source-only read-path diagnosis and proposed change — unactivated

The measured 22,782,859,264 logical feature bytes already count **both** the norm
and loss/adjoint passes: see [measurement arithmetic](converter_training/tools/content_pair_measurement.py),
lines 200–203, and [loss passes](converter_training/tools/content_loss.py),
lines 183–191 and 205–225. The much larger reported kernel counters include
other reads; neither `rchar` nor `read_bytes` establishes disk, network or
cache traffic separately. No filesystem or hardware fault is diagnosed.

The concrete amplification candidate is
[decoder_interface.py](converter_training/tools/decoder_interface.py),
lines 271–299: each convolutional block repeatedly requests
`data[:, channel, row:row+count, column]` across all 4,712 images. The admitted
feature array is contiguous, uncompressed C-order `[image,channel,height,width]`
(lines 326–338), while the returned unit order advances height first. Thus each
thin column request spans strided values through every image, and neighboring
columns revisit those same storage regions. Packed finite flags are also read
afresh for each block (lines 277–281). The provided counters imply roughly 64 KiB
per read call, consistent with buffered read amplification, but source inspection
alone does not identify the issuing library or quantify its share of traffic.

There is no image decoding or full feature-bank hashing in this update loop.
[feature_protocol.py](pipeline_conformance/tools/feature_protocol.py), lines 85–119,
hashes the structural image CSV during input preparation and explicitly reuses
the previously authenticated image-container identity. The pinned model loader
hashes at most 64 KiB of public source once
([schedule.py](converter_training/tools/schedule.py), lines 48–78).
Decoder coefficients and terminal records are reread on both passes
([decoder_assembly.py](decoder_training/tools/decoder_assembly.py), lines 524–554),
and each block is copied into frozen Torch tensors
([content_loss.py](converter_training/tools/content_loss.py), lines 83–120).
These are additional costs, but the inspected path does not support attributing
the multi-terabyte counter to repeated whole-payload hashing or decoding.

**Minimal proposed implementation:** change only the private feature-reading
path to hold one bounded C-order column stripe for the selected active channel,
then gather the same at most 256 Fortran-ordered units into the existing float32
output block. A 64 MiB stripe holds about 15 columns at the largest 224×224 layer;
one active-channel finite-byte buffer adds at most about 29.6 MB. Reuse each stripe
across its consecutive decoder blocks, including blocks crossing stripe edges,
and release both buffers at pass completion/cancellation. This keeps all 4,712 rows,
the fixed channel schedule, both passes, coefficient checks, exact C-bit/F-unit
mapping and per-block float32 subtraction/division unchanged. It creates no
durable feature copy and does not cache all layers. Existing RSS/deadline checks
remain applicable; the extra buffer headroom is not yet established.

Proposed validation is limited to focused contiguous-HDF5 fixtures comparing
old/new standardized blocks exactly, including stripe boundaries, short tails,
packed-bit boundaries, nonfinite refusal and cancellation, followed by the
existing loss/adjoint oracle with unchanged block/reduction order. Only after
separate selection and resource admission could a successor operational
measurement compare elapsed time, RSS and the same kernel counters. No speedup,
production feasibility, bitwise training equivalence, new access or successor
execution is claimed; **no source change or new measurement is activated here**.

**October10 update:** all33 new brain and Finding A component checks passed in
the EP07 preparation parent47334643; see the [evidence](runs/secondary_component_checks_job_47334643/).
Earlier unrun descriptions below are historical. This qualifies the tested
components, not full-scale memory/runtime or empirical results. Production
and evaluation remain unactivated; the content resource measurement continues.

Prepared 2026-10-07 from pinned source and existing protocol/code metadata only.
Production remains an implementation proposal. The separate four-update
resource measurement below was authorized October 10; see the
[adoption record](experiment_log.md#2026-10-10-1602-pacific--content_resource_measurement-authorized-and-selected).

The current [full-decoder grant](decoder_training_full_authorization.json)
remains decoder training/assembly only: 500 aggregate allocated CPU-hours,
the original eight-worker limit amended on October 7 to at most 42 concurrent
1-CPU/7-GiB workers within the unchanged shared caps, and 500 GB new durable output.
It grants no converter, tau, scoring, reconstruction, OOD or automatic-successor
access. A converter allocation requires its own finite resource and access
authorization. The separate bounded measurement below is now approved;
its completion or throughput is not promised. The [question](../GOAL.md), [dataset roles](../DATASETS.md) and
[search policy](../SEARCH_POLICY.yaml) remain controlling.

## Proposed full-size resource measurement — October 9

The [authorized measurement profile](converter_measurement_proposal.json) and
[batch script](converter_training/content_measurement.sbatch) describe one
disposable four-update measurement for **sub-03 → sub-01**, the largest canonical
parameter count. It uses all **4,712 source-unique images, 19 layers, 12,897 source
and 12,294 target eligible voxels**, CPU float32, the existing explicit sparse
backend, 256-unit blocks and 512-row arithmetic chunks. Production remains all
20 directed pairs with 1,024 full-batch updates each; this disposable measurement
does not create a resumable production prefix or fitted-model result.

The selected replacement grant is **three CPUs, 16 GiB, no GPU, at most two hours**
(at most six allocated CPU-hours), 16 MiB operational output and 256 MiB scratch.
The source-derived 4.604-GiB subtotal excludes possible allocator/library
temporaries and is not peak memory. The October 10 submission established that
bigmem requires at least256GB and rejected the original one-CPU/16-GiB request
before creating a job. Refreshed `sh_part` discovery gives russpold8GB per CPU:
the first accepted job actually received three CPUs and the launcher's exact
two-CPU check refused before imports or data access. Request three CPUs explicitly
for attempt002 while keeping every numerical kernel at one thread. Preserve
the failed attempt and its18allocatedCPU-seconds. This resource adjustment falls within the scientist's explicit approval
to increase compute. No hardware feature is assumed. This job is separate from the completed decoder grant;
its remaining allowance cannot authorize converters.

The [new input binder](converter_training/tools/empirical_content_inputs.py)
consumes the existing corrected summed-loss/gradient evidence before payload
access, all five frozen normalization/axis/ID records for canonical twenty-pair
initialization, only sub-03 normalized source responses and source-unique
features, and the fixed sub-01 target decoder. It does not load other source
response matrices or target responses to source images. Exact source/target
image disjointness remains required. Tau, OOD and evaluation stay closed.

The new measurement wrapper reuses the unchanged production driver and raises
a private stop immediately after the fourth Adam/LR update, before any final
snapshot or publisher. The separate resolver admits only binding and this
measurement; full training/publication is refused. A separate budget implements
the existing reader interface without enlarging or misrepresenting the old
7-GiB decoder WorkBudget. The CLI refuses the profile while `authorized=false`.
The one fixed attempt is claimed exclusively; failure/timeout leaves it closed
and incomplete, without automatic retry.

Only operational preparation/step time, process CPU, peak RSS, kernel I/O
counters, actual versions and completion flags are reported. First-update time
includes selected-pair setup. Four completed updates request22,782,859,264
logical feature bytes, not measured physical storage bytes; partial-step reads
and final snapshot/write cost are unmeasured. No loss, model, prediction or
scientific score is reported. Four steps cannot establish whole-training
numerical behavior, all-channel/cache costs or a guaranteed20-pair runtime.

New binder/measurement/admission checks run first within120seconds plus5seconds
grace. The empirical process has a6,300-second internal deadline and a6,320-second
external cap plus10seconds grace. The remaining sequential companion envelope
is630seconds. October9's updated tail selects only new EP06 component-scoring/nomination,
EP17 endpoint-evaluator/primary-audit adjudicator/nested prediction composition
and EP19 final-refit/offset checks, plus changed ASTRA validation. Twenty EP17
methods share235+5seconds, and twelve EP06 methods share60+5seconds. The full-
roster EP17 fixture instruments numerical kernels; its planned memory envelope
is up to7GiB and runtime is unmeasured. Total configured tail is625seconds
within the unchanged630-second allowance; no extra allocation is added.
Each companion needs sufficient actual remaining job time.
The earlier EP15 reader, EP19 selector and EP17 dependency/adapter checks have
since passed and are removed from this measurement batch. No package install
or repeated passing suite remains. New composition checks remain unrun.
No standalone tiny test job, payload copy, training continuation or scoring
branch is selected. Only the separately authorized measurement is selected;
production remains **unactivated**.

## Prepared implementation and observed checks — October 7, 03:20 Pacific

### October 9: orchestration and recovery-aware reader implementation

The unactivated [content-pair driver](converter_training/tools/content_pair_driver.py)
now prepares all five manifest-bound native axes and all twenty canonical
initializations, then joins supplied source data to the existing full training
loop and exclusive final publisher. There is no empirical CLI, automatic source
response loading, new stage admission or retry. The complete decoder collection
still supplies the only target axes and its fixed training moments.

The converter reader forwards the exact accepted I/O recovery review while
retaining spatial ordering. Each content provider now takes a bounded private
review snapshot before admitting the complete catalog and recovery evidence
once. Exact review/allocation checks run at pass entry and exhausted-pass exit,
after budget callbacks. Between-pass mutation refuses before reads; mid-pass
mutation cannot alter the private authority and refuses before pass completion
and optimizer advancement. Ordinary per-block checks remain live, and standalone
public readers still perform fresh admission. No global or numerical cache was
introduced. This removes the provider's formerly projected 38,912 full recovery
journal/evidence replays per pair, without claiming measured performance.

Independent source review is complete; **all ten reader and three orchestration
checks passed** in3.714seconds inside the optional tail of parent47101165.
ASTRA schema and semantic validation also passed. The actual test runtime
reports `torch=2.4.0a0`, despite the selected module's2.4.1 label. The completed
[target decoder collection](decoder_training/full_continuation_001/assembled_decoder_manifest.json)
now covers all five participants and19 layers. This satisfies the assembly
prerequisite without activating converter access. Unaffected component evidence
remains valid. See the [closed execution and accounting](experiment_log.md#2026-10-09-0837-pacific--laion_decoder_training-full-decoder-collection-assembled-and-parent-closed).

Full-scale runtime and memory remain unmeasured. The subsequently prepared
binder and measurement caller are described above; their new checks and finite
resource/access grant remain pending. No execution authority changed.

### Earlier component evidence

The implementation below is written, independently reviewed, and passed its
**20 focused component checks** in parent **46881096** (3.860 seconds).
The same bounded child passed all 12 decoder-assembly checks (2.547 seconds)
and the corrected source-loss/gradient fixture. The
[closed check status](converter_training/module_path_repair_job_46881096/status.txt)
and [loss correction report](pipeline_conformance/converter_loss_correction_001/report.json)
establish implementation evidence, not empirical converter execution or
full-scale resource qualification.

| Component | Prepared code |
| --- | --- |
| Selective source state, exact source feature rows, eligible/native target mapping and bounded channel reader | [decoder_interface.py](converter_training/tools/decoder_interface.py) |
| Complete active-block weights, summed error and response adjoint; explicit sparse or bounded dense backend | [content_loss.py](converter_training/tools/content_loss.py) |
| Shared permutation schedule, pinned model loader and canonical pair initialization | [schedule.py](converter_training/tools/schedule.py) |
| All 1,024 full-batch updates with the tagged Adam/LR order | [train_content.py](converter_training/tools/train_content.py) |

The new [brain-loss CV library](converter_training/tools/brain_loss.py) is
also implemented. Its separate **five focused checks passed in 0.112 seconds**
inside work_0007 parent **46891123**, under the selected 60-second limit plus
five-second kill grace. See the [closed transcript](decoder_training/remainder_admission_checks_job_46891123/brain_loss_checks.txt).
Passing component tests does not open tau-train.

The checks used small fixtures and public source, with no scientific payload
access or additional allocation. The current empirical adapter still refuses
execution. Stage admission, final artifact
publication, an authorized resource budget, and measured numerical/resource
behavior remain unfinished; these library components are not an executable
empirical stage. The brain-loss library below likewise has no empirical driver.

## Content-loss execution to implement

For each of the 20 directed nonself pairs, use all 4,712 source-unique,
role-assigned image averages and their VGG19 features. The frozen target decoder
uses the target's disjoint 4,712 unique images. Target neural responses to source
images are not inputs. Tau-test, OOD and external-site branches remain outside
this training proposal.

Pinned V1.0.0 `NCC_content_loss/NCC_train.py` establishes:

- Lines 128–142 and 208–221: `batchSize` sizes buffers; each iteration performs
  one forward/backward/Adam step over **all rows**. EP21 buffers must use 4,712,
  not the source default 6,000. Minibatching is not the selected algorithm.
- `prepare_indices`, lines 100–125: each convolutional layer visits a shuffled
  permutation of every channel, then reshuffles that permutation. Lines 281–290
  reverse the 19-layer order (`fc8` through `conv1_1`) and prepare one schedule
  shared across all pairs. All FC units participate every iteration.
- Lines 243–246 and 293: seeds are set once before the ordered pair loop.
  Independent jobs must bind initialization to a canonical pair order; simply
  resetting every pair to seed 42 changes the tagged initialization sequence.
- Lines 49–63: source-image features use the target decoder's per-unit mean and
  ddof-1 standard deviation. Converted responses already occupy the target
  decoder's standardized input space; no second input normalization belongs
  inside its linear prediction.

The two-hidden-layer architecture, half-input hidden width, InstanceNorm,
LeakyReLU(0.2), Adam schedule, 1,024 iterations and final-only checkpoint are
already selected. The corrected source loss is normalized inverse-squared
active-block norms multiplied by **summed squared error** per block. Its
caller-aware fixture and gradient check passed in parent 46881096; see the
[observed correction](pipeline_conformance/converter_loss_correction_001/report.json).
Historical lock008 and its reports must remain unchanged; future converter
admission must consume the explicit correction evidence.

## Native axes and the missing interface

`train_retained_prefix.normalize_neural` retains full-native mean/std vectors
but removes zero-variance columns from `normalized_X` using source-unique
training only. Therefore source input width is the eligible count, not the
stored mean-vector length. Hidden width follows that eligible input width.

The proposed target output axis is the target decoder's already fixed
`eligible_native_voxel_ids`, in canonical native order, consistent with the
adopted training-only validity filter. This requires an explicit adapter binding:
`decoder_assembly.read_sparse_block` returns `native_columns` into the **full**
native axis, whereas an eligible-only target output requires lookup through
`eligible_native_voxel_ids`. A full-native output would contain unused
coordinates with undefined zero training scale. The adapter should map each
selected native ID to its eligible-axis position and reject any missing ID;
it must not perform another data-driven voxel selection. Do not infer the target width
from the retained full-width mean vector or silently use full-axis column
numbers on an eligible-only tensor. Preserve native IDs, training-derived
eligibility, and the reader's explicit C/Fortran feature coordinates.

The existing `execution_adapter.bind_empirical_context` still rejects real
execution. The prepared reader selectively loads normalized source state
without its Gram, binds source features by exact image identity/order, maps
the target eligible axis and supplies one bounded scheduled-channel block at
a time. The existing sparse iterator now has optional channel selection with
its default behavior preserved; the new converter reader additionally orders
physical fragments by spatial position. Its focused fixture checks passed;
actual converter-scale I/O and memory remain unmeasured.
The prepared orchestration now joins the training and final-publication pieces.
The final empirical caller must still admit stage access and resources and
consume the correction evidence. No complete admitted consumer exists yet.

## Memory and a possible equivalent streaming implementation

One active map from each convolutional layer plus all FC units contains 151,096
units. At 4,712 rows, one complete float32 target or prediction bank alone is
2,847,857,408 bytes. Source responses, model/gradient/Adam state, activations,
decoder coefficients and autograd intermediates are additional. The literal
source retains all active target buffers and the loss graph until backward;
fitting it into an existing 7-GiB worker is not established.

The prepared streaming adjoint remains unactivated: it first establishes
complete active-block norms and their common normalization, then streams
bounded feature/weight blocks, accumulates the full converted-response gradient
and performs one converter backward/Adam step. It requires forward/gradient
equivalence checks before use. Chunked float32 reduction order differs from
the literal complete source tensors; bitwise equivalence is not claimed.
The current two-pass reader also rereads targets, so full-scale I/O cost is
unqualified. It preserves sample support and optimizer-step count and creates
no dense duplicate decoder archive. Resource sizing and a finite execution
budget remain unresolved.

### October 9: resource arithmetic from closed metadata — no allocation selected

The existing completion records expose the eligible widths without opening
normalization arrays, responses, features or decoder coefficients:

| Participant | Eligible width | Closed metadata evidence |
| --- | ---: | --- |
| sub-01 | 12,294 | [Prefix status](decoder_training/retained_prefix_001/prefix_status.json), line 14672 |
| sub-03 | 12,897 | [State status](decoder_training/full_continuation_001/shared/sub-03/state_status.json), line 28 |
| sub-05 | 10,891 | [State status](decoder_training/full_continuation_001/shared/sub-05/state_status.json), line 28 |
| sub-06 | 12,023 | [State status](decoder_training/full_continuation_001/shared/sub-06/state_status.json), line 28 |
| sub-07 | 12,057 | [State status](decoder_training/full_continuation_001/shared/sub-07/state_status.json), line 28 |

For source width `S`, target width `T`, hidden width `H = S // 2` and
`N = 4712`, the six learned tensors in
[the pinned model shape binding](converter_training/tools/final_content_artifact.py)
(lines 32–36) give `P = S*H + H*H + H*T + 2*H + T` parameters. The
[full-batch Adam loop](converter_training/tools/train_content.py) and
[driver copies](converter_training/tools/content_pair_driver.py) imply the
following logical payloads; these are **not measured peak RSS, GPU memory or
safe allocation limits**:

| Component | Bytes, excluding runtime overhead |
| --- | --- |
| Float32 model, gradients and two Adam moments | `16*P`, plus small step counters |
| Original NumPy source and copied Torch source | `8*N*S` combined; placement depends on the explicit device |
| Converted responses and full response adjoint | `8*N*T` |
| One hidden activation | `4*N*H`; six logical linear/normalization/activation outputs sum to `24*N*H`, with actual liveness and workspace unmeasured |
| Source-feature block of `B` units | `4*N*B` per NumPy/Torch copy |
| Decoder block | Sparse COO nominal `20*B*500`, or dense `4*T*B`; provider arrays, sorting/coalescing and temporary copies are additional |
| Row-adjoint product for `R` rows | `4*R*T`, plus predictions, residuals and overlapping temporaries |
| Final CPU model snapshot | Additional `4*P`; copied axes, RNG state and metadata are additional |

Block formulas follow [content_loss.py](converter_training/tools/content_loss.py),
lines 83–132 and 201–223. They do not imply that every temporary has been counted.
The current interface permits `1 <= B <= 256` and `1 <= R <= 4712`; no block
size, backend, device or resource profile is selected here.

The largest parameter count among the fixed twenty pairs is **sub-03 → sub-01**:
`H = 6448`, `P = 204033462`, and one float32 model occupies **816,133,848 bytes**.
The first three table rows sum to **4,214,135,328 bytes (3.925 GiB)**. Adding six
logical hidden outputs gives **4.604 GiB**, still not a bound on actual peak
memory: autograd, Adam workspace, finite-value masks, provider buffers, native
libraries and allocator retention remain additional. After the training loop
returns and the driver drops its source tensor, model gradients still exist;
model + gradients + snapshot + original source give `12*P + 4*N*S`, or
**2,691,484,200 bytes**, before metadata and retained allocator memory. Snapshot
copying and model deletion are explicit in the
[driver](converter_training/tools/content_pair_driver.py), lines 187–191, and
[artifact helper](converter_training/tools/final_content_artifact.py), lines
176–222. Across all twenty models, raw float32 parameter tensors sum to
**14,499,833,216 bytes**, excluding artifact metadata. GB uses decimal bytes;
GiB uses `2**30` bytes.

The unchanged [nineteen-layer schedule](converter_training/tools/schedule.py)
selects 151,096 feature units per iteration. With all 4,712 rows, one logical
float32 pass requests **2,847,857,408 bytes**. The two full passes in
[content_loss.py](converter_training/tools/content_loss.py), lines 183–208,
therefore request **5,695,714,816 bytes per iteration**, **5,832,411,971,584 bytes
per 1,024-step pair**, and **116,648,239,431,680 bytes across all twenty pairs**.
These are requested feature-value bytes, **not measured disk traffic**:
strided HDF5 access, cache behavior, finite flags, coefficients and metadata
are separate. Reusing fixed, target-standardized feature norms could avoid
the repeated norm-pass reads in a future equivalent implementation. That is
only a potential I/O optimization; it is neither implemented nor selected by
this resource assessment.

A future full-size operational measurement would need all 4,712 rows and all
nineteen scheduled active blocks, both passes, backward and Adam, including
first-step optimizer allocation and subsequent steady-state behavior. It would
record actual peak memory, read/arithmetic time and snapshot-copy/write cost.
**That measurement remains UNSELECTED.** Before any run, its relation to the
fixed final-only checkpoint and no-restart rule must be resolved explicitly;
an interrupted production pair cannot silently become a resumable sizing run.
No full-pair walltime or throughput follows from these calculations. The
primary objective remains all twenty pairs and 1,024 full-batch updates each.
The [current policy](../SEARCH_POLICY.yaml), lines 28–46, still grants decoder
training/assembly only; separate finite converter access/resources and the
explicit loss-correction admission remain required. No budget, run, algorithm
change or payload access was adopted in this subsection.

## Final content-model artifact preparation

The unactivated [final artifact helper](converter_training/tools/final_content_artifact.py)
can snapshot a supplied completed fit without new fitting or empirical reads.
It preserves final CPU float32 weights, both frozen eligible native axes and
moments, the 4,712 source/target training IDs, the one seed42 shared channel
schedule, captured canonical-pair RNG states, and target decoder/state references.
The fixed source-feature directory must belong to the source participant.
The recorded seed describes the existing one-seed canonical initialization
sequence, not a new seed reset for each pair.

Publication uses an exclusive sibling `.partial`, fsync, and an atomic exclusive
final hard link. Failures preserve the partial or visible final according to
the point of failure; no overwrite, automatic retry or false-success return is
provided. Loading uses CPU `weights_only=True`. The caller must account for
additional CPU model-copy memory and bound output bytes/time. Four tiny fixtures
are selected inside a substantive decoder job; this prepares an implementation,
not a converter execution grant, authentic fit-completion record or result.
The admitted empirical caller and full-scale resource qualification remain open.

## Brain-loss comparator

The selected comparator has 897 paired, independently image-averaged tau-train
images, five common image folds (180/180/179/179/179 validation images), and
alphas 620, 640, 660, 740, 780, 800, 820, 1000, 1200, 1400: 1,000 CV fits plus
20 final refits across the 20 pairs. Source/target means and ddof-1 scales are
fit separately on each training fold. Aggregate validation MSE by image count;
an exact tie selects the smallest alpha. A zero required scale makes the pair
unavailable, not eligible for further voxel deletion.

Pinned FastL2LiR0.9 with `n_feat=0` uses a dense float32 primal solve with a
penalized intercept. An exact-dual implementation is an unqualified optimization
option, not an existing automatic switch. Full-native planning sizes are about
0.475–0.665 GB per Gram and 0.524–0.634 GB per final mapping (about 11.575 GB for
all 20 mappings), excluding fit workspace and temporary copies. No runtime is
inferred from these sizes.

The new array-only library binds all 897 authenticated image IDs and supplied
fixed native axes, aligns independently ordered participants, and constructs
the common folds from a local RandomState42 permutation of bytewise-sorted IDs
with permutation-position modulo five. Rows within each subset retain sorted
image order. Each fold computes its own source/target mean and ddof-1 standard
deviation, then reuses that fold's float32 Gram/RHS across the ten primal
solves. Pinned `ncc_test.py` appends default float64 ones when multiplying the
trained float32 mapping; the validation prediction preserves that promotion.
All-image moments and the selected final mapping are retained together.

The panel visits the canonical 20 directed pairs once, streams each completed
model to a caller callback and keeps compact CV summaries. Scientifically
unavailable pairs remain explicit; incomplete fit counts are unknown rather
than falsely reported as zero. Budget/cancellation, programming and publication
callback failures propagate. No smaller axis, favorable subset, alpha repair,
alternate dual solve or automatic retry is introduced. The five pending checks
include one actual 897-row CV fixture with only three source/two target columns
and a direct comparison with the pinned FastL2LiR no-selection source; this is
implementation evidence only and does not establish full-scale runtime.

The separately prepared [final brain-model helper](converter_training/tools/final_brain_artifact.py)
now snapshots a supplied completed `BrainLossFit` and publishes/loads one
bounded CPU model using the existing exclusive writer pattern. It binds the
caller-admitted pair, all897 tau-train image IDs and complete native axes;
preserves the float32 mapping including penalized bias, final source/target
means and scales, all10-by5 CV scores, image counts, selected alpha, common
fold membership and51-fit completion; and records the fixed source method.
It does not fabricate per-fold moment arrays absent from the fitting result.
These full brain-comparator axes/moments remain distinct from content-decoder
eligible axes and subject-unique training normalization. Failed partials and
existing finals are preserved without overwrite or retry.

Source review found no concrete defects. The [three publication checks](converter_training/tests/test_final_brain_artifact.py)
are **unrun** and remain outside the active resource-measurement job's selected
tail. They perform no fitting or empirical reads. This is an unactivated
library, not a model result, execution grant or authenticated tau reader.

The dormant [tau-train input adapter](converter_training/tools/empirical_brain_inputs.py)
now joins the existing authenticated regular-role metadata and protocol lock
to the completed neural export and independent native-mask identity. It checks
both participants' complete training-row metadata before either response read,
then loads only the named tau-train arrays and full native axes. Every stored
repetition contributes to its participant's own float32 image mean, retaining
stored order. It returns 897 aligned image averages, common folds and separate
repetition counts, with no decoder normalization or eligible-column filtering.
Read-only maps close on success and failure; the returned averages own their
memory. External operation admission and actual allocation/budget checks precede
I/O and remain the future caller's responsibility.

Independent source review found no concrete defects. The [four reader checks](converter_training/tests/test_empirical_brain_inputs.py)
cover differing per-person trial orders/counts, float32 averaging, role/native
binding, access refusal and cancellation/nonfinite handling. They are **unrun**,
along with the three final brain-artifact checks above, and remain outside
the active measurement job's unchanged tail. This code does not authorize or
perform tau access, normalization, fitting, publication or evaluation.

The dormant [single-pair driver](converter_training/tools/brain_pair_driver.py)
now connects those bound inputs to the unchanged fifty CV fits plus final
refit and existing final-only publisher. It requires a separate external
`train_and_publish_brain_pair` operation for the exact pair, destination and
byte limit in the original allocation, rechecking after fitting and immediately
before writing. Input-binding authority alone is insufficient. Missing output
parents and existing final/partial files refuse before fitting. Numerical,
deadline, incomplete-fit and publication failures propagate without retry or
success. This helper executes one specified pair, not a complete twenty-pair
panel, and creates no CLI, admission grant or scheduler operation.

Independent source review found no concrete defect. Its [three composition checks](converter_training/tests/test_brain_pair_driver.py)
mock only fitting and exercise the real final-model helper; they do not repeat
the already tested numerical CV. All ten new publisher/reader/driver methods
remain **unrun** and outside the current measurement tail. No real tau data
or fitted brain model has been produced. The separately inactive entrypoint
proposal below now supplies a concrete execution route; its checks and
full-scale resource qualification remain outstanding.
Brain fitting itself does not need
decoders, but the current combined ASTRA converter stage consumes full decoder
assembly; this proposal does not change that dependency or open tau-train.

### First full brain-pair job proposed — October 10, unactivated

The [inactive profile](brain_pair_proposal.json), [entrypoint](converter_training/tools/run_brain_pair.py)
and [batch](converter_training/brain_pair.sbatch) prepare one complete
sub-03 to sub-01 fit: all 897 tau-train images, 12,897 source and 12,295 target
native voxels, the ten-alpha/five-fold CV and selected final refit. A successful
job retains the final model and CV selection metadata. It does not retain the
fifty intermediate models, reduce the training dimensions, or claim completion
of the required twenty-pair panel. Tau-test/OOD, evaluation, reconstruction,
other pairs and automatic retries remain outside the proposal.

| Bound | Proposed value |
| --- | --- |
| Allocation | One russpold job, 1 CPU, 7 GiB, 24 hours, no GPU; all numerical threads 1 |
| Work deadline | 84,600 seconds internally; 84,620 seconds externally plus 10-second termination grace |
| Process RSS | 6.5 GiB |
| Final model | At most 768 MiB; exclusive final-only publication |
| Durable / node-local scratch | 2 GiB / 256 MiB |
| Prerequisite checks | Thirteen new publisher/reader/driver/admission methods, 120 seconds plus 5-second grace, before empirical reads |
| Budget | At most 24 allocated CPU-hours on later activation; separate from decoder and content-measurement accounts |

The float32 matrix alone is 634,323,640 bytes, about605MiB. Solver temporaries,
library allocation and actual51-fit runtime are unmeasured. If the job cannot
complete within these limits, it reports incomplete where possible and
preserves any failed partial or visible final for diagnosis; there is no
automatic resume, extension, alternate solver or smaller-axis fallback.

The entrypoint admits only input binding and the exact fit/publication operation,
with its own allocation, metadata-only operational report and one exclusive
attempt claim. Internal tau-train CV is part of fitting; no held-out evaluation
or scientific score is released. The batch validates the changed ASTRA program
within30seconds plus5seconds grace when actual remaining time permits.

This profile remains `authorized:false` with no approval source. It has not
reserved compute or created an attempt directory, and the active contracts
have not been amended. The new admission checks and all ten earlier brain
component checks are unrun. Code preparation does not open tau-train access.

### Finding A prediction bridge — prepared, unactivated

The [array/model-only bridge](converter_training/tools/finding_a_prediction.py)
now connects caller-admitted 224-image responses, final frozen content/brain
artifacts and decoder blocks to the three prediction paths. Within-source
preparation uses its own decoder moments and can be reused across outgoing
targets. Content preparation reconstructs the pinned final model and preserves
its float32 round trip from standardized output through physical target values
and back. It does not directly normalize already-standardized output as if it
were physical data. Brain prediction uses all native target columns, restores
physical values with tau-training moments and then applies the target decoder's
subject-unique normalization; target columns are joined by native identity.

The existing [implementation binding](protocol_implementation_bindings_2026-09-30.yaml)
and protocol lock specify float32 normalized decoder inputs followed by default
float64 FastL2LiR prediction. This remains the selected common EP21 interface.
Bounded source inspection found that the literal NCC brain helper instead
forwards float64 normalized inputs directly. That arithmetic deviation is
disclosed; literal NCC brain-call equivalence is not claimed and the existing
lock is unchanged.

Each bounded decoder projection restores coefficients in training-eligible
native-axis order, retains bias and feature moments and emits both F/C unit
indices. It returns named float64 standardized/restored intermediates and
`source_file_features`, the float32 representation saved by both tagged NCC
callers. Eventual serialization and source-style metrics must respect that
precision boundary. No feature file is written here. All channels are available
to the caller; the sampled training-channel schedule does not select evaluation
units, and one block makes no completeness claim for a layer or pair panel.

Root and independent source review found no concrete defects. The separately
authored [five focused checks](converter_training/tests/test_finding_a_prediction.py)
remain **unrun**, outside the active measurement tail. They cover identities,
normalization and precision-sensitive references, F/C order and refusal of
mismatched, overlapping or nonfinite inputs. The caller still owns data/role
admission, complete decoder coverage, qualified source factory and resources.
Actual tau-test access and scientific evaluation remain closed. Complete
collection assembly remains unimplemented; the metric and primary-inference
components prepared below do not open evaluation. Content production
resource selection still awaits the current measurement.

### Finding A complete-layer correlations — prepared, unactivated

The [metric helper](converter_training/tools/finding_a_metrics.py) consumes
bounded blocks of caller-supplied float32 predictions, true features and fixed
decoder-training feature moments. It requires all224 image IDs and every C-order
unit of the actual registered layer exactly once, including every convolutional
channel. Each block contains at most256 units; full feature banks are not
assembled. It retains the complete profile vector and frozen moments, using
approximately55MiB for the largest layer's result plus bounded working storage.

Pinned BdPy0.25 `bdpy/evals/metrics.py` lines9–34 defines raw-value profile
Pearson correlations across images. Lines37–73 defines pattern correlations
after both arrays use the decoder's training feature moments. Here those
supplied values/moments and the subtraction/division remain float32 before
float64 centered covariance arithmetic. Within-source uses source moments;
converted methods use target moments. Stable centered block merging covers
the complete pattern axis without cancellation-prone raw second moments or
averaging block correlations. This preserves the selected metric and precision
boundaries, not bitwise equality with a full-array covariance reduction.

Undefined correlations keep their image/unit identities and counts; all
undefined means unavailable. There is no column deletion, near-zero epsilon,
Fisher transform or outcome-selected support intersection. Paired layer
contrasts require identical defined masks, and shared decoder owners must
also have identical training IDs/moments. All19 layer means receive equal
weight; missing/unavailable required layers leave that summary unavailable.
Subtracting two cross-layer summaries does not replace the per-layer paired
support checks. No comparability margin or inference is supplied here.

The independently authored [seven metric checks](converter_training/tests/test_finding_a_metrics.py)
use complete fabricated fc8 arrays and independent scalar/full-array references,
with identity, undefined-support, within-source and nineteen-layer reducer
cases. Two new cases check repeated-image resampling: original unique IDs stay
fixed while immutable draw indices select raw rows before both metrics. Exact
draw positions are required across paired and cross-layer summaries; profiles
are recomputed and degenerate draws remain unavailable. The checks remain
**unrun**, outside the live measurement tail. Root source
review, independent review and static API review found no concrete defect.
Changed ASTRA validation remains pending. No real tau-test response, feature or
scientific score was accessed, and no job or resource grant was added.

### Finding A primary panel and image inference — prepared, unactivated

The [panel and bootstrap module](converter_training/tools/finding_a_inference.py)
joins all19 layers and20 directed pairs. Each source/layer scores its within-
source baseline once, followed by each target's content and brain predictions.
Paired support is checked before reducing contrasts. Four targets receive
equal weight within source, five sources receive equal weight in the group,
and all19 layer values receive equal weight. Explicit missing inputs and
undefined contrasts retain their affected slots without changing these weights.
Ordinary callback, scoring and cancellation failures propagate without retry.

The point panel retains only the current baseline/content/brain full-layer
vectors, plus compact means, defined/undefined counts and missingness reasons.
A caller-owned optional sink receives point LayerScores for eventual full
image/unit artifact persistence; the caller owns that sink's storage and
memory. The five decoder-training ID tuple bindings are snapshotted before
callbacks and across the complete bootstrap operation.

Exactly1000 NumPyRandomState42 image draws are shared across every method,
source, target and layer. Each draw rebuilds the complete endpoints. Per-layer
and cross-layer dyad/source/group intervals use linear2.5/97.5 percentiles;
a missing point or any required missing draw makes that particular interval
unavailable. There is no available-draw percentile, replacement draw, new
permutation test, comparability margin or participant-population claim.

The direct implementation makes1001 full panel passes:855855 method/layer
scores when all inputs exist. Its retained numeric draw summaries use roughly
43.1MB plus a1.8MB draw schedule; raw-feature replay I/O and full-scale runtime
remain unmeasured. No feasible production allocation is inferred from this
memory arithmetic or source review. The secondary summaries described below
are now implemented; artifact assembly and empirical execution remain unfinished.

The [four orchestration tests](converter_training/tests/test_finding_a_inference.py)
retain the full roster and draw count while instrumenting numerical scoring;
the seven metric tests separately check repeated-row arithmetic. Root,
independent contract review and static API review found no concrete defects.
All eleven tests and changed ASTRA validation remain **unrun**, outside the
live measurement tail. No empirical data, score, resource grant or job was
opened by preparing this component.

### Secondary Finding A summaries prepared — October 10, unactivated

The [secondary module](converter_training/tools/finding_a_secondary.py) consumes
the original compact point panel. It averages four incoming sources per target,
retaining source-owned baselines and qualified contrasts. Overall summaries
require every layer. One local RandomState42 schedule supplies 1,000 descriptive
dyadic draws: sequential source/target proposals, rejected self pairs, duplicate
edges retained, twenty accepted rows and a 1,000-proposal cap per draw. Missing
original cells or capped draws make affected intervals unavailable without
replacement or reweighting. Primary image draws are rejected as inputs.

The algorithm follows the pinned source's choices; the adopted local seed does
not reproduce its unspecified historical seed. These secondary intervals do
not establish population inference or add a hypothesis test. Four independent
tests and changed ASTRA validation remain **unrun**; source/API review found
no concrete defects. No empirical access or launch was activated.

Source identity: `KamitaniLab/InterSiteNeuralCodeConversion` V1.0.0, commit
`edbff02edd6a08d3c6c829af45a957fd14bdc460`; preserved archive member prefix
`KamitaniLab-InterSiteNeuralCodeConversion-edbff02/`. Relevant additional source
is `NCC_content_loss/models.py`, `NCC_content_loss/utils.py` and
`NCC_brain_loss/ncc_train.py`; LAION adaptations are in
[the adopted decisions](protocol_decisions_2026-09-30.yaml).
