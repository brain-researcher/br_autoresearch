# Experiment log

## 2026-10-12 — baseline_models — prospective callback source prepared; full-scale backend remains unresolved

`code/production_continuous_designs.py` and its two focused methods in
`test_production_continuous_designs.py` are prepared but inactive and untested.
They bind explicit released EEG/image/label/event/candidate artifacts to the
existing tuning API and all15 designs. No numeric recipe, configuration,
full-trial controller, empirical launch or score is selected by this source.
Independent callback review remains open. Complete30-person candidate-bank
inputs, bounded cohort persistence, baseline selection/freeze and selected
design diagnostics are still required.

A scoped source assessment identifies a concrete current-backend limit:
`continuous_tuning.py` materializes the base and all10 added-slot matrices
together. At the earlier estimated500000 training rows, one8-function
categorical matrix is approximately55GiB and nine are approximately497GiB,
already exceeding the existing256GiB/job ceiling. FIR is larger. This is an
implementation bound conditional on the stated row count, not an empirical
runtime measurement or proof that an exact streamed algorithm is impossible.
The current `_expand` and dense capacity path have no connected GPU/streaming
production alternative; adding parallel jobs alone cannot admit those shapes.

Earlier source review found an exact direct-cuSOLVER float64 QR/SVD path
plausible but unimplemented, with actual workspace/convergence/full-slot cost
unknown. Ordinary Torch SVD copies do not fit the full74160-direction case.
An actual complete-slot cost execution would require a concrete exact backend
and resource admission first. Do not claim that the current source makes a
complete original trial runnable before the unchanged October13 21:30UTC
deadline. Active real-data preparation and the selected first-person false-bank
cost chain remain useful and unchanged; no simulation expansion is selected.

## 2026-10-12 16:55 Pacific — development_eeg_preprocessing — complete30-person empirical preparation observed

The production operational report finished at23:51:06.188809UTC with
`complete_development_EEG_preprocessing`: all30 released participants and2160
voltage intervals complete,646.077450seconds scientific preparation,
860045312bytes process peakRSS and9560266517durable bytes. No model scores or
audit access occurred. The batch exit record reports stage
`complete_development_EEG_preparation`,exit0,715scriptseconds, both EP18 cache
and independently scoped EP21 hardware checks exit0. Evidence:
`runs/eeg_preprocessing/development_20261012_001/operational.json` and
`runs/eeg_preprocessing/operations_20261012_001/initial_47572924/exit.json`.
The16:56Pacific scheduler observation closes47572924 COMPLETED0:0,
16:39:03–16:51:09Pacific,726seconds at one allocatedCPU. Charge exactly
0.201666666667allocatedCPUh once to EP18; the EP21 companion is not charged again.
The script's715seconds and scientific646.08seconds are nested durations,
not additional charges. All active image/false-bank
dependencies remain unchanged. This is complete real EEG preprocessing, not
an encoding-model result or admission of the unfinished complete fitting route.

## 2026-10-12 16:19 Pacific — false_partition_preparation — corrected queue binding verified

Coordinator squeue at23:19:37UTC confirms47571099 PENDING/Dependency with
13CPUs,96GiB and4h, waiting for the original assembly chain. All22 active
image-feature workers remain RUNNING, four fast tasks wait on their original
array concurrency limit, and assembler47551339 waits on dependencies. This is
the sole live first-participant cost request. It has not begun processing;
no candidate bank or EEG result is claimed.


## 2026-10-12 16:18 Pacific — false_partition_preparation — corrected dependent job submitted

The scheduler accepted the corrected request as47571099 at23:18:34UTC:
normal13CPU/96GiB/4h, afterok47551339. Submission is not execution or bank
completion. Retain the selected52CPUh nominal reservation and64CPUh rounding
ceiling; actual usage is charged from closed accounting. The cancelled
47570748 and rejected bigmem submission remain historical, with no compute
charge. No additional first-participant request is selected.


## 2026-10-12 16:18 Pacific — false_partition_preparation — pending request cancellation verified; corrected wrapper reviewed

Slurm confirms47570748 CANCELLED by the coordinator, zero allocatedCPUs,
zero elapsed time and no start. It incurred no compute charge. The corrected
13CPU wrapper was source-reviewed: CPU rounding is bounded13–16, actual CPU
count and four-hour reservation are recorded, one numerical thread is retained,
and memory/dependency/input/deadline behavior is unchanged. Select one corrected
submission now. The old request cannot run; no duplicate is left pending.

The companion parent47565811 closed COMPLETED0:0; its full2141CPU-seconds are
charged once to EP15 and are not charged again here. The latest resource-only
ASTRA edit will be validated by the selected job before data processing; the
preceding schema/semantic program and all numerical/config checks passed.


## 2026-10-12 16:17 Pacific — false_partition_preparation — scheduler-adjusted CPU binding diagnosed before execution

Submission returned47570748 at23:17:01UTC. Coordinator inspection at23:17:13UTC
confirmed PENDING/Dependency afterok47551339, normal96GiB/4h, no runtime, but
the scheduler normalized NumCPUs/CPUsPerTask/MinCPUsNode to13 while ReqTRES
retained the requested12. The submitted wrapper's exact12CPU assertion would
therefore be inappropriate for this recorded scheduler adjustment.

Select cancellation of this still-dependent, unstarted request and one
corrected13CPU/96GiB/4h submission, nominally52allocatedCPUh. The wrapper may
accept scheduler CPU rounding from13 through16 and records actual allocation
CPU count; reserve at most64CPUh, within the existing episode budget. Exactly
one worker/thread remains, all scientific/storage/deadline choices unchanged.
Preserve the original submission facts and do not alter unrelated jobs. No
new short test allocation is warranted for this resource-only correction;
the submitted job retains changed ASTRA validation before scientific access.


## 2026-10-12 16:16 Pacific — false_partition_preparation — corrected resource checks passed

The normal12CPU/96GiB/4h wrapper passed shell syntax and changed ASTRA schema
and semantic validation in parent47565811. Evidence:
`runs/false_partition_preparation/false_partition_operations_20261012_001/wrapper_normal12_syntax_47565811.txt`
and `astra_normal12_validation_47565811.txt` in that same operations directory.
Submit the corrected finite request once under the preceding admission; all
input and numerical checks remain applicable without rerunning them.


## 2026-10-12 16:16 Pacific — false_partition_preparation — scheduler memory minimum; resource correction selected

At23:15:45UTC, sbatch refused the96GiB bigmem request because that partition
requires at least256GB. No job ID was issued, no allocation began, and no
compute charge is added. This is a documented submission constraint reported
by the scheduler, not a failed scientific run or infrastructure fault.

Select the discovered normal partition with12CPU/96GiB/4h, satisfying its
8GiB/core maximum while retaining the measured96GiB memory request. Exactly
one worker and numerical thread still run; the other CPUs satisfy the memory
ratio. The reservation is now at most48allocatedCPUh, superseding the earlier
8CPUh proposal, within the12000CPUh episode budget and128parallelCPU cap even
with the22 active feature workers. The256GiB per-job memory limit is preserved.
No numerical recipe, data scope, cache/search cap, seed, dependency or global
deadline changes. Recheck the modified wrapper syntax and ASTRA, then submit
once; there is no duplicate pending job from the rejected request.


## 2026-10-12 16:15 Pacific — false_partition_preparation — wrapper checks passed; single dependent submission admitted

In parent47565811 the new wrapper's shell syntax, exact30-person release and
production configuration checks passed. Context has855 channels and search
array admission is76700852096bytes, within the80GiB cap. The first check
invocation mistakenly supplied a relative configuration path and was refused;
correcting the check's argument to the absolute path already used by the
production wrapper passed without a production-code change. Both records are
preserved. Changed ASTRA passed schema and semantic validation. Evidence:
`runs/false_partition_preparation/false_partition_operations_20261012_001/`.

The reviewed wrapper and selected recipe are ready for the one previously
selected afterok47551339 submission. No full image payload, source event or EEG
was read by these checks. Reuse the already passing search/reader checks;
no additional synthetic experiment or repeated validation is selected.


## 2026-10-12 16:09 Pacific — false_partition_preparation — complete first-participant production cost selected

Select the source-reviewed `code/run_false_partition_cost_001.sbatch`, subject
to its focused wrapper/config checks in parent47565811. The next reservation
is bigmem2CPU/96GiB/4h with one worker and all numerical threads1, at most8
allocatedCPUh; the second CPU meets bigmem's discovered64GiB/core limit.
The existing global deadline1791927000 is also enforced against actual
allocation start/end. Submit once afterok:47551339, the current full-image
assembler; no duplicate, automatic retry or successor. Fresh operations:
`runs/false_partition_preparation/false_partition_operations_20261012_001/`.

The four-field release binds the existing complete22448-image/six-family
assembly, private UUID routing, development manifest and literal30-person
development event paths. Preparation selects only structural fields and binary
target/response flags, never RT or accuracy. Under the existing coarse-button
delegation, fix that flag at its associated stimulus onset; this is contextual
information, not authenticated button timing. Preserve all22248 main images
in each participant's search. No pixels, captions, EEG amplitudes or audit data
are released to this stage. Runtime assembly admission must pass before use.

Prepare all30 metadata records within300seconds/128MiB durable output. Then
build the complete first participant sub-07 cache within3h/72GiB durable cap;
only a completed cache permits the remaining at-most1h search segment. Search
retains8 banks,64 starts each,100000 legal swaps/start and seed20261012, with
80GiB array/90GiB softRSS caps. A resource stop preserves partial work and any
actually written checkpoint, without claiming the bank complete. Measured cost
will determine feasible continuation; no30-person cache expansion is selected.
Image/context acceptance still does not establish exact selected-design
spectrum/leverage/EDF qualification. Historical charges and all original
participants, controls, tolerances and resource limits remain.

## 2026-10-12 16:08 Pacific — input_role_preparation — complete main-sequence geometry bound

All30 development participants,2160 main sequences and667440 main image events
passed the exact source/frozen/E1/E3/segment bindings and original62-channel
order at23:03:24.006297UTC. The metadata-only operation took72.277426seconds,
184692736bytes peakRSS. Evidence:
`runs/feature_preparation/development_main_geometry_20261012_001/operational.json`
and `geometry_binding.json`. No EEG amplitudes or behavioral values were used.

All three changed reader checks passed in2.972seconds; changed ASTRA passed.
Evidence is under
`runs/feature_preparation/sequence_geometry_fir_operations_20261012_001/`.
The prospective129-tap causal filter and actual-E3 span rule were already
selected above; this completes their development geometry binding, not EEG
fitting or false-partition qualification. Companion work used parent47565811,
charged once to EP15. A resource-bounded production false-partition cost
wrapper is being reviewed; it is not yet submitted.

## 2026-10-12 15:59 Pacific — input_role_preparation — complete metadata geometry binding admitted

The narrow original channel-order release succeeded with62unique labels, and
the new `code/bind_development_sequence_geometry.py` passed independent source
review. Select its full30-person/2160-sequence metadata binding after the three
affected reader checks pass in parent47565811. Inputs are the exact development
manifest and join, literal development source event/header/marker leaves and
the narrow frozen-channel release; EEG files are statted only. Behavioral
columns are not selected or used. Main source/frozen/E1/E3/segment identities,
all667440 main events and original62-channel order must agree.

Fresh output is `runs/feature_preparation/development_main_geometry_20261012_001/`.
Use295seconds internal/300seconds external,2GiB observedRSS and8MiB durable
allowance, bounded by the same1CPU/7GiB parent. The whole parent is charged once
to EP15, not again here. A complete report contains metadata/support counts,
not EEG evidence or comparison approval. Categorical source-frame reporting
was added to retain a useful failure location without values or locals.

## 2026-10-12 15:56 Pacific — input_role_preparation — recorded geometry and frozen channel field selected

Select raw sequence extent from actual preceding E3 through last image+800ms,
inclusive, and one common score window from first image−800ms through last
image+800ms, inclusive. Use the same valid recording-grid samples for primary
and mandatory negative-lag comparisons; the primary model's event-response
horizon remains0–800ms. The causal filter's entire128-sample history must be
inside the same sequence. No nominal cue-gap tolerance or clock shift is used.

Authorize a narrow metadata projection of only `common_recorded_channel_order`
from the existing `runs/input_role_preparation/20261003-freeze_metadata01/frozen_inputs_and_roles.json`
into `runs/feature_preparation/recorded_channel_order_release001.json`. Other
mixed-role fields and all audit/amplitude outcomes stay closed. The prepared
geometry caller will bind the adjacent development_manifest.json, its exact30
development event paths and this original frozen62-label order; the recent
first-header order is not a substitute. This selects metadata joins and scoring
geometry, not EEG amplitudes, fitting or a new cohort.

The actual-E3 reader repair passed independent source review. Select its three
focused metadata/reader tests in the existing attended parent47565811 after
the active EP17 response diagnosis and EP15 quality census; no new short job.

## 2026-10-12 15:53 Pacific — input_role_preparation — prospective causal filter fixed from metadata and coefficient evidence

Select the129-tap100Hz Kaiser8.6 causal recipe under the existing
`event_and_sample_rules` delegation, before EEG/model comparisons; exact choices
are in SEARCH_POLICY and the paper plan. Preserve recording-grid phase0,
128past/0future samples,64ms delay and the filtered-EEG interpretation. The
metadata inspection does not by itself release amplitudes, select score spans,
qualify main-event/channel joins or admit full fitting.

Read-only BR `pipeline_plan_review` received this supplied prospective recipe
and returned approve/low-risk with no findings. Its actual checklist only
recognized four steps and absence of registration; no KG rules were consulted
or enforced. This limited automated response is not EEG-specific validation or
canonical scientific approval. The choice rests on the measured frequency
response/support, explicit interpretation and source review. The reader's fixed
one-second gap is being replaced by exact source-labelled E1-to-recorded-E3
binding; no tolerance, row removal or audit access is authorized.

## 2026-10-12 15:48 Pacific — feature_preparation — all image delivery complete; 22 feature workers running

Coordinator-confirmed builder47551249 completed successfully with Slurm
`COMPLETED 0:0`. Its report finished at22:43:20.597UTC: all22448 original
image IDs (22248 main,200 repeated),702 pixel packs and101754313015 pack
bytes were delivered. The process reported5634.521seconds elapsed and
686328KiB peak RSS. Slurm's5653seconds on one CPU charges1.570277778
allocatedCPUh once to this selected EP18 chain. Evidence:
`runs/feature_preparation/image_full_operations_20261012_001/builder_single_47551249/execution.txt`.

At the coordinator's22:48:07UTC scheduler observation, fast-array47551335
tasks0–5 and BLIP-array47551337 tasks6–21 were all running:22 active feature
workers. Fast tasks22–25 waited for the existing array concurrency limit;
assembler47551339 waited on successful dependencies. This is the original
26-task extraction chain, not duplicate submissions or a new resource grant.
Builder completion establishes full pixel delivery only. Six-family feature
completion and exact all-ID assembly remain outstanding at this observation;
no EEG fitting or scientific result is implied.

## 2026-10-12 15:44 Pacific — input_role_preparation — development geometry and prospective FIR inspection completed

The selected serial companion completed in substantive EP15 parent47565811
at22:44:02UTC, taking19.9335seconds and reporting127762432bytes peak RSS
(121.84MiB). Evidence is
`runs/feature_preparation/sequence_geometry_fir_20261012_001/operational.json`
and `metadata_and_fir.json`. The parent allocation remains charged once to
EP15; this is not a separate compute reservation or EEG fit.

All30 development recordings report1000Hz and an observed common set of62
channels. Each has84 nonempty E3 intervals:72 with309 E1 markers and12 with200.
Every recorded interval supports the proposed last-E1-plus800ms endpoint and
first-E1-minus800ms scoring start with the full128-sample causal filter support.
There are no empty E3 intervals. These are header/marker geometry results;
the EEG binaries were statted only. No EEG amplitude, event TSV, frozen event
join, behavioral value or audit input was read. The previously frozen channel
order and exact main-sequence identities were not rechecked by this inspection.
An interval's event count is descriptive, not a substitute for that join.

The single prospective129-tap100Hz Kaiser-beta8.6 FIR has DC gain1 and a
64ms group delay, with128ms past and zero future support. On the computed
frequency grid, maximum gain error over0–80Hz is0.0005349652 and minimum
attenuation over125–500Hz is86.7262dB. The analytic between-grid envelopes
give maximum passband error0.0005982048 and stopband attenuation79.2245dB;
these envelopes explicitly exclude numerical roundoff. The100Hz cutoff is
the half-amplitude point, not a flat-passband claim. Exact float64 taps and
their SHA256 are retained in the report.

The report does not adopt a filter or span recipe. A prospective causal
recipe would keep event/sample clocks unchanged and target filtered EEG over
the existing0–800ms horizon. Its64ms delay is material: late unfiltered
responses can extend beyond that fitted interval. Negative-lag and shifted-label
controls remain mandatory; causality alone is not evidence that they pass.
No high-pass or detrending is selected by this inspection.

The geometry exposes a concrete integration correction before EEG access:
the current reader accepts only one exact `cue_to_first_samples` value, but
the2160 intervals with309 markers have measured E3-to-first-E1 gaps of999
(18 intervals),1000(2052),1001(89) and1025(1) samples. The repeated-count
intervals also include994. Preserve these recorded coordinates and bind each
main sequence to its actual E3 through the source/frozen structural event
join; do not round the onset, discard a sequence or infer primary membership
from309 events. Reader source changes and their execution remain unselected
in this entry. Full-image features, false-bank qualification and practical
exact-capacity execution are separate unfinished stages; no empirical EEG
comparison or scientific result follows from this metadata report.

## 2026-10-12 15:36 Pacific — input_role_preparation — recorded sequence geometry and prospective FIR inspection selected

Root reviewed inspect_sequence_geometry_and_fir.py, including nonempty-interval
and explicit unknown/failure aggregation. Select one serial companion in the
forthcoming substantive EP15 allocation:1CPU,180s plus5s grace,2GiB observedRSS,
16MiB durable output and one numerical thread. This soft RSS bound is checked
between bounded reads; the parent7GiB Slurm limit is the hard allocation bound.

Release only the existing development_manifest.json under
`runs/input_role_preparation/20261003-freeze_metadata01/` and its thirty literal
development .vhdr/.vmrk pairs below
`$OAK/data/br_autoresearch_data/things_eeg1/openneuro-ds003825-v1.2.0/source`;
corresponding .eeg files are stat-only. Header cap8MiB, marker cap128MiB per
person, one person at a time. No event TSV/image join, EEG amplitudes, image
arrays, behavior or audit input is opened. Write fresh
`runs/feature_preparation/sequence_geometry_fir_20261012_001`.

Report every recorded E3 interval, not inferred main-only identities, and
compute one prospective129-tap100Hz Kaiser8.6 causal FIR's numerical response.
Its0..128ms support and64ms delay are explicit; timestamps and proposed0..800ms
filtered-EEG horizon are not shifted. This inspection does not adopt a filter,
span, main-sequence join or empirical EEG comparison. Changed ASTRA must pass
within the same parent. Parent cost is charged once to EP15; active image jobs
and their source files are untouched.


## 2026-10-12 15:29 Pacific — false_partition_preparation — search checks and changed ASTRA passed

Both focused search methods passed in0.163s in parent47553427: exact two-group
updates agree with full recomputation and checkpoint continuation preserves
the RNG, assignments and consumed counters. Changed ASTRA passed schema and
semantic validation. Evidence under
`runs/feature_preparation/image_full_operations_20261012_001/`:
`false_partition_search_checks_47553427.txt` and
`astra_false_partition_search_validation_47553427.json`.
No real false-partition cache/search, EEG preparation recipe or EEG comparison
was run. Parent resources are charged once to EP17.


## 2026-10-12 15:19 Pacific — false_partition_preparation — concrete search implemented; two affected checks selected

The new false_partition_search.py implements fixed fold/family cosine caches,
physical-sequence context and exact two-group swap updates, complete image/context
distribution acceptance and diversity. It preserves all eight members,64starts
and100000legal proposals per start. Manual continuation retains RNG, assignments
and consumed counters; provisional matching is distinct from final design/EDF
qualification. Independent review found one soft-RSS continuation defect, now
repaired; actual allocation errors remain failures. No empirical recipe or search
allocation is adopted by this implementation.

Select the two focused functional methods in test_false_partition_search.py,
<=90s plus5s grace, serially in existing parent47553427 after EP17's active work.
They compare actual two-group updates against full recomputation and exact
checkpoint continuation on invented arrays. Revalidate changed ASTRA. Existing
reader, tuning, capacity and design passes are reused. No EEG, real cache or
new partition is opened by these checks, and no separate test-only job is added.

## 2026-10-12 15:04 Pacific — baseline_search and false_partition_preparation — reader and exact design checks passed

The authentic BrainVision reader's two methods passed in0.614s. The shared
slot-capacity module's five methods passed in0.009s; all four selected-design
diagnostic methods passed in0.053s. Changed ASTRA passed schema and semantic
validation. Evidence is under
`runs/feature_preparation/image_full_operations_20261012_001/` in
`brainvision_reader_checks_47553427.txt`,
`slot_capacity_shared_checks_47553427.txt`,
`selected_design_checks_47553427.txt` and
`astra_selected_design_validation_47553427.json`.

Before execution, independent source review identified an underestimated
dense-array peak. The estimate was corrected to include caller inputs,
selected base factors and reduction temporaries; the existing resource test
now covers the tall-array lower bound. No mathematical definition, empirical
threshold or passing assertion was weakened. The exact operator checks compare
the selected Schur/leverage results to an explicit full joint smoother.

All checks ran serially with one numerical thread in EP17 parent47553427;
its allocation is charged once to EP17. No EEG, candidate score, accepted
false partition or full-scale resource qualification is implied. The remaining
search implementation and structured-capacity cost issue recorded below persist;
the active submitted image jobs were untouched.

## 2026-10-12 14:59 Pacific — false_partition_preparation — exact design diagnostics implemented; prospective definition fixed

The new `code/prepare_false_partition_inputs.py` binds the completed six-family
image bank to all30 development participants' existing UUID/event identities,
preserving all22248 main images and source target/button flags. Root and
independent source review found no concrete defect. It is unrun, needs the
completed image bank and an explicit metadata release, and neither reads EEG
nor creates a qualified partition bank.

Select the named design diagnostic definition before empirical use: singular
values of the same conditional design F used by term EDF, and the diagonal of
the full joint weighted-data smoother at the selected penalties. Retain squared
singular values separately; apply the .10 relative spectrum bound to singular
values. Apply both the existing standardized distribution bounds and additional
relative quantile bounds to spectrum/leverage. Use the fixed five linearly
interpolated quantiles, true-distribution sample SD, and absolute1e-12 equality
for a zero reference scale/quantile. These are explicit prospective analyst
definitions where the policy wording is underspecified, not a claim that the
source uniquely prescribes them. Term EDF and its .05 capacity bound remain.

`code/selected_design_diagnostics.py` evaluates a shared base and each slot
sequentially, then all eight frozen slots in all six folds without retuning or
regeneration. The shared factorization in `code/slot_capacity.py` preserves its
public13-penalty/term-EDF interface. Root source review found no concrete defect.
Select the five existing slot-capacity checks plus four new diagnostic methods,
each module<=90seconds plus5seconds grace, in parent47553427 after its active
EP17 image process exits. They use fixture arrays only; revalidate ASTRA then.
No EEG or false-bank qualification follows from passing software checks.

The full-scale issue is now explicit:500000 rows by74160 FIR columns needs
296.64GB decimal (about276.27GiB) for one float64 slot, already above the256GiB
per-job limit. FIR is a required search branch; it cannot be silently dropped.
Exact structured fitting is being assessed. The new diagnostic's memory guard
does not solve full-scale execution, and a resource refusal is not a scientific
underidentification result. Submitted image jobs and their source are unchanged.

## 2026-10-12 14:45 Pacific — baseline_search — authentic development reader connected; empirical preparation remains unselected

`code/brainvision_sequence_reader.py` now binds the existing development
manifest's exact header/marker/binary leaves, converts per-channel resolution
and source voltage units, anchors physical spans to documented E3 triggers and
feeds bounded intervals to the existing72-sequence bridge. All structural
event/span/mask checks precede the lazy reader's first EEG interval. It has
no scientific defaults, automatic access grant, sequence clipping or audit
reader. Root source review found no concrete defect.

Two focused methods in `code/test_brainvision_sequence_reader.py` cover actual
tiny binary offset/unit conversion and the complete72×309 metadata-to-E3 span
construction, including refusals. They remain **unrun**. Select them≤90seconds
plus5seconds termination, serially in the EP17 parent after its active image
process completes, using the qualified EP18 runtime and one numerical thread.
Revalidate changed ASTRA in the same parent; its previous validation predates
these reader inputs. These checks use fixtures, not acquired EEG.

The preprocessing/span recipe is still unselected. Algorithm and seed for
false-partition generation must freeze before EEG; all eight partitions must
qualify before candidate-discriminating comparisons. Those are distinct
boundaries, and absent final EDF matching is not a new blanket prohibition on
otherwise-authorized baseline-only work. The proposed symmetric FIR is not
adopted; compatibility with negative-lag controls and nuisance modeling must
be explicit. No raw EEG was accessed or prepared by this implementation.


## 2026-10-12 14:39 Pacific — baseline_search — tuning checks complete; sequence-trigger source resolved

All four selected tuning methods passed in19.419seconds in parent47553427;
changed ASTRA passed schema and semantic validation. Evidence:
`runs/feature_preparation/image_full_operations_20261012_001/continuous_tuning_checks_47553427.txt`
and `astra_tuning_validation_47553427.json`. This is software evidence, not an
empirical EEG comparison. Raw-reader integration, explicit full-run choices,
features/false partitions and empirical resource admission remain unfinished.

The original data paper explicitly identifies E3 as sequence start, E1 as image
onset and E2 as offset (page3), resolving the metadata-only reader's question
about E3 meaning. Its1000Hz and0.0488281microvolt resolution agree with the
authorized sub-07 development header. Source:
[Grootswagers et al.,2022, author-hosted paper](https://tijl.github.io/tijl-grootswagers-pdf/Grootswagers_et_al_-_2022_-_Human_EEG_recordings_for_1,854_concepts_presented_.pdf).
Actual event-to-E3 offsets, segment-confined spans and post-stimulus padding
must still be checked from each authorized development record; the paper does
not supply their exact sample arrays or silently select an analysis recipe.

Disclosure: the reader agent's bounded initial head of the explicitly allowed
sub-07 events TSV also exposed response/rt/correct fields for the first three
development rows (0/0.0/0), before column selection. These are behavioral values;
do not describe this inspection as entirely outcome-blind. No EEG, model
outcome or audit row was opened, and no choice used those response fields.
Further event inspection selects structural fields only.


## 2026-10-12 14:37 Pacific — baseline_search — preparation checks passed; tuning composition selected for verification

All three selected raw-preparation bridge checks passed in6.111seconds inside
the attended EP17 parent47553427. Evidence:
`runs/feature_preparation/image_full_operations_20261012_001/continuous_eeg_bridge_checks_47553427.txt`.
This verifies the explicit-recipe software interface, not empirical EEG
preprocessing or a scientific result. Authentic source-reader integration and
the full-run preprocessing recipe remain unfinished.

The new `code/continuous_tuning.py` composes the existing36 outer/inner splits,
13-point ridge grid, training-only preparation and exact term effective degrees
of freedom. It retains every feasible fold-local capacity match, requires the
outer-training match too, and exposes only the permitted baseline losses to
selection. Independent review repaired the missing nested-design-prefix check.
The caller must still supply explicit penalties, capacity matching/ties,
covariance/rank conventions, loss normalization and dense-memory admission;
this implementation freezes none of those scientific choices automatically.
Select its four focused existing methods, one numerical thread, at most180
seconds, serially in parent47553427 using the qualified EP18 runtime. Validate
the changed ASTRA object in that same parent. No raw EEG or acquired image
access is part of these fixtures. Cost remains included once in the EP17 parent.

BR's read-only review request returned directive
`ext_review_dir_bcda429cab3d` (21:20:27UTC), with generic criteria concerning
within-fold transforms, nested selection, temporal dependence and biological
units; it supplied no tailored checks and read no local artifact. These criteria
inform the source review above. This is advisory assistance, not BR approval,
canonical launch or scientific acceptance. The running full-image chain and
its submitted code remain unchanged.


## 2026-10-12 14:20 Pacific — baseline_search — continuous EEG preparation bridge implemented and checks selected

The new `outputs/code/prepare_continuous_eeg.py` preserves all72 physical
sequences and22248 main events, joins original events/markers to the frozen
metadata, checks disjoint segment-confined spans before any reader call, and
applies only explicit sequence-local detrending/FIR/decimation recipes with
full filter-support masks. It retains the62 ordered recorded channels; the
existing fold runner applies the fixed62-to61 projection once. The caller
must supply authentic source spans, header-scaled voltage samples, filtering,
grid alignment, scoring intervals and memory/time admission. No recipe or EEG
access is selected by this implementation.

Independent and owner review corrected an overly strict onset tolerance to
the already qualified round(onset*1000)==source_sample rule; the corresponding
rounding-bin case is included in an existing test. No other concrete source
defect remains. Select the three focused `test_prepare_continuous_eeg` methods
inside EP17's attended1CPU/4GiB parent47553427, serially after its image pilot,
with one numerical thread and a90second external cap. Use the qualified EP18
runtime. Tests use constructed signal fixtures only, with no raw EEG/images.
They remain unrun until observed. This short companion is charged once inside
the EP17 parent. Full image preparation47551249 and its submitted dependents
remain unchanged; complete tuning and authentic raw-reader integration are
separate unfinished work.

## 2026-10-12 14:13 Pacific — feature_preparation — builder running and independent companion stopped

The 21:13:06 UTC queue observation confirms builder47551249 RUNNING on
sh03-06n09, with the submitted one CPU, 7 GiB and six-hour allocation.
Its configuration admission and ASTRA schema/semantic validation passed.
The trusted builder's `execution.txt` records completed real-image pixel
packs, establishing actual preparation beyond startup. Fast array47551335,
BLIP array47551337 and assembler47551339 remain pending on successful
dependencies. No feature-array or empirical EEG completion is claimed.

The independent EP17 companion returned exit1; its captured stderr contains
the Slurm task exit but no Python or private-data message. Accounting records
step47551249.0 FAILED1:0 after one second. Its EP17 operational directory was
not created, so no launcher checks, password delivery or image pilot is
observed. Root is diagnosing the pre-log startup checks; the cause is not
established. As selected, this separate failure did not stop EP18's builder.
No retry, submitted-source edit or additional accounting charge is selected.
Evidence is under `image_full_operations_20261012_001/builder_single_47551249/`.

## 2026-10-12 14:05 Pacific — feature_preparation — full real-image extraction selected

The completed fixed32 pilot establishes successful delivery, all six families,
singleton repeatability and measured resource costs. Select full extraction of
all **22448 existing opaque IDs × six unchanged families**, batch1, with no
EEG or audit-event access. The reviewed full producer/worker/assembler reuse
their three passing focused tests. A source review repaired duplicate-invocation
failure markers before selection; previous successful outputs remain immutable.

Selected config: `outputs/config/full_image_extraction_20261012_001.json`.
Fresh run `image_features_full_20261012_001`; fresh restricted temporary cache
`outputs/cache/image_pixels_full_20261012_001`. Decode the original archive
once into32-image packs, each≤1GiB, total≤256GiB. The trusted builder privately
uses this archive's existing authorized notice. Workers receive only pixels,
opaque IDs, qualified assets and public resource configuration. No new terms
acceptance, secret export, credential reuse from another episode or download.

Family shard counts are **2/1/1/2/16/4**:26 tasks. Submit fast indices
`0-5,22-25%6` at2h and BLIP indices `6-21%16` at4h: maximum22 concurrent
single-CPU workers. Builder6h and assembler1h give **91 allocated CPU-hours
maximum**, including all startup, validation and the separately scoped EP17
companion below. Each EP18 allocation uses1CPU/7GiB/zeroGPU and one numerical
thread. The two full-inventory ResNet tasks each allow256MiB of outputs;
other tasks128MiB, assembly2GiB, metadata8MiB. Their5.5078125GiB reservation
fits the8GiB run cap with operational/failure headroom. Quota observation
showed Oak171.9TB/250TB; use the established episode-local layout and local
worker caches. The previously stopped Scratch route is not probed or retried.

Runtime deadlines retain the actual scheduler start: minimum of
StartTime+phasecap−60, EndTime−60 and **2026-10-13 21:30UTC**. Scheduler
Start/End timestamps can include a few seconds of slack, as in pilot47547023;
their exact difference is not used as a TimeLimit check. Root submits explicit
phase time limits. Every process has a terminating timeout; no worker polls
the scheduler, retries, requeues or submits a successor. Root submits one
builder, two afterok worker arrays, then one assembler after both arrays.

Budget reconciliation: documented EP18 charges are0.6900925926CPUh. Historical
shared parents46425841/46427768/46429138 used3346×4,1285×2 and2077×1 CPU-seconds.
Even assigning all those shared-parent costs to this admission bound gives
5.008611CPUh, so the conservative total including the new91h is **96.698704h**,
far below12000h. This upper bound is not a second debit: retain each existing
shared charge once. No prior explicit calendar origin was recorded. For this
admission, conservatively retain **October3 00:00UTC**, the beginning of the
earliest documented execution day, as a reference, including all intervening
waits. The selected absolute end is261.5h later, within336h. This is an explicit
conservative admission reference, not a recovered historical timer or reset
from today; September27 records a design revision, not an execution attempt.

The builder first validates the selected coverage/caps and changed ASTRA
inside its allocation. It then hosts EP17's independently selected first32
real-image pilot in an actual1CPU/4GiB/15-minute Slurm step, with separate
paths, interpreter, credentials and reports. That step's entire cost remains
inside this parent's6h reservation and is charged once to EP18. Its failure
is recorded and does not prevent independent EP18 delivery. No EP17 output
is an EP18 scientific input. Both scopes preserve their data-role boundaries.

The ASTRA program records delivery, complete shards and assembly under
`feature_preparation`; full feature freeze, false partitions and EEG analysis
remain separate unfinished work. Submission IDs and observed startup follow
below after side effects; no full-extraction completion is asserted here.

Submission completed at **21:07:09UTC**: builder **47551249**, fast array
**47551335** (`0-5,22-25%6`), BLIP array **47551337** (`6-21%16`), assembler
**47551339**. Both arrays require successful builder completion; assembly
requires both complete arrays. This is28 finite tasks including delivery and
assembly, represented by four scheduler submissions. The21:07UTC scheduler
snapshot confirms1CPU/7GiB and6h/2h/4h/1h requests, the selected array limits
and dependencies. Builder is PENDING; no new compute or companion startup is
claimed. No duplicate pending job is selected.

Operational records live in `image_full_operations_20261012_001/`, including
the exact returned IDs in `submission.txt`. The source review's final correction
caps the outer EP17 step-start wait and execution at min(remaining,900s), plus
10s termination grace, so a waiting companion cannot consume the builder's
entire reservation. The current submitted wrapper also reuses the previously
passing explicit ASTRA Python package path. Submitted source is now preserved;
any observed failure is diagnosed before a separately recorded repair attempt.

## 2026-10-12 13:49 Pacific — feature_preparation — all six real-image pilot families complete

Repaired attempt `image_feature_pilot_47547023_repair001` completed at
20:49:28.036080 UTC, exit 0, in **299.136370 seconds** inside parent47547023.
All six families encoded the same 32 real images, plus the selected one-image
repeat per family. Each repeat's maximum absolute difference was zero and
its details agreed exactly. Peak family RSS was 1504868 KiB (BLIP); no GPU,
EEG or audit-event access occurred. This establishes image delivery and
operational feasibility, not a neural result or permission-blinded evaluation.

The trusted builder decoded 143339898 RGB bytes in 10.257942 seconds; the
metadata-free pack was 143349392 bytes. Family outputs before the aggregate
report totalled 1016614 bytes. See the
[operational report](runs/feature_preparation/image_feature_pilot_47547023_repair001/image_feature_pilot_report.json).
The failed first attempt and its diagnosis below remain preserved.

Linear extrapolations to 22448 images, excluding model loading and I/O, are
8917.90/3058.73/3094.18/5093.32/96275.83/16899.69 seconds for low-level,
supervised ResNet, DINO ResNet, CLIP image, BLIP/CLIP text and DINOv2,
respectively: approximately 37 CPU-hours of encoding. These are estimates
from the fixed first32 UUIDs, not full-data measurements or job reservations.
BLIP dominates; full extraction should split its work more finely than the
faster families while retaining useful job durations. Image delivery projects
to about two CPU-hours and 94 GiB of decoded packs, with further allowances
needed for variability and I/O.

Four new full-extraction source files now separate bounded pixel packs from
per-family job ranges and require exact six-family/all-image completion.
Select their three focused small-fixture delivery/coverage/failure tests
inside the existing attended parent, capped at 180 seconds, one numerical
thread, with no new reservation or empirical extraction. Independent source
review is in progress. Full extraction remains unlaunched; its explicit
resources, deadline and storage allowances must be recorded before submission.

The three full-runner checks initially passed in0.126seconds. Source review
then identified a concrete history-preservation defect: refusal of a duplicate
invocation could append a failure marker to the prior successful directory.
The narrow repair records ownership only after this invocation's successful
exclusive directory creation; only owned paths may receive a failure marker.
The existing coverage test now also invokes all three duplicate CLI routes and
verifies prior reports remain byte-identical with no added failure markers.
All three affected checks passed in **0.119seconds**, and independent source
review accepted the ownership repair. Check output is in
`image_pilot_operations_47547023/full_image_runner_ownership_repair_checks.txt`.
No full extraction, production-resource selection or scientific successor ran.

The attended shell exited at **20:55:14 UTC**. Slurm records parent47547023
COMPLETED0:0, **908seconds ×1CPU =0.252222 allocated CPU-hours**, charged once
to EP18 with all prior charges preserved. The overlapping step reports
897seconds and3304456K MaxRSS; it is not an additional charge. The two-hour
reservation is closed and unused time released. Durable pilot and check
artifacts remain in the episode; allocation-local decoded packs were temporary
and are reproducible from preserved inputs. No compute shell remains active.

## 2026-10-12 13:43 Pacific — feature_preparation — private-notice parser diagnosis and bounded repair selected

In verified parent **47547023**, all five new image-delivery checks passed in
0.427 seconds and all four complete fold-composition checks passed in 11.570
seconds. Changed ASTRA schema and semantic validation passed. Their outputs
are in `runs/feature_preparation/image_pilot_operations_47547023/`.
The fold bridge is tested source integration, not a complete raw-EEG pipeline,
thirteen-penalty search, EDF qualification or empirical EEG result.

The real-image attempt `image_feature_pilot_47547023` started at
20:42:11.331144 UTC and failed after **1.746346 seconds** with categorical
`PasswordNoticeError`, before any image was decoded or family executed. The
builder had reached the encrypted archive's private notice; the original
report and empty local pack are preserved. This is a parser limitation,
not an authentication rejection or infrastructure failure.

Private on-compute format inspection reported only categorical properties:
the notice has exactly one explicit password-labelled field with descriptive
text before its colon and one unquoted ASCII token as its value. Neither the
notice nor the credential was printed or stored in a report. The original
parser accepted only a bare token or a label beginning the line. Select the
narrow repair to accept the observed explicit labelled-field format, retain
ambiguity rejection and test that behavior using an invented fixture.

After the changed password test passes and source review agrees, select one
fresh attempt `image_feature_pilot_47547023_repair001` inside this same attended
parent, with a fresh corresponding local directory. Keep all 32 UUIDs, six
families, singleton repeatability, qualified runtime, exact archive and the
original **1791844508** deadline. No new reservation, clock reset, full
extraction or automatic retry follows. The failed attempt remains visible;
all parent use is charged once. The combined small failed report and new
pilot outputs must remain within the selected 64-MiB durable allowance.

The changed password-format method passed on compute in **0.004 seconds**;
the prior unaffected eight methods and ASTRA retain their passes. Source
review confirmed the narrow parser change. The fresh repaired pilot began at
**20:44:28 UTC** inside the same parent and original deadline. The encrypted
archive delivery succeeded and the first family completed all 32 selected
images; complete six-family success remains pending. No credential was logged.

## 2026-10-12 13:37 Pacific — feature_preparation — attended real-image cost-pilot allocation selected

Select one attended `srun --pty` parent in the currently advertised
`russpold,normal` partitions: one CPU, 7 GiB, two hours, zero GPUs and one
numerical thread. Reserve at most two additional allocated CPU-hours once to
EP18, preserving every prior charge and the existing 12000-CPU-hour ceiling.
The parent will run the focused new runner checks and changed ASTRA validation,
then the selected first-32-real-image/all-six-family pilot after source review.
No EEG or audit payload is admitted. There is no automatic retry or successor.

Use the already qualified CPU interpreter from allocation 46842156. The shared
work deadline is actual Slurm StartTime plus 6900 seconds, bounded by EndTime
minus 60; a terminating outer timeout bounds native calls. Slurm enforces the
7-GiB memory allocation; the pilot additionally stops on observed historical
process RSS above 6.5 GiB at interpreter checks. Local decoded-pixel pack is
limited to 256 MiB, all pilot-local files to 1 GiB, and durable pilot output to
64 MiB. No new environment, asset transfer, namespace probe or scratch-access
retry is selected. Use the established OAK output and allocation-local layout.

The original image archive is the recorded steward acquisition's
`archives/images_THINGS.zip`; its existing academic/noncommercial acceptance
receipt was rechecked as documentation only. The trusted builder may consume
the adjacent `terms/password_images.txt` privately if selected members require
decryption. Never expose a credential or source crosswalk to a worker, log or
terminal; no secret content was opened by the coordinator. Preserve the archive.

BR's advisory `pipeline_plan_review` accepted the textual custom-code plan
with no findings, but consulted no KG rules and applies no enforcement. Its
generic approval is not scientific validation or launch authority; the fixed
contract, independent source review and compute evidence govern this action.
The first malformed request returned a missing-tool-field error and no review.

Slurm accepted attended parent **47547023** and reported it queued at
20:37:40 UTC. The session is still waiting for resources; no compute-node
shell, test or image inference has started at this observation. Keep this
single request and its two-CPU-hour maximum reservation; do not duplicate it.

The actual PTY shell is now verified on **sh04-04n24.int**, parent 47547023,
one CPU/7168 MiB/no GPU. Slurm supplied StartTime epoch1791837608
(20:40:08 UTC) and EndTime1791844815. The shared selected deadline is
**1791844508 (22:35:08 UTC)**. Allocation-local space was available. The five
image-runner and four fold-runner checks have been started serially there;
their results and changed ASTRA validation belong to
`runs/feature_preparation/image_pilot_operations_47547023/`.

Independent source review found no blocking delivery/composition defect.
To satisfy the registry's numerical-consistency requirement at the selected
singleton batch size, select one repeat of the first image per family:
32 unique images/33 inferences, atol1e-6/rtol1e-5 plus exact generated-caption,
token and truncation details. This is recorded before real-image processing,
does not select another batch size and makes no cross-batch equivalence claim.
Its measured cost is retained separately from the 32-image extrapolation.

## 2026-10-12 13:34 Pacific — feature_preparation and baseline_search — implementation resumed under adopted procedural boundary

Selected next work under standing resume authority: compose a real-image cost
pilot using the existing six image adapters and a complete nested-fold EEG
runner using the already tested numerical core. Source inspection corrects one
stale item in the 13:23 entry: the fixed 62-to-61 sensor projection and explicit
training-residual whitener already exist in `code/training_transforms.py` and
passed their October 6 checks. The missing EEG work is their fold composition,
raw-data preparation and complete fitting/evaluation procedure; do not replace
or requalify the unchanged projection merely because the summary was stale.

The image pilot selects the first 32 existing opaque UUIDs in lexical order,
all six prospectively fixed image families, sequential fresh CPU workers and
batch size one. The trusted builder alone may use the already frozen private
image routing and authenticated archive join; workers receive only decoded RGB
arrays and opaque IDs. Reuse the qualified exact CPU runtime and public assets.
The proposed one-CPU/7-GiB/two-hour parent includes focused changed-code checks
and the substantive real-image pilot, with no EEG access, full extraction,
network acquisition or automatic successor. Final finite byte/deadline limits
and the returned allocation will be recorded before execution. No allocation
or image inference is claimed by this selection.

The complete fold component will retain six outer and five inner splits, the
fixed sensor basis, training-only transforms, common masks and direct joint
fits. It returns paired per-sequence whitened errors. The present contract
does not explicitly distinguish mean sequencewise loss ratios from a ratio of
equal-sequence mean errors, so this implementation does not silently select a
scientific aggregation rule. Raw preprocessing and final scoring remain
separate admitted work.

Unplanned role-path listing disclosure. During source discovery, the review
agent ran `rg --files outputs/code outputs/configs outputs` piped through a
filename filter and `head -n 105`. This mistakenly listed the path strings
`runs/input_role_preparation/20261003-freeze_metadata01/audit_manifest.json`
and `audit_event_image_join.tsv.gz`, plus the private-routing pathname under
`runs/feature_preparation/20261003-mapping01/`. It did not open those files,
their attributes or their contents, and read no audit participant identity,
EEG, predictions or outcome. The discovery was stopped; further inspection is
restricted to explicit source/config/document paths. This is an unplanned
violation of the procedural no-listing rule and remains disclosed in the
preparation history. No EEG result has been produced to mark stale, and no
audit evaluation opening or technically enforced/blinded boundary is claimed.

## 2026-10-12 13:23 Pacific — scientist decision: procedural same-account separation adopted

At about 13:10 Pacific, in a portfolio review of evaluation separation for
EP02, EP15, EP17 and EP18, the scientist (zijiao) was told that the same
account can read the evaluation data and that the sandbox route failed on
October 3 (namespace `ENOSPC`, EP18 job 46427768). The recommendation was
procedural separation with a disclosed weaker claim, as EP07 already ran,
because these public data can be re-downloaded by the same account anyway;
the alternative was sending the SRCC draft below. The scientist replied:
"follow your recommendation".

EP18 adopts the procedural-only boundary that the
[October 6 route review](paper_plan.md#october-6-execution-route-recommendations--unadopted)
named as a reviewable alternative with a disclosed weaker evaluation claim.
That sentence named the option without terms, so short terms were added: a
dated amendment in [GOAL.md](../GOAL.md#data-readiness-and-access-boundary),
`access_boundary.amendment_2026_10_12` in `SEARCH_POLICY.yaml`, and notes in
[DATASETS.md](../DATASETS.md#development-and-audit-access). It replaces the
requirements for verified/enforced EEG development/audit separation and for
an externally controlled pixel-only image worker. The boundary is procedural,
not OS isolation; every audit or evaluation result must say so. The SRCC
draft is not sent. The paper plan's October 6 "not adopted" wording and the
enforced deployment described in `inputs/README.md` (per the October 3 entry;
not opened here) are superseded and were not edited.

The separation precondition is satisfied procedurally; no flag was set true.
The audit-seal flag stays false. EEG-driven candidate comparison remains
prohibited until the other `qualification_before_candidate_comparison` items
pass, and the audit still opens at most once, after analysis choices are
frozen and recorded. Still gating the next steps:

- remaining image/caption features: outcome-blind cost pilot and extraction,
  with decoded pixels and opaque IDs only inside the worker role, then the
  complete `frozen_feature_record`;
- the eight false partitions: those features, then EEG-blind construction and
  qualification against every tolerance in every fold;
- the EEG runner: the 61-dimensional sensor transform and the complete
  continuous-EEG fitting/evaluation pipeline are not yet implemented.

Roles, EP17 treatment, effective-df definition, penalty grid, 5% tolerance,
opening counts and budgets are unchanged. Every audit-role access is recorded
in this log from now on; this task made none. Documentation-only on sh04-ln04
without an allocation: no compute, payload, input, audit or outcome access,
scheduler call or contact. ASTRA was not edited: `baseline_search` and
`audit_evaluation` still depend on
`input_role_preparation.verified_access_boundary`. Refresh
`outputs/astra/v0.0.14/astra.yaml` to the procedural boundary and validate it
at the next compute session.

## 2026-10-11 23:44 Pacific — input_role_preparation current unsent isolation support handoff

Prepared this current handoff from the retained October 3 observation; the
failure has **not** been reobserved. The exact CPU runtime and model assets
subsequently qualified in job **46842156**. The historical Torch/build and
asset-transfer requests below are therefore not the current dependency.
No support message was sent, infrastructure inspected, or runtime retried.

**Not sent; not needed after the 2026-10-12 decision (entry above). Text kept below.**

**Draft for the user to review and send to SRCC, if desired:**

> Subject: Supported isolated image/EEG evaluator route for a Slurm study
>
> On October 3, 2026 at 20:04:09 UTC, job 46427768 on
> sh02-01n18.int ran a sentinel-only isolated-runtime check. Bubblewrap exited
> before child launch with `Creating new namespace failed: nesting depth or
> /proc/sys/user/max_*_namespaces exceeded (ENOSPC)`. That route was stopped;
> this is the retained October 3 event, not a fresh failure report.
>
> Could you identify a supported Slurm execution pattern for an externally
> controlled image worker and EEG development/evaluator separation? The image
> worker should receive only decoded pixels and opaque IDs. The development
> process must be unable to read sealed audit EEG, mixed-source outcomes or
> evaluator outputs, or re-download audit EEG from public sources; it must not
> be able to restore host mounts or network access itself. We need an effective
> filesystem/process/network boundary, not same-account file lists alone.
>
> Exact Torch 2.4.1/Torchvision 0.19.1 and the selected model assets have since
> qualified in a private project environment. This request concerns a supported
> isolation route, not software installation or downloads. We have selected no
> further namespace probe, retry or weaker isolation.
>
> Retained evidence:
> `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability/outputs/runs/input_role_preparation/20261003-runtime_sentinel01/runtime_feasibility.json`.

The required boundary is described in [inputs/README.md](../inputs/README.md).
A supported pattern could also help satisfy EP15/EP17 reader requirements;
this is not evidence that either episode experienced the same namespace
failure. Existing image/EEG access restrictions and prior history remain intact.

## 2026-10-07 02:21 Pacific — feature_preparation public-label geometry completes

The separately authorized label-only child in EP21 parent **46881096**
completed on **sh04-04n24.int**. Its adjacent
[wrapper log](runs/feature_preparation/label_geometry_46881096.txt) records
`label_geometry_exit=0`; the closed
[report](runs/feature_preparation/label_geometry_46881096/label_geometry_report.json)
records `label_geometry_cache_complete`, **33.248759 seconds**, peak RSS
**318,180 KiB**, and **507,723,286 cached array bytes**, within the planned
600,000,000-byte allowance. This byte count describes the reported cached
arrays, not a fresh recursive scan or a new enforcement mechanism.

All **1,854 public concept labels**, **27 requested rank triples** and **81
rank/bandwidth mappings** are retained through **six distinct geometries** and
**18 kernels**. Numerical ranks are 512 for concept-name embeddings, 12 for
human properties and 53 for human categories. Equivalent requested ranks
share the same prepared geometry without repeating equivalent fits. All
kernels retain rank 1,854; the largest reported relative reconstruction error
is **3.79008e-15**, below the fixed 1e-10 tolerance. These are label preparation
and numerical facts, not empirical EEG prediction results.

The runner explicitly records no EEG/image/event payload access, no complete
feature-freeze record, no audit-seal verification, no capacity or penalty
selection, and no full temporal-operator resource qualification. The image/EEG
reader-isolation blocker and stopped namespace route remain unchanged. Use
these completed label caches when the remaining empirical access and modeling
conditions are satisfied; do not repeat unchanged preparation.

The 33.248759-second runner time sits within the selected 1,800-second child
allowance and is **included in the EP21 parent charge**. It excludes wrapper
startup/shutdown and is not a separate physical allocation charge. The parent
remained active fitting EP21 decoders at the latest scheduler observation;
its final actual runtime is not yet known.

## 2026-10-07 00:47 Pacific — feature_preparation public-label geometry selected for shared worker

Under the scientist's standing multi-episode resume instruction, the
coordinator selects the prepared public-label PCA/RBF production step for
the next fresh EP21 global array index **12 / work_0013**. This is a separate
EP18 work scope aggregated with substantive decoder fitting, not an EP21
scientific input. **Production remains unrun**: no parent job ID, startup,
cache, measured runtime or completion is established by this selection.
The earlier statement that no allocation or reservation was selected is a
historical preparation snapshot, superseded by this prospective child scope.

The serial child uses **one CPU / 7 GiB / zero GPUs**, the already qualified
`$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python`,
and `code/prepare_label_geometry.py --max-seconds 1750`. Its complete lifetime,
including setup and shutdown, is bounded by **1,795 seconds to TERM plus a
five-second KILL grace**, at most **1,800 seconds**. All worker and numerical
thread counts remain one. Reuse the 27 passing component checks and changed
ASTRA validation from 46864089; no repeated test or new validation program is
selected. The first necessary kernel remains the retained cost pilot, with
the existing projection guard permitted to return a documented partial cache.
There is no automatic continuation or padding to reach a minimum runtime.

Inputs are only the completed public concept-name embeddings, aggregate
human-feature table, their identity/qualification records, the feature
registry, and permitted structural/frozen-inventory reports. Every requested
rank triple and bandwidth mapping is retained through the existing effective
rank aliases. The output is a fresh
`outputs/runs/feature_preparation/label_geometry_${SLURM_JOB_ID}/` directory
with an adjacent `label_geometry_${SLURM_JOB_ID}.txt` log. No image pixels,
private routing, EEG, event payload, sibling scientific input, access amendment,
complete feature freeze or scientific comparison is included.

The parent remains a **one-CPU / 7-GiB / eight-hour EP21 worker**. Its full
reservation and eventual actual allocated time, including this child, are
charged **once to EP21** within its existing 500-CPU-hour ceiling. Record the
EP18 child elapsed time against a nested allowance of at most **1,800
CPU-seconds, INCLUDED in that parent charge**; do not create another physical
allocation charge or subtract it from the EP21 accounting. The prospective
EP18 storage allowance is **600,000,000 bytes**. Fixed array shapes give an
analytic estimate of approximately **0.51 GB**; reconcile actual retained
bytes from the runner report and adjacent log. This allowance is not a newly
implemented byte-cap enforcement claim.

The coordinator owns the wrapper and EP21 execution records. Child failure
must be recorded independently, after which the decoder worker restores its
own environment and clips fitting to the original allocation deadline. Frozen
EP21 authority, assignments, work manifest and accounting journal are not
changed by this EP18 documentation. No tests, imports, payload reads, scheduler
calls or submission occurred in this documentation task.

## 2026-10-06 22:59 Pacific — baseline_search component checks pass in EP21 job 46864089

The four prepared component suites completed successfully inside the
coordinator-owned EP21 allocation **46864089**, one CPU / 7 GiB on
**sh04-04n24.int**. Direct inspection of the episode-local receipts establishes
the following results:

| Component | Passed tests | Reported suite runtime | Evidence |
| --- | ---: | ---: | --- |
| Training transforms | 8 | 0.010 s | [test_training_transforms.txt](runs/component_checks_job_46864089/test_training_transforms.txt) |
| Continuous sequence model | 10 | 0.012 s | [test_continuous_sequence_model.txt](runs/component_checks_job_46864089/test_continuous_sequence_model.txt) |
| Added-slot capacity | 5 | 0.010 s | [test_slot_capacity.txt](runs/component_checks_job_46864089/test_slot_capacity.txt) |
| Label RBF geometry | 4 | 0.009 s | [test_label_rbf.txt](runs/component_checks_job_46864089/test_label_rbf.txt) |

All **27 tests passed**; their reported suite runtimes sum to **0.041 s**.
That sum excludes interpreter startup, ASTRA validation and the rest of the
parent workload. The [status receipt](runs/component_checks_job_46864089/status.txt)
records exit zero for each suite and ASTRA. The
[ASTRA output](runs/component_checks_job_46864089/astra_validation.json)
explicitly reports successful schema and semantic validation of
`outputs/astra/v0.0.14/astra.yaml`.

These array-only fixture checks verify the prepared numerical components and
their tested interfaces. They do not execute `prepare_label_geometry.py`,
materialize public-label PCA/RBF caches, qualify the complete temporal
operator's empirical resource use, release an image/EEG reader, establish an
access seal, fit empirical EEG or produce a scientific score. The earlier
pending descriptions are historical; label preparation and the existing
image/EEG isolation and pipeline dependencies remain unfinished. Reuse these
passes rather than repeating unchanged checks.

The coordinator reports the parent job still running at this documentation
update. Its entire allocation cost is charged **once to EP21**, as also stated
in the receipt; no separate EP18 charge or final parent runtime is recorded.
This documentation task read only the named check receipts and episode
records, with no source payload, import, test or scheduler operation.

## 2026-10-06 21:32 Pacific — baseline_search nonlinear label component reviewed

**Historical preparation snapshot:** the component checks passed later in
46864089, and the October 7 shared-worker selection above supersedes this
entry's then-unselected allocation status. Label production itself remains
unrun until that new worker starts.

Resume continues under the existing authorization. The array-only
[`label_rbf.py`](code/label_rbf.py) implements the registered three-bandwidth
RBF slot from the same training-transformed name, property and category
features as the linear label model. Its square-root kernel coordinates give
the intended kernel-ridge penalty geometry. Concept IDs only join rows;
they never enter distances. Explicit numerical rank and reconstruction
tolerances preserve and report discarded directions without adding a
diagonal ridge or changing the bandwidth grid.

The coordinator and an independent worker found no blocking static issue.
Four focused kernel-ridge/sequence-capacity checks remain **unrun**, alongside
the eight transform, ten affected sequence and five capacity checks already
recorded below. Changed ASTRA validation is also pending. These are method
implementation checks, not EEG results, and will be bundled with substantive
authorized work rather than submitted as another short allocation.

The existing structural preparation reports establish one instance of every
concept in every main block for all 46 eligible participants, and the freeze
builder verifies their common concept inventory. Consequently all permitted
eight-block inner and ten-block outer training sets have the same 1,854 unique
label-fitting units. Identical label transformations can be reused for those
identical inputs; this does not authorize sharing image-family transforms
across held-out exemplars. The finite
[`prepare_label_geometry.py`](code/prepare_label_geometry.py) runner is now
prepared and independently statically reviewed, preserving all requested rank
combinations while avoiding repeated work for equivalent capped ranks.
At most six distinct label geometries and
eighteen bandwidth kernels are needed; actual numerical ranks remain unmeasured.
This work alone has no established ten-minute runtime, so no new allocation
or reservation is selected and no redundant work will be added to extend it.

The runner stores shared PCA parameters and one compact eigensystem per
distinct kernel, releasing each before continuing. The first necessary kernel
is retained as the timing pilot; a twofold timing allowance plus 30 seconds
for completion must fit the bounded runtime before proceeding. Fixed numerical
choices are eigenvalue relative tolerance `1854 * float64_epsilon`, zero
absolute eigenvalue tolerance, relative kernel Frobenius error at most
`1e-10` and maximum entry error at most `1e-9`. Discarded spectra remain
reported. These finite-kernel checks do not qualify the eventual temporally
expanded operator or complete capacity match. The ASTRA recipe uses the
qualified runtime, one CPU / 7 GiB and at most 30 minutes when bundled with
substantive work. Actual preparation and all 27 pending EP18 component checks
remain unrun.

No feature payload, EEG, image pixels or private routing was read in this
review. Enforced image delivery and development/audit separation remain
unresolved; no feature freeze, empirical fit or reader release is claimed.

## 2026-10-06 21:21 Pacific — baseline_search exact capacity component prepared

[`slot_capacity.py`](code/slot_capacity.py) now implements both named EDF
curves on the thirteen original penalties using an augmented-base SVD and
a Schur-root SVD. It supports rank-deficient unpenalized nuisance columns
with an explicit caller rank tolerance, lists all admissible five-percent
capacity pairs and selects no penalty. The [method derivation](paper_plan.md#october-6-pacific-added-slot-capacity-definition-and-implementation)
records the operative term-EDF definition, incremental-model diagnostic,
common-base requirements and dense storage/runtime limitation.

Coordinator static review found no blocking algebraic issue. Five focused
checks in `code/test_slot_capacity.py` and changed ASTRA validation are
**unrun**. Add that named unittest module, under a 60-second timeout with
the qualified runtime, to the next substantive compute opportunity alongside
the eight transform and ten affected sequence checks already recorded below.
No separate allocation, data read, capacity-qualified empirical model or
complete EEG pipeline is established. RBF coefficient geometry, production
resource qualification, penalty selection and enforced delivery remain open.

## 2026-10-06 21:17 Pacific — baseline_search explicit added-slot capacity definition

Before any EEG fit, the coordinator bound the policy's **added-slot** effective
degrees of freedom to the trace of that term's influence in the joint weighted
ridge model. This operationalizes the existing term-level comparison; the
thirteen registered penalties and maximum 5% label-slot excess are unchanged.
The [official mgcv method documentation](https://stat.ethz.ch/R-manual/R-patched/library/mgcv/html/summary.gam.html)
defines term EDF through its influence trace. The policy now states the
definition and separately requires reporting incremental full-model EDF.

The distinction matters with penalized shared predictors. For the algebraic
one-row example `X_base = X_slot = [1]`, both penalties one, base EDF is 1/2,
joint EDF 2/3 and added-term EDF 1/3; incremental full-model EDF is only 1/6.
Neither quantity may silently stand in for the other. This is a derived
numerical example, not an EEG result. The planned exact routine will report
both on the original penalty grid, preserving common base predictors, weights
and penalties across compared slots and selecting no penalty from outcomes.
An explicit augmented-design SVD tolerance handles identified nuisance space;
full empirical capacity matching and RBF feature construction remain unfinished.

A bounded read-only BR-KG search for `effective degrees of freedom ridge
additive model` returned four unrelated task/concept/dataset labels, supplying
no method evidence or review verdict. The implementation decision uses the
primary documentation and independent algebraic review, not those search hits.
No BR canonical state, compute reservation or scientific data access changed.

## 2026-10-06 21:08 Pacific — baseline_search training transforms reviewed; compute pending

The resumed workers and coordinator independently reviewed
[`training_transforms.py`](code/training_transforms.py) against the frozen
feature registry. It implements training-only population scaling, full SVD
with the registered sign and rank conventions, retention of zero-rank feature
families, a fixed 62-to-61 channel projection, and explicitly supplied
training-residual covariance shrinkage. The sequence model now accepts a
named zero-column feature family when other predictors remain. No data reader
or statistical shrinkage recipe is selected by these array-only interfaces.

The eight checks in `code/test_training_transforms.py`, the ten affected
`code/test_continuous_sequence_model.py` checks, and validation of the changed
ASTRA program remain **unexecuted for these changes**. The earlier ten-check
pass in 46845646 applies to the previous sequence implementation. Run the
two named unittest files serially using the qualified CPU runtime, each with
a 60-second timeout, and the pinned ASTRA validator inside the next suitable
substantive allocation. No separate short-check allocation or new reservation
is selected; no extra workload is added to meet a runtime target.

No scientific payload was opened. Completed public assets and all 1,854
concept-name embeddings remain preserved. Enforced image delivery and EEG
reader separation still prevent empirical fitting; this code review neither
retries the stopped namespace route nor changes those data roles.

## 2026-10-06 18:58 Pacific — baseline_search combined integration allocation completed and charged

Terminal Slurm accounting reports **46845646 COMPLETED 0:0**, one allocated
CPU, **98 elapsed seconds**, October 6 **18:55:01–18:56:39 Pacific**. The
attended outer shell exited normally; the `.0` step used 202,744 KiB peak RSS.
Charge **98 allocated CPU-seconds (0.0272222222 hours)** once to EP18 and
release the unused **1,702 seconds** of its 30-minute reservation. `.0` and
`.extern` are not additional charges; EP02/15/20 receive no duplicate cost.
No EP18 allocation remains live or reserved from this session.

All selected numerical checks passed on their first compute invocation:
EP02 six / 1.893 seconds, EP15 seven / 0.021 seconds, EP18 ten / 0.011 seconds,
EP20 five / 0.093 seconds. All four changed ASTRA programs passed schema and
semantic validation. Each episode retains its own evidence under
`outputs/runs/shared_integration_46845646/`. The work ended sooner than the
attended diagnosis allowance; the coordinator closed the allocation without
padding time or running additional tests. No scientific payload was opened.

These three recent EP18-owned allocations (46842156, 46843116, 46845646)
total **811 CPU-seconds (0.2252777778 hours)**. All earlier EP18 costs remain
preserved; this subtotal is not the lifetime episode spend. No successor
allocation or wider validation program is implied by these passes.

## 2026-10-06 Pacific — baseline_search sequence-local fitting checks pass

The coordinator reports all **ten** focused checks for
[`continuous_sequence_model.py`](code/continuous_sequence_model.py) **passed
in 0.011 seconds**, followed by successful changed ASTRA validation
(`astra_exit=0`). The serial grouped implementation session **46845646** ran
on **sh04-18n32**, with **one CPU / 7 GiB** and all worker/BLAS thread limits
one. The [retained session outputs](runs/shared_integration_46845646/)
record these results. Final allocation disposition and its single shared
charge are recorded separately by the coordinator.

The checks cover actual sub-sample event timing and overlap, the operator's
adjoint, physical-sequence boundaries, direct joint weighted ridge with
unpenalized nuisance, zero-weight sample exclusion, fitted-event ablation,
sequence/column and event-mask alignment, invalid penalties or empty fitting
support, refusal of unconverged fits, and rejection of complex input before
real conversion. These finite algebra fixtures establish the numerical
component's tested behavior; they are not a recovery cohort, empirical fit,
scientific effect or qualification of the full EEG pipeline.

No image, event, learned-feature or EEG payload was read by these checks.
Completed runtime/assets and all 1,854 label features remain preserved. The
future caller still supplies the selected temporal/filter recipe, verified
event alignment, training-only feature transforms, sensor transform/whitening,
folds and penalties. Enforced image-worker delivery and EEG reader separation,
remaining image/caption features, false partitions and the complete evaluation
runner remain unfinished. Neither access seal nor a scientific release opens;
the stopped namespace route remains stopped.

## 2026-10-06 18:55 Pacific — baseline_search aggregated attended implementation session selected

Select the prepared continuous sequence fitting checks and changed ASTRA validation within
one coordinator-owned attended session aggregating EP02/15/18/20 integration,
diagnosis and any required focused repairs. Request **one CPU / 7 GiB /
30 minutes / zero GPUs** in the discovered development partition. Reuse the
qualified EP18 runtime and run each episode serially with all thread limits
one. The coordinator owns the shell; no independent agent submits or runs tests.
This replaces the provisional wait for a future EP21 batch without touching
those occupied workers. It is one combined implementation session, not four
short scheduler submissions or a simulation sweep.

Reserve at most **1,800 allocated CPU-seconds once to EP18**, then replace that
reservation with the actual terminal allocation charge; no duplicate cost is
assigned to EP02/15/20. Existing consumed budgets and results remain intact.
No empirical payloads, reader-release or scientific recipe change is selected.
Verify the actual compute shell before commands, preserve initial failures,
and exit the attended shell as soon as selected work is finished, without
padding runtime or adding an automatic successor.

## 2026-10-06 Pacific — baseline_search sequence-local fitting core prepared

**Historical preparation snapshot:** the checks described as unrun below
subsequently passed in the coordinated session recorded above.

Added `code/continuous_sequence_model.py` and ten focused algebra tests in
`code/test_continuous_sequence_model.py`, with ASTRA dependencies. The
matrix-free design evaluates basis functions at actual sample-minus-event
times without rounding, includes every overlapping event within its physical
sequence, and fits nuisance plus event columns together. Explicit column
penalties allow unpenalized nuisance; fixed sample weights preserve the caller's
mask. Multiple sensor contrasts are fit serially. The ablation operation
subtracts only the specified fitted added-slot event contribution over its
complete support without refitting or changing the sample mask.

Coordinator static review checked the operator/adjoint, direct weighted ridge
and ablation equations. A conversion issue was repaired before execution:
complex arrays now refuse before any cast could discard imaginary parts.
Ten focused checks and changed ASTRA validation remain **unrun**. The proposed
command is `timeout --signal=TERM --kill-after=5s 60s
$GROUP_HOME/users/zijiao/ep18_model_runtime_cpu241_py312_46842156/bin/python -B
outputs/code/test_continuous_sequence_model.py`, with the qualified module
environment, one CPU and all six thread limits set to one, inside the next
coordinated substantive allocation. No standalone checking job is selected.

This numerical component reads no files and supplies no access isolation,
preprocessing, sensor transform, temporal recipe, folds or tuning choice.
No EEG or image data were read, and no empirical fit, overall feature freeze
or reader release is claimed. Completed model/label artifacts are unchanged.

## 2026-10-06 18:42 Pacific — feature_preparation all label features complete and attended allocation closed

The [full-label report](runs/feature_preparation/all_name_text_46843116/label_feature_report.json)
records `all1854_label_text_features_completed`: all **1,854 by 512 float32**
vectors passed the runner's finite-value, unit-norm and row-alignment checks.
Total extraction time was **169.454814 seconds**, including **160.462050
seconds** of encoding; peak process RSS was **609,144 KiB**. There were zero
truncated prompts and zero provider-name fallbacks. The fixed batch size was
32. The matrix and aligned IDs/token metadata are retained beside the report.
The driver exited zero; its two focused release-gate tests passed.

The updated full-label ASTRA program passed schema and semantic validation:
[validator output](runs/feature_preparation/label_session_46843116/episode18_full_labels_astra.json)
and [serial check results](runs/feature_preparation/label_session_46843116/check_status.txt).
Both earlier dependency-path retries also passed. Original failed validation
attempts are retained. No image pixels, private routing, events or EEG were
read; this completes one feature family, not the overall feature freeze,
image-worker boundary, EEG audit seal or scientific comparison.

The coordinator exited the outer compute shell after the selected commands.
Terminal Slurm accounting reports **46843116 COMPLETED 0:0**, one CPU,
**366 elapsed seconds**, October 6 **18:35:22–18:41:28 Pacific**. Charge
**366 allocated CPU-seconds (0.1016666667 hours)** once to EP18 and release
the unused 834 seconds of its 1,200-second reservation. Do not add `.0` or
`.extern` as additional allocation charges; the `.0` step's measured peak RSS
was 1,307,488 KiB. This allocation also covered the EP20 document validation.
Together with provisioning 46842156's 347 seconds, these two completed
allocations cost **713 CPU-seconds (0.1980555556 hours)**; all earlier consumed
EP18 costs remain unchanged. No EP18 allocation remains reserved here.

Next work is enforced pixel/EEG delivery and the remaining feature/EEG
implementation under the existing contract. The stopped namespace route is
not retried and no automatic successor is launched.

## 2026-10-06 18:37 Pacific — feature_preparation name pilot passed and full label extraction released

Attended **46843116** is verified on `sh04-04n22`, with actual scheduler
allocation one CPU / 7 GiB / 20 minutes / zero GPUs. The three focused pilot
checks passed in 0.008 seconds. The
[first-64 report](runs/feature_preparation/name_text_pilot_46843116/name_text_pilot_report.json)
records completion in **16.344671 seconds**, including 4.902691 seconds to load
the text model and 5.631201 seconds to encode the 64 names. Peak RSS was
597,208 KiB. All 64 normalized float32 512-vectors are finite, no prompt was
truncated or required a provider-name fallback, and the four singleton checks
agree with their batched results (maximum absolute difference 1.49e-7).

The measured encoding-only extrapolation to 1,854 names is **163.128868
seconds**, not an observed full runtime. It fits the proposed 600-second
limit with the measured load and memory well within the existing allocation.
The coordinator therefore releases the already-defined **all-1,854 label-only
extraction**, unchanged batch size 32, prompts, pinned CLIP model and runtime,
under its explicit `--run-full-labels` invocation. Run two focused gate checks
first, retain the pilot, use a fresh `all_name_text_46843116` output directory,
and stop at 600 seconds without automatic continuation. This uses the current
attended session and adds no job or resource reservation.

Both earlier ASTRA retries now pass schema and semantic validation with the
diagnosed dependency-path correction; results are in
`runs/feature_preparation/label_session_46843116/*_astra_retry.json` and the
same directory's `check_status.txt`. Original missing-YAML failure logs remain.
Any new full-label program entry will receive its own changed-document check.
No image/caption worker, image pixels, EEG comparison, audit seal or overall
frozen feature record is released by the label-only step.

## 2026-10-06 18:31 Pacific — feature_preparation attended session 46843116 queued

The bounded foreground `srun --pty` request was accepted as **46843116** and
is waiting for resources. Requested resources are one CPU / 7 GiB / 20 minutes
/ zero GPUs, partitions `russpold,normal`. The coordinating agent owns the
PTY; actual compute hostname/allocation confirmation is pending. No test or
pilot command will run in a login shell. The selected tasks and maximum
reservation are recorded below; no supervisor or automatic renewal exists.

## 2026-10-06 18:30 Pacific — feature_preparation exact runtime and all public model assets qualified

Parent **46842156** is **COMPLETED 0:0**, one allocated CPU, **347 elapsed
seconds**: charge **0.0963888889 allocated CPU-hours** once to EP18 and release
the unused portion of its two-hour reservation. `.batch`/`.extern` are not
additional charges. Peak batch RSS was 4,445,572 KiB. Earlier consumed costs
remain preserved.

The fresh private environment imported Python 3.12.1, Torch 2.4.1+cpu,
Torchvision 0.19.1+cpu, Transformers 4.46.3 and every other registry version
exactly; `pip check` passed. All 2,142,873,276 declared asset bytes downloaded
through public HTTPS in about 30 seconds. The
[offline qualification](runs/feature_preparation/exact_cpu_provision_46842156/model_runtime/model_runtime_qualification.json)
verified source integrity for all five selected assets and loaded all six
selected model routes. Its status is `asset_runtime_qualification_completed`,
elapsed time 11.670318 seconds, peak RSS 1,299,076 KiB and zero forward passes.
No concept prompts, image pixels or EEG were read by this preparation.

The exact-version and asset-delivery blockers are resolved. The prior shared
alpha-module route and old environment remain preserved as failed history.
Pixel-worker and EEG audit boundaries remain unverified, so this completion
does not release those analyses. The already-selected first-64-name label-only
cost pilot may now proceed separately. All-1,854 text extraction still requires
that pilot's measured feasibility and coordinator release.

The missing-YAML ASTRA invocation remains a separate incomplete check. The
new qualified runtime contains PyYAML 6.0.3; a subsequent serial compute step
can give the unchanged pinned ASTRA executable its own package directory first
and this new environment's package directory second on `PYTHONPATH`. Preserve
the failed validation logs and write new results without claiming they passed.

## 2026-10-06 18:30 Pacific — feature_preparation attended label pilot and document validation selected

Select one bounded attended Slurm session, **one CPU / 7 GiB / 20 minutes /
zero GPUs**, for the existing first-64-name CLIP cost pilot and the two
diagnosed ASTRA-environment retries. The fixed pilot uses only the existing
label-name table and verified local CLIP assets, performs no EEG-driven
comparison, and emits runtime/memory/truncation evidence. It does not activate
full-label extraction or any image/EEG worker. Reserve at most one-third of an
additional allocated CPU-hour to EP18; close the shell and reconcile actual
terminal cost when the selected work ends. No unrelated running allocation is
suitable because all eight EP21 one-CPU workers are occupied.

## 2026-10-06 18:25 Pacific — feature_preparation provisioning running and shared checks complete

Job **46842156** started on `sh04-04n22`, with its actual shell confirming
one CPU / 7,168 MiB. The three transfer-context regressions passed in
**0.016 seconds**. The separately scoped EP09 alias reader returned zero and
EP20's eight coupled-circuit checks passed in **0.007 seconds**; each episode
retains its own evidence. These share the EP18 allocation charge once.

Both changed ASTRA documents remain **unvalidated**: their invocations failed
before validation with `ModuleNotFoundError: No module named 'yaml'`. This is
a missing dependency in the selected validator environment, not a reported
schema or semantic defect. Preserve both failure logs under
`runs/feature_preparation/exact_cpu_provision_46842156/`; retry only in a
properly configured serial compute opportunity. This optional tool failure
does not stop authorized provisioning.

At 01:25:44 UTC, `runtime_setup.log` reports successful downloads of the exact
Torch 2.4.1+cpu and Torchvision 0.19.1+cpu wheels and installation in progress.
Full runtime imports, model acquisition and offline loading are not yet
complete. No source-image pixels, concept prompts, EEG or inference were
released. The full two-hour reservation remains outstanding until terminal
Slurm accounting is observed.

## 2026-10-06 18:19 Pacific — feature_preparation provisioning job 46842156 submitted

Slurm accepted **46842156**, one CPU / 7 GiB / two hours / zero GPUs,
partitions `russpold,normal`, no requeue. Reserve 7,200 CPU-seconds once to
EP18; startup and all selected stage results are pending observation.
The exact run directory is `runs/feature_preparation/exact_cpu_provision_46842156`;
outer logs are `runs/feature_preparation/provision_exact_cpu_001-46842156.out`
and `.err`. No duplicate EP18 job existed at the prelaunch queue observation.
The eight EP21 workers remain running and were not modified.

## 2026-10-06 18:19 Pacific — feature_preparation exact CPU provisioning selected

Select one bounded outcome-blind provisioning job under the scientist's
standing resume/repair instruction: **one CPU, 7 GiB, two hours, zero GPUs**.
This is the concrete supported replacement for two diagnosed operational
problems. Sherlock's DTN has no interactive login, so the old DTN-shell
downloader could not execute; the selected model URLs are public HTTPS
endpoints and need no account credentials. The advertised shared Torch/vision
modules imported alpha builds, so preserve that stopped route and its existing
environment while installing the unchanged exact releases in a fresh private
GroupHome environment. No shared software or access restriction is altered.

The fixed `code/provision_exact_cpu_runtime_001.sbatch` requests no requeue.
It installs official CPU Torch 2.4.1 / Torchvision 0.19.1 and the unchanged
registry dependencies with a 2,400-second cap, acquires only the existing
2,142,873,276-byte public asset list with a 1,800-second cap, then runs the
existing sequential offline integrity/layer loader for at most 1,200 seconds.
Pip and asset retries are disabled; existing outputs and partial failures are
preserved. Temporary/cache files use node-local job scratch, the fresh runtime
uses GroupHome, and durable assets remain at the selected episode cache.
Module discovery was reused and quota inspection found adequate space.

Three narrow transfer-context regressions run before acquisition. Separately
scoped EP09 provider-identity metadata and EP20 coupled-circuit checks run
serially in this same substantive allocation; both changed ASTRA programs
are validated. Their cost is charged once to this EP18 parent, never again
to the other episodes. Reserve at most **two additional allocated CPU-hours**
against EP18's existing envelope; reconcile to actual terminal cost.

No source-image pixels, concept prompts, EEG, feature inference or optional
text pilot is selected. The stopped namespace route and unverified pixel/EEG
isolation remain separate prerequisites. This mechanical correction preserves
all scientific versions, roles, comparisons, resource caps and prior costs.
Submission and observed results follow separately.

## 2026-10-06 — completed input and feature preparation reconciled into readiness

Updated DATASETS and GOAL readiness from existing completed source records:
one-based sample handling, 1,133,808 preserved events, fixed 30/16 roles,
62-channel order, complete human-feature/private-routing metadata and the
qualified EP17 overlap. Exact official runtime distribution availability is
distinguished from the stopped unqualified module route and unacquired model
assets. Physical pixel/EEG isolation, false-group qualification and scientific
fitting remain unfinished. This was documentation-only; no role, version,
access seal, source payload, test, job or scientific rule changed.

## 2026-10-06 Pacific — official runtime recipes and pinned asset metadata recorded

Added the [verified public-source findings](paper_plan.md#october-6-public-source-update--exact-release-recipes-exist-runtime-untested)
to the existing paper plan and navigation. PyTorch's official release page
documents EP18's exact 2.4.1/0.19.1 pairing for CPU and named CUDA variants.
That establishes a public distribution route, not local runtime qualification
or permission to restart the stopped shared-module route. The existing pinned
CLIP configuration endpoint returned metadata; BLIP/DINOv2 metadata requests
did not yield usable content and therefore establish neither presence nor
absence. No checkpoint was downloaded, validated or loaded.

All selected versions, model revisions, transfer route, role/seal requirements,
budgets and scientific limits are unchanged. This source-documentation task
used already verified URLs and bounded local text reads/edits on sh04-ln04
without allocation: no install, import, test, payload or closed-outcome access,
scheduler call, authenticated transfer, external contact or job occurred.

## 2026-10-06 10:57 Pacific (PDT) — supplied review reconciled; execution-route alternatives unadopted

Revised the existing paper plan from the current asset map, feature registry,
access policy and preserved failure records. The EP17/EP18 overlap decision
is complete. Model-asset transfer, actual runtime qualification, pixel-worker
delivery and EEG audit separation remain distinct unresolved dependencies.

The review's proposed 2.1-GB compute-node download and acceptance of another
episode's alpha build are not adopted. Retain the DTN/Globus route and exact
EP18 release recipe unless a concrete replacement is selected within its own
scope. Namespace `ENOSPC` is the observed failure, not a diagnosed identity
with Codex's expected sandbox limitation. Appropriately configured supported
runtime isolation is possible in principle; no container is accepted merely
by name and none is ruled out categorically. A procedural boundary is a
possible explicit amendment, not current permission for EEG comparisons.

Documentation-only work on sh04-ln04 without a Slurm allocation. No runtime
probe, import, test, scheduler action, transfer, payload access, contact or
contract change. Prior failures remain preserved; no new job is selected.

## 2026-10-05 15:08 UTC — shared preparation allocation closed and charged once

Analysis `feature_preparation`. Attended46624388/PTY2853 closed at15:07:06UTC,
COMPLETED exit0. Slurm records1148seconds and2 parent allocated CPUs, despite
the one-CPU application step/request. Total0.6377777777777778 allocated
core-hours is split once across EP01/EP15/EP18; charge this episode
0.2125925925925926. Both preserved runtime attempts are included, not extra
charges. The shared shell is closed; no imports may fall back to login.
The selected module route remains stopped on exact-release mismatch, and
authenticated DTN asset transfer remains pending. No inference has run.

## 2026-10-05 15:04 UTC — feature_preparation: regex repaired; module runtime route stopped

The project-only regex correction installed successfully. Its distribution
metadata is2024.7.24, code/internal version2.5.146, both within the selected
episode venv. Preserve it and the previous six wheels. The repeated qualifier
then refused the Torch version. One bounded actual-import observation explains
the mismatch: `py-pytorch/2.4.1_py312` reports `torch.__version__=2.4.0a0`,
and `py-torchvision/0.19.1_py312` reports `0.19.1a0`, each from its advertised
module path. Transformers4.46.3, Pillow10.2.0, NumPy1.26.3, SciPy1.12.0 and
scikit-image0.24.0 match their selected versions. Evidence is the failed
repair log and `observed_module_versions.json` in
`runs/feature_preparation/20261005-runtime-regex-repair-46624388`.

The affected module-based runtime route is stopped. The coordinator has told
the user the observed release-label/import-version discrepancy; no alpha
version is silently accepted, shared module changed, broader infrastructure
probe or replacement workaround attempted. No model loading or inference ran.
Supported exact-version build availability or provider clarification remains
required; checkpoint transfer is a separate pending authenticated-user action.

Optional user-reviewed support draft (not sent): On2026-10-05 around15:02UTC,
job46624388 onsh04-18n31.int, loading py-pytorch/2.4.1_py312 and
py-torchvision/0.19.1_py312 produced import versions2.4.0a0 and0.19.1a0.
The experiment requires releases2.4.1 and0.19.1. Could you clarify these
build identities or identify supported matching builds? Full module-origin
paths and observed versions are in the linked project report above.

## 2026-10-05 15:00 UTC — feature_preparation: missing project regex dependency diagnosed

The failed call is `importlib.metadata.version('regex')` inside Transformers,
not a model or numerical failure. The project recipe wrongly treated module
discovery as evidence that this application dependency was qualified. Available
evidence does not establish a shared-system fault; no shared installation is
probed or changed. Stop relying on that unqualified regex module and preserve
the six successfully installed wheels.

Select the minimal application correction: one official
[regex2024.7.24 wheel](https://pypi.org/project/regex/2024.7.24/), CPython3.12
manylinux2014 x86_64,790,854 bytes, source SHA256
`d55588cba7553f0b6ec33130bc3e114b355570b45785cebdc9daed8c637dd440`.
Install it with `--no-deps` only into the existing episode environment, load
the same qualified module recipe without py-regex, and repeat the previously
failed version/import check in46624388,1CPU/8GiB,180s bound plus10s grace.
Verify regex code and distribution metadata both resolve inside that venv;
retain its distinct internal version string. Fresh report directory:
`runs/feature_preparation/20261005-runtime-regex-repair-46624388`.
No model loading, inference or scientific payload is selected by this repair.

## 2026-10-05 14:58 UTC — feature_preparation: six wheels installed; import check fails

The fresh selected GroupHome environment was created in attended46624388,
and all six exact official wheels (13,637,089 bytes) installed successfully
without dependency expansion. The subsequent version-import check exited1
because Transformers could not find `regex` distribution metadata, despite
the selected module-loading recipe. The overall runtime is not qualified.
Evidence: `runs/feature_preparation/20261005-runtime-environment-46624388/`
contains execution, setup output and stage exit. No model asset was loaded,
no inference ran, and no image/EEG/concept payload was opened.

Preserve this installation and failed attempt. A bounded diagnosis of the
project's module/runtime recipe is assigned before any retry; no shared
software changes, blind reinstall or infrastructure workaround is selected.
The changed ASTRA program passes schema and semantic validation in the same
allocation. The actual model assets remain blocked on authenticated DTN
transfer; the independently reviewed finite transfer script and exact command
have been supplied to the user. No authentication retry occurred.

## 2026-10-05 14:50 UTC — feature_preparation: selected runtime and official asset preparation

The registry fixes Torch2.4.1, Torchvision0.19.1 and Transformers4.46.3.
Module discovery supplies the exact first two and frozen low-level stack;
Transformers4.46.3 is absent. Available older tokenizer/hub modules are
incompatible, so no version substitution is selected. Official HF revision
metadata and PyTorch/DINO source metadata identify the five exact assets and
required local config/tokenizer files, totaling2,142,873,276 bytes in
`config/model_runtime_asset_map_proposal.json`. No checkpoint has yet been
acquired or loaded by this preparation.

Select a fresh environment at
`$GROUP_HOME/users/zijiao/ep18_model_runtime_py312`, retaining site modules for
exact Torch/vision, NumPy1.26.3, SciPy1.12.0, scikit-image0.24.0,
Pillow10.2.0, safetensors0.4.2 and regex. Additional exact wheels are
Transformers4.46.3, tokenizers0.20.3, huggingface-hub0.26.2, ftfy6.3.1,
wcwidth0.2.13 and tqdm4.66.5. Module metadata supplies remaining dependencies.
Install only these six packages with `--no-deps`; actual imports/version
checks run in attended46624388, one CPU/8GiB, with600s setup bound. Preserve
module paths and load plain Pillow last. Caches/temp files use task scratch
or node-local storage; never overwrite an existing environment.

Select the previously unspecified optional text cleaner prospectively as
ftfy6.3.1 for both concept and caption text, with wcwidth0.2.13. Official
[CLIP requirements](https://raw.githubusercontent.com/openai/CLIP/main/requirements.txt)
use ftfy; the pinned
[Transformers tokenizer](https://raw.githubusercontent.com/huggingface/transformers/v4.46.3/src/transformers/models/clip/tokenization_clip.py)
otherwise changes to BasicTokenizer. Reject that fallback rather than let
ambient installation determine the recipe. No inference/outcome informs this
choice; the main frozen feature-family registry remains unchanged.

The asset-only sequential CPU loader is under review,1200s internal and
proposed1CPU/8GiB/30m envelope after acquisition. It has no forward pass,
image/EEG/concept access or isolation claim. A first64 sorted-name text pilot
still requires successful actual loading.

At14:47UTC ordinary batch-mode SSH to `dtn.sherlock.stanford.edu` returned
`Permission denied (gssapi-keyex,gssapi-with-mic,password,keyboard-interactive)`
and exit255 before remote payload transfer. That authentication route is
stopped; no alternate identity, stored factor or login-node bulk download is
attempted. A bounded transfer handoff is being prepared for the user's
authenticated DTN session. Independent software setup and EP01/EP15 proceed.

2026-10-02 — Concept figure: generated/edited and visually inspected [ep18_conceptual_question-v2.png](ep18_conceptual_question-v2.png) with built-in image generation; exact prompts in [ep18_conceptual_question-v2-prompt.md](ep18_conceptual_question-v2-prompt.md). Edited presentation-rate and model-basis labels, removed illustrative primary winner bars, and qualified the remaining alternative-pattern sketches and fitted ablation interpretation. Earlier assets retained; no analysis, outcome access or scientific-rule change.

2026-10-02 — [Scope/novelty review (consolidated)](paper_plan.md): Reused the source-backed Holm/Kim/Rong/Watson comparison. Reframed the opening as categorical versus continuous sharing; distinguished post-fit predictive importance from unique/causal contribution and equivalence from an unestablished advantage. No EEG or audit outcomes were opened.

## 2026-09-27 — scientific design revision

- Reframed the episode around held-out exemplars and a strong image-and-label
  reference.
- Added a visual-distance prediction that distinguishes concept stability
  from residual visual similarity.
- Replaced ambiguous event-bin scoring with a full-sequence, post-fit
  distant-event contribution ablation and fixed its simultaneous audit test.
- Generated and scientifically reviewed the synthetic conceptual figure with
  the built-in image-generation tool.
- No EEG values, candidate scores, audit data, or protected outcomes were
  accessed.

## 2026-10-03 — input_role_preparation: outcome-blind source qualification

The scientist authorized resuming feasible work and completing episode jobs;
the coordinator separately assigned EP18 preparation. The current contract
explicitly permits outcome-blind setup before the separate-evaluator boundary
exists. It prohibits EEG-driven candidate comparison until that boundary is
verified. No EEG/image payload, audit outcome, sibling result or role assignment
was opened during preparation.

Read the exact OpenNeuro preservation `SOURCE.md`, provider README, participant
and event dictionaries, one text BrainVision header/marker prefix, and the
steward metadata/source receipts. The public release excludes identifiable
health information and uses study IDs; the permitted age/gender/language fields
are used only for the contract's assignment purpose. Existing acquisition
verification is reused. The September 22 image receipt records explicit
authorization for noncommercial academic use within this project without
redistribution or modification; no fresh acceptance is required for this scope.
THINGSplus-CC0 remains a separate control, not an image-join substitute.

Selected attempt: `outputs/runs/input_role_preparation/20261003-metadata01`,
using [qualify_structural_inputs.py](code/qualify_structural_inputs.py), one CPU,
2 GiB RAM, at most ten minutes inside the coordinator's existing allocation.
The complete per-event working join goes to
`$SCRATCH/br_autoresearch/episode18_things_eeg_concept_stability/20261003-metadata01/`;
only the compact qualification report belongs in the episode outputs.

The reader uses `participants.tsv`, the 46 non-provider-excluded event TSVs,
text `.vhdr`/`.vmrk` files, the pinned `image-paths.csv`, and ZIP central-directory
filenames. It **stats but never opens `.eeg`**, never opens a ZIP image member,
and does not use behavioral correctness, RT or response columns. It checks
the declared 12 blocks/72 main sequences, 1,854 concepts × 12 distinct images,
200 repeated images × 12, recording bounds, explicit duration conversion,
exact source-filename joins and common recorded channels. It records timing
diagnostics without silently changing source conventions. BrainVision's
one-based marker convention follows the
[manufacturer's format explanation](https://pressrelease.brainproducts.com/markers/).
No taxonomy, feature family, split-optimization substitute or neural-quality
exclusion is selected. Any failed integrity check is reported for repair and
does not silently remove a participant. Job ID and actual results are pending.

After structural qualification, the exact-global participant assignment must
follow the fixed minimax/squared-imbalance/lexicographic objective, four audit
members in each acquisition-order quartile, and one of sub-49/sub-50 per role.
The complete event/image join, role views and common 62-channel/61-dimensional
transform then need freezing. Remaining scientific work includes named feature
and taxonomy records, all eight accepted EEG-blind false partitions, real-design
overlap/recovery simulations, the outcome-blind cost pilot, and the EP17/EP18
exposure record. No same-account ACL will be described as true audit blinding.

Observed **46425841**, `sh03-06n50.int`, at 19:22:45 UTC:
[metadata qualification](runs/input_role_preparation/20261003-metadata01/structural_input_report.json)
completed in **52.165662 s** for all 46 potentially eligible participants.
The exact four provider exclusions match the contract. Every participant
passed main/repeated event structure, recording bounds, source duration and
image-filename joins; the metadata and archive each expose 26,107 image keys,
and the common channel inventory has exactly 62 labels. No EEG/image values
were opened. The sole failed check was the reader's initial assumption that
TSV `sample` was zero-based. The retained report contains offset-0 and offset-1
marker match counts; the observed examples match all 24,648 markers at offset
zero and none at offset one, with `onset*1000` equal to source `sample` up to
floating representation. Source `time_stimon` has a different clock origin
and millisecond variation. This is an index-convention diagnosis, not participant
exclusion or scientific failure. Official ASTRA schema/semantic validation
passed in [validation_20261003_metadata.json](astra/v0.0.14/validation_20261003_metadata.json).

Selected repair attempt:
`outputs/runs/input_role_preparation/20261003-sample_origin01`, using
`normalize_event_indices.py`, one CPU, 2 GiB, five minutes. It first requires
the same exact offset-zero diagnosis in **all 46** recorded reports, reuses
the passing structural checks, and preserves the original sample/onset while
adding `sample_zero_based = sample - 1` and its recording-clock onset. The
normalized working join goes to the corresponding new Scratch attempt. Only
source event text and the previous metadata join are reread; no EEG/image
values, image archive or unchanged file checks are reopened. Display-clock
diagnostics remain explicit and unadjudicated, and neither roles nor the join
is declared frozen by this repair. The original attempt remains unchanged.

Observed repair on **46425841**:
[sample-origin report](runs/input_role_preparation/20261003-sample_origin01/sample_origin_report.json)
completed in **33.66247 s**, preserving and normalizing **1,133,808 events**
across all 46 participants (24,648 each). The one-based convention is confirmed
throughout. Current ASTRA schema/semantic validation also passed in
[validation_20261003_sample_origin.json](astra/v0.0.14/validation_20261003_sample_origin.json).
Two display-versus-recording clock ranges require explanation: sub-36 spans
134.237 s and sub-38 spans 190.295 s; other participants' spans are approximately
0.001–0.013 s. Sample/marker agreement alone does not verify display timing.

Selected next metadata diagnosis:
`outputs/runs/input_role_preparation/20261003-clock01`, using
`diagnose_clock_segments.py --participants sub-36 sub-38`, one CPU, 2 GiB,
five minutes in the coordinator allocation. It reads only their text event and
marker records; matches offset changes against BrainVision `New Segment`
positions, physical-sequence and block boundaries; and reports within-segment
clock spans and relative recording-segment timestamp intervals. Actual calendar
timestamps are not emitted. The 100-ms nominal RSVP interval is used only to
describe large clock discontinuities, never to remove events or participants.
All 46 participants and the normalized join remain intact. No display-clock
tolerance or semantic trial exclusion is selected by this diagnosis.

Selected metadata-only assignment profile:
`outputs/runs/input_role_preparation/20261003-role_profile01`, using
`profile_role_metadata.py`, one CPU, 256 MiB, one minute. It uses only the
46 fixed eligible IDs, age, gender, native-English status, language-profile
category and home-language metadata, with definitions from `participants.json`.
Proposed categorical encoding retains all recorded nominal levels and an
explicit missing category, without an invented language taxonomy or participant
exclusion. Age remains continuous and its exact rational resolution is reported;
missing age is reported without imputation. Age scale is sample SD with divisor
N−1. Categorical scale is pooled Bernoulli SD `sqrt(p*(1-p))`, with p from the
full 46-person cohort. For each nonconstant indicator or age feature, imbalance
is `(audit mean - development mean)/scale`. The fixed acquisition quartiles
also enter this objective. Only zero-variance fields are dropped, per policy.
The profile does not freeze the encoding, choose roles or relax exact global
optimization. It supplies the actual finite levels and age resolution needed
for the exact minimax, squared-imbalance and lexicographic optimization stages.

Observed clock diagnosis on **46425841**, 19:28:52 UTC:
[clock_segment_report.json](runs/input_role_preparation/20261003-clock01/clock_segment_report.json)
links both large offsets to BrainVision `New Segment` markers and existing
physical-sequence boundaries. Sub-36 changes between main sequence 5 and 6
(block 0 to 1); the segment timestamp interval exceeds recorded sample elapsed
time by **134.238813 s**. Sub-38 changes between sequence 39 and 40 within
block 6, with **190.279787 s** of timestamp-versus-recorded-time difference.
This diagnoses recording-pause/segment semantics, not an arbitrary event shift
or missing-trial exclusion. Maximum display/recording offset span within a
physical-sequence/segment is **1.01536 ms** and **1.02992 ms**, respectively.
Both participants retain all 24,648 events. The existing hard physical-sequence
boundaries already contain these recording breaks; no sequence is bridged,
event deleted or participant reclassified. Use normalized marker-based sample
indices for recording joins and retain display-clock fields as their original
clock. This does not establish an independent absolute display-timing accuracy
bound or authorize neural outcome access.

Observed [assignment-field profile](runs/input_role_preparation/20261003-role_profile01/role_metadata_profile.json)
on **46425841**, 19:31:03 UTC: all 46 participants have complete required
fields; ages are integers 17–28 with integer lattice step one. Gender/native
English/language profile/home language have 2/2/3/5 nominal levels. Source
language profile is explicitly Bilingual/Monolingual/Trilingual; it is not
treated as a numerical ordinal. The exact combined home-language label is
retained, not split or recategorized. No imputation is needed.

Before optimization, fixed the justified within-contract encoding in
[role_encoding.json](config/role_encoding.json): all nominal levels, continuous
age, explicit pooled Bernoulli and sample-age variances, all four quartile
indicators, and retention of distinct registered columns even if identical.
Dropping a reference category or deduplicating columns would change the squared
imbalance objective and is not done. An independent agent reviewed the exact
formulation against policy and this metadata profile; this was a mathematical
review, not an executed solver or scientific outcome review.

Selected next attempt:
`outputs/runs/input_role_preparation/20261003-role_proposal01`, one CPU, 2 GiB,
five minutes (240-second internal optimizer bound) for the full 46-person
metadata-only optimizer in the remaining allocation. Before it, run
the focused `test_exact_role_assignment.py` fixture on compute: compare all
three objective tiers against exhaustive feasible small-cohort enumeration,
and check retained categorical complements/nonzero quartile terms.
`exact_role_assignment.py` compares objective values as exact fractions.
It binary-searches the complete finite minimax-squared grid with integer MILP
feasibility, enumerates all aggregate count/sum states inside that bound in
exact squared-imbalance order, retains every feasible tied optimum, and prefers
inclusion of each ascending ID only while some tied optimum remains feasible.
Squaring the nonnegative minimax preserves its order. Integer constraints and
final rational objectives are verified exactly for every accepted witness.
HiGHS infeasibility, feasible witnesses, solver status/gap and objective bounds
are retained; a time limit or unknown result cannot become an infeasibility
certificate. There is no candidate cap or weighted-objective approximation.
No proposal is emitted unless all three tiers complete. The five-minute wall
bound is a resource bound only. Output is a **role proposal**, not an audit seal
or authorization to access EEG, pending exposure/access review.
The solver explicitly forwards HiGHS `threads=1`, in addition to the
coordinator's one-thread BLAS/OpenMP environment. SciPy may emit its expected
warning that this option is forwarded to HiGHS; it is intentional.

Observed **46425841**, 19:36:11 UTC: both focused
[exhaustive/encoding tests](runs/input_role_preparation/20261003-role_solver_tests01/tests.log)
passed in **0.020 s**. The complete 46-person
[exact role proposal](runs/input_role_preparation/20261003-role_proposal01/role_proposal.json)
completed in **0.087851 s**, using **53 MILP calls**, **88 aggregate states**
within the minimax bound and one feasible optimum aggregate. Minimax squared
imbalance is exactly **529/10125**, with preceding infeasible level
**34983/762880**. The second-tier objective and its equal lower/upper bounds are
**635707579471/2527731360000**. The exact lexicographic audit proposal is
`sub-02, sub-03, sub-04, sub-05, sub-15, sub-16, sub-17, sub-19, sub-33, sub-36,
sub-37, sub-39, sub-40, sub-41, sub-42, sub-50`; the other 30 eligible IDs are
the development proposal. It keeps four audit people in every acquisition
quartile and exactly one of sub-49/sub-50. Zero-variance drops were unnecessary.
The expected `threads` forwarding warning is retained in the logs; it is not
a solver failure. Current
[official ASTRA validation](astra/v0.0.14/validation_20261003_role_proposal.json)
passed schema and semantics.

The independent reviewer subsequently inspected the actual solver source and
confirmed complete minimax/aggregate coverage, complete objective ties, exact
integer witness checks, failure on unknown/time-limited solves, and lexicographic
checking across all tied states. No code repair or rerun was indicated. This
review did not execute code or inspect EEG/image values. **Participant roles
remain proposals**, not frozen access assignments. The EP17/EP18 exposure record,
role-filtered views and a genuinely separate evaluator/mount remain outstanding;
same-account ACLs are not claimed to provide that boundary. Subsequent image-only
feature preparation remains outcome-blind; EEG-driven candidate comparison and
the audit remain closed under the existing contract.

Selected next outcome-blind attempt, before any execution side effect:
`outputs/runs/input_role_preparation/20261003-freeze_metadata01`, using
`outputs/code/freeze_metadata_views.py`. This is a new bounded compute task
after the coordinator closed allocation 46425841, not an allocation renewal.
Request one CPU, 2 GiB and ten minutes; expected runtime is under two minutes
for a single streaming pass over the existing 1,133,808-row compressed metadata
join and the pinned CNeuroMod public stimulus CSV. Aggregate with other short
authorized work if available. No EEG, image, checkpoint or sibling payload is
opened. Preserve both prior source-normalization attempts and the role proposal.

The initialized contract-required
[EP17/EP18 exposure record](config/ep17_ep18_exposure_record.json) documents
reviewed first-access history, source-image nonidentity and official checkpoint
reuse candidates. The September central-directory receipt maps all 8,640
CNeuroMod archive image identities to THINGS, but all differ in byte size and/or
CRC. The next pass records exact intersections with EP18's main and repeated
presented-image/concept inventories. These are source-inventory intersections;
EP17's eventual common analysis universe remains unqualified, and no EP17 live
cache or feature output may substitute for independently authorized sources.
EP17 and EP18 remain related, design-exposed THINGS studies, not independent
stimulus-family confirmation. The history record states its reviewed scope
instead of claiming exhaustive knowledge of all past human/session exposure.

Reviewed eligibility, normalized marker indices, contained recording breaks
and the exact role solution support freezing the deterministic assignment and
join under the existing contract. No separate human choice is required for
this already prescribed assignment. The selected pass preserves every event
and original column in two disjoint durable compressed metadata views; their
union is the frozen join. It also writes separate participant/file manifests
without copying or linking raw payloads. Freeze completion is asserted only by
the final `frozen_inputs_and_roles.json`, emitted after count/identity checks.
It explicitly leaves the audit seal and EEG-comparison authorization false.
The deployment requirement is concrete in [inputs/README.md](../inputs/README.md):
an externally controlled development identity/runtime can access only its own
inputs and cannot reach or redownload public audit EEG. No account/network
configuration changes or premature approval request are part of this work.

Before launch, independent metadata review identified additional catch/test
rows in the pinned CNeuroMod CSV: catches have blank THINGS indices and test
paths are flattened under `images_test_fmri`. Corrected the prepared overlap
reader to use explicit **one-based `things_image_nr`** into the already
qualified THINGS `image-paths.csv`, cross-checking filenames and experimental
folders. Small bounded source examples establish the indexing convention;
the compute pass verifies every mapped row. Blank catch mappings remain
unmapped image identities and never create a THINGS concept. The existing byte
nonidentity receipt covers only the 8,640 experimental archive entries.
Selected two focused synthetic identity checks before the actual pass:
`python -m unittest discover -s outputs/code -p test_source_overlap_identity.py`.
They cover exp/test/catch mapping and reject conflicting indices/partial
mappings. No test has run on login; unchanged role optimization is not rerun.

Observed completion on **46427768**, `sh02-01n18.int`,
**2026-10-03 19:57:23 UTC**: both focused
[identity checks](runs/input_role_preparation/20261003-freeze_metadata01-tests.log)
passed in **0.001 s**. The
[metadata freeze](runs/input_role_preparation/20261003-freeze_metadata01/frozen_inputs_and_roles.json)
completed in **49.539415 s**, preserving all **46 participants and 1,133,808
events**. The immutable role solution is now instantiated in durable disjoint
metadata joins: development has **30 people / 739,440 events**; audit has
**16 people / 394,368 events**. All original source fields and normalized
marker indices remain, and the common 62-channel order is retained. No EEG,
image or checkpoint payload was opened. The separate
[development](runs/input_role_preparation/20261003-freeze_metadata01/development_manifest.json)
and [audit](runs/input_role_preparation/20261003-freeze_metadata01/audit_manifest.json)
manifests list future controlled-mount inputs; they create neither symlinks nor
access controls. **Roles/join are frozen; audit seal and EEG-driven candidate
comparison authorization remain false.**

The [source-only overlap report](runs/input_role_preparation/20261003-freeze_metadata01/source_overlap.json)
contains all **8,840 CNeuroMod CSV image rows**: 8,640 experimental images from
720 concepts, 100 test images from 100 concepts and 100 catch images with no
THINGS mapping. Every mapped image index agrees with the preserved THINGS
filename; experimental folders also agree. EP18's 22,248 main images share
**8,640 image identities and 720 concepts** with the CNeuroMod experimental
inventory. EP18's 200 repeated images share **100 image identities**, all with
the CNeuroMod test inventory. All **200 repeated concepts** occur among the
CNeuroMod experimental concepts, but none of those repeated image identities
is an experimental image. Catch rows contribute neither matches nor invented
concepts. The September byte-nonidentity receipt applies to the 8,640
experimental archive entries, not to test/catch CSV rows. Exact EP17 analysis
overlap remains unresolved until its common source-event universe qualifies;
the source counts cannot substitute for that step. These studies remain
design-exposed and stimulus-related, not independent stimulus-family evidence.

The initialized [exposure record](config/ep17_ep18_exposure_record.json) now
links this completion while preserving prior acquisition/access history and
its limits. Official [ASTRA validation](astra/v0.0.14/validation_20261003_freeze_metadata.json)
passed schema and semantics. Independent code review confirmed the repaired
identity mapping, whole-person routing and false access-seal flags; it did not
execute tests or inspect EEG/image values.

The remaining deployment is the concrete evaluator-controlled runtime
described in [inputs/README.md](../inputs/README.md): stage only the development
manifest's 30 EEG filesets and development event view, deny access to all
mixed/audit preservation paths and host execution credentials/interfaces, and
enforce network restrictions preventing public audit redownload. The evaluator
retains the audit fileset for the registered one-shot gate. Verification must
run inside the actual worker context and establish both denied paths and denied
downloads without opening audit payloads. No such deployment is claimed here.
The contract does not require a new human approval or a second Unix account:
a trusted coordinator can control a suitably isolated development child,
provided that child cannot bypass the mount/network boundary or reach its
coordinator's host context. Feasibility with existing supported tools remains
to establish using harmless sentinel inputs on compute only.

Independent outcome-blind work remains authorized before that handoff:
construct the fixed 61-dimensional common-average sensor basis from the frozen
channel order; instantiate the full image/label feature registry and exact
human-metadata crosswalk without substituting sibling caches; prepare an
opaque-ID image-only input builder; then, under a separately selected compute
step, extract the required core/expanded visual features, freeze the visual
distance panel, build all eight matched false partitions and qualify the
real-event overlap simulations. Image-only workers must receive pixels and
opaque IDs only, without concept folders/names or event labels. The present
task identifies these next steps but launches none and opens no pixels or EEG.

Selected sentinel-only runtime feasibility, before execution:
`outputs/runs/input_role_preparation/20261003-runtime_sentinel01`, via
`code/check_isolated_runtime.py`, **1 CPU, 512 MiB, one minute** inside the
coordinator's current allocation 46427768. Lightweight module discovery found
the supported `system bubblewrap/0.11.2`; no installation, shared configuration
probe or namespace execution occurred on login. The
[upstream runtime documentation](https://github.com/containers/bubblewrap)
states that security depends on the chosen mounts/namespaces and warns about
host sockets and terminal injection; the selected wrapper uses isolated
user/PID/network/mount namespaces, no host sockets, `--new-session`, cleared
environment, dropped capabilities and disabled nested user namespaces.

The coordinator creates only fake development/audit text sentinels and a
temporary loopback listener inside Slurm. The child sees the fake development
directory read-only and Python/system libraries only. Checks cover denied host
sentinel paths/process-root routes/descriptors, namespace separation, no new
privileges/capabilities, denied write/remount/nested-user-namespace operations,
and inability to reach the coordinator's positively verified loopback listener.
It makes no external connection and mounts no real source data. Context-managed
listener/files and a 30-second subprocess timeout bound cleanup. A pass is
**runtime feasibility only**: the actual feature/EEG worker's reviewed mount
list, dependencies and controlled output area still need deployment and checks
before `verified_access_boundary` can be emitted. The audit seal stays false.

Independent review identified and repaired a launcher boundary before the
sentinel run: resolved interpreter prefixes must be `/usr` software or an
explicit module/version under `/share/software/user/open`, must not overlap
the supplied scratch tree, and must contain the selected executable. A
home/group/Oak/Scratch virtual environment is rejected **before** any child
mount or output-directory creation. The coordinator's selected Python3.12.1
module satisfies the intended software-only scope. No namespace weakening or
runtime installation is used if the supported configuration is unavailable.

Observed sentinel attempt01 on **46427768**, `sh02-01n18.int`,
**20:04:09 UTC**: [runtime feasibility](runs/input_role_preparation/20261003-runtime_sentinel01/runtime_feasibility.json)
failed **before the child launched**, in **0.048899 s**, with bubblewrap exit 1:
`Creating new namespace failed: nesting depth or /proc/sys/user/max_*_namespaces exceeded (ENOSPC)`.
The fake development file was unchanged; the coordinator's listener and all
temporary sentinels were cleaned up. There was no scientific payload read,
external connection or shared infrastructure change. This affected runtime
route is stopped: no sysctl/configuration probe, flag weakening, other-node
retry or workaround is selected. The coordinator informed the user and asked
whether an existing supported isolated evaluator/runtime is available.
The contract's deployment remains unresolved; the real audit seal stays false.
The failed runtime check does not undo the completed metadata/role freeze.

Concrete independent preparation remains available from the existing metadata
snapshot `things_metadata_osf_jum2f_20260922/payload`: join all 1,854 main concept
keys to `concepts-metadata_things.tsv` by `uniqueID`; profile the relevant
columns and missingness in `property-ratings.tsv`; and instantiate one declared
broad-category crosswalk from `category53_wide-format.tsv`. The source describes
that 53-category table as membership associations, while the 27-category files
represent different human/WordNet/manual schemes. Prefer the explicit 53-column
membership table for the initial registry, preserving its full membership
structure, and use aggregate property means with the current arousal field
instead of `deprecated_property_arousing_mean`. Do not include raw naming
responses, worker demographics or response-time summaries as model features.
If typicality is included, `typicality53_mean-ratings.tsv` is the corrected
task-v2 source; the `_deprecated` files mix flawed task-v1 data. These are
prospective registry choices and source checks, not EEG-based selection.
Exact feature columns, missing-value/scaling rules and taxonomy must be frozen
in the feature record before use; every learned transform remains training-only.

The next image-only work can use the already-frozen event/archive join and
accepted image terms to prepare a trusted opaque-ID builder, but must first
instantiate named core models (including vision-language and caption families)
and an independently trained expanded encoder with licenses, preprocessing,
layers/pooling and dimensions. The three documented ResNet checkpoints are
possible official-source assets only; they do not by themselves fill this
required panel or authorize EP17 feature-cache reuse. No image pixels, new
downloads, feature computation or additional jobs were opened by this closeout.

The [sentinel program validation](astra/v0.0.14/validation_20261003_runtime_sentinel.json)
passed the official schema and semantic checks. This validates the experiment
description, not the failed runtime boundary.

Selected next authorized source-only pass before execution:
`outputs/runs/feature_preparation/20261003-human_metadata01`, via
`code/prepare_human_metadata.py`, **1 CPU, 1 GiB, two minutes** in the remaining
coordinator allocation 46427768. It reads the first complete development
participant's frozen metadata inventory (the prior freeze already verified
inventory equality across all 46), the provider concept/aggregate property/
53-category tables, and [human_metadata_recipe.json](config/human_metadata_recipe.json).
All 1,854 concepts must join by exact `uniqueID`; no concept is dropped.
The selected source recipe retains the ten nondeprecated object-property
means, corrected arousal and perceived size (12 raw dimensions), plus all 53
broad-category memberships, including multiple memberships. The key and
provider name/context remain in a separate crosswalk, not numerical predictors.
No source responses, RTs, demographics or deprecated fields become features.
Missing values remain missing and are reported by exact concept/field; no
imputation or fitted scaling/PCA is selected silently. A complete feature-panel
freeze, concept-name embedding rule, licenses and training-only transforms are
still separate requirements. No EEG, image or checkpoint values are opened.

Observed [human metadata pass](runs/feature_preparation/20261003-human_metadata01/human_metadata_report.json)
on **46427768**, `sh02-01n18.int`, **20:09:17 UTC**, completed in
**2.632346 s**. All **1,854** frozen concepts match exactly and uniquely in all
three source tables, with no extra/missing source concepts and **no missing
values** in the 12 property/size dimensions or 53 category indicators. Thus no
imputation is needed for this pinned inventory. The untransformed feature table
and separate name/identity crosswalk are durable under that attempt. Multiple
memberships remain intact. **406 concepts have an all-zero 53-category row**:
this means no membership in those particular broad categories, not missing
numeric data or grounds for exclusion. All 406 remain in every image/concept
analysis and retain their property and later name-embedding inputs. Broad-
category influence summaries must disclose this nonexhaustive coverage.

The provider documents property/arousal means on rating scales and perceived
size on its own rating axis; no physical unit or shared scale is inferred from
observed ranges. Scaling/PCA remain unapplied and training-only. No complete
feature panel or concept-name embedding is yet frozen; no image, EEG or
checkpoint values were opened. The
[official ASTRA validator](astra/v0.0.14/validation_20261003_human_metadata.json)
passed both schema and semantics. The next selected work is a complete
prospective feature registry under the current policy, without downloads or
feature extraction and without changing the false audit-seal state.

## 2026-10-03 — feature_preparation: prospective complete panel

The [feature recipe](config/feature_registry.json) now selects the complete
panel using the existing policy's delegated pre-score choices. This is a
prospective recipe, not an emitted `frozen_feature_record` or a successful
feature-extraction run. No model, image or EEG payload was opened; no downloads,
tests or compute jobs were launched in this step. Coordinator allocation
46427768 ended at 20:13:27 UTC after the earlier selected metadata tasks.

| Required component | Selected representation | Raw dimensions |
| --- | --- | ---: |
| Core low-level color/frequency/texture/geometry | Fixed RGB moments/histograms, frequency power, Gabor texture, HOG and global geometry/contrast | 851 |
| Core supervised vision | Torchvision ResNet-50 ImageNet-1K V1, final average pool | 2,048 |
| Core self-supervised vision | DINO ResNet-50, final average pool | 2,048 |
| Core vision-language image | CLIP ViT-B/32 projected image embedding | 512 |
| Core automatic captions | Image-only BLIP base greedy caption, then CLIP text embedding | 512 |
| Additional expanded encoder | DINOv2 ViT-B/14, final CLS plus mean patch embedding | 1,536 |
| Concept-name labels | Same CLIP text encoder, fixed provider-context prompt | 512 |
| Human property and category labels | Current 12 property/size dimensions plus all 53 broad memberships | 12 + 53 |

The required families, retained core in the expanded panel, image-worker
input restrictions, nested transforms and rank/search budgets come directly
from policy. The particular models, layers, descriptors, deterministic caption
rule, name prompt, full-precision inference, source versions and scaling
conventions are now selected before scores; they are not scientist-only
approval points. DINOv2's ViT-B was distilled from its own ViT-g teacher and
has a distinct training lineage from the core DINO ResNet, rather than being
a second layer or transform of the same checkpoint. The expanded model adds
this encoder without removing any core family.

The record pins official CLIP, BLIP and DINOv2 revisions and exact official
ResNet checkpoint identities. It distinguishes software licenses from model
cards and dataset/weight notices. In particular, Torchvision's official notice
does not warrant relabeling every pretrained weight with the code license;
CLIP's inspected HF metadata lacks a separate checkpoint-license field. The
source terms must be documented for the actual assets before candidate scores,
as already required by policy. No new blanket human-approval requirement is
introduced. The three known EP17 ResNet assets alone would not satisfy this
panel. The exposure record now separates prospective DINO/supervised selection
from actual loading, which has not occurred in EP18.

Training-only mean/SD scaling and per-family PCA retain the policy's
32/64/128 rank requests with explicit algebraic dimension/rank caps for small
human families. No artificial padding, family deletion or held-out fitted
transform is introduced. The capacity-matched label kernel receives the same
continuous name/property/category vectors as the linear-label baseline, never
concept numbers. The selected training-only RBF reference is median positive
pairwise distance, with the policy's unchanged bandwidth multipliers and df
requirements. The explanatory visual panel is fixed to the three core
low-level/supervised/self-supervised families; its equal-weight normalized
cosine distances and equal-count tertiles are specified independently of EEG.
This does not narrow false-partition matching over the registered image
families. The shared CLIP space across image/name/caption representations and
unknown pretraining image overlap remain explicit limitations.

Next substantive preparation is implementation of those selected extractors
and the trusted opaque-ID builder, then qualification of the actual official
assets/runtime and the required outcome-blind cost pilot. The source name
crosswalk still needs its declared blank-context fallback and text-truncation
accounting when text features are generated; all numeric human entries are
already complete. Batch sizing and software discovery are routine engineering
choices, not opportunities to substitute models or expand the search after
scores. Continuous-model, false-partition and recovery implementations remain
separate substantive work under their existing contracts. The actual audit
seal remains false; no EEG-driven comparison may begin on the basis of these
metadata records. The affected bubblewrap runtime route remains stopped.

The ASTRA program now consumes this prospective recipe under its existing
`feature_preparation` ID, while `frozen_feature_record` remains a future
qualified output. The prior human-metadata validation predates this small
program edit; a current official validation can be grouped with the next
coordinator-owned compute step, without launching a separate checking job.

Independent source/recipe review found no substantive contract conflict in
the complete registry. A declaration typo was corrected before implementation:
the six unchanged low-level component sizes sum to **851**, not 951. A bounded
read of the preserved metadata `payload/LICENSE` also identifies its grant as
Attribution CC BY, without a version number in that notice; source-specific
exceptions and the original-image research terms remain distinct. No metadata
payload needed to be reopened for that license clarification.

Selected bounded implementation is now present in
[feature_extractors.py](code/feature_extractors.py) and
[prepare_feature_metadata.py](code/prepare_feature_metadata.py). Image routes
accept only ordinary decoded uint8 RGB arrays and opaque IDs. Model loading
uses explicit local paths with no implicit downloads or remote code. The
independent metadata adapter prepares private UUID routing for the trusted
builder and separate provider-context name prompts; that private crosswalk
must never enter an image worker. The adapters do not claim filesystem
isolation. Full feature inference has not run and no asset was loaded.

The low-level implementation uses FFT convolution with a symmetric padded
image to implement the selected half-sample-reflect Gabor boundary efficiently;
this retains the declared kernel/formula. A single focused
[constant-field regression](code/test_feature_formula.py) checks the corrected
851 dimensions, known color/histogram/geometry values, and zero spatial
frequency/texture/HOG for a constant image. It remains unrun and is intended
for the next substantive allocation, not a separate test-only job.

Before its future side effect, select the source-only mapping attempt
`outputs/runs/feature_preparation/20261003-mapping01`: **1 CPU, 1 GiB, 1 minute**.
It reads the existing frozen development join and human crosswalk only, retains
all 1,854 concepts and main/repeated image identities, and records name-context
fallback counts without tokenizing, opening models or decoding images/EEG.
The exact executable recipe is the existing `feature_preparation` analysis's
new `private_feature_metadata` output in ASTRA. Actual launch/job/result remain
unrecorded because the coordinator has no active allocation. Full selected
asset loading and cost measurement are separate later steps; there is no
implicit namespace retry or access-seal change.

Independent adapter review confirmed the CLIP projected-submodel API and
BLIP's unconditional one-BOS/SEP stopping semantics. It requested one concrete
additional numerical check: an asymmetric edge-impulse image compares the FFT
Gabor operator directly with `scipy.ndimage.convolve(mode="reflect")` at
float64 tolerance. The focused file now contains **two unrun tests**. Strict
DINO backbone loading remains in force pending actual checkpoint inspection;
it will not be relaxed merely because the upstream hub helper uses
`strict=False`. HF model loads reject missing or mismatched selected weights,
which otherwise could produce randomly initialized features without stopping.

Lightweight module discovery found existing scikit-image **0.24.0**, SciPy
**1.12.0**, NumPy **1.26.3** and plain Pillow **10.2.0** modules. The prospective
runtime recipe now selects these available versions before any features or
scores instead of requiring installations for the initially proposed numeric
patch versions. For the focused compute step, load
`devel math viz py-scikit-image/0.24.0_py312 py-pillow/10.2.0_py312`, with plain
Pillow last because scikit-image also loads Pillow-SIMD as a dependency. Record
actual imported versions/paths in the future compute step. The two tests need
**1 CPU, 2 GiB, under two minutes** as an estimate; BLAS/OpenMP threads stay at
one. The independent source-only mapping command remains stdlib-only. No
software was installed or imported for project execution on the login node.

Observed [private routing/name preparation](runs/feature_preparation/20261003-mapping01/feature_metadata_report.json)
on coordinator allocation **46429138**, `sh03-06n50.int`, at
**2026-10-03 20:34:17 UTC**: **22,248 main image identities**, **200 repeated
image identities**, **22,448 union IDs** and **1,854 concept-name prompts** were
preserved. Every prompt used the nonblank provider context; **zero** fallback
names were needed. The durable random UUID mapping is fixed by this attempt
and must not be regenerated for existing feature artifacts. Its source routing
table remains private to the trusted builder; it is not an image-worker input
or filesystem isolation. No EEG, source-image pixel or checkpoint value was
opened, actual worker delivery is unverified, and the audit seal stays false.

The coordinator reports both focused formula tests **passed in 1.543 s** in
the same verified allocation. Actual imports were NumPy **1.26.3**, SciPy
**1.12.0**, scikit-image **0.24.0** and plain Pillow **10.2.0**, with Pillow
resolved to
`/share/software/user/open/py-pillow/10.2.0_py312/lib/python3.12/site-packages/PIL/__init__.py`.
This checks the fixed low-level numerical formula and reflection implementation
on the stated fixtures; it is not actual THINGS feature extraction, neural
model loading, a complete model-runtime qualification or an access-boundary
test. No unchanged check is selected for rerun.

The [official current ASTRA validation](astra/v0.0.14/validation_20261003_feature_recipe.json)
passed schema and semantics. Remaining substantive work is qualification of
the actual selected local model assets and supported image-worker delivery,
then the required outcome-blind feature cost pilot/extraction. No completed
`frozen_feature_record` is asserted. EEG-driven comparisons still require the
independent development/audit separation; the stopped namespace route has not
been retried or weakened.

## 2026-10-04 — cross_episode_exposure: qualified universe and claim decision

At **23:53 UTC**, record the coordinator's completed metadata comparison and
adopted **correlated, design-exposed** choice. Both existing designs, EP17's
480/120/120 concept roles and EP18's 30/16 participant roles remain unchanged.
This resolves the earlier unknown EP17 analysis-universe overlap without
replacing the October 3 source-inventory comparison or changing eligibility.

The coordinator supplied the immutable, episode-local
[exposure handoff](runs/cross_episode_exposure/ep17_handoff001/exposure_report.json)
after the EP17 comparison completed with exit **0**, from
**23:52:26.433238 to 23:52:30.642990 UTC**, on job **46531931**,
`sh02-02n43.int`. Elapsed time was **4.209763 s**, peak RSS **37,684 KiB**,
and the report size **3,666,473 bytes**. The selected envelope was one CPU,
at most 1 GiB, 60 seconds internal and 90 seconds external; shared-allocation
accounting remains with the coordinator. Static coordinator and independent
reviews passed; no additional fixture suite or source-array run was needed.

The comparison used a coordinator-frozen EP18 metadata snapshot, EP17's
completed concept/image assignments and the two exact source metadata tables.
Provider `image_path` selected one-based `things_image_nr` in the preserved
THINGS path list. Folder, filename, category, concept and inherited role/fold/
block identities agreed. No filename approximation, pixel equality test or
live sibling feature/cache input was used. Exact identity lists are retained
in the handoff.

| EP17 image scope | All images | Development | Support calibration | Sealed audit | Shared with EP18 main |
| --- | ---: | ---: | ---: | ---: | ---: |
| Observed | 4,320 | 2,880 | 720 | 720 | 4,320 |
| Common to four participants | 3,840 | 2,560 | 640 | 640 | 3,840 |
| Repeat-qualified | 3,634 | 2,423 | 605 | 606 | 3,634 |

All **720 concepts** overlap EP18's main inventory. Across each EP17 image
scope, overlap with EP18 repeated-session image identities is **zero**;
the repeated-session **concept** overlap is **200**, split **126/35/39**
across EP17 development/calibration/audit roles. This differs from the earlier
source inventory's 100 shared test-image identities: those source test rows
are not part of EP17's observed experimental image universe.

The updated [exposure record](config/ep17_ep18_exposure_record.json) freezes
the allowed interpretation: EP18 may provide new-participant EEG evidence
conditional on its fixed THINGS inventory, but cannot be independent
stimulus-family confirmation of EP17. DINO and supervised ResNet50 V1 share
official checkpoint identities; EP18 has its own selected final pooled
features and DINO bicubic transform, while EP17 fixes four pooled stages and
common bilinear preprocessing. Checkpoint reuse does not establish identical
features or clean pretraining exposure. The prior archive size/CRC distinction
and documented-history limitations remain in force.

This attempt opened no neural, image-pixel, feature or checkpoint payloads,
changed no selection and opened no audit. It establishes no physical access
seal. EEG-driven comparisons remain prohibited until the existing independent
runtime separation requirement is met. The failed namespace route remains
stopped; no retry, workaround or weaker boundary was selected. ASTRA ownership
and any validation remain with the coordinator.

At23:55UTC root verified the exposure JSON and its result reference on compute,
and added the current exposure record plus immutable handoff as explicit
dependencies of the eventual feature freeze. The first ASTRA check rejected an
uppercase input ID; its failed output remains in
`astra/v0.0.14/validation_20261004_exposure.json`. Root corrected that ID to
lowercase; schema and semantics then passed in
`astra/v0.0.14/validation_20261004_exposure_repaired.json`. This mechanical
repair changes no scientific rule or result. Allocation46531931 completed0:0
at23:55:36UTC. Its entire562core-seconds are charged once to EP17, including
this brief handoff/document validation; no additional EP18 charge applies.

## 2026-10-12 16:32 Pacific — development_eeg_preprocessing — complete development EEG preparation selected

The coordinator selects actual EEG access for fixed preprocessing only, under
the already adopted 30-person development role and prospective filter recipe.
The five-field release is `outputs/config/development_eeg_release_20261012_001.json`.
It opens exactly the development manifest, its30 named recordings, original
62-channel order, complete metadata geometry and stored coefficients. Audit
EEG, image features, behavioral outcome values and model/scientific scores
remain outside this preparation. Existing same-account procedural separation
is preserved; it is not an OS-enforced seal.

All30 people and72 main sequences each are retained. Reuse the stored129tap
100Hz Kaiser8.6 causal filter,128past/0future support, absolute-grid phase0
250Hz decimation, no high-pass or detrending and original acquisition times.
The64ms delay remains disclosed. Save all62 float64-voltage channels, every
full-grid row, original common mask and structural event/clock joins. Learned
transforms and the single62-to-61 projection remain fold-runner operations.
Target/button covariates must later join their separately authorized event
arrays; absence here never means zero. Source review verified the structural
source/frozen-join schema excludes target/response/RT/accuracy columns.

Complete geometry supplies70,419,836 raw sequence samples,17,496,937 scored
250Hz samples and a maximum32,633-sample interval. The full-grid signal upper
bound is about8.134GiB, with times/masks, bounded metadata and publication
records admitted within12GiB durable output. Process one participant at a
time; raw interval buffers are limited to64MiB, participant/local publication
to512MiB and metadata to64MiB/person. Completed private participant caches
survive a stop; only exact all30 completion publishes the full manifest.
Independent source review found no actionable cache/caller defect. Exactly
two focused cache/join tests precede EEG access in the substantive parent.

Select one normal1CPU/7GiB/2h parent, one numerical worker,6GiB softRSS,
actual-StartTime+7140seconds bounded by EndTime-minus60 and the unchanged
October13 21:30UTC episode deadline. Reserve2allocatedCPUh once for EP18;
this includes the separately scoped EP21 seven-method hardware software
companion and changed ASTRA validations, with no duplicate EP21 charge.
No GPU, scientific subset, model, score, automatic retry or successor is
selected. Runtime is unmeasured; this is full production preprocessing,
not a reduced scientific pilot. Existing source reader/bridge checks are reused.

Wrapper: `outputs/code/run_development_eeg_preprocessing_001.sbatch`.
Fresh data output: `outputs/runs/eeg_preprocessing/development_20261012_001`.
Operational root: `outputs/runs/eeg_preprocessing/operations_20261012_001`.
The new cache/caller and wrapper will freeze at submission. Image extraction
and the already submitted false-partition chain remain independent.

Submission returned **47572924** at23:32:58UTC. Cache/caller/wrapper and the
EP21 hardware delta are now frozen. Independent source reviews passed for
both; compute test and empirical preparation outcomes remain pending.

Fresh `sh_part` discovery confirms the user's russpold owner partition is
available with a7-day limit and sufficient per-core memory for the unchanged
1CPU/7GiB/2h request. Sherlock's running-jobs documentation distinguishes this
dedicated group partition from the preemptible shared `owners` partition.
Select `russpold,normal` as alternative eligible partitions for the same
pending47572924, preserving its ID, script, limits, no-requeue and outputs.
No duplicate job or priority/QOS change is selected. The normal-only sbatch
header remains the preserved submission source; this scheduler-field amendment
is the operative partition request. Source:
https://www.sherlock.stanford.edu/docs/user-guide/running-jobs/#owner-partitions

The same discovered group-partition alternative is selected for pending
false-partition parent47571099: `russpold,normal`, with13CPU/96GiB/4h and
its original afterok:47551339 dependency unchanged. The wrapper already admits
the actual CPU/memory allocation independently of partition name; no source,
cache recipe, search count, deadline or resource reservation changes.

## 2026-10-12 16:40 Pacific — development_eeg_preprocessing — allocation running and checks passed

Parent47572924 started at16:39:03Pacific onsh02-08n53 in normal; the running
shell verified its own job ID,1CPU/7168MiB and actual start/end epochs. The
russpold,normal alternative was accepted, but the observed start was in normal;
no causal scheduling-speed claim is made. Its common deadline is1791855483.
The two cache checks passed in14.480seconds; EP21's independent seven checks
passed in18.906seconds. Both changed ASTRA programs passed. The wrapper then
proceeds to the complete development EEG preparation. It does not fit a model.
No completed participant/cache or full30-person completion is claimed yet.

## 2026-10-12 — baseline_search — exact capacity implementation remains unresolved

Source analysis found that sequential full-root QR/SVD can reduce peak memory
but cannot make the straightforward dense CPU backend practical within the
12,000CPUh allowance. For q=74160, one float64 square matrix is40.98GiB.
Using approximate400k inner and500k outer rows,21 unique training-block sets
and30people×9categorical slots gives about2.52e19 QR operations alone, or
583GFLOP/s per allocated core if it consumed the entire CPU allowance. This
is an illustrative operation count for that concrete backend, not a lower
bound excluding all algorithms. Complete SVD, leverage, nonlinear-label
capacity, other temporal branches and fitting add work. Sparse plus low-rank
conditioning may help, but the current code has no qualified exact full-spectrum
backend. The separate512GPUh allowance and an exact GPU route remain unresolved;
this CPU count does not establish their impossibility. No dense-QR implementation,
benchmark, approximation, shortened FIR or reduced scientific panel is selected.
Preprocessing, images and false-group input construction proceed independently.

An independent source review proposed concrete prospective callback choices
for currently unspecified nuisance bases, temporal endpoints, training-only
covariance, weights, fixed base penalties and deterministic ties. They are not
adopted or executed by this note. BR pipeline_plan_review was called once with
that explicit proposed recipe, use_kg=false and no empirical scores; it returned
approve/low risk/no findings, with no KG rules and only a generic six-step
checklist. This is limited advisory feedback, not substantive validation of the
statistical choices, a practical solver, launch authority or scientific evidence.

## 2026-10-12 16:41 Pacific — development_eeg_preprocessing — first real EEG participant cache published

At16:41Pacific the first participant's complete publication was observed:
sub-07,72sequences,22248main events,586832full-grid samples and583232 retained
samples. `development_20261012_001/sub-07/complete.json` records296802198bytes
of arrays plus21860769bytes of structural metadata. No model scores were
computed. This establishes actual real-EEG preparation progress; it does not
establish complete30-person preparation or authorize model comparison.
