# Source-preserving neural role materialization

**Root adopted this reviewed scope at 2026-10-04 22:30 UTC for implementation,
focused fixtures and the fixed 64-row pilot**, conditional on completed full001
roles, the exact layout cap and a bounded capacity observation before the first
neural read. The original full twelve-product copy was unselected pending
pilot costs; the current column-copy selection after successful pilot003 is
recorded below. No reader release is authorized. This specification
was prepared from the contracts, completed metadata definitions and existing
attempt003 array headers; no neural, pixel, behavioral-response or model-feature
values were opened. The full metadata assignment's terminal completion remains
a prerequisite rather than an assumed result.

## October 9 implementation and bounded repair

The adopted pilot001 ran in allocation 47085635 after current capacity was
established. It stopped before any neural read with
`private_directory_permissions_invalid`: the new, task-owned `outputs/cache/private`
directory inherited setgid from its 2770 parent, producing mode 2700 where
the implementation required exactly 0700. No role-materialization directory
or private pilot product was created. Preserve its operational report at
`outputs/runs/role_materialization/pilot001/pilot_report.json` and its original
runner unchanged. The coordinator checked that this new private directory
was empty and normalized only that directory to 0700.

The narrowly reviewed replacement is
`outputs/tools/materialize_neural_roles_pilot002.py`. It invokes the unchanged
pilot core with fixed fresh pilot002 private/public paths and
`outputs/tools/neural_role_private_paths.py`: new owned directories receive
exactly 0700 immediately after creation, before content, while existing
unexpected modes and symlinks still refuse. This changes no role, source
extent, byte, memory or time bound. The three new path/regression fixtures
and six full-copy fixtures passed in 0.432 seconds in allocation 47086577;
the original ten passing copy fixtures are reused.

Pilot002 then stopped in 4.329 seconds, before neural reads, with
`source_layout_exceeds_decoded_bound` (peak RSS 454,496 KiB). The coordinator's
allowed header observation identifies sub-01/B chunks `(12780,1,1,1)`, gzip
level 3 and dtype `<f4`. A logical 64-row request across all 227,214 columns
would decode 11,615,179,680 bytes, exceeding the unchanged 256 MiB bound.
Pilot002 and both stop reports remain preserved. The coordinator subsequently
adopted the column-stripe implementation and bounded pilot described below.
Neither the original fixed pilot nor the row-based full runner may silently
change its source scope, chunk rule or resource bounds to continue.

The historical `outputs/tools/materialize_neural_roles_full.py` implements the
original row-based twelve-product plan without selecting its execution. Its CLI
requires `--pilot-attempt pilot001` or `--pilot-attempt pilot002`, a current
`--capacity-observation` record, and a positive explicit `--max-seconds N`.
The chosen pilot must have completed successfully; pilot001's stopped report
cannot admit a full copy. Root must first review the actual completed pilot's
cost, layout and RSS and record a finite full bound in the experiment log.
The external command must use `timeout --signal=TERM --kill-after=10s` with
an `N + 60` second limit and an allocation that also fits the ten-second grace
and remaining episode budgets. No default duration, automatic full run,
reader release or new source-access grant follows from implementation.

The full runner parses the frozen metadata once, checks all twelve headers
and every planned 64-row decoded-work bound before any neural read, and
rechecks each product before use. It copies the selected pilot's verified
first 64 rows through ordinary new writes without a logical mixed-source
reread of those rows, validates exact inherited index maps, and reopens every written segment
for dtype/shape/bit equality. It tracks known output files, including private
partials and the preserved pilot, and limits each file write against the
remaining 160 GiB aggregate ceiling. Failed attempts remain private and
ineligible; the original reader/scoring flags remain false. Full execution
of that row-based implementation remains **UNSELECTED**; the separately
selected column-copy implementation below supersedes its traversal.
If a source chunk straddles row 64, later hyperslabs can decode that chunk's
earlier rows again. The full report conservatively records those decoded
extents and audit-byte exposure; logical pilot reuse does not claim that
physical chunk decoding never overlaps the preserved rows.

## Current adopted column-stripe amendment: pilot003 complete, full attempt002 selected

Root adopted this engineering amendment after inspecting only the twelve
named source headers in allocation 47086577. The fixed metadata record is
`outputs/runs/role_materialization/compute_job_47086577/all_source_layout_headers.json`.
All twelve products use source chunks `(N_source_rows,1,1,1)`, dtype `<f4`,
gzip level 3. No neural value was read by the observation. The earlier
row-based modules, tests and pilot001/pilot002 reports remain unchanged;
their traversal cannot run this source layout under the existing bound.

`outputs/tools/materialize_neural_roles_columns.py` now implements the
prospectively selected traversal. The completed fixed cost pilot used
sub-01/B, all 12,780 original rows and original voxel columns `[0,512)`.
Its logical and decoded source extent is **26,173,440 bytes**, with every
original row, inherited role, eligibility flag and float32 bit pattern retained.
It has the same one CPU, 2 GiB, 120-second internal/180-second external deadline
plus ten-second grace, and 256 MiB decoded/output ceilings. Its fresh private
and public paths end in `pilot003`. It opens only the exact sub-01/B source
dataset; other observed headers are read solely from the saved metadata record.
The command, inside the selected compute allocation and child memory bound, is:

```sh
timeout --signal=TERM --kill-after=10s 180s python3.12 -B \
  outputs/tools/materialize_neural_roles_columns.py --mode pilot \
  --capacity-observation outputs/runs/role_materialization/capacity_observation002.json
```

The current-job capacity record must still match the actual allocation; its
filename alone is not evidence. The pilot routes all selected source rows into
ordinary private role arrays with shape `(role_rows,1,1,512)` and preserves
the full original row map. Output chunks are
`(min(1024,role_rows),1,1,min(512,output_columns))`, without compression or
filters. Reopened stripe readback requires exact shape/dtype/byte equality.
The first raw neural/audit-byte access is reported conservatively before I/O;
this is trusted copying, not an audit model evaluation or scientific result.

In allocation 47086577, all six column-copy checks passed in 0.289 seconds.
Pilot003 then **completed in 5.270826 seconds**, with peak RSS 456,100 KiB,
0.004007607 seconds of header inspection, 0.506149549 seconds of source reading
and 0.228693355 seconds of routing/writing/reopened readback. It copied exactly
26,173,440 logical bytes into 33,926,074 private output bytes, with all copy
bytes equal. Neural values and raw audit-response bytes were read for trusted
copying; the scientific audit evaluation count remains zero, and all reader,
isolation and scoring flags remain false.

Root now selects full **`attempt002`**, conditional on a fresh capacity record
bound to its own job, with **9,000 seconds internal, 9,060 seconds external
plus ten-second termination grace, one CPU, 2 GiB and a three-hour batch
allocation**. Keep the 2 GiB child virtual-address cap. Its CLI is
`--mode full --max-seconds 9000 --capacity-observation PATH`. The root-owned
wrapper `outputs/tools/run_neural_columns_full_001.sbatch` binds the full
capacity observation to the actual allocation. It does not resubmit itself.
This records selection, not successful submission or full-copy completion.

The measured read plus write/readback cost is 0.734842904 seconds per pilot
stripe. Multiplying by all 5,169 stripes gives approximately 3,798.4 seconds;
doubling for filesystem variation gives approximately 7,596.8 seconds. The
9,000-second internal limit includes further metadata, header, map and setup
allowance. The current attended allocation has about 104 minutes remaining,
which cannot contain that conservative envelope, so root will close it and
use the fresh finite batch. Charge each allocation's actual allocated core-hours once to
EP17; the new three-hour reservation does not reset or extend episode budgets.

Full execution requires successful pilot003 and revalidates its exact row
maps, column extent and file identities.
All twelve current source headers and every planned source stripe must pass
before any full-copy payload access. The twelve products contain 5,169
512-column stripes including final shorter stripes. Reuse sub-01/B's first
512 columns through ordinary private reads/new writes; source column chunks
are disjoint, so this traversal neither logically rereads those columns nor
re-decodes their source chunks. Original fourth-axis positions remain unchanged.

For each nonempty role array with `n` rows and `V` columns, calculate its
uncompressed padded chunk payload as
`4 * a * b * ceil(n/a) * ceil(V/b)`, with `a=min(1024,n)` and
`b=min(512,V)`. Sum all 36 full products, add the exact serialized index-map
bytes, actual preserved pilot bytes and failed operational reports, a 1 GiB
engineering HDF5 metadata/slack allowance and a 1 MiB terminal-report reserve.
The coordinator's metadata-only projection has 154,958,561,280 padded array
bytes; its conservative provisional 256 MiB pilot reserve plus 1 GiB slack
and 1 MiB report reserve totals 156,301,787,136 bytes before exact maps.
The runner separately adds the exact map bytes and known preserved files.
The allowance is not a mathematical HDF5 metadata bound: the existing file
write limiter independently enforces the **160 GiB actual aggregate ceiling**,
including partial files, and stops if exhausted. Pilot planning instead uses
16 MiB metadata slack and its unchanged 256 MiB aggregate cap.

Six new source-free checks cover fixed pilot/full traversal and bit identity,
pilot reuse binding, all-header preflight, failed readback/partial evidence,
actual HDF5 padding at row/column edges, and explicit scope/cap admission.
All six passed in allocation 47086577, and independent source review found
no blocking defect. This selected full copy introduces no automatic successor,
larger stripe, raised cap, reader activation, response estimator or candidate
scoring.

## Purpose and retained scientific boundary

[DATASETS.md](../DATASETS.md#role-separated-handoffs) and
[SEARCH_POLICY.yaml](../SEARCH_POLICY.yaml) require ordinary role-filtered
files and prohibit mounting the mixed source to the adaptive search worker.
The proposed trusted builder performs only literal row routing into three
private neural products, plus their metadata index maps. It makes no response
estimator, finite-voxel selection, anatomical choice, rescaling or score.

Preserve every source presentation, including presentations whose image is
ineligible for the primary analysis, and attach its already inherited concept
role and eligibility flag. Copying an ineligible row does not make it eligible.
All original columns remain in their original order at B, C and D, including
NaNs, infinities, signed zero and any other source float32 bit pattern. Do not
average repetitions, change dtype, flatten away the two source singleton
dimensions, normalize, clip, impute, mask, rank, compute reliability, inspect
sample values, or compute candidate comparisons.

This creates **builder-private staging products**, not activated search or
evaluator handoffs. Copy completion is distinct from actual reader enforcement,
response units, anatomical/voxel correspondence, finite-response eligibility,
calibration construction and neural-scoring readiness.

## Exact metadata prerequisites and routing

Require the completed fixed source metadata `runs/source_metadata/attempt003`,
event reconciliation `runs/released_event_reconciliation/retained001`,
eligibility `runs/source_qualification/attempt002`, chronology
`runs/source_chronology/attempt001`, and the selected complete role assignment
`runs/concept_role_assignment/full001`. No alternative attempt is selected by
directory order or by a newer modification time.

The full001 prerequisite is `assignment_report.json` with terminal
`status: completed` and `metadata_assignments_complete: true`, plus complete
`concept_assignments.csv`, `image_assignments.csv` and
`presentation_assignments.csv`. `source_roles_complete`, neural materialization
and physical isolation can remain false there; the metadata assignment alone
does not establish them. A pilot report, partial CSV or interrupted optimizer
cannot supply an eligible map.

Use the adopted [concept-role specification](concept_role_assignment_specification.md)
and [chronology definition](source_chronology_specification.md) unchanged:
720 concepts; 480 development, 120 support/calibration, 120 sealed audit;
six development folds of 80, 24 development blocks of 20, twelve audit blocks
of 10; and 3,634 repeat-qualified common images. The broader metadata contains
4,320 observed image identities and all 49,980 retained presentation rows.
Ineligible images inherit the same whole-concept role and remain explicitly
marked. No neural value changes any identity, eligibility, half, fold or block.

Before the first neural read, join all metadata identities exactly and require
one role per concept across images, participants and stages, with the original
hard capacities. For each participant, its numeric
`glmsingle_trial_index_zero_based` must be a unique complete interval
`0..N_s-1`; the known participant row counts sum to 49,980. Sort routing only
by this source coordinate, not the input CSV's incidental row order. Preserve
actual acquisition rank and eligible-image repeat-half labels as metadata;
they must not reorder the copied B/C/D arrays. Ineligible images retain their
missing analysis-half label. This keeps GLMsingle order distinct from actual
chronology, including the documented out-of-order run.

For participant s and role r, assign local index j to the j-th source row with
that role in increasing original GLMsingle index. The **same source-to-local
index map** applies to B, C and D. Every source row is routed to exactly one
role; no discarded, duplicated, relabeled or newly synthesized presentation is
allowed. Unknown concepts, extra/catch/test rows outside the frozen retained
mapping, changed shape or conflicting roles stop before the affected read;
they are not silently sent to development.

## Exact source and output arrays

The only proposed neural input root is the existing read-only source:

`$OAK/users/zijiao/br_autoresearch_data/restricted/cneuromod-things-1.0.1-restricted-raw`

For `sub-01`, `sub-02`, `sub-03`, and `sub-06`, open only dataset **`modelmd`**
in these three paths beneath that root:

`THINGS/glmsingle/<participant>/glmsingle/output/T1w/TYPEB_FITHRF.mat`

`THINGS/glmsingle/<participant>/glmsingle/output/T1w/TYPEC_FITHRF_GLMDENOISE.mat`

`THINGS/glmsingle/<participant>/glmsingle/output/T1w/TYPED_FITHRF_GLMDENOISE_RR.mat`

Original annex links may resolve only to the identified file inside that
read-only source root, as in the existing metadata adapter. No other dataset,
HDF5 external/soft link, virtual dataset, dataset with external raw-data storage
(`dataset.external` must be absent), neighbor neural product, mask payload,
ROI label volume, public D ceiling, source behavior, pixel archive or model
file is an input to this task. The first pilot opens only its named B product.

The permitted attempt003 header report establishes:

| Participant | Stored shape at every B/C/D stage | Dtype | Bytes per stage | Bytes for all three stages |
| --- | --- | --- | ---: | ---: |
| sub-01 | `(12780, 1, 1, 227214)` | float32 | 11,615,179,680 | 34,845,539,040 |
| sub-02 | `(12780, 1, 1, 236701)` | float32 | 12,100,155,120 | 36,300,465,360 |
| sub-03 | `(12780, 1, 1, 210349)` | float32 | 10,753,040,880 | 32,259,122,640 |
| sub-06 | `(11640, 1, 1, 207095)` | float32 | 9,642,343,200 | 28,927,029,600 |

All twelve payloads total **132,332,156,640 bytes = 123.243925 GiB =
0.120356 TiB**. Partitioning rows across roles does not triple that number:
each original value goes to one role file. There are twelve source products
and at most 36 nonempty role-filtered output arrays.

At an adopted launch, recheck the exact stored shape and four-byte floating
dtype from each allowed dataset header. Record and preserve its actual
`dtype.str`, including byte order; the existing report names float32 but does
not separately record endian metadata. Output dataset shape is
`(n_s_role, 1, 1, V_s)` with the exact original dtype. The source voxel index
is the fourth-axis coordinate `0..V_s-1`; no physical-coordinate interpretation
is invented. B/C/D header equality does not establish common finite voxels or
response units: attempt003 explicitly leaves those properties unverified.

## Private paths, ordinary files and publication

Use only this episode's workspace. Proposed private roots are:

`$OAK/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability/outputs/cache/private/role_materialization/pilot001/`

`$OAK/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability/outputs/cache/private/role_materialization/pilot002/`

`$OAK/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability/outputs/cache/private/role_materialization/pilot003/`

`$OAK/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability/outputs/cache/private/role_materialization/attempt001/`

`$OAK/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability/outputs/cache/private/role_materialization/attempt002/`

The current column amendment completed pilot003 and selects full attempt002
under the measured finite bound above; the earlier roots remain historical
or unselected.

The existing repository ignore rule for episode `outputs/cache/` covers this
entire private tree, including `.partial` files and index maps. Verify that
coverage before creation; no Git staging, new tracked neural files or ignore
rule mutation is selected. Git exclusion is not an access-control boundary.

Within each fresh attempt use the exact role directory names `development`,
`support_calibration`, and `sealed_audit`. Each nonempty participant/stage
product is `<role>/<participant>/<stage>.h5`, containing an ordinary local
`modelmd` dataset. No symlink, hard link back to source or pilot, HDF5 external
storage, external/soft dataset link, or virtual-dataset indirection is allowed.
The three roles are separate physical files, not three views in a mixed file.

Each role also has its ordinary `source_index_map.csv`, with participant,
stage-independent local row index, original GLMsingle row index, exact image
and concept IDs, eligibility/reason, role, folds/blocks, source session/run/
trial coordinates, retained acquisition rank and original repeat-half labels.
Copy those fields from the completed map; do not reinterpret blanks as zero
or construct a new chronology. Columns retain source integer/identifier
semantics. A compact metadata description states the fixed original voxel
index range and stage-specific source path/dtype/shape. It does not pretend
to be an anatomical or common-finite-voxel mapping.

Set private attempt/role/participant directories to mode 0700 and private
array/index files to 0600 during creation, retaining restrictive effective
permissions before any content is written. A fresh attempt may not overwrite
an old one. Stage files as ordinary `.partial` files in their own private
role directory; close and validate them before publishing their final name.
No consumer may use any attempt until its complete terminal report succeeds.
Failure leaves private partial evidence ineligible; it does not remove old
attempts or publish a subset as a complete handoff.

Only bounded operational reports are exported to the controller-visible
`outputs/runs/role_materialization/pilot001/pilot_report.json` and later
`outputs/runs/role_materialization/attempt001/materialization_report.json`.
The October 9 replacement publishes its separate
`outputs/runs/role_materialization/pilot002/pilot_report.json`; it never
overwrites the stopped pilot001 report.
The column amendment uses its own `pilot003/pilot_report.json` and later
`attempt002/materialization_report.json` under the same public parent.
Allowed fields are fixed input/output identities, shapes/dtypes, row counts,
metadata-only role/eligibility totals, bytes, timings, peak RSS, permissions,
completion/error codes and copy-equality booleans. Export **no response values,
NaN/finite counts, ranges, means, histograms, correlations, neural payload
digests, images, embeddings or sample array representations**. Error handling
must not serialize an array or a value-bearing exception payload.

Unlike the preceding metadata builders, an adopted copier **does read neural
values internally**. Its report must truthfully set `neural_values_read: true`
when the first payload block is read, record the bounded participant/stage/row
scope and first-access time, and distinguish raw audit-response bytes accessed
for trusted copying from an audit model evaluation. Record whether the copied
block includes audit rows using metadata alone. Never describe a completed
neural copy as metadata-only or claim audit response bytes were never accessed.
No agent-readable value summary or candidate/audit score is released, and the
one-shot scientific audit evaluation count remains zero.

Mode 0700/0600 restricts other ordinary readers; it does not separate agents
running as the same UID or prevent that UID from reaching the mixed source.
Keep `physical_role_isolation_verified: false`, `reader_handoffs_activated:
false`, `source_roles_complete: false` and `candidate_scoring_authorized:
false`. The trusted builder alone handles these staging files. A later
independently qualified reader boundary is required before any search-worker
or audit-evaluator release. This proposal neither implements nor substitutes
for that boundary, and authorizes no workaround for a stopped EP18 isolation
route.

## Original row-based streaming and checks (superseded by the column amendment)

Process participants in lexical order and stages B, C, D, with one worker.
Read source rows in contiguous blocks of at most 64, always all original
voxel columns. Route each block's rows by the metadata map, preserving source
order within each role. The largest decoded 64-row payload is 60,595,456 bytes
for sub-02; a final shorter block is retained. Each source row is logically
read once for a complete copy. Do not form a full participant matrix in RAM.

Output HDF5 uses no compression, quantization or lossy filter. For a nonempty
role, use chunks `(min(16, n_s_role), 1, 1, V_s)` and the known final shape;
a zero-row pilot role gets only its explicit empty metadata record. Cap source
raw-chunk cache at 16 MiB and each open destination cache at 16 MiB; keep at
most the current source and its three role products open. Do not reserve an
in-memory cohort or model process alongside the copier.

Source chunk layout is absent from the permitted saved headers. Before any
pilot values are read, inspect only that dataset's layout header. For a
chunked source, bound decompression work by the count of chunks intersecting
the fixed row block times `product(chunk_shape) * dtype.itemsize`, including
edge padding. For contiguous storage, use the selected logical block size.
Require this bound to be **at most 256 MiB** for the pilot and each later
64-row block. A header exceeding the bound stops that route before value
access; it does not silently change the selected pilot or memory request.
Report header/layout facts without viewing values. No chunk geometry is
claimed verified in this proposal.

Before any full-copy payload read, inspect the exact twelve allowed dataset
headers and enumerate their planned 64-row block bounds. All twelve products
must pass the same shape/dtype/local-storage/layout rules. Recheck the current
product header before its first read; any changed or oversized product stops
before that product's values are read and leaves any already copied products
private and the whole attempt incomplete. Do not silently reduce block size,
raise the chunk bound or drop a participant/stage to continue.

For every routed write, read the just-written destination segment back and
compare its dtype, shape and byte representation with the in-memory source
rows. This byte comparison preserves NaN payloads and signed zero; floating
`equal_nan` equality alone is insufficient. Discard the source/readback buffers
before the next block. Publish only equality/count/byte facts, not mismatch
values or value locations. Internal numerical summaries are unnecessary.
Completion also requires metadata bijection/partition checks and unchanged
source/role row order at all stages. Content verification does not choose a
voxel or test a scientific hypothesis.

## Original row-based pilot and full-copy planning record

The outcome-independent pilot is the first lexical participant/stage and
first source row block: **sub-01, stage B, GLMsingle rows `[0,64)`**, all
227,214 original columns and both singleton dimensions. It reads exactly
**58,166,784 logical neural bytes (55.472168 MiB)**, routes those same rows by
their inherited roles, writes and verifies the three disjoint ordinary
products. Do not change the block to make every role appear: an empty pilot
role is valid and recorded. No response-dependent sample or alternative
participant is permitted.

Propose one CPU, **2 GiB RAM**, one worker, no GPU, all numerical thread caps
one; **120 seconds internal, 180 seconds external plus 10 seconds termination
grace**. Source input is the one fixed dataset/block, with at most 256 MiB of
decoded source-chunk work; private neural output is at most 58,166,784 payload
bytes, with a **256 MiB whole pilot-output ceiling** for files and metadata.
All prerequisite metadata inputs are individually bounded to 32 MiB, parsed
once, and never replaced by raw events. The pilot is resource/copy qualification
only: its report cannot set full materialization, physical reader separation
or scientific readiness true. No retry or full dispatch is automatic.

The full twelve-product plan has **2,346 source row blocks** at size 64. Its
uncompressed role payload is the 123.243925 GiB total above, plus ordinary HDF5
and index overhead. Reserve a **160 GiB total new-output ceiling**, including
the preserved pilot and any partial output of this one full attempt, well
below the existing 4 TiB storage limit. With the 256 MiB per-source-block cap,
full decoded source work is bounded by 2,346×256 MiB = 586.5 GiB; writing and
readback add about 246.49 GiB of logical I/O. These are workload bounds, not
claims of physical filesystem throughput or compressed source size.

A completed pilot may supply its exact 64 verified sub-01/B source rows to
the later full copy through ordinary private file reads and ordinary new
writes, preserving its metadata source indices. No link or changed endpoint
is introduced. Record reused rows separately; do not logically re-read them from mixed
source or count them as a second source-copy comparison. A later source
hyperslab may decode an overlapping chunk; retain its conservative decoded
scope in exposure accounting. Preserve the pilot
artifact, and keep its rows subject to full-attempt byte and partition checks.

Select the full walltime only after the measured pilot reports header/open,
read, route/write/readback time and RSS. Base planning on all twelve products,
2,346 blocks, the largest documented voxel count and measured byte throughput,
with explicit allowance for repeated product opens and filesystem variation.
Root must record a finite full internal/external bound fitting its allocation
and remaining 6,000-CPU-hour/240-hour budgets before dispatch. Do not assume
the whole pilot duration is pure transfer time or fit the walltime to a
nearly expired allocation. Full execution remains unselected here.

The existing documentation reports an approximately 128 GiB acquired source;
that is **not available destination space**. No current free-space or quota
observation for the proposed private parent was found in the permitted
metadata. Do not infer capacity from the 4 TiB policy ceiling, scan directories
or claim a verified free-space figure. Before any adopted pilot/full launch,
root must supply an existing current capacity observation or obtain one bounded
parent-filesystem capacity/quota observation, covering the 256 MiB pilot or
160 GiB full reservation respectively and concurrent reserved work. This is
an operational launch condition, not permission for a recursive inventory.

Proposed coordinator commands, in its verified compute shell, use only the
known existing episode cache parent and the site's quota report:

```bash
EP17_ROLE_PARENT="$OAK/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability/outputs/cache"
sh_quota
df -B1 -- "$EP17_ROLE_PARENT"
```

Record the applicable quota headroom and that filesystem's available bytes,
accounting for concurrent reservations; filesystem free space alone is not
quota availability. These commands are proposed, not run here. Unknown or
insufficient capacity stops dispatch without a recursive size scan.

## What can remain blocked after a successful copy

| Condition | Consequence |
| --- | --- |
| Complete role/fold/block assignment absent or changed | Do not read neural values; no valid row routing exists |
| Root has not adopted this copy scope/pilot | No neural copy is authorized by this proposal |
| Source layout, memory/time or destination-capacity bound fails | Stop the technical attempt; preserve it and review a concrete repair before any retry |
| Ordinary copied products complete but reader enforcement unqualified | Keep every neural product builder-private; no adaptive worker or evaluator activation |
| Units, voxel/geometry correspondence or common finite support unqualified | Literal copying may complete, but B/C/D comparison and anatomical qualification remain unfinished |
| Calibration estimators, ROI quotas, checkpoint/exposure records, margins or scoring conventions unfinished | No neural candidate score, calibration-derived release, model features or audit access follows |

This separates useful source-preserving preparation from the consequential
scientific choices and access enforcement it cannot settle. No candidate trial
is earned, no scientific audit evaluation occurs and no primary search is
launched by materialization alone. The trusted builder's raw-response access
for copying remains explicitly recorded rather than erased from exposure
history.
