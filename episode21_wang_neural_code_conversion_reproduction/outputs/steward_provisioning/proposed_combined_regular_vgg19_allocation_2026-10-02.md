# Approved EP21 continuation and historical allocation proposal

## 2026-10-05 continuation proposal — APPROVED

The scientist explicitly approved both named pending proposals with **“agree,
please go ahead”**. This EP21 scope is recorded at 2026-10-05 21:41:06 UTC in
`outputs/regular_vgg19_continuation_authorization_2026-10-05.json` and the
separate continuation section of `SEARCH_POLICY.yaml`. It authorizes the two
finite stages below, including conditional staging release, without changing
historical approvals, their consumed budgets, the original lock or scientific
bindings. Existing DUA, source approval, roles and runtime/conformance are
reused. Preparation of these submission files is not a claim that either job
has run; scheduler events belong in the coordinator's experiment log.

### Retained work and exact continuation

The original allocations consumed **17 + 34 + 28,569 = 28,620 seconds
(7:57:00)**. The remaining three minutes cannot finish the export.
Job 46424955's terminal log records an incomplete publication, not a scientific
failure. Bounded metadata inspected on 2026-10-05 establishes:

| Partition | Exact retained location | Status and intended action |
| --- | --- | --- |
| sub-01, sub-03 | `outputs/steward_provisioning/regular_vgg19_sealed.building.46424955/{sub-01,sub-03}` | Each has all 4,712 index rows and complete feature/mask files; reuse without new forwards or payload copying. |
| sub-05 | `$SCRATCH/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/vgg19-export-46424955/sub-05` | Complete staged 4,712-image partition; publish all three files into a fresh continuation directory. |
| sub-05 old partial | `outputs/steward_provisioning/regular_vgg19_sealed.building.46424955/sub-05` | Preserve the 138,084,876,288-byte interrupted feature copy unchanged. |
| sub-06, sub-07, tau_train, tau_test | Original regular handoff's frozen CSV ordinal selections | Only new forwards: 4,712 + 4,712 + 897 + 224 = 10,545 images. |

Each complete subject partition contains `features.h5` (280,105,563,680 bytes),
`finite_packed.h5` (8,753,306,088 bytes) and its 4,713-line `image_index.tsv`.
The old exporter writes an index row only after all 19 layers and finite checks,
closes all staged files, then starts copying. Sub05's complete index and staged
files, followed by a partial durable copy, identify publication as the stopping
point. No feature values were read for this diagnosis. The adapter reuses that
execution evidence and checks exact row identities/HDF5 headers before reuse;
it does not rerun conformance or a new feature-validation study.

### Approved finite resources and storage

1. **One CPU publication job: 1CPU, 4GiB, no GPU, 2h walltime.** Internal cap 7,080s,
   leaving 120s for terminal closure. Copy only complete staged sub05 to
   `outputs/steward_provisioning/regular_vgg19_continuation_20261005/sub-05`.
2. **One remaining-export job: 1GPU, 4CPU, 16GiB, 10h walltime.** Internal cap 35,880s,
   leaving 120s for closure. Run only after successful publication and staging
   release; perform the 10,545 remaining forwards and their publication. The
   coordinator verified at 21:40 UTC that public `gpu` accepts up to two days;
   the CPU publication uses the existing `russpold` partition (seven-day limit).
   Neither stage uses the preemptible `owners` partition.

These are two newly approved finite jobs, **10 additional GPU-hours and 42
additional allocated CPU-hours maximum**, not a reset of old consumption.
Actual accounting remains the coordinator's once-only allocation ledger.
File modification times show 138.085GB copied during 52m49s, implying about
**1h50m** for a fresh complete 288.859GB copy at that observed rate. For new
exports, whole-attempt image throughput projects about **5.9h**; scaling the
slowest completed 4,712-image partition interval (4h16m10s) projects about
**9.6h**. These metadata-derived estimates include variable I/O and are not
guarantees. Either stage stops at its cap, retaining evidence without retry.

Approved episode feature-output ceiling: **1,700,000,000,000 durable bytes**.
The full 1.5130TB feature/mask payload plus the preserved 138.085GB interrupted
copy needs approximately 1.6511TB before small overhead. Reused partitions are
hardlinked into the final handoff only after all seven are complete; hardlinks
do not create a second payload copy or modify old file contents. Byte accounting
counts physical files once, retains the original 10MB public-stage reserve,
counts protocol/claim/continuation metadata and reserves 1MB for final records.

The **340,000,000,000-byte temporary ceiling remains unchanged**. Approval
explicitly includes removal of precisely the three old staging files
`vgg19-export-46424955/sub-05/{features.h5,finite_packed.h5,image_index.tsv}`
and their empty parent directory **only after** the fresh copy is fully written,
fsynced, size-matched, header/index-checked, sealed, and recorded durably.
The complete values/history then reside on Oak; the old partial Oak file stays
untouched. The same condition releases each newly generated staging partition
after publication. No other old staged inputs, weights, artifacts or paths are
deleted or moved. This explicit release prevents two 288.859GB partitions from
overlapping within the 340GB ceiling. Original staged regular images and pinned
model/mean files are reused in place, avoiding another bulk source copy.

### Bound entrypoint and approval record

New code: `outputs/steward_provisioning/tools/continue_regular_vgg19.py`.
The original exporter, original job claim, lock 008 and passing conformance 007
remain unchanged. The adapter validates those existing bindings using the old
authority as historical provenance, and separately requires a new approved
`ep21.regular_vgg19_continuation.v1` record. It never presents the new resources
as if they were the old approval.

The approved record is
`outputs/regular_vgg19_continuation_authorization_2026-10-05.json`.
It must contain exactly the substantive field values in the adapter's
`AUTH_FIELDS` constant, including both resource envelopes, the conditional
staging release and every closed-role flag. It may carry the coordinator's
approval quotation/log reference. The created record contains those exact
substantive fields plus the quotation and unchanged runtime/input bindings.

Independent once-only claims are created with exclusive creation:
`regular_vgg19_continuation_20261005_publish-sub05_claim.json` and
`regular_vgg19_continuation_20261005_remaining-export_claim.json` under
`outputs/steward_provisioning/`. Existing claims are not deleted or reset.
Any failed stage retains its files/status and prevents automatic restart.
The canonical `regular_vgg19_sealed/` and `regular_vgg19_structural_report.json`
are published only after all seven exact partitions are complete. Failure does
not create a successful handoff or authorize any neural consumer.

Use the already accepted environment/interpreter from the original wrapper:
`module load devel math viz python/3.12.1 py-numpy/1.26.3_py312 py-h5py/3.10.0_py312 py-pillow/10.2.0_py312 py-pytorch/2.4.1_py312`.
Its actual Torch build remains the accepted **2.4.0a0**, not a relabeled release.
The interpreter is
`$GROUP_HOME/users/zijiao/brain-researcher-astra/releases/a9188f0ede69912179a8b2e926f0b890c046a75d-5734fb966834bc284ec523913b573e3130038be3-modules/venv/bin/python`.
Set all six adapter-listed thread variables to 1 for publication or 4 for GPU
export, `PYTHONDONTWRITEBYTECODE=1`, and GPU `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
These are the approved compute-node commands, not login checks:

```bash
"$EP21_PYTHON" -B outputs/steward_provisioning/tools/continue_regular_vgg19.py \
  --stage publish-sub05 \
  --authorization outputs/regular_vgg19_continuation_authorization_2026-10-05.json \
  --max-seconds 7080

"$EP21_PYTHON" -B outputs/steward_provisioning/tools/continue_regular_vgg19.py \
  --stage remaining-export \
  --authorization outputs/regular_vgg19_continuation_authorization_2026-10-05.json \
  --max-seconds 35880
```

Independent static source review passed against the original exporter and
role/lock helpers. No execution or numerical conformance is claimed.

The new fixed wrappers are
`outputs/steward_provisioning/tools/continue_regular_vgg19_publish_sub05.sbatch`
and `outputs/steward_provisioning/tools/continue_regular_vgg19_remaining.sbatch`.
They retain the exact internal caps; outer TERM bounds are 7,100 and 35,900
seconds, each with a 10-second KILL grace, inside their two- and ten-hour jobs.
The publication job performs only source syntax checks and one bounded,
non-gating validation of the changed ASTRA recipe before the substantive copy.
No unchanged scientific conformance suite is repeated. The GPU job is submitted
with an `afterok` dependency on that publication job; the adapter independently
requires its completed publication status and released staging before export.
Both jobs are no-requeue and use unique job-ID logs; neither submits a successor.

At 21:40 UTC the coordinator observed Scratch usage 5.8TB/100TB. The Oak quota
query timed out and returned cached 158.5TB/250TB; this is not a verified current
free-capacity reading. No new quota probe is requested. The adapter enforces the
approved physical-file byte ceilings and preserves partial output on a storage
or deadline failure. Old payloads, claims and bindings are not modified to make
space; only the exact verified redundant staging described above is released.

No model, weights, transforms, layer set, precision, feature roles, fitting,
scoring, OOD access or successor authority changes. Prepared code has not been
imported or executed; no new tests/conformance/data qualification are claimed.

## Historical original allocation proposal (2026-10-02)

Status: proposed, not approved or submitted. This is a continuation of the
existing episode and Goal. It does not replace the scientific question,
adopt a new model, open neural outcomes, or grant authority by itself.

The next substantive empirical prerequisite is extraction of the pinned
19-layer VGG19 representation for all 24,681 provisioned regular images.
The corrected public-runtime harness runs its already approved check as a
bounded prelude in this allocation. It does not create a separate short job.

## Requested allocation and scope

One job in the discovered public `gpu` partition, with **1 GPU, 4 CPUs,
16 GiB RAM, an 8-hour wall-time limit, at most 1,600,000,000,000 bytes of
durable output and 340,000,000,000 bytes of temporary storage**. These are
parent-job totals, including the public stage. The public stage retains its
existing 1-CPU/4-GiB/30-minute/no-GPU/10-MB-output/2-GB-temporary limits.

Approve this allocation only for the existing public check, required
outcome-blind protocol conformance and lock completion, and regular-image
VGG19 feature extraction. It permits no neural-value access, decoder or
converter fitting, neural scoring, reconstruction, OOD access, external-site
execution, automatic extension/resubmission, or successor episode.

## Why the storage request is large

Retaining all 19 pinned pre-ReLU layer maps as float32 requires
59,445,152 feature bytes per image. Packed finite masks add 1,857,661 bytes
per image. Thus 24,681 images require **1,513,014,727,653 bytes** before small
container/index overhead. The largest 4,712-image partition requires
288,858,854,856 bytes, permitting sequential partition staging within the
340-GB temporary cap. The 1.6-TB durable cap includes that overhead and a
10-MB reserve for the public stage. This is an arithmetic estimate from the
already selected layer shapes, not an observed throughput or feature result.
No precision reduction, layer dropping, or outcome-based feature selection
is proposed to reduce the request.

Durable features stay sealed under this episode's Oak outputs. Temporary
work uses `$SCRATCH/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/`.
Large stimulus/model/partition I/O is staged there; no large I/O uses HOME
or GROUP_HOME. No credentials are written.

## Conditions before real image access

1. Complete the already approved public source check; reuse a genuine passing
   report only when its binding matches.
2. Complete only the remaining contract-required conformance and prospective
   branch dispositions using permitted public/structural evidence.
3. Publish a valid outcome-blind protocol lock. Preserve code-faithful V1.0.0
   choices and all adopted mask/provider-measurement amendments. Comparability
   subclaims without margins stay not evaluable; unavailable reconstruction,
   identification and external branches receive their actual dispositions.
4. Bind a real feature-only execution context to that lock and this separate
   approval before opening regular-image values. A synthetic context cannot
   substitute for it.
5. Extract only the authenticated regular-image roles, preserving source-index
   ordinal alignment, all 19 layers, pinned preprocessing and role partitions.
   Report structural coverage/failures without opening neural values.

The existing feature script remains a draft: its real lock/context integration
is unfinished. This proposal does not claim launch readiness, runtime
conformance, an existing lock, or completed feature extraction. Approval
would authorize completing that implementation and running the bounded job
once the genuine prerequisites are ready. Failure of a prerequisite or a cap
stops the affected stage; there is no automatic retry or alternative protocol.

## Evidence reused and execution state

- Regular stimuli, independent masks, and sealed regular-neural exports are
  already complete; no re-export or unchanged conformance rerun is proposed.
- The repaired harness's 19 focused regressions passed. This is engineering
  evidence, not a live public-runtime or empirical result.
- One current scheduler inspection found no active/pending EP21 job. Jobs in
  other episodes were not inspected or reused.
- `sh_part` confirmed the public GPU partition supports the proposed duration.
- `sh_quota` reported Scratch 5.5 TB / 100 TB. Its Oak query timed out and
  displayed cached usage of 155.8 TB / 250 TB; current Oak free capacity is
  therefore not verified. This is a storage-report limitation, not an
  experimental outcome or authority to exceed a cap.
- Prior BR scientific retrieval/critique is reused. No scientific choice or
  substantive result changed here; no new critique call or figure is needed.
- Existing 17-analysis ASTRA program and successful harness closeout export
  remain applicable. No scientific terminal export or canonical action occurs.

The only new human decision requested is the explicit allocation above. DUA,
Low Risk classification, approved mask/provider amendments, delegated
outcome-blind choices, and the public-only check approval are already recorded.
