# Proposed regular-only neural steward export — 2026-09-30

Status: **prepared, not authorized or submitted**. The approved fixed-provider measurement exception resolves the former custom-beta eligibility gate. This proposal performs only regular-response provisioning; no decoder/converter fitting, feature extraction, reliability estimation, prediction or scoring.

## Exact requested allocation

One Slurm job, 1 CPU, 4 GiB RAM, 12-hour maximum, 0 GPUs; at most 10,000,000,000 bytes durable output and 10,000,000,000 bytes temporary storage. The previous stimulus/mask allocations were specific completed jobs and do not cover this export. Runtime is an unmeasured conservative cap, not a benchmark or prediction. A timeout is a stop, not automatic authority to extend or resubmit.

## Real metadata estimate

| Participant | Regular trial rows | Fixed voxels | Float32 matrix bytes |
| --- | ---: | ---: | ---: |
| sub-01 | 30,372 | 12,295 | 1,493,694,960 |
| sub-03 | 30,373 | 12,897 | 1,566,882,324 |
| sub-05 | 30,372 | 10,891 | 1,323,125,808 |
| sub-06 | 30,372 | 12,023 | 1,460,650,224 |
| sub-07 | 30,372 | 12,057 | 1,464,780,816 |
| Total | 151,861 | — | 7,309,134,132 |

The matrix payload is 6.807 GiB; bit-packed finite flags add approximately 228.4 MB plus aligned TSVs, native voxel IDs and compact structural metadata. Subject-unique/tau-train/tau-test float32 totals are 4,534,657,088/2,187,577,316/586,899,728 bytes. Original repetitions remain separate. Image averaging belongs to the downstream locked pipeline.

The 150 existing source gzip objects total 176,525,339,806 compressed bytes and expand sequentially to 1,863,070,763,840 bytes. Stream them directly rather than stage expanded sessions. One spatial volume is 11,696,828 bytes; the largest full session would be 12.48 GB and must never be loaded as a full array. Reaching the 12-hour cap requires aggregate expansion throughput about 43.1 MB/s; actual throughput remains unknown.

## Concrete extraction and isolation

Use `regular_trial_index.tsv`'s session-global `beta_volume_index`, not run-local `beta_index`. Read/validate each actual NIfTI header's shape, dtype, endian, offset, scaling and native-grid precision using the already repaired grid-binding helper. Derive NIfTI Fortran storage offsets from the approved ascending C-order native voxel IDs without changing the scientific voxel ordering or resampling.

Sequential gzip traversal necessarily processes the mixed source encoding. Discard excluded volumes as opaque byte buffers; never form, retain, summarize or emit OOD sample arrays. The fixed-measurement ancestry exception does not grant direct OOD scientific access. Do not claim zero physical processing of excluded encoded bytes. Only regular masked rows may be decoded into scientific arrays or leave the steward.

Write participant × role response matrices, aligned trial/image/session/run IDs, bit-packed finite flags and canonical voxel IDs in an owner-only sealed payload under `outputs/steward_provisioning/regular_neural_sealed/`. It remains outside the protocol resolver. A separate `regular_neural_structural_report.json` may contain row/voxel counts, native/header binding, finite/NaN coverage without values and publication status. Preserve native values under declared NIfTI scaling; no new normalization, imputation, response selection or averaging.

Use the prepared single sequential subject-stream worker; stage/publish subject outputs without keeping a second full export copy. Validate declared caps before publication and publish only after all required participants/roles succeed. Partial work has no valid handoff; no direct scoring interface is opened on failure. No stable neural interface or scoring authority is created by this proposal.

## Required authorization and next step

The worker requires a separate scientist-approved allocation record and Slurm job context before opening beta payloads. On approval, submit this single export-only job, babysit it, inspect only the redacted structural report and publish the regular-only sealed artifact. Model/runtime binding, required remaining conformance, protocol lock and later fitting/scoring authority remain separate. No successor is authorized.
