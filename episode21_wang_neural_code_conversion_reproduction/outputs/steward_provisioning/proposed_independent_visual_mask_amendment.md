# Historical proposal adopted 2026-09-30

The scientist approved this proposal with “ok” after clarification. The adopted
specification is `outputs/protocol_mask_amendment_2026-09-30.yaml`. Its one
mask-only job46141362 completed successfully; the original proposal below is
retained as review history.

# Proposed independent visual-mask amendment — not adopted

Prepared 2026-09-30, before EP21 neural outcome access. No mask values have
been read or materialized. This proposal does not open model execution.

## Recommended route

Replace the primary provider whole-visual-cortex definition with an explicitly
adapted native visual-cortex union. For each of the five participants, use the
union of the following 29 provider masks, whose filenames and label metadata
are present for every participant. Choose the same component identities in
all five subjects based only on their documented independent-localizer origin
and availability, with no main-experiment response-based additions.

| Family | Frozen proposed labels |
| --- | --- |
| retinotopy | V1d, V1v, V2d, V2v, V3d, V3v, V3A, V3B, hV4, LO1, LO2, TO1, TO2, VO1, VO2, IPS0 |
| body | EBA, FBA |
| face | OFA, FFA-1, FFA-2, pSTS-faces |
| object | l-objects, v-objects |
| place | PPA, OPA, MPA |
| character | VWFA-1, VWFA-2 |

Subjects: sub-01, sub-03, sub-05, sub-06, sub-07.
Source path template, relative to the user's OAK tree:
`br_autoresearch_data/steward_acquisition/laion_fmri_mixed_role_20260922/dataset/derivatives/rois/{sub}/{family}/{sub}_space-T1w_res-1pt8_label-{label}_mask.nii.gz`.
Family label tables are
`{sub}_space-T1w_res-1pt8_seg-{family}_dseg.tsv` in the same directories.

Provider source: the pinned LAION-fMRI archive at commit
`ffb6abef8f787ca37f9a7faa9dd4a97119b6d603`, `docs/source/rois.rst`, and
https://laion-fmri.hebartlab.com/rois.html. These define retinotopic ROIs from
the retinotopy localizer; body/face/place/character ROIs from fLoc; and object
ROIs from the object localizer. These sources distinguish them from the
main-experiment-selected `laion-general` family.

## Locked construction and limits, if approved

Use the logical union of the provider's supplied binary native masks. Preserve
native grid, affine, voxel identities and canonical voxel order. Stop if a
component is missing, malformed, nonbinary, or grid-incompatible; do not fill,
threshold response maps, interpolate, dilate, or make outcome-based exclusions.
Exclude every `laion-*` mask, SPCS/IPCS, and undocumented extra AFP/pSTS-words
labels. No main-experiment R² or beta values, tau-test/OOD responses, images,
features, or sibling-episode outputs enter this construction.

Coverage is adapted and incomplete relative to whole visual cortex. Optional
masks are deliberately outside the common core: MT/MST are absent for sub-05;
SPL is absent for sub-01; mfs-words is absent for sub-07. A separately named EVC
volume was not present in the checked native filenames; its six V1–V3 component
masks are present and included. Filename support is not a voxel coverage result.
Report actual mask support after authorized construction and disclose this
primary-ROI deviation in every affected scientific report. Do not claim
anatomical equivalence to the source study or invent a coverage/comparability
margin. Unsupported subarea crosswalk claims remain not evaluable.

This changes `PROTOCOL_TABLE.yaml:roi_contract` and the corresponding
`response_and_ROI` decision, so it needs explicit scientist approval before
adoption. All question, participant/image roles, model choices, inference,
OOD reservation, scientific budget, and stopping criteria stay as specified.
This amendment resolves mask definition only. Provider beta/preprocessing
provenance, neural handoff, model assets/runtime, required conformance, valid
protocol lock, and later neural execution authorization remain separate.

## Separate bounded mask-only resource request

One CPU-only Slurm job: 1 CPU, 2 GiB RAM, 10-minute limit, no GPU, at most
250 MB durable output and 500 MB temporary storage. This allocation is new;
it does not reuse the completed one-job stimulus-export allocation. Read only
the 29 independent masks per subject and their small metadata; stage selected
mask files as needed. Produce five union masks, native voxel indices, the
component/grid provenance report, and an actual mask-support QC figure. No
neural or feature analysis, no model fitting/scoring, no R²-map reads.
Do not submit until this allocation and scientific amendment are approved.

## Alternative preserving the existing ROI contract

Keep the current whole-VC requirement and obtain authoritative construction
provenance for an eligible provider broad mask. The precise, unsent request is
`mask_provenance_request.md`. If it proves tau-test/OOD exclusion, bind that
mask without this amendment. If prohibited inputs contributed, it is ineligible.
A training-response-derived replacement would require its own prospectively
specified estimator/thresholds, eligible input provenance and allocation;
none is proposed or authorized here.
