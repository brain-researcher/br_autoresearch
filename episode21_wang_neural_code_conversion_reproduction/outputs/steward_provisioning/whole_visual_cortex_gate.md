# Whole-visual-cortex gate — real provider evidence

EP21's primary input is native whole visual cortex. The current contract
requires an authenticated provider whole-VC mask and prohibits mask/voxel
selection using `tau_test` or OOD responses (`PROTOCOL_TABLE.yaml`,
`roi_contract`). No named whole-VC provider mask has yet been bound.

Provider documentation in the pinned LAION-fMRI code archive at commit
`ffb6abef8f787ca37f9a7faa9dd4a97119b6d603`, `docs/source/rois.rst`, describes:

- `laion-general`: selection from GLMsingle R² of the LAION main experiment.
- `laion-EVC`: retinotopic V1–V3 intersected with that selection mask.
- LAION dorsal/lateral/ventral parcels: subdivisions of the selected mask.
- Plain retinotopic EVC: an independently defined union of V1–V3, covering
  early visual cortex rather than the primary whole-VC region.

The inspected documentation does not identify the exact trial universe used
for main-experiment selection or authenticate exclusion of held-out/OOD
responses. The inspected LAION ROI tree has no JSON sidecar establishing that
provenance. This is **unknown eligibility**, not a measured contamination
result. No ROI values, R² maps, neural values, or outcome plots were inspected.

An independently eligible whole-VC mask and its selection provenance are
required before the primary neural branch can be locked runnable. Plain EVC
is not silently substituted. A narrower EVC analysis would need an explicit
prospective scientific amendment because the current primary ROI is whole VC.

## Selective neural-export constraint

The same pinned provider's `Subject.get_betas()` loads all beta volumes before
applying its stimulus filter (`subject.py`, acquisition loader). Its regular
image filter therefore cannot protect against reading OOD values. A later
authorized neural export must select allowed session volumes before payload
loading and preserve native voxel order. The provider beta sidecar declares
one volume per acquisition-ordered trial; the TSV `beta_index` resets per run.
The prepared `regular_trial_index.tsv` retains both run-local `beta_index` and
session-global `beta_volume_index`, with a focused real regression recorded in
`trial_binding_regression.json`.

No neural export or fitting/scoring is authorized by the completed stimulus
export. The unmasked regular-volume estimate from actual headers is
1,776,291,996,908 bytes, far outside the 5 GB stimulus-only output allocation.
This is a storage estimate, not a neural outcome or a requested allocation.


## Requested mask-provenance double-check — 2026-09-30

The scientist asked us to double-check and re-supplied the already recorded
stimulus access acceptance. This supplied no new mask provenance, ROI
amendment, neural export allocation, or fitting/scoring authority.

The current official provider ROI documentation was checked at
https://laion-fmri.hebartlab.com/rois.html. It still defines `laion-general`
from main-experiment GLMsingle R² and its LAION parcels as subdivisions.
It does not specify a selecting trial/image pool or authenticate exclusion
of `tau_test` and OOD responses.

A narrowly scoped metadata inspection found no construction JSON/README in
the stored ROI tree; segmentation TSVs identify parcels and colors only.
The stored specification-only sidecar at
`br_autoresearch_data/steward_acquisition/laion_fmri_mixed_role_20260922/dataset/derivatives/glmsingle-tedana/sub-01/sub-01_task-images_space-T1w_stat-rsquare_desc-R2mean_statmap.json`
defines an across-session `np.nanmean` of TYPED R² maps, with sessions 01–30
and 32–34 listed as sources. It does not identify a training-only image pool,
state tau-test/OOD exclusions, or directly link this R2mean map to construction
of `laion-general`. These source specifications therefore do not prove mask
contamination and do not establish the required independence.

Conclusion: whole-VC mask eligibility remains unknown under
`PROTOCOL_TABLE.yaml:roi_contract`. An authoritative construction reference
or an eligible replacement whole-VC mask with selection provenance is still
needed; a narrower ROI requires a prospective scientific amendment. The
regular-only stimulus export remains complete and reusable. No voxel values,
R² map values, neural outcomes, or OOD payloads were read. No new scientific
attempt, job, protocol change, validation replay, or successor was launched.
The original Goal remains incomplete at the same human/scientific gate.


### Concrete mask remedy prepared — not adopted, 2026-09-30

A metadata-only inventory identifies 29 documented independent-localizer
native T1w mask identities common to all five subjects. The proposed fixed
union, exclusions, coverage limitations, source paths and a separate bounded
mask-only allocation are in
`steward_provisioning/proposed_independent_visual_mask_amendment.md`.
This would change the primary ROI definition and requires explicit approval.
No masks were materialized, protocol fields changed, or jobs submitted.
The contract-preserving alternative is the unsent provider provenance request.
Mask amendment would not authenticate beta preprocessing or open neural fitting.


### Scientist-approved mask remedy — job 46141362

The scientist's “ok” adopts the proposed fixed 29-component independent
localizer union and separate mask-only allocation. The previous whole-VC
construction-provenance requirement is superseded for EP21's primary ROI by
this explicit, pre-outcome amendment. The masks remain an adapted union with
incomplete coverage; do not claim source whole-VC equivalence.

`native_visual_mask_preparation`, attempt-001, Slurm46141362 is the single
submitted mask-only job. No duplicate, neural job or successor may be launched.
The worker reads only the 145 named independent masks, preserves native grids,
returns C-order full-grid voxel identities and actual support QC. Resource
limits remain 1 CPU/2 GiB/10 minutes/250 MB durable/500 MB temporary.

The full16-analysis ASTRA program was validated with zero warnings. Its amended
compatibility snapshot also exported successfully to a named revision,
`astra/compatibility/v0.0.14/plan/independent-mask-amendment-2026-09-30/astra.yaml`,
with the prior snapshot and full program preserved. No terminal export ran.
The exact next action is to babysit this job and inspect its real mask-only
report/figure; do not repeat qualification or unchanged checks.

Remaining scientific dependencies are regular neural provisioning, eligible
beta/preprocessing provenance and native beta-grid binding, exact runtime/assets,
required conformance, valid lock and later neural execution authority.
No beta or main-experiment R² values are authorized by the mask allocation.


### native_visual_mask_preparation — attempt-001 completed — 2026-09-30

Slurm46141362 COMPLETED, exit0:0, elapsed00:00:33; batch MaxRSS131196KiB.
Exactly one approved mask-only job ran. All145 independent components passed
3D/finite/binary/nonempty and exact within-subject grid checks. Five saved unions
and ascending full-grid C-order voxel index arrays passed real round-trip
regression. Native support: sub-01 12,295; sub-03 12,897; sub-05 10,891;
sub-06 12,023; sub-07 12,057 voxels. Adapted/incomplete visual support is
disclosed; no source whole-VC equivalence or comparability margin is claimed.

Published artifacts: `steward_provisioning/independent_visual_masks/`, through
the new read-only `inputs/laion_independent_visual_masks/` interface. Source
components, hashes, coded geometry and identities are in `mask_report.json`.
Publication evidence: `steward_provisioning/mask_publication_check.json`; native
accounting: `steward_provisioning/mask_job_completion.json`. Total durable
output including delivered QC figure copies is917,905bytes, within250MB;
temporary storage was bounded within500MB and removed on successful finish.

QC figure: `figures/independent_visual_mask_support_qc_2026-09-30.png` (also SVG
and JSON), rendered from actual union counts, visually inspected and shared.
It establishes native spatial support under the explicit amendment; it is not
neural-conversion evidence or a whole-cortex coverage measurement.

A required metadata-only native-grid binding read five session01 beta headers
(348bytes each; no responses). Bitwise affine equality failed because it was
stricter than NIfTI-1 float32 header precision. The diagnosed comparison was
repaired using the gamma8 forward-error bound at all8 voxel-box corners, with
finite/nonsingular affines, same subject/shape/unit tags and a strict nearest-
native-index guard in both directions. The focused five-real-header regression
passed; max corner discrepancy2.8876e-5voxel. Original coded headers and all
mask/voxel data remain unchanged; unknown unit tags remain unknown. Report:
`steward_provisioning/native_mask_beta_header_binding.json`; the failed strict
comparison is retained in the `_strict_float_equality.json` sibling. This is
engineering precision compatibility, not identical-construction provenance or
a scientific comparability margin. Every future exported-session header must
use the same native-grid binding before payload extraction.

The amended mask dependency is now satisfied. The original Goal remains
incomplete at the closed neural boundary: eligible beta/preprocessing
provenance, physically regular-only neural provisioning, exact model assets/
runtime, required remaining conformance, valid lock and later execution
authority remain necessary. No beta values, main-experiment R² values, neural
features/scores, OOD payloads/derivatives or sibling results were opened. The
original scientific budget, training/test roles and stopping rules are intact.

Full16-analysis ASTRA schema/semantics passed with zero warnings and the passing
result was reused after unchanged program content. Changed contract fields
passed focused parsing/closed-authority checks. The closeout compatibility
export succeeded as an idempotent named revision (`astra/mask-closeout-export.log`),
with the full program and prior snapshot preserved. No export problem remains
for this EP21 route; shared legacy tools are unchanged. No terminal export,
canonical scientific acceptance/reward/transition, successor or further job
is launched. Exact next step: obtain/authenticate the remaining eligible
regular neural/preprocessing inputs within the existing authority before any
new export or fitting/scoring; do not repeat completed mask/stimulus work.
