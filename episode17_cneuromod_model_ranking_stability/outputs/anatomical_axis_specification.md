# Anatomical labels and the original GLMsingle voxel axis

The missing-unit repair is adopted on October 4 at 23:15 UTC. The completed
[units001 witness](runs/anatomical_axis/units001/unit_witness_report.json)
authenticates all four original source-mask headers as millimetres, with zero
corner difference from the saved union geometry. The mapper may interpret an
unknown union unit as millimetres only when this exact complete, source-bound
report and the current numeric geometry agree. It changes an in-memory header
copy only, reports observed unknown units separately from the derived unit,
and never overrides explicit units. Preserve failed attempt001; execute the
repaired mapping in fresh attempt002 with the original resource bounds.
Independent review and all 23 focused fixtures passed before this adoption.

Status: bounded structural stage adopted by the coordinator on 2026-10-04
at22:45UTC after independent review; implementation and compute checks precede
source execution. The adopted scope is exactly the eight structural volumes
and resource/output bounds below. The optional DESIGNINFO scalar remains
unselected. This stage does not establish response units, select finite
voxels, or freeze `N`, spatial quotas, banks, reliability thresholds or margins.
The coordinator owns execution. Literal role-filtered copies of the original
B/C/D columns are scientifically independent of response-unit qualification;
their own required capacity observation remains unresolved after the separate
Oak quota-query timeout recorded in the execution log.

## Fixed contract and present evidence

The nominated ROIs remain DK `ctx-lh-fusiform` (1007) and
`ctx-rh-fusiform` (2007), separately weighted equally within each of sub-01,
sub-02, sub-03 and sub-06, then equally across participants. They are anatomical
fusiform parcels, not functional FFA. GOAL states: “Do not select another
anatomical region because the nominated relation fails.” DATASETS requires
nearest-neighbor mapping to the GLMsingle T1w mask, aligned B/C/D axes, common
finite support, small-ROI refusal and fixed support quotas before scoring.

The completed header-only
[`source_metadata/attempt003` report](runs/source_metadata/attempt003/metadata_report.json)
establishes matching B/C/D stored dimensions and records union-mask headers.
It explicitly leaves voxel values and response units unopened. Equal shapes
alone do not establish a voxel correspondence. No structural or neural payload
was opened while preparing this proposal.

The provider source is the release's pinned `THINGS/glmsingle` commit
`1f16b9e982ed47cb5d07ae1cf03d479c2d8521a3`, under
`/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/restricted/cneuromod-things-1.0.1-restricted-raw`.
The exact code evidence below is relative to that submodule's
`code/glmsingle/` directory.

## Source-supported column mapping

`GLMsingle_preprocBOLD.py:47–109` constructs and saves the original union mask,
applies it to each T1w BOLD series, and writes the resulting ordered units to
HDF5. `GLMsingle_run.m:43–46` reads and transposes that matrix for MATLAB;
it does not permute the units. `GLMsingle_betasPerTrial.py:50–70,98–99`
explicitly checks the raw `modelmd` column count against that original mask
and unmasks the raw columns through it before any later clean-mask selection.

The provider's pinned requirements specify Nilearn 0.9.2. Its
[masking implementation](https://raw.githubusercontent.com/nilearn/nilearn/0.9.2/nilearn/masking.py)
uses C-order Boolean indexing. Thus, for a three-dimensional binary original
mask `M`, raw zero-based column `j` maps to

```
flat_indices = flatnonzero(M.ravel(order="C"))
(i, k, l) = unravel_index(flat_indices[j], M.shape, order="C")
world_xyz = mask_affine @ [i, k, l, 1]
```

Nilearn's `unmask(order="F")` allocation preference does not change the
Boolean-index column order. Do not replace this mapping with MATLAB-style
Fortran flattening of the anatomical volume.

GLMsingle accepts already collapsed units and reshapes those units to a
singleton spatial representation. The recorded h5py storage shape is
`[presentations,1,1,voxels]`; the corresponding access convention is
`modelmd[p,0,0,j]`. Preserve this explicit convention and all original columns,
rather than inferring axes from whichever dimensions happen to have matching
lengths. The structural stage needs only the already recorded B/C/D headers,
not any value from `modelmd`.

The exact original mask for participant `SS` is:

```
THINGS/glmsingle/sub-SS/glmsingle/input/sub-SS_task-things_space-T1w_label-brain_desc-union_mask.nii
```

`GLMsingle_cleanmask.py:55–61` derives `unionNonNaN` by examining standardized
BOLD across runs. It is a later signal-derived selection, not the original
raw-beta axis or an independently qualified finite-response universe. Neither
its payload, `unionNaN`, provider noise ceilings, z-scored trial-beta exports,
nor image-average exports belongs in this structural stage.

## Provider-native response convention; physiological units remain unverified

Coordinator adoption, October 4 at 23:17 UTC: existing pinned provider-code
evidence sufficiently qualifies the convention for the planned named-model
comparisons as **delivered GLMsingle B/C/D trial estimates from runwise,
voxel-standardized BOLD, retaining the provider's scaling**. Independent review
agrees. Historical binary ambiguity stays disclosed; neither DESIGNINFO
inspection nor physiological amplitude calibration is an additional gate.
Spatial mm concerns coordinates only and does not resolve response units.

A fixed nonzero per-voxel response multiplier cancels from voxelwise R² when
training means and predictions scale consistently, and from repeat-half
Pearson reliability when both halves share that scaling. This does not assert
that B/C/D are merely rescalings: beta edges compare the delivered stage
recipes. Cross-voxel CKA and absolute variance thresholds can depend on scaling;
their conventions must be frozen accordingly. Finite support, nondegenerate
variance, physical handoffs and other scoring prerequisites remain required.
No physiological percent-BOLD, physical amplitude change or uniquely causal
historical implementation claim follows. No neural value was opened.

The provider preprocesses each run with per-voxel `zscore` before dropping the
first two TRs, then calls GLMsingle without setting `wantpercentbold`.
Consequently the fitted input is standardized BOLD, not an unmodified scanner
baseline. This establishes the documented input transformation, not the
delivered models' physiological units.

There is a concrete version ambiguity: the checked-in GLMsingle gitlink is
`1ab54a65edd3ea41a6133d4b4ecb78a9c7296684`, whereas the release's THINGS README
names the Courtois fork at `c4e298e`. Both inspected implementations default
`wantpercentbold` to one and rescale C/D by the absolute mean volume. The
gitlinked version saves an `opt` structure in `DESIGNINFO.mat`. These facts
support checking a saved option; they do not prove the historical executable
or establish physiological percent-BOLD units.
[Gitlinked implementation](https://raw.githubusercontent.com/cvnlab/GLMsingle/1ab54a65edd3ea41a6133d4b4ecb78a9c7296684/matlab/GLMestimatesingletrial.m),
[README-named fork](https://raw.githubusercontent.com/courtois-neuromod/GLMsingle/c4e298e/matlab/GLMestimatesingletrial.m).

If separately selected, the smallest useful configuration read is one named
scalar per participant:

```
THINGS/glmsingle/sub-SS/glmsingle/output/T1w/DESIGNINFO.mat
HDF5 candidate: /opt/wantpercentbold
```

Inspect only that path's type/shape and, if it is a direct real numeric
one-element scalar or `(1,1)` dataset of at most eight bytes, its finite
zero/one value. Missing file/path, references, a different layout or another
value is unresolved; do not enumerate HDF5 groups, follow references or read
neighboring design/neural datasets. The code documents a MATLAB structure,
not this file's actual HDF5 layout, so the nominated HDF5 path remains
conditional. The older fork does not provide the same `DESIGNINFO` evidence.
Even a value of one confirms an option rather than a physiological baseline;
zero also requires investigation because the inspected C/D rescaling is
unconditional. No `meanvol` payload is selected here. Preserve
`response_units_verified: false` in structural reports because that flag is
not a physiological-unit certificate. The source-convention decision above
is distinct. The optional scalar route remains unselected and is unnecessary
for that decision.

## Exact atlas objects and source-reference bridge

DATASETS calls acquired `aparcaseg` volumes candidates. The fixed release
points `THINGS/fmriprep` to
`5155ba0d43836a4451141b90b5a70507f41205cf` in the official
`courtois-neuromod/things.fmriprep` repository. The local directory is not an
initialized submodule: Git there resolves to the parent release. Its empty
tracked-file listing therefore cannot establish anatomical file absence.

A bounded lookup of `sub-SS/anat` at that fmriprep version returned HTTP404;
these were incorrect layout assumptions, not evidence of absent anatomy.
The pinned root metadata and `.gitmodules` identify anatomy through
`sourcedata/smriprep`, pointing to the official `courtois-neuromod/anat.smriprep`
commit `a69af03816ae28730b91ed334e873bfad2391e04`. The release separately pins
its initialized local `anatomical/smriprep` checkout at
`4726ea8258371256455d43cc649452caacc00930`.

The four exact files below have identical Git symlink blobs at these two
anatomical versions. This is a specific object-identity bridge despite the
different repository commits; it is not an assumption that arbitrary
derivatives are interchangeable. Paths are relative to
`anatomical/smriprep/` under the restricted release root:

| Participant | Exact path | Shared Git symlink blob | Existing annex content identity |
| --- | --- | --- | --- |
| sub-01 | `sub-01/anat/sub-01_desc-aparcaseg_dseg.nii.gz` | `c4548de42676e7ce1caf6819647c5db48567ea26` | `MD5E-s1882667--2ba3c3702f7b3239a5520995c4b5714a.nii.gz` |
| sub-02 | `sub-02/anat/sub-02_desc-aparcaseg_dseg.nii.gz` | `e954136823dd71c2af609a066b248dc75b6980e8` | `MD5E-s1967036--aff7cfe74e25391963fc46adf8059ecd.nii.gz` |
| sub-03 | `sub-03/anat/sub-03_desc-aparcaseg_dseg.nii.gz` | `9f75dacef6061923b137180d2e52a4f2093b9e52` | `MD5E-s1935990--f87c0b09c80c4c79ace45407167d8402.nii.gz` |
| sub-06 | `sub-06/anat/sub-06_desc-aparcaseg_dseg.nii.gz` | `89cb93181acf0cbbc0ce7016733df5055667b80c` | `MD5E-s1756349--6fe93a02e27c1fc6c590db53fd92c311.nii.gz` |

All four symlinks are 140-byte Git objects. Their annex declarations total
7,542,042 compressed bytes. Payload availability, declared-length/content
agreement, image headers and geometry remain uninspected. This proposal
selects no download. A missing acquired object is reported explicitly.

The pinned fmriprep `dataset_description.json` identifies version 20.2.5;
the local anatomy descriptor identifies fMRIPrep 20.2.1. Preserve both as
provider provenance. The anatomical gitlink and identical segmentation
objects support the native-anatomy bridge; the proposed geometry check must
still establish the selected coded T1w frame. A filename alone is insufficient.
The matching-version [fMRIPrep output documentation](https://fmriprep.org/en/20.2.1/outputs.html)
states that anatomical outputs without a `space-` entity use the original
T1w space and distinguishes that space from FreeSurfer's conformed `fsnative`
space. No historical launch command has independently been verified here.

The proposed exact crosswalk is label 1007 → left fusiform and label 2007 →
right fusiform. It is confirmed directly by the pinned fmriprep
[`desc-aparcaseg_dseg.tsv`](https://raw.githubusercontent.com/courtois-neuromod/things.fmriprep/5155ba0d43836a4451141b90b5a70507f41205cf/desc-aparcaseg_dseg.tsv)
and supported by the
[FreeSurfer DK label definition](https://raw.githubusercontent.com/freesurfer/freesurfer/v7.3.2/distribution/FreeSurferColorLUT.txt).
White-matter labels 3007/4007 are not members. This reference fixes label
semantics; it does not assert that the local derivatives were produced with
FreeSurfer 7.3.2. Source metadata must authenticate DK aparc+aseg and its
participant/T1w reference, excluding a Destrieux, DKT-only, aseg-only or
functional-ROI substitution.

## Proposed structural mapping, after exact input selection

Read exactly four original union masks and four authenticated DK segmentation
volumes, sequentially by fixed participant order. Do not read T1w intensities,
BOLD, beta values, reliability maps or any response-based mask. The method is:

1. Require each mask to be three-dimensional, finite and exactly binary after
   NIfTI scaling, with its count of ones equal to every corresponding B/C/D
   header's original voxel count. Preserve every selected mask column.
   Require each segmentation's scaled labels to be finite integers; never
   round a noninteger field into a valid atlas.
2. Require source metadata to identify both volumes in the same participant
   T1w reference. Record shape, orientation, units, coded qform/sform and
   selected affine. Select only a finite invertible coded affine
   authenticated as the shared T1w reference. When both forms are eligible
   in that same frame, prefer sform, consistent with NiBabel's documented
   precedence. If only one is authenticated in that frame, select that form
   explicitly; do not let a valid other-frame sform override a T1w qform.
   An unresolved or invalid active frame stops qualification rather than
   triggering an automatic fallback. With both codes zero, the fallback
   affine does not qualify the coordinate frame. Require
   millimeter spatial units or the prospectively fixed exact conversions
   from meters (multiply coordinates by1000) or microns (multiply by0.001).
   Unknown units are not inferred from voxel size. Different image grids
   are expected and are not a failure.
3. If both qform and sform are coded as the same coordinate frame, require
   their eight voxel-center corners to agree within 0.001 mm, a prospective
   floating-encoding check. If they encode different frames, retain both
   and require source metadata to identify which is the shared T1w frame;
   do not infer that from image overlap. An unresolved frame/affine conflict
   stops mapping. Header consistency alone does not prove anatomical
   registration accuracy.
4. Resample the integer atlas to the exact original mask grid with
   nearest-neighbor interpolation and outside-FOV label zero. The coordinate
   map is `inv(atlas_affine) @ mask_affine`; use the recorded SciPy/Nilearn
   order-zero resampling implementation with constant-zero boundary mode
   and no prefilter. Do not smooth, dilate, fill holes, reorient the original
   mask, fit a registration, or choose an alignment using outcomes. Record
   the implementation version and its nearest-neighbor boundary convention.
5. Enumerate the C-order union-mask column map and attach the resampled label
   to each original column. Report left/right counts separately for all four
   participants, source-label counts, out-of-FOV mask-column counts and the
   world-coordinate extent of each parcel. Keep nonfusiform columns in the
   axis artifact with their labels; no neural or role-filtered array is
   rewritten by this stage.

The affine policy follows the
[NiBabel NIfTI documentation](https://nipy.org/nibabel/nifti_images.html);
the proposed sampling operation follows the
[Nilearn resampling implementation](https://raw.githubusercontent.com/nilearn/nilearn/0.9.2/nilearn/image/resampling.py).
The source documentation and geometry establish a structural mapping only;
they cannot certify common finite response support.

An empty nominated parcel in any participant is an explicit anatomical
refusal, not an invitation to replace the ROI, merge hemispheres or drop a
participant. A one-voxel parcel is explicitly too small for any nontrivial
change in support membership. Counts of two or more merely permit later
feasibility assessment; they do not declare the scientific sample adequate.
After structural counts are known, select a concrete `N`, spatial bins,
quotas and bank in a separate prospective specification, before neural/model
scores. Later common-finite and calibration eligibility can only reduce
capacity; if a frozen quota becomes infeasible, refuse that configuration
rather than silently reduce `N` or move voxels between bins.

## Bounded implementation and outputs to select

Proposed allocation contribution: one CPU, 4 GiB, 300 s internal deadline and
360 s external timeout plus 10 s kill grace, no GPU, retries or network fetches.
Combine this short stage with another already useful task-owned compute
allocation. Its cost comes from eight structural volumes and one resampling
per participant. Process/release one participant at a time. Before decoding,
cap each mask at 4 million grid voxels and each atlas at 32 million, with at most
eight bytes per stored scalar; refuse larger inputs without enlarging the
scope automatically. These bounds keep working arrays within 4 GiB without
assuming unknown file sizes. Record elapsed time and maximum RSS.

Proposed fresh output directory: `outputs/runs/anatomical_axis/attempt001/`.
Publish one compressed tabular `voxel_axis.csv.gz` with participant,
zero-based model column, zero-based C-flat index, voxel `i,j,k`, world
`x,y,z` in millimeters, integer atlas label and nominated hemisphere (blank
outside the two parcels), plus `anatomical_axis_report.json`. Cap the combined
uncompressed table at 256 MiB; preserve any interrupted table as incomplete.
The terminal report lists all eight ROI counts, exact selected source paths,
geometry decisions, count agreement and explicit unresolved/refusal reasons.

Set `structural_voxel_axis_complete` and
`anatomical_fusiform_mapping_complete` only after all four participants,
their B/C/D header-count checks and all outputs succeed. On any failure or
deadline, clear both flags. Structural counts may still be reported as
partial evidence with participant scope. Always leave
`finite_response_support_complete`, `response_units_verified`,
`reliability_support_complete`, `source_roles_complete` and
`candidate_scoring_run` false. No `N`, quota, selected voxel bank or neural
calibration result is an output of this proposal.

The exact atlas inputs now have the source-reference bridge above.
Implementation may begin when the coordinator selects this bounded payload
scope; image-header and geometry evidence is obtained inside the structural
stage. Ordinary indexing/resampling conventions are delegated engineering
choices; neither the current contract nor this proposal requires a blanket
scientist-approval pause for recovering public source metadata.
