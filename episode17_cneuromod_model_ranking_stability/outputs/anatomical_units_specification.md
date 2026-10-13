# Original union-mask spatial-unit witness

Status: the coordinator adopted this exact four-object diagnostic on
2026-10-04, after anatomical attempt001 stopped with
`spatial_units_unresolved`. Implementation and source-free fixtures precede
coordinator execution. This is a diagnostic within `anatomical_axis`, mode
`unit_witness`; it does not complete the anatomical mapping or qualify neural
response units. The original anatomy runner and source files remain unchanged.

## Why these four headers answer the observed failure

The release's GLMsingle submodule is pinned to
`1f16b9e982ed47cb5d07ae1cf03d479c2d8521a3`.
[GLMsingle_preprocBOLD.py](https://github.com/courtois-neuromod/things.glmsingle/blob/1f16b9e982ed47cb5d07ae1cf03d479c2d8521a3/code/glmsingle/GLMsingle_preprocBOLD.py)
sorts `sub-SS/ses-*/func/*_part-mag_space-T1w_desc-brain_mask.nii.gz`,
constructs the union with `intersect_masks(..., threshold=0)`, and saves it.
The provider pins Nilearn0.9.2.
[That implementation](https://github.com/nilearn/nilearn/blob/0.9.2/nilearn/masking.py)
uses the first input's affine, requires all inputs to have the same shape and
affine, and passes that affine to `new_img_like` without `copy_header=True`.
[The image constructor](https://github.com/nilearn/nilearn/blob/0.9.2/nilearn/image/image.py)
defaults to `copy_header=False`: it retains the affine and supplies no copied
header to the image class. This is a concrete route by which spatial units
can disappear while the numeric geometry is preserved. It is not itself
evidence that the original units were millimetres.

The pinned fMRIPrep version is20.2.5. Its
[T1w resampling workflow](https://github.com/nipreps/fmriprep/blob/20.2.5/fmriprep/workflows/bold/registration.py)
uses the native T1w reference and mask to generate the sampling reference and
resample the brain mask. Its
[output sink](https://github.com/nipreps/fmriprep/blob/20.2.5/fmriprep/workflows/bold/outputs.py)
writes this product with `space=T1w`, `desc=brain`, `suffix=mask`.
These conventions support the native participant T1w frame; actual header
units and coded geometry must still pass the checks below. A two-millimetre
voxel size or apparent atlas overlap is insufficient.

## Fixed source identities and transport addresses

The release pins `courtois-neuromod/things.fmriprep` to
`5155ba0d43836a4451141b90b5a70507f41205cf`.
Bounded nonrecursive participant and first-session directory metadata show
that `ses-01` is the first lexical session for every participant. Each
`ses-01/func` contains three matching masks, with `run-1` first. This is the
provider's original glob order, not an inference from the retained-run roster.

For each `SS` in01,02,03,06 the exact path is:

```
sub-SS/ses-01/func/sub-SS_ses-01_task-things_run-1_part-mag_space-T1w_desc-brain_mask.nii.gz
```

| Participant | Func Git tree | Pointer Git blob | Annex directory | Exact annex key |
| --- | --- | --- | --- | --- |
| sub-01 | `599d47a225ee10b1f4149388cd37c2498552112d` | `c58b7e29bc132eacac90a2074376b2aff81c3b41` | `XZ/5k` | `MD5E-s7602--aaf135ebac3bced281281b703b20fe9f.nii.gz` |
| sub-02 | `1c1cbb31a3d7383a45229f3343b33e8c511d8792` | `872a9a32863a5c3538c47e173a04f88cd77864c9` | `95/14` | `MD5E-s7127--2510838dea8c0347033e769384730d38.nii.gz` |
| sub-03 | `a19052c2bfe5ad2dbb0538eb92ea1a345dd5942b` | `b368132e9d2772b51882fb164143d4c955506ded` | `Wg/7x` | `MD5E-s6862--f0b3582ca9870346b989f6fac8814e80.nii.gz` |
| sub-06 | `8fb0ffbbce7135e4150515e49f2851a8385a3340` | `461c49c12417693384a1e47610f9c50fb3d8a129` | `X2/Xp` | `MD5E-s6478--1ef4ce1667283f97a566f505908b9a11.nii.gz` |

Each Git entry is mode120000 and its pointer is137 bytes. The decoded pointer
Git SHA was verified during metadata inspection; the runner rechecks the
fixed pointer reconstruction before each selected request. The pointer is
`../../../.git/annex/objects/{directory}/{key}/{key}`.
The same pinned repository's `.datalad/config` gives dataset ID
`81d8c126-b10f-4e8b-9eab-6825964c8552`.
Using the existing CONP RIA layout, the exact four request URLs are:

```
https://sftp.conp.ca/users/cneuromod/ria-conp/81d/8c126-b10f-4e8b-9eab-6825964c8552/annex/objects/XZ/5k/MD5E-s7602--aaf135ebac3bced281281b703b20fe9f.nii.gz/MD5E-s7602--aaf135ebac3bced281281b703b20fe9f.nii.gz
https://sftp.conp.ca/users/cneuromod/ria-conp/81d/8c126-b10f-4e8b-9eab-6825964c8552/annex/objects/95/14/MD5E-s7127--2510838dea8c0347033e769384730d38.nii.gz/MD5E-s7127--2510838dea8c0347033e769384730d38.nii.gz
https://sftp.conp.ca/users/cneuromod/ria-conp/81d/8c126-b10f-4e8b-9eab-6825964c8552/annex/objects/Wg/7x/MD5E-s6862--f0b3582ca9870346b989f6fac8814e80.nii.gz/MD5E-s6862--f0b3582ca9870346b989f6fac8814e80.nii.gz
https://sftp.conp.ca/users/cneuromod/ria-conp/81d/8c126-b10f-4e8b-9eab-6825964c8552/annex/objects/X2/Xp/MD5E-s6478--1ef4ce1667283f97a566f505908b9a11.nii.gz/MD5E-s6478--1ef4ce1667283f97a566f505908b9a11.nii.gz
```

These are metadata-derived addresses, unprobed during implementation. No
alternative endpoint, mask, session, redirect or automatic retry is selected.

## Bounded check and interpretation

Use one CPU, at most1GiB, one worker,120 seconds internally and180 seconds
externally plus a10-second kill grace. Make four sequential GETs, each with a
15-second timeout and at most8192 bytes. Their exact compressed lengths are
7602,7127,6862,6478 bytes, totaling28,069. Require HTTP200, the unchanged URL,
identity content encoding, no transfer encoding, and exact Content-Length.
Read only the declared length, bounded by the aggregate limit, and verify
the complete compressed object's existing annex MD5 before interpreting it.
Missing HTTP framing is unresolved rather than authorization for a fallback.

Whole compressed structural objects are transported in memory for identity
authentication. This is structural-byte access, even though no voxel array
is decoded. Record request/read timestamps and actual transported counts.
Persist no compressed objects. Use `zlib.decompressobj(wbits=31).decompress`
with `max_length=352`, once per object, to decode only the NIfTI-1 header and
four-byte extension indicator. Do not flush, follow extensions, invoke an
image loader or request an array. Refuse a short/non-NIfTI-1 header, non-3D
shape, overlapping data offset, or units other than explicit `mm`.

Reuse the adopted anatomy geometry rules: active forms must have codes1 or2,
their frames must be resolved as native T1w, their finite affine matrices
must be invertible, and two eligible forms must agree at all eight
voxel-center corners within0.001mm. Prefer an eligible sform, otherwise the
eligible qform. Never use an uncoded fallback.

Read only the existing bounded source-metadata report to reconstruct each
original union's recorded shape, pixdim, coded forms, srows and quaternion.
Do not reopen a union mask or any B/C/D, atlas, image or signal-derived mask.
Require equal shape and maximum selected-affine corner difference at most
0.001mm. Temporarily labeling the reconstructed geometry as millimetres is
the hypothesis being compared against the witness's explicit units; it is
not an observation of units in the saved union header. Report it explicitly
as witness-derived only after the source-chain and geometry checks pass.

The output is
`outputs/runs/anatomical_axis/units001/unit_witness_report.json`.
All four participants and exact aggregate transport accounting must complete
before `anatomical_spatial_units_witness_complete=true`. Any failure or late
deadline clears completion while preserving observed accesses. Partial
participants cannot qualify the four-person axis. Success supports the
inherited spatial-unit convention for these original union geometries; it
does not qualify response units, finite neural support, atlas mapping,
registration accuracy, role handoffs, audit access or model scoring. Root
must separately select the focused anatomy-runner integration and attempt002.

Coordinator commands from the episode directory, using its module interpreter:

```bash
timeout -k 10s 120s "$EP_PYTHON" -m unittest discover -s outputs/tests -p test_anatomical_units.py -v
timeout -k 10s 180s "$EP_PYTHON" outputs/tools/qualify_anatomical_units.py --out outputs/runs/anatomical_axis/units001
```

The fixtures use synthetic headers and mocked transport only. No source
requests, test execution or header/array reads occurred while writing them.
