# EP15 anatomical common support

Selected prospectively by the coordinator on2026-10-03, after source metadata
qualification and before anatomy values, Task-A reliability or candidate
scores. The seven scientific margins remainNULL. This specification applies
identically to all24 participants and all eventual model classes.

The [provider dataset route](https://github.com/DiedrichsenLab/Functional_Fusion/blob/main/Functional_Fusion/dataset.py)
passes the individual forward SUIT transform and native cerebellar mask to
[AtlasMapDeform](https://github.com/DiedrichsenLab/Functional_Fusion/blob/main/Functional_Fusion/atlas_map.py).
The transform is a pull field: at template coordinates it stores absolute
native-world XYZ coordinates, not displacement vectors. Inverse SUIT images
are unnecessary. The implementation below preserves this direction but does
not copy the provider's optional smoothing, clipped indices, mask-weight
renormalization or NaN-summing behavior.

Only the following exact members of each `MDTB_<pid>.zip` may be read:

- `<pid>/anat/<pid>_space-SUIT_xfm.nii`
- `<pid>/anat/<pid>_desc-cereb_mask.nii`
- `<pid>/anat/<pid>_label-GMc_probseg.nii`

No T1, beta, variance, Task-A/B signal, atlas fitted to MDTB, private reliability
or participant role enters this stage. Prior all-participant header and metadata
reports are reused to identify all24 people and detect changed source geometry.

## Coordinates and lattice

Use each image's coded sform as its zero-based voxel-index-to-world affine,
consistent with [NIfTI affine precedence](https://nipy.org/nibabel/nifti_images.html).
Require a finite invertible affine and explicit millimetre spatial units. No
fallback affine, guessed unit conversion, voxel reorientation or axis flip is
allowed. A missing/contradictory unit stops before that image's values are read.
The current common forward-field shape is141×95×87×1×3, with1mm spatial steps.
All24 live transform grids must agree with the saved headers and each other.

Define template voxel indices `(i,j,k)` as the Cartesian product
`range(0,141,3) × range(0,95,3) × range(0,87,3)`. Order them lexicographically
in `(i,j,k)`, with `k` varying fastest (NumPy C order for the3-D support mask).
This gives43,616 candidate points in a47×32×29 lattice. Template world coordinates
are `A_SUIT @ [i,j,k,1]`; the lattice affine multiplies the first three columns
of `A_SUIT` by3 and retains its origin. Its endpoints are
`(70,-100,-75)` and `(-68,-7,9)`mm under the qualified source grid.

This is an **EP15 SUIT-derived3mm lattice**, not the canonical `SUIT3` atlas.
It loads no canonical functional mask or fitted atlas. Because every point is
an exact source-grid voxel centre, read the transform's three XYZ components
directly there; do not interpolate or smooth the deformation itself.

NIfTI voxel storage is decoded in Fortran order (`i` varies fastest in the
stored image), independently of the C-order output-point ordering above.
Respect the header's datatype, byte order, voxel offset and scale. For a finite
nonzero `scl_slope`, values are `stored*scl_slope+scl_inter`, requiring a finite
intercept. NIfTI slope0 or NaN means no scaling (1,0); an infinite slope fails.
No implicit division by255 is permitted. The finite decoded GMc range must fit
probability units within1e-6 of[0,1], allowing only floating-point encoding
roundoff; no values are clipped and the inclusion threshold remains exactly0.5.
No finite GMc values is an unresolved encoding; sampled nonfinite corners fail
the local support rule below, without invalidating other locations globally.

## Anatomical inclusion and strict sampling

For a template point, read its scaled forward XYZ vector `p`. Nonfinite `p`
makes that participant's support false at the point. Compute native image
coordinates separately as `u_GM = inv(A_GM) @ [p,1]` and
`u_mask = inv(A_mask) @ [p,1]`. The GMc and functional-mask grids need not be
the same. Use the actual coded affine for each.

For each trilinear sample use `floor(u)` and the usual eight Cartesian corner
weights. Every strictly positive-weight corner must lie inside the image and
have a finite scaled value. Exact zero-weight corners are ignored, including
the otherwise-outside corner at an exact last voxel centre. Coordinates are
never clamped, snapped to a nearby voxel or padded. Missing/nonfinite values
never become zero, and surviving weights are never renormalized. Invalid
samples return unknown internally and fail anatomical support.

A participant supports the template point only when:

1. all three forward XYZ values are finite;
2. strict trilinear GMc sampling is valid and its interpolated scaled probability
   is at least0.5; and
3. **every** positive-weight trilinear corner on the native cerebellar-mask grid
   is finite and strictly positive.

The third rule defines the future functional-beta stencil on the already
matched native mask grid. It is stricter than retaining any positive mask
weight. This stage does not read or validate beta values; the future reader
must check every consumed beta's grid and all positive-weight values. No later
Task-A/B result may revise this support.

**Amendment, October 12, 2026 (scientist):** for the observed-beta descriptive
route only, the derived common finite availability mask defined in the
[finite-support follow-up](proposed_finite_support_followup.md) replaces this
support (`scientist_amendment_20261012_finite_support` in `SEARCH_POLICY.yaml`).
This specification, its 208-location result and the no-revision rule are
otherwise unchanged and still govern every other route.

Take the Boolean intersection across all24 participants, completing everyone
even if the running intersection becomes empty. No person, optional inverse
transform or outcome-based criterion can be omitted. An empty final support
is a preparation failure to resolve, not a scientific result or permission
to lower the GM threshold or reduce the cohort.

## Artifacts, resources and limits

`common_support.npz` contains only the all24 Boolean mask, its lattice affine,
selected template world coordinates and selected source-grid indices in the
fixed order. Export no participant-native coordinates, deformation, tissue
image or individual support mask. The JSON report records completion, aggregate
sampling counts, observed header scaling and resource use. No partial common
mask is exported if the stage fails or times out.

The prior headers bound72 selected payloads at633,483,020 bytes total, with
largest payload13,984,380 bytes. Read them sequentially and retain only one
participant's three small arrays at a time, with no extraction or anatomical
copy on disk. The full24 run uses1CPU/2GiB, a10minute envelope and an internal
540second deadline. Preparation scratch is capped at2GiB; in this implementation
there are no extracted scratch images, and the fixed common arrays are below
4MiB uncompressed. The report includes first-participant elapsed time as an
internal cost observation. No automatic budget expansion or restart is allowed.

Freezing this anatomical intersection precedes any Task-A reliability work.
It does not assign the12/12 roles, prove functional finite coverage or independent
measurement errors, activate scientific margins, or authorize candidate scores.
