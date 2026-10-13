# EP15 first-participant Task-A mapping and cost pilot

Adopted by the coordinator on 2026-10-04 at 20:37 UTC after review of the
specification, implementation and nine proposed fixtures, before any Task-A
numeric access. Compute validation and source execution remain pending. This is the
next `source_qualification` stage recorded in `outputs/README.md`, not candidate
scoring, private reliability, role assignment, or activation of the seven
scientific margins. The coordinator records adoption and the attempt before
execution. No Task-B image stream is authorized.

Select the lexicographically first participant from the same 24 unique IDs in
the completed metadata, anatomical-header, and anatomical-support records.
There is no participant argument or outcome-dependent substitute. All reports
must indicate successful prerequisite completion and identify the same source
root. Use the completed `metadata_join001/regressor_members.tsv` to select
that person's `ses-s1` Task A. Require its full 736-row design, consistent
instruction flags, exact provider member names and one member per run/regressor,
all 16 runs, and the same 29 noninstruction condition identities and per-run
multiplicities as the metadata result. Instruction betas are excluded. The
ZIP index may be inspected for these identities; its Task-B images may never
be opened. The permitted numeric members are exactly the selected person's
forward SUIT field and the noninstruction Task-A beta names from this plan.

Reuse `anatomical_support001/common_support.npz` without modification. Validate
its Boolean mask, selected source-grid indices, template coordinates, affine,
and ordering against the already adopted stride-three lattice and the saved
all-participant intersection; require exactly 208 selected points. This is
consistency validation of the existing support, not a new support search.
Read the selected person's forward field with the existing NIfTI reader and
its saved expected header. Sample its absolute native-world XYZ values at the
208 exact source-grid centers. Keep native coordinates only in memory.

Every consumed beta must have the same live spatial grid as both the saved
selected Task-A header and the prior native cerebellar-mask header. The saved
headers must agree before opening beta values; each live beta must match that
grid before its voxel payload is read. Require coded invertible sform and
millimetre spatial units; honor datatype, byte order, voxel offset and actual
NIfTI scaling. Reuse `read_image`, `scale_values`, `world_to_voxel`, and
`strict_trilinear` from `build_anatomical_support.py` without changing them.
The reused reader's historical error wording says “anatomical”; the pilot
records its actual stage and permitted Task-A access separately. Its datatype,
offset, 32 MiB member bound and scaling checks apply unchanged.

Map each beta to all 208 points using the adopted strict trilinear rule.
Every positive-weight corner must be in bounds and finite; zero-weight
corners are ignored. Never clamp coordinates, replace nonfinite values,
renormalize surviving weights, shrink support, or exclude a condition, run,
or participant. The prior all24 intersection establishes the native-mask
stencil; no new mask or tissue values are read. A changed geometry or invalid
functional stencil is a preparation failure.

For each condition and run, compute the arithmetic mean of all its planned
noninstruction regressors. Require the exact planned multiplicity, including
repeated regressors if any. Average those run maps with equal run weights over
runs 1–8 and independently runs 9–16. This yields two 29-by-208 half-map arrays
in memory only. Require finite arrays. Report only whether every half-map has
nonzero centered spatial variation, using the equivalent exact test that at
least one location differs from its first value; no magnitude threshold is
introduced. A constant map makes the later correlation undefined and stops
the pilot without dropping that condition. Do not calculate or export any
reliability correlation, participant reliability score, half-map, native
coordinate, beta magnitude, model score, or participant role.

The sole output is `task_a_mapping_pilot.json` in a fresh coordinator-selected
`outputs/runs/source_qualification/` attempt. It records the deterministic
participant ID, operational counts, unchanged-support and finite/nonconstant
booleans, actual uncompressed stream bytes and decoded payload bytes, elapsed
time, peak RSS, and completion/failure status. A failure report records the
stage and diagnostic but no array or value. No partial map is saved. Finite and
nonconstant qualification is preparation evidence only; it does not establish
independent errors or scientifically useful reliability.

Run serially with one CPU, 4 GiB memory, an internal 540-second deadline and
external 600-second timeout inside the coordinator's bounded attended session.
Numeric thread counts are one. The preparation scratch cap is 2 GiB; no image
is extracted or written to scratch, and the retained map array is under 1 MiB.
Each image is read sequentially with the existing 32 MiB member bound. No
automatic retry or resource expansion is authorized. Measure the complete
first-person Task-A work before proposing any all24 mapping/private-reliability
allocation. Passing this pilot does not launch that panel or assign the 12/12
roles.

Coordinator command, after adoption and recording the attempt:

```sh
python outputs/tools/pilot_task_a_mapping.py --output-dir outputs/runs/source_qualification/task_a_mapping_pilot001 --max-seconds 540
```

Focused tests cover exact Task-A selection and denied Task-B/instruction opens,
run/condition multiplicity, pre-payload grid failure, real NIfTI scaling and
strict finite stencils, equal-regressor/equal-run arithmetic, and the report's
absence of maps, native coordinates and reliability values. Run them only in
the coordinator's verified compute allocation.
