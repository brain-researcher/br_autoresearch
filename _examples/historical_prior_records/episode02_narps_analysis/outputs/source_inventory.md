# Source inventory

Status: **pass for bounded successor testing**. This inventory accessed filenames,
tabular metadata, and NIfTI headers only; it did not read BOLD or neural outcome
voxel values.

## Source resolution and scope

- All three workspace references are readable Sherlock-native absolute OAK symlinks.
- Raw participants: 108 (54 EI,
  54 ER).
- Expected task runs: 432 (four for every participant).
- Complete required-test-source run intersection:
  432.
- Required availability counts: raw BOLD 323, events
  432, MNI 2 mm fMRIPrep BOLD 432,
  MNI 2 mm masks 432, confounds
  432, and FitLins run designs 432.
- Optional raw BOLD links readable: 323/432;
  dangling raw annex links: 109. The bounded recomputation
  uses complete fMRIPrep BOLD, so this does not block either test.
- Every inspected run design contains `gain_demean`, `loss_demean`, and an intercept.

The machine-readable audit and per-run manifest are
`outputs/source_inventory.json` and `outputs/source_inventory_runs.tsv`.

## Image and design compatibility

- MNI preprocessed BOLD and masks use one common spatial signature:
  `True`.
- Existing gain/loss subject effect and variance maps are complete
  (432/
  432) and use one spatial
  signature: `True`.
- FitLins reports version `0.11.0.post0.dev16`, estimator
  `nilearn`, space
  `MNI152NLin2009cAsym`, and configured smoothing
  `5:run:iso`.

The existing FitLins statistics are usable as provenance and a matched 5 mm
reference, but they cannot by themselves identify the v1 0-versus-8 mm smoothing
mechanism. A bounded recomputation is therefore needed only for the prospectively
selected smoothing endpoints; the v1 3 x 3 x 3 grid is not needed.

## Candidate support

- Amplitude-versus-shape: supported by matched 0/8 mm group maps from a single
  fixed first-level design, evaluated before and after spatial scale normalization.
- Mask-edge/weighting: supported by the same subject effect/variance estimates,
  outcome-blind run-mask coverage shells, and equal versus inverse-variance
  aggregation.
- Task-version/sample composition: supported as a stratified robustness/control
  readout (EI and ER separately), not as independent confirmation.
- Continuous-versus-thresholded: supported by fixed continuous metrics and one
  frozen BH-FDR q=0.05 readout; threshold is not tuned.

## Compute and storage plan

Inventory ran on compute node `sh03-06n06.int` inside Slurm job
`39398079`. Durable artifacts remain in `outputs/`.
High-frequency GLM checkpoints and transient arrays will use an episode-specific
directory below `/scratch/users/zijiao` (available and writable:
`True`), with only final reproducibility
artifacts copied to this workspace. No analysis will run on a login node.

## Stage-0 decision

No required source is missing, and semantic compatibility is sufficient to freeze
a two-test successor contract. The raw checkout's broader annex completeness is
not required for these tests, and no replacement data will be downloaded.
