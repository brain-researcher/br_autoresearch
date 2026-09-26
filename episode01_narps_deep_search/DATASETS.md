# Episode 01 data and exposure contract

## Status

This data contract records known identities, exposure roles, and the firewall
required before candidate-discriminating work and protected audit access. An
explicit scientist instruction may start the episode, but this file alone does
not authorize source access, materialization, computation, or audit opening.

## Role table

| Resource | Frozen role | Outcome exposure | Permitted use |
| --- | --- | --- | --- |
| `narps_prior_v1` and `narps_prior_v2` | Historical development lineage | Fully exposed | Hypothesis generation, historical reproduction, ingestion tests; never new evidence |
| OpenNeuro `ds001734` | Adaptive development | Fully exposed by the prior runs | Search, grouped folds, ablation, reliability, and full-development replication |
| Synthetic fixtures | Outcome-blind calibration | Generated | Operator/metric recovery, degeneracy tests, and targeted audit-boundary tests |
| OpenNeuro `ds000005` | One-shot external transport audit | Identity, metadata, counts, and contrast names visible; neural arrays forbidden to discovery | Exactly one locked evaluation through a permission-separated runner |

No partition of `ds001734` is untouched confirmation. Its subject folds are
algorithmic development checks inside an exposed dataset. `ds000005` is
independently collected, but because it is public and locally present, its
status is procedure-sealed for this program rather than globally pristine.

## Historical prior lineage

The machine-readable prior declaration is
[`inputs/prior_lineage/PRIOR_LINEAGE_MANIFEST.json`](inputs/prior_lineage/PRIOR_LINEAGE_MANIFEST.json).
It records the repository revision, legacy canonical identities, exposure
status, and hashes of selected legacy artifacts. Those existing hashes are
retained because the two prior runs had overlapping historical names and live
elsewhere in the repository: matching a hash once for each small artifact
actually imported is the lightest direct way to prove which exposed prior was
used. This is an identity and leakage control, not a general checksum policy.

Candidate search does not require materializing every declared legacy file.
If a legacy artifact is actually consumed, copy that artifact into the
episode-local, read-only prior packet and match it once to the identity already
declared in the lineage manifest. That episode-local copy plus its one-time
match to the existing lineage hash is the verified content-addressed prior
packet required by repository policy. Unused files need not be materialized,
and no new hashes are generated. Do not create a second manifest, aggregate
byte-count receipt, independent-review gate, or repeated verification pass.
The run must not symlink, hardlink, or read live sibling outputs. Large
historical maps remain excluded; recompute them from the identified
`ds001734` source unless a scientist explicitly provides a read-only snapshot.

The two prior records have different canonical states. `narps_prior_v1` was
observed complete with `closed_no_candidate`; `narps_prior_v2` was observed at
`AWAITING_REWARD`. Its returned `record_reward` action expired on
2026-08-18 and is not replayable authority. This episode neither resolves nor
inherits either state. If this episode later enters Brain Researcher review,
it must use a new record; any service-reported collision is reconciled
separately rather than by acting on a legacy record from this workspace.

## Development source: NARPS `ds001734`

### Locally observed references

| Component | Historical OAK location |
| --- | --- |
| Raw metadata/events | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/input/ds001734` |
| fMRIPrep derivatives | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/fmriprep/ds001734/derivatives` |
| FitLins products/designs | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses/ds001734` |

Historical inventories reported 108 participants, 54 EI and 54 ER, four
expected runs per participant, and 432 runs in `MNI152NLin2009cAsym` 2-mm
space. These counts are priors to revalidate, not verified input facts.

| Identity field | Previously observed value |
| --- | --- |
| OpenNeuro DOI | `10.18112/openneuro.ds001734.v1.0.5` |
| Raw checkout commit | `0ad017e7f83ecda943fb85cc93bb3a8a122b2e60` |
| FitLins analysis checkout commit | `a3c7aeb3406634e2d11e37ddcb6efb7d4e5b99e3` |
| fMRIPrep version | `21.0.2` |
| License | `CC0` in the locally observed metadata |

A provider version or Git commit does not by itself establish scientific
usability. Before development scoring, record the source version and the paths
actually used, directly check retained subject/run completeness and file
readability, and record any source-tree difference that changes those inputs.
Per-file hashes and a source Merkle root are not required unless a concrete
identity discrepancy is observed.

### Required development records

Record the following outcome-blind definitions once before they are used for
candidate scoring. A concise table or configuration file is sufficient; no
custom schema, independent attestation, or checksum layer is required:

- source/version and the retained participant/run inventory;
- participant/run eligibility and exclusion reasons;
- deterministic, task-version-stratified subject folds created without
  outcome values;
- gain/loss and EI/ER environment definitions and weights;
- S0, S8, and global exact-`K8_ref` operator DAGs;
- comparison mask, resampling, edge-shell, and degeneracy rules;
- contrast units/sign crosswalk and nuisance construction;
- versioned analysis code, seeds, and the resource ceilings; and
- any synthetic fixture actually used, with its expected score or failure mode.

Every scored trial reports all four primary environments separately plus the
frozen aggregate. Fold creation uses identifiers and declared stratifiers, not
outcome values.

## Audit source: OpenNeuro `ds000005`

### Previously observed metadata-only identity

| Field | Observed value |
| --- | --- |
| Dataset | `ds000005`, mixed-gambles task |
| Repository commit | `4d5640924a477a4b4402bfe04a8fde3e19e78fbe` |
| Participants | 16 subject directories |
| Runs | 3 per participant; 48 preprocessed BOLD filenames observed |
| fMRIPrep | `21.0.1` in local metadata |
| FitLins | `0.11.0.post0.dev16`; Nilearn estimator |
| Existing derivative smoothing | 5-mm run-level smoothing; forbidden for candidate selection |
| Space | `MNI152NLin2009cAsym` in FitLins metadata |
| Primary contrasts | `paragain`, `paraloss` |
| Nonprimary contrasts | `paragainvloss`, `distindiff`, `rt` |
| License | Public Domain Dedication and License v1.0 in local metadata |

The inventory previously observed filenames for 240 effect and 240 variance
maps across five contrasts. That observation establishes feasibility only. It
does not make those maps development inputs and does not authorize opening
their voxel values.

### Why this audit is scientifically bounded

`ds000005` is independently collected, uses parametric gain and loss contrasts,
and is small enough for a CPU-only S0/S8 recomputation. It is therefore a
useful transport test for a locked methods mechanism. With only 16
participants, it cannot support a broad population-neuroscience claim; paired
map-change prediction, simultaneous uncertainty, deterministic halves, and
leave-one-subject-out stability remain visible.

### Required technical firewall

Maintain two direct access roles:

1. A development surface containing the identified `ds001734` inputs,
   deterministic folds, frozen scoring definitions, and only the legacy prior
   artifacts actually used.
2. A protected audit surface unavailable to the proposal model, search
   workers, development evaluator, caches, logs, and ordinary episode
   workspace until an immutable configuration-lock version exists. Before
   then, those processes may see only `ds000005` identity, metadata, counts,
   contrast names, and compatibility information that contains no neural
   array values.

A visible symlink, a path plus prose warning, or an instruction-only same-UID
task boundary is insufficient. The permission-separated audit runner must
reject access before lock and reject a new scientific opening after any
candidate-discriminating output has been visible. Keep one simple access log
with the lock version, opening/start/end times, whether outcome information was
emitted, and any concrete infrastructure failure. Payload Merkle roots,
complete file manifests, report schemas, receipt chains, exact runner or
interpreter attestations, and generic network/mount audits are not required.

A retry is permitted only for a concrete infrastructure failure that emitted
no candidate-discriminating value. Record it as a continuation of the same
logical opening, reuse the immutable scientific configuration, and do not
change the candidate, data, contrasts, thresholds, or uncertainty procedure.

## Compatibility checks before any audit neural access

Using metadata, any targeted synthetic fixture, and development data only:

- verify subject/run completeness and events/confounds/design readability;
- obtain scientist signoff on semantic, sign, and unit correspondence between
  NARPS `gain_demean`/`loss_demean` and audit `paragain`/`paraloss`;
- freeze spatial grids, affines, masks, and resampling rules;
- demonstrate that S0 and S8 can be recomputed from fMRIPrep BOLD with
  unchanged task regressors and auditable nuisance construction;
- verify contrast estimability for every retained run;
- validate deterministic bootstrap, interval multiplicity, LOSO, and
  half-split code on synthetic data;
- demonstrate rejection before lock and after a consumed transaction, and
  permit continuation only after a recorded no-outcome infrastructure failure;
  and
- make a practical CPU, memory, scratch, and wall-time estimate before the
  relevant Slurm launch.

Compatibility failure may make candidate scoring or audit opening ineligible
and may cause a technical terminal. It may not expose audit map values or
motivate a mechanism.

## Prohibited data use

- Do not use existing `ds000005` 5-mm subject or group maps to choose a
  mechanism, mask, threshold, floor, operator, or uncertainty procedure.
- Do not expose audit neural arrays through a symlink, inherited mount, cache,
  notebook, log, or environment variable.
- Do not replace the audit dataset, add subjects, switch contrasts, promote the
  runner-up, or resume development after an unfavorable audit.
- Do not pool NARPS EI/ER into an unqualified gain-minus-loss claim.
- Do not count prior-run cells, internal folds, public availability, or a
  filesystem checkout as new independent confirmation.

## Minimal readiness checkpoints

Before development scoring:

- [ ] Legacy records are treated only as exposed priors, expired actions are not replayed, and no live sibling output is read.
- [ ] Each legacy artifact actually used is an episode-local copy matching its existing lineage identity; unused legacy files need not be materialized.
- [ ] The `ds001734` source/version, retained subjects and runs, exclusions, and readable inputs are recorded.
- [ ] Subject split, environments, contrasts, mask, S0/S8 pathways, exact-`K8_ref` operator, metric, and degeneracy rule are frozen.
- [ ] The chronological trial record and proposal-before-execution rule are active.

Before the one-shot audit opening:

- [ ] Contrast semantics, sign, units, numerical thresholds, and simultaneous uncertainty rule are frozen and scientist-approved.
- [ ] One immutable configuration-lock version names the selected mechanism, comparator set, code version, data roles, split, scoring definitions, and report fields.
- [ ] The permission-separated runner rejects pre-lock access and search workers cannot see audit neural arrays.
- [ ] A practical resource estimate fits the episode budget.

These checkpoints are performed once at the stage they govern. Generic
checksum manifests, schema validators, receipts, attestations, and repeated
preflights are not additional gates.
