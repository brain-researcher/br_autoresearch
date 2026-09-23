# Design-example data contract: NARPS development and mixed-gambles audit

## Status

This is a planning contract. It records locally observed file-system facts and
the required data firewall; it does not authorize source access or a run.

## Data roles

| Resource | Role | Outcome exposure | Permitted use |
| --- | --- | --- | --- |
| EP01/EP02 curated artifacts | Historical lineage | Fully exposed | Generate mechanisms, reproduce historical facts, test ingestion |
| OpenNeuro `ds001734` | Adaptive development | Fully exposed by EP01/02 | Multi-round search, grouped CV, ablation, reliability |
| OpenNeuro `ds000005` | External transport audit | Metadata and filenames inventoried; audit neural values not inspected during drafting | One frozen audit after configuration lock |

No partition of `ds001734` is described as untouched confirmation. Subject
folds are algorithmic generalization checks inside an outcome-exposed dataset.

## Historical lineage packet

Before any future instantiation, copy or content-address a minimal read-only
packet from the two historical examples rather than reading mutable sibling
outputs. It should include:

- frozen contracts and terminal bundles;
- source and multiverse manifests;
- variance-attribution and mechanism tables;
- mask/scaling diagnostics;
- review objections and unresolved-question record; and
- hashes and explicit parent-loop identities.

The packet may motivate hypotheses but may not be counted as new evidence.

## Development source: NARPS `ds001734`

### Locally observed references

| Component | Historical OAK location |
| --- | --- |
| Raw metadata/events | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/input/ds001734` |
| fMRIPrep derivatives | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/fmriprep/ds001734/derivatives` |
| FitLins products/designs | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses/ds001734` |

Historical EP01/EP02 inventories reported 108 participants, 54 EI and 54 ER,
four expected runs per participant, and 432 runs in MNI152NLin2009cAsym 2 mm.
These counts must be revalidated and hash-bound; they are not guaranteed merely
because the paths exist.

The drafting inventory additionally observed:

| Field | Observed value |
| --- | --- |
| OpenNeuro DOI | `10.18112/openneuro.ds001734.v1.0.5` |
| Raw checkout commit | `0ad017e7f83ecda943fb85cc93bb3a8a122b2e60` |
| FitLins analysis checkout commit | `a3c7aeb3406634e2d11e37ddcb6efb7d4e5b99e3` |
| fMRIPrep | `21.0.2` |
| License | `CC0` in local dataset metadata |

A Git commit is not by itself a content manifest for DataLad/annex payloads or
a potentially modified worktree. Before launch, hash every BOLD, events,
confounds, mask, design, and contrast-definition file actually used and record
any worktree difference from the commits above.

### Development folds

Create deterministic, task-version-stratified subject folds from opaque hashes,
not outcome values. The fold manifest is immutable before search. Each scored
trial must report gain/loss and EI/ER separately plus the frozen aggregate.

Because all NARPS outcomes were previously exposed, fold locking prevents
implementation leakage during this program but does not restore independent
confirmation status.

## Audit source: OpenNeuro `ds000005`

### Locally observed identity

| Field | Observed value |
| --- | --- |
| Dataset | `ds000005`, Mixed-gambles task |
| Local repository | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses/ds000005` |
| Observed repository commit | `4d5640924a477a4b4402bfe04a8fde3e19e78fbe` |
| Participants | 16 subject directories |
| Runs | 3 per subject; 48 preprocessed BOLD files observed |
| fMRIPrep | `21.0.1` in local dataset metadata |
| FitLins | `0.11.0.post0.dev16`; Nilearn estimator; existing output used 5 mm run-level smoothing |
| Space | FitLins metadata declares `MNI152NLin2009cAsym` |
| Relevant contrasts | `paragain`, `paraloss`; also `paragainvloss`, `distindiff`, and `rt` are present but nonprimary |
| Existing run-level products | 240 effect and 240 variance maps across five contrasts were observed by filename inventory |
| License in local metadata | Public Domain Dedication and License v1.0, with community attribution norms |

These observations establish feasibility, not audit results. During this draft,
no `ds000005` BOLD or effect-map voxel values were opened for candidate
selection.

### Why `ds000005` is the primary audit

It is independently collected, uses a mixed-gambles paradigm with parametric
gain and loss contrasts, and has locally materialized raw, fMRIPrep, FitLins,
design, effect, and variance layers. It therefore provides a closer transport
test than splitting NARPS subjects again while remaining small enough for a
CPU-only S0/S8 recomputation.

Its sample size is only 16. The audit therefore emphasizes paired map-change
prediction, contrast agreement, deterministic halves, bootstrap uncertainty,
and leave-one-subject-out stability rather than a strong population-neuroscience
claim.

## Required audit firewall

Before launch, materialize two different access surfaces:

1. `inputs/development/`: lineage packet plus immutable `ds001734` references,
   available throughout search.
2. `inputs/audit_manifest/`: dataset identity, counts, hashes, contrast names,
   and schema only; no readable BOLD, effect, variance, or group-map arrays.

The actual audit payload must be held outside the discovery process and exposed
only to a separate one-shot audit command after configuration lock. Preferred
implementations are a server-owned scorer or a permission-separated read-only
mount. A visible symlink plus a Markdown instruction is not sufficient.

The audit runner must record:

- the frozen configuration hash it received;
- the exact audit source commit and file manifest;
- first and last access timestamps;
- that it emitted only declared metrics and diagnostics; and
- that no subsequent development trial was accepted.

## Compatibility checks before freezing

Without opening audit neural arrays, verify:

- subject/run completeness and readable events/confounds/designs;
- corresponding semantics and units for NARPS `gain_demean`/`loss_demean` and
  audit `paragain`/`paraloss`;
- spatial grids, affine conventions, masks, and resampling requirements;
- that S0 and S8 can be recomputed from fMRIPrep BOLD with unchanged task
  regressors and auditable nuisance construction;
- contrast estimability in every retained run;
- software/container availability and numerical tests; and
- required OAK and scratch capacity.

Failure of a compatibility check may change feasibility or block launch, but it
must not reveal audit effect-map values or motivate a candidate.

## Prohibited data use

- Do not use the existing `ds000005` 5 mm group or subject maps to choose the
  mechanism.
- Do not tune masks, thresholds, floors, kernels, or uncertainty procedures on
  audit outcomes.
- Do not replace `ds000005`, add subjects, or switch contrasts after audit
  access because the result is unfavorable.
- Do not pool NARPS EI/ER into an unqualified gain-minus-loss claim.
- Do not treat a filesystem checkout, internal fold, or prior publication as
  an independently untouched confirmation by itself.

## Reuse checklist

- [ ] Content-addressed lineage packet created without live sibling reads.
- [ ] `ds001734` source identities, counts, versions, and hashes revalidated.
- [ ] `ds000005` commit and complete audit manifest revalidated.
- [ ] Contrast crosswalk approved by a scientist before neural audit access.
- [ ] Development fold manifest frozen.
- [ ] Permission-separated audit runner demonstrated on synthetic fixtures.
- [ ] Search policy registered canonically and bound by `search_policy_ref`.
- [ ] CPU, scratch, wall-time, and failure budgets accepted.
- [ ] Audit thresholds frozen after synthetic calibration.
