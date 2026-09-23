# Episode 01 data and exposure contract

## Status

This is a pre-launch data contract. It records known identities, exposure
roles, and the firewall that must exist before execution. It does not authorize
source access, materialization, computation, or audit opening.

## Role table

| Resource | Frozen role | Outcome exposure | Permitted use |
| --- | --- | --- | --- |
| `narps_prior_v1` and `narps_prior_v2` | Historical development lineage | Fully exposed | Hypothesis generation, historical reproduction, ingestion tests; never new evidence |
| OpenNeuro `ds001734` | Adaptive development | Fully exposed by the prior runs | Search, grouped folds, ablation, reliability, and full-development replication |
| Synthetic fixtures | Outcome-blind calibration | Generated | Operator/metric recovery, degeneracy tests, ledger tests, audit-runner rejection tests |
| OpenNeuro `ds000005` | One-shot external transport audit | Metadata/schema visible; neural arrays forbidden to discovery | Exactly one locked evaluation through a permission-separated runner |

No partition of `ds001734` is untouched confirmation. Its subject folds are
algorithmic development checks inside an exposed dataset. `ds000005` is
independently collected, but because it is public and locally present, its
status is procedure-sealed for this program rather than globally pristine.

## Historical prior lineage

The machine-readable prior declaration is
[`inputs/prior_lineage/PRIOR_LINEAGE_MANIFEST.json`](inputs/prior_lineage/PRIOR_LINEAGE_MANIFEST.json).
It pins the repository revision, representative artifact hashes, legacy
canonical identities, exposure status, and required materialization set.

The current packet is manifest-only. Before launch, a controller-owned step
must copy every declared small artifact into a content-addressed, read-only
packet under this episode, verify all hashes, and record aggregate byte counts.
It must not symlink, hardlink, or read live sibling outputs during the search.
Large historical maps are deliberately excluded. They may be reused only from
an immutable controller-provisioned snapshot with a complete file manifest;
otherwise they are recomputed from the pinned `ds001734` source.

The two prior records have different canonical states. `narps_prior_v1` was
observed complete with `closed_no_candidate`; `narps_prior_v2` was observed at
`AWAITING_REWARD`. Its returned `record_reward` action expired on
2026-08-18 and is not replayable authority. This episode neither resolves nor
inherits either state. A fresh canonical prepare must create a new identity; a
collision, if the service actually reports one, is reconciled separately rather
than by acting on the legacy record from this workspace.

## Development source: NARPS `ds001734`

### Locally observed references

| Component | Historical OAK location |
| --- | --- |
| Raw metadata/events | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/input/ds001734` |
| fMRIPrep derivatives | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/fmriprep/ds001734/derivatives` |
| FitLins products/designs | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses/ds001734` |

Historical inventories reported 108 participants, 54 EI and 54 ER, four
expected runs per participant, and 432 runs in `MNI152NLin2009cAsym` 2-mm
space. These counts are priors to revalidate, not launch facts.

| Identity field | Previously observed value |
| --- | --- |
| OpenNeuro DOI | `10.18112/openneuro.ds001734.v1.0.5` |
| Raw checkout commit | `0ad017e7f83ecda943fb85cc93bb3a8a122b2e60` |
| FitLins analysis checkout commit | `a3c7aeb3406634e2d11e37ddcb6efb7d4e5b99e3` |
| fMRIPrep version | `21.0.2` |
| License | `CC0` in the locally observed metadata |

A Git commit does not content-address DataLad/annex payloads or prove a clean
worktree. Before development, hash every BOLD, events, confounds, mask, design,
and contrast-definition file actually used, record all worktree differences,
and freeze a source Merkle root.

### Required development manifests

The following must be generated outcome-blind, independently reviewed, hashed,
and bound in `SEARCH_POLICY.yaml` before candidate-discriminating work:

- source and payload inventory with file bytes and hashes;
- participant/run eligibility and exclusion reasons;
- opaque-hash, task-version-stratified subject folds;
- gain/loss and EI/ER environment definitions and weights;
- S0, S8, and global exact-`K8_ref` operator DAGs;
- comparison mask, resampling, edge-shell, and degeneracy rules;
- contrast units/sign crosswalk and nuisance construction;
- software/container, code, seed, and resource-metering manifests; and
- synthetic fixtures with expected scores and failure modes.

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

Before launch, materialize distinct access surfaces:

1. `inputs/development/`: verified prior packet, immutable `ds001734` payload,
   folds, manifests, and synthetic fixtures.
2. `inputs/audit_manifest/`: dataset identity, schema, counts, source Merkle
   commitment, contrast names, and compatibility receipts only. It contains no
   readable BOLD, effect, variance, residual, or group-map arrays.
3. An external trusted audit surface unavailable to the proposal model,
   search workers, development evaluator, cache, log process, and ordinary
   episode workspace until a valid configuration-lock hash exists.

A visible symlink, a path plus prose warning, or the current instruction-only
launcher boundary is insufficient. The audit runner must accept exactly one
configuration-lock hash, reject a second scientific audit transaction, disable
undeclared mounts and network access, and record:

- exact source commit, payload Merkle root, and file manifest;
- received configuration-lock and ledger-prefix hashes;
- first and last payload-access timestamps;
- emitted metric/diagnostic schema and whether anything candidate-
  discriminating became visible;
- infrastructure failures and whether a retry is mechanically eligible; and
- permanent rejection of subsequent development records.

A frozen eligible retry is permitted only when the failed attempt emitted no
candidate-discriminating value. It must reuse the same lock hash, payload root,
runner, and command; remain inside the same logical audit transaction; and add
a hash-chained attempt receipt. The exact eligible-failure list and maximum
attempt count are launch-blocking and require scientist signoff.

## Compatibility checks before any audit neural access

Using metadata, schemas, synthetic fixtures, and development data only:

- verify subject/run completeness and events/confounds/design readability;
- obtain scientist signoff on semantic, sign, and unit correspondence between
  NARPS `gain_demean`/`loss_demean` and audit `paragain`/`paraloss`;
- freeze spatial grids, affines, masks, and resampling rules;
- demonstrate that S0 and S8 can be recomputed from fMRIPrep BOLD with
  unchanged task regressors and auditable nuisance construction;
- verify contrast estimability for every retained run;
- validate deterministic bootstrap, interval multiplicity, LOSO, and
  half-split code on synthetic data;
- prove rejection before lock and after a consumed transaction, plus acceptance
  only of a receipt-linked eligible continuation; and
- profile CPU, memory, concurrency, scratch, and wall time inside the frozen
  episode budget.

Compatibility failure may block launch or cause a technical terminal. It may
not expose audit map values or motivate a mechanism.

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

## Launch-readiness checklist

- [ ] Fresh canonical read confirms that neither legacy identity/action is reused; any reported prepare collision is handled separately.
- [ ] Prior packet materialized locally, content-addressed, and independently verified.
- [ ] `ds001734` identities, payloads, counts, versions, and worktree state revalidated.
- [ ] Development source, split, environment, mask, operator, and container hashes frozen.
- [ ] `ds000005` exposure history and complete payload commitment independently audited.
- [ ] Contrast semantics/sign/units and all numerical thresholds scientist-signed.
- [ ] Permission-separated evaluator and one-shot audit runner synthetically qualified; eligible retry failures and maximum attempts signed.
- [ ] Memory, concurrency, CPU, scratch, and wall-time profile fits the registered budget.
- [ ] Canonical adaptive program and immutable `search_policy_ref` registered.
- [ ] Fresh canonical EP01 Goal/loop identities created; neither prior identity reused.
