# Dataset Contract — Episode 17

## Source and access status

CNeuroMod-THINGS 1.0.1 is the only neural source. Its required 128.145 GiB
neural/structural subset and release-provenance archive are currently
stored outside this canonical checkout at
`/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/restricted/cneuromod-things-1.0.1-restricted-raw`.
The root location manifest identifies that tree as `ep17_restricted_raw`; it
was relocated by same-filesystem rename on 2026-09-24. Operational acquisition
logs, checksums, and the transfer program remain with the restricted source and
are intentionally not versioned in this code repository. Candidate scoring and
audit access remain closed until outcome-blind role assignment, role-filtered
handoffs, and the remaining scientific contracts are complete.

Provisioning source bytes is not permission to inspect neural arrays.

## Physical-location record

[`DATA_LOCATION_MANIFEST.json`](../DATA_LOCATION_MANIFEST.json) is the only
repository record that maps the logical locations below to machine paths:

- `ep17_restricted_raw` is the canonical restricted neural/structural source at
  `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/restricted/cneuromod-things-1.0.1-restricted-raw`;
  and
- `private_steward_acquisition` contains the CNeuroMod stimulus archive at
  `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition`.

Both trees were relocated by same-filesystem rename on 2026-09-24.

Neither current source is a role-filtered episode handoff. No path listed in
the manifest changes the access boundary or opens protected neural outcomes.

## Frozen source

Primary sources:

- data paper: <https://www.nature.com/articles/s41597-026-06591-y>;
- official repository:
  <https://github.com/courtois-neuromod/cneuromod-things>;
- archived release: <https://doi.org/10.5281/zenodo.17881592>; and
- THINGS stimulus terms: the `LICENSE.txt` in the pinned stimulus repository.

| Component | Frozen identity |
| --- | --- |
| CNeuroMod-THINGS release | tag `1.0.1` |
| Root commit | `0f22b6c5eca4001e86fa936f3b6c2569a0179970` |
| `THINGS/glmsingle` gitlink | `1f16b9e982ed47cb5d07ae1cf03d479c2d8521a3` |
| `THINGS/behaviour` gitlink | `b84bf6d5c18e53e78c6278bfcb5d4e0e6afff214` |
| `anatomical/smriprep` gitlink | `4726ea8258371256455d43cc649452caacc00930` |
| Participants | `sub-01`, `sub-02`, `sub-03`, `sub-06` |
| Spatial representation | native T1w GLMsingle products |

Moving branches are inadmissible. A replacement release requires fresh source
verification and a pre-outcome amendment.

The repository releases the participant data under CC0 and project code under
MIT. Those terms do not cover the THINGS image pixels, whose separate
research/noncommercial terms must be accepted and recorded independently.

## Why one dataset is sufficient for this episode

The primary question is not cross-dataset transport. It is whether controlled
model relations change under isolated measurement operations and whether a
development result survives on concepts that never influenced fitting.

CNeuroMod supplies:

- four intensively sampled participants;
- a common THINGS concept structure;
- repeated image presentations;
- aligned Type-B, Type-C, and Type-D GLMsingle products; and
- enough concepts to separate development, support/calibration, and audit.

This makes B→C and C→D identifiable within one derivative family and removes
the missing-intermediate problem that affected the former NSD design.

It does not create external biological replication. The four people are the
only biological units, and the audit reuses those same people.

## Verified restricted source inventory

The canonical restricted source contains the four participants' B/C/D matrices,
public D-ceiling maps, GLMsingle designs, masks and run metadata, restricted
per-trial annotation sources, and anatomical `aparcaseg` candidates required
by this episode. The neural/structural subset is approximately 128.145 GiB;
the official `cneuromod-things-1.0.1.tar.gz` release archive is retained beside
it as provenance. Neither payload is part of a code-only checkout.

The root location manifest records the restricted source by logical identifier;
the machine path remains outside Git and is disclosed only to the trusted
builder; downstream consumers receive
approved, role-filtered handoffs. Source-level acquisition evidence stays with
that workspace rather than being duplicated as tracked manifests, receipt
files, or checksum sidecars.

The fMRI stimulus archive was transferred separately and is not included in
the neural-source byte total above. The private-steward-held, pinned
`images_fmri.zip` is 917,286,854 bytes. Its research/noncommercial terms were
accepted and recorded on 2026-09-22.
The archive remains encrypted and unextracted in steward quarantine; controlled
extraction, per-image hashing, exact event-image alignment, and a role-safe
feature handoff remain feature-computation blockers. None weakens the
neural-data seal.

## Acquisition state and access boundary

Source provisioning completed on 2026-09-21 against the frozen release and
gitlink identities above. The required files and provenance archive were
verified before the final tree was made read-only; its annex links resolve
inside the same restricted tree. No MAT, NIfTI, HDF5, or annotation array was
parsed as part of acquisition.

The exact transfer logs and integrity inventory are operational records beside
the restricted source. They are not episode inputs and are not carried by Git.
A new checkout must receive a separately verified, role-filtered handoff; it
must not infer data-role access from repository files.

The raw source remains mixed-role and restricted after transfer. It must not be
mounted to the adaptive search worker.

## Neural products

For every participant, the canonical restricted source contains:

| Stage | File | Scientific interpretation |
| --- | --- | --- |
| B | `TYPEB_FITHRF.mat` | fitted HRF, no GLMdenoise, no ridge |
| C | `TYPEC_FITHRF_GLMDENOISE.mat` | fitted HRF plus GLMdenoise, no ridge |
| D | `TYPED_FITHRF_GLMDENOISE_RR.mat` | fitted HRF plus GLMdenoise and ridge |
| D public ceiling | `*_stat-noiseCeilings_statmap.nii.gz` | positive-control diagnostic only |

The B/C/D matrices physically contain development, calibration, and audit
trials together. Their presence on OAK is not outcome access.

The public D ceiling cannot be called a B or C ceiling. It may contain
information aggregated across stimulus roles, so it stays sealed from the
search worker and is opened only as a post-lock, nonterminal positive control.
EP17 recomputes stage-specific B, C, and D ceilings from the frozen calibration
concepts under one estimator.

For initial isolated coverage, B→C and C→D use the same
outcome-independent common anatomical support; support-policy edges hold the
beta stage at D; and the required raw-versus-normalized ceiling edge also uses
D on one fixed NC-eligible voxel set. Stage-specific B/C ceiling analyses are
prespecified successors and diagnostics, not permission to change voxel
identity on a pure beta edge.

## Outcome-blind eligibility and roles

### Step 1: build the common universe

The trusted builder reads only an allowlist of events fields (image/concept
IDs, presentation index, session/run/trial position, acquisition order, and
repetition lag, plus the provider-defined schedule-exception flag), design
indices, shapes, affine/voxel metadata, and structural missingness flags.
Recognition responses, correctness, reaction time, and other behavioral
outcomes are excluded. The builder does not summarize neural values.

An image is primary-eligible only if:

- its immutable image and concept IDs agree across all four participants;
- its required presentations are structurally complete in all four;
- its trial mapping agrees across B, C, and D; and
- it passes prespecified, outcome-independent integrity rules.

The publication-level expectation is 720 common concepts. Public summaries
also suggest that `sub-01`, `sub-02`, and `sub-03` have 4,320 unique
images while `sub-06` has 3,840. Those are planning facts, not the primary
local intersection. The events-derived manifest is authoritative.

If exactly 720 eligible concepts cannot be formed, no role count is silently
changed. The episode stays blocked until a pre-outcome scientist amendment or
is abandoned as infeasible.

### Step 2: assign concepts globally

One deterministic, versioned metadata-only procedure assigns:

- 480 concepts to `development`;
- 120 concepts to `support_calibration`; and
- 120 concepts to `sealed_audit`.

Every exemplar, presentation, participant row, beta stage, and derived value
for a concept receives the same role. Concepts are the leakage boundary.

The split balances externally defined THINGS/THINGSplus taxonomy, five- versus
six-exemplar status, session/acquisition order, and repetition lag within
frozen tolerances. Candidate features and neural values cannot influence
balance or tie-breaking.

The taxonomy is a versioned input, not an informal label source. Before role
assignment, freeze its release, exact file and hash, concept-ID crosswalk,
unmapped and multilabel handling, group-merging rule, and minimum cell size.
The same frozen mapping must be used for both split balance and any terminal-
driving semantic stratum. The source annotation TSVs remain restricted mixed
files because they also contain behavioral outcomes; only the builder's
materialized allowlisted projection is outcome-blind.

Development concepts are divided into six fixed 80-concept outer folds.
They also receive 24 fixed 20-concept uncertainty blocks, independent of the
outer-fold labels. Audit concepts receive twelve fixed 10-concept influence
and uncertainty blocks.

### Step 3: create permission-separated handoffs

| Handoff | Contents | Who may read it |
| --- | --- | --- |
| Metadata-only | allowlisted structural IDs, roles, repetitions, sessions, folds, blocks, and stage/voxel mappings; no behavioral outcomes | builder, controller, evaluator |
| Calibration raw | role-filtered B/C/D repetitions for support/calibration concepts | trusted builder only |
| Development | development B/C/D values plus frozen calibration-derived reliability, support, and stage-specific ceiling artifacts; no raw calibration rows | adaptive search worker |
| Sealed audit | role-filtered B/C/D values for audit concepts only | trusted evaluator after lock |

Handoffs are ordinary materialized files, not symlinks back into the mixed raw
source. Each has a content manifest, immutable hash, owner/group, permission
report, and no-unexpected-link check.

## Response and repeat contract

Within participant and beta stage, retained trialwise betas for a unique image
are averaged with equal repetition weight. Each resulting image has equal
weight in fitting and scoring.

Calibration concepts alone define:

- repeat reliability;
- reliability-selected voxel supports; and
- stage-specific B/C/D noise ceilings.

The trusted builder performs those fixed operations. The adaptive search
worker receives their frozen outputs, not raw calibration responses.

Development and audit images never contribute to support selection or ceiling
estimation. First-presentation and repeat-index endpoints are prespecified
nonterminal diagnostics and never trigger refitting on audit data.

## ROI and geometry contract

The acquired `aparcaseg` volumes are candidate anatomical sources, not yet a
frozen visual-ROI definition. Before neural-outcome scoring:

1. freeze the parcel-to-visual-ROI crosswalk;
2. resample labels to each GLMsingle T1w mask with nearest-neighbor
   interpolation;
3. record source and target affine, shape, orientation, and checksums;
4. intersect with voxels finite across B/C/D before outcome scoring; and
5. predeclare empty/small-ROI refusal thresholds.

Participant-specific retinotopy or functional localizers may be
three-participant sensitivities. They cannot substitute a reliability-derived
fourth-person ROI into the primary four-participant rule.

## Controlled models and image pixels

The model registry requires six to eight exact checkpoints, three controlled
promotion pairs, and a trained-versus-random falsifier. Each checkpoint needs
training lineage and an explicit THINGS/target-image exposure label.

Model features require the exact stimulus pixels. The pinned CNeuroMod
`images_fmri.zip` and the full original THINGS archive are now
acquisition-verified under recorded terms, but neither is an episode handoff.
A central-directory audit found that all 8,640 CNeuroMod entries map by name to
the full archive but differ in both uncompressed size and CRC32, so the exact
CNeuroMod archive cannot be reconstructed by selecting and renaming full-THINGS
entries. The trusted builder must therefore extract the pinned CNeuroMod source
under the accepted terms, hash every image, align it to the event image IDs,
and materialize only the role-safe feature artifacts needed by the episode.
File names or concept labels are not substitutes for pixels.

## EP18 exposure boundary

EP17 and the planned THINGS-EEG episode share the THINGS stimulus ecosystem.
They are not independent simply because one measures fMRI and one EEG.

Before any EP17 audit neural outcome is opened, either:

1. reserve disjoint concept roles for the two episodes; or
2. freeze EP18's complete stimulus roles, feature controls, and decision rule,
   then label later evidence as correlated/design-exposed.

Downloading and hashing EP17 source files does not open a neural outcome and
does not, by itself, consume the EP18 design seal.

## Audit firewall

CNeuroMod is public, so the seal is prospective campaign governance, not a
claim that the world has never inspected the data. A valid audit requires:

1. a signed human/agent/cache/publication exposure ledger;
2. one primary relation and at most two disclosed secondary checks;
3. a lock binding models, features, final development-only voxel
   coefficients, contracts, ROIs, margins, concept blocks, code, environment,
   and evaluator;
4. no search-worker access to audit neural values or partial metrics;
5. direct application of the frozen predictor, with no audit refit;
6. one evaluator opening across all four participants; and
7. no post-open subset rescue, margin change, or rerun.

## Requirements before neural-outcome access

Candidate scoring and audit access remain closed until:

- controlled extraction of the acquired pinned stimulus archive and exact
  per-image hash/event-ID alignment are complete;
- the versioned THINGS/THINGSplus taxonomy, hashes, concept crosswalk,
  label-handling rules, group merges, and minimum cell size are frozen;
- the local events-derived common image/concept universe is frozen;
- the 480/120/120 roles, six folds, 24 development uncertainty blocks, and
  twelve audit blocks pass balance and completeness checks;
- role-filtered, non-symlinked handoffs and permissions are verified;
- B/C/D trial axes, response units, geometry, and voxel indices align;
- B/C/D stage-specific ceiling feasibility passes;
- the anatomical ROI crosswalk and spatial quota supports are frozen;
- the exact checkpoint/exposure manifest exists;
- the EP17/EP18 exposure decision is signed; and
- the trusted evaluator passes an outcome-free synthetic dry run.

A qualification failure before neural outcome access produces no scientific
terminal. Narrow, valid audit evidence that contradicts
development may close as `closed_heldout_concept_nonreplication`; wide audit
bounds close as `closed_audit_underidentified`. Integrity failures are
technical failures, not null results.
