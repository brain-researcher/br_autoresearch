# EP04 adaptive data contract

## Status and known source

This contract does not assert that absent data have been acquired. The intended
source is the LAION-fMRI release. Before scoring, the episode must identify a
read-only release and its scientific joins rather than read mutable run outputs.

Reported source-release structure:

- five participants and 30 main `task-images` sessions per participant;
- 25,052 distinct images;
- per participant, 4,712 subject-unique regular images plus 1,492 images shared
  by all five participants;
- shared images comprise 1,121 regular and 371 OOD images;
- shipped GLMsingle TYPED single-trial betas, beta-to-trial tables, ROI/noise
  ceiling files, human captions, object segmentations for shared images,
  pretrained embeddings, metadata, and predefined splits; and
- subject-unique images span reported LAION-natural, THINGS, and THINGSplus
  sources, while the shared pool also contains NSD-origin images.

Verify counts, identifiers, access terms, missing trials, and exact usable rows
from the identified release. Planning counts are not observed data.

### Physical-location record

The canonical checkout does not contain these payloads and does not have a
repository-local `.steward_acquisition` directory. Resolve physical paths only
through the root
[`DATA_LOCATION_MANIFEST.json`](../DATA_LOCATION_MANIFEST.json):

- `private_steward_acquisition` identifies the canonical private steward root
  at
  `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition`;
  the acquisition tree was relocated there by same-filesystem rename on
  2026-09-24; and
- `scratch_ep04` identifies an existing legacy runtime copy under
  `$SCRATCH/autoresearch/episode04_shared_scene_geometry/`. It is a cleanup
  candidate after durable-destination verification, not a migration
  destination.

The steward source has been relocated, but no source has been provisioned into
this episode. These location records are inventory pointers, not role-filtered
handoffs, scientific qualification, or permission to inspect neural or
stimulus outcomes.

## Evidence exposure status

| Slice | Adaptive role | Exposure statement |
| --- | --- | --- |
| synthetic/permuted fixtures and structural metadata | implementation and QC | no target neural claim |
| 4,712 regular subject-unique images per participant | adaptive development and selection | development-exposed by design; never confirmation |
| 1,121 regular shared images | historical prior/reproduction only | neural outcomes opened in the earlier EP04 round; excluded from trial scoring and audit |
| 371 shared OOD images | conditional one-shot boundary audit | eligible only if an outcome-blind exposure review establishes that target responses and comparison summaries remained sealed |
| future compatible participants/dataset | alternative independent audit | absent; must be prospectively acquired, authenticated, and sealed |

The 240 NSD-origin shared images are not an independent image-source
replication of NSD. Regular shared exposure cannot be repaired by choosing new
metrics or withholding some participants after the fact.

## Required frozen development inputs

Before neural scoring, the episode must record:

1. the LAION-fMRI provider release/package identifier, accessible location,
   license, and stimulus DUA disposition;
2. beta-to-trial-to-image identity for every included presentation, including
   participant, session, repetition, missingness, and exclusions;
3. one canonical GLMsingle representation and its space, finite/NaN, session
   standardization, and brain-coverage validation;
4. ROI masks and noise-ceiling files with a frozen retinotopic-EVC versus
   LAION-sector crosswalk;
5. human-caption rows with source/count/rank, object metadata, model embedding
   `image_ids`, `valid` masks, dtype, normalization, and released
   model/checkpoint identifiers;
6. a frozen subject-unique fold assignment that keeps all repetitions and
   near-neighbour clusters together and balances source strata; and
7. a concise exposure record for subject-unique, regular-shared, and OOD
   responses and comparison results.

The readiness record must include a subject-unique image-by-feature coverage
matrix for human captions, object/category inventory, every retained frozen
visual embedding, and low-level covariates. Current release notes explicitly
report object segmentations for shared images; they do not prove object
coverage for the 4,712 subject-unique images per participant. The central
object comparator must not enter neural scoring until this matrix passes with
the frozen common-eligibility rule. A new detector requires stimulus DUA
approval, a model/version frozen before extraction, and complete coverage
before any neural outcome is accessed.

Primary search should use released captions, embeddings, betas, ROIs, and
metadata. Raw packed images are not required. If a proposed operator requires
raw images or a new detector, stop until the separate DUA and model/version are
approved and freeze a revised policy before outcome access.

## Development and selection firewall

- Search only on subject-unique response outcomes with image-disjoint,
  repetition-grouped folds within each participant.
- Fit scaling, PCA/PLS, whitening, reliability rules, readouts, fusion weights,
  and hyperparameters using training images only.
- Use identical image folds and eligible voxels for candidate/comparator pairs.
- Do not use regular-shared or OOD neural results for early stopping, rank,
  representation, ROI, caption construction, exclusions, or thresholds.
- Keep regular-shared historical results in a separately labeled prior capsule;
  the adaptive evaluator must not import it.

## Audit firewall

The one-shot audit source is chosen before search:

1. **Conditional OOD route:** use the 371 shared OOD images only after an
   outcome-blind exposure review establishes that the relevant responses and
   summaries were never opened. Keep response bytes outside the search
   worker's permissions; reveal them only to the locked evaluator.
2. **Independent-data route:** if any OOD exposure is found, acquire and seal a
   compatible new participant/dataset. Do not quietly fall back to regular
   shared images.

The audit custodian verifies the write-once configuration lock and then permits
one complete execution. Partial category scores stay hidden until the run is
complete. An exact software retry is allowed only if no score was released and
the locked pipeline and settings are unchanged. After reveal, no candidate,
ROI, category, threshold, or aggregation may change.

## Missing assets and blockers

- No immutable adaptive LAION-fMRI input pack is yet bound here.
- A complete exposure audit is required to establish whether all 371 OOD
  responses remain sealed; this contract does not assume that conclusion.
- The exact subject-unique eligible images, folds, missing trials, ROI coverage,
  released model versions, and legal-use clearance must be established.
- Subject-unique caption/object/visual/low-level feature coverage, especially
  the object-only comparator, has not been verified and blocks neural search.
- The packed stimulus-image DUA is unresolved for any raw-image-dependent
  extension; such extensions remain excluded by default.
- No new independent participant or compatible dataset has been acquired.
- The scoring code has not yet been shown to implement the frozen search
  policy; this blocks candidate scoring and audit opening, not explicit task
  startup.

## Storage boundary

Inputs are read-only and large source material remains outside Git. Durable
source/fold records, code, trial and exposure records, the configuration lock,
reports, and compact predictions belong in the episode workspace. New
transient beta matrices and search caches
belong under
`$SCRATCH/br_autoresearch/episode04_shared_scene_geometry/`. The old path
represented by `scratch_ep04` remains legacy runtime state and must not be used
as the durable source or planned destination. Historical EP04 outputs are
immutable prior evidence, not a writable cache or audit source.
