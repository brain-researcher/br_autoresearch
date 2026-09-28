# Dataset contract — Episode 21

EP21 uses the existing LAION-fMRI release. It creates no new data asset and
grants no new access by itself. The study requires regular natural images,
GLMsingle single-trial betas, beta-to-trial metadata, ROI and noise-ceiling
files, human captions, object metadata, released model features, and official
split definitions.

## Data source

| Item | Current value |
| --- | --- |
| Dataset | LAION-fMRI |
| Source study | Doerig et al. (2025), 10.1038/s42256-025-01072-0 |
| Participants | Five |
| Primary fitting pool | 4,712 regular subject-unique images per participant |
| Primary evaluation pool | 1,121 regular images shared by all five participants |
| Prohibited pool | 371 shared out-of-distribution (OOD) images and all derived payloads |
| Episode-local payload | None provisioned under `inputs/` |
| Release identity and row counts | Reported for planning; not yet verified for EP21 |
| Data-use agreement and risk status | Unresolved; must be qualified before any payload access |

The reported release structure, still to be verified before scoring, is:

- five participants with 30 main `task-images` sessions each;
- 25,052 distinct images;
- 4,712 regular subject-unique images per participant;
- 1,492 images shared by all five participants, comprising 1,121 regular and
  371 OOD images;
- three human captions per subject-unique image and five per regular shared
  image;
- official `tau` and `cluster_k5` split metadata; and
- GLMsingle betas, trial tables, ROI/noise-ceiling files, captions, object
  segmentations for at least the shared set, and released image embeddings.

These values are planning statements, not observed row counts. Before anyone
materializes or inspects human neural data, the release must be confirmed as
deidentified Low or Moderate Risk data permitted on Sherlock, and its
data-use agreement (DUA) must permit the intended use. High Risk data are out
of scope.

## What has already been seen

The neural outcomes for the 1,121 regular shared images were previously opened
in the EP04 lineage. They are retrospective reproduction evidence, not fresh
or independent confirmation. EP04 and EP21 share evidence and must be treated
as correlated analyses.

The 240 NSD-origin images in the shared regular pool also prevent a clean claim
of fully independent image-source replication. EP21 keeps the full 1,121-image
regular pool primary and reports the prespecified 240-image exclusion only as
a sensitivity analysis.

No empirical payload is provisioned under this episode's `inputs/` directory.
The 371-image OOD payload remains unavailable to and prohibited for EP21.

## Where the data live

The root `DATA_LOCATION_MANIFEST.json` inventories two durable upstream assets
under the private steward root:

| Logical asset | Inventory role |
| --- | --- |
| `asset_ep04_laion_mixed_role` | Mixed-role LAION-fMRI source |
| `asset_ep04_laion_raw_stimuli_dua` | Restricted raw-stimulus asset |

These logical IDs remain stable even though EP21 shares the underlying bytes.
Their presence is inventory only: it does not provide an EP21 role-filtered
handoff, prove source integrity or scientific eligibility, resolve the image
DUA, or authorize protected outcomes. EP21 must never use the legacy
`scratch_ep04` tree.

Three read-only, role-filtered EP21 views are required:

1. a redacted structural manifest for qualification and protocol locking;
2. a caption/stimulus/feature handoff without neural values; and
3. a regular-only neural scoring handoff that scorers may consume only after
   the protocol lock exists.

All three views must exclude the 371 OOD raw images, OOD-derived features,
neural responses, predictions, scores, and candidate-comparison summaries. The
structural and feature handoffs contain no neural values or derived neural
scores.

## Who may see what

| Stage or slice | Permitted use | Still unavailable |
| --- | --- | --- |
| Structural metadata, identifiers, captions/features, ROI headers, and synthetic/permuted fixtures | Establish joins, qualification, and implementation conformance | Neural values and neural-derived scores |
| 4,712 regular subject-unique images per participant | Fit the prespecified encoding and decoding maps | Treating training performance as scientific evidence |
| 1,121 regular images shared by all five participants | Fixed retrospective transfer evaluation after protocol lock | Model or analysis selection outside the narrow prespecified robustness exception |
| Official `tau` and `cluster_k5` partitions in the regular pool | Fixed secondary robustness analyses | Changing the core fit, primary verdict, global method, or another fold |
| 371 shared OOD raw images, derived features, neural responses, predictions, scores, and summaries | No EP21 use; published category-level documentation may be cited | The complete payload, which is conditionally reserved for EP04 |
| Future compatible participants or dataset | Possible future confirmation | Absent and not part of EP21 |

## What must be verified before scoring

Before any neural score, record only the source facts needed by the fixed
program:

1. provider release/package identifier, accessible read-only location,
   license, DUA, and Sherlock data-risk disposition;
2. participant, session, trial, repetition, and image IDs, including missing
   presentations and all exclusions;
3. the chosen GLMsingle beta representation, native space, finite/NaN status,
   brain coverage, and regular-only within-session standardization behavior;
4. retinotopic early visual cortex (EVC) and LAION ventral, lateral, and dorsal
   masks, category-ROI masks, noise ceilings, reliability maps, their exact
   stimulus universes, and the prespecified NSD-to-LAION interpretation
   crosswalk;
5. human-caption rows and caption counts; object/segmentation coverage; every
   retained feature's `image_ids`, valid mask, dtype, normalization, model,
   checkpoint, layer, and preprocessing;
6. exact membership and overlap of the subject-unique, regular-shared, OOD,
   `tau`, `cluster_k5`, and 240 NSD-origin subsets;
7. an exposure statement for each neural slice and confirmation that the
   EP21 worker cannot see the 371 OOD outcomes; and
8. the authenticated upstream paper code commit, RCNN weight identities,
   ten-instance set, final-layer/final-time-step definition, and original
   comparator availability.

The qualification report must distinguish verified present, verified absent,
and unknown. Its lock-facing form is restricted to identifiers, roles, shapes,
dtypes, finite/NaN masks, coverage, missingness, and provenance; it must not
contain neural values, neural summaries, predictions, or model--brain scores.
A model-asset inventory covering code, checkpoints, tokenizers, parsers, and
weights is a required qualification record. It may certify absence and lock
Finding 3 as `not_evaluable` without blocking Findings 1 or 2. Any
provider-derived noise ceiling, reliability map, or
normalization statistic whose construction included OOD trials is prohibited;
it must be recomputed from regular rows or the affected endpoint is
`not_evaluable`. A missing core asset yields `not_evaluable` or stops the
affected module; it does not authorize a substitute discovered after neural
scoring.

Exact image-ID overlap between subject-unique fitting and shared evaluation is
a qualification failure. Before scoring, fix either a versioned
near-duplicate detector with threshold and action or an explicit no-screen
policy with the resulting limitation; neural outcomes cannot inform it.

## Required fields and features

The fixed common image universe for each comparison requires complete,
image-ID-aligned coverage of every feature in that comparison. At minimum:

- separately embedded and image-averaged `all-mpnet-base-v2` human captions;
- separately embedded/averaged words;
- nouns and verbs produced by one pinned parser and concatenation rule;
- one prespecified object-inventory construction from authorized metadata;
- caption-image shuffled features generated from recorded seeds;
- recurrent convolutional neural network (RCNN) activations extracted with the
  released LLM-trained and category-trained weights at the exact source-specified
  layer, time step, and ten instances;
- authenticated original comparator features needed for Finding 3; and
- DINOv2, OpenCLIP, PEcore, and SigLIP2 embeddings for the secondary
  cross-modal extension, each bound to an immutable model/checkpoint revision,
  layer, pooling, input preprocessing, and output normalization.

The shared-caption count sensitivity uses all ten fixed combinations of three
of the five provider-ordered captions, reports every combination and their
unweighted mean, and never selects a subset from neural results.

LAION segmentation nouns are a proxy for the original COCO category labels
and must be labeled an adapted comparison. Object metadata reported for shared
images does not establish subject-unique coverage. If Finding 1 fitting or a
control requires subject-unique object features, complete coverage and the
construction rule must be fixed before scoring.

Decoder dictionaries are not valid merely because original code can load
them. Qualification separately records the ID, hash, entry universe, and
preprocessing of the original qualitative lookup dictionary and the adapted
LAION target-containing rank dictionary. For the latter it also fixes the
eligible correct entry or entries for every evaluation image, absent-target
action, similarity metric, rank direction, tie rule, and participant summary.
The combined-sector decoder input must likewise name and deduplicate its
native-space voxel union. If these cannot be established without neural
results, only the affected retrieval subclaim is `not_evaluable`.

Nearest-caption lookup in the original source dictionary is a qualitative
source-style reconstruction. A correct-caption rank on LAION is an adapted
endpoint and requires a prespecified candidate set that actually contains the
target captions; it must not be reported as an original-paper rank.
For Finding 2, the participant/ROI RDM noise-ceiling estimator must be
authenticated separately from any provider voxelwise noise-ceiling map.

Regular raw-stimulus access is allowed only if the DUA and risk review permit
it and only for the prespecified RCNN/image-feature extraction required by the
contract. The 371 OOD raw images remain forbidden regardless of that review.
The regular raw-image asset may not be opened merely to explore alternative
features.

## Training and evaluation boundary

- Group all repetitions of one image on the same side of every split.
- Filter out OOD rows before session response normalization, noise-ceiling or
  reliability estimation, repetition averaging, and every other
  response-derived statistic.
- The prespecified within-session z-score uses all and only regular trials in
  that participant/session/voxel and is an explicit transductive response-only
  exception. Fit feature scaling, PCA, regularization, voxel filters,
  nuisance models, and variance-partitioning components on prespecified training
  rows only.
- Use identical eligible images, voxels, folds, and transforms for paired
  model comparisons.
- Except for nested estimator tuning confined to the prespecified `tau` or
  `cluster_k5` training rows, do not use the 1,121 outcomes to choose features,
  ROIs, hyperparameters, parsers, comparators, exclusions, thresholds, or
  stopping. That narrow robustness exception cannot change the global method,
  core fit, primary verdict, or another fold. Prior campaign exposure is
  acknowledged; the EP21 procedure is still fixed prospectively relative to
  new EP21 scoring.
- Do not inspect or derive any 371-OOD neural result. Directory traversal,
  broad searches, mixed-role mounts, cached predictions, and terminal output
  all count as access.
- EP21 artifacts must not be imported into EP04's adaptive scorer or audit.

## Fixed transfer and robustness splits

The primary predictive transfer fits each participant's 4,712 regular
subject-unique images and evaluates the resulting procedure unchanged on the
1,121 regular shared images. Primary ROI RSA uses the regular shared pool under
the same fixed image and eligibility rules.

The provider's exact split manifest is authoritative after qualification.
Planning counts for `tau` are 897 shared regular training images and 224 shared
regular test images. Encoding and decoding fit the train portion and evaluate
unchanged on the test portion; RSA is reported separately for both. In
`cluster_k5`, predictive models and every learned transform are fit on four
clusters and evaluated on the fifth, while parameter-free RSA is computed only
within each held-out cluster. The procedure rotates through all five folds,
reports every fold, and averages correlations on the Fisher-z scale (with an
inverse transform for display). Its inference is descriptive and cannot alter
a core verdict.

Any mismatch between these planning counts and authenticated release metadata
must be resolved outcome-blind. Do not repair a mismatch by dropping rows
after viewing neural results.

## Initialization status — 2026-09-28

- No read-only EP21 role-filtered LAION-fMRI handoff exists under `inputs/`.
- Release identity, exact counts, joins, masks, features, and splits have not
  been qualified for EP21.
- The paper code commit, RCNN weights, comparator panel, and decoder dictionary
  are not yet pinned in this episode.
- Regular raw-image DUA permission for RCNN and image-feature extraction is
  unresolved.
- The noun/verb parser, object-inventory proxy, fractional-ridge solver,
  ROI-RDM ceiling correction, shuffle construction, multiplicity table,
  interval implementation, and expected-direction/exception tables are not
  yet fixed.
- `PROTOCOL_TABLE.yaml` is intentionally unlocked; it must resolve every
  shared field and every field for each runnable module, or freeze an
  unavailable module as `not_evaluable`, before producing
  `outputs/protocol_lock.yaml` and permitting neural scoring. In particular,
  paired-comparison inference and the decoder dictionary/rank convention still
  require outcome-independent validation.
- The complete 371-image OOD payload is intentionally unavailable to EP21.

These facts block neural execution, not creation of the episode contract.

## Storage

Inputs are read-only after provisioning. Durable code, manifests, compact
features, predictions, null summaries, reports, and execution records belong
under this episode's `outputs/`. Large transient beta matrices, RDMs,
activations, permutation shards, and caches belong under
`$SCRATCH/br_autoresearch/episode21_doerig_laion_fmri_reproduction/` and are
reconstructible. Scratch is temporary and must never hold the only copy of a
required artifact.
