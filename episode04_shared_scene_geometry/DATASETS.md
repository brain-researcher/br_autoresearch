# EP04 fixed-panel VLM–brain alignment data contract

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

| Slice | Current role | Exposure statement |
| --- | --- | --- |
| synthetic/permuted fixtures and structural metadata | implementation and QC | no target neural claim |
| 4,712 regular subject-unique images per participant | fixed-panel development and selection | selection-exposed by design; never confirmation |
| 1,121 regular shared images | excluded opened prior; no reproduction analysis | neural outcomes opened in the earlier EP04 round; excluded from panel scoring and audit |
| 371 shared OOD images | conditional one-shot boundary audit | eligible only if an outcome-blind exposure review establishes that target responses and comparison summaries remained sealed |
| future compatible participants/dataset | alternative independent audit | absent; must be prospectively acquired, authenticated, and sealed |

The current VLM–brain alignment analysis makes no replication claim from the
regular-shared or NSD-origin images. Regular shared exposure cannot be repaired
by choosing new metrics or withholding some participants after the fact.

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
5. exactly one release-supplied final pooled image embedding for each of
   OpenCLIP, PEcore, SigLIP2, and DINOv2, plus the required low-level control;
   their `image_ids`, `valid` masks, dtype, dimensionality, normalization, and
   released model/checkpoint identifiers; optional caption and object features
   are recorded separately if qualified;
6. a frozen subject-unique fold assignment that keeps all repetitions and
   near-neighbour clusters together and balances source strata; and
7. a concise exposure record for subject-unique, regular-shared, and OOD
   responses and comparison results.

Before scoring, verify subject-unique image coverage for every retained VLM
image embedding and for the DINOv2 and low-level controls. These mandatory
features define the common eligible-image set for the primary analysis.
Caption and object/category coverage is checked separately and can restrict
only its labeled secondary analysis. Current release notes explicitly report
object segmentations for shared images; they do not prove object coverage for
the 4,712 subject-unique images per participant. Missing object coverage must
therefore be reported but does not block the primary VLM-versus-DINOv2 test. A
new detector requires stimulus DUA approval, a model/version frozen before
extraction, and complete coverage before neural scoring.

For each frozen model/checkpoint, record the native embedding dimensionality
and the fixed L2-normalization plus cosine-distance model-RDM definition. No
layer, rank, distance, or transformed geometry can enter the primary candidate
set. The primary neural RDM is correlation distance over
repetition-averaged, session-standardized beta patterns in each frozen ROI.
Cosine neural distance is mandatory sensitivity; crossnobis is optional only
if valid independent repeat partitions can be constructed. The development
association is image-disjoint Spearman RSA, Fisher-z aggregated across folds,
equally across the three high-level sectors, and then equally across the five
participants. These fold scores select the fixed-panel winner and are not final
confirmation.

Model-RDM edges are not independent samples: split images before constructing
RDMs, place both endpoints of every scored edge in the same evaluation fold,
and resample whole images or frozen near-neighbour clusters when quantifying
stimulus uncertainty. Because subject-unique development image pools differ by
participant, development heterogeneity mixes participant and stimulus-sample
variation; the shared OOD images are the fixed-stimulus comparison.

The primary repeat rule requires at least two valid presentations per image
and averages every valid presentation after frozen session standardization.
Before scoring, structural trial metadata determine the modal planned repeat
count; the mandatory complete-repeat sensitivity retains only images reaching
that count. No repeat or image may be dropped after its RSA contribution is
known.

Primary RSA should use released VLM image embeddings, control features, betas,
ROIs, and metadata. Raw packed images and reproduction of the earlier analysis
are not required. If a proposed extension requires raw images or a new
detector, stop until the separate DUA and model/version are approved and freeze
a revised policy before neural scoring.

## Development and selection firewall

- Compare the fixed panel only on subject-unique response outcomes with
  image-disjoint, repetition-grouped folds within each participant.
- Apply only the fixed L2 normalization to the native embeddings in the
  primary analysis; it has no neural fit. Fit reliability and nuisance rules
  on training images only.
- Restrict PCA, whitening, covariance matching, and other geometry transforms
  to labeled sensitivity analyses. They cannot select or rescue the native-RSA
  candidate.
- Restrict PLS, learned fusion, and predictive readout fitting to the labeled
  secondary encoding diagnostic, using training images only; it cannot select
  or rescue the primary RSA candidate.
- Use identical image folds and eligible voxels for candidate/comparator pairs.
- Do not use regular-shared or OOD neural results for model selection, ROI,
  caption construction, exclusions, or thresholds.
- Do not import the regular-shared responses or their earlier comparison
  results into the current evaluator; no reproduction branch is planned.

## Audit firewall

The one-shot audit source is chosen before fixed-panel scoring:

1. **Conditional OOD route:** use the 371 shared OOD images only after an
   outcome-blind exposure review establishes that the relevant responses and
   summaries were never opened. Keep response bytes outside the comparison
   worker's permissions; reveal them only to the locked evaluator.
2. **Independent-data route:** if any OOD exposure is found, acquire and seal a
   compatible new participant/dataset. Do not quietly fall back to regular
   shared images.

The audit custodian verifies the write-once configuration lock and then permits
one complete execution. Partial category scores stay hidden until the run is
complete. An exact software retry is allowed only if no score was released and
the locked pipeline and settings are unchanged. After reveal, no candidate,
ROI, category, threshold, or aggregation may change.

The locked OOD estimator uses native VLM and DINOv2 cosine RDMs and
correlation-distance neural RDMs. It computes Spearman RSA separately from the
within-category edges of every prespecified OOD category, applies Fisher-z,
then weights categories and the three high-level sectors equally to one effect
per participant. The all-pair RDM is secondary. Audit success requires the
participant-balanced VLM-minus-DINOv2 gain to reach 0.02 Fisher-z units, at
least four of five participant effects to be positive, all participant/sector/
category leave-one-out aggregates to remain positive, and positive
unsubtracted VLM-to-brain RSA to exceed the 95th percentile of 10,000
within-category whole-image label permutations shared across participants and
ROIs.

Before any OOD neural response is opened, verify complete common coverage for
the locked VLM, DINOv2, required low-level geometry, prespecified category
labels, and valid image IDs across all 371 images. The OOD decision additionally
requires the participant-balanced VLM-minus-DINOv2 partial-RSA gain and every
leave-one-participant-out gain to remain positive after within-category control
for the low-level RDM. Missing pre-open coverage makes the audit technically
unevaluable.

## Missing assets and blockers

- No immutable fixed-panel LAION-fMRI input pack is yet bound here.
- A complete exposure audit is required to establish whether all 371 OOD
  responses remain sealed; this contract does not assume that conclusion.
- The exact subject-unique eligible images, folds, missing trials, ROI coverage,
  released model versions, and legal-use clearance must be established.
- Subject-unique VLM, DINOv2, and low-level feature coverage has not been
  verified and blocks primary neural scoring. Caption/object coverage remains
  unresolved but blocks only the corresponding secondary anchor.
- The packed stimulus-image DUA is unresolved for any raw-image-dependent
  extension; such extensions remain excluded by default.
- No new independent participant or compatible dataset has been acquired.
- The scoring code has not yet been shown to implement the frozen comparison
  policy; this blocks candidate scoring and audit opening, not explicit task
  startup.

## Storage boundary

Inputs are read-only and large source material remains outside Git. Durable
source/fold records, code, model-evaluation and exposure records, the
configuration lock, reports, and compact predictions belong in the episode
workspace. New transient beta matrices and evaluation caches belong under
`$SCRATCH/br_autoresearch/episode04_shared_scene_geometry/`. The old path
represented by `scratch_ep04` remains legacy runtime state and must not be used
as the durable source or planned destination. Historical EP04 outputs are
immutable prior evidence, not a writable cache or audit source.
