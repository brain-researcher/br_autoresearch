# EP04 fixed-panel VLM–brain alignment data contract

## Status and known source

The LAION-fMRI development source was qualified and the complete fixed panel
finished with `closed_no_candidate` at October 6, 01:21 UTC. All 23,560
development images have complete finite 3,150-coordinate low-level controls;
the 400 observed comparisons, required sensitivities, encoding diagnostics and
full null panel were admitted. See the [source handoff and completed evidence](outputs/source_handoff.md#completed-development-evidence--october-6-2026)
and [terminal decision](outputs/experiment_log.md#2026-10-06-0121-utc--candidate_decision--closed_no_candidate-reporting-complete).
No model met the registered selection criteria, so no configuration lock or
OOD audit follows. The source specifications below preserve the fixed data
roles and historical qualification requirements; they are not outstanding
requests to restart the completed development branch.

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

The completed source records distinguish reported whole-release counts from
qualified development support: 23,560 images, 94,216 presentations and 14,285
frozen grouping units. The metadata preserve 23,536 four-repeat and 24
three-repeat images separately from intended four-repeat coverage.

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

The location record alone does not authorize source access. Subsequent
episode-local selected-development handoffs and their qualification evidence
are linked in [source_handoff.md](outputs/source_handoff.md); the mixed sources,
regular-shared responses and OOD outcomes do not become evaluator inputs.

## Evidence exposure status

### Public online resources — checked October 6, 2026

- [Provider access documentation](https://laion-fmri.hebartlab.com/data_access.html): fMRI, derivatives, metadata, captions, embeddings and segmentations are public CC0 assets with anonymous access. The packed raw-image HDF5 is a separate DUA-controlled asset. EP04 already has its accepted DUA and selected development handoffs; this is no new acceptance or download request.
- [Provider stimulus derivatives](https://laion-fmri.hebartlab.com/stimulus_derivatives.html): documents four float16 embedding arrays with common `image_ids` and `valid` fields, 1,024 coordinates for CLIP/DINOv2/PEcore and 1,152 for SigLIP2. It reports three human captions per unique image, but object masks only for the 1,492 shared images. Caption availability is therefore distinct from absent unique-image segmentation coverage. The page also flags the September 28 transparency correction for 116 OOD images; existing local development arrays remain bound, and any later OOD use needs the correct version rather than an unrecorded refresh.
- [Provider GLMsingle documentation](https://laion-fmri.hebartlab.com/glmsingle_betas.html): public TYPED-only native-T1w beta, trial-mapping, sidecar and noise-ceiling specifications. Each trial TSV row aligns with one beta volume/column. Its current prose still describes `beta_index` as session-wide; the earlier local source diagnosis found run-local resets. Preserve original row position as the session beta-axis coordinate, as documented in the existing source handoff, rather than replacing that qualified join with web prose.

These resources supply usable documentation and public assets, but do not
establish local optional-caption coverage, OOD exposure/reader isolation or an
independent participant cohort. No image, beta, caption-content or embedding
payload was opened in this lookup. The completed `closed_no_candidate`
disposition and its data roles are unchanged.

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

On 2026-10-05 the scientist approved the exact
[RGB block-moment control](outputs/low_level_control_proposal.md), using the
already accepted September 22 LAION-fMRI raw-image acquisition. The authorized
reader joins frozen `20261003-metadata_folds04` identities to the source CSV
before reading only the selected HDF5 image elements: first the fixed 60-image
representative, then all 23,560 unique development images. The grids are
5/10/20, with fixed mean/population-SD scaling, 3,150 float64 coordinates and
Euclidean nuisance geometry. Shared and OOD images remain excluded. This
approval preceded extraction. Complete finite coverage for all 23,560 images
was subsequently established by job 46674207, as recorded in the
[October 5 completion](outputs/experiment_log.md#2026-10-05-2331-utc--full-low-level-coverage-complete-observed-panel-running),
and admitted by the final fixed-panel decision.

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

## Remaining scope after development closure

Development identities, folds, selected beta/ROI handoffs, all four native
embedding arrays, the approved low-level control and required computation
were sufficient for the admitted terminal decision. They are no longer
missing-development blockers. Subject-unique object masks are absent from
the release and the optional object anchor was not supplied; optional anchors
do not rescue or reopen the native-panel result.

OOD exposure/permission separation, its corrected embedding-version binding
and complete feature coverage were not established by development. They remain
conditional audit requirements, but no audit is activated after
`closed_no_candidate`. The accepted DUA and October 5 approval cover only the
specified unique-development RGB control; other raw-image extensions remain
outside that scope. No new independent participant or compatible dataset was
acquired. A negative candidate decision is not a missing-information request
or evidence that every possible VLM lacks an advantage.

## Storage boundary

Inputs are read-only and large source material remains outside Git. Durable
source/fold records, code, model-evaluation and exposure records, the
configuration lock, reports, and compact predictions belong in the episode
workspace. New transient beta matrices and evaluation caches belong under
`$SCRATCH/br_autoresearch/episode04_shared_scene_geometry/`. The old path
represented by `scratch_ep04` remains legacy runtime state and must not be used
as the durable source or planned destination. Historical EP04 outputs are
immutable prior evidence, not a writable cache or audit source.
