# Dataset contract — Episode 21

EP21 is an adapted reproduction of Wang et al. (2025), [*Inter-individual and
inter-site neural code conversion without shared
stimuli*](https://doi.org/10.1038/s43588-025-00826-5). Its primary empirical
resource is LAION-fMRI. NSD and THINGS form a separate, provisioning-contingent
inter-site branch. Deeprecon is part of the source study's context but is not
an active EP21 dataset under the present contract.

No empirical payload is currently provisioned under `inputs/`. Counts below
are planning values from the proposal and existing LAION-fMRI documentation;
they must be authenticated from release manifests before execution.

## Source implementation identity

The fixed software reference is the official repository's `V1.0.0` tag at
commit `edbff02edd6a08d3c6c829af45a957fd14bdc460`, not its moving default
branch. The associated paper-linked archive is Zenodo record
[14910040](https://doi.org/10.5281/zenodo.14910040); qualification must verify
the downloaded archive inventory and expected MD5
`bb80dcc1a038737d6d8fc2c95b3e5586` before use.

The source paper and tagged code contain fidelity questions that must be
resolved prospectively. The manuscript describes ReLU in the converter while
the tag uses LeakyReLU; the exact stochastic convolutional feature-map sampling
and loss weights must be recovered from code; and the manuscript's eight-layer
reconstruction description must be reconciled with a tagged configuration
that appears to request all 19 VGG19 layers. Qualification must document the
paper-faithful and code-faithful options, select one with an outcome-blind
rationale before scoring, and mark affected findings `not_evaluable` if an
authentic choice cannot be made.

## Dataset roles

| Dataset | Planned participants | EP21 role | Current availability |
| --- | ---: | --- | --- |
| LAION-fMRI | 5 | Primary 20-directed-pair inter-individual reproduction | Not provisioned or qualified |
| NSD | 4 | External-site targets and sources paired bidirectionally with LAION-fMRI | Not provisioned or qualified |
| THINGS | 3 | External-site targets and sources paired bidirectionally with LAION-fMRI | Not provisioned or qualified |
| Deeprecon | 5 in the source study | Source-study context only | Not an EP21 input |

If all planned LAION, NSD, and THINGS participants qualify, the inter-site
branch contains 70 directed LAION↔external pairs:

```text
LAION -> NSD:     5 x 4 = 20
NSD -> LAION:     4 x 5 = 20
LAION -> THINGS:  5 x 3 = 15
THINGS -> LAION:  3 x 5 = 15
Total:                       70
```

NSD↔THINGS pairs are not part of this adapted branch. The 70-pair count is not
an entitlement to data or a claim that every planned participant will qualify.

## LAION-fMRI source

The planned LAION-fMRI material is:

- five participants;
- GLMsingle single-trial betas in each participant's native space;
- whole-visual-cortex masks, with V1, V2, V3, V4, and higher-visual-cortex
  reporting masks where valid crosswalks can be established;
- 4,712 regular subject-unique images per participant for native decoder or
  content-converter training, depending on that participant's role;
- 1,121 regular images shared by all five participants, with an authenticated
  `tau` train/test partition planned as 897/224 images; and
- 371 shared out-of-distribution images that are closed to EP21.

The exact release identifier, participant IDs, session structure, trial counts,
image IDs, repetitions, exclusions, beta representation, mask definitions, and
`tau` membership remain unverified for this episode.

## Non-overlapping training roles

The no-shared-stimulus claim depends on image-role separation, not merely on
using different neural rows. For each directed LAION pair `source -> target`,
the required roles are:

| Component | Neural input | Stimulus role | May contain paired source/target responses? |
| --- | --- | --- | --- |
| Target VGG19 decoder | Target participant | Target's 4,712 subject-unique regular images | No |
| Content-loss converter | Source participant | Source's different 4,712 subject-unique regular images | No |
| Brain-loss comparator | Source and target participants | Authenticated shared regular `tau` training images | Yes; this is the shared-stimulus baseline |
| Common method evaluation | Source and target participants as required by the locked endpoint | Disjoint shared regular `tau` test images | Evaluation only |

Before fitting, qualification must demonstrate:

1. zero exact image-ID overlap between the source participant's 4,712
   content-converter images and the target participant's 4,712 decoder images;
2. zero overlap of either subject-unique training set with the shared `tau`
   test set;
3. zero overlap between shared `tau` train and `tau` test image IDs;
4. every repetition of an image remains in one role and one split; and
5. no OOD image contributes to any training, evaluation, normalization,
   reliability, voxel-selection, or derived feature calculation.

An exact-ID failure stops the affected directed pair. Any near-duplicate policy
must be chosen from non-neural image information and locked before neural
outcomes are available; it may not be invented after an unfavorable score.

### Target-decoder data

The target participant's decoder is fitted on the target's 4,712
subject-unique regular images. It predicts the fixed VGG19 feature hierarchy
from native-space target activity. Voxel selection, ridge regularization,
feature centering/scaling, and any dimensionality reduction must be fitted on
these target training rows only.

The target decoder is frozen before content-converter fitting. Its training set
does not include a source participant's subject-unique image, a `tau` test
image, or any OOD image.

### Content-loss-converter data

The nonlinear content-loss converter is fitted on the source participant's
4,712 subject-unique regular images. For each source image it consumes source
activity and the true VGG19 representation of that image. Its loss is evaluated
after the converted activity passes through the already frozen target decoder.
No target neural response to the source image is used or required.

Consequently, the source converter-training and target decoder-training image
sets must be disjoint. This is the primary no-shared-stimulus condition.

### Brain-loss data

The linear brain-loss converter requires paired source and target activity and
is the only method allowed to use the shared `tau` training split. It may not be
fitted on either participant's subject-unique images, because those images are
not shared, and it may not see a `tau` test image. Evaluation is confined to the
disjoint `tau` test split.

The planning counts of 897 `tau` train and 224 `tau` test images are accepted
only after the provider manifest reproduces them exactly or an outcome-blind
scientific amendment resolves a discrepancy.

An optional, separately labeled sample-matched content sensitivity may use
only the source side of the 897 `tau` training images, never the target
responses. It does not replace the 4,712-image primary content model and cannot
support the no-shared-stimulus claim. If used, a fixed 897-image subset of the
source-unique training role should also be reported to separate sample-count
from shared-image effects.

## External-site data

The inter-site branch uses openly available NSD and THINGS participants only
after their exact releases, permissions, native preprocessing, ROI masks,
stimulus roles, and decoder inputs qualify. The source proposal reports the
following source-study scales:

| Dataset | Source-study training images | Source-study test images | Stimulus description |
| --- | ---: | ---: | --- |
| NSD | 9,000 | 100 | COCO natural images |
| THINGS | 8,640 | 100 | Object images |

These are contextual planning counts, not authenticated EP21 row counts.

For each inter-site direction, the target decoder is trained on the target
participant's native training images, and the content-loss converter is trained
on the source participant's native training images. The two training universes
must have zero exact image-ID overlap for the no-shared analysis and must not be
forced into an artificial paired set. Any detected overlap is removed by a
prespecified outcome-blind rule. Evaluation uses the source dataset's declared
held-out test role and the target's fixed decoder, following the authenticated
source-pipeline convention.

There is no brain-loss inter-site comparator unless genuinely shared source and
target stimuli are prospectively documented. EP21 will not create one by image
matching, semantic similarity, or reuse of another dataset's neural responses.

Because LAION-fMRI documentation indicates that 240 regular images may be
NSD-origin stimuli, any LAION↔NSD no-shared analysis additionally requires an
exact cross-dataset identity audit. Qualification must show that none of those
images bridges a target-decoder and source-converter training role, or apply a
prospectively locked exclusion and sensitivity rule. Until that audit passes,
the LAION↔NSD no-shared subclaim is `not_evaluable`.

The external branch is independent in execution from the 20-pair LAION branch.
Missing NSD or THINGS data make only the affected direction or dataset
`not_evaluable`.

## Required stimulus and model features

EP21 must reproduce the source pipeline's feature definitions rather than use
convenient modern substitutes.

### VGG19 content and decoder targets

The planned hierarchy is VGG19 `conv1_1` through `conv5_4` plus `FC6`, `FC7`,
and `FC8`. Qualification must pin the implementation, checkpoint, layer names,
input resize/crop, color order, pixel scaling, mean subtraction, tensor layout,
flattening, and feature normalization. It must also establish whether the
source code and released assets use a framework-specific convention that
cannot be reproduced by layer names alone.

Every feature artifact requires image IDs, valid masks, dtype, shape, layer,
preprocessing identity, and extraction status. VGG19 features for LAION-fMRI
are new metadata to be generated only from authorized regular images or an
authorized non-neural feature handoff.

### AlexNet identification features

Natural-image identification uses the source pipeline's five convolutional and
three fully connected AlexNet layers. Qualification must lock the checkpoint,
preprocessing, correlation definition, candidate universe, true candidate,
pair construction, averaging order, rank direction, and tie rule. AlexNet
features are scoring metadata, not converter-training targets.

### Reconstruction assets

The reconstruction branch requires the authenticated gradient-based feature
optimization code, multi-layer weights, deep-generator prior, and DISTS
structure/texture implementation and weights. Iterations, optimizer,
initialization, seeds, image range, stopping, and failure handling must be
fixed before any reconstruction is inspected. Missing authentic assets make
Findings B–E, as applicable, `not_evaluable`; they do not authorize an
after-the-fact replacement method.

## Neural preprocessing and ROIs

LAION-fMRI uses the declared GLMsingle single-trial beta representation in
native participant space. Qualification must record the beta variant, trial to
image join, session and run labels, missing presentations, finite/NaN masks,
repetition policy, and whole-visual-cortex mask identity. V1–V4 and HVC reports
require source-compatible masks or an explicitly adapted crosswalk.

Response normalization, reliability estimation, voxel selection, nuisance
regression, and all other response-derived transformations must use only the
training rows allowed for that fitted object. If a response-only session
normalization is proposed across train and test trials, it must be declared as
a transductive exception, demonstrated not to use OOD trials, and frozen in the
protocol lock.

NSD and THINGS remain in their authenticated native preprocessing and native
subject spaces. Differences in acquisition, preprocessing, voxel counts, and
mask definitions are part of the inter-site estimand and must be reported, not
silently harmonized after outcome inspection.

## Prior exposure and cross-episode boundary

Neural outcomes for LAION-fMRI regular shared images were previously opened in
the EP04 lineage. EP21's regular-image evaluation is therefore retrospective
and correlated with EP04. It is not an independent confirmation, even though
the Wang analysis and code are newly fixed for EP21.

The 371-image LAION-fMRI OOD pool is conditionally reserved for EP04. The
proposal's abstract-shape or illusion analogue cannot run under the current
contract. EP21 may cite published category-level documentation, but it may not
access:

- raw OOD images or pixel-derived features;
- OOD neural responses or response-derived normalization/reliability values;
- OOD predictions, reconstructions, scores, nulls, or summaries; or
- live EP04 outputs, candidate rankings, runtime state, or legacy Scratch.

No OOD handoff path is defined. An explicit cross-episode amendment, new
role-isolated input, and regenerated protocol lock would be required before an
artificial-image branch could exist. Until then it is
`not_evaluable_under_current_contract`.

## What must be qualified before any neural scoring

The structural qualification report must distinguish verified present,
verified absent, and unknown for:

1. provider release/package IDs, licenses, data-use terms, read-only locations,
   and Sherlock Low/Moderate Risk eligibility;
2. all participant, site, session, run, trial, repetition, and image IDs;
3. subject-unique, shared `tau` train, shared `tau` test, and OOD membership,
   including all exact overlaps and exclusions;
4. the 20 directed LAION pairs and, separately, every eligible member of the
   planned 70 directed inter-site pairs;
5. beta variants, response shapes, finite/NaN status, native-space masks,
   voxel coverage, and preprocessing;
6. VGG19 and AlexNet implementations, checkpoints, layers, preprocessing, and
   complete regular-image coverage;
7. the upstream
   `KamitaniLab/InterSiteNeuralCodeConversion` code revision, configuration,
   converter architecture, ridge decoder, voxel-selection, reconstruction,
   generator, and DISTS assets;
8. exact training, evaluation, baseline, bootstrap, permutation, and
   comparability-margin specifications; and
9. prior exposure for every neural slice plus physical exclusion of all OOD
   payloads and live EP04 outputs.

The lock-facing qualification record contains identifiers, roles, counts,
shapes, dtypes, missingness, coverage, masks without values, overlap results,
and provenance. It contains no neural values, decoded features,
reconstructions, model comparisons, or other outcome summaries.

## Stable handoff classes

EP21 requires separate read-only interfaces for:

1. a redacted structural manifest and upstream asset inventory for
   qualification;
2. authorized non-neural LAION regular-image/features metadata;
3. post-lock LAION regular neural scoring data;
4. authorized non-neural NSD/THINGS stimulus/features metadata; and
5. post-lock NSD/THINGS neural scoring data.

The external interfaces are optional for completing the within-LAION branch.
A mixed-role handoff or a handoff containing any OOD payload is invalid.
Details and stable names are in `inputs/README.md`.

## Current availability — 2026-09-28

- No EP21 empirical handoff has been provisioned.
- No release, participant list, row count, split, overlap, mask, code revision,
  checkpoint, or reconstruction asset has been qualified for execution.
- No protocol lock exists.
- No neural value, feature score, prediction, or reconstruction has been
  accessed or produced by EP21.
- No local process or Slurm job has run.
- Contract authoring does not authorize empirical access or execution.

## Storage

Provisioned inputs are read-only. Durable code, compact features, model
manifests, fitted-object manifests, predictions, reconstructions, summaries,
reports, and execution records belong under this episode's `outputs/` after
separate authorization. Large reconstructible matrices, model states, feature
caches, and reconstruction intermediates belong under:

```text
$SCRATCH/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/
```

Scratch is temporary and may not hold the only copy of a required artifact.
EP21 must never use an EP04 Scratch tree or another episode's outputs as a live
input.
