# EP18 data: THINGS-EEG1, images, and participant roles

**October 13 standing-authority continuation.** The unchanged core image/task
calibration resumes on the already released 30-person development inputs.
Its complete-cohort G/T estimates precede concept/false-group comparison and
do not require new data access. The existing audit and all qualification
requirements for candidate-discriminating EEG scores remain closed/in force.
Historical calendar and per-job limits below are superseded by the repository's
standing compute authority, while all prior consumption remains recorded.

**October 13 search-only continuation amendment.** The separately adopted
[authority](outputs/config/bounded_generator_continuation_20261013_authority.json)
permits one continuation of the existing sub-07 image/context search from its
saved checkpoint, until at most October 14, 21:43:13 UTC. It reuses the same
released inputs and complete cache; it opens no new dataset, EEG fitting,
participant expansion or audit data. Original source/configuration/cache and
checkpoint identities are preserved. The original study deadline remains in
the historical record; only this bounded control search has a new window.

EP18 uses THINGS-EEG1 to ask whether a true-concept basis predicts EEG for new
exemplars better than a capacity-matched label basis built on the same image
model. The scientific comparisons are defined in [GOAL.md](GOAL.md); this
file explains which data are available, how they must be divided, and what
still has to be built before EEG-driven model comparison.

## Public online resources — checked October 6, with sequence-trigger update October 12, 2026

| Public source | Available information and EP18 use | Remaining limit |
| --- | --- | --- |
| [Original data paper, author-hosted copy](https://tijl.github.io/tijl-grootswagers-pdf/Grootswagers_et_al_-_2022_-_Human_EEG_recordings_for_1,854_concepts_presented_.pdf) | Page3 identifies E3 as sequence start, E1 as image onset and E2 as offset; it documents1000Hz acquisition and0.0488281microvolt resolution. Page2 describes72 main sequences of309 images. Checked October12 for the continuous-reader implementation. | Exact E3/event alignment, segment-confined sample spans and padding still come from each authorized record. The provider's epoch-based preprocessing is context, not an adopted EP18 continuous-data recipe. |
| [OpenNeuro ds003825 v1.2.0 README](https://raw.githubusercontent.com/OpenNeuroDatasets/ds003825/1.2.0/README) | The fixed-release public README describes 50 recorded participants, 1,854 concepts, 22,248 main images and 10-Hz presentation with an orthogonal fixation task; it links the authors' OSF `hd6zk` information/example-code archive. Use for original experimental provenance and finding the provider methods. | These design totals do not replace the fixed local eligibility/event checks or 30/16 roles. No public EEG payload or example outcome was opened. |
| [Official PyTorch 2.4.1 distribution recipes](https://pytorch.org/get-started/previous-versions/#v241) | Exact `torch==2.4.1` / `torchvision==0.19.1` installation recipes exist for CPU and CUDA 11.8/12.1/12.4, with wheel and Conda routes. The required release pair is publicly available rather than unspecified. | Public availability alone did not qualify Sherlock execution. The separate exact CPU-wheel runtime subsequently qualified in allocation 46842156; the alpha-module route remains stopped. See the preparation evidence below. |
| [Pinned CLIP ViT-B/32 configuration](https://huggingface.co/openai/clip-vit-base-patch32/raw/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/config.json) | The already-selected model revision exposes readable configuration: 224-pixel vision input, 32-pixel patches, vision hidden size 768 and projection dimension 512. Use to cross-check the prepared asset/feature interface. | Configuration retrieval alone did not establish checkpoint integrity or loading. Both subsequently qualified in allocation 46842156, and label-only encoding completed in 46843116; clean image-exposure claims remain a separate unresolved requirement, and isolated pixel/EEG readers were replaced by procedural separation on 2026-10-12. |

The online resources resolve original-study pointers, exact release
distribution and model-shape documentation. They do not require changing the
episode's versions, feature choices or closed audit, and no model weights,
images, EEG or secrets were downloaded in this check.

## What is available

**October 17 method cross-check.** The original paper's technical-validation
methods (page 4 of the [author-hosted paper](https://tijl.github.io/tijl-grootswagers-pdf/Grootswagers_et_al_-_2022_-_Human_EEG_recordings_for_1,854_concepts_presented_.pdf))
used 0.1-Hz high-pass and 100-Hz low-pass Hamming-windowed FIR filtering,
average referencing and 250-Hz sampling, followed by stimulus epochs from
−100 to 1,000 ms. EP18's frozen continuous recipe instead has no high-pass or
detrending and uses a causal 129-tap low-pass with an unchanged acquisition
clock. Consequently, the paper's decoding validation is not positive-control
validation of this same continuous-prediction pipeline. This source check
changes no preprocessing, data role, score or threshold; the complete-cohort
signal/error diagnostic is recorded in `outputs/experiment_log.md`.

| Resource | Current state | EP18 use |
| --- | --- | --- |
| THINGS-EEG1 `ds003825`, release `v1.2.0` | Acquired and preserved read-only | Raw continuous EEG, events, markers, and participant metadata |
| Original THINGS images | Acquired under their source terms; archive remains in steward-controlled storage | Exact images shown in the experiment and input to image-feature extraction |
| THINGS metadata | Acquired; consume the recorded source version through a read-only working copy | Concept names, image identities, and human feature tables |
| THINGSplus-CC0 archive | Available as a separate control resource | Cross-checks only; it cannot replace the exact event-image join |

No large EEG, image, or feature payload belongs in Git. The episode repository
contains the readable design and small records describing how those payloads
are used.

### Source references

| Item | Source |
| --- | --- |
| Dataset | THINGS-EEG1 |
| OpenNeuro accession | `ds003825` |
| Release | `v1.2.0` |
| Dataset DOI | <https://doi.org/10.18112/openneuro.ds003825.v1.2.0> |
| Data paper | <https://doi.org/10.1038/s41597-021-01102-7> |
| Official code archive | <https://doi.org/10.17605/OSF.IO/HD6ZK> |
| EEG license | CC0 |
| Preservation root | `/oak/stanford/groups/russpold/data/br_autoresearch_data/things_eeg1/openneuro-ds003825-v1.2.0` |
| BIDS payload | `/oak/stanford/groups/russpold/data/br_autoresearch_data/things_eeg1/openneuro-ds003825-v1.2.0/source` |

The THINGS images are not part of the OpenNeuro release and do not inherit its
CC0 license. Extraction, feature computation, and redistribution must follow
the image archive's recorded research terms. The conceptual figure for EP18
therefore uses synthetic illustrations, not THINGS source images.

## Experiment structure

The main experiment contains 22,248 images: 12 exemplars for each of 1,854
concepts.

| Level | Structure | Role in EP18 |
| --- | --- | --- |
| Main block | 12 blocks; every concept appears once per block | Defines train/test exemplars |
| Physical sequence | 6 per block; 309 images per sequence | Hard boundary for filtering and temporal modeling |
| Image event | One onset every 100 ms; image displayed for about 50 ms | One event in the continuous time-expanded design |

The six outer folds hold out block pairs `{0,6}`, `{1,7}`, `{2,8}`, `{3,9}`,
`{4,10}`, and `{5,11}`. A fold fits on the other ten blocks, so every concept
contributes ten training exemplars and two held-out exemplars. Every image is
scored in exactly one outer fold.

The remaining five complementary block pairs form the inner folds used to
choose preprocessing, feature rank, temporal basis, and regularization using
training data only.

### Repeated-image session

After the main session, participants saw 200 validation images 12 times each.
These data are reserved for signal-recovery and reliability checks. They do
not enter the primary concept score, feature selection, candidate comparison,
or false-group construction, and they cannot rescue a failed main result.

## Reconstructing events correctly

Several event fields have similar names:

| Field | Main-session values | Meaning |
| --- | --- | --- |
| `blocksequencenumber` | 0–11 | Complete 1,854-concept block |
| `sequencenumber` | 0–71 | Physical EEG sequence |
| `withinsequencenumber` | 0–5 | Physical-sequence position within a block |
| `presentationnumber` | 0–308 | Image position within a physical sequence |
| `stimnumber` | 0–11 | Exemplar number within the true concept |

In release `v1.2.0`, `onset` is in seconds and `sample` is the integer sample
at 1,000 Hz. The source `duration` value is `50`, meaning 50 ms even though a
generic BIDS reader may interpret it as seconds. Preserve the source field and
create an explicit `duration_seconds = 0.05` in the working event table.

Use `sample` as the raw-signal join and compare it with `onset`, BrainVision
markers, `time_stimon`, and recording bounds. For every eligible participant,
confirm:

- 12 main blocks with one event for each of 1,854 concepts;
- 12 distinct exemplars per concept across blocks;
- six physical sequences of 309 images per block;
- 200 unique repeated images shown 12 times each; and
- agreement among event order, sample timing, and raw markers.

These checks establish the observations that enter the analysis.

The completed outcome-blind source preparation retains all 1,133,808 events
for the 46 eligible participants. It established that source `sample` values
are **one-based**; the frozen working join retains originals alongside
normalized recording indices. The large sub-36/sub-38 clock discontinuities
were localized to recording-segment/physical-sequence boundaries without
dropping events. See the [frozen metadata and source-diagnosis records](outputs/README.md).

## Exact image-event join

Before feature extraction or EEG modeling, build one row for every main and
repeated image event containing:

- participant and event location;
- concept and exemplar identity;
- canonical image identity;
- the recorded metadata version; and
- the exact local image match.

Resolve mismatches from source metadata and experiment records, not from EEG
responses. Freeze the join before any image-feature job or EEG-driven model
comparison. A feature job that is declared image-only receives pixels and an
opaque image ID; it cannot read filenames, folders, concept names, or event
labels.

## Participant eligibility and split

The provider marks four of 50 participants for exclusion:

| Participant | Provider note |
| --- | --- |
| `sub-01` | Missing the final repeated-image session |
| `sub-06` | Did not complete the experiment |
| `sub-18` | Poor signal at documented electrodes |
| `sub-23` | Monitor problems during the experiment |

This leaves 46 potentially eligible participants. Before assignment,
eligibility may use provider notes and fixed checks of files, headers, events,
and markers. It may not use neural signal quality, reliability, model
predictions, or candidate effects.

Assign 30 whole participants to development and 16 whole participants to
audit using the fixed metadata-only rule in `SEARCH_POLICY.yaml`. Because
`sub-49` and `sub-50` used a different acquisition montage, place exactly one
in each role if both pass the fixed integrity checks. Freeze all IDs before
EEG-derived work. No participant may later be swapped, excluded, or replaced
because of signal quality or effect direction.

This assignment is complete. The frozen development/audit metadata views
contain 739,440 and 394,368 events respectively, with the exact 30/16 identities
and montage balance fixed by the reviewed metadata rule. These manifests are
not runtime access controls; the EEG audit seal remains false.

The participant is the inference and replication unit. Blocks, sequences,
images, electrodes, folds, and samples are repeated measurements within a
participant.

## Common sensor space

Most participants used a standard 63-channel EEG setup with Cz reference.
`sub-49` and `sub-50` used a larger setup with FCz reference. Exactly 62
recorded channel labels are shared across the two arrangements.

The primary analysis uses those 62 labels in a frozen order for every
participant, subtracts the across-channel mean, and represents the result in
one fixed 61-dimensional orthonormal basis. Participant- or fold-specific bad
channel removal and interpolation are not allowed in the primary analysis.
Extra channels from `sub-49` and `sub-50` are sensitivity analyses only.

Estimate whitening from the outer-training blocks in the 61-dimensional
space. Do not invert the singular 62-channel common-average covariance.

## Continuous-data boundaries

The primary analysis starts from raw continuous EEG. For every outer fold:

- physical sequences remain separate during filtering, detrending, temporal
  expansion, and trimming;
- learned preprocessing and covariance use only the ten training blocks;
- the held-out pair is transformed with training-derived quantities;
- a model-independent scoring mask is shared by every compared model; and
- no raw or transformed sample contributes to both fitting and scoring.

Missingness and sample retention cannot depend on concept labels, candidate
residuals, or observed effects. Ordinary prestimulus baseline correction is
not used in the primary RSVP analysis.

## Feature and false-group records

Before candidate-discriminating EEG scores are released, record and freeze:

1. the low-level image features and named vision or vision-language model
   families used in the core and expanded panels;
2. image preprocessing, selected layers, pooling, and output dimensions;
3. the concept-name and human feature tables, missing-value rules, scaling,
   and concept crosswalk;
4. the separate visual-only distance panel used for the explanatory analysis;
   it includes low-level, supervised-vision, and self-supervised-vision
   features and excludes captions, label features, and vision-language
   embeddings;
5. the training-only reduction and regularization choices permitted by the
   search; and
6. the eight accepted false partitions and their image-feature, temporal, and
   fitting-capacity diagnostics.

Exact software and model versions belong in this feature record, where they
are needed to reproduce the chosen analysis. They do not need to dominate the
scientific question or create a generic readiness gate.

## Development and audit access

| View | Contents | Permitted use |
| --- | --- | --- |
| Preservation source | All 50 released participants | Archive and outcome-blind setup only |
| Development view | 30 eligible participants | Model development, qualification, and finalist selection |
| Audit view | 16 eligible participants | One application of the frozen procedure by an evaluator |

The same user account can currently retrieve the public EEG for all
participants. The planned split is therefore only a procedural holdout until
a separate evaluator identity or separately controlled mount prevents the
development process from reading or re-downloading the audit EEG. ACLs under
the same account do not create that seal.

Outcome-blind setup may proceed while this is unresolved. Under the current
contract, EEG-driven candidate comparison is prohibited and the one-shot
audit cannot open until the separation is verified.

**2026-10-12 amendment:** the scientist adopted procedural same-account
separation with a disclosed weaker claim, replacing the verified-separation
requirement above and the externally controlled pixel-only worker.
Development reads only the 30 development participants (pixels only inside
the image-feature worker); audit EEG is not opened, listed, searched or
re-downloaded before the single recorded opening, and every audit result must
state that the separation was procedural. Terms are in
[GOAL.md](GOAL.md#data-readiness-and-access-boundary).

Audit participants use the same six-fold participant-refit procedure: ten
blocks fit the model and two blocks are scored. This is replication in new
participants, not transfer of a development participant's fitted EEG
coefficients.

## What must exist before model comparison

The event/identity join, fixed 30/16 roles, common 62-channel order, prospective
feature registry, private image routing and human-feature crosswalk are
complete. All 1,854 concepts join to 12 property/size dimensions and 53
broad-category indicators; 406 concepts with no membership in those categories
remain included. The private routing preserves 22,448 image identities and
1,854 name prompts without exposing names to a pixel worker. Feature adapters
are prepared, but no image-model inference or EEG comparison has run.

The [exact-runtime and model-asset qualification](outputs/runs/feature_preparation/exact_cpu_provision_46842156/model_runtime/model_runtime_qualification.json)
completed in allocation **46842156**: all five fixed public assets passed
source-integrity checks and all six selected routes loaded offline with the
exact stable CPU runtime, including Torch 2.4.1 and Torchvision 0.19.1. No
forward pass or experimental data access occurred in that qualification.
After the successful first-64-name pilot and explicit coordinator release,
[all concept-name features](outputs/runs/feature_preparation/all_name_text_46843116/label_feature_report.json)
completed in allocation **46843116**: **1,854 by 512 float32** CLIP vectors,
with zero token truncations and zero provider-name fallbacks. This label-only
preparation read no source pixels, private image routing, events or EEG. It
does not establish a complete `frozen_feature_record` or either access seal.

The exact EP17 qualified-universe comparison is also complete: all 720 EP17
concepts and 3,634 qualified image identities occur in EP18's main experiment.
Both frozen designs are retained as correlated, design-exposed studies. The
[exposure record](outputs/config/ep17_ep18_exposure_record.json) distinguishes
source-inventory overlap from that qualified-universe result and does not
claim identical pixels/features or complete historical exposure knowledge.

The remaining work includes:

- effective pixel-worker delivery and EEG reader separation; the namespace
  `ENOSPC` route remains stopped, with no weaker boundary substituted
  (**superseded 2026-10-12** by the adopted procedural separation; see
  [Development and audit access](#development-and-audit-access));
- the remaining image/caption feature extraction and complete continuous-EEG
  fitting/evaluation pipeline; the fixed 61-dimensional sensor transform and
  supplied-recipe fold composition are implemented and tested, while raw-data
  preprocessing, complete nested tuning/EDF integration and scoring remain;
- feature-derived visual-distance/ablation and false-group qualification under
  the fixed prospective definitions;
- eight acceptable EEG-blind false partitions;
- the required leakage/recovery and inferential qualification, without
  restarting an unrequested simulation expansion.

The historical DTN authentication failure and alpha-module mismatch remain
preserved in the [execution log](outputs/experiment_log.md). The later finite
public HTTPS transfer and fresh exact CPU-wheel environment resolved asset
delivery and runtime compatibility without credentials, a retry of either
stopped route, or a weaker image/EEG access boundary.

If the event structure, sample separation, design rank, false-group matching,
reliability, or projected 16-participant precision is inadequate, the study is
underidentified. It must not substitute folds or trials for participants or
relax the comparison after seeing the effect.
