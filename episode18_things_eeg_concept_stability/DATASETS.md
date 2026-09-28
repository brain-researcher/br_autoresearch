# EP18 data: THINGS-EEG1, images, and participant roles

EP18 uses THINGS-EEG1 to ask whether a true-concept basis predicts EEG for new
exemplars better than a capacity-matched label basis built on the same image
model. The scientific comparisons are defined in [GOAL.md](GOAL.md); this
file explains which data are available, how they must be divided, and what
still has to be built before EEG-driven model comparison.

## What is available

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

Audit participants use the same six-fold participant-refit procedure: ten
blocks fit the model and two blocks are scored. This is replication in new
participants, not transfer of a development participant's fitted EEG
coefficients.

## What must exist before model comparison

The next setup stage must produce:

- a validated event table and exact image join;
- the fixed 30/16 participant assignment and role-filtered views;
- the common 62-channel order and 61-dimensional transform;
- the image and label feature records;
- the frozen visual-only distance panel and distant-event ablation rule;
- eight acceptable EEG-blind false partitions;
- outcome-blind simulations of overlap, leakage, and recovery; and
- one EP17/EP18 exposure record covering image, concept, and feature reuse.

If the event structure, sample separation, design rank, false-group matching,
reliability, or projected 16-participant precision is inadequate, the study is
underidentified. It must not substitute folds or trials for participants or
relax the comparison after seeing the effect.
