# EP18 — data and split instructions

EP18 uses THINGS-EEG1 to test whether, with the same image-derived predictors,
a categorical concept template predicts held-out-exemplar EEG better than a
capacity-matched basis built from continuous label and human features. The
scientific comparison and decision rule are defined in [GOAL.md](GOAL.md).

The complete EEG release, authorized original THINGS image archive, and
separate THINGSplus-CC0 control archive have been acquired, hash-verified, and
preserved read-only. The THINGS metadata snapshot is acquired and
hash-verified in steward quarantine, but its payload is currently
owner-writable; setup must verify it against the recorded checksums before use
and freeze the exact input hashes. EP18 is therefore source-ready for a future
outcome-blind setup phase, but no episode-managed run may launch until its
input/output guards and seven workspace projections exist. The exact
image-event join, participant roles, role-filtered views, feature records, and
alternative partitions are outputs to build and freeze during the episode,
not missing external datasets.

## Current status

| Component | Status |
| --- | --- |
| Complete EEG release | Ready and read-only |
| Original THINGS image archive and terms | Verified in steward quarantine; encrypted/unextracted; controlled extraction is an episode setup task |
| THINGS metadata | Acquisition-verified in steward quarantine; currently owner-writable, so verify recorded checksums at use |
| THINGSplus-CC0 control archive | Acquisition-verified and read-only; not a substitute for the exact event-image join |
| Exact image-event join | Build and validate during setup; freeze before feature extraction and model fitting |
| 30/16 participant roles | Assign outcome-blind during setup; freeze before EEG-derived work |
| Development and audit views | Provision after assignment; audit EEG remains evaluator-only |
| Feature and alternative-partition records | Generate under the rules below; freeze the finalist before audit |

## EEG source and image dependency

| Item | Source |
| --- | --- |
| Dataset | THINGS-EEG1 |
| OpenNeuro accession | `ds003825` |
| Release | `v1.2.0` |
| Dataset DOI | <https://doi.org/10.18112/openneuro.ds003825.v1.2.0> |
| Data paper | <https://doi.org/10.1038/s41597-021-01102-7> |
| Official code archive | <https://doi.org/10.17605/OSF.IO/HD6ZK> |
| License | CC0 |
| Preservation root | `/oak/stanford/groups/russpold/data/br_autoresearch_data/things_eeg1/openneuro-ds003825-v1.2.0` |
| BIDS payload | `/oak/stanford/groups/russpold/data/br_autoresearch_data/things_eeg1/openneuro-ds003825-v1.2.0/source` |
| Original THINGS archive | Steward quarantine; exact operational location is outside Git |
| THINGS metadata | Steward quarantine; exact operational location is outside Git |
| THINGSplus-CC0 control archive | Steward quarantine; exact operational location is outside Git |

The preserved release contains 412 files and 59,435,654,496 bytes
(55.35 GiB), including 41.40 GiB in the raw participant trees. Its `SOURCE.md`
contains the acquisition and verification record.

The release provides raw continuous BrainVision EEG, BIDS events, participant
metadata, and public derivatives for 50 participants. The
[EEG sidecar](https://github.com/OpenNeuroDatasets/ds003825/blob/1.2.0/task-rsvp_eeg.json)
reports 1,000 Hz sampling, 63 recorded EEG channels, Cz reference, and no
software filter. The primary analysis starts from the raw continuous EEG;
provider derivatives may be used only for reproduction or diagnostics.

The THINGS stimulus images are not part of the OpenNeuro payload. Their access
and redistribution terms do not follow from the EEG release's CC0 license. The
original archive was acquired separately under recorded research/noncommercial
terms and remains encrypted and unextracted in steward quarantine. During
EP18 setup, the authorized coding workflow may create a read-only working
extraction and join every event to an exact, hashed image using stimulus and
event metadata. This operation does not require audit EEG values. Extracted
images, archive credentials, and large derived features remain outside Git and
retain the source terms.

## Experiment structure

The main experiment contains 22,248 images: 12 exemplars for each of 1,854
concepts.

| Level | Structure | EP18 role |
| --- | --- | --- |
| Main block | 12 blocks; every concept appears once per block | outer fitting and scoring unit |
| Physical sequence | 6 per block; 309 images per sequence | continuous-signal boundary |
| Image event | one onset every 100 ms; image shown for about 50 ms | row in the time-expanded design |

Each participant therefore has 72 main physical sequences. The six outer
folds hold out block pairs `{0,6}`, `{1,7}`, `{2,8}`, `{3,9}`, `{4,10}`, and
`{5,11}`. A fold fits on the other ten blocks and assigns its named pair to
held-out scoring, so every image belongs to one held-out fold.

Each block contains one exemplar per concept, so every fold fits ten exemplars
and scores two for each concept. The physical sequence remains the
signal-processing unit: filtering, detrending, FIR construction, trimming, and
state resets may not cross its boundary.

### Repeated-image session

After the main session, participants saw 200 validation images 12 times each,
once in each of 12 additional physical sequences. These rows have
`sequencenumber` 72–83 and are separate from the 72 main sequences.

The repeated images are reserved for measurement reliability, reproducible
image-response checks, injection calibration, and provider-quality
reproduction. They do not enter the primary concept score, feature selection,
candidate comparison, or alternative-partition construction. Development
participants may be used to qualify those procedures. Audit-participant
repeated-image EEG remains evaluator-only until the procedure is fixed. Their
repetition, adaptation, and late-session position prevent treating them as a
primary noise ceiling, and they cannot rescue the main result.

## Event fields and timing

The [event sidecar](https://github.com/OpenNeuroDatasets/ds003825/blob/1.2.0/task-rsvp_events.json)
uses several names that are easy to confuse.

| Field | Main-session values | Meaning |
| --- | --- | --- |
| `blocksequencenumber` | 0–11 | complete 1,854-concept block |
| `sequencenumber` | 0–71 | physical EEG sequence |
| `withinsequencenumber` | 0–5 | physical-sequence position within the block |
| `presentationnumber` | 0–308 | image position within the physical sequence |
| `stimnumber` | 0–11 | exemplar within the true concept |

For the repeated-image session, `blocksequencenumber` and
`withinsequencenumber` are `-1`.

In `v1.2.0`, `onset` is in seconds and `sample` is the integer acquisition
sample at 1,000 Hz. The source `duration` value is `50`, meaning 50 ms even
though BIDS software normally expects seconds. Preserve that source value and
create an explicit `duration_seconds = 0.05`; never let a loader interpret it
as 50 seconds. Use `sample` as the raw-signal join and check it against
`onset`, the BrainVision markers, `time_stimon`, and recording bounds.

For every participant retained, confirm that:

- the main session has 12 blocks of 1,854 image events;
- every concept occurs once per block and uses 12 distinct exemplars overall;
- every block has six physical sequences of 309 images;
- the repeated session has 200 unique images with 12 presentations each; and
- event order and timing agree with the raw BrainVision markers.

Response time was updated only at screen refresh and is not highly precise.
Use response-locked nuisance timing only after validating its reconstruction;
otherwise use a coarser fixed response term and report the limitation.

## Participant eligibility and assignment

The provider marks four participants for exclusion:

| Participant | Provider note |
| --- | --- |
| `sub-01` | no data for the final repeated-image session |
| `sub-06` | did not complete the experiment |
| `sub-18` | poor signal at several documented electrodes |
| `sub-23` | monitor problems during the experiment |

These exclusions leave 46 potentially eligible participants. Before assigning
roles, eligibility may use provider flags, acquisition notes, and fixed checks
of files, headers, events, and markers. It may not use signal-derived quality,
reliability, model predictions, or candidate scores.

`sub-49` and `sub-50` used a 128-channel setup with only 64 channels connected.
If both pass the fixed checks, the assignment procedure must place one in
development and one in audit. `SEARCH_POLICY.yaml` defines one metadata-only
assignment algorithm, seed, balance objective, and tie rule; there is no
post-hoc choice between alternative splits. The algorithm must stop before
EEG access if the eligible count is not exactly 46.

### Montage harmonization

The standard recording sidecar declares 63 EEG channels with Cz reference.
For `sub-49` and `sub-50`, the sidecar declares a 128-channel setup with FCz
reference, while each BrainVision header exposes 127 recorded data channels.
Exactly 62 recorded channel labels are shared with the standard montage.

The primary analysis uses those 62 shared labels for every participant. Any
participant- or fold-specific bad-channel removal and interpolation is
prohibited. Subtract the across-channel mean and project into a fixed
61-dimensional orthonormal basis for that common-average subspace. Estimate
the fixed Ledoit-Wolf whitening covariance on training blocks in this
full-rank basis. Additional channels from
`sub-49` and `sub-50` are sensitivity-only and may not influence candidate
selection or rescue the primary result. Freeze the exact channel list, order,
the primary interpolation rule (`none`), contrast basis, and sensor-space loss
before any EEG-derived candidate comparison.

If another participant fails a fixed integrity rule, revisit the sample size
and split before model development; do not select a replacement after audit
outcomes are available. The participant is the inferential and replication
unit. Blocks, physical sequences, images, electrodes, folds, and EEG samples
are repeated measurements within a participant.

## Development and audit access

The planned split is 30 whole development participants and 16 whole audit
participants. After the fixed integrity checks, assign and record exact IDs
before examining any signal-derived measure or beginning model development.
Participant roles never change thereafter. Reliability, raw-signal quality,
model fit, and effect direction cannot be used to exclude, swap, or replace a
participant after assignment.

| Location or view | Contents | Permitted use |
| --- | --- | --- |
| Preservation source | all 50 released participants | archive only |
| Development view | 30 eligible participants | model development |
| Audit view | 16 eligible participants | evaluator only |

The same user account can currently reach all public EEG files, so the audit is
not yet technically sealed. A valid seal requires a separate evaluator
principal or separately controlled mount and must prevent the development
runtime from re-downloading audit EEG from the public provider. ACLs controlled
by the same `zijiao` account are insufficient. Until these controls are
verified, the state is `procedural_holdout_only`: setup may proceed, but
EEG-driven candidate comparison remains blocked. Proceeding with a merely
procedural holdout would require an explicit contract revision and could not
be described as a permission-blinded audit.

Stimulus order and task-event metadata may be used before the split to build
EEG-blind nuisance terms and alternative partitions. Neural values, neural QC,
predictions, and scores from audit participants may not inform development.

For each audit participant, the evaluator applies the fixed six-fold
procedure: fit ten blocks and score two. This is participant-refit replication,
not zero-shot transport of a development-participant EEG template. Audit
reliability and positive controls are computed only after lock. If they fail,
the cohort-level audit supports neither a positive conclusion nor a biological
null. Individual failures never create an analyzed subset, and no participant
is replaced.

All participants viewed the same main image collection. A scored exemplar is
held out from that participant's fold-specific fit, but is not globally unseen
by the development process.

## Setup records to build and freeze

These records are produced within EP18. The image/event join, participant
roles, development view, and primary channel transform must be frozen before
EEG-driven candidate comparison. The admissible upstream feature menu is fixed
without EP18 EEG; only declared downstream choices may use training-block
development EEG. Alternative partitions may evolve from stimuli and event
metadata, never EEG. Their finalist records must be frozen before audit
access.

1. **Image index and exact event join.** Include event path, concept ID,
   exemplar ID, canonical image identity, source-metadata hashes, and exact
   local image match for every main and repeated image.
2. **Feature record and matrices.** For every family, record the source or
   checkpoint, software version, weights hash, license, training-data lineage,
   image preprocessing, selected layers, pooling, and output dimension. Align
   rows to the exact image index. An image-derived job receives only pixels and
   an opaque image ID; paths, folders, filenames, concept fields, and prompts
   made from those fields are unavailable to it. Concept-name and human
   concept features are generated in a separate label-feature job. Freeze that
   job's missing-value indicators, column drops, 1,854-concept centering and
   scaling, equal-family weighting, column order, crosswalk, and matrix hash.
   Upstream encoders and raw feature choices cannot use EP18 EEG.
3. **Participant roles, channels, and folds.** Record the fixed 30/16 IDs,
   assignment rule, 62-channel list and order, the primary interpolation
   prohibition, fixed
   61-dimensional contrast basis, covariance regularization and sensor-space
   loss, six held-out block pairs, and physical-sequence boundaries.
4. **Alternative partitions.** Record the EEG-blind generator, features,
   fold-specific tolerances, seed and acceptance order, diversity rule, retry
   ceiling, eight accepted partitions, and matching diagnostics. Because block
   and sequence assignments vary across participants, accepted files may be
   participant-specific while the generator contract remains shared. Every
   participant-specific partition uses every main image exactly once, has
   1,854 groups of 12 images, places one image from each main block in a group,
   and never repeats a true concept within a group. Matching must pass in all
   six outer folds for held-out-to-training visual geometry, design spectrum,
   leverage, effective degrees of freedom, and temporal context over the full
   filter, FIR, target, and response horizon. Failure to build all eight within
   the frozen retry budget is underidentification, not permission to loosen a
   tolerance after EEG access.
5. **EP17/EP18 exposure ledger.** Record exact image and concept overlap,
   feature or checkpoint reuse, and first-access history.

Large EEG files, images, and feature matrices remain outside Git. The episode
directory contains the readable instructions and small records needed to
reproduce how they were used.

## Continuous-data invariants

- The primary analysis begins with raw continuous EEG.
- A physical sequence is always a signal boundary.
- Adaptive preprocessing and learned transformations use only the ten training
  blocks in an outer fold.
- After filtering, FIR construction, and edge trimming, no raw or transformed
  EEG sample may contribute to both fitting and scoring.
- One model-independent scoring mask is frozen per participant and outer fold;
  every retained continuous sample is scored exactly once.
- Every model and partition in a fold uses that same mask and the same
  training-derived whitening covariance.
- Missingness, artifact, and retention decisions cannot use a concept label,
  candidate residual, or observed effect.

The detailed model, matched controls, and temporal falsification tests belong
to [GOAL.md](GOAL.md).

## EP17 exposure boundary

EP17 and EP18 share the THINGS stimulus ecosystem. Maintain one ledger for
exact image and concept overlap, feature or checkpoint reuse, and access
history. Establish overlap from acquired files, not filenames or an assumed
upper bound. EP18 may provide new EEG evidence, but it is not an independent
stimulus-family confirmation of EP17.

## Readiness

The EEG and image archives have been acquired, hash-verified, and preserved
read-only. The metadata snapshot is acquired and hash-verified but must be
checked against its recorded hashes when consumed. These sources make
outcome-blind setup feasible, but EP18 remains an incomplete local draft and
no episode-managed run may launch until its input/output guards and seven
workspace projections exist. Once that scaffold is complete, freeze the exact
event join, participant roles, development view, primary channel transform,
and admissible upstream feature menu before EEG-driven candidate comparison,
and verify that the development runtime cannot access audit EEG. Before audit,
freeze the finalist feature and partition records, complete the shared
EP17/EP18 exposure ledger, and let only the evaluator open the audit view. The
qualification conditions in [GOAL.md](GOAL.md) must also pass.
