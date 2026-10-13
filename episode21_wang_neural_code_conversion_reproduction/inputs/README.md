# EP21 inputs

The approved native mask interface `inputs/laion_independent_visual_masks/`
now points to `outputs/steward_provisioning/independent_visual_masks/`.
It contains five read-only native unions of the fixed 29 independent-localizer
mask identities, C-order full-grid voxel index arrays, and the actual component/
grid/QC report. Scientist amendment:
`outputs/protocol_mask_amendment_2026-09-30.yaml`. This is adapted, incomplete
visual support; no source whole-VC equivalence is claimed. Five reference beta headers are consistent at float32 header precision; every
future exported session still needs this binding. Beta/preprocessing eligibility
remains unresolved. The interface
contains mask values only, no neural responses or model outcomes.

EP21 has a provisioned regular-only raw-stimulus interface as of 2026-09-30:
`inputs/laion_regular_feature_handoff/` points to the owner-only read-only
`outputs/steward_provisioning/regular_stimuli/` export. It contains 24,681
regular images, aligned metadata and an export manifest; all 371 OOD images
are excluded. It contains no neural values or precomputed VGG19/AlexNet
features. The neural scoring interface remains absent and closed.

The two original redacted structural/public-source inventory interfaces remain
read-only historical records. Real provider role/split/header metadata is now
in `outputs/steward_provisioning/source_structural_manifest.json`, explicitly
consumed by the existing ASTRA conformance analysis. The paths below are
stable interfaces for the Wang et al. neural-code-conversion reproduction, not
evidence that a payload exists or that access and execution are authorized.
Once provisioned, every object under `inputs/` is read-only.

The scientist supplied provider confirmation of Terms v2026-05-12 acceptance
and Low Risk, then authorized a separate stimulus-only export with explicit
1CPU/4GiB/30-minute/5GB-durable/10GB-temporary limits. Slurm job 46135794
published the handoff above. Records:
`outputs/steward_provisioning_authorization_2026-09-30.md` and
`outputs/steward_provisioning/publication_check.json`.
The independent-mask amendment and its real materialization are complete.
Beta/preprocessing provenance, native beta-grid binding, neural provisioning,
exact runtime/assets and the real protocol lock remain unresolved. Model
fitting/scoring is closed.

## Stable interfaces

| ASTRA input ID | Expected path | Permitted content |
| --- | --- | --- |
| `conversion_structural_manifest` | `inputs/conversion_structural_manifest.json` | Redacted dataset, participant, site, trial/image-role, split, overlap, shape, dtype, mask, missingness, and exposure metadata; no neural values or outcome summaries |
| `upstream_pipeline_asset_inventory` | `inputs/upstream_pipeline_asset_inventory.json` | Source-code revision and VGG19, AlexNet, decoder, converter, generator, and DISTS asset identities or verified absence |
| `laion_regular_feature_handoff` | `inputs/laion_regular_feature_handoff/` | Authorized non-neural regular-image inputs or image-aligned VGG19/AlexNet features and their manifests; no neural values and no OOD payload |
| `laion_regular_scoring_handoff` | `inputs/laion_regular_scoring_handoff/` | Post-lock LAION-fMRI regular-only neural data, IDs, native-space visual masks, and allowed preprocessing metadata |
| `external_site_feature_handoff` | `inputs/external_site_feature_handoff/` | Authorized non-neural NSD/THINGS stimulus inputs or VGG19/AlexNet features and manifests |
| `external_site_scoring_handoff` | `inputs/external_site_scoring_handoff/` | Post-lock NSD/THINGS neural data, IDs, native-space masks, and allowed preprocessing metadata |

The external-site interfaces are provisioning-contingent. Their absence makes
the affected inter-site branch unavailable but does not block the 20-directed-
pair LAION branch.

There is deliberately no OOD or artificial-image input interface.

## Structural manifest requirements

`conversion_structural_manifest.json` must allow outcome-blind verification of:

- release/package, license, data-use, and Sherlock risk status;
- dataset, site, participant, session, run, trial, repetition, and image IDs;
- LAION subject-unique, shared `tau` train, shared `tau` test, and OOD roles;
- exact pairwise overlap counts between every source and target training set;
- exact disjointness of `tau` train and `tau` test and grouping of all image
  repetitions;
- planned and eligible directed-pair tables: 20 within LAION and, if fully
  qualified, 70 bidirectional LAION↔NSD/THINGS pairs;
- the identity and role disposition of the 240 regular LAION images documented
  as NSD-origin before any LAION↔NSD no-shared claim;
- beta representation, array shapes, dtypes, finite/NaN masks without values,
  missingness, native-space ROI identities, and coverage; and
- prior-exposure labels plus confirmation that OOD payloads and live EP04
  outputs are physically absent from every EP21 handoff.

It may contain counts and Boolean overlap results. It may not contain neural
values, response summaries, decoded features, reconstructions, identification
scores, model comparisons, or candidate rankings.

## Asset inventory requirements

`upstream_pipeline_asset_inventory.json` records identities, compatibility, and
availability without model-brain outcomes. At minimum it covers:

- the authenticated revision of
  `KamitaniLab/InterSiteNeuralCodeConversion`, fixed to tag `V1.0.0` at commit
  `edbff02edd6a08d3c6c829af45a957fd14bdc460`, its paper-linked Zenodo archive
  and checksum, and any required configuration;
- the precise VGG19 and AlexNet implementations, checkpoints, layer maps, and
  image preprocessing;
- target ridge-decoder and 500-voxel-selection implementations;
- nonlinear content-loss and linear brain-loss converter implementations;
- reconstruction optimization code, deep-generator prior, and DISTS model;
  and
- software/runtime compatibility needed to reproduce the released pipeline.

Verified absence is a valid inventory result. Missing required assets freeze
only the affected branch as `not_evaluable`; they do not authorize a substitute
after neural outcomes are visible.

## LAION handoff roles

The LAION feature and scoring handoffs must jointly support three strictly
separated roles for each directed `source -> target` pair:

1. target decoder fitting on the target participant's 4,712 subject-unique
   regular images;
2. no-shared content-loss converter fitting on the source participant's
   different 4,712 subject-unique regular images; and
3. brain-loss fitting on authenticated shared regular `tau` train images and
   evaluation only on disjoint shared regular `tau` test images.

The structural manifest must prove zero image-ID overlap between roles 1 and 2
and between all fitting roles and `tau` test. Neural values cannot be mounted
to resolve or revise these roles.

`laion_regular_feature_handoff` may include authorized raw regular stimuli only
when the applicable data-use and risk decisions allow feature extraction. If
precomputed features are supplied instead, their image IDs, valid masks,
dtypes, shapes, checkpoints, layers, preprocessing, and normalization must be
complete. Neither form may contain a raw or derived OOD image.

`laion_regular_scoring_handoff` is inaccessible until a valid protocol lock
exists and empirical execution is separately authorized. It must be
regular-only and role-filterable without traversing a mixed OOD source.

## External-site handoff roles

The NSD/THINGS interfaces must preserve each provider's authenticated native
training and held-out test roles. For each inter-site direction:

- train the target decoder only on the target participant's native training
  images;
- train the content-loss converter only on the source participant's native
  training images, after verifying zero exact overlap with the target training
  universe; and
- evaluate on the source dataset's declared held-out test role under the
  locked source-pipeline convention.

No shared-image brain-loss comparator may be synthesized for inter-site pairs.
NSD↔THINGS pairs are outside the current adapted design. Each dataset and
direction must be independently disableable if its input is absent or invalid.

## Stage-specific visibility

| Stage | Inputs that may be visible | Inputs that remain unavailable |
| --- | --- | --- |
| Source qualification | Structural manifest and upstream asset inventory | All neural values, feature/scoring handoffs, reconstructions, scores, and OOD payloads |
| Pipeline conformance and protocol locking | Contracts, qualification report, asset metadata, synthetic/permuted fixtures | Empirical neural values and neural-derived outcomes |
| Feature preparation | Applicable non-neural feature handoff after lock | Neural scoring handoffs unless separately needed by a later stage; all OOD payloads |
| Decoder/converter fitting and evaluation | Only the applicable post-lock role-filtered regular scoring handoff and frozen feature manifests | Other episode outputs, live EP04 state, legacy Scratch, and all OOD payloads |

The lock resolver may not receive neural values, predictions, reconstructions,
identification scores, or comparison summaries. A protocol lock is necessary
but does not itself grant empirical execution authority.

## OOD and cross-episode prohibition

The proposal's artificial-image analogue uses LAION-fMRI abstract-shape or
illusion stimuli that fall within the 371-image OOD pool reserved for EP04.
Under the current contract, `inputs/` must not contain or link to:

- OOD raw images or pixel-derived VGG19/AlexNet features;
- OOD neural responses or response-derived statistics;
- OOD predictions, reconstructions, scores, nulls, or summaries;
- live or mutable EP04 outputs, rankings, runtime state, or candidate records;
  or
- legacy EP04 Scratch content.

Published category-level descriptions may be cited in documentation only. An
executable artificial branch would require an explicit cross-episode decision,
a newly declared role-isolated interface, and a regenerated protocol lock. No
such interface or authorization currently exists.

## Provisioning record

Before adding any empirical payload, record in the episode log:

- the interface ID and read-only path;
- release and provider identity;
- license, data-use, and Sherlock risk decision;
- datasets, participants, sites, stimulus roles, and permitted analyses;
- exact regular/OOD filtering and the exposure classification;
- whether the handoff is structural, non-neural, or neural; and
- verified absence or presence of every required upstream asset.

Generated code, fitted models, features created by EP21, reconstructions,
predictions, caches, reports, and execution logs belong under `outputs/` or the
episode-specific Scratch path, never under `inputs/`.


Provider beta eligibility remains unverified after retrieving frozen processing
JSON metadata. Assessment: `outputs/steward_provisioning/provider_beta_eligibility_assessment_2026-09-30.json`.
The settings/reference-pool provenance does not establish tau-test/OOD
exclusions. No neural handoff or neural execution is opened by this metadata.


## Scientist-approved fixed provider measurement — 2026-09-30

The scientist approved standard released GLMsingle v1.2 final TYPED `betasmd`
regular-trial rows as fixed measurement inputs for this adapted retrospective
replication. Record: `outputs/protocol_provider_beta_amendment_2026-09-30.yaml`.
Their upstream HRF/noise-PC/ridge/autoscaling/internal-normalization/percent-
signal-change role exclusions remain unverified. Possible inherited tau-test/
OOD influence is accepted and disclosed; no custom replacement release or raw
BOLD reprocessing is required for this measurement definition. This explicitly
supersedes earlier blanket upstream-eligibility gates for these fixed regular
measurements only. All direct OOD payload/statistic/outcome access stays closed.

EP21 downstream normalization, voxel ranking, model fitting, CV, tuning and
stopping remain training-only. No-shared-stimulus claims refer to decoder and
converter training image IDs, not the whole upstream estimation pipeline.
Method contrasts do not establish cancellation of upstream dependence. Results
are conditional on fixed provider measurements, regular-image support and the
five participants; no clean-measurement, fresh-confirmation, OOD or population
claim follows. Neural export resources, role-isolated missingness metadata,
exact runtime/assets, required remaining conformance, a valid protocol lock
and separate fitting/scoring authority are still required.


## Regular-neural provisioning completion — 2026-10-01

The single authorized steward job46172551 completed successfully in55:02. All5participants and150native-session bindings are provisioned in `outputs/steward_provisioning/regular_neural_sealed/`; the readable structural completion marker is `outputs/steward_provisioning/regular_neural_structural_report.json`. It records151,861preserved regular trial repetitions,0nonfinite elements and7,550,850,918B durable payload. No OOD arrays were formed/retained/summarized. The accepted inherited preprocessing limitation remains unverified, not clean. This provisioned payload stays sealed: no scoring handoff, empirical fitting/scoring or lock is authorized. Required public runtime conformance remains pending the separately requested source-only allocation. Existing question, budget, stopping criteria, masks/role definitions and passing checks remain unchanged.


Public-runtime allocation update2026-10-01: source-only limits are approved and retained in `outputs/public_runtime_conformance_authorization_2026-09-30.json`. Execution is not launched: the prepared short bundle is held under Sherlock AGENTS rule10 useful-work aggregation. See `outputs/pipeline_conformance/public_runtime_scheduling_disposition_2026-10-01.md`. No scientific contract, lock, result or empirical authority changed.


Harness correction2026-10-02: required public conformance no longer implies a standalone short job. It is a bounded prelude within a separately authorized substantive EP21 allocation, with unchanged scientific lock/data gates and no new compute authority. See `outputs/pipeline_conformance/harness_repair_2026-10-02.md`.19focused regressions pass; actual public model/fixture execution remains pending. Existing source-only approval and completed data/QC are retained.
