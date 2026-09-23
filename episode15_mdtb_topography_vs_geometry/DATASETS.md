# Dataset Contract — Episode 15

This episode uses the acquired Functional Fusion MDTB v1.0 derivative release.
It provisions nothing, authorizes no mixed-archive outcome access, and does
not turn a public release into a pristine external confirmation source.

## Fixed release

| Item | Fixed value |
| --- | --- |
| Dataset | Functional Fusion Multi-Domain Task Battery (MDTB) |
| Version | v1.0 |
| Zenodo record | `16788784` |
| DOI | `10.5281/zenodo.16788784` |
| Publication date | 2025-08-10 |
| Read-only shared root | `/oak/stanford/groups/russpold/data/br_autoresearch_data/functional_fusion_mdtb/zenodo-16788784-v1.0` |
| Acquired | 2026-08-21 |
| Primary paper | King et al., *Nature Neuroscience* (2019), `10.1038/s41593-019-0436-x` |
| Raw-data lineage declared by release | OpenNeuro `ds002105`, version `1.1.0` |
| Conservative use rule | Attribution plus noncommercial use pending license reconciliation |

Zenodo's record says CC BY 4.0, the bundled `README.md` and
`dataset_description.json` say CC0, and a provider page has reported CC BY-NC
3.0. Until provenance owners reconcile these statements, the most restrictive
observed terms control: retain attribution and do not use the data
commercially. No participant map or data derivative may be committed here.

## Identity pins

| Local record | SHA-256 |
| --- | --- |
| `SOURCE.md` | `b2fcfcdcdaa99ef4a213083656833277c6af57607da58162ea9c3a716f5c7f89` |
| `source/participants.tsv` | `43eb5c2c1704578ba5a458985eaf10c51e33f5a33769bbd8c5956cdb8f1bacde` |
| `source/README.md` | `c8464e2920ee0b4be4e2a23fe92237ee129ad4ba6a27c4b0c1d1c4234fd22368` |
| `source/dataset_description.json` | `86f0e10edd530063d24338bd1fe5e0e0c8a777fbdf424b359d19eedd3dffd2d5` |
| `source_metadata/record.json` | `8be08f7f62d68e00f336c3720fc6d350a484fefc5b9b267aac08281b6babb5d7` |
| `source_metadata/download_manifest.json` | `e952ac05636ab79efff8a80fe02295b3d1c0a1a1b90d70589ec2d1b997c95f90` |

Before extraction, an operator must validate the publisher MD5 values and
write a content-addressed local manifest with SHA-256 for all 24 participant
archives and every extracted file used by the episode.

## Observed structural inventory

The release contains 24 participant ZIP archives totaling 14,594,246,151
bytes, plus `README.md`, `participants.tsv`, and `dataset_description.json`
(14,594,249,364 bytes in `source/`). Central-directory inspection on
2026-09-20 found for every participant:

- one Task-A (`ses-s1`) and one Task-B (`ses-s2`) `reginfo.tsv`;
- 16 runs per task set;
- 736 Task-A beta NIfTI files (`46 regressors x 16 runs`);
- 784 Task-B beta NIfTI files (`49 regressors x 16 runs`);
- a cerebellar mask and nominal `space-SUIT_xfm.nii`; and
- anatomy, tissue maps, mean BOLD, MNI transforms, design matrices, residual
  variance maps, and cortical surfaces.

Regressor tables are byte-identical across participants. The released design
matrices have shapes `(9568, 752)` for Task A and `(9568, 800)` for Task B.

Seven archives (`sub-04`, `sub-09`, `sub-15`, `sub-17`, `sub-19`, `sub-21`,
and `sub-29`) contain 1,545 rather than 1,547 members. Each lacks both
`desc-overlap_mask.nii` and `space-SUIT_inv_xfm.nii`; these are exactly the
participants with `ses-rest=0`. All retain the nominal SUIT transform,
cerebellar mask, and Task-A/B masks and betas. This must not become a silent
seven-participant exclusion. A common route must use assets present for all
24, or reproducibly regenerate the missing products before roles are frozen.
Transform direction, reference grid, interpolation, software/template
versions, and whether an inverse is required remain unauthenticated launch
blockers.

The derivative contains beta estimates, not raw time series, events/confounds,
residual time series, or resting-state data. The task-design `rest` beta is not
resting-state fMRI. This episode cannot re-estimate the GLM, remove new motion
confounds, or use resting-state connectivity.

## Task structure and endpoint

After instruction regressors are excluded, each task set contains 17 task
families. Across A and B there are 26 distinct families and 47 distinct
conditions. Eight families and 14 conditions are shared.

| Role | Families | Conditions | Use |
| --- | ---: | ---: | --- |
| Task A only | 9 | 15 | Target calibration diversity and development diagnostics |
| Shared A/B, including rest | 8 | 14 | Task-A calibration; locked post-decision Task-B diagnostic only |
| Task B only | 9 source families | 18 | Primary prediction endpoint, reweighted as seven domains |

Primary eligibility is `instruction == 0 && common == 0` in Task B. The source
families and conditions are:

| Source family | Conditions | Primary domain |
| --- | --- | --- |
| `CPRO` | `CPRO` | `CPRO` |
| `prediction` | `Prediction`, `PredictViol`, `PredictScram` | `prediction` |
| `spatialMap` | `SpatialMapEasy`, `SpatialMapMed`, `SpatialMedDiff` | `spatialMap` |
| `natureMovie` | `NatureMovie` | `movie` |
| `romanceMovie` | `RomanceMovie` | `movie` |
| `landscapeMovie` | `LandscapeMovie` | `movie` |
| `mentalRotation` | `MentalRotEasy`, `MentalRotMed`, `MentalRotDiff` | `mentalRotation` |
| `emotionProcess` | `BodyMotionIntact`, `BodyMotionScram` | `emotionProcess` |
| `respAlt` | `RespAltEasy`, `RespAltMed`, `RespAltDiff` | `respAlt` |

The seven primary domains have equal weight; conditions have equal weight
within a domain. All 18 conditions remain separate for condition geometry,
with domain-balanced condition-pair weights. Provider spelling is preserved,
and any cleaned alias must retain a reversible mapping.

## Planned participant roles: 12 development / 12 audit

The earlier 16/8 hash split is withdrawn and must not be reused. The revised
estimands require a signal gate, equivalence bounds, adequacy, and model-class
discrimination; eight audit participants are not accepted by default.

No new role manifest has yet been instantiated. Before any candidate-
discriminating development Task-B score, an outcome custodian must:

1. verify that all 24 participants pass archive, anatomy, Task-A, transform,
   and expected-member checks without reading a Task-B numeric map or score;
2. compute one frozen Task-A reliability scalar per participant as the median
   centered spatial cross-half similarity across all non-instruction Task-A
   conditions, using runs `1--8` versus `9--16`, with no exclusions based on
   that scalar;
3. enumerate all 12/12 assignments and minimize, in order, the maximum and
   then the sum of absolute development-versus-audit count imbalances across
   the observed levels of sex, native-English status, and `ses-rest`; the
   absolute mean difference and then one-dimensional Wasserstein distance for
   age standardized over all 24 participants; and the same two quantities for
   the Task-A reliability scalar; and
4. break an exact tie with SHA-256 of the sorted audit IDs under the namespace
   `ep15-topography-vs-geometry-role-v1`.

Missing categorical values are treated as explicit levels; a missing age or
uncomputable Task-A reliability blocks role freeze rather than triggering
imputation or exclusion. The reliability code, environment, input hashes,
selected IDs, complement IDs, each vector component, and assignment hash must
be human-signed and committed to the lock ledger. The custodian returns only
the role manifest and balance report; participant-level audit Task-A
reliability values are not exposed to the search controller before panel lock.
After lock, the audit calibrator may read Task A only for the prespecified
target-personalization step.

Before this assignment, outcome-blind simulations must run the entire
selection and audit decision tree and show acceptable correct-class and
abstention rates for all decision edges at `n_dev=12`, `n_audit=12`. Scientific
margins may not be widened to make the split pass. If the simulation instead
supports 16/8, that is a contract amendment requiring human approval before
roles or development outcomes are exposed; it is not an automatic fallback.

All assigned participants remain in the accounting. After lock, the evaluator
applies the frozen finite-value, coverage, and geometry rules. An unscorable
audit participant yields `technical_failure`; there is no replacement or
reduced-N primary analysis.

## Physical firewall

The acquired ZIPs mix Task A and Task B and are not launch-safe. An operator
outside the candidate/controller process must create three immutable,
content-addressed views after role assignment:

1. development anatomy plus Task A and Task B for 12 participants;
2. audit anatomy plus Task A only for 12 participants; and
3. audit Task B for those 12 participants, mounted only to the trusted
   one-shot evaluator after the panel and predictions are locked.

The candidate workspace must not mount mixed audit ZIPs, derived atlas caches,
or any path from which audit Task B can be reconstructed. Audit Task A may
personalize only the already locked recipes after panel lock; it may not
update shared priors, choose a branch, tune a margin, or alter stopping.

## Frozen map construction

Before development Task-B outcomes guide any candidate, freeze and hash:

1. the authenticated SUIT route and one common cerebellar gray-matter support
   available for all 24 participants, fixed from anatomy and development data
   without audit-B intensities, audit-B header/coverage-driven selection, or
   audit-B QC adaptation;
2. affine/header checks, interpolation, resampling, missing-data behavior,
   voxel ordering, equal voxel weights, and spatial centering;
3. arithmetic condition-map aggregation from run-level betas with no
   outcome-adaptive weighting, whitening, smoothing, or participant-specific
   Task-B scaling, plus one common rule for an optional positive scalar fitted
   from target Task A and applied uniformly across all voxels and conditions;
4. map halves `runs 1--8` and `runs 9--16`;
5. geometry partitions `runs 1--4`, `5--8`, `9--12`, and `13--16`, plus a
   proof that every eligible condition is represented as expected;
6. the primary geometry pairing `(1,2)` versus `(3,4)` and any fixed
   non-selective alternative pairings;
7. the development-only M3 source geometry `K0`, its positive scale, and a
   source construction whose actual Gram geometry matches that definition;
8. the seven-domain table and condition/pair weights;
9. participant-held-out development rules that exclude the pseudo-target's
   Task B from source maps, priors, fitting, normalization, and thresholds;
   and
10. an append-only identity/QC ledger for every warning, coverage value,
    transform failure, exclusion, retry, and output hash.

The independence assumption for map halves and geometry partitions must be
examined using Task A and frozen synthetic/noise fixtures before Task-B model
comparison. A shared nuisance that makes the cross-products invalid blocks the
primary estimand; favorable Task-B results cannot waive the assumption.
Evaluator-side audit-B header and finite-value checks may declare a frozen
participant unscorable, but may never modify the common support or replace the
participant.

## Exposure, prior work, and novelty boundary

This is a targeted direct-overlap audit as of 2026-09-21, not a claim that the
literature has been exhausted. MDTB is repeatedly analysed, and all 24
participants are publication-exposed.

- King et al. (2019; `10.1038/s41593-019-0436-x`) derived group and
  participant-specific cerebellar parcellations. Its lower-bound analysis fit
  on one MDTB task set and evaluated boundaries on the other set's unique
  tasks, reversed the direction, and found individual parcellations superior
  to group parcellations. It also analysed crossvalidated condition
  distances. Cross-task prediction of individual boundaries is prior work.
- Nettekoven et al. (2024; `10.1038/s41467-024-52371-w`) fitted a hierarchical
  probabilistic atlas using seven task datasets including MDTB. Its precision-
  mapping analysis personalized an atlas with 1--16 first-set MDTB runs and
  evaluated it on the second set, including novel tasks. A hierarchical
  parcel model is therefore a mandatory incumbent, not a new endpoint.
- Nettekoven et al. (2026; `10.64898/2026.03.09.710558`), a bioRxiv preprint
  at the contract date, used Task A or rest in the 17 participants with rest
  to estimate spatial covariance, individual parcellations, and connectivity,
  then used the 18 unique B conditions for evaluation. Its primary covariance
  is brain-location by brain-location. Its connectivity prediction also uses
  Task-B cortical activity and is not anatomy-plus-A-only cerebellar map
  prediction.
- Arafat et al. (2026; eLife Reviewed Preprint
  `10.7554/eLife.111868.1`) used all 24 participants, selected short batteries
  from Task B, estimated individual parcellations, and predicted Task-A maps
  with leave-one-participant-out response profiles; supplementary analyses
  included cerebellum. It also estimated crossvalidated task-by-task Gram
  matrices and released MDTB-derived group-average activity patterns.

Consequently, this episode must not claim the first cross-task
individualization, first generalizing parcellation, first task-general spatial
covariance, or first crossvalidated condition geometry. Its narrower target is
development-refit discrimination among parcel membership, smooth relocation,
certified isometry, and bounded non-isometry for stable between-person B-only
variation. The geometry endpoint is distinctive only as an individual
prediction target: an A-derived model must predict B-only deviations from the
development source geometry.

Task set and session are confounded. A non-isometric result supports stable
geometry-changing variation across sets; it does not establish task
dependence. This remains internal mechanistic reanalysis, not confirmation.

### Published-parameter firewall

An artifact is audit-B-derived if any numeric value was fitted, selected,
averaged, normalized, or otherwise computed using an assigned audit
participant's Task B. Publication, anonymization, group averaging, resampling,
or incorporation into an atlas does not remove that status. The firewall
applies to parameters, derived maps, initializers, and caches, not only raw
files.

Papers, equations, algorithm descriptions, source code, anatomy-only
templates, labels, and condition metadata may inform the frozen grammar. The
following numeric artifacts are prohibited before the primary audit report:

- King-2019 full-MDTB atlases, condition/individual maps, and parcellations,
  including `atl-MDTB10` and `con-MDTB*`;
- released `NettekovenSym*` or `NettekovenAsym*` label/probability maps,
  arrangement priors, emission parameters, hierarchies, response profiles,
  checkpoints, and connectivity weights;
- Nettekoven-2026 covariance matrices, individual parcellations, connectivity
  weights, evaluation tables, and Task-B-selected settings;
- Arafat-2026 task-library maps (Zenodo `18793343`), MDTB-derived Gram
  matrices, battery rankings, cached batteries, and B-to-A predictions; and
- any basis, warp, transform, normalization, support mask, winner, or other
  descendant of those artifacts.

Re-estimating only response profiles does not sanitize an exposed functional
atlas: its arrangement prior or hierarchy may already encode audit Task B.
An exposed initializer remains exposed. Such artifacts may be inspected only
after the primary report as labelled descriptive oracles and cannot rescue or
reinterpret the outcome.

### Mandatory refitting and cross-fitting

Every data-dependent class obeys these rules:

1. For development pseudo-target `i`, every shared prior, emission model,
   hierarchy, response profile, source map, basis, regularizer, threshold, and
   normalization is fitted without `i`'s Task B. Anatomy and Task A enter only
   through the frozen target-personalization rule.
2. For audit, shared parameters are refitted once on development participants,
   hashed, and locked. Audit anatomy and Task A estimate only prespecified
   participant-specific parameters; audit-cohort Task A cannot update shared
   parameters or hyperparameters.
3. A claimed external or leave-MDTB-out prior needs an immutable training
   manifest proving that neither `ds002105`, an MDTB derivative, nor an audit
   participant contributed. Otherwise it is treated as exposed and refitted.
4. Implementations disable implicit downloads and inspect package, user, and
   job caches. Every fitted artifact records participant IDs, sessions,
   conditions, upstream hashes, code commit, seed, and output hash.

## EP03 collision and campaign dependence

Repository inspection on 2026-09-20 found no explicit MDTB reference in
EP03's current contract, but EP03's planned accession manifest is absent while
MDTB declares lineage to OpenNeuro `ds002105` v1.1.0. Before roles are frozen,
crosswalk EP03's eventual manifest against `ds002105`. A match binds EP03 and
EP15 into one correlated exposure family and prohibits an independence claim.

## Launch blockers

Episode 15 is not launch-ready until all of the following are present and
verified:

- full local SHA-256 manifest matching the provider record;
- license/provenance reconciliation under the conservative use rule;
- authenticated common-space route and common cerebellar support for all 24;
- four-partition map manifest and evidence supporting the error-independence
  assumptions;
- human-signed scientific margins, numerical certificates, exact contrast
  family, multiplicity code, and outcome-blind 12/12 power/abstention record;
- outcome-blind 12/12 role manifest and balance report;
- role-filtered immutable handoffs and a permission-separated evaluator;
- exposure ledger covering EP03, prior MDTB work, people, agents, caches,
  downloaded atlases, and prepared derivatives;
- development-only refit implementation for M1 and a certified construction
  for M2--M4, including the exact `K0` source geometry;
- synthetic fixtures and full model-class confusion matrix;
- frozen 20-row coverage manifest, trial schema, lock manifest, retry policy,
  and one-open audit receipt format; and
- a canonical adaptive-search program binding.

None is created by this dataset contract.
