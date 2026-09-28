# EP15 data: MDTB Task A, Task B, and participant roles

EP15 uses the Functional Fusion Multi-Domain Task Battery (MDTB) v1.0
derivative to ask why individual cerebellar task maps differ. [GOAL.md](GOAL.md)
defines the scientific comparison; this file describes what is available, how
participants and task sets are separated, and what still blocks analysis.

## Source

| Item | Value |
| --- | --- |
| Dataset | Functional Fusion MDTB |
| Release | v1.0 |
| Zenodo record | `16788784` |
| DOI | `10.5281/zenodo.16788784` |
| Raw-data lineage | OpenNeuro `ds002105`, version `1.1.0` |
| Primary paper | King et al., *Nature Neuroscience* (2019), `10.1038/s41593-019-0436-x` |
| Read-only source | `/oak/stanford/groups/russpold/data/br_autoresearch_data/functional_fusion_mdtb/zenodo-16788784-v1.0` |

The source was acquired on 2026-08-21. It contains participant derivatives,
not a ready-made EP15 input. Availability does not authorize reading audit
Task-B maps.

License statements differ across the Zenodo record, bundled metadata, and a
provider page. Until the owners reconcile them, retain attribution and use the
data noncommercially. Do not commit participant maps or derived participant
data to this repository.

## What the release contains

The release has 24 participant archives. Every participant has:

- one Task-A session (`ses-s1`) and one Task-B session (`ses-s2`);
- 16 runs per task set;
- 736 Task-A beta images and 784 Task-B beta images;
- Task-A and Task-B regressor tables and design matrices;
- a cerebellar mask and a nominal transform to SUIT space; and
- anatomy, tissue maps, mean BOLD, variance maps, and cortical surfaces.

The product contains beta estimates, not raw time series, events/confounds,
residual time series, or resting-state data. The task-design `rest` beta is not
resting-state fMRI. EP15 cannot re-estimate the GLM, add a new motion model, or
use resting-state connectivity.

Seven participants lack an overlap mask and inverse SUIT transform but retain
the forward transform, cerebellar mask, and Task-A/B betas. They are exactly
the participants without the optional rest session. This cannot become a
silent exclusion. EP15 must authenticate one registration route that works for
all 24 participants or reproducibly regenerate the missing products before
roles are assigned.

## Task structure

After instruction regressors are removed, Task A and Task B contain 26 task
families and 47 distinct conditions. Fourteen conditions are shared. The
primary endpoint is the 18 Task-B-only conditions:

| Domain | Conditions |
| --- | --- |
| CPRO | `CPRO` |
| prediction | `Prediction`, `PredictViol`, `PredictScram` |
| spatial map | `SpatialMapEasy`, `SpatialMapMed`, `SpatialMedDiff` |
| movie | `NatureMovie`, `RomanceMovie`, `LandscapeMovie` |
| mental rotation | `MentalRotEasy`, `MentalRotMed`, `MentalRotDiff` |
| emotion processing | `BodyMotionIntact`, `BodyMotionScram` |
| response alternation | `RespAltEasy`, `RespAltMed`, `RespAltDiff` |

Each of the seven domains has weight `1/7`; conditions divide that weight
equally within their domain. All 18 conditions remain separate when building
the condition-geometry matrix. Maps are never averaged within a domain before
cross-products, because opposing condition effects could cancel.

The 14 shared conditions, including `rest`, are used for Task-A calibration
where permitted and for a locked post-decision Task-B diagnostic. They do not
enter the primary Task-B-only endpoint.

## Participant roles

Twelve whole participants are assigned to development and twelve to audit.
The earlier 16/8 proposal is withdrawn. The 12/12 roles have not yet been
instantiated.

The assignment is Task-B-blind:

1. verify archive, anatomy, Task A, transform, and expected-file availability
   for all 24 participants without reading a Task-B numeric map or score;
2. compute one Task-A-only reliability summary for each participant: the
   median centered spatial correlation between runs 1–8 and runs 9–16 across
   all non-instruction Task-A conditions, without excluding anyone based on
   that value;
3. enumerate every 12/12 assignment and balance sex, native-English status,
   optional-rest availability, age, and Task-A reliability in a fixed order;
4. break an exact tie with the lexicographically first sorted audit-ID tuple.

Missing categorical values form their own level. Missing age or an
uncomputable Task-A reliability blocks the split; it does not trigger
imputation, exclusion, or participant replacement. The custodian returns the
roles and balance report, not participant-level audit reliability values.

Outcome-blind simulations must show that the complete selection and audit
decision is usable with 12 development and 12 audit participants. Scientific
margins cannot be widened to make that simulation pass. A different split
would require an explicit design amendment before either role's Task-B
outcomes are exposed.

## Physical separation

The participant archives mix Task A and Task B and cannot be mounted to the
search process. After the split, a trusted operator creates three ordinary,
role-filtered handoffs:

| Handoff | Contents | Reader |
| --- | --- | --- |
| Development | Anatomy, Task A, and Task B for 12 development participants | Development worker |
| Audit calibration | Anatomy and Task A for 12 audit participants | Calibrator, only after panel lock |
| Sealed audit | Task B for the same 12 audit participants | Trusted evaluator, once after prediction lock |

The candidate workspace must not contain mixed audit archives, exposed
MDTB-fitted atlases, or caches from which audit Task B can be reconstructed.
Audit Task A may estimate only the participant-specific parameters of already
locked recipes. It cannot update shared priors, choose a branch, set a margin,
or change stopping.

Every assigned audit participant remains in the analysis. A frozen
evaluator-side finite-value or geometry failure causes technical failure; no
participant may be replaced and no reduced-N primary result may be substituted.

## Map construction to freeze before scoring

One common route must be used for all four model classes. Before a development
Task-B score can favor a candidate, freeze:

- the authenticated SUIT transform direction, reference grid, interpolation,
  and common cerebellar gray-matter support for all 24 people;
- affine, orientation, coverage, missing-value, voxel-order, equal-weight, and
  spatial-centering rules;
- arithmetic condition maps from run-level betas with no adaptive smoothing,
  whitening, or Task-B-derived participant scaling;
- one optional positive global gain estimated from target Task A and applied
  uniformly to all voxels and Task-B conditions;
- map halves `runs 1–8` and `runs 9–16`;
- geometry partitions `runs 1–4`, `5–8`, `9–12`, and `13–16`;
- the primary geometry pairing and fixed non-selective sensitivity pairings;
- the seven-domain weights and condition-pair weights; and
- the actual development-only M3 source geometry `K0` and its positive scale.

The map halves and geometry partitions rely on independent measurement error.
Task-A and synthetic/noise checks must support that assumption before Task-B
model comparison. Shared nuisance that invalidates the cross-products blocks
the estimand; a favorable model score cannot waive the problem.

For each development pseudo-target, all source maps, priors, bases, response
profiles, regularizers, thresholds, and normalizations exclude that
participant's Task B. For audit, they are refitted once on development
participants and frozen before audit Task A is used.

## Prior work and the novelty boundary

MDTB is heavily studied, and all 24 participants are publication-exposed.
Relevant prior work already includes:

- King et al. (2019): participant-specific cerebellar parcellation,
  cross-task boundary evaluation, and crossvalidated condition distances;
- Nettekoven et al. (2024), `10.1038/s41467-024-52371-w`: hierarchical
  probabilistic atlases personalized with one task set and evaluated on the
  other, including novel tasks;
- Nettekoven et al. (2026), `10.64898/2026.03.09.710558`: Task-A/rest spatial
  covariance, individual parcellation, and B-only evaluation; and
- Arafat et al. (2026), `10.7554/eLife.111868.1`: individualized task-map
  prediction and crossvalidated task-by-task geometry using MDTB.

EP15 must not claim the first cross-task individualization, generalizing
parcellation, task-general covariance, or condition geometry. Its narrower
question is whether Task-A-derived information distinguishes parcel
membership, smooth relocation, geometry-preserving remapping, and bounded
geometry change for B-only individual variation.

The key novel prediction is participant-specific: Task A must predict the
signed deviation of the full B-only condition-geometry matrix, not merely show
that a group-average geometry exists.

## Published-parameter firewall

Papers, equations, algorithm descriptions, source code, anatomy-only
templates, labels, and condition metadata may inform the design. Numeric
artifacts fitted or selected with an assigned audit participant's Task B may
not enter a model, initializer, synthetic generator, support, or selection
rule—even if they are public, anonymized, averaged, or resampled.

This excludes MDTB-fitted atlases and parcellations, Nettekoven response
profiles and covariance products, Arafat task-library maps and Gram matrices,
and descendants of those artifacts. An exposed functional atlas is not made
clean by refitting only its response profiles. Implementations must disable
implicit downloads and inspect package, user, and job caches.

EP03 may share the OpenNeuro `ds002105` lineage. Before participant roles are
frozen, compare its eventual source inventory with MDTB. A match makes EP03 and
EP15 correlated exposure, not independent evidence.

## What remains before analysis

EP15 is not analysis-ready. It still requires:

- one verified registration route and common support for all 24 participants;
- the fixed four-partition condition table and error-independence checks;
- the Task-B-blind 12/12 role assignment and three permission-separated
  handoffs;
- scientist-approved signal, adequacy, increment, and geometry margins;
- exact M1–M4 capacities, identifiability and operator certificates, source
  geometry, inference implementation, and scratch limit;
- outcome-blind synthetic qualification of the complete decision tree;
- the published-parameter/cache and EP03 exposure review; and
- a dry run showing that all audit predictions can be written while audit
  Task B is unavailable.

Until those items are complete, candidate scoring and audit Task-B access stay
closed. Setup failure is a readiness limitation, not evidence for or against a
scientific class.
