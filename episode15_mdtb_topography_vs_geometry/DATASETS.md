# EP15 data: MDTB Task A, Task B, and participant roles

## Adopted common-valid audit support — October 21, 2026

The scientist adopted the [complete exploratory common-support followup](outputs/paper_plan.md#proposed-daily-decision--october-19-2026).
The same 12 audit people, 18 B-only conditions, four partitions, all 16 variants
and every control remain. Reuse existing role-filtered containers and the
original188-trained frozen predictions; do not recalibrate or refit.
Renewed B access may construct one complete strict-sampler availability
intersection over every required selected regressor and freeze it before any
loss. No beta magnitude, sign, variance or score selects locations.

Only this availability-conditioned spatial target is scored, after restricting
and recentering observations and predictions identically. Fewer than two
locations stops without an alternative support search. Shared B14 and private
reliability stay closed. Preserve the original failed complete188 evaluation;
prior/renewed access and same-account procedural separation are disclosed.
The result is exploratory, with no untouched audit, original188 pass, winner,
mechanism or whole-cerebellum claim. No dataset or data agreement is added.

## Adopted audit data roles — October 19, 2026

The scientist adopted the complete [observed-map evaluation](outputs/proposed_observed_map_audit.md).
The original 12 audit people (sub-02,03,04,06,09,10,19,22,26,28,29,30) now
serve this prospective predictive evaluation. Shared quantities use only the
12 original development people. After panel lock, audit anatomy and all 29
Task-A conditions may calibrate individual predictions. Only after all ordinary
and control predictions freeze may the evaluator open their 18 B-only
conditions, once, on the development-frozen 188 locations and four partitions.
Every variant and control is retained; required missing/nonfinite values stop
the complete panel without exclusions, imputation or support reselection.

The shared 14 Task-B conditions and private reliability remain closed. Existing
role-filtered containers are reused; no new dataset is added. Separation remains
procedural within the same account. Older audit-closed statements below apply
to their earlier routes; this is the sole adopted exception. The original
mechanism question remains unresolved, and these outcomes cannot later be
untouched confirmation for it. The full specification controls inference and
any disclosed unchanged engineering resumption.

EP15 uses the Functional Fusion Multi-Domain Task Battery (MDTB) v1.0
derivative to ask why individual cerebellar task maps differ. [GOAL.md](GOAL.md)
defines the scientific comparison; this file describes what is available, how
participants and task sets are separated, and what still blocks analysis.

**October 12, 23:18 UTC:** the adopted common finite-support amendment replaces
the original208-location axis only for the observed-beta route. Its single mask
intersects strict sampling validity across all9024 released development maps,
retaining all people, conditions, runs and regressors. Original anatomy support
stays unchanged. Target Task-B availability contributes to the common axis, so
this is an availability-conditioned descriptive estimand, not wholly unseen
Task-B prediction. See the [adopted specification](outputs/proposed_finite_support_followup.md).
The October12 22:29 description below records the preceding208-point route.

**October 12, 22:29 UTC:** the scientist adopted the
[observed-beta descriptive route](GOAL.md#adopted-observed-beta-descriptive-route--october-12-2026).
It permits the complete fixed twelve-person/eighteen-condition/sixteen-variant
development panel while cross-partition error support is unresolved. It changes
no participant, source map, 208-location support, partition, control or access
boundary. The qualified fit still requires the coordinator's explicit
development release and resource admission. Reporting is limited to unadjusted
observed-map losses and clearly assumption-conditional signed summaries;
reliable-signal/recovery fractions and audit access are prohibited. The
derivative's estimation-scope information remains useful for the original
question, but is not a new prerequisite for this descriptive route.

The October 6 scientist agreement opens an exploratory real-data development
direction after the existing reader/source/model requirements are met. It
adopts only the intended 80% recovery and five-percentage-point increment
standards; it does not validate inference, adopt the other five margin values,
release the opaque containers or open audit Task B. See the
[adopted scope](GOAL.md#adopted-exploratory-real-data-direction--october-6-2026).

## Public online resources — checked October 6, 2026

| Public source | Available information and EP15 use | Remaining limit |
| --- | --- | --- |
| [Functional Fusion MDTB v1.0 on Zenodo](https://zenodo.org/records/16788784) | Participant derivative archives, `README.md`, `participants.tsv` and dataset-description metadata; documents Task A=`ses-s1`, Task B=`ses-s2` and the Functional Fusion import layout. Use for release/file identity and interpreting the derivative. | The provider explicitly excludes resting-state data and time series from this release. Availability does not release audit Task B. |
| [Official Functional Fusion dataset implementation](https://raw.githubusercontent.com/DiedrichsenLab/Functional_Fusion/main/Functional_Fusion/dataset.py) | `DataSetMDTB` defines the two sessions, condition/half fields and `CondHalf` default; the native-space mapper links deformation and cerebellar-mask files. Use as a readable source-format/mapping reference. | The current `main` implementation also enables baseline subtraction and exposes optional preprocessing. These defaults are not EP15's adopted arithmetic or a frozen historical software version. |
| [Official deformation/sampling implementation](https://raw.githubusercontent.com/DiedrichsenLab/Functional_Fusion/main/Functional_Fusion/atlas_map.py) | `AtlasMapDeform` documents world-coordinate deformation, mask inputs and nearest-neighbor/trilinear/smoothed interpolation choices. It supports the source interpretation recorded in EP15's sampling specification. | It does not replace EP15's fixed 208-location support, strict missing-stencil rules or role-separated reader. |

These are public documentation/code resources, not missing participant results.
The remaining statistical and reader work is listed under
[analysis readiness](#what-remains-before-analysis); no provider defaults or
new payloads were adopted by this source check.

### Run-error provenance follow-up — checked October 7, 2026

The exact [v1.0 release README](https://zenodo.org/records/16788784/files/README.md?download=1)
still leaves preprocessing details incomplete. The [original study's Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC8312478/)
describe two scanning sessions per task set, eight runs each, run-specific GLMs,
FAST temporal autocorrelation modeling and an implicit rest baseline. These
describe the original analysis, not an authenticated execution record for the
2025 derivative.

The original author code maps runs 1–8 and 9–16 to the two actual scanning
sessions, then groups all sixteen run-specific designs in one task-set SPM
object for estimation: pinned [session mapping](https://github.com/maedbhk/MDTB-Cerebellum/blob/b905f6162ee1745f27b225aba6aea73270c39c64/sc1_sc2_imana.m#L548),
[specification](https://github.com/maedbhk/MDTB-Cerebellum/blob/b905f6162ee1745f27b225aba6aea73270c39c64/sc1_sc2_imana.m#L1549)
and [estimation](https://github.com/maedbhk/MDTB-Cerebellum/blob/b905f6162ee1745f27b225aba6aea73270c39c64/sc1_sc2_imana.m#L1804).
This organization does not itself prove correlated errors. It also does not
justify interpreting run-specific regressors as independently estimated
preprocessing throughout. If the release preserves that run order, EP15's
geometry pairs 1×2 and 3×4 each pair disjoint runs **within one actual scanning
session**; the Task-A/Task-B `ses-s1`/`ses-s2` labels should not be confused
with those within-task-set scanning sessions.

A concrete release-provenance gap remains: the [pre-release Functional Fusion
importer](https://github.com/DiedrichsenLab/Functional_Fusion/blob/342c3be9cb53e82aa3f22e32e582cc832d0b06b8/scripts/import/import_mdtb.py#L92)
references `GLM_firstlevel_7`, whereas the inspected original-paper script
defines GLM4/5/6. The partly commented importer is a clue, not a completed
export record. Its [import helper](https://github.com/DiedrichsenLab/Functional_Fusion/blob/342c3be9cb53e82aa3f22e32e582cc832d0b06b8/Functional_Fusion/import_data.py#L146)
copies beta images and a separate shared task-set residual-variance image.
Later [Functional Fusion extraction](https://github.com/DiedrichsenLab/Functional_Fusion/blob/2be43b20080f3e7042b2c1db04bff7a4b85b4b69/Functional_Fusion/dataset.py#L666)
performs additional residual normalization, contrast recombination and optional
baseline subtraction; EP15's direct arithmetic beta reader does not invoke
those operations.

The smallest useful remaining provider item is a **release-bound GLM7
specification/export record** identifying run/session mapping and the estimation
scope of temporal covariance, nuisance terms and beta normalization. That would
support assessment of the existing cross-partition error assumption, not
mathematically prove independence. Zero cross-partition error covariance is
still unverified. This review changed no partitions, payloads, preprocessing,
access rights or statistical claim.

## Source

### Public-source follow-up — October 16, 2026

The release-specific [README](https://zenodo.org/records/16788784/preview/README.md?include_deleted=0)
still leaves its preprocessing description unfinished. The current official
[MDTB importer](https://raw.githubusercontent.com/DiedrichsenLab/Functional_Fusion/main/scripts/import/import_mdtb.py)
still contains a commented GLM7 example with inconsistent source/destination
task arguments; it does not authenticate the actual export. This is not evidence
that the released task labels are wrong. The official
[import helper](https://raw.githubusercontent.com/DiedrichsenLab/Functional_Fusion/main/Functional_Fusion/import_data.py)
copies betas and a separate residual-variance image; the later
[extraction code](https://raw.githubusercontent.com/DiedrichsenLab/Functional_Fusion/main/Functional_Fusion/dataset.py)
applies additional normalization/recombination that EP15's arithmetic reader does
not invoke.

For interpretation under standing authority, retain the release's Task-A/Task-B
labels and assume its run indices preserve the author's chronology. Analyze
beta amplitudes as supplied. This source-supported chronology assumption does
not add an assumption of zero shared measurement-error covariance. Existing
matched controls can be reported now; stronger inference needs explicit
conditional sensitivity to shared errors, not indefinite provider contact.
No partition, support mask, formal margin or audit access changes here.

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
data noncommercially. Raw participant maps remain outside Git. The scientist's
October 14, 2026 publication instruction authorizes released figures and compact
derived result reports, including deidentified participant scores and dataset
labels, in this repository. This does not open the sealed audit or change source
data agreements.

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
silent exclusion. The authenticated common route uses the available forward
SUIT pull fields, so no inverse transform or participant exclusion is needed.
The [adopted sampling specification](outputs/anatomical_support_specification.md)
defines absolute native-world coordinates, strict trilinear sampling and one
source-derived 3 mm lattice. The completed anatomy-only intersection retains
208 locations shared by all 24 people. It is fixed; it is neither a canonical
SUIT3 atlas nor evidence of whole-cerebellum coverage. The source and support
completion records are linked in [the output index](outputs/README.md).

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
The earlier 16/8 proposal is withdrawn. The Task-B-blind 12/12 assignment
completed October 4; the fixed identities and aggregate balance are linked in
[the output index](outputs/README.md). Private reliability values remain
with the custodian. Opaque materialization subsequently completed, but reader
isolation/release and candidate scoring are still closed.

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

For the original formal class-decision route, outcome-blind simulations must
show that the complete selection and audit
decision is usable with 12 development and 12 audit participants. Scientific
margins cannot be widened to make that simulation pass. A different split
would require an explicit design amendment before either role's Task-B
outcomes are exposed. The October 6 exploratory branch retains the existing
split while deferring that whole-procedure qualification; it cannot claim the
formal route is complete.

## Physical separation

The participant archives mix Task A and Task B and cannot be mounted to the
search process. Opaque copying has produced 36 role-filtered containers with
all 36,576 expected members, retaining the following three handoff types.
The containers remain unreleased: same-UID staging does not establish the
required reader isolation. See the [materialization completion](outputs/experiment_log.md#2026-10-05-1513-utc--full-role-materialization-complete-and-charged-once).

**October 12, 2026:** the scientist adopted procedural same-account separation
in place of enforced reader isolation; see the
[amendment](GOAL.md#adopted-procedural-evaluation-separation--october-12-2026).
The handoffs and readers below are unchanged. The same account can still read
every container, so this boundary is procedural, not OS isolation. The
containers stay unreleased until a separately recorded development release.

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

For the original reliable-variation estimands, the map halves and geometry
partitions rely on independent measurement error.
Task-A and synthetic/noise checks must support that assumption before Task-B
model comparison. Shared nuisance that invalidates the cross-products blocks
the estimand; a favorable model score cannot waive the problem.
The adopted observed-beta route is the explicit exception to withholding the
descriptive full panel: it does not certify these estimands, change their
partitions or treat the Task-A diagnostic as an independence pass.

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

The next paragraph is a historical preparation snapshot from before the adopted
observed-map result; development release is no longer pending for that route.

The source route, 208-location support, Task-A-only reliability preparation,
fixed 12/12 roles, condition/run-partition member plan and opaque handoff
materialization are complete. The pre-role EP03 exposure review conservatively
records potentially correlated exposure of unknown extent; it does not claim
independence. The [selected endpoint metric](outputs/paper_plan.md#prospective-metric-and-decision-contrasts--selected-conventions-2026-10-05)
retains all 171 unique signed geometry entries, with equal weight over 28
domain-pair blocks. M3/M4 algebra, covariance, paired-jackknife and endpoint
arithmetic have passed their focused checks; these are preparation results,
not participant findings or a qualified inference procedure.

The adopted observed-map route completed its development handoff release,
source/cache protection, functional checks and complete model/control panel on
188 common valid locations. Its [released result](outputs/README.md) and
[numeric report](outputs/figures/observed_beta_development001/observed_beta_report.json)
are descriptive out-of-fold predictions, not reliable-signal or mechanism
inference. The 188-location amendment does not change the original route's
208-location estimand. Reader separation is procedural under the
[adopted amendment](GOAL.md#adopted-procedural-evaluation-separation--october-12-2026);
the separately adopted October 19 audit now follows the prediction-before-B
access order above.

For the original reliable-variation route, functional validity must hold on
its own support and cross-partition error independence remains unresolved.
The release-bound GLM7 specification/export record described
[above](#run-error-provenance-follow-up--checked-october-7-2026) is still missing.
The observed-map exception does not resolve these original-route requirements.

The formal route additionally needs the remaining two signal/geometry and
three native-signature margins, calibrated joint inference, the registered
whole-procedure qualification and the audit-prediction dry run. The accepted
`tau_remaining = 0.20` and `tau_increment = 0.05` resolve two intended standards
only. The seven-fixture and Bonferroni alternatives remain proposals. These
formal requirements no longer block the separately labeled exploratory point
report or the separately adopted observed-map audit. The original formal
class route remains unqualified. Setup failure is not evidence for or against
a scientific class.
