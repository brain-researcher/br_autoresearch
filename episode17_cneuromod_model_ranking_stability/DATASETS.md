# EP17 data: CNeuroMod responses, concepts, and measurement roles

EP17 uses one source, CNeuroMod-THINGS 1.0.1, to ask whether a controlled
visual-model relation is stable or changes for a predictable measurement
reason. [GOAL.md](GOAL.md) defines the scientific comparisons; this file
describes the available data, concept roles, and remaining setup work.

The 2026-09-30 paper focus nominates DINO versus category-supervised ResNet-50,
anatomical bilateral fusiform, and the isolated D-stage `A_N → R_N` support
edge. [GOAL.md](GOAL.md) lists three concrete pair proposals. Nominations are
not evidence of checkpoint control, clean image exposure, atlas compatibility,
sufficient voxel support, or executable readiness. The operative roles,
five-edge coverage and closed outcomes below are unchanged. The 2026-10-02
amendment permits named recipe-confounded pairs for a controlled measurement
comparison, not causal attribution to the recipe; checkpoint and exposure
qualification are still required.

## What is available

| Resource | Current state | EP17 use |
| --- | --- | --- |
| CNeuroMod-THINGS 1.0.1 neural and structural subset | Acquired in restricted, read-only storage | B/C/D GLMsingle responses, anatomical candidates, events, masks, and run metadata |
| Official release archive | Preserved with the source | Provider provenance and recovery |
| CNeuroMod fMRI stimulus archive | Acquired under its source terms; encrypted and unextracted | Exact pixels for model features and event-image alignment |
| THINGS/THINGSplus taxonomy | Available but not yet frozen for EP17 | Role balance and one optional semantic-stratum analysis |

The restricted source is approximately 128 GiB and contains all four planned
participants: `sub-01`, `sub-02`, `sub-03`, and `sub-06`. Source availability
does not authorize reading neural arrays. The mixed source still contains
development, support/calibration, and audit rows together.

### Source references

| Item | Source |
| --- | --- |
| Dataset | CNeuroMod-THINGS |
| Release | `1.0.1` |
| Data paper | <https://www.nature.com/articles/s41597-026-06591-y> |
| Official repository | <https://github.com/courtois-neuromod/cneuromod-things> |
| Archived release | <https://doi.org/10.5281/zenodo.17881592> |
| Restricted source | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/restricted/cneuromod-things-1.0.1-restricted-raw` |
| Steward stimulus area | `/oak/stanford/groups/russpold/users/zijiao/br_autoresearch_data/steward_acquisition` |

Exact provider versions and acquisition details stay with the restricted
source record. They do not need to appear as a chain of identifiers in the
scientific question.

The participant release is CC0 and project code is MIT. Those terms do not
cover THINGS image pixels. Image extraction, feature computation, and any
redistribution must follow the recorded THINGS research terms.

## Neural products

Each participant has one aligned GLMsingle response family:

| Stage | Product | What changed from the previous stage? |
| --- | --- | --- |
| B | `TYPEB_FITHRF.mat` | Fitted HRF; no GLMdenoise or ridge regularization |
| C | `TYPEC_FITHRF_GLMDENOISE.mat` | Adds GLMdenoise |
| D | `TYPED_FITHRF_GLMDENOISE_RR.mat` | Adds ridge regularization to the response estimate |

The B/C/D matrices must agree on trial identity, response units, voxel index,
geometry, and image mapping before a stage comparison is valid.

The source also provides a public D-stage noise-ceiling map. It may be used
only as a post-freeze positive-control diagnostic because its construction may
mix stimulus roles. EP17 must estimate B-, C-, and D-specific ceilings from
support/calibration concepts only.

## The common image and concept universe

The trusted builder first uses event and derivative metadata, not neural
values, to identify the exact images shared by all four participants. A
primary image must:

- have the same provider image ID and concept ID in all four participants;
- meet the fixed repeat-completeness rule in every participant;
- map to the same trial across B, C, and D; and
- pass only structural checks defined before model features or neural values.

The public design suggests 720 common concepts, but that count is not treated
as an observed local fact. If the event-derived universe is not exactly 720
eligible concepts, stop before neural access and revise the design explicitly
or abandon it.

The metadata used for eligibility and balance may include image and concept
IDs, presentation indices, session/run/trial position, acquisition order,
repetition lag, and provider schedule exceptions. Recognition answers,
correctness, reaction time, other behavioral outcomes, candidate features,
and neural values are excluded.

## Concept roles

Assign each whole concept globally:

| Role | Count | Permitted use |
| --- | ---: | --- |
| Development | 480 | Nested model fitting, measurement comparisons, explanatory modeling, and finalist selection |
| Support/calibration | 120 | Repeat reliability, reliable-voxel support, and stage-specific noise ceilings only |
| Sealed audit | 120 | One direct application of the frozen development predictor |

Every exemplar, repetition, participant row, beta stage, and derived neural
value inherits its concept's role. No concept may cross roles.

Use one deterministic metadata-only assignment that balances the frozen
THINGS/THINGSplus taxonomy, five- versus six-exemplar status, session and
acquisition order, and repetition lag. Freeze the taxonomy release, concept
crosswalk, unmapped and multilabel handling, group-merging rule, minimum cell
size, balance objective, tolerances, and tie rule before assignment.

Development concepts form six outer folds of 80 concepts and 24 uncertainty
blocks of 20 concepts. Audit concepts form twelve blocks of 10 concepts. The
block assignment is independent of model features and neural responses.

## Role-separated handoffs

The restricted source physically mixes all roles and must never be mounted to
the adaptive search worker. A trusted builder creates four materialized
handoffs:

| Handoff | Contents | Reader |
| --- | --- | --- |
| Metadata | IDs, roles, folds, blocks, repeat and stage mappings; no neural or behavioral outcomes | Builder, controller, evaluator |
| Calibration neural | B/C/D repetitions for the 120 support concepts | Trusted builder only |
| Development | Development responses, IDs, folds, role-safe model features, and frozen calibration-derived reliability, support, and ceiling artifacts | Search worker |
| Sealed audit | Audit responses, IDs, twelve blocks, and model features for the already locked predictor | Trusted evaluator after the final relation is frozen |

These handoffs must be ordinary role-filtered files rather than links back to
the mixed source. Check their intended readers and targets before release.
The search worker never receives raw calibration or audit responses.
Audit feature arrays and embeddings are evaluator-only until the write-once
model, contract, and forecast lock; the worker cannot inspect them to choose a
candidate.

## Response construction

Within participant, stage, unique image, and voxel, average retained trialwise
betas with equal repetition weight. Give each unique image equal weight in
fitting and scoring. Trialwise or inverse-variance-weighted endpoints are
sensitivities only.

Support/calibration concepts define:

- repeat reliability at B, C, and D;
- the change in reliability across B→C and C→D;
- reliable-voxel support under frozen spatial quotas; and
- stage-specific noise ceilings and their eligibility floor.

For reliability, sort retained presentations of each image by acquisition
order and alternate them between two deterministic repeat halves. Average
within each half, then correlate the two half-response vectors across
calibration images for each participant, ROI, voxel, and beta stage. The beta
operator uses the Fisher-z change between stages; `R_N` ranks the D-stage
calibration reliability within each participant and ROI. The numerical
Fisher-z boundary rule is fixed before scores.

The exact noise-ceiling estimator, its stage-specific floor, and the finite
ceiling-quality threshold still need to be chosen before launch. They cannot
be filled in after candidate scores are visible.

The builder releases those fixed artifacts, not the underlying calibration
responses. Development and audit concepts never select voxel support or
estimate a ceiling.

## Anatomical support and geometry

The acquired `aparcaseg` volumes are candidates, not yet a finished visual-ROI
definition. Before model scoring:

1. freeze a parcel-to-visual-ROI crosswalk;
2. resample labels to each participant's GLMsingle T1w mask with
   nearest-neighbor interpolation;
3. verify affine, shape, orientation, and voxel correspondence across B/C/D;
4. retain the common finite voxel universe before any model output is used;
5. define empty or small-ROI refusal rules; and
6. freeze `N`, the anatomical subset-bank size, spatial bins, quotas, and
   deterministic bank construction.

The primary reliable support `R_N` and every anatomical subset in `A_N` must
contain the same number of voxels in every subparcel and spatial bin. Without
that match, a support comparison also changes spatial composition.

For a pure support comparison, create one common out-of-fold prediction bank
for all eligible voxels and apply both support masks afterward. Separate
support-specific model fitting is not allowed.

## Images, features, and model exposure

The exact CNeuroMod stimulus archive cannot be replaced by filenames or by
selecting similarly named images from another THINGS archive. The trusted
builder must extract the authorized source, join each provider image ID to the
event table, and supply role-safe image features.

Before neural scoring, freeze three controlled promotion pairs, one
trained-versus-random falsifier, and at most one optional pair. For each
checkpoint, record its named version, architecture, objective, training data,
parameter count, license, image preprocessing, eligible layers, feature
shape, and known THINGS or target-image exposure.

Known exact or near-duplicate exposure prevents a checkpoint from supporting
a clean promotion claim. Unknown exposure remains unknown.

## EP18 overlap

EP17 and EP18 share the THINGS stimulus ecosystem. Before either sealed
outcome is opened, record exact concept and image overlap, feature or
checkpoint reuse, and first-access history. Either reserve disjoint concepts
or freeze both designs and describe later evidence as correlated and
design-exposed. Different measurement modalities do not make the stimulus
evidence independent.

## What must exist before neural model comparison

EP17 is not analysis-ready. The next setup stage must produce or freeze:

- controlled extraction and the exact event-image join;
- the verified 720-concept common universe and repeat-completeness rule;
- the deterministic 480/120/120 assignment, folds, and concept blocks;
- role-filtered handoffs and their access checks;
- B/C/D trial, response, geometry, and voxel alignment;
- the visual-ROI crosswalk, `N`, subset-bank size, spatial bins, and quotas;
- the calibration reliability maps, ceiling estimator, floor, and eligible
  voxel set;
- named controlled model pairs, exposure records, layers, PCA and ridge grids;
- practical effect margins, operator-prediction error margins, and the exact
  crossed-bootstrap procedure;
- the five-edge contract table, every endpoint's same-stage/same-support audit
  homolog, separate raw and normalized margins, and audit adequacy widths; and
- the EP17/EP18 exposure decision and an outcome-free evaluator dry run.

Until these are complete, neural candidate scoring and audit access remain
closed. A failure of role identity, data alignment, or access separation is a
technical limitation, not a scientific null.
