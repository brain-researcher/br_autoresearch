# EP17 data: CNeuroMod responses, concepts, and measurement roles

The October12 [adopted block-variance amendment](outputs/proposed_score_specific_validity_followup.md)
retains every original development image, column, participant, ROI, stage and
uncertainty block. For primary R² only, a block need not have positive centered
variance; it must still have finite responses/computed moments and positive
finite SST in every column around the matching frozen training mean. No
support is selected or repaired using development values. All training and
validation variance, calibration support and secondary-statistic rules remain.
The same block rule applies to the sealed audit only after its original
predictor/relation freeze; audit access and opening count are unchanged.

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

## Public online resources — checked October 6, 2026

| Public source | Available information and EP17 use | Remaining limit |
| --- | --- | --- |
| [Official CNeuroMod-THINGS repository README](https://raw.githubusercontent.com/courtois-neuromod/cneuromod-things/main/README.md) | Navigates the event/preprocessing records under `THINGS/fmriprep`, trial/image beta and analysis code under `THINGS/glmsingle`, anatomical products, and fLoc/retinotopy methods. It explains selective DataLad retrieval without fetching the full collection. Use the source tree to identify the specific format/processing documentation needed by an adapter. | This is the current public repository description, not a change to the fixed 1.0.1 input. Available functional-localizer or public noise-ceiling products do not replace the frozen anatomical ROI or calibration-only estimator. |
| [Pinned stimulus README](https://raw.githubusercontent.com/courtois-neuromod/things.stimuli/f8f5162f79fb2e9fb0d8cbb8bac610db11a54180/README.md) | Names `images_fmri.zip` and its documented `password_images.txt` interface. Also identifies `annotations/task-things_desc-manual_annotation.tsv` as CC0 and THINGSplus annotations as CC BY. Use for archive identity, annotation provenance and the distinction between annotation and pixel terms. | No password file was opened. Research/noncommercial pixel terms and authorized private adapter delivery still apply; annotation licensing does not grant pixel redistribution. |
| [Official access guide](https://docs.cneuromod.ca/contents/access.html) | Lists all four EP17 participants among the openly shared subset and distinguishes that route from the agreement for the entire six-person databank. | Public neural availability does not establish episode-specific reader isolation or change the accepted stimulus terms. No new access application is needed merely to read these public instructions. |

These online resources supply actionable format, acquisition-route and
annotation information. The remaining task is implementing authorized delivery
and role separation, not asking the scientist to identify an unspecified
public source. No neural arrays, pixels, secrets or result notebooks were
opened during this check.

## What is available

| Resource | Current state | EP17 use |
| --- | --- | --- |
| CNeuroMod-THINGS 1.0.1 neural and structural subset | Acquired in restricted, read-only storage | B/C/D GLMsingle responses, anatomical candidates, events, masks, and run metadata |
| Official release archive | Preserved with the source | Provider provenance and recovery |
| CNeuroMod fMRI stimulus archive | Acquired under its source terms; encrypted and unextracted | Exact pixels for model features and event-image alignment |
| THINGS/THINGSplus taxonomy | Metadata crosswalk and deterministic role assignment completed October 4 | Fixed role balance; optional semantic analysis remains subject to its contract |

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

The completed metadata qualification establishes 720 eligible concepts and
3,634 repeat-qualified image identities (from 4,320 observed and 3,840 common
identities). The retained role counts are 2,423 development, 605 calibration
and 606 audit images. These are metadata findings, not decoded pixel identity
or neural response qualification. If the event-derived universe is not exactly 720
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

The metadata assignment completed October 4: the 480/120/120 concept roles,
six development folds, 24 development uncertainty blocks and twelve audit
blocks are fixed. [The completion record](outputs/experiment_log.md)
and [source-role specification](outputs/source_roles_proposal.md) preserve the
exact assignment and bounded optimization rule. This does not release neural
or audit-feature handoffs.

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

The [prediction/calibration specification](outputs/prediction_qualification_specification.md)
and [endpoint contracts](outputs/measurement_contract_specification.md) were
adopted October 4 before scores, including the repeat-aware ceiling estimator,
0.10 denominator floor and numerical eligibility rules. The actual
calibration maps and eligible voxel set have not been computed; those
prospective choices cannot be revised after candidate scores are visible.

The builder releases those fixed artifacts, not the underlying calibration
responses. Development and audit concepts never select voxel support or
estimate a ceiling.

## Anatomical support and geometry

The acquired `aparcaseg` volumes now have a qualified structural mapping to
the nominated bilateral fusiform ROIs. The following rules govern model
scoring; structural mapping and quota construction are complete, while the
calibration-dependent finite universe remains unfinished:

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

Structural mapping has completed with original-column identity preserved
across all 881,359 source columns and every nominated anatomical component.
The adopted bilateral fusiform route fixes N=256, eight spatial bins and
32 voxels per bin for each participant/ROI. Private calibration-dependent
finite support, reliability membership and ceilings are now computed under
those unchanged definitions; see the
[empirical calibration completion](outputs/experiment_log.md#2026-10-09-0141-pacific--calibration_support-full-empirical-maps-and-supports-completed).
The
[structural and calibration specifications](outputs/prediction_qualification_specification.md)
separate these completed geometry decisions from neural eligibility.

For a pure support comparison, create one common out-of-fold prediction bank
for all eligible voxels and apply both support masks afterward. Separate
support-specific model fitting is not allowed.

## Images, features, and model exposure

The exact CNeuroMod stimulus archive cannot be replaced by filenames or by
selecting similarly named images from another THINGS archive. The trusted
builder must extract the authorized source, join each provider image ID to the
event table, and supply role-safe image features.

The [pinned provider README](https://raw.githubusercontent.com/courtois-neuromod/things.stimuli/f8f5162f79fb2e9fb0d8cbb8bac610db11a54180/README.md)
names `password_images.txt` for `images_fmri.zip`. This documents the interface;
authorized local delivery and bounded use are unverified. No secret is stored
here. Existing accepted research terms remain in force.

Before neural scoring, freeze three controlled promotion pairs, one
trained-versus-random falsifier, and at most one optional pair. For each
checkpoint, record its named version, architecture, objective, training data,
parameter count, license, image preprocessing, eligible layers, feature
shape, and known THINGS or target-image exposure.

Known exact or near-duplicate exposure prevents a checkpoint from supporting
a clean promotion claim. Unknown exposure remains unknown.

## EP18 overlap

The completed metadata comparison establishes that all 720 EP17 concepts and
all 3,634 qualified EP17 image identities occur in EP18's main experiment.
Both designs and roles are retained as correlated, design-exposed studies;
neither supplies independent stimulus-family confirmation of the other.
This is an identity/crosswalk finding, not proof of pixel-byte or feature-array
identity. Known checkpoint reuse and access history, with their limitations,
are retained in the [completed exposure record](outputs/experiment_log.md#2026-10-04-2353-utc--cross_episode_exposure-exact-overlap-completed).

## What must exist before neural model comparison

The common universe, concept roles/folds/blocks, structural atlas/column
mapping, N/quotas, model-feature definitions, linking/calibration conventions,
scale-specific margins, seven-endpoint/five-edge audit contracts and EP18
exposure disposition are recorded. The calibration/support/operator/uncertainty
cores passed 33 focused checks. The adopted margins do not require a new
personal ratification; see the [decision provenance](outputs/source_roles_proposal.md#october-6-access-and-decision-handoff--recommendations-unadopted).

EP17 is not analysis-ready. The remaining setup must supply:

- authorized stimulus delivery/extraction and pixel-to-event verification;
- enforced reader release and access checks for the completed private
  role-filtered copies (amended 2026-10-12: procedural same-account
  separation replaces the enforced boundary, per the
  [GOAL amendment](GOAL.md#data-readiness-and-access-boundary); the
  development view is still released separately, and audit/sealed access is
  logged and disclosed);
- remaining development/audit response qualification and calibration-operator
  applicability; completed private B/C/D calibration maps, ceilings and fixed
  supports do not establish those separate conditions;
- role-safe features and full common-grid feature feasibility, with unknown
  pretrained-image exposure retained as unknown; and
- the complete evaluator and an outcome-free dry run.

Until these are complete, neural candidate scoring and audit access remain
closed. A failure of role identity, data alignment, or access separation is a
technical limitation, not a scientific null.
