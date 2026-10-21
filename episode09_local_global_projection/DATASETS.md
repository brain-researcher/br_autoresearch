# EP09 data: registered dendrites and distal axonal arbors from the same neurons

EP09 needs whole-neuron reconstructions in which the dendritic tree and distal
axon belong to the same identified cell. [GOAL.md](GOAL.md) explains the
scientific question; this file explains which measurements support it, how a
valid target observation is constructed, and what must be settled before the
experiment can launch.

<a id="primary-source"></a>
## Current primary source: CEBSIT whole-cortex projectome

The scientist adopted **CEBSIT / Gao, Zenodo 16909126** on October 16 and
**Cux2-Cre plus Wfs1-Tg2-CreERT2** on October 17. These are the operative
source and line-set amendments in [SEARCH_POLICY.yaml](SEARCH_POLICY.yaml).
SEU-A1876 is the historical, infeasible source, not a current primary input.

| Contract item | Adopted specification |
| --- | --- |
| Resource | Gao whole-cortex projectome, Zenodo 16909126; the verified `data_v3.zip` release contains 18,621 registered cortical reconstructions. |
| Primary population | Cux2 L2/3 and L4; Wfs1 L2/3. No L5 IT claim. Scnn1a remains sensitivity-only. |
| Coordinates | CCFv3-registered, soma-centred rotation-invariant dendritic features. Native SWCs are not released; a native-versus-CCF sensitivity is unavailable. |
| Layer and grouping | Registered soma area/layer with the adopted coordinate-disagreement corrections; one fMOST brain treated as one animal under the identity assumption below. |
| Frozen cohort and roles | 1,703 cells: 1,114 development cells in 27 whole-brain groups and 589 audit cells in 16 groups. The original 12/8 group counts remain minimums, not the assigned split. |
| Measurement | Adopted v1.2 distal/arbor/observability rules; stage-A `n_coverage=6`, radial edges 50/100/150 µm, branch-order ranges 1–3/4–5/>5 and R_cap 600 µm. |

The [qualification record](outputs/runs/metadata_feasibility/attempt005/README.md)
documents 6,302/6,469 IT-line cells passing the dendrite screen. Its pending
Wfs1 wording describes the earlier qualification decision point; the later
adopted line-set amendment supersedes it. The
[role record](outputs/runs/measurement_and_roles/attempt002/README.md),
[stage-A record](outputs/runs/measurement_and_roles/attempt006/README.md) and
[current output navigation](outputs/README.md) establish the operative bindings
and access stage. The donor-population/radius amendment is still a proposal.

**Identity evidence and assumption.** The
[catalog-only source review](outputs/runs/metadata_feasibility/attempt004/README.md#cebsit-check-i-ran-this-catalog-only)
reports unique `sample_id` values for IT-line brains. Its
[retained census](outputs/runs/metadata_feasibility/attempt004/cebsit_catalog_census.json)
records 382 sample rows, 382 unique fMOST IDs and no fMOST ID with multiple
sample rows. These observations support, but do not independently certify,
the adopted interpretation of one fMOST brain per mouse; no explicit
provider animal/acquisition crosswalk is cited in these records. Animal-level
wording is conditional on that interpretation. Do not treat the source
amendment itself as independent evidence or export it to another episode as
an animal-identity certificate. The frozen whole-brain roles remain unchanged.

<a id="proposed-primary-source-cebsit-whole-cortex-projectome--unadopted"></a>
## Historical October 16 proposal: CEBSIT before adoption

The original proposal below records the earlier design state. Its pending
source/line decisions, counts, labelling uncertainties and no-download wording
are historical and are superseded by the adopted specifications above.
SEU-A1876 could not supply the registered design: at most 3 of 12 IT-driver
animals had donor pools. The scientist requested another dataset; the
[attempt004 search](outputs/runs/metadata_feasibility/attempt004/README.md)
then produced this proposal.

| Item | Proposed source |
| --- | --- |
| Release | Gao et al., *Neuron* 2025, [doi:10.1016/j.neuron.2025.10.019](https://doi.org/10.1016/j.neuron.2025.10.019). 18,621 SWC files mapped to CCFv3 on [Zenodo 16909126](https://zenodo.org/records/16909126): CC BY 4.0, one 13.6 GB archive. Catalog at the [CEBSIT portal](https://mouse.digital-brain.cn/projectome/whole-cortex). |
| Labelling | Sparse `CSSP-YFP` AAV injected regionally into cortical source areas of Cre-line mice. Labelling is source-defined. Whether the vector depends on Cre is still to be confirmed from the STAR Methods. |
| IT label | The driver line, independent of each cell's axon: Cux2-Cre (L2/3, L4), PlexinD1-2A-CreER (L2/3, L5), Tlx3-Cre_PL56 (L5), Rorb-IRES2-Cre-D (L4, L5). Wfs1 (L2/3) and Scnn1a (L4) would be a sensitivity only. |
| Groups | One fMOST brain per mouse. IT-line brains have unique sample IDs. |
| Support | Pools are animal × line × CCF area × layer. **57 core-IT animals** have a pool of ≥ 5 cells and 35 a pool of ≥ 10, against the 20 required. 2,796 eligible core-IT cells are flagged `Axon_and_dendrite`. |
| Lineage | No brain shared with Peng 2021 or SEU-A1876, so this source is independent of the earlier lineage. |

**Changes that would need approval:**

1. Replace SEU-A1876 with this source as EP09's primary dataset.
2. Make CCF-registered dendrites the primary representation, with soma-centred
   rotation-invariant features. Native-space SWCs appear unreleased, so the
   native-versus-CCF sensitivity would become unavailable.
3. Define layer by registered soma position, with a depth caliper and an
   interior-only sensitivity. No curated layer exists here.
4. Choose a whole-animal development/audit split stratified by line. The
   12/8 minimums are met with room to spare.
5. Build a shared exposure ledger with EP10 and EP11, which also use Gao-lab
   data, before any outcome is opened.
6. Store the archive in the Oak steward area, outside this repository.

**First steps after approval, before any outcome:**

- verify the archive's MD5;
- count dendrite completeness per cell from dendrite nodes only;
- confirm the CSSP vector's Cre-dependence;
- recheck novelty against the dataset's own paper;
- build the ledger, then assign roles.

**Claim limits:**

- mice are male only;
- cells cluster near the injection sites;
- morphology is CCF-registered;
- one lab and one pipeline, so held-out animals give internal grouped
  reproducibility, not independent replication.

## Public online resources — checked October 6, 2026

| Public resource | Available information and use for EP09 |
| --- | --- |
| [Original SEU-A1876 Zenodo release](https://zenodo.org/records/13944322) | The open v1 record lists `Full_morphometry.xlsx`, native `Full_morphology_RAW.zip`, registered `Full_morphology_CCFv3.zip`, separate dendritic/axonal arbor archives and published file sizes/MD5s. Its inventory describes 1,876 full reconstructions. This identifies the exact metadata/native-dendrite/CCF-axon source objects required by EP09 without reading their payloads. |
| [Provider BrainTell genetic table](https://braintell.org/projects/fullmorpho/genetic_info.txt) | The row keyed `16124` contains `Gnb4-IRES2-CreERT2;Ai139-hyg_236174-16124`; other rows retain image/genetic-background strings for documented Cux2, Plxnd1 and Tlx3 lines. This is original-provider metadata for reconciling specimen keys, not an outcome-derived class label. |
| [Primary TeraVR Methods, available through PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6677772/) | The Data preparation section identifies brain `236174` as `Gnb4-IRES2-CreERT2;Ai139`, independently corroborating the genotype. |
| [Liu et al. Supplementary Data 1](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-024-54745-6/MediaObjects/41467_2024_54745_MOESM3_ESM.xlsx), retained public workbook row 81 | Explicitly separates Animal ID `236174` from Image ID `16124`, with fMOST technique, the same Gnb4 genotype and `BIL meta (20231223)` provenance. The [bounded observation](outputs/runs/metadata_feasibility/attempt002/provider_alias236174_observation_job_46842156.json) authenticates the alias without reading neuron outcomes. |

## Source facts recovered on October 16, 2026 (no provider contact)

Rules were fixed before use in
[ASSUMPTIONS.md](outputs/runs/metadata_feasibility/attempt003/ASSUMPTIONS.md).
The outcomes, with evidence and sensitivities, are in the
[attempt003 register](outputs/runs/metadata_feasibility/attempt003/README.md).
Additional public sources read:

| Source | What it settles |
| --- | --- |
| [Liu et al. 2024 Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11599929/) | 140 neurons newly annotated from 19 fMOST brains, plus refined versions of 1,736 neurons released earlier (Peng 2021). Registration to the 25 µm CCFv3 with mBrainAligner. Regions auto-labelled from the deformation. Six structural reconstruction criteria. Axes x = AP, y = DV, z = ML. |
| [Peng et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8494643/), [Supplementary Table 2](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03941-1/MediaObjects/41586_2021_3941_MOESM4_ESM.xlsx) | Names of the 1,741 released cells, registered and manually corrected soma regions, and curated `Cortical_layer`. Ext. Data Fig. 7: layers confirmed by visual inspection, and L4-like cells called from dendrite and local-axon morphology. `Subclass_or_type`, `Notes` and the matrix were never decoded. |
| [Peng 2021 Supplementary Table 1](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03941-1/MediaObjects/41586_2021_3941_MOESM3_ESM.xlsx) | Age, sex, tamoxifen schedule and targeted types per brain. Targeted types agree with Supplementary Data 1 for all 9 overlapping IT-driver brains. |
| [Provider analysis code](https://github.com/SEU-ALLEN-codebase/full_spectrum_sources), commit `22491dd` | Reads cortical layers from a "manually curated" cell-type table. Only code and README files were read. The clone in EP09 scratch also contains projection, cell-type and morphology data, which stay closed. |

**Lineage.** 1,736 SEU-A1876 cells are of Peng 2021 lineage: 1,727 match a
name exactly, and 9 are Peng's `210254_*` cells renamed `15257_*` (animal
210254 = image 15257). The 140 new cells come from exactly 19 brains.

**`isManuallyChecked`.** The flag equals this split exactly: 1 marks the
refined cells of Peng lineage, 0 marks the new cells. The two uses of 1,736
noted below are therefore the same set of cells. The flag is a lineage
covariate, not a per-cell quality score. No other per-cell quality or batch
field is released.

**Layers.** SEU-A1876 recorded layers are Peng 2021 curated layers (510/510
agree).

- 74 blank layers of Peng lineage were recovered from Peng 2021, assuming
  each blank is an omission.
- The 140 new cells have no curated layer anywhere. Coordinate-derived layers
  agree with curated layers at κ 0.41, below the pre-set κ ≥ 0.60, so they
  serve only as unvalidated sensitivities.

**Donor support.** Pools are animal × source × layer × line with ≥ 5 cells.
At most 3 of the 12 IT-driver animals qualify in any scenario: 374712,
382148, and 380470 (the last via recovered layers). The registered 12/8
design and the unadopted nine-animal route are therefore infeasible on this
source.

**RAW → CCF.** The exact per-brain mBrainAligner transform is not released
for any IT-driver brain, so it is skipped. The transform would comprise RPM
markers, an FFD grid and local landmark pairs. The searches covered:

- Zenodo, and Zenodo-wide;
- the Liu 2024 Google Drive mirror;
- both papers' data statements;
- the BIL catalog via its metadata API;
- the mBrainAligner examples, which hold a demo for brain 18458 only.

BIL folder contents could not be listed: BIL storage is offline for a
data-centre move into the week of October 18.

Native orientation is resolved without the field:

- BIL image metadata gives the axes and voxel sizes (0.2–0.35 µm in x/y,
  1 µm in z) for 15 brains.
- In all 11 brains that also have a soma affine fit (median residual
  132 µm), the two sources agree.
- Raw z = AP in every brain. Raw x/y map to DV/ML in a brain-specific order,
  and brain 18453 is mirror-reflected.

### October 6 identity note (still valid)

The public identity/genotype gap for `236174` is now resolved: its provider
animal and image identifiers differ. The original exact-Image-ID join remains
preserved, with this separate observation as correction evidence. This adds no
authenticated L2/3–L5 IT eligibility, missing layers or matched-donor support;
the current cohort-feasibility limits below remain unchanged.

## Historical primary source: SEU-A1876

This section preserves the former source inventory and identity work. Its
primary/native-input labels and unassigned-role statements describe the
superseded SEU design only; they do not authorize SEU payload use in the
adopted CEBSIT study.

| Item | Role in EP09 |
| --- | --- |
| [SEU-A1876](https://zenodo.org/records/13944322) | Primary same-cell morphology collection |
| `Full_morphometry.xlsx` | Cell identities, brain identifiers, soma regions, layers, morphology summaries, and provider QC |
| `Full_morphology_RAW.zip` | Primary native dendritic reconstructions |
| `Full_morphology_CCFv3.zip` | Whole-neuron CCFv3 reconstructions used for axons and coordinate validation |
| `Axonal_arbor_CCFv3.zip` | Candidate source for distal-arbor detections |
| `Dendritic_arbor_CCFv3.zip` | Secondary sensitivity representation; not the primary dendritic input |
| Allen Mouse CCFv3 annotation and structure graph | Source/target assignment, ontology families, and distance geometry |
| [CCF-ME v2](https://zenodo.org/records/13801372) | Optional external representation sensitivity after cell-lineage review |

The release reports 1,876 morphologies across 39 `fMOST Brain ID` values, 92
soma regions, 1,736 manually checked cells, and 308 cells without a provider
`Projection class`. These are source inventory counts, not the EP09 sample.
Eligibility depends on verified biological grouping, paired dendrite–axon
identity, coordinate integrity, observability, and the frozen target rules.

The number 1,736 also appears in a different and scientifically important
lineage statement: Liu et al. 2024 report that 1,736 of the 1,876 morphologies
are refinements of previously released neurons and 140 were newly annotated.
These two uses of 1,736 were once kept apart as possibly different sets. The
October 16 cross-tabulation (above) shows they are the same cells:
`isManuallyChecked` = 1 exactly for the refined cells of Peng lineage. EP09 is
therefore predominantly an adjudicative reanalysis of the Peng-2021 data
lineage, not an independent biological replication.

The official source archives are preserved read-only at
`$OAK/users/zijiao/br_autoresearch_data/steward_acquisition/seu_a1876_zenodo_13944322/archives`.
They remain mixed-role steward inputs, not an episode-visible candidate pack.
Large archives and participant-level derived data remain outside Git.

The October 3 steward metadata check recovered a non-axon provenance lead from
[Liu et al. Supplementary Data 1](https://www.nature.com/articles/s41467-024-54745-6#Sec32):
38 of 39 fMOST brain IDs join exactly to **38 distinct animal IDs** and transgenic
metadata. The immediate shortage is authenticated IT-label support, not a
finding that those 38 brains collapse into fewer animals. Brain 236174 remains
unmatched in that preserved historical exact-Image-ID join; its identity and
genotype are now authenticated separately below. Only cell identity, brain identity,
soma region and cortical layer were decoded from the source workbook; its
projection-class column and morphology/outcome values remained closed.
The explicitly documented cortical L2/3/4/5 IT driver families in this join
(six Cux2, five Plxnd1 and one Tlx3) span **at most 12 authenticated animals**
before layer, donor and quality restrictions. The now-identified Gnb4 animal
does not supply additional authenticated primary-population IT support. Thus
this route cannot supply the required 20 development-plus-audit groups. Other
driver lines cannot be relabelled IT from axonal projection classes.
See the [actual metadata support](outputs/runs/metadata_feasibility/attempt002/genotype_support_summary.json)
and [execution log](outputs/experiment_log.md). An additional authenticated,
non-axon cell-label source or a separately approved scientific downgrade is
needed for further primary-cohort execution. No roles have been assigned.

**Authenticated correction, October 6 Pacific:** the metadata-only check in
job 46842156 returned exactly one matching public-supplement row, row 81:
Animal ID `236174`, Image ID `16124`, fMOST,
`Gnb4-IRES2-CreERT2;Ai139`, provenance `BIL meta (20231223)`, and target
description `Claustral and cortical Car3+ IT`, referencing Peng et al. 2021.
Its provider source locator is
[`mouseID_236174`](https://download.brainimagelibrary.org/7f/53/7f537a62e521a26a/mouseID_236174/).
That animal ID does not already occur in the retained join. The 43-cell source
group's original key `236174` is preserved; the mismatch arose because the
original join searched Image ID alone. This resolves the earlier crosswalk
question from original-provider metadata, without using any EP11 assignment
or output. See the [direct observation](outputs/runs/metadata_feasibility/attempt002/provider_alias236174_observation_job_46842156.json)
and [source note](outputs/paper_plan.md#october-6-public-source-lead--brain-236174-and-provider-key-16124).

The historical join and genotype-summary JSON files are not overwritten.
Any future integrated join must be a separate derived output preserving the
original brain key and these explicit animal/image fields; no general alias
fallback is adopted. This identity correction supplies neither cell-level
release lineage nor the required primary cortical-layer/class or donor
qualification. BrainTell's older 45-neuron gallery listing is not substituted
for the retained 43-cell group. No eligibility or biological role is assigned.

The retained per-animal source/layer counts further limit this route under
the currently authenticated metadata: only animals 374712 and 382148 have
driver-compatible exact source/layer strata with at least five recorded cells,
necessary for four nonself donors. Their ten strata contain 72 recorded cells
before finer non-axon class, calipers, quality and other qualification. The
[complete 12-animal table](outputs/paper_plan.md#october-6-metadata-feasibility-and-unadopted-exploratory-route)
is a direct reading of the existing permitted join, with no new workbook or
payload access. Two is a bound for these recorded keys, not an impossibility
claim after authentic metadata corrections or additional labels. Missing
layers remain unknown rather than verified absence. *(October 16 update:
after the recoveries above, the bound is 3 animals and 85 cells under the
primary rule, and at most 3 animals under any sensitivity.)* Brain 236174's identity
and genotype no longer require a provider answer; finer eligibility remains
unestablished.

The existing policy already permits stopping or downgrading to grouped
development evidence when group support is inadequate. It does not activate
the [proposed nine-animal exploratory route](outputs/paper_plan.md#october-6-metadata-feasibility-and-unadopted-exploratory-route),
which the current metadata does not support.
The current 12-development/8-audit requirement remains binding. No episode
power analysis establishing 20 as a sufficient sample size was identified in
the reviewed contract and feasibility records; the number is a design
requirement, not demonstrated precision. Nine would likewise be a proposed
feasibility floor, not a power guarantee.

## What one eligible neuron must contribute

An eligible cell needs all of the following:

- a stable cell identifier linking its metadata, native dendrite, and CCFv3
  axon;
- a conservative animal, brain, or specimen group identifier;
- source region, layer when available, soma coordinates, and acquisition or
  reconstruction batch;
- enough native dendrite to compute the frozen morphology blocks;
- enough axonal coverage to decide at least some target outcomes;
- acquisition and dendrite-quality measures defined independently of axonal
  geometry;
- a cortical IT label established by genetic, experimental, or source metadata
  whose lineage does not use that cell's axonal target vector; and
- original-release versus newly annotated lineage.

The provider `Projection class` field is axon-derived. It cannot define the
primary IT cohort, matching strata, predictors, or dendritic feature selection.
If no outcome-independent IT label exists with adequate group, target, and
donor support, the headline primary is not executable.

Axon-derived coverage, completeness, total length, or projection breadth may
determine observability or enter a stratified sensitivity analysis. They are not
context predictors merely because they were computed without reading the
specific target label; they can disclose other parts of the axonal outcome.

The dendritic and axonal files are parsed independently and joined only through
the verified identity ledger. Duplicate cells, derivative copies, truncated
trees, disconnected components, invalid parent links, unit or axis ambiguity,
and mismatched soma coordinates must be resolved before role assignment.

`Projection class` and every label computed from the axon are outcomes or
descriptions of outcomes. They cannot be predictors, split variables,
imputation aids, or ingredients in dendritic feature selection.

## Biological groups and the shared role ledger

Cells from the same biological preparation are correlated. EP09 therefore
splits and evaluates whole biological groups, never random neurons. The grouping
priority is:

1. animal, when a verified animal identifier exists;
2. brain, when it is the most conservative verified unit;
3. specimen or preparation, only when neither animal nor brain provenance can
   be recovered.

The primary design requires at least 12 development groups and 8 sealed audit
groups, with adequate source-population support on both sides. A provider field
must not be reinterpreted as an animal merely to meet those counts. If the
provenance or counts fail, the study becomes grouped development evidence or
stops; it does not fall back to a neuron-level split.

EP09, EP10, and EP11 may expose the same cells and axonal outcomes. They need
one shared ledger containing:

- stable cell and biological-group identifiers;
- duplicate and publication-lineage notes;
- whole-group development or audit roles;
- every prior exposure to outcomes, target prevalence, derived features, or
  fitted artifacts; and
- the date on which each episode froze its target and analysis contract.

Opening an audit group for one episode exposes it for all three. All three
episodes must therefore freeze their role manifests, outcome rules, target
vocabularies, and primary analyses before any shared audit outcome is opened.

## Development and audit roles

Role assignment uses identity, provenance, coverage, and technical feasibility
only. It cannot use dendrite–axon associations, target prevalence, model scores,
or visually interesting cells.

| Role | Minimum groups | Assigned CEBSIT groups | Permitted use |
| --- | ---: | ---: | --- |
| Development | 12 | 27 | Build the observation pipeline, determine training-fold target support, tune models, run falsification analyses, and freeze predictions within the adopted staged access. |
| Audit | 8 | 16 | One evaluation of the locked primary and explanatory predictions after the shared lock and authorized extraction. |

Audit axonal detections, target prevalence, and outcome-derived feature caches
remain in a separately readable location until the full EP09/10/11 lock. The
search process may receive audit anatomy and outcome-blind coverage only when a
frozen recipe explicitly needs them; it may not receive detection labels or
summaries from which those labels can be inferred.

This split supports internal grouped reproducibility within CEBSIT, conditional
on the identity assumption above. CEBSIT is separate from the Peng/SEU lineage,
but its own development and audit groups remain one laboratory/reconstruction
resource. A held-out CEBSIT subset is not external replication. The earlier
option to use newly annotated SEU cells does not assign them a current role.

## Constructing a target observation

The unit recorded by the evaluator is a cell–target pair. A frozen axonal-arbor
rule assigns exactly one of three states:

| State | Meaning |
| --- | --- |
| `detected` | Qualifying distal axon is present inside the target under the frozen rule |
| `not_detected` | The axonal reconstruction and target are observable, but no qualifying arbor is present |
| `uncertain` | Coverage, clipping, registration, or reconstruction quality does not support either conclusion |

`uncertain` remains missing. It is never converted to `not_detected`. A valid
negative therefore requires positive evidence that the cell and target were
observable, not merely an absent row in an arbor table.

Before outcome construction, validate:

- tree semantics and parent–child topology;
- coordinate units, axis order, orientation, and soma registration;
- atlas version and transform direction;
- clipping, incomplete terminal capture, disconnected axonal branches, and
  provider-arbor linkage; and
- whether a target is far enough from the soma/source region to count as
  distal under one frozen definition.

The primary arbor threshold is fixed before scoring. A small set of
non-selectable threshold sensitivities may show how measurement definition
affects the result, but none may replace the primary result after audit.

## Exact-target vocabulary and secondary target families

Primary outcomes are exact, non-overlapping nodes from one pinned Allen CCFv3
structure graph. The vocabulary is chosen from anatomy and observation support,
not from which targets yield a strong dendritic association. Broad families
are prespecified explanatory summaries and hierarchy falsifiers; they cannot
replace the exact-target primary.

Two forms of support are distinct:

- **Outcome-blind observation support:** enough cells have adequate coverage to
  determine a state for the target.
- **Development prevalence support:** enough independent development groups
  contain detections and nondetections for fitting and scoring.

Before detections are read, freeze an integer `n_coverage >= 3` from
outcome-blind cell counts and development-only precision simulations. A target
can enter the audit score set only if every development and audit group has at
least `n_coverage` observable cells for that target. Audit observability is
allowed for this intersection; audit detection states are not. A later drop
below the frozen minimum is a technical invalidation, not permission to remove
the target or change its weight.

In each development outer fold, a target must have detections in at least three
independent training groups and valid nondetections in at least three. Every
candidate in that fold is scored on the same fold-specific target set, and
outer folds receive equal weight. The held-out group never supplies prevalence
eligibility. The final audit set requires at least four positive and four
negative full-development groups, so holding out any one development group
leaves the three-per-state training minimum.

The explanatory target families use one outcome-independent, non-overlapping
anatomical cut of the Allen ontology rather than assuming that one numeric tree
level is balanced. The final set needs at least three families and at least
three scored targets in every family. Every eligible target belongs to exactly
one family. Family definitions cannot be redrawn around an attractive result.

## Predictor blocks

### Context-only predictors

The M0 model may use prospectively available variables that do not describe
the distal axon:

- source region and cortical layer;
- soma location and prespecified spatial transforms;
- acquisition or reconstruction batch;
- acquisition and dendrite-reconstruction quality whose lineage is independent
  of axonal geometry; and
- independently defined labels whose lineage contains no axonal outcome.

### Registered dendritic predictors

M1 adds soma-centred, rotation-invariant summaries from the adopted CEBSIT
CCFv3 dendrites. Candidate blocks include:

- branch topology and branch order;
- soma-centered radial or Sholl-like profiles;
- path length, extent, tortuosity, and branching geometry;
- orientation and hemispheric symmetry as a prespecified secondary extension; and
- prespecified persistent-homology summaries.

Physical-radius shells and branch-order blocks for the explanatory analysis
are frozen after unit, sampling, and coordinate validation but without viewing
target detections. The source amendment makes registered CCFv3 morphology the
primary representation; native morphology is unavailable. Registration and
interior-only sensitivities therefore matter for the interpretation.

An external pretrained representation is admissible only if its training data,
cell identities, and publication lineage are documented and do not expose the
audit outcomes. Otherwise it is excluded.

## Matched control data

The evaluator must be able to generate the following from the same eligible
cells and grouped folds:

- 20 frozen nuisance draws that independently derange standardized residual
  rows for each dendritic block after outcome-free, cross-fitted conditional
  location-and-scale modeling from M0 context;
- coherent whole-dendrite residual swaps within biological group, source,
  layer, independently defined non-axon IT class, frozen soma-position and
  quality calipers, and optionally a development-frozen dendrite-only
  neighborhood;
- rotation-invariant and soma-centered versions of orientation-sensitive
  features;
- size-only and QC-only feature blocks; and
- 20 matched shuffled target hierarchies preserving family size within frozen
  development-prevalence, observation-rate, and CCFv3 source-centroid distance
  bins.

Nuisance blocks and matched-donor swaps are different experiments. Nuisance
generation uses a different no-fixed-point donor assignment for each dendritic
block, preserving within-block covariance while destroying the cross-block
coherence of a tree. The primary swap transfers the entire standardized
dendritic residual vector from one tightly matched donor, preserving coherent
cross-block morphology while breaking the recipient–dendrite pairing.

Both routes fit their outcome-free conditional location, scale, whitening, and
training-only imputation inside the training partition; transform the donor
residual to the recipient’s context; and restore the recipient’s missingness
mask. Training/calibration and held-out groups are generated separately. No row
moves across partitions and no axonal outcome enters the generator.

Before launch, development morphology must support conditional overlap and
joint support across source, layer, soma position, batch, and axon-independent
QC, including mean, covariance, heteroscedasticity, multimodality, and missingness
diagnostics. Every nuisance group and every frozen source × layer × class
matched-swap pool needs the prespecified donor count. Failure makes the headline
primary unavailable rather than triggering looser matching or silent cell
removal after outcomes are known.

Freeze every hierarchy distance/prevalence/observation caliper and assignment
before audit. If 20 valid hierarchy shuffles cannot be constructed, that
explanatory claim is unresolved; its constraints are not relaxed.

The nuisance controls receive the one development-selected M1 pipeline and are
refit from scratch on their false feature blocks. They do not receive separate
adaptive searches. The correct-versus-donor primary is different: the M1
outcome model, preprocessing, and calibrator are fitted on correct development
pairings and held fixed. Donor residuals are substituted only when generating
held-out predictions, with the recipient's context and missingness preserved.

The scale-block analysis similarly fits one all-block E1 on correct
development pairings and holds it fixed during held-out evaluation. It
replaces one block at a time using the same frozen coherent matched-donor bank
as the primary, averaging across the frozen donor plans rather than refitting
on swapped training data. Within a plan, donor identity remains the same
across blocks. These are matched conditional substitutions, not unrestricted
feature permutations, and they estimate predictive dependence rather than
causal importance.

All paired comparisons use the same cells, targets, folds, preprocessing,
calibrator eligibility, and score weights. Tuning is identical where the
scientific comparison calls for refitting; fixed-pipeline controls do not claim
unperformed search exposure.

## What remains sealed

Before the audit, the search workspace may contain development data and the
frozen audit input required to generate predictions. It may not contain:

- audit target detections or prevalence;
- audit-derived outcome labels, feature selections, or fitted target models;
- reports, caches, figures, or logs that reveal audit performance;
- exact-target or target-family refinements based on audit outcomes; or
- artifacts from EP10 or EP11 that indirectly disclose the same outcomes.

A trusted evaluator joins the frozen predictions to the sealed outcomes,
computes group-level scores and explanatory summaries, and returns the locked
report once. Audit outcomes cannot be used to refit, recalibrate, select a
feature block, change a margin, or choose a more favorable target set.

## Prior work and the novelty boundary

The source collection and related literature already support broad claims about
whole-neuron morphological diversity, projection motifs, and relationships
between dendritic form and projection identity:

- [Peng et al. (2021)](https://doi.org/10.1038/s41586-021-03941-1)
  reported that within cortical IT populations, individual target-subset
  selection appeared unrelated to soma depth or dendritic morphology.
- [Liu et al. (2024)](https://doi.org/10.1038/s41467-024-54745-6)
  analyze neuronal diversity and stereotypy at multiple scales using
  SEU-A1876 and document its relationship to previously released cells.
- [Gao et al. (2023)](https://doi.org/10.1038/s41593-023-01339-y) relate
  dendritic and axonal organization in mouse prefrontal cortex while also
  showing that simple one-to-one correspondences do not always hold.
- [Liu et al. (2026; online 2025)](https://doi.org/10.1038/s41593-025-02119-6)
  build an atlas from dendritic microenvironments and examine correspondence
  with long-range projection organization.
- [Sorensen et al. (2026)](https://doi.org/10.1038/s41586-026-10424-8)
  directly connect local morphology, cortical location, multimodal identity,
  and specific projection targets.

EP09 cannot use “dendrites are associated with axons” as its novelty claim.
The proposed contribution is the held-out comparison of the correct dendritic
residual against coherent tightly matched donor residuals at exact-target
resolution, followed by a scale × exact-target replication test. M1 beating M0
is a required benchmark, not the discovery.

## Remaining dependencies under the adopted source

CEBSIT acquisition, source/line qualification, shared roles, compartment-code
resolution, measurement v1.2 and stage A are recorded in the links above.
Do not repeat the historical SEU provisioning sequence. Before the relevant
later access/scoring stages, the remaining requirements are:

- resolve the scoped donor-supported-population/matching amendment and
  confirmatory gates without inferring their adoption from implementation work;
- complete the prospective model/donor specifications and outcome-free donor
  qualification, preserving actual supported-group counts and the full primary
  exact-target scope;
- perform the adopted technical stage and development measurements only under
  their staged access rules; audit dendrites and axons remain closed until the
  shared lock and authorized stage E;
- freeze qualified nuisance controls, E0/E1, scientific margins, exact-target
  template/sign rules, calibrated inference and nested development checks;
- retain the 20-shuffle secondary hierarchy requirement and its own availability
  rule without treating secondary failure as primary failure;
- demonstrate measured workload feasibility using finite resource requests
  under standing authority, retaining historical consumption.

The [attempt006 navigation](outputs/runs/measurement_and_roles/attempt006/README.md#october-20-prospective-implementation-defaults-under-standing-authority)
separates adopted implementation defaults from pending scientific choices.

## Storage

Keep source archives immutable and outside the repository. Temporary archive
expansions and feature matrices belong under
`$SCRATCH/br_autoresearch/episode09_local_global_projection/`. Durable episode
outputs are the source inventory, cell/group ledger, parsers and validation
results, frozen role and target manifests, prediction files, evaluator report,
and paper figures. None should contain protected audit outcomes before the
one-time evaluation.
