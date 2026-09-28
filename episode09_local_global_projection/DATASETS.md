# EP09 data: native dendrites and distal axonal arbors from the same neurons

EP09 needs whole-neuron reconstructions in which the dendritic tree and distal
axon belong to the same identified cell. [GOAL.md](GOAL.md) explains the
scientific question; this file explains which measurements support it, how a
valid target observation is constructed, and what must be settled before the
experiment can launch.

## Primary source

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

The data are not yet provisioned in an episode-local read-only location. Large
archives and participant-level derived data remain outside Git.

## What one eligible neuron must contribute

An eligible cell needs all of the following:

- a stable cell identifier linking its metadata, native dendrite, and CCFv3
  axon;
- a conservative animal, brain, or specimen group identifier;
- source region, layer when available, soma coordinates, and acquisition or
  reconstruction batch;
- enough native dendrite to compute the frozen morphology blocks;
- enough axonal coverage to decide at least some target outcomes; and
- acquisition and dendrite-quality measures defined independently of axonal
  geometry.

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

| Role | Minimum groups | Permitted use |
| --- | ---: | --- |
| Development | 12 | Build the observation pipeline, determine training-fold target support, tune models, run falsification analyses, and freeze predictions |
| Audit | 8 | One evaluation of the locked primary and explanatory predictions in unseen biological groups |

Audit axonal detections, target prevalence, and outcome-derived feature caches
remain in a separately readable location until the full EP09/10/11 lock. The
search process may receive audit anatomy and outcome-blind coverage only when a
frozen recipe explicitly needs them; it may not receive detection labels or
summaries from which those labels can be inferred.

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

## Target vocabulary and target families

Targets must be non-overlapping nodes from one pinned Allen CCFv3 structure
graph. The vocabulary is chosen from anatomy and observation support, not from
which targets yield a strong dendritic association.

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

### Native dendritic predictors

M1 adds summaries computed from `Full_morphology_RAW.zip`. Candidate blocks
include:

- branch topology and branch order;
- soma-centered radial or Sholl-like profiles;
- path length, extent, tortuosity, and branching geometry;
- orientation and hemispheric symmetry; and
- prespecified persistent-homology summaries.

Physical-radius shells and branch-order blocks for the explanatory analysis
are frozen after unit, sampling, and coordinate validation but without viewing
target detections. Native coordinates are primary because the question concerns
the measured dendritic tree rather than an atlas-warped version. CCFv3
dendrites are a locked sensitivity only.

An external pretrained representation is admissible only if its training data,
cell identities, and publication lineage are documented and do not expose the
audit outcomes. Otherwise it is excluded.

## Matched control data

The evaluator must be able to generate the following from the same eligible
cells and grouped folds:

- 20 frozen nuisance draws that independently derange standardized residual
  rows for each dendritic block after outcome-free, cross-fitted conditional
  location-and-scale modeling from M0 context;
- intact whole-dendrite residual swaps within a development-frozen,
  dendrite-only morphology class and biological group;
- rotation-invariant and soma-centered versions of orientation-sensitive
  features;
- size-only and QC-only feature blocks; and
- 20 matched shuffled target hierarchies preserving family size within frozen
  development-prevalence, observation-rate, and CCFv3 source-centroid distance
  bins.

Nuisance blocks and cell-identity swaps are different experiments. Nuisance
generation uses a different no-fixed-point donor assignment for each dendritic
block, preserving within-block covariance while destroying the cross-block
coherence of an intact tree. The identity test transfers the entire standardized
dendritic residual vector from a donor in the same frozen morphology class,
preserving coherent morphology while breaking cell identity.

Both routes fit their outcome-free conditional location, scale, whitening, and
training-only imputation inside the training partition; transform the donor
residual to the recipient’s context; and restore the recipient’s missingness
mask. Training/calibration and held-out groups are generated separately. No row
moves across partitions and no axonal outcome enters the generator.

Before launch, development morphology must support conditional overlap and
joint support across source, layer, soma position, batch, and axon-independent
QC, including mean, covariance, heteroscedasticity, multimodality, and missingness
diagnostics. Every nuisance group and every group-by-morphology-class swap cell
needs at least four donors. Failure makes that control unavailable rather than
silently dropping cells after outcomes are known.

Freeze every hierarchy distance/prevalence/observation caliper and assignment
before audit. If 20 valid hierarchy shuffles cannot be constructed, that
explanatory claim is unresolved; its constraints are not relaxed.

The nuisance controls receive the one development-selected M1 pipeline and are
refit from scratch on their false feature blocks. They do not receive separate
adaptive searches. The scale-block analysis instead uses one fixed all-block
E1 model, learns an outcome-free conditional replacement law inside each
development training fold, and averages 10 complete replacement refits in which
the block is replaced in both training and held-out data. Post-fit test-only
permutation is not used.

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
- target-family refinements based on audit outcomes; or
- artifacts from EP10 or EP11 that indirectly disclose the same outcomes.

A trusted evaluator joins the frozen predictions to the sealed outcomes,
computes group-level scores and explanatory summaries, and returns the locked
report once. Audit outcomes cannot be used to refit, recalibrate, select a
feature block, change a margin, or choose a more favorable target set.

## Prior work and the novelty boundary

The source collection and related literature already support broad claims about
whole-neuron morphological diversity, projection motifs, and relationships
between dendritic form and projection identity:

- [Peng et al. (2024)](https://doi.org/10.1038/s41467-024-54745-6) analyze
  neuronal diversity and stereotypy at multiple scales using SEU-A1876.
- [Gao et al. (2023)](https://doi.org/10.1038/s41593-023-01339-y) relate
  dendritic and axonal organization in mouse prefrontal cortex while also
  showing that simple one-to-one correspondences do not always hold.
- [Muñoz-Castañeda et al. (2025)](https://doi.org/10.1038/s41593-025-02119-6)
  build an atlas from dendritic microenvironments and examine correspondence
  with long-range projection organization.

EP09 cannot use “dendrites are associated with axons” as its novelty claim.
The proposed contribution is the held-out, same-cell decomposition of context,
native dendritic information, dendritic scale, and independently defined target
families. A focused novelty review must verify that this exact prediction has
not already been established in comparable data.

## Launch blockers

The following items must be resolved before candidate scoring:

- provision the SEU-A1876 archives and pin the exact Allen CCFv3 annotation and
  structure graph in read-only storage;
- verify animal/brain/specimen provenance and build the shared EP09/10/11 role
  and exposure ledger;
- authenticate cell joins, native-to-CCF coordinate relationships, and the
  dendrite/axon parsing route;
- freeze the distal-arbor, observability, valid-nondetection, target-vocabulary,
  and outcome-independent target/source family cuts;
- verify the minimum development and audit group counts after all eligibility
  rules;
- freeze `n_coverage`, positive/negative group support, family support,
  dendritic blocks, conditional-generator diagnostics, independent-block
  nuisance seeds, dendrite-only morphology classes, intact-swap seeds, and all
  20 shuffled hierarchies;
- freeze E0/E1, the exact group and family scores, `delta_morph`, `delta_cell`,
  `eta_matrix`, `rho_matrix`, required sign entries, source-family scope, and
  calibrator eligibility;
- demonstrate with separate, prespecified calibration and validation simulations
  that the planned lower-tail, upper-tail, cell-equivalence, nuisance, and
  hierarchy critical values meet their coverage requirements;
- confirm that all fixed-pipeline nuisance, replacement, and hierarchy refits
  fit inside the resource ceiling; and
- complete the focused novelty review.

Optional CCF-ME absence does not block the primary analysis. Failure of a
primary item does.

## Storage

Keep source archives immutable and outside the repository. Temporary archive
expansions and feature matrices belong under
`$SCRATCH/br_autoresearch/episode09_local_global_projection/`. Durable episode
outputs are the source inventory, cell/group ledger, parsers and validation
results, frozen role and target manifests, prediction files, evaluator report,
and paper figures. None should contain protected audit outcomes before the
one-time evaluation.
