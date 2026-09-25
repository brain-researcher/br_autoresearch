# Dataset contract — EP10

EP10 asks whether specific single-neuron target combinations recur across
animals and soma positions, what measured alternatives explain them, and how
complete axon trees implement them. Resource roles will be chosen during the
run's first data pass; no biological analysis has started.

The [paper plan](outputs/paper_plan.md) maps these data requirements to the
claims, figures, failure interpretations, and external evidence needed for a
manuscript. It does not assign a resource or expand access.

## Candidate resources

| Resource | Potential role | Current limits |
| --- | --- | --- |
| **SEU full-morphology release** | Measurement development and, if biological-group support is adequate, grouped discovery or internal evaluation | Metadata, full CCFv3 morphologies, and derived arbor files were acquired into unopened mixed-role steward quarantine. Animal identity, source support, observation quality, outcome-derived metadata removal, and release overlap remain unresolved. |
| **Gao cortical projectome, 18,621 SWCs** | Main within-source and across-position candidate if its metadata support independent biological groups and complete target observation | Not acquired locally. Animal mapping, source coverage, reconstruction completeness, prior exposure, and overlap with other releases must be established before assigning a role. A large cell count does not establish animal replication. |
| **MouseLight** | Candidate external validation from a different acquisition and reconstruction workflow | Comparable source and target coverage, animal identity, sampling differences, reconstruction completeness, and duplicate lineage are unverified. It cannot be called external replication until those checks pass. |
| **Projection-TAGs** | Targeted test of whether independently measured population composition explains a frozen co-projection relationship | Raw-data access and usable accession details remain pending. Target-panel overlap, MOp or SSp support, independent labels, animal replication, and the assay's detection model require separate verification. Tag non-detection is not equivalent to morphological target absence. |

[Yuan et al.'s 2024 axonal BARseq study](https://www.nature.com/articles/s41467-024-52756-x)
is a prior-work and measurement benchmark, not an assigned EP10 validation
resource. It mapped more than 8,000 auditory-cortex neurons in one male mouse
and already related co-target status to laminar distribution in a shared
cortical target. Its barcode rolonies sample projection distribution but do not
provide a continuous full tree or synaptic partners.

Allen CCFv3 remains the intended common anatomical frame. Its orientation,
voxel axes, label hierarchy, laterality, and compatibility with each resource
must be verified before regions are harmonized.

## Source-population strategy

MOp is the priority feasibility candidate because it can connect the analysis
to existing whole-morphology and multi-target studies. This is not a frozen
source choice. MOp is eligible only if an outcome-blind metadata review
establishes enough independent animals, overlapping soma positions,
independent labels where needed, and reliable target observation.

MOs and SSp are later candidates for testing applicability boundaries. They
must not be pooled with MOp merely to increase cell or animal counts. Any
multi-source analysis requires a stated pooled estimand, adequate support in
each source, and an explicit treatment of source heterogeneity.

Choose the primary source using biological rationale, metadata, independence,
coverage, and measurement quality. Do not scan target-pair effects, model
scores, or visually striking morphologies to choose it.

## First data pass

Before assigning analytical roles, produce a source-by-group support table
with one row for every observed or expected combination:

| Source | Verified animal/specimen | Soma-position stratum | Independent label | Cells available | Observation status | Proposed role | Exclusion reason |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| pending | pending | pending | pending | pending | pending | pending | dataset build not completed |

The first pass should establish:

- cell-to-metadata and morphology joins;
- the most conservative defensible animal, brain, or specimen identity;
- coverage of prespecified soma-position strata within animals;
- provenance and availability of layer, driver-line, molecular, or other
  labels that are independent of the axonal outcomes;
- acquisition batch, reconstruction quality, and usable-cell counts; and
- whether source, position, label, animal, and batch are too confounded to
  distinguish the proposed explanations.

Before roles are assigned, do not inspect exact A+B/A+C counts. Whether A, B,
and C are prespecified or selected by the target-set controller, assess exact
eligibility only in development groups after the split. Keep final-group
outcomes unopened until evaluation. At that point, tabulate B/C exclusivity,
`K`, other targets, qualifying arborization in A, and common-position support.
If final groups later contain too few qualifying events, the endpoint is
inconclusive; do not redefine the combination or retroactively change
eligibility.

The biological replicate is the verified animal. If animal identity remains
unresolved, use the most conservative verified specimen grouping and limit the
claim to cross-group evidence. More neurons from one group do not replace
biological replication.

## Avoiding duplicated samples

Use stable identifiers and basic morphology/metadata matching to keep the same
cell or biological group from crossing the development/final split. If EP09,
EP10, or EP11 use the same cells, treat those results as correlated evidence,
not independent replications. A different repository or paper does not by
itself establish a new sample.

## Assigning discovery and validation roles

Assign roles from metadata, biological independence, source and position
coverage, label availability, and measurement comparability. Never assign a
resource or animal according to the size, direction, significance, or visual
appeal of a candidate co-projection effect.

The current contract requires at least 12 whole biological groups for
development and at least 8 for locked final evaluation. Role assignment occurs
before candidate association outcomes or outcome-derived support are
inspected. Additional groups may be required by the coverage and precision
analysis.

A held-out subset of one public release provides internal cross-animal
evaluation. External replication requires an independent resource with a
comparable frozen source, target definition, and estimand. Combined development
must preserve a genuinely unused validation source; data used to choose the
rule cannot later be relabeled as independent validation. With no defensible
final role, use grouped cross-validation for exploratory work only.

## Complete target-set observation

For every eligible neuron `i`, freeze:

- `V_i`, the non-overlapping candidate targets that could have been scored;
- `S_i`, the complete set of qualifying detected targets within `V_i`;
- `K_i = |S_i|`, the detected target count supplied equally to all models; and
- `Omega_i`, every size-`K_i` subset of `V_i` scored by each model.

The primary question is conditional on `K_i`; it does not explain why target
count varies. Every model must score the same neurons and target sets with
exact normalization.

Before testing the hypothesis, freeze the target resolution, non-overlapping
vocabulary, laterality convention, atlas-boundary rule, and operational rule
separating terminal arborization from a passing axon. Full reconstructions are
the primary source for target calls. Provider-derived arbors may be used only
after their agreement and failure modes have been checked against full trees.
Putative boutons are not direct observations of synaptic connectivity.

For each cell-target pair, distinguish:

1. a qualifying detected target;
2. a verified non-detection under the frozen observation rule; and
3. unknown because reconstruction or atlas assignment is incomplete or
   unreliable.

Unknown is never recoded as zero. If every candidate target cannot receive a
defensible detection or non-detection status, stop the complete-set endpoint.
A positive-only or missing-data analysis needs a different scientific question
and observation model.

Axon-derived `Projection class` cannot define the source, predictors, split,
candidate targets, imputation, or validation labels. Any label used to test a
population-mixture explanation needs provenance independent of the target
outcomes.

## Observation decisions

Retain a reviewable table of accepted targets, passing fibers, boundary cases,
unknowns, and exclusions. Include the full-tree evidence, agreement with
provider arbors, clipping or coordinate failures, repeatability for ambiguous
cases, and the coverage lost under each rule. Target granularity must balance
biological meaning, observation reliability, recurring-pair support, and
exact-normalization cost; it cannot change after final outcomes are seen.

For any within-target endpoint, additionally verify registration and the
target-intrinsic coordinate inside A, terminal-tree completeness and clipping,
stable boundary assignment, and repeatability of normalized terminal profiles.
Because both A membership and the profile in A come from the same
reconstruction, check whether reconstruction quality changes inclusion or the
apparent profile differently across A+B and A+C.

## Cross-resource comparison

Before transfer, document source and soma-position overlap; non-overlapping
target and laterality mappings; reconstruction, sampling, and detection
differences; biological-group support and lineage; permissible recalibration; and how
assay-specific zero and unknown states alter the estimand. Comparability failure
narrows or prevents an external claim and does not license post hoc remapping.
Projection-TAGs can test mixture only for a prespecified overlapping relation
with an explicit assay model; it does not enter the morphology likelihood.

## Full-tree anatomical characterization

For a small pair list frozen in development, quantify how full axon trees
implement each supported combination. Freeze measures of:

- the divergence point of the paths reaching the two targets;
- normalized shared path length before divergence;
- distinct collaterals and terminal branches; and
- continuous terminal trees split into two targets by an atlas boundary.

### Shared-target terminal organization

For an eligible candidate, compare neurons with qualifying arborization in a
shared target A under frozen A+B and A+C definitions. Before final evaluation,
freeze whether B and C are mutually exclusive, the treatment of other targets
and `K`, the coordinate within A, one primary terminal-profile metric, the
predicted contrast, common-support rule, and multiplicity procedure.

The primary profile should describe normalized qualifying terminal-arbor length
within A. Report total qualifying arbor length in A separately so that a change
in distribution is not confused with a change in amount. Prespecified
supporting summaries may include terminal-branch or reconstruction-endpoint
density, centroid, spread, and focality. Reconstruction endpoints are quality
indicators and anatomical summaries, not observed synapses.

Display the pooled A profile beside the A+B and A+C profiles and freeze one
summary of the heterogeneity hidden by pooling; do not infer information loss
from visual separation alone.

Use the existing whole-group development and final allocation; do not create a
neuron-random split or new roles for this endpoint. Compare biological-group-
level effects on common support—the animal when verified, otherwise the most
conservative specimen—under a frozen adjustment or matching rule for soma
position, source layer or independent labels, `K` and other targets, total
morphology, target observability, registration, clipping, and reconstruction
quality. Treat entry route into A as a prespecified alternative or mediator,
not an automatic matching variable.

Use reference pairs matched on target distance, soma position, source label,
prevalence, `K` and other co-targets, total axon extent, registration and
clipping, and observation quality. Report representative trees with
group-level distributions rather than only favorable examples. A shared root
or common ancestor branch is universal, not a finding.

### Leakage boundary

Predictive geometry must come from an independent atlas, external reference, or
development-frozen rule and be computable for every candidate set without the
test neuron's realized targets.

Actual routes, branch points, shared paths, collaterals, and terminal trees are
outcome characterization only. They cannot be predictive geometry, selection
or eligibility inputs, imputation variables, or reasons to revise vocabulary.

With B/C context labels blinded, development profiles inside A may establish
technical repeatability and choose the most reliable option from a prespecified
set of coordinates or profile summaries. They cannot select A, B, or C, define
B/C exclusivity, or choose a predicted direction. The A+B versus A+C contrast
cannot choose the coordinate or metric, revise target rules, enter M0, M1, or
M2, or serve as an independent label. Target identity and terminal anatomy from
the same reconstruction are complementary outcomes, not independent
replications.

Adult morphology describes implementation, not developmental order, energetic
cost, functional coordination, or a causal wiring mechanism.

The morphology resources can resolve arbor topography and branching. They do
not identify postsynaptic or input partners, synapse number or strength, or the
cell-type-specific synaptic connectivity sought by EP12.

## What the first run decides

EP10 may start directly from an explicit scientist task. Its first stage asks
whether one source supports the 12/8 group split, a repeatable
complete-target rule, overlapping target and position strata, any labels used
for a mixture analysis, feasible exact normalization, and a defensible anatomy
analysis. External resources are not needed for that stage; if a genuinely
comparable independent resource is available, its frozen transfer test is
required for a Route A claim.

Outcome modeling begins only after those scientific choices are made without
looking at candidate effects.

Insufficient development A+B/A+C support makes the shared-target endpoint
ineligible and forbids that claim. Insufficient final support makes an activated
endpoint inconclusive. Neither outcome changes the preserved target-set primary
endpoint or justifies pooling incomparable groups.

If these conditions fail, change the data strategy, narrow the claim, or stop.
Do not pool incomparable areas, assays, specimens, or releases to create
apparent support.
