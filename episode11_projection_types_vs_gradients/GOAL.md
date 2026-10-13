# Do nearby somatosensory neurons use reusable output patterns?

## Adopted local closeout — October 6, 2026

The scientist adopted closure of the original cross-animal primary as
**unresolved/incomplete** at its expired execution and qualification bound.
This is not evidence for continuous organization, absence of reusable
patterns, or an informative biological null. Prior evidence and charges are
retained; real projection outcomes remain closed. No successor or renewed
window is selected. The optional descriptive design remains unadopted.
See the [adoption record](outputs/experiment_log.md#2026-10-06--population_h_optimizer_recovery--scientist-adopted-primary-closeout).

Two neurons can sit at similar positions in trunk somatosensory cortex
(`SSp-tr`) yet distribute their axons differently across downstream regions.
That diversity is already known. The open question is whether a small set of
allocation patterns adds useful, transferable information beyond a flexible
continuous description of position, depth, and nearby-cell variation.

The paper must resolve a named anatomical statement, not announce another
cluster catalogue: **at comparable source positions, does a development-defined
target-allocation pattern recur in held-out replicate units, or does a
qualified continuum predict the allocation distribution just as well?**
The intended replicate is an independently verified animal. Current metadata
establish only brain/sample units, so cross-animal transfer is not yet a
property of this cohort.

![EP11: nearby neurons, allocation descriptions and group transfer](outputs/ep11_conceptual_question-v2.png)

*Design schematic, not observations. Target sets, colors, allocation strips
and distributions are illustrative; no contrast, pattern or winner is
selected. Allocation means regional axon centerline length, including passing
grey-matter axon, not terminal mass or synapses. The group icons do not certify
independent animals. [Imagegen prompt](outputs/ep11_conceptual_question-v2-prompt.md).*

## The smallest informative paper result

The biological deliverable is one principal, development-selected allocation
rule on the existing fixed target vocabulary, with a named source, targets,
laterality, spatial domain, effect scale, and uncertainty. For example, a
contrast between the relative axon length assigned to two named target sets
could change smoothly with depth, or show recurring distributional patterns
among spatially overlapping neurons. The actual target sets and direction are
selected only in development and frozen before audit; this example is not a
preregistered target choice or evidence of exclusive pathways.

The operational primary remains `H_ref - C_ref` on the same 56-coordinate
axon-allocation composition. Its purpose is to distinguish the usefulness of
two descriptions, not prove that neurons have intrinsically discrete types.
Groups and continuous latent coordinates are integrated out when scoring new
neurons. An outcome-derived posterior label is not an independently observed
predictor and cannot establish prospective identification of an individual
neuron's group.

A positive group comparison needs both a meaningful held-out gain and a
reproduced named allocation rule. A continuous answer needs absolute adequacy,
sensitivity to meaningful group effects, and an upper bound on added group
value. A wide interval is unresolved, not evidence for continuity. Regional
axon length is not synapse count, terminal-arbor mass, or functional output.

## What this resource can support now

The prospective primary cohort has 206 axon/dendrite-distinguished cells in
36 distinct fMOST/provider-sample pairs. Those are not 206 independent
replicates, and the 36 pairs lack provider-certified animal identities. Only
12 development and eight audit units are assigned roles in the intended v2
design; the remaining 16 are reserves, not additional audit observations.
No role assignment is locked and no real regional projection outcome is open.

The existing estimator has not passed production recovery or precision. A
population-`C` pilot pass addresses one optimizer diagnostic, not that
limitation. The current resource therefore supports an anatomical design and
outcome-blind feasibility work, **not yet an adjudication of output patterns**.
Improving the prose does not repair power.

Keep two questions separate: can the evaluator recover the prespecified
effect, and does the available replicate layout yield a sufficiently precise
answer? Repairing the former cannot create independent animals or overcome
the latter. If the fixed design remains inadequate, report that result and
prospectively choose a narrower estimand or additional data before outcomes;
do not quietly relax the support or effect rules. A focused named-target
analysis would be a separately specified design, not a fallback significance
test on the current primary.

The compact paper argument and prior-work boundaries are in
[paper_plan.md](outputs/paper_plan.md). The execution history and unchanged v2
contract are retained below; this writing revision launches no experiment and
does not activate a new data-access stage.

## Recorded execution status — not a biological result

**Current evidence boundary, reconciled October 6:** the subsequent H attempt
46143347 timed out and was closed as incomplete qualification; its terminal
scientific artifact remains unverified. The renewed window expired at
2026-10-06 01:12:23 UTC. No real projection outcomes or audit data opened,
and no successor is selected. See the [limitation closeout](outputs/experiment_log.md#2026-10-05-0146-utc--population_h_optimizer_recovery-local-record-limitation-closeout)
and [current data boundary](DATASETS.md#current-evidence-boundary--october-6-documentation-reconciliation).

**Historical observation, 2026-09-30 16:21 PDT:** the existing synthetic H job
`45736553` completed `0:0` but failed the frozen diagnostic with
`revise_population_H_optimizer_iteration_cap`. The empirical start reached
1,200 iterations, and the two converged H fits differed by
`0.129459 ± 0.000104 MCSE` on fresh draws against a `0.002` tolerance.
The best truth-initialized fit's truth deficit upper bound passed
(`0.001997 < 0.010`), so initialization recovery is the diagnosed blocker.
Panel A remains incomplete and real outcomes remain closed. This attempt has
ended; any revised start strategy and another launch require a prospective
decision and the existing budget accounting. The earlier status narrative
below records the path to this attempt; its in-progress H wording is superseded
by this observation. The research question, boundaries and thresholds are
unchanged. Actual result: `outputs/stage0/population_h_recovery_summary.json`.

**Historical v2 preparation narrative, before the September 30 terminal observation:**
the original SEU-A1876 `SS` design (v1) completed Stage 0 as
`closed_insufficient_source_support`. EP11 was reopened as design v2 for
Stage-0 qualification of the public Gao whole-cortex `SSp-tr` cohort. The
targeted SWCs and source metadata have been acquired and structurally checked,
and the outcome-blind cortex-geometry support gate is feasible for 36 distinct
fMOST/provider-sample units. A fixed 56-coordinate projection-composition
definition and its synthetic constructor tests now exist. The completed
five-target recovery surrogate failed group-recovery and interval-precision
checks. In the pre-repair 56-coordinate implementation, all 280 synthetic
comparisons were numerically valid but no reusable-group route was recovered
even at the maximum tested effect. The code now enforces the exact factorial
pairing (`H = C + groups`, `D = B + the same groups`). Its current synthetic
smoke subset also passes whole-unit development-fold separation, freezes a
`C`-selected noise scale into `H`, estimates the shared group term conditional
on the frozen `C` backbone, and passes structural and zero-group-degeneracy
tests. Its effect calibration is still only a projected-oracle proxy and the
completed 400-comparison smoke result is `revise`: numerical validity and the
no-group false-predictive-gain screen passed, but strong-group predictive gain
was never supported and only 1.75% of intervals met the 0.05-nat precision
target. A subsequent 32-replicate, common-random-number diagnostic showed that
an infeasible 27-development/8-audit ceiling did not improve recovery, while
an infeasible 12-development/23-audit ceiling reduced the strong-fixture
one-sided half-width only to 0.076. Giving the estimator the true synthetic
development labels raised `H - C` only to about 0.04 nats/neuron, versus a
projected-oracle proxy near 0.10--0.12. Outcome-blind geometry also shows that
at most 33 of the 36 units can be assigned under either tested depth
convention; a single cross-convention assignment has not been established.
These results prioritize estimator, gating, signature, and full-`C`
calibration repair over further role-count searches. The identified-coordinate
likelihood and optimizer are now repaired and exact truth anchors pass, but a
representative 32-draw `C` MLE overfits its finite pool and cannot define the
population projection. The original positive-amplitude `C`
`R=128/512/2048/8192` group-stratified SAA ladder returned
`revise_population_C_SAA_pilot_gates` because its two-start `R=128` rung
exposed a persistent empirical-initializer bad basin. The prospectively frozen
`r128_three_fresh_v1` revision retained every rung and threshold, used fresh
validation and terminal streams, and completed with all ten pilot gates true.
Its exact status is
`pass_population_C_SAA_pilot_gates_not_Panel_A_calibration`; the final ladder
change was `0.004545 < 0.005`, the `R=8192` best-two validation difference was
`0.0000595 < 0.002`, and terminal truth-minus-C MCSE was `0.000357 < 0.005`.
This closes only the positive-amplitude population-`C` stability item. A single
synthetic fitted-`H` optimizer-recovery job (`45736553`) has now been launched
and is being babysat. Panel A, outcome-bearing access, and all real projection
outcomes remain closed. The
subset has not completed registered recovery
qualification. Real regional projection
outcomes have not been constructed or modeled. The 36 units form a
strict one-to-one fMOST-ID/provider-sample-ID mapping; the public release does
not expose an `animal_id` or explicitly certify one unit per independent
animal. Therefore no development/audit split is locked, cross-animal analysis
is not yet authorized, and audit open count remains zero. The v1 closure
remains a historical data--design result, not a result about continuous
organization or reusable projection groups.

## Scientific question

Within one well-defined source population, can neuron-to-neuron differences in
a fixed quantitative regional projection distribution be adequately described
by continuous variation? Or does adding a small number of projection groups,
learned in development and reused without re-clustering, provide a stable and
scientifically meaningful improvement in unseen animals? If reusable groups
are supported, does continuous variation remain within them?

Continuity does not imply a Gaussian or even a unimodal marginal distribution.
A connected, curved projection trajectory can look like several clusters when
neurons are sampled unevenly or its middle is sparsely observed. The continuous
reference must therefore permit nonlinear trajectories, nonuniform density,
heteroskedastic or heavy-tailed variation, and smooth relationships with soma
position. A positive result is necessarily model-relative: it means that
reusable groups add information beyond the prespecified, qualified continuous
family, not that the biological population has been proved intrinsically
discrete.

Soma position, depth, and eligible anatomy are part of the organization to be
explained, not nuisances to remove before the question is asked. Reconstruction
quality, acquisition, registration, and batch remain adjustment and
falsification variables. The biological replicate for the intended primary
paper claim is the verified independent animal. In the current Gao release,
only distinct brain/sample units have been verified. Those units may establish
feasibility, but they cannot silently satisfy a cross-animal claim or its 12/8
minima. Outcome-bearing development requires either provider confirmation that
the units represent distinct animals or an explicit prospective redesign to a
cross-brain/sample-unit estimand with correspondingly narrower language.

## Relationship to EP10 and novelty boundary

EP10's revised biological follow-up asks whether terminal implementation
inside a shared target changes with positively verified co-target context.
Its preserved target-association analysis is a separate frozen track; the
original M2 objective is now closed incomplete.
EP11 instead asks whether a regional allocation distribution benefits from
reusable patterns beyond flexible continuous organization. Neither question
can be answered merely by discovering clusters, and a positive result in one
does not independently confirm the other.

[Winnubst et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6754285/)
already used complete axonal reconstructions to describe projection subtypes
and long-range organization. [Timonidis et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10722239/)
examined projection motifs along a morphological continuum in somatosensory
thalamus. Thus neither axonal diversity nor a motif-versus-gradient picture
is itself the advance. EP11's proposed contribution is conditional predictive
usefulness and a frozen, held-out named allocation rule in `SSp-tr`; it is not
the first study connecting projection motifs and continuous variation.

[Peng et al. (2021)](https://doi.org/10.1038/s41586-021-03941-1) reported
extensive projection diversity within major projection types, topography among
the principles organizing that diversity, and imperfect correspondence between
fine morphological distinctions and transcriptomic subtypes. EP11 therefore
does not treat a visually separated embedding or a morphology-derived cluster
as a new cell type. Its narrower contribution is to test whether a few
development-defined projection patterns transfer across animals beyond a
flexible continuous account.

[Yufeng Liu et al. (2024)](https://doi.org/10.1038/s41467-024-54745-6)
directly analyzed all 1,876 SEU-A1876 morphologies, including spatially tuned
morphology clustering, projection organization, and axonal motifs. EP11 cannot
claim the first analysis of diversity, stereotypy, spatial clustering, or
projection-defined organization in this exact release.

[Lijuan Liu et al. (2025)](https://doi.org/10.1038/s41592-025-02621-6) built a
150-dimensional potential-connectivity barcode based on axon--dendrite overlap,
defined connectivity subtypes in 31 brain regions, combined that estimand with
soma-distance affinity for spatially tuned clustering, and examined secondary
motor cortex and thalamocortical pathways. Its potential-connectivity barcode
and EP11's regional axon-allocation vector are different estimands; neither is
treated as intrinsically superior. EP11 cannot claim novelty from producing
another connectivity clustering, showing subtype separation, or describing
MOs or thalamocortical diversity.

[Xiong et al. (2025)](https://doi.org/10.1038/s41592-025-02784-2) used a
near-identical 1,877-neuron morphology resource to construct probabilistic
arbor- and bouton-level connectomes and analyze spatial and anatomical
modularity. EP11's regional projection outcome is not synaptic connectivity
and cannot inherit connectome-level language from that analysis.

[Gao et al. (2025/2026)](https://doi.org/10.1016/j.neuron.2025.10.019)
released 18,621 Allen-CCFv3 cortical projectomes and reported 346
projection-defined subtypes. That resource is the primary v2 cohort. EP11
cannot claim the first classification of these neurons, the first cortical
projectome atlas, or novelty from rediscovering the released `class` labels.
Those labels and all provider projection summaries are excluded from model
construction. EP11 instead asks whether, inside one named source area, a
qualified nonlinear continuum is sufficient or whether development-defined
patterns add reproducible prediction in unseen animals after accounting for
soma position, depth, layer, Cre-line sampling, and technical variation.

EP11 must instead test, under one fixed quantitative regional projection
outcome, how much organization is continuous and position-related, whether
fixed group definitions add prediction in unseen animals on common spatial
support, and which named downstream target-allocation rule changes as a
result. Before novelty language is used, the report must state whether prior
work shares cells, animals/brains, acquisitions, reconstruction versions, or
derived features with EP11. Shared cells or brains are reanalysis, not
independent confirmation; a shared reconstruction pipeline is also
shared-source evidence.

Episodes 09, 10, and the v1/secondary-sensitivity part of EP11 reuse the
SEU-A1876 release. Their results are correlated sibling analyses, never
independent replications. The Gao whole-cortex v2 primary cohort has its own
EP11-only whole-brain split; A1876 cells do not count toward its development or
audit minima.

## Paper-level biological object

Model comparison is the test, not the final biological claim. The intended
paper must answer a concrete projection-organization question:

> In `SSp-tr`, how do neurons allocate axon across named
> downstream regions? Is that allocation mainly a continuous function of soma
> position and depth, or do spatially overlapping neurons retain a reproducible
> division of projection patterns across animals?

## Version-2 primary study population

The v2 primary source is the harmonized Allen `SSp-tr` population in the Gao
whole-cortex projectome. Provider labels `SSp-tr1`, `SSp-tr2/3`, `SSp-tr4`,
`SSp-tr5`, `SSp-tr6a`, and `SSp-tr6b` map to the same source area. Their layer
suffix is retained as measured anatomy and continuous depth is modeled; layers
are not treated as projection groups or as separate source populations.

The official inventory contains 214 downloaded cells, 36 unique fMOST IDs, and
36 unique provider sample IDs. The two identifiers have a strict one-to-one
mapping. The primary cohort is the 206 `Axon_and_dendrite` reconstructions, for
which the provider distinguishes axon from dendrite; this label does not certify
that each axon is complete. The eight `Axon_only` files remain sensitivity data
and do not enter primary retention, precision, role, or outcome-model
denominators. Every brain/sample unit retains at least one primary-eligible
cell. The release supplies neither an `animal_id` crosswalk nor an explicit
one-pair-per-mouse statement, so this is evidence for 36 distinct brain/sample
units, not yet 36 verified independent animals. The C57BL/6J subset contains 74
cells in nine brain/sample IDs and is a prespecified within-resource
sensitivity, not the primary cohort, because it cannot satisfy the current
12/8 design on its own.

Cre line and layer are strongly related by construction. The primary role
split must therefore retain line representation across development and audit,
and every positive group claim must survive explicit tests that membership is
not simply decoding Cre line, layer, laboratory, or reconstruction batch. The
single Tle4 brain/sample is prospectively excluded from primary role assignment
but retained as a descriptive sensitivity. The five A1876 provider-labelled
`SSp-tr` cells are a cross-resource sensitivity only.

All 206 primary cells have valid cortical streamlines and all 36 brain/sample
units remain represented. A line-balanced
12-development/8-audit/16-unassigned allocation exists at a 500-µm
pial-surface geodesic radius with a 0.10 normalized-depth tolerance while
requiring each retained cell to have neighbors in at least two distinct
development brain/sample units, at least 0.90 retention within every role
unit, and at least 0.95 retention within each role overall. The stricter
250-µm radius is infeasible at depth tolerances 0.10 and 0.20, but feasible at
0.30. The fixed `LineString3D` depth-projection sensitivity gives the same
boundary. This establishes operational spatial-design feasibility for the 36
verified units; it does not certify animal independence, choose or disclose a
role assignment, or provide evidence for groups or for a continuum.

Development may identify a small number of target-allocation contrasts within
the fixed primary outcome, but it must freeze the named source, targets,
direction, spatial domain, effect scale, and multiplicity rule before audit.
The held-out report must show the actual target distributions and spatial
relationships, not only `H - C`. Each candidate report must state which Peng-
or prior resource-specific organization it overlaps, what those descriptions
did not adjudicate, and what biological interpretation would change if EP11
supports groups versus a continuum.

## A motivating example

Suppose some neurons in one source project mainly to A and B while others
project mainly to C and D. At least three organizations could produce that
picture:

| Explanation | What could be observed |
| --- | --- |
| **Continuous variation** | Projection preference changes gradually from A/B to C/D along a straight or curved trajectory; uneven sampling can leave few observed intermediate cells. |
| **Reusable groups** | A small set of projection distributions recurs in different animals under one fixed definition. |
| **Both** | Reusable projection patterns recur, while projection distributions also change continuously with soma position within each pattern. |

Sparse intermediate observations alone do not distinguish the first two.
Animal-specific sampling of different positions can also manufacture apparent
groups. EP11 must establish common spatial support before interpreting a gap as
biological organization.

## Operational explanations and model hierarchy

The primary comparison is **continuous variation plus reusable groups versus
continuous variation alone**. All adjudicative models use the same cells,
folds, outcome, coordinate system, and scoring measure.

| Model | Operational explanation | Role |
| --- | --- | --- |
| `B` | An outcome-blind-frozen whitelist of declared design and technical adjustments, including Cre/strain line and acquisition variables where estimable. It has no soma-position, cortical-layer/depth, unregistered morphology, or latent biological coordinate. | Baseline for the position contrast within harmonized `SSp-tr`. |
| `P` | Exactly `B` plus smooth projection change with folded tangential soma position and continuous depth, with provider layer retained as a registered anatomical covariate rather than a source stratum. | Describes continuous topographic organization within `SSp-tr`; `P - B` quantifies what measured position and depth add. |
| `C` | `P` plus a low-dimensional connected continuous latent coordinate for residual differences among cells at similar positions. Its marginal density may be multi-peaked. | Qualified continuous reference for the primary comparison; `C - P` tests nearby-cell continuous heterogeneity. |
| `D` | Exactly `B`'s non-position adjustment and animal nuisance plus a small number of fixed reusable projection signatures. Group prevalence may vary smoothly with position, but the projection signature itself has no continuous position or latent-coordinate outcome trajectory within a group. | Group-dominant reference for whether within-group continuity is needed. |
| `H` | `D`'s adjustment, animal nuisance, groups, and gating plus the same position-driven and continuous-latent outcome structure available to `C`, allowing both spatial and residual continuous variation within groups. | Candidate grouped explanation; compared primarily with `C` and secondarily with its nested `D` pair. |

Simple Gaussian (`G`) and flexible unimodal (`U`) models remain useful
diagnostic anchors. They do not define the scientific continuous null and
cannot promote a grouping claim merely because `D` or `H` predicts better than
they do. The `C` backbone must receive the same smooth and latent-continuum
capacity available inside `H`, apart from the finite group variable.

The `B` whitelist is frozen before real projection outcomes. It may contain
only explicitly declared acquisition/design variables and prospectively
validated technical QC; any sex, age, strain/genotype or Cre line, labeling,
modality, hemisphere, laboratory, batch, or reconstruction-version term must
be named. Folded tangential position, provider layer, and continuous depth
enter `P`. The provider's `SSp-tr*` layer labels are collapsed to one source
population and cannot themselves be called projection groups. Axon-derived fields, released projection class, and
unregistered dendritic or other morphometric predictors are excluded. The same
whitelist is used in every matched pair and in the final model lock.

The continuous latent dimension is restricted to one or two, its smoothness
and noise families are bounded in advance, and the same limits apply in `C`
and `H`. Every `H` trial has a registered, qualified `C` using the identical
continuous-backbone family and capacity, with only the finite-group signatures
and gating removed from its grammar. The pair shares cells, folds,
outcome/observation contract, predictors, continuous hyperparameters and their
tuning treatment, inner data, convergence criteria, and optimization-effort
rule. Each family must nevertheless optimize its own likelihood: coefficients
estimated under `C` are not frozen inside `H`, because doing so prevents `H`
from reallocating signal that `C` absorbed. Nestedness is established by the
parameter restriction that coincident group signatures reduce `H` to `C`, not
by forcing independently fitted models to have elementwise-identical
coefficients. Effective-capacity tolerances apply to the shared continuous
family, not to the deliberately added group terms.
Across the complete archive, however, `C_ref` and `H_ref` are selected
independently by each family's best absolute development predictive score under
equal search budgets, with complexity and a deterministic canonical-
configuration ordering as tie breaks. The primary contrast is
`H_ref - C_ref`, never a pair chosen to make `C` weak; all
family selection is replayed in each full-search null. Below, unqualified
`H - C` is shorthand for this family-reference contrast. `D` and `H` use the
same allowed `K`, canonical-label rule, group-signature parameterization, and
gating family so that `H - D` tests the added within-group continuum rather
than a new clustering grammar. Their fitted coefficients may differ after
each family optimizes its own likelihood. Likewise, the selected `P_ref` carries
`B_pair_for_P_ref`, and `C_ref` carries `P_pair_for_C_ref`, with the same nuisance, residual, and
tuning treatment. Thus `P - B` and `C - P` remain nested incremental contrasts
rather than differences between unrelated best-family fits.
Because a flexible continuum and a finite mixture can approximate one another,
the comparison concerns predictive usefulness, stability, and interpretability
among these validated descriptions, not a unique ontology.

Many neurons come from each animal, so every model also contains the same
frozen hierarchical animal-level nuisance/random-effect structure. The primary
estimand is marginal per-neuron predictive density for an equal-weight new
animal, not the joint density of all its neurons. The new animal effect is
integrated over the population distribution learned from development animals;
it is not refit or inferred from other held-out outcomes. Uncertainty is
clustered at the verified-animal level, and synthetic and fitted-`C` nulls draw
one shared animal effect plus within-animal correlated residuals per animal.
Otherwise an animal-level projection shift could masquerade as a reusable
neuron group.

The prespecified contrasts are:

1. `H_ref - C_ref`, primary: do reusable groups add cross-animal predictive information
   beyond qualified continuous organization?
2. `P_ref - B_pair_for_P_ref`, secondary: how much additional projection organization is
   predicted by soma position and depth?
3. `C_ref - P_pair_for_C_ref`, secondary: do nearby-position cells retain continuous residual
   heterogeneity?
4. `H_ref - D_pair_for_H_ref`, secondary: after reusable groups are added, is continuous
   within-group organization still needed?
5. `D_pair_for_H_ref - C_ref`, descriptive: is a group-dominant account competitive with the
   continuous reference?

Predictive gain and a development-frozen outcome-scale deviance or variance
summary should both be reported for `P - B`. Neither is a causal fraction of
biological variance.

## Fixed primary outcome and common scoring space

The v2 primary outcome is the fixed
[`allen_28_named_target_relative_axon_length_composition_v1`](outputs/stage0/projection_outcome_contract.yaml):
a 56-coordinate composition formed by 28 non-overlapping named Allen
grey-matter targets crossed with ipsilateral/contralateral laterality. Each
coordinate is registered axon centerline length assigned to that named target
and side, divided by total length over the same 56 named coordinates. Total
named-target mass is therefore discarded by normalization. Exact measured
zeros are retained without a pseudocount; a cell with zero total named-target
length is a construction failure, not a basis for post-outcome target or cell
selection. Unclaimed generic grey matter, fibre tracts, ventricular systems,
grooves, retina, and annotation-zero/outside-volume length are reported as QC
quantities outside both the 56 coordinates and their denominator. All
attributed grey-matter axon length is counted, so a passing axon inside grey
matter can contribute; the outcome is not a synapse, bouton, or terminal-arbor
measure. The fixed coordinate order is recorded in
[`allen_target_vocabulary.csv`](outputs/stage0/allen_target_vocabulary.csv).

The constructor operates on provider reconstructions in which axon and
dendrite are distinguished. That distinction is not evidence that a
reconstruction contains a complete axon. Completeness, truncation, registration,
and provider-hemisphere agreement remain separate outcome-blind eligibility
checks. The constructor has passed synthetic geometry and normalization tests,
but it has not been run on the real SWCs. In version 2, source selection, atlas
resolution, vocabulary, laterality, passing-tract rule, normalization, zero and
missingness handling, eligibility, and the coordinate/base measure are chosen
from anatomy, measurement documentation, reconstruction QC, and synthetic or
precision work **before any real regional projection summary is opened**.
“Target observability” at this stage means outcome-blind completeness and
registration QC, not observed target mass. If a later version uses real
projection outcomes to choose a vocabulary, resolution, zero filter, or
normalization, that entire choice must be replayed inside every development
fold and every full-search-null replicate; it cannot be called fixed afterward.

Every primary model must produce a normalized predictive density for that same
object in that same fixed-dimensional space. Raw log density in a 4-dimensional
embedding cannot be compared with raw log density in a 32-dimensional
embedding. PCA or another low-rank construction may be an internal fitting
device only if it yields a normalized density back on the frozen primary
outcome under the same scoring measure; reduced-space density is never the
primary score.

The aggregation estimand is also frozen. Metadata-only coordinates determine
which cells lie in the common-support population. Every model is scored on the
same retained cells on the same harmonized `SSp-tr` position/depth support;
provider layer and Cre/strain line enter the registered adjustment or
falsification plan rather than defining separate source strata. For the
intended cross-animal estimand, cell log scores are averaged within each
verified animal and animals receive equal weight. A sparse animal therefore
cannot dominate through its cell count. If outcome-blind precision work instead
justifies a spatially standardized estimand, its strata and weights must be
fixed before outcomes and applied identically to every model. No post-outcome
block deletion, reweighting, or imputation is allowed, and an animal need not
occupy every arbitrary grid block merely to enter the conditional predictive
estimand. This scoring rule cannot be activated for a cross-animal claim while
the 36 public units lack animal certification.

Before outcome access, the contract must also choose one observed-data
likelihood and base measure for the nonnegative vector: whether total axon mass
is modeled, conditioned on, or discarded by compositional normalization. A raw
vector uses a product of point mass at zero and continuous measure on positive
values; a composition uses probability over nonempty active target sets plus
the appropriate measure on each simplex face. Each model must therefore score
both the active target pattern and positive values. Any log or log-ratio
transform is bijective on the relevant support and returns density to the
original outcome measure with its observation-specific Jacobian. Unknown,
missing, censored, or truncated coordinates are distinct from structural zero
and are summed or integrated through one frozen observation operator, not zero-
filled or renormalized away. The same likelihood, observation mask/exposure,
transform, and base measure are used by `B/P/C/D/H`, simulations, and null
reruns. A Gaussian or Student-*t* distribution may describe residual
coordinates only after this shared contract has been defined; it is not by
itself a likelihood for the original projection vector.

Topology/path summaries, provider-derived arbors, alternative target
resolutions, and alternative normalization rules are prespecified sensitivity
tracks. Each track compares models only within its own fixed outcome space. It
cannot select, replace, or rescue the primary conclusion, and its log density
is never ranked against the primary score.

Configuration-density averaging is disabled for adjudication in this first
version. Averaging two one-mode densities can be multimodal, and averaging
continuum fits can emulate groups. Any ensemble shown as a predictive
benchmark is non-adjudicative and cannot determine the winning explanation.

## First analysis stage: identity, observability, and common support

The first practical action was a metadata-only source-support census. It fixed
harmonized Gao `SSp-tr` as the v2 source, verified 36 distinct one-to-one
fMOST/provider-sample units, and established an outcome-blind feasible geometry
for a 12/8/16 allocation. Before group-discriminating outcome search, EP11
must still:

- authenticate cells, the provider's axon/dendrite distinction, and the CCF
  binding; apply a separate outcome-blind reconstruction-completeness,
  truncation, and registration eligibility rule; and either verify independent
  animals or prospectively redefine the replicate estimand;
- report, for the selected source population, its number of verified replicate
  units, cells per unit, layer/depth coverage, soma-position overlap, batch
  composition, and sampling of plausible intermediate regions;
- record sex, age, strain/genotype or Cre line, labeling strategy, imaging
  modality, hemisphere, source laboratory, acquisition/registration batch,
  and reconstruction version; prespecify which strata define the source and
  which are adjustment or falsification variables;
- retain or revise the selected source only from identity, reconstruction
  quality, target observability as defined by outcome-blind completeness/
  registration QC, and spatial overlap rather than observed regional
  projections or clustering;
- freeze the primary target vocabulary, quantitative outcome, observation
  rules, and common score;
- report a source-by-replicate-unit-by-soma-position/depth support table, including
  batch, gaps, exclusions, and cells outside common support;
- show overlap between development and audit replicate units and enough local
  overlap to compare putative groups at similar positions; and
- profile synthetic qualification, the 99 complete search-null reruns, and the
  one-shot audit inside the fixed compute budget.

The 12-development/8-audit minima must be satisfied by verified independent
animals within the selected source and its common-support domain; the number of
brain/sample units in the full release does not count as that certification.
Those counts are necessary, not sufficient. Before real outcomes, freeze
non-null rules for minimum effective cells per animal, animals per spatial
neighborhood/layer, an overlap metric and maximum unsampled gap,
missing/unknown/truncated outcome tolerance, and the precision needed to
resolve the meaningful `H - C` margin. Every putative group must also have
multi-animal development support before audit. If only distinct brain/sample
units can be authenticated, the intended cross-animal study is infeasible
unless its scope is explicitly redesigned. No claim that EP11 can adjudicate
cross-animal organization is made until replicate identity and every frozen
support rule pass.

The 12/8 counts are design-specific requirements of this version, not universal
scientific thresholds. Their adequacy must be supported by outcome-blind
precision calculations and synthetic recovery at the declared meaningful
effect. If they are infeasible or inadequate, investigators may prospectively
version a narrower question, different allocation, revised sample-size rule,
or added dataset and repeat qualification before outcome access (or begin a new
round after exposure). They may not lower the rule after seeing favorable or
unfavorable projection patterns. `closed_insufficient_source_support` means
only that the current resource cannot execute the current design; it is neither
evidence against projection groups nor a judgment that EP11 lacks value.

Primary inference is restricted to a development-frozen common-support domain.
Euclidean CCF-radius calculations are preliminary feasibility screens, not the
final geometry. Final qualification requires a validated CCF axis/ontology
binding, separate cortex-aware tangential and depth tolerances, and direct
support for each retained cell from at least two distinct development animals
inside harmonized `SSp-tr`. The same rule is applied to audit cells and to
development cells with their own animal left out; graph chains or a single
bridge do not establish common support. Per-animal and overall retention floors
are fixed at 0.90 per role animal and 0.95 within each role overall; 0.80 is a
sensitivity analysis, not sufficient for a positive group claim. Final
cortical geometry folds homologous hemispheres, uses surface-geodesic distance
between pial streamline endpoints for tangential position, and measures depth
along the local cortical streamline as a fraction of local thickness.

### Historical v1 support screen (retained, not active)

In the historical v1 screen, candidate `SS` was evaluated as a hierarchical analysis of fixed
exact provider-label strata, not as a homogeneous parent area. The global
necessary-support rule excluded generic `SSp` before role enumeration because
it occurred in only one geometry-eligible animal, leaving 318 cells in eight
fixed strata across 22 animals. These labels are not all siblings at one
ontology depth. Every adjudicative model would condition on provider-label
stratum, spatial smooths would be stratum-specific, and a
claimed group must recur within same-label overlapping neighborhoods rather
than merely decode provider-label identity. The allocation to compare is a
deterministic 12-development/8-audit design with two prospectively unassigned
geometry outliers versus a 14-development/8-audit all-animal design. Neither
allocation is selected until frozen simulations compare type-I error, recovery,
interval precision, retention, and animal influence.

That completed v1 cortex-aware screen found no qualifying allocation. Five of 324
cells lacked a valid cortical streamline and were excluded before retention
denominators. Across surface-geodesic radii 250--1,500 µm and normalized-depth
tolerances 0.10, 0.20, and 0.30, the best unconstrained per-animal maximin was
0.778 for 12/8 plus two unassigned animals and 0.500 for forced 14/8, below even
the 0.80 sensitivity floor. No fully valid split existed under the primary
exact-polyline depth definition or the fixed `LineString3D` sensitivity. Even
with depth unrestricted, the five-animal `SSp-tr` stratum had no 3-development/
2-audit assignment satisfying direct two-development-animal support. Thus no
role design was selected and outcome-bearing search did not start in v1. It is
retained as prior feasibility evidence and is not pooled into the v2 primary
role split.

### Active v2 support design

Version 2 fixes a 12-development/8-audit/16-unassigned design on the 36 Gao
`SSp-tr` brain/sample units; 14/8 is not an active alternative. It requires
each nonsingleton Cre/strain line to appear in both assigned roles and does not
lock a split merely because one feasible allocation exists. The geometry and
retention screen has an eligible allocation, but the roles remain unlocked and
cannot be treated as animal roles until identity is resolved. Provider animal
confirmation or an explicit cross-brain/sample-unit redesign, reconstruction
completeness/truncation eligibility, precision, and full registered synthetic
recovery must all pass before regional axon distributions are constructed.

The first completed outcome-blind recovery surrogate does not pass that gate.
Under the fixed 12/8/16 design, all 4,000 requested fits were valid and the
false reusable-group rate was zero, but recovery was also zero at the strongest
registered group effect and the replicate-level intervals never reached the
0.05 nats/neuron precision target. This failure diagnoses the surrogate/
evaluator or the attainable design; it is not evidence for a real continuum
and it cannot be reclassified as launch authorization.

The repaired 56-coordinate crossfit smoke also does not pass. All 400
comparisons were finite and no zero-group condition produced a lower-bound
predictive gain above 0.05 nats/neuron. But at the projected-oracle 2x fixtures,
the fitted `H - C` mean was -0.0176 nats/neuron, predictive-gain recovery was
zero, and planted-label accuracy averaged only 0.570. Across all conditions,
only 0.0175 of repeats had a one-sided 95% half-width below 0.05; for the 2x
group fixtures the mean half-width was 0.145. Independent verification of the
projected-oracle proxy was itself too noisy, with maximum absolute target error
0.0476 nats/neuron. This removes the old gross factorization artifact but shows
that the current estimator and 12/8 precision design remain inadequate. It is
not evidence for continuous biology and it does not authorize outcome access.

The next synthetic diagnostic separated development-size, audit-size, and
estimator ceilings using the same full 36-unit synthetic draw in each
replicate. At the strong reusable-group fixture, moving all 15 reserve units
to development (`D27/A8`) changed fitted `H - C` from -0.0367 to -0.0434
nats/neuron and planted-label accuracy from 0.584 to 0.581. Replacing only the
training labels with their synthetic truth yielded 0.0389 and 0.0441,
respectively, so label discovery alone does not explain the failure. Moving
the reserve to audit (`D12/A23`) produced fitted `H - C = -0.0317`, a mean
one-sided half-width of 0.0759, and precision success in 0.219 of 32 repeats;
the observed unit variance would require about 51 audit units for a half-width
below 0.05. The label-integrated oracle density exceeded its projected-`C`
proxy by about 0.103--0.117, confirming that the fixture contains group
information relative to that proxy, although its independent calibration SE
was 0.016 and the proxy is not the population-optimal registered `C` family.
All zero-group arms had zero false predictive-gain declarations.

Those `D27/A8` and `D12/A23` arms are deliberately infeasible diagnostic
ceilings, not candidate role designs. A companion outcome-blind joint geometry
test found that assigning all 35 nonsingleton units in either direction is
infeasible. Independent count-frontier solves assign at most 33 units; count
pairs `D13/A20` through `D25/A8` are feasible under each depth convention
separately, but may use different hidden allocations. No redesign is selected:
the active contract remains `D12/A8/16 unassigned` while group discovery,
signature estimation, smooth position-dependent gating, and full `C/H`
calibration are repaired and isolated. These 32-replicate results are
diagnostics, not registered qualification, evidence about real biology, or
authorization for outcome-bearing search.

A post-run audit found that this completed diagnostic standardized synthetic
position and gate coordinates inside the index subset supplied to the
generator. Its 20-unit calibration and full-36-unit design run therefore used
slightly different synthetic population definitions. The nested role arms
still share each replicate's full-36 draw and remain useful paired
estimator/design diagnostics, but their absolute "projected-oracle 0.10"
effect label is not qualified. The generator now freezes these covariates on
the complete metadata layout and tests full-versus-subset invariance. Any
calibrated recovery claim requires a new run under that repair.

The repaired generator has now been used for a 256-replicate synthetic
group-module attribution. Its exact truth-gate arm reproduced the oracle
density and the zero-signature arms correctly collapsed at zero effect. The
current finite-D12 reference still failed at the strong fixture (`H - C =
-0.0344` nats/neuron, zero of 32 predictive-gain detections, mean one-sided
half-width 0.115). Conditional point estimates suggest substantial loss in
group discovery and signature estimation, and the D12-fitted smooth gate did
not recover the truth gate. However, the prespecified joint Monte Carlo
precision rule failed at its 256-replicate cap (maximum monitored MC SE 0.0108
versus 0.005), so module-level attribution remains unresolved. The nominal
0.100-nat projected-proxy fixture realized only 0.0880 in this run and is still
not a full-family calibration. These results justify estimator repair, not a
biological conclusion or outcome-bearing launch.

The full-family diagnostic now removes the exact A8 design null space and the
compositional value gauge, leaving 3,109 identified `C` coordinates and 3,224
identified `H` coordinates. Its analytic-gradient optimizer converges on a
representative 32-draw `C` benchmark; exact parameter anchors reproduce `C` at
zero group amplitude and `H` at every tested amplitude; and support and gate
likelihoods now use stable, unclipped log-domain evaluation. However, the
32-draw `C` maximum-likelihood fit is not a population projection. It has 2,432
synthetic rows for 3,109 parameters and overfits the finite pool, yielding a
fresh-draw truth-minus-`C` gap of about 1.13 nats/neuron even at zero amplitude,
where truth is exactly contained in `C`. Panel A therefore remains unresolved.
The initial `R=128→512` population-SAA pilot validated execution but failed
its cross-rung stability rule. The subsequent complete
`R=128/512/2048/8192` ladder had every scheduled start satisfy the
physical-coordinate KKT rule and no best solution on a numerical boundary.
Its final `R=2048→8192` fresh-validation change was
`0.003678 ± 0.001172` nats/neuron, fixed-start polishing gained `0.0`, the
terminal best-two difference was `0.000252 ± 0.000127`, and the best
terminal truth-minus-`C` MCSE was `0.000376`; those checks passed their frozen
thresholds.

The ladder nevertheless returned `revise_population_C_SAA_pilot_gates`
because the two `R=128` starts differed on fresh validation by
`0.518406 ± 0.002456` nats/neuron, above the prespecified `0.002`
start-stability threshold. This was the only false pilot-acceptance boolean,
so the failure is localized to the first-rung start-stability requirement, but
the registered conjunction still fails. The empirical initializer converged
to an interior KKT solution about `0.522` objective units worse than the rung
best at every ladder level, whereas later rungs added the preceding optimum as
a warm start. The `R=128` difference is about 211 MCSE, so this is persistent
start-dependent local-basin structure, not validation imprecision. The
prospectively frozen `r128_three_fresh_v1` schedule then ran three fresh
`R=128` starts followed by two fresh plus one previous-best warm start at each
later rung, using validation stream `5202` and untouched terminal stream
`5303`. It passed all ten unchanged C pilot gates with exact status
`pass_population_C_SAA_pilot_gates_not_Panel_A_calibration`. Population-C
stability is therefore accepted for this synthetic diagnostic, and the
prespecified fitted-H optimizer-recovery attempt is now running as Slurm job
`45736553`. Panel A remains incomplete; this is synthetic estimator work and
does not authorize outcome-bearing search.

Even a successful four-rung run at the current diagnostic amplitude `1.25`
would close only the positive-amplitude `C_pop` stability item. It would not by
itself calibrate the required `0.100`-nat population fixture, establish fitted
`H_pop` recovery, replace the fitted zero-amplitude C/H checks, test D12-to-A8
transport, or complete Panel A. Those are separate synthetic gates.

An unsampled intermediate position is a sampling gap, not evidence for a
biological interval. If animal, batch, and position are inseparable, or if
neighboring-position comparisons have no overlap, the grouping question is
unresolved rather than positive. If critical experimental strata are missing
or perfectly confounded with animal, position, or a proposed group, the report
must narrow the population or stop the relevant interpretation. Pooling
distinct source populations to create support is forbidden.

## Evidence required for a reusable projection group

A held-out animal is not independently clustered and then matched back to a
development cluster. Instead:

1. Development animals determine `K`, group projection signatures, canonical
   label identities, all parameters, the position-dependent mixing rule, and
   the assignment and uncertainty rules.
2. Each held-out development fold receives definitions learned without that
   fold. The final audit receives the definitions refit once on all development
   animals and frozen in the configuration lock.
3. Held-out log density marginalizes over every unobserved group and continuous
   latent coordinate using frozen development rules. It may not use an oracle
   group label or an outcome-fitted latent coordinate as a predictor.
4. Posterior memberships or latent coordinates computed after scoring are
   post-score evidence for recurrence, separation, and trajectory shape. They
   are not predictors and cannot trigger refitting, per-animal re-clustering,
   label permutation or rematching, prototype changes, or audit-prevalence
   recalibration.
5. Every claimed group must have multi-animal support in development and recur
   under its fixed label in at least three eligible audit animals, with
   adequate effective membership. Cross-animal target-signature similarity is
   evaluated within harmonized `SSp-tr` after standardizing to the same folded
   soma-position/depth support and applying the frozen provider-layer and
   Cre/strain-line adjustment or falsification rules. Raw group means under
   different layer, line, or position sampling are not recurrence evidence.
6. Separation must be visible within overlapping soma-position support. A
   group cannot primarily decode animal, batch, QC, reconstruction, disjoint
   territory, sex, age, strain/genotype or Cre line, labeling strategy,
   imaging modality, hemisphere, or source laboratory.
7. The animal-level one-sided lower bound for `H_ref - C_ref` must exceed a
   meaningful, outcome-blind-frozen margin under the common retained-cell,
   equal-animal score, and pass calibration, full-search-null, influence,
   technical-decoding, and synthetic-recovery gates.
8. A one-cell animal may contribute to marginal new-animal prediction, but it
   cannot by itself establish within-animal coexistence, prevalence, a spatial
   gradient, group recurrence, or `H - D`. Those interpretations require a
   simulation-frozen minimum number of multi-cell animals and effective group
   members.

The active v2 allocation is fixed at 12 development, 8 audit, and 16
unassigned units. With only eight audit units, uncertainty remains
replicate-limited; moving units from the unassigned reserve into development is
not a post hoc alternative. Qualification requires calibrated replicate-level
intervals, whole-pipeline false-group control under curved and depleted-middle
continua, and adequate recovery at twice the frozen meaningful effect under
the fixed 12/8/16 design. Failure triggers
`closed_insufficient_source_support` or a prospectively versioned redesign,
not an enlarged neighborhood, weakened neighbor rule, or unregistered 14/8
analysis. For the intended primary claim, all eight audit units must first be
certified as independent animals.

Exact minimum development animals per label, audit recurrence, effective
membership, position-standardized target-signature similarity, held-out
separation, predictive gain, calibration, technical-decoding, local-overlap,
and influence thresholds are chosen from outcome-blind precision calculations
and synthetic qualification, then frozen before real candidate-discriminating
outcomes are searched. They cannot be relaxed after a weak result.

## Required reporting

Regardless of which explanation wins, the report must answer five questions:

- how much held-out projection organization is added by continuous soma
  position and depth (`P - B`);
- whether a low-dimensional continuum explains additional differences among
  nearby-position cells (`C - P`);
- whether neurons at overlapping, nearby positions retain distinct reusable
  projection patterns under fixed definitions;
- whether adding groups improves cross-animal prediction beyond qualified
  continuous variation (`H - C`); and
- whether continuous variation remains within supported groups (`H - D`).

Show animal-level effects, uncertainty, calibration, effective membership,
component target distributions, spatial support, and exclusions. A global
embedding, silhouette score, or pooled-neuron likelihood is not sufficient.

## A valid comparison versus evidence for groups

Stage 0 freezes the estimand and primary outcome construction, source and role
rules, common-support definition, admissible model grammar, tuning/search
budgets, scoring and selection rules, meaningful margins, falsifiers, and audit
boundary. It does **not** fit the scientific models or predetermine all fitted
parameters. Within those frozen rules, development outcomes may estimate noise
and nuisance parameters, choose regularization, smoothness, latent dimension,
covariance option, and other registered candidate configurations, and drive the
adaptive successor sequence. Those choices are evaluated only by animal-held-
out development scoring and are replayed by the full-search null. Adding an
unregistered family, changing the outcome, moving animals between roles, or
choosing a source because its groups look attractive remains forbidden.

Comparison validity and support for reusable groups are separate decisions. A
complete comparison first becomes **pre-null eligible** when data roles and
outcome are fixed, common-space scoring and the paired `C/H` controls are fair,
`B/P/C/D/H` pass calibration and numerical checks, and synthetic qualification
and deterministic replay pass. Development independently selects each family's
best absolute-score qualified reference under equal budgets, then freezes that
package and its `H_ref - C_ref` statistic. Completing the 99 reruns then makes
that comparison valid; this ordering avoids requiring a selected statistic to
run a null that was itself required for selection. Poor separation, low
component prevalence, a failed null *p*-value, or better real-data performance
by `C` does not invalidate the completed comparison.

Accordingly, separation, minimum group prevalence, and group-recurrence gates
may support a positive grouped interpretation but cannot filter out an
otherwise valid comparison or block its continuous-result path. A qualified
comparison must be able to lock and reach audit when `C` is best, subject to
the same data, scoring, calibration, null, and replay requirements.

Separation, prevalence, local spatial overlap, cross-animal recurrence, a
meaningful `H - C` gain, and a passing null *p*-value are additional conditions
only for a positive grouping conclusion. Development must provisionally select
one complete package even when `B`, `P`, or `C` predicts best. After all 99 null
controllers complete, that package is valid and receives its final lock
regardless of the group-support *p*-value; only then does it enter the one-shot
audit, where a continuous, grouped, or unresolved answer can complete the same
workflow.

## Synthetic qualification and falsifiers

Before real adaptive search, the evaluator must demonstrate that it does not
turn difficult continuous structure into reusable groups. Required no-group
fixtures include straight and curved connected trajectories, arc or horseshoe
geometry, strongly nonuniform latent sampling, deliberately depleted
intermediate regions, heteroskedastic and heavy-tailed variation, smooth
spatial organization, animal-specific position coverage, animal-level shifts
and correlated residuals, and experimental-stratum/batch/QC confounding.
Required positive fixtures include reusable groups without
gradients and groups with within-group gradients. Missing and truncated
outcomes must also be represented.

The qualified `C` family must fit the continuous fixtures and control false
group promotion; `D` and `H` must recover reusable patterns when present. A
fitted-`C` null is not adequate if `C` first fails these stress tests.

Positive-fixture amplitudes must be calibrated before qualification to the
oracle expected predictive gain on the same 56-coordinate score: 0, 0.5, 1,
1.5, and 2 times the frozen meaningful `H - C` margin. A raw shift multiplier
is not itself a multiple of a predictive margin. Calibration and recovery are
reported separately from whether the simulated groups satisfy the frozen
multi-unit reuse criteria. The same fixed role assignment or outcome-blind role
selection algorithm is replayed across fixtures; favorable seeds, splits,
labels, or `K` values may not be selected after recovery is observed.

Here `C_pop` means the maximizer of expected equal-unit log predictive density,
not the maximizer on one finite synthetic pool. Exact `C`-at-zero and `H`-truth
parameter anchors are structural scoring checks; they do not by themselves
establish recovery of a fitted population optimum. At positive group
amplitudes, the `C` population projection must have a stable increasing-SAA
limit and pass untouched independent verification before an amplitude is
called calibrated.

The real-data falsification set includes:

- exactly 99 fixed-seed fitted-`C` null datasets, each beginning with an empty
  adaptive proposal history and rerunning the complete deterministic search
  controller while leaving the shared EP09--11 role/exposure table unchanged
  and never opening, generating, or using audit outcomes;
- conditional residual randomization and soma-graph surrogates preserving the
  frozen spatial dependence;
- animal, batch, QC, truncation, cell-count, spatial territory, sex, age,
  strain/genotype or Cre line, labeling, modality, hemisphere, and laboratory
  prediction from posterior group membership;
- capacity-matched noise features and group/outcome shuffles;
- leave-one-development-animal and leave-one-spatial-territory influence;
- full-axon versus validated provider-arbor and fixed alternative-outcome
  sensitivity tracks; and
- explicit `B`, `P`, `C`, `D`, and `H` comparisons on identical cells and
  folds.

The Monte Carlo rule is
`p=(1 + #{null statistic >= observed}) / 100`, with `p <= 0.05` required for a
positive grouping conclusion. Freeze controller code, proposal
model/version/prompt, sampling configuration, generator-selection rule, and
the random-seed table before real outcomes. If deterministic replay or the complete
null budget is infeasible, revise the contract before outcome search rather
than weakening the null.

## Search, lock, and one-shot audit

The bounded adaptive search retains 48--120 pre-null-eligible development
trials, at least
16 coverage trials, patience of 18 valid trials after trial 48, a 1,500
CPU-core-hour total ceiling, no GPU, and a 120-hour wall-clock ceiling. Search
maintains an archive of every pre-null-eligible complete model comparison plus a
Pareto
view over predictive score, calibration, group reuse, capacity, and influence.
`H` need not win for a comparison to enter the archive or audit. An incumbent
is a nonterminal development object; reaching a budget or patience boundary
freezes the independently selected family references and statistic for null calibration
and is not scientific success. After every required null rerun finishes, the
comparison becomes valid regardless of whether its group-support *p*-value
passes; it can then be locked and audited.

The 1,500 CPU-core-hour and 120-hour limits are global across the observed
controller, all 99 null controllers, synthetic qualification, and replay. Before
outcome access, profiling must freeze one identical per-controller cap and
reserve 100 times that amount, plus qualification/replay, within the global
envelope; the observed search cannot spend the 99 null-controller reserves.
The observed and null controllers have the same proposal opportunities, trial
bounds, compute cap, and stop/selection behavior. Because the minimum entails
48 complete trials in the observed controller and in each of 99 null
controllers, feasibility profiling covers at least 4,800 complete `B/P/C/D/H`
fits at the trial-set level. Null controllers use the same grammar, trial
bounds, stop rule, and deterministic selection through the pre-null stage, and
they do not launch nested nulls. If the identical-controller reservation cannot
fit 1,500 core-hours, the planned full-search null is infeasible and the
contract must be revised before outcomes; a smaller null search is not a valid
substitute.

After the development stop rule fires, provisionally freeze one package
containing independently selected `P_ref/C_ref/H_ref` and their registered
`B_pair_for_P_ref/P_pair_for_C_ref/D_pair_for_H_ref`, together with the primary
outcome and score, continuous backbone, group definitions, thresholds,
group/block weights, code and environment, exclusions, decision table, null
generator, and random-seed table. Run the 99 null controllers without changing that
package. Only after all finish are their results attached and the valid package
finally locked. The audit runner then opens all sealed independent
animals once and applies every frozen model directly. No outcome,
representation, family, `K`, prototype, label, prevalence, threshold,
exclusion, or claim may be retuned after the provisional freeze.

At least 12 complete development animals and 8 complete sealed-audit animals
are required within the selected source and common-support domain. No random-
neuron split may substitute for animals.

## Evidence chain

1. **Observation and support.** Establish one measurable source population,
   fixed regional projection outcome, animal identity, and overlapping spatial
   support.
2. **Continuous organization.** Estimate continuous projection structure and
   the incremental contribution of soma position without assuming a unimodal
   marginal distribution.
3. **Reusable groups.** Ask whether a few development-defined patterns improve
   prediction in unseen animals without re-clustering or rematching.
4. **Within-group organization and boundaries.** Test remaining gradients,
   technical and spatial alternatives, outcome sensitivities, and domains where
   the conclusion ceases to transfer.

## Possible outcomes

| Outcome | What would be needed | Scientific interpretation |
| --- | --- | --- |
| **Reusable groups plus within-group continuum** | `H - C` exceeds the meaningful margin and all reuse gates; `H - D` also shows meaningful transfer. | A small number of cross-animal projection patterns add information beyond the tested continuum, and each pattern retains continuous organization. |
| **Reusable groups without supported within-group continuum** | `H` beats qualified `C` and passes reuse gates, while the upper bound on `H - D` is below the meaningful margin with adequate sensitivity. | Reusable projection patterns are supported at the tested resolution; a within-pattern gradient is not supported. |
| **Reusable groups; within-group organization unresolved** | `H - C` and all reuse gates support groups, but `H - D` is not precise enough to support or exclude a meaningful within-group continuum. | Reusable projection patterns are supported, while their internal organization remains unresolved. |
| **Continuous organization adequate** | Qualified `P/C` models pass absolute calibration and trajectory checks, a named target-allocation relationship transfers across animals, and the upper bound on `H - C` is below the meaningful margin with adequate sensitivity. | The tested finite groups add less than the meaningful margin beyond continuous spatial or latent organization at this resolution. Only if apparent modes were separately documented may the report say this continuous description accounts for them. This is not a claim of unimodality or proof that no groups exist. |
| **Unresolved** | Power, common support, continuous-model adequacy, or the `H - C` contrast is insufficient. | The data and prespecified models do not distinguish the explanations. |
| **Insufficient source support** | Metadata qualification finishes but no source has 12 development and 8 audit animals with adequate cell counts and overlapping position/depth support. | The intended cross-animal comparison is infeasible in this resource; no projection-organization conclusion is drawn. |
| **Technical failure** | Identity, observability, calibration, synthetic recovery, deterministic replay, or audit integrity fails. | No scientific comparison is valid. |

## Active version-2 qualification requirements

The user-authorized v2 redesign must freeze the
source population; target vocabulary, laterality, and quantitative projection
measure; common-support rule; continuous latent dimension and smoothness
ceiling; group count range and canonical-label rule; outcome-scale and
predictive meaningful margins; effective-membership, reuse, calibration, and
influence thresholds; and audit report template. These choices must use
identity, measurement, synthetic recovery, and precision rather than favorable
real grouping results.

Historical v1 note: simply dropping `SSp-tr` was not a repair to v1. It would have left
21 animals and seven fixed strata after also excluding unsupported generic
`SSp`, making 14/8 impossible and changing the allocation comparison to a new
12/8/1-versus-13/8 problem. That narrower design requires new support,
precision, and simulation qualification or additional animals before outcome
access.

## Claim boundary

The strongest positive claim is that, within the pinned source population,
target vocabulary, observation rule, and common-support domain, a small number
of cross-animal reusable projection patterns provide predictive information
beyond a qualified continuous model. If `H - D` transfers, the claim may also
state that continuous projection organization remains within those patterns.

A completed continuous result may instead claim that a specific downstream
allocation follows a reproducible spatial or low-dimensional continuum and
that any added value of the tested finite groups is bounded below the frozen
meaningful margin. Either paper-level claim must name the source, targets,
spatial relationship, and prior description it changes.

This is not evidence for molecular, transcriptomic, developmental, or
functional cell types; synaptic connectivity; causal wiring programs; whole-
brain generality; or independent confirmation of EP09/10. Even a positive
`H - C` result does not prove intrinsic discreteness outside the tested
continuous family. A negative result does not exclude grouping at another
resolution, in another source, or with independent molecular or functional
measurements.

All cross-animal terms in this contract describe the intended primary
estimand, not a property already established for the Gao cohort. Unless the
provider confirms independent-animal identity, any prospectively redesigned
analysis must use `brain/sample unit` throughout its methods, results, and
claims and must not imply animal independence.

## Authority and local execution

This is the current local Episode 11 v2 contract, governed by
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md). It grants no
scientific acceptance, reward, or canonical Landscape transition. The v1
Stage 0 ended without opening outcome-bearing data. Version 2 is currently
limited to targeted acquisition, identity/readiness qualification, and
outcome-blind soma geometry, fixed 56-coordinate outcome construction, and
synthetic score-path checks. No real regional projection vector has been
constructed. Before it opens outcome-bearing development it requires identity
resolution, outcome-blind reconstruction eligibility, full registered recovery
qualification, the frozen policy, and the audit boundary described above. Any later Brain Researcher
registration is optional post-run governance and cannot turn shared-source
evidence into independent replication.

## Explicit wall-clock amendment — 2026-09-30

The user instructed “ok you should reset 120hours from now.” The renewed
120-hour continuous clock is recorded from September30,18:12:23 PDT
(2026-10-01T01:12:23Z), ending October5,18:12:23 PDT
(2026-10-06T01:12:23Z). Queue and gaps consume this window. Prior history
and CPU usage remain recorded; the 1500 allocated-core-hour ceiling, zero
GPUs, question, data boundaries, gates and scientific stopping rules are
unchanged. This clears the wall-clock-origin gate for one prepared H revision,
not Panel A, biological access or a successor. The proposed request is
3 CPUs/16GB/72h (at most216core-hours), with an absolute scheduler deadline
that prevents scheduling a job unable to finish in this new window.
