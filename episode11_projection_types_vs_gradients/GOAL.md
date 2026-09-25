# EP11 — Continuous variation and reusable projection groups within a source population

**Status:** local Stage-0 metadata/readiness qualification active; real
projection outcomes and audit remain unopened, and no EP11 result is reported.

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
falsification variables. The biological replicate for the primary paper claim
is the verified independent animal. A conservative specimen group may be used
for feasibility or a separately narrowed cross-group description, but it cannot
satisfy the cross-animal claim or silently count toward its 12/8 minima.

## Relationship to EP10 and novelty boundary

EP10 asks how projection targets are associated within neurons: whether
particular target combinations recur beyond prevalence, spatial routing, and
measured population composition. EP11 asks how differences among neurons'
complete projection distributions are organized. A reproducible co-projection
rule does not imply discrete neuron groups, and a reusable projection group
does not by itself establish an unexplained association among its targets.

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

EP11 must instead test, under one fixed quantitative regional projection
outcome, how much organization is continuous and position-related, whether
fixed group definitions add prediction in unseen animals on common spatial
support, and which named downstream target-allocation rule changes as a
result. Before novelty language is used, the lineage dossier must trace overlap
at the cell, animal/brain, acquisition, reconstruction-version, and derived-
feature levels. Shared cells or brains are reanalysis, not independent
confirmation; a shared reconstruction pipeline is also shared-source evidence.

Episodes 09, 10, and 11 reuse the SEU-A1876 release. Their cell identity,
independent-animal/brain mapping, duplicate lineage, exclusions, development/audit
roles, and first-outcome-access times must live in one shared ledger. Their
results are correlated sibling analyses, never independent replications.

## Paper-level biological object

Model comparison is the test, not the final biological claim. The intended
paper must answer a concrete projection-organization question:

> In one named source population, how do neurons allocate axon across named
> downstream regions? Is that allocation mainly a continuous function of soma
> position and depth, or do spatially overlapping neurons retain a reproducible
> division of projection patterns across animals?

Development may identify a small number of target-allocation contrasts within
the fixed primary outcome, but it must freeze the named source, targets,
direction, spatial domain, effect scale, and multiplicity rule before audit.
The held-out report must show the actual target distributions and spatial
relationships, not only `H - C`. Each candidate dossier must state which Peng-
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
| `B` | An outcome-blind-frozen whitelist of declared design and technical adjustments, without position/anatomical proxies, unregistered morphology, or a latent biological coordinate. | Baseline for the position contrast. |
| `P` | Exactly `B` plus smooth projection change with soma coordinates and depth, and a registered layer term when layer is not fixed by the source definition. | Describes continuous topographic organization; `P - B` quantifies what measured position adds. |
| `C` | `P` plus a low-dimensional connected continuous latent coordinate for residual differences among cells at similar positions. Its marginal density may be multi-peaked. | Qualified continuous reference for the primary comparison; `C - P` tests nearby-cell continuous heterogeneity. |
| `D` | Exactly `B`'s non-position adjustment and animal nuisance plus a small number of fixed reusable projection signatures. Group prevalence may vary smoothly with position, but the projection signature itself has no continuous position or latent-coordinate outcome trajectory within a group. | Group-dominant reference for whether within-group continuity is needed. |
| `H` | `D`'s adjustment, animal nuisance, groups, and gating plus the same position-driven and continuous-latent outcome structure available to `C`, allowing both spatial and residual continuous variation within groups. | Candidate grouped explanation; compared primarily with `C` and secondarily with its nested `D` pair. |

Simple Gaussian (`G`) and flexible unimodal (`U`) models remain useful
diagnostic anchors. They do not define the scientific continuous null and
cannot promote a grouping claim merely because `D` or `H` predicts better than
they do. The `C` backbone must receive the same smooth and latent-continuum
capacity available inside `H`, apart from the finite group variable.

The `B` whitelist is frozen before real projection outcomes. It may contain
only explicitly declared acquisition/design strata and prospectively validated
technical QC; any pooled sex, age, strain/genotype or Cre line, labeling,
modality, hemisphere, laboratory, batch, or reconstruction-version term must
be named. Layer and subregion are either part of the source definition or enter
`P`, never `B`. Axon-derived fields, released projection class, and
unregistered dendritic or other morphometric predictors are excluded. The
whitelist hash is carried in every matched pair and the final lock.

The continuous latent dimension is restricted to one or two, its smoothness
and noise families are bounded in advance, and the same limits apply in `C`
and `H`. Every `H` trial has a registered, qualified `C` using the identical
continuous backbone with only the tested finite-group signatures and gating
removed. The pair shares cells, folds, outcome/observation contract, predictors,
continuous hyperparameters and their tuning treatment, inner data, convergence
criteria, and optimization-effort rule. Effective-capacity tolerances apply to
that shared continuous backbone, not to the deliberately added group terms.
Across the complete archive, however, `C_ref` and `H_ref` are selected
independently by each family's best absolute development predictive score under
equal search budgets, with complexity and canonical-hash tie breaks. The
primary contrast is `H_ref - C_ref`, never a pair chosen to make `C` weak; all
family selection is replayed in each full-search null. Below, unqualified
`H - C` is shorthand for this family-reference contrast. `D` and `H` use paired
`K`, label, and group-signature
rules so that `H - D` isolates the added within-group continuum rather than a
new clustering. Likewise, the selected `P_ref` carries
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

The primary outcome is one quantitative regional projection-distribution
vector derived from authenticated complete axons on a frozen, non-overlapping
target vocabulary. In version 1, source selection, atlas resolution, vocabulary,
laterality, terminal-arbor versus passing-tract rules, normalization, zero and
missingness handling, eligibility, and the coordinate/base measure are chosen
from anatomy, measurement provenance, reconstruction QC, and synthetic or
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

The aggregation estimand is also frozen. Metadata-only coordinates define the
common-support blocks; log scores are averaged first across cells within each
animal-by-block cell, then equally across retained blocks within an animal, and
finally equally across verified animals. Each retained animal must meet the
frozen cell minimum in every retained primary block. Empty blocks are not
post-outcome reweighted or imputed: the block or animal is prospectively
excluded under the common-support rule.

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

The first practical action is a metadata-only source-support census. Before
group-discriminating outcome search, EP11 must:

- authenticate cells, complete axons, CCF assets, and verified independent
  animals, then freeze the shared EP09--11 exposure and role ledger;
- enumerate every candidate source population and report its number of
  independent animals, cells per animal, layer/depth coverage, soma-position
  overlap, batch composition, and sampling of plausible intermediate regions;
- record sex, age, strain/genotype or Cre line, labeling strategy, imaging
  modality, hemisphere, source laboratory, acquisition/registration batch,
  and reconstruction version; prespecify which strata define the source and
  which are adjustment or falsification variables;
- select one source population from that table, reconstruction quality, target
  observability as defined by outcome-blind completeness/registration QC, and
  spatial overlap rather than observed regional projections or clustering;
- freeze the primary target vocabulary, quantitative outcome, observation
  rules, and common score;
- report a source-by-animal-by-soma-position/depth support table, including
  batch, gaps, exclusions, and cells outside common support;
- show overlap between development and audit animals and enough local overlap
  to compare putative groups at similar positions; and
- profile synthetic qualification, the 99 complete search-null reruns, and the
  one-shot audit inside the fixed compute budget.

The 12-development/8-audit minima must be satisfied by verified independent
animals within the selected source and its common-support domain; the number of
animals in the full release does not count. Those counts are necessary, not
sufficient. Before real outcomes, freeze non-null rules for minimum effective
cells per animal, animals per spatial neighborhood/layer, an overlap metric and
maximum unsampled gap, missing/unknown/truncated outcome tolerance, and the
precision needed to resolve the meaningful `H - C` margin. Every putative group
must also have multi-animal development support before audit. If only
conservative specimen groups can be authenticated, the intended cross-animal
study is infeasible unless its scope is explicitly redesigned. No claim that
EP11 can adjudicate cross-animal organization is made until the table exists
and passes every frozen support rule.

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
   evaluated after standardizing to the same soma-position/depth support; raw
   group means under different position sampling are not recurrence evidence.
6. Separation must be visible within overlapping soma-position support. A
   group cannot primarily decode animal, batch, QC, reconstruction, disjoint
   territory, sex, age, strain/genotype or Cre line, labeling strategy,
   imaging modality, hemisphere, or source laboratory.
7. The animal-level one-sided lower bound for `H_ref - C_ref` must exceed a
   meaningful, outcome-blind-frozen margin under the common cell-within-block,
   equal-block-within-animal, equal-animal score, and pass calibration, full-
   search-null, influence, technical-decoding, and synthetic-recovery gates.

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

The real-data falsification set includes:

- exactly 99 fixed-seed fitted-`C` null datasets, each beginning with an empty
  adaptive proposal-history ledger and rerunning the complete deterministic
  search controller while preserving the immutable EP09--11 exposure/role
  ledger and never opening, generating, or using audit outcomes;
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
the seed manifest before real outcomes. If deterministic replay or the complete
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
generator, and seed manifest. Run the 99 null controllers without changing that
package. Only after all finish are their outputs and hashes appended and the
valid package finally locked. The audit runner then opens all sealed independent
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

## Decisions still open

Before candidate-discriminating development, investigators must freeze the
source population; target vocabulary, laterality, and quantitative projection
measure; common-support rule; continuous latent dimension and smoothness
ceiling; group count range and canonical-label rule; outcome-scale and
predictive meaningful margins; effective-membership, reuse, calibration, and
influence thresholds; and audit report template. These choices must use
identity, measurement, synthetic recovery, and precision rather than favorable
real grouping results.

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

## Authority and local execution

This is the current local Episode 11 contract, governed by
[`../ADAPTIVE_SEARCH_PROTOCOL.md`](../ADAPTIVE_SEARCH_PROTOCOL.md). It grants no
scientific acceptance, reward, or canonical Landscape transition. The explicit
local Stage-0 launch began on 2026-09-25 UTC and may perform only metadata and
readiness qualification, synthetic checks, and trusted-handoff preparation
without opening outcome-bearing data. Outcome-bearing development still requires
the role-safe data, shared ledger, frozen policy, and audit boundary described
above. Any later Brain Researcher
registration is optional post-run governance and cannot turn shared-source
evidence into independent replication.
