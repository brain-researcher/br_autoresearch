# EP11 paper plan — From spatial projection continua to reusable target allocation

**Status:** prospective plan; no source population, target pattern, or EP11
result has been selected.

## Paper question

Within one well-sampled source population, how do individual neurons allocate
axon across named downstream regions? Is that allocation mainly a continuous
function of soma position and depth, does a low-dimensional continuum remain
among nearby cells, or do spatially overlapping neurons show a small number of
projection divisions that recur under fixed definitions in new animals?

The paper should not end at “a grouped model predicts better.” It must identify
the source population, target regions, spatial relationship, held-out
allocation rule, and prior biological description that the result changes.

## Closest prior work and the required advance

[Peng et al. (2021)](https://doi.org/10.1038/s41586-021-03941-1) established
extensive projection diversity within major projection types, topographic
organization, and limits to fine morphology--transcriptome correspondence.

[Yufeng Liu et al. (2024)](https://doi.org/10.1038/s41467-024-54745-6)
analyzed the exact 1,876-neuron SEU-A1876 resource, including spatially tuned
morphology clustering, projection organization, and axonal motifs. EP11 cannot
claim the first spatial clustering, projection-defined organization, diversity,
or stereotypy result in this release.

[Lijuan Liu et al. (2025)](https://doi.org/10.1038/s41592-025-02621-6)
constructed a 150-dimensional potential-connectivity barcode based on axon--
dendrite overlap, defined connectivity subtypes in 31 brain regions, combined
that estimand with soma-distance affinity for spatially tuned clustering, and
described MOs and thalamocortical pathways. Its potential-connectivity barcode
and EP11's regional axon-allocation vector answer different measurement
questions; EP11 does not presume that either estimand is superior. It cannot
claim the first connectivity subtype, separated embedding, or MOs/thalamic
diversity result.

[Xiong et al. (2025)](https://doi.org/10.1038/s41592-025-02784-2) used a
near-identical 1,877-neuron resource to build probabilistic arbor- and bouton-
level connectomes. EP11 must distinguish regional axon allocation from
putative synaptic connectivity and connectome modularity.

EP11 must add four things:

1. an outcome-blind-frozen regional projection-distribution estimand and common
   observed-data likelihood, distinct from but not privileged over prior
   connectivity or morphology estimands;
2. a fair comparison with qualified nonlinear spatial and continuous-latent
   descriptions, including curved and unevenly sampled continua;
3. development-defined group signatures applied without re-clustering,
   rematching, or prevalence recalibration to held-out animals; and
4. a named target-allocation rule whose biological interpretation changes
   under the grouped versus continuous answer.

For shared cells or brains, EP11 is a reanalysis, not independent replication;
shared acquisition, reconstruction versions, or derived-feature pipelines also
constitute shared-source evidence. The lineage dossier must quantify all five
levels before novelty language is finalized.

## Current source-support verdict

The source files are acquired only in mixed-role steward quarantine. Existing
inventory reports 1,876 reconstructed morphologies, 39 `fMOST Brain ID` values,
92 soma-region labels, 1--225 cells per brain ID, and cortical-layer labels for
511 cells. These are not source-specific animal counts.

The metadata lacks an authenticated `fMOST Brain ID` to animal/specimen map and
the shared development/audit ledger. It also contains the forbidden
axon-derived `Projection class`. Therefore no source population is currently
known to satisfy the 12-development/8-audit requirement, and EP11 is not yet
qualified for cross-animal adjudication. A trusted redacted build and lineage
resolution come first.

## Study route

### 1. Select one sufficiently sampled source population

The first deliverable is a source-by-animal-by-position/depth coverage table,
constructed without projection outcomes. For every candidate source it shows:

- verified independent animals; conservative unresolved specimen groups are
  shown separately and cannot count toward the cross-animal minima;
- eligible and excluded cells per animal;
- soma-coordinate, depth, and layer coverage within each animal;
- overlap of those distributions between development and audit roles;
- sex, age, strain/genotype or Cre line, labeling strategy, modality,
  hemisphere, laboratory, batch, reconstruction version, registration, and
  completeness composition; and
- whether positions likely to contain intermediate cells were actually
  sampled.

Source selection uses this table and outcome-blind reconstruction/registration
QC. Here “target observability” cannot include observed regional target mass.
It does not use an embedding, cluster count, target contrast, or grouped-model
score. Whole-release animal counts cannot substitute for counts inside the
selected source and common-support domain. The 12 development and 8 audit
independent animals are necessary but not sufficient: outcome-blind work first
freezes minimum effective cells per animal, animals per neighborhood/layer,
mutual-support and maximum-gap rules, missing/truncation tolerance, and the
required precision. Every proposed label also needs multi-animal development
support. A valid census with no passing source yields
`closed_insufficient_source_support`, not a biological continuum result and not
pooling unrelated sources.

### 2. Describe position and projection allocation

Before real regional outcomes, freeze one quantitative regional projection
distribution on a named target vocabulary using atlas, measurement, and QC
rules. Also freeze the common likelihood for structural zeros and positive
values, transform/Jacobian, and marginalization of missing or truncated
coordinates. Show how expected allocation changes over soma coordinates and
depth using `P`, the position-driven continuous model, relative to adjustment-
only `B`.

The biological readout is not merely `P - B`. It is a map and uncertainty band
for named target proportions or contrasts over the shared spatial domain. The
report identifies where a target preference changes, where the data have
support, and where any interpolation would be extrapolation.

### 3. Test additional organization among nearby cells

Use `C` to ask whether a one- or two-dimensional connected latent continuum
accounts for residual variation among cells at similar positions. Then compare
`H` with `C` to ask whether adding fixed reusable groups improves cross-animal
prediction. `D` and `H` determine whether supported groups retain continuous
within-group organization, using paired `K`, labels, and group-signature rules
rather than two independently interpreted clusterings.

Local evidence must come from overlapping soma-position neighborhoods. A
pattern that separates animals, batches, layers with no overlap, or distinct
sampled territories is not a reusable projection division.

All models share a qualified hierarchical animal nuisance structure. Prediction
for a new animal integrates that effect from the development distribution
without using its other outcomes; uncertainty is animal-clustered. Simulations
and the fitted-continuum null include animal-level shifts and correlated
within-animal residuals so that such dependence cannot masquerade as neuron
groups.

### 4. Validate a concrete target-allocation rule

Development selects a small, multiplicity-controlled allocation contrast from
the already frozen outcome. Before audit, lock:

- the source and layer scope;
- the downstream targets and allocation scale;
- the predicted position relationship;
- if applicable, the fixed group signatures and label rule;
- the common-support domain and smallest meaningful effect; and
- the animal-level uncertainty and exclusion rules.

The audit applies the complete `B/P/C/D/H` comparison and this rule directly to
all eligible held-out animals. It reports target distributions by animal and
position. Posterior group membership is post-score recurrence/separation
evidence and cannot act as a predictor, refit, or relabel the model.

## Candidate dossier required before audit

For each development-selected candidate, record:

| Item | Required content |
| --- | --- |
| Source | Exact atlas region, layer/depth scope, verified independent-animal count, and common-support domain. |
| Targets | Named downstream regions, laterality, terminal-arbor rule, and quantitative allocation contrast. |
| Spatial organization | Frozen coordinate/depth relationship and local-overlap evidence. |
| Competing descriptions | Predictions from `P`, `C`, `D`, and `H` on the same outcome and cells. |
| Prior overlap | Closest Peng, Yufeng Liu, Lijuan Liu, and Xiong result; shared cells, animals/brains, acquisition, reconstruction, or features; and the unresolved point. |
| Decisive audit result | What would favor reusable groups, a continuum, or an unresolved conclusion. |
| Knowledge change | The anatomical statement that becomes more accurate than the existing source-average or subtype description. |

MOs, VPM, VP, or another Liu-analyzed source is eligible only if the dossier
states which reported allocation is being retested and what new adjudication
EP11 supplies. Repeating its clustering is not sufficient.

## Main comparisons

| Contrast | Paper question | Required interpretation |
| --- | --- | --- |
| `P - B` | How much organization follows measured soma position and depth? | A concrete topographic allocation, with held-out uncertainty and common-support boundaries. |
| `C - P` | Do nearby cells retain continuous heterogeneity? | A low-dimensional residual trajectory, not a finite-type claim. |
| `H - C` | Do reusable groups add stable cross-animal information? | The primary grouped-versus-continuous test under a common score. |
| `H - D` | Does continuity remain within groups? | A boundary between group-dominant and hybrid organization. |

All log scores refer to the same fixed outcome, coordinate measure, cells,
folds, and weights. Topology or provider-arbor sensitivities are shown
separately and cannot select the paper route. `C_ref` and `H_ref` are selected
independently by absolute development predictive score under equal search
budgets, with frozen complexity/hash tie breaks; the primary contrast is
`H_ref - C_ref`, not an `H/C` pair selected to make the continuum perform
poorly. Incremental `P - B`, `C - P`, and `H - D` results use
`B_pair_for_P_ref`, `P_pair_for_C_ref`, and `D_pair_for_H_ref` carried by the selected larger model rather
than unrelated best-family fits. The complete selection is replayed in every
full-search null.

## Complete-result routes

### Route A — Spatially overlapping reusable projection divisions

The animal-level lower confidence bound for `H - C` exceeds the frozen
meaningful margin in audit, the full-search null passes, position-standardized
fixed signatures recur across animals, and the named target allocation is not
decoded by experimental strata, batch, QC, or disjoint position. The paper may
conclude that the source contains cross-animal reusable projection patterns
beyond the tested continuous organization. `H - D` can then support internal
gradients, exclude them within the meaningful margin, or remain unresolved;
all three are complete grouped outcomes.

### Route B — Continuous organization is adequate at the tested resolution

The named target allocation transfers, `P/C` passes absolute calibration and
trajectory checks, simulations show sensitivity to meaningful groups, and the
audit upper bound on `H - C` lies below the frozen margin. The paper may
conclude that the tested finite groups add less than that margin beyond a
qualified spatial or latent continuum. It may say that a continuum accounts
for apparent modes only when those modes and absolute fit were separately
documented.

This is a full scientific result, not a failed candidate. A nonsignificant
group contrast alone is insufficient; Route B needs precision and synthetic
sensitivity.

### Route C — Unresolved or infeasible

Wide intervals, inadequate local overlap, model ambiguity, failure to reproduce
the named allocation, or insufficient source-specific animals prevents either
claim. The report identifies whether the limiting factor is sampling,
observability, continuous-model adequacy, or cross-animal instability.

## Proposed figure sequence

1. **Source and support.** Animal-by-position/depth coverage, exclusions,
   outcome-blind reconstruction/registration observability, and the frozen
   common-support domain.
2. **Continuous topography.** Named target allocations over soma position,
   with `B/P/C` predictions and held-out animal uncertainty.
3. **Nearby-cell alternatives.** The same cells under connected-continuum and
   fixed-group descriptions, including sampling density rather than only a
   colored embedding.
4. **Direct held-out transfer.** Frozen target and group predictions in each
   audit animal, with no rematching, plus the primary `H - C` interval.
5. **Mechanism boundaries and falsifiers.** Within-group gradients, technical
   decodability, influence, curved-continuum simulations, and separate outcome
   sensitivities.

Audit outcomes cannot choose the displayed source, targets, direction, spatial
region, or example cells. Example-selection and multiplicity rules are frozen
in development, while quantitative panels report every eligible audit animal.

## Claim boundary

The grouped route supports reusable projection patterns, not molecular,
functional, or developmental cell types. The continuous route supports a
specific topographic or latent description within a frozen source, outcome,
and resolution, not the universal absence of types. Neither route establishes
synaptic connectivity or independence from prior analyses reusing the same
cells.
