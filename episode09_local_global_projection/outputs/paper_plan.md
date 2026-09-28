# EP09 paper plan

## Working title

**Local dendritic structure foreshadows distal axonal target families in
single neurons**

The title is provisional. “Foreshadows” is appropriate only if the locked
same-cell audit succeeds; it does not imply development, causation, or synaptic
connectivity.

## The paper in one sentence

Among neurons matched for source anatomy and reconstruction quality, test
whether each cell’s native dendritic tree improves prediction of its own distal
axonal-arbor targets, then ask whether the useful dendritic scale depends on an
independently defined target family and repeats in new brains.

## Why this could matter

A neuron receives much of its input through dendrites and sends output through
its axon. If local dendritic organization reliably anticipates distant axonal
target families, the two compartments are not merely independent descriptions
of the same cell. They carry a reproducible, cell-wide organizational relation.

That broad idea is not new. Whole-brain morphology work already describes
dendritic diversity, distal arbors, and projection motifs in SEU-A1876
([Peng et al., 2024](https://doi.org/10.1038/s41467-024-54745-6)). Work in
mouse prefrontal cortex has already found dendrite–projection associations and
exceptions
([Gao et al., 2023](https://doi.org/10.1038/s41593-023-01339-y)). A dendritic
microenvironment atlas also reports correspondence with long-range projection
organization
([Muñoz-Castañeda et al., 2025](https://doi.org/10.1038/s41593-025-02119-6)).

The paper therefore cannot end with “a model using dendrites predicts axons.”
Its contribution must be the stricter decomposition:

> How much is already explained by where the neuron starts and how well it was
> reconstructed; how much belongs to that neuron’s own dendritic tree; and does
> the remaining information follow a reproducible dendritic-scale × target-family
> rule?

## Competing explanations

### H1 — Structured cell-wide organization

Native dendrites improve held-out distal-target prediction beyond context and
matched nuisance features. Particular dendritic scales contribute to particular
atlas-defined target families, the dendritic increment under the true family
hierarchy beats matched shuffled hierarchies, and the pattern repeats in sealed
audit brains.

### H2 — Residual context or measurement quality

Dendritic features act as complicated proxies for soma position, source,
layer, cell size, batch, or reconstruction quality. The apparent gain fails
the capacity-matched nuisance, common-pair, leakage, calibration, or group
influence gates.

### H3 — Shared morphology-class organization

The overall and scale-by-family results repeat, but a context-residualized
dendrite from another cell in the same brain predicts as well as the correct
cell’s dendrite. The relation is organized at a shared morphology-class level,
not shown to be unique to the individual neuron.

### H4 — Diffuse dendritic information

The own-cell dendritic tree improves prediction, but the useful information is
distributed or unstable. The overall effect replicates while the prespecified
scale-by-family pattern or true-ontology advantage does not.

### H5 — Context is sufficient at the tested scale

The audit interval excludes the frozen scientifically meaningful dendritic
gain, and synthetic tests show that the design could have detected such a gain.

### H6 — The data cannot decide

Biological grouping, coverage, target support, probability calibration, or
sensitivity is inadequate. This is unresolved, not evidence for H5.

## Data and prospective split

The primary source is SEU-A1876. For each eligible cell, the analysis joins a
native dendritic reconstruction, source and soma context, technical-quality
descriptors, and CCFv3 distal axonal-arbor observations through a verified cell
ledger.

Whole biological groups—not neurons—are assigned to roles before any
dendrite–axon association is inspected:

- at least 12 groups for development, model comparison, and hypothesis freeze;
- at least 8 different groups for one sealed audit.

EP09 shares possible axonal outcomes with EP10 and EP11. Their common cell,
group, role, and exposure ledger must be frozen before any of those episodes
opens a shared audit outcome.

Each cell–target state is `detected`, `not_detected`, or `uncertain`. A negative
requires evidence that the target was observable; `uncertain` remains missing.
The endpoint is distal-arbor detection in the reconstruction, not a synapse or
functional connection.

## Analysis 1 — Does the neuron’s own dendrite add information?

### Paired models

- **M0, context:** source region, layer, soma location, batch, and acquisition or
  dendrite quality known independently of axonal geometry.
- **M1, context + native dendrites:** exactly the same context plus the cell’s
  native dendritic topology and geometry.
- **N0, context + false dendrites:** exactly the same context plus a frozen
  bank of dimension- and missingness-matched dendritic permutations or nuisance
  blocks.

M0 and M1 use the same cells, targets, outer grouped folds, preprocessing,
learners, tuning budget, calibrators, and adaptive-search exposure. Development
selects one M1 pipeline. Each of 20 false-dendrite members is then refit with
that locked pipeline and calibrator type, without a separate adaptive search.
This tests capacity honestly without claiming an unbudgeted 20-fold search.

### Primary outcome

For each biological group, compute Brier error by first averaging across
observable cells within each frozen target, then weighting targets equally.
Compare M0 and M1 within the group and finally weight groups equally:

\[
D_g=B_g(M0)-B_g(M1), \qquad
\Delta_{\mathrm{Brier}}=\operatorname{mean}_g D_g.
\]

Positive values favor native dendrites. The audit success rule requires the
small-group uncertainty interval to clear a frozen scientific margin
`delta_morph`, M1 to beat the simultaneous nuisance envelope, and every
leave-one-group summary to remain positive.

The anatomy-defined candidate vocabulary and the scored target set are
different. A scored target must have at least a frozen
`n_coverage >= 3` observable cells in every development and audit group, at
least four development groups with a detection, and at least four with a valid
nondetection. Audit observability may establish coverage, but audit detection
states remain sealed. Within each outer development fold, target prevalence is
recomputed from training groups alone using a three-positive/three-negative
minimum. Every candidate in that fold uses the same target set.

For nuisance member (j),

\[
R_{gj}=B_g(N_j)-B_g(M1).
\]

The 20 nuisance blocks come from outcome-free conditional location-and-scale
models of each dendritic block given M0 context. Standardized residual rows are
deranged independently for topology, radial, path-geometry, orientation, and
topological-summary blocks, then transformed to the recipient’s context. This
preserves within-block structure but breaks the cross-block coherence of a real
tree. M1 must have a positive simultaneous lower bound against every member.
Conditional covariance, heteroscedasticity, multimodality, missingness, overlap,
and joint support must pass development diagnostics; otherwise the nuisance gate
is unresolved.

The `0.001` search-patience value is not `delta_morph`. The scientific margin is
chosen from development-only simulation and domain interpretation, not from a
convenient audit result.

With only eight audit groups, use separate one-sided 95% bounds

\[
L_D=\bar D-c_Ls_D/\sqrt G,
\qquad
U_D=\bar D+c_Us_D/\sqrt G.
\]

The lower-tail statistic is \((\bar D-\mu)/(s_D/\sqrt G)\); the upper-tail
statistic reverses its sign. Separate \(c_L\) and \(c_U\) are necessary under
skew. Each is at least the ordinary one-sided Student-\(t\) value. For every
frozen normal, skewed, heavy-tailed, and heteroscedastic scenario, 50,000
calibration replicates set a one-sided 99% upper confidence bound for the
corresponding 95th percentile. A disjoint 50,000-replicate validation set must
then give a 99% Clopper–Pearson lower coverage bound of at least 0.95 for each
tail. Scenario parameters, counts, and both seed sets freeze in advance; the
same separation governs nuisance, hierarchy, and cell-identity critical values.
Simulations cannot certify the unknown audit population. The procedure also assumes
independent biological groups, finite group variance, a fixed target set and
model, and an equal-group target population. Removing one group may change the
mean by no more than `delta_morph / 2`.

### What this analysis can establish

If successful, Analysis 1 shows that native dendrites contain reproducible
incremental information. It does not yet establish which biological
organization produced that information. That requires Analyses 2 and 3.

## Analysis 2 — Which dendritic scale carries information for which targets?

Dendrites are divided into a small number of outcome-blind, interpretable
blocks, such as soma-centered physical-radius shells and branch-order ranges.
Targets are grouped by one outcome-independent, non-overlapping anatomical cut
of the Allen ontology. The scored set must contain at least three families and
three targets per family.

The adaptive primary winner may omit a block, so scale interpretation uses a
separate fixed pair. E0 contains context only; E1 is a ridge-regularized
multi-target logistic model with the common basis for every prespecified
dendritic block. E1 cannot replace the primary winner, and a structured claim
requires its own audit dendritic increment to have a positive lower bound.

For family \(f\), define \(B_{gf}\) by averaging cell errors within target and
then targets equally within the frozen family. For block \(b\), family \(f\),
and biological group \(g\), measure the
loss after replacing that block with a dimension-matched, within-group and
context-matched false block:

\[
L_{bfg}=B_{gf}(E1^{[b]})-B_{gf}(E1).
\]

Development data freeze the expected sign and ordering of the complete
block-by-family matrix and one matrix-replication statistic. Audit data test
that prediction once. The paper does not select the most attractive audit cell
or relabel the families after seeing the heat map.

Replacement is not a post-fit audit permutation. Inside each development outer
split, learn an outcome-free conditional law for the selected block given E0
context and the remaining dendritic blocks. For each of 10 frozen draws,
replace the block in both training/calibration and held-out data, rerun
preprocessing, refit E1 and its calibrator, and average the 10 scores.
The audit replacement law is learned on full development only. Failed
conditional-generator diagnostics make that block result unresolved.

Development freezes a robust scale for every matrix entry and the subset of
entry signs that clear a simultaneous development threshold. The template and
each audit-group matrix are vectorized in a fixed order, divided by those
development scales, and compared by cosine similarity. A structured result
requires: template and every audit matrix norm above `eta_matrix`; an
equal-group cosine lower bound above `rho_matrix`; every required sign repeated;
and a positive fixed-E1 dendritic increment. Undefined or low-norm matrices and
failed generators make the explanation unavailable, not biologically diffuse.

Nested development pseudo-audits repeat the full primary search, generator
fitting, E0/E1 fitting, template learning, scale estimation, sign selection, and
decision. Calibration and validation pseudo-audits use disjoint groups or seed
sets, so the error rate includes template uncertainty and winner selection.

The interpretable result is not simply “proximal branches mattered most.” It is
a conditional rule, for example:

> Near-soma branching carries most of the incremental information for one
> independently defined target family, whereas outer dendritic extent carries
> information for another, and this signed relationship repeats across new
> brains.

The actual pattern is learned and frozen in development; the example is not a
claim about the unseen data.

## Analysis 3 — Is the apparent organization anatomically real and cell specific?

### True target families versus matched false hierarchies

Extend fixed E1 with family-specific dendritic deviations. E0 is fit once and
its predictions are identical for every hierarchy; family labels affect only
E1’s dendritic deviation terms. The penalty grid and nested selection procedure
are identical under the true anatomy and 20 frozen shuffled families. For
hierarchy \(h\), its dendritic increment is

\[
H_g(h)=B_g(E0)-B_g(E1_h).
\]

For each shuffle \(j\), test

\[
Q_{gj}=H_g(\mathrm{true})-H_g(\mathrm{shuffle}_j).
\]

The shuffles preserve family size and fall within frozen bins for development
prevalence, observation rate, and CCFv3 source-to-target centroid distance. All
20 simultaneous lower bounds must exceed zero. If 20 valid matched shuffles
cannot be constructed, the hierarchy explanation is unresolved.

Because the comparison is a difference in dendritic increments, true families
cannot win merely because context-only base rates already follow the ontology.

### Correct dendrite versus a matched wrong dendrite

Development dendrites alone define a stable morphology-class partition; audit
cells use its frozen assignment rule. Within brain and morphology class, give
each cell a no-fixed-point donor’s **entire** standardized dendritic residual
vector while retaining the recipient’s context-specific location, scale, and
missingness. Unlike the nuisance bank, this preserves an intact, coherent
dendrite while breaking cell identity. Conditional overlap, covariance,
missingness, and joint support must pass, with at least four donors per
group-class.

Let \(C_g=B_g(E1_{swap})-B_g(E1_{correct})\). A cell-specific claim requires
the one-sided 95% lower bound to exceed frozen `delta_cell`. A shared
morphology-class claim uses two one-sided 5% tests: the 90% interval formed with
separately calibrated and validated \(c_{C,L}\) and \(c_{C,U}\) must lie wholly
inside `[-delta_cell, +delta_cell]`. If a valid interval establishes neither
superiority nor equivalence, the organization may still be structured but cell
specificity remains unresolved. If donors or diagnostics are unavailable, the
identity explanation itself is unavailable; those are different outcomes.

## Analysis 4 — Validity, explanation, and scope

Four kinds of evidence have different consequences:

- **Common design-validity gates:** verified identity and observability, common
  prediction pairs, no axonal leakage, valid and available nuisance construction
  and diagnostics, group-level influence, probability support, calibration, and
  synthetic recovery. These are required for both a positive result and an
  informative negative result.
- **Positive-only nuisance gate:** M1 must beat all 20 nuisance members for a
  positive incremental-information claim. Directional superiority is not needed
  to conclude context sufficiency when the upper bound excludes the scientific
  margin, although the nuisance construction must still be valid and available.
- **Explanation discriminators:** the scale-by-family replication,
  hierarchy-by-dendrite interaction, and same-cell swap. Failure narrows a
  successful primary result; it does not rewrite that score.
- **Scope sensitivities:** rotation-invariant morphology, native versus CCFv3
  dendrites, arbor threshold, registration/clipping strata, and source-family
  heterogeneity. Disagreement limits coordinate, measurement, or population
  scope; it is not by itself a proxy result.

Axon-derived coverage and completeness may define observability or a
sensitivity stratum but never enter M0 because even target-blind summaries can
reveal projection extent or breadth. Size/QC comparisons use dendrite size and axon-independent
acquisition or reconstruction quality.

A favorable source subgroup cannot rescue a failed overall primary endpoint.

## Statistical principles

- The biological group supplies the degrees of freedom. Cell–target pairs do
  not create independent brains.
- The primary group interval uses a frozen small-sample procedure whose
  critical value is chosen or falsified—not generally certified—by
  development-only simulations.
- Target prevalence support is recomputed inside each outer training fold.
- Target, family, block, and nuisance-bank definitions are frozen before audit.
- Family, nuisance, and hierarchy inference use prespecified simultaneous
  procedures.
- Every coverage- and development-supported target enters; there is no
  favorable-target analysis posing as confirmation.
- Calibration receives the same training data and tuning opportunity across
  M0 and M1. Controls refit the locked calibrator type. Isotonic calibration is
  prohibited; shared Platt calibration requires at least eight training groups
  and each state in at least four groups.
- The audit is opened once and cannot change the model, target set, margin,
  explanatory prediction, or paper conclusion rule.

## Main figures

### Figure 1 — The question and the decisive predictions

Use the synthetic conceptual schematic
[`ep09_conceptual_question.png`](ep09_conceptual_question.png).

- **A:** Context-matched neurons with different local dendrites and distal
  target-family outcomes.
- **B:** The paired M0 versus M1 comparison, with the nuisance bank beside it.
- **C:** Prespecified dendritic scales linked to a fixed atlas family tree and
  a development-frozen scale-by-family matrix; show the matched shuffled
  hierarchy interaction as the alternative.
- **D:** One audit in new brains, separating structured replication, overall
  gain only after a valid contradiction, unavailable explanation, context
  sufficiency only when the upper bound excludes the margin under a valid and
  sensitive design, invalid evidence, and unresolved support.

Scientific judgment: what result would distinguish a cell-wide organizational
rule from prediction, proxy, or insufficient evidence?

### Figure 2 — What was actually observed

- Verified cell-to-dendrite-to-axon joins and biological-group structure.
- Source, layer, soma-location, batch, and QC balance across roles.
- Target observability, detections, nondetections, and uncertainty by group.
- Frozen target vocabulary and ontology families.
- Coordinate, clipping, registration, and reconstruction completeness results.

Scientific judgment: are detections and valid negatives comparable across
brains, or could missingness manufacture the question?

### Figure 3 — Native dendrites versus anatomical context

- Groupwise M0 and M1 Brier scores joined within brain.
- Groupwise native-dendrite gains with the scientific margin and interval.
- Real-dendrite gain versus the frozen nuisance envelope.
- Reliability and calibration plots using identically supported pairs.
- Leave-one-group influence and prespecified source-family scope analysis.

Scientific judgment: do a cell’s own dendrites add reproducible information
beyond context and modeling capacity?

### Figure 4 — The scale-by-target-family prediction

- Development block-by-family contribution matrix with uncertainty.
- Development-scaled cosine template, minimum-norm requirement, and frozen
  required-sign conjunction beside the audit matrix.
- Groupwise similarity between each audit matrix and the development template.
- Contribution curves for the prespecified dendritic scale blocks.

Scientific judgment: is the predictive information organized in the way
development data predicted, or is it diffuse and unstable?

### Figure 5 — Anatomy and cell identity as competing explanations

- True versus matched shuffled hierarchy effects on the dendritic increment,
  not raw model score.
- Correct-cell dendrites versus matched wrong-cell swaps.
- Native, rotation-invariant, size/QC-only, and CCFv3 sensitivity results.
- Registration, clipping, missingness, and arbor-threshold sensitivities.

Scientific judgment: is the result organized by real target anatomy, and does it
belong to the individual cell or only to a shared morphology class?

### Figure 6 — What the result changes

Use a compact decision panel rather than another score summary:

| Primary gain | Family pattern | Cell swap | Validity | Supported conclusion |
| --- | --- | --- | --- | --- |
| Pass | Pass | Correct cell clears `delta_cell` | Pass | Structured, cell-specific local-to-global organization |
| Pass | Pass | Equivalence is established | Pass | Shared morphology-class organization |
| Pass | Pass | Valid interval shows neither superiority nor equivalence | Pass | Structured organization; cell specificity unresolved |
| Pass | Valid test contradicts prediction | Any | Pass | Diffuse dendritic information only |
| Pass | Test unavailable | Unresolved | Pass | Primary increment established; explanation unresolved |
| Apparent pass | Any | Any | Fail | No valid incremental-information claim |
| Meaningful gain excluded | Not needed | Not needed | Common design valid and sensitivity adequate; nuisance direction not required | Context sufficient at the tested scale |
| Inconclusive | Inconclusive | Inconclusive | Inadequate | Unresolved |

Scientific judgment: which statement does the complete evidence permit, and
which attractive statements remain unsupported?

## Supplementary analyses

- complete feature and learner comparison from grouped development folds;
- exact nuisance-generation and matched-hierarchy procedures;
- target-by-target estimates for all frozen targets, without target selection;
- simultaneous source- and target-family intervals;
- full synthetic calibration and power results;
- dendritic block boundary and atlas-level sensitivities selected before audit;
- parser, coordinate, topology, and lineage validation; and
- complete audit decision trace showing that no outcome changed the contract.

## Narrative under each possible result

### If the full structured prediction succeeds

Lead with the reproduced scale-by-family organization, then show that the
overall predictive gain, true hierarchy, and same-cell identity controls all
support it. Architecture search remains a methods detail.

### If only the primary gain succeeds and explanation tests are valid

Lead with the narrower result: native dendrites add information, but the study
did not recover a stable scale or target-family rule. Do not turn a new audit
pattern into a confirmed mechanism.

If a generator, matrix norm, donor set, or matched hierarchy is unavailable,
report “primary increment established; explanation unresolved” instead. Missing
explanatory evidence is not evidence for diffuse biology.

### If the family pattern succeeds and swap equivalence is established

Lead with the narrower organized result: morphology predicts target-family
structure at a shared class or neighborhood level, but the study does not show
that a neuron’s particular dendrite is uniquely informative.

If the family pattern succeeds but a valid swap interval establishes neither
superiority nor equivalence, report “structured organization; cell specificity
unresolved.” Do not force that interval into either the individual-cell or
shared-class conclusion.

### If controls explain the gain

Lead with the falsification: a flexible dendritic representation can appear
predictive because it recovers location, quality, or group context. This is a
useful caution about whole-neuron morphology analyses, not a failed attempt to
be hidden.

### If the meaningful effect is excluded

Report the bound and synthetic sensitivity. The conclusion is that source
anatomy and measurement quality are adequate at the prespecified scale in this
dataset, not that dendrites and axons are biologically unrelated.

### If support is inadequate

Call the result unresolved. Do not equate a wide interval or missing targets
with an informative null.

## Claim boundary

Even the strongest result concerns reconstructed distal axonal-arbor detection
within the eligible SEU-A1876 populations. It does not establish synapses,
functional communication, causal wiring, developmental mechanism, or a
universal neuron taxonomy. External data would be needed for broader
generalization.

## Work required before analysis

1. Provision and authenticate the source archives and exact CCFv3 resources.
2. Verify animal/brain/specimen provenance and create the shared EP09/10/11
   role and exposure ledger.
3. Freeze the arbor, observability, distal, target, and anatomical family-cut
   rules.
4. Freeze `n_coverage`, three-per-state outer-fold support, four-per-state audit
   support, and validate exact eligible group, target, and family counts.
5. Freeze the conditional location/scale and support diagnostics, 20
   independent-block nuisance seeds, dendrite-only morphology classes, intact
   swap seeds, 10 block-replacement draws, and 20 matched target hierarchies.
6. Freeze E0/E1, `delta_morph`, `delta_cell`, `eta_matrix`, `rho_matrix`, matrix
   scaling and required signs, the tail-specific `c_L`, `c_U`, `c_C_L`, and
   `c_C_U` values, and all simultaneous procedures.
7. Freeze simulation scenarios, counts, and separate 50,000-replicate
   calibration and validation seed sets; require their coverage rules to pass.
8. Complete the focused novelty review.
9. Verify that the complete primary, fixed-pipeline, nested pseudo-audit, and
   explanatory refit workload fits the resource
   ceiling.
10. Freeze the complete analysis, generate audit predictions, and open the audit
   once.
