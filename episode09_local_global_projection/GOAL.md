# At what biological resolution does dendrite–projectome coupling live?

[Scope and novelty review, 2026-10-02](outputs/scope_novelty_review_20261002.md).

## The question in plain language

Two neurons can belong to the same cortical IT population, sit in the same source
region and layer, and look equally well reconstructed, yet send axons to different
sets of exact brain targets.

EP09 asks whether that remaining target variation belongs only to anatomical context
or a shared cell class, or whether it is coupled to the particular neuron's own
dendrite. Within held-out biological groups from SEU-A1876, the primary comparison
gives each recipient either its own native dendritic representation or a coherent
dendritic representation from a tightly matched peer. The axonal target vector always
stays with the recipient.

A context-only model and the conventional native-dendrite-versus-context comparison
remain required benchmarks. They are not the novelty claim. The paper-level question
is whether the correct dendrite beats matched wrong dendrites for **exact distal
targets**, after source, layer, independently defined non-axon class, soma location,
quality, and biological group have been held fixed.

## What would make the result scientifically useful?

A small M1-over-M0 improvement would establish only that dendrites add information, a
question close prior work has already addressed. A paper-level result must establish
the following hierarchy:

1. The correct dendrite beats coherent matched-donor dendrites by a frozen
   `delta_cell` margin in held-out biological groups.
2. That advantage is present for exact atlas targets rather than only after pooling
   them into broad families.
3. The correct-minus-matched-donor advantage has a reproducible dendritic-scale ×
   exact-target structure.
4. The result survives context, quality, capacity, donor-overlap, calibration, and
   group-influence controls.

The intended contribution is therefore a biological-resolution adjudication:
anatomical context versus shared class versus within-class cell-level residual
coupling. Even the strongest result cannot prove unique neuronal identity, because an
unmeasured finer subtype could still explain the residual association.

## Why this is not already answered

[Peng et al. (2021)](https://doi.org/10.1038/s41586-021-03941-1)
reported a scoped tension that EP09 can adjudicate: within cortical L2/3, L4,
and L5 IT populations, individual target-subset selection appeared unrelated to soma
depth or dendritic morphology. This was a descriptive observation, not a general
theorem of random wiring.

The broad relationship is no longer open. [Gao et al.
(2023)](https://doi.org/10.1038/s41593-023-01339-y) established dendrite–axon
correspondence and important exceptions in complete single neurons. [Liu et al.
(2026; online 2025)](https://doi.org/10.1038/s41593-025-02119-6) linked dendritic
microenvironments to long-range organization. Most decisively, [Sorensen et al.
(2026)](https://doi.org/10.1038/s41586-026-10424-8) directly connected local
morphology, cortical position, multimodal identity, and specific projection targets.
Therefore, a basic M1-over-M0 result is a benchmark or confirmation, not the new
discovery.

The source lineage also limits the rhetoric. [Liu et al.
(2024)](https://doi.org/10.1038/s41467-024-54745-6) report that 1,736 of the
1,876 SEU-A1876 cells are refinements of previously released neurons and only 140 are
newly annotated. EP09 is predominantly a prospective adjudicative reanalysis of the
Peng-2021 data lineage. Holding out biological groups supports internal grouped
reproducibility; it is not independent biological replication.

## Explanations that the experiment must separate

| Explanation | What we should observe | What would count against it |
|---|---|---|
| **Within-class cell-level residual coupling** | The correct dendrite beats coherent matched-donor dendrites for exact targets, and the scale × exact-target advantage repeats across held-out biological groups. | Matched donors predict equally well, or the exact-target structure does not recur. |
| **Residual context or measurement proxy** | Apparent dendritic value disappears when position, source, layer, total size, batch, and reconstruction quality are matched or when nuisance features receive the same modeling opportunity. | The own-cell dendritic advantage survives all matched controls and is not concentrated in one group. |
| **Shared class or neighborhood organization** | Native dendrites beat context, but a tightly matched peer's coherent dendritic residual predicts about as well as the correct cell's. | The correct-cell dendrite has a reproducible advantage over matched donors. |
| **Benchmark-only dendritic information** | Native dendrites improve over context, but neither correct-over-swap nor a stable exact-target structure is established. | Correct-over-swap and the frozen exact-target template both pass. |
| **Context is sufficient** | The interval rules out the frozen meaningful M1-over-M0 gain and synthetic checks show the benchmark could have detected it. | Native dendrites clear the frozen benchmark margin in held-out groups. |

These are prospective alternatives. A benchmark result cannot rescue a failed or
unavailable correct-versus-matched-donor primary. A positive primary supports
within-class cell-level residual coupling, not unique neuronal identity.

## Data and unit of analysis

The primary resource is the SEU-A1876 collection of whole-neuron reconstructions in
Allen CCFv3 space. The decisive population is cortical L2/3, L4, and L5 IT
neurons **only if** IT membership is available from genetic, experimental, or source
metadata whose lineage does not use the cell's axonal target vector. The provider
`Projection class` field is axon-derived and cannot define eligibility. If an
outcome-independent IT definition and its matched-donor support do not exist, the
headline primary is not executable and the episode must stop or downgrade to a broad
benchmark study.

Each eligible neuron contributes:

- one native dendritic reconstruction near the soma;
- source-region, cortical-layer, soma-location, batch, and dendrite/acquisition-quality
  descriptors that do not depend on axonal geometry;
- a distal axonal-arbor outcome for each frozen target region;
- a biological group identifier, preferably animal or brain and otherwise the most
  conservative documented specimen unit; and
- outcome-independent IT-label provenance plus original-release versus newly
  annotated source lineage.

The basic observation is a **cell–target pair**, but splitting and uncertainty are at
the biological-group level. Neurons from one brain may never be divided between
development and audit merely to increase sample size.

For each cell and target, the outcome is:

- `detected`: the frozen arbor rule finds qualifying distal axon in the target;
- `not_detected`: the target is observable and no qualifying arbor is found;
- `uncertain`: coverage or reconstruction quality does not support either conclusion.

`uncertain` is missing, never a negative label. The endpoint is reconstructed distal
arbor detection, not synaptic connectivity, transmission strength, or causal influence.

## The primary biological test

Every eligible cortical IT cell and every frozen exact target enters the comparison.
There is no post-hoc selection of targets with favorable results. Broad target
families are explanatory summaries, not the primary outcome resolution.

### M0: anatomical context only

M0 predicts distal target detections using information available without seeing the
cell’s dendritic tree:

- source region and cortical layer;
- independently defined non-axon class;
- soma coordinates and prespecified spatial transforms;
- batch or specimen descriptors;
- acquisition and dendrite-reconstruction quality known without measuring axonal
  extent, target breadth, or target detections.

Axon-derived coverage or completeness may decide whether a target is observable and
may stratify a sensitivity analysis. It never enters M0 because even a target-blind
axon summary can reveal projection extent or breadth.

### M1-correct: the same context plus the neuron's own dendrite

M1-correct receives every M0 variable plus outcome-blind summaries of the native dendritic
tree. Candidate summaries may cover topology, Sholl-like radial profiles, path length,
branch order, tortuosity, orientation, and prespecified topological descriptors. All
feature definitions, transformations, and missingness rules are learned or frozen
using development data only.

M0 and M1-correct must use the same cells, targets, outer folds, prediction family, tuning
opportunity, calibration opportunity, and adaptive-search exposure. The comparison is
about the information source, not unequal model capacity.

### M1-swap: the same fitted model plus a coherent matched-donor dendrite

Development selects and fits the M1 pipeline using correct development cell–dendrite
pairings. For primary evaluation, the outcome model and calibrator remain fixed. Each
held-out recipient is evaluated repeatedly after replacing its entire commonly
observed standardized dendritic residual representation with that of a prespecified
matched donor while retaining the recipient's context, exact target vector, and
missingness support.

Donors are matched within biological group × source × layer × independently defined
non-axon IT class, with frozen calipers on soma position and axon-independent quality.
A stable development-only dendritic neighborhood may be added to tighten matching but
cannot use axonal information. Multiple frozen no-fixed-point donor plans are averaged
so the result cannot depend on one convenient derangement. No outcome model is refit
on swapped training pairings for this primary comparison.

Because the residual is transported into the recipient's context and recipient
missingness is reapplied, this comparator is called a **coherent matched-donor
dendritic residual**, not a literal untouched native dendrite. Failed overlap,
covariance, missingness, or donor-count diagnostics make the primary unavailable.
Matching is never loosened after outcomes are inspected.

### N0: supporting capacity-matched nuisance dendrites

A larger feature block can improve prediction even when its entries are biologically
meaningless. N0 is therefore a bank of 20 deterministic false-dendrite blocks, not one
convenient random baseline. Inside each outer split, an outcome-free conditional
location-and-scale model is fit separately for each prespecified dendritic block using
M0 context. It produces standardized multivariate residual rows for topology, radial,
path-geometry, orientation, and topological-summary blocks.

For each seed, residual rows are deranged without fixed points **independently for each
block** within a biological group, then transformed back using the recipient cell’s
context-specific location and scale. This preserves within-block dimension and
covariance while breaking the cross-block coherence of a real dendritic tree. Training,
calibration, and held-out groups are generated separately, and outcomes are never used.
Training-only imputation supplies a donor value before the recipient’s original
missingness mask is reapplied.

The development search selects one M1-correct representation, learner, calibrator, and set of
hyperparameters. Each nuisance member then receives that locked pipeline and is refit
on its own false features; nuisance members do not run 20 additional searches. Thus
they match the chosen model’s capacity without pretending to have search exposure that
the resource budget does not provide.

For nuisance member \(j\), define

\[
R_{gj}=B_g(N_j)-B_g(M1_{\mathrm{correct}}).
\]

Positive values favor the real dendrite. M1-correct must have a positive simultaneous lower
bound against every frozen nuisance member. The 20 generation seeds and conditional
model freeze after development morphology validation. Diagnostics must show overlap
and balance across source, layer, soma position, batch, and axon-independent QC, and
must test conditional covariance, heteroscedasticity, multimodality, missingness, and
joint support. Every participating group needs at least four eligible cells. Failed
diagnostics or inadequate group size make the nuisance gate unresolved rather than
triggering outcome-dependent cell removal. If the complete nuisance workload cannot
fit within the resource ceiling, the episode is not ready to launch.

## Exactly how the primary score is computed

First define an anatomy-based candidate vocabulary without reading detections. Then
freeze an integer \(n_{\mathrm{coverage}}\geq3\) using outcome-blind coverage counts and
development-only precision simulations. The audit score set \(T\) is the intersection
of candidate targets that:

- have at least \(n_{\mathrm{coverage}}\) observable cells in every assigned
  development and audit group; and
- have detections in at least four independent development groups and valid
  nondetections in at least four independent development groups.

Audit coverage may determine whether a pair is observable but cannot reveal whether it
is detected. If any frozen target later has fewer than \(n_{\mathrm{coverage}}\) valid
pairs because of a data error, the run is technically invalid; the target is not
dropped and the endpoint is not renormalized.

Within biological group \(g\), model \(m\)'s Brier score is calculated in three steps:

1. average squared probability error across observable cells for each target;
2. average those target scores with equal target weight;
3. compare models within the same group.

In notation,

\[
B_g(m)=\frac{1}{|T|}\sum_{t\in T}
\frac{1}{n_{gt}}\sum_{i\in O_{gt}}
\left(y_{git}-\hat p_{git}^{(m)}\right)^2,
\]

where \(O_{gt}\) contains only observable cell–target pairs and
\(n_{gt}=|O_{gt}|\geq n_{\mathrm{coverage}}\). For target family \(f\), define the
corresponding score

\[
B_{gf}(m)=\frac{1}{|T_f|}\sum_{t\in T_f}
\frac{1}{n_{gt}}\sum_{i\in O_{gt}}
\left(y_{git}-\hat p_{git}^{(m)}\right)^2,
\]

where \(T_f\) is the frozen subset of scored targets in family \(f\). This family score
is secondary. The exact-target score used to build the explanatory matrix is

\[
B_{gt}(m)=\frac{1}{n_{gt}}\sum_{i\in O_{gt}}
\left(y_{git}-\hat p_{git}^{(m)}\right)^2.
\]

For frozen matched-donor plan \(j\), the group-level correct-cell advantage is

\[
C_{gj}=B_g(M1_{\mathrm{swap},j})-B_g(M1_{\mathrm{correct}}),
\qquad
C_g=\frac{1}{J}\sum_{j=1}^{J}C_{gj}.
\]

Positive values favor the recipient's correct dendrite. The headline primary estimand
gives every biological group equal weight:

\[
\Delta_{\mathrm{cell}}=\frac{1}{G}\sum_g C_g.
\]

The conventional group-level native-dendrite gain remains a required supporting
benchmark:

\[
D_g=B_g(M0)-B_g(M1_{\mathrm{correct}}),
\qquad
\Delta_{\mathrm{morph}}=\frac{1}{G}\sum_gD_g.
\]

This order prevents a large brain, a common target, or a target with unusually complete
coverage from dominating either comparison. The same recipient cells, targets, and
observable pairs must be used for M0, M1-correct, and every M1-swap plan.

For development fold \(k\), \(T_k\) contains only candidate targets with the frozen
coverage in every group participating in that fold and at least three positive and
three negative independent outer-training groups. All candidates in that fold are
compared on the same \(T_k\), and the development objective weights outer folds
equally. The held-out fold never decides its own prevalence eligibility. Requiring four
positive and four negative groups in full development ensures that a final target can
retain three of each when any one development group is held out. The final audit set
\(T\) is frozen from full development prevalence plus outcome-blind audit coverage;
audit detections remain sealed.

## What counts as a positive primary result

The primary success rule is frozen before any shared audit outcome is opened. It
requires all of the following:

1. the group-level one-sided lower bound for \(\Delta_{\mathrm{cell}}\) exceeds the
   positive scientific margin `delta_cell`;
2. the supporting M1-correct-versus-M0 lower bound exceeds `delta_morph`;
3. M1-correct beats the simultaneous capacity-matched nuisance envelope;
4. every correct-versus-swap and benchmark leave-one-group summary is positive and no
   single group controls the conclusion;
5. donor balance, joint support, calibration, probability support, and common-pair
   requirements pass; and
6. synthetic tests show that the locked procedure can recover effects at both frozen
   margins under the available group structure.

M1-correct beating M0 without the correct-over-swap result is benchmark evidence only.
If the donor pools or diagnostics are unavailable, the headline primary is unresolved;
the matching rules are not relaxed and the benchmark cannot be relabelled as a positive
primary.

Because the audit may contain only eight biological groups, inference must be designed
for small \(G\). Let \(\bar C\) and \(s_C\) be the equal-group mean and standard
deviation of \(C_g\). Use separate one-sided 95% critical values for the lower and
upper primary bounds:

\[
L_C=\bar C-c_{C,L}s_C/\sqrt G,
\qquad
U_C=\bar C+c_{C,U}s_C/\sqrt G.
\]

The supporting \(D_g\) benchmark receives separately calibrated \(c_{D,L}\) and
\(c_{D,U}\) values; it does not borrow the more favorable tail from the primary.

For scenario mean \(\mu_s\), the signed errors are

\[
Z_{C,L}=(\bar C-\mu_s)/(s_C/\sqrt G),
\qquad
Z_{C,U}=(\mu_s-\bar C)/(s_C/\sqrt G).
\]

Each critical value is the larger of the ordinary one-sided Student-\(t\) value and a
one-sided 99% upper confidence bound for the corresponding 95th percentile of
\(Z_{C,L}\) or \(Z_{C,U}\), estimated from 50,000 calibration replicates per frozen
scenario. The benchmark uses the same construction with \(D_g\) and its own tails. This
separation matters when group contrasts are skewed. A disjoint set of 50,000 validation
replicates then evaluates both values; the 99% Clopper–Pearson lower bound on each tail’s
coverage must be at least 0.95 in every scenario. Scenario parameters, replicate
counts, and calibration and validation seeds freeze before either simulation set is
run.

The scenarios preserve outcome-blind group sizes and support and cover normal, skewed,
heavy-tailed, and heteroscedastic contrasts. They choose, validate, or falsify the
critical value; they do not certify the unknown audit population or substitute
simulated data for audit contrasts. A failed independent validation requires more audit
groups or a different prelaunch procedure and leaves the question unresolved.

The interval assumes verified independent biological groups, a fixed target set and
model, finite group variance, an equal-group target population, and a scenario envelope
broad enough to include development-observed heterogeneity. Every leave-one-group mean
must be positive. Removing any group may change \(\bar C\) by at most
`delta_cell / 2` and \(\bar D\) by at most `delta_morph / 2`. A neuron-level interval that treats cells from the same brain as
independent is not acceptable. Nuisance and hierarchy max-\(t\) critical values use the
same separated calibration/validation design, with Bonferroni-\(t\) as the conservative
fallback.

Four different quantities must not be confused:

- `0.001` is an optimization patience threshold used during development;
- `delta_cell` is the minimum scientifically meaningful correct-over-swap primary
  advantage;
- `delta_morph` is the minimum supporting M1-correct-over-M0 benchmark improvement;
- `0.002` is a secondary scope threshold for source-family heterogeneity, not a
  primary success margin.

Both scientific margins and the exact source-family definition and support remain
unresolved launch blockers until development-only feasibility work justifies and
freezes them.

## The deeper prediction: a scale × exact-target correct-cell advantage

The explanatory analysis asks whether the useful signal has a reproducible structure.
The adaptive primary winner may omit a dendritic block, so it is not used to declare
that an omitted scale is biologically unimportant. Instead, one separate explanatory
pair is frozen before audit:

- **E0:** the same context-only inputs as M0;
- **E1:** a ridge-regularized multi-target logistic model containing the common,
  prespecified basis for **every** dendritic scale block.

E1 is not eligible to replace the primary winner. Its form and penalty grid are fixed,
and its penalty is selected inside grouped development folds. A structured claim first
requires a positive group-level lower bound for \(B_g(E0)-B_g(E1)\).

Before outcomes are inspected, dendrites are divided into biologically interpretable
blocks such as soma-centered radial shells and branch-order ranges. Numeric boundaries
are frozen only after coordinate and sampling validation. The decisive matrix retains
each exact non-overlapping CCFv3 target as a separate column. One outcome-independent
anatomical family cut is retained only for prespecified aggregation and hierarchy
falsification.

For dendritic block \(b\), exact target \(t\), and group \(g\), define

\[
S_{btg}=\frac{1}{J}\sum_j
\left[B_{gt}(E1^{[b\leftarrow\mathrm{donor}\ j]})
-B_{gt}(E1_{\mathrm{correct}})\right],
\]

where the replacement block comes from the same frozen matched-donor bank used by the
primary, with recipient context and missingness retained. E1 is trained and calibrated
on correct development pairings; at held-out evaluation only block \(b\) is replaced.
The donor identity is coherent across blocks within a plan, rather than being redrawn
independently to manufacture a scale effect.

Diagnostics cover conditional means, covariance, heteroscedasticity, multimodality,
missingness, and joint support. Failure makes that block's explanation unavailable. A
post-hoc target or donor choice is prohibited. Positive \(S_{btg}\) means that the
recipient's own block improves prediction for exact target \(t\) relative to a matched
donor block; it does not establish causal importance.

Development freezes the all-block basis, exact-target entry order, anatomical family
summary cut, and a robust scale for every entry. It also freezes a set of required sign entries whose
development evidence clears a simultaneous threshold. The development template and
each audit-group matrix are vectorized in that order and divided by the development
scales. Matrix similarity is the cosine between the two vectors, which avoids rank ties.

The structured scale-by-exact-target prediction succeeds only if all of the following hold:

1. the development template norm clears a positive `eta_matrix` threshold;
2. every audit-group matrix clears the frozen positive `eta_matrix` norm threshold;
3. the equal-group lower bound on cosine similarity exceeds `rho_matrix`;
4. every required sign entry repeats under the frozen simultaneous procedure; and
5. the fixed E1 dendritic increment is positive in audit groups.

An undefined cosine, inadequate matrix norm, or failed generator makes the explanation
unavailable, not evidence for diffuse biology. `eta_matrix`, `rho_matrix`, required
signs, scaling, and multiplicity freeze before audit.

Development pseudo-audits must be nested: each one repeats the complete primary search
and freeze, donor-bank construction, explanatory-model fitting, template
estimation, sign selection, scaling, and decision. Calibration and validation
pseudo-audits use disjoint groups or disjoint simulation seeds. This carries template
uncertainty and winner selection into the error rate instead of comparing audit data to
a template estimated once from all development groups.

## Secondary anatomical falsification and primary interpretation

### Does real target organization beat an arbitrary hierarchy?

A family-deviation extension of fixed E1 is used only for this explanation. E0 is fit
once and produces identical predictions for every hierarchy. Hierarchy labels affect
only E1’s family-specific dendritic deviations; the penalty grid, nested selection rule,
and fitting opportunity are identical for the true anatomy and 20 frozen matched
shuffled hierarchies. For hierarchy \(h\), define its dendritic increment

\[
H_g(h)=B_g(E0)-B_g(E1_h).
\]

For shuffle \(j\), the interaction is

\[
Q_{gj}=H_g(\mathrm{true})-H_g(\mathrm{shuffle}_j).
\]

The true anatomy must have a positive simultaneous lower bound against every shuffled
hierarchy. Thus it cannot win merely because target base rates or context-only
predictions follow the ontology.

Shuffles must preserve family size within frozen bins for development prevalence,
observation rate, and CCFv3 centroid distance from the source region. The tolerances,
centroid-distance bins, and 20 assignments freeze before audit. If 20 valid matched
assignments or the minimum family support cannot be constructed, the hierarchy
explanation is unresolved rather than relaxed.

### How the correct-versus-donor primary is interpreted

The primary deliberately preserves a coherent donor residual morphology, unlike the
independent-block nuisance bank. The frozen class rule, donor plans, minimum donor
count, common analysable set, and joint-support diagnostics are therefore part of the
primary contract rather than a downstream explanation.

A within-class cell-level residual-coupling claim requires the one-sided 95% lower
bound for \(C_g\) to exceed `delta_cell`. A shared class or neighborhood interpretation
uses two one-sided 5% tests: its 90% equivalence interval

\[
[\bar C-c_{C,L}s_C/\sqrt G,\;\bar C+c_{C,U}s_C/\sqrt G]
\]

must lie wholly inside `[-delta_cell, +delta_cell]`. The two primary critical
values are calibrated and independently validated for the lower and upper signed tails
under the same frozen 50,000-plus-50,000 simulation design. Failure to prove either
superiority or equivalence leaves the primary biological resolution unresolved. Even
superiority is not evidence of unique neuronal identity: unmeasured finer subtype
structure remains a possible explanation.

## Validity gates, explanation tests, and scope sensitivities

These categories must not be interpreted as interchangeable.

**Common design-validity gates** are verified identity and observability, no outcome
leakage, the same prediction pairs across models, a valid and available nuisance
construction, group-level inference and influence, and adequate probability support
and calibration. These are required for both positive and informative-negative claims.

**The directional nuisance-superiority gate**—M1-correct beating all 20 nuisance members—is
required only for a positive incremental-information claim. A true zero dendritic effect
need not beat nuisance features for context sufficiency to be interpretable; the
nuisance generator and its diagnostics must simply be valid and available.

**Primary biological-resolution evidence** is the correct-versus-matched-donor
comparison. **Explanation tests** are the scale-by-exact-target replication and the
secondary true-versus-shuffled family-hierarchy increment. They determine whether a
successful primary effect has stable anatomical structure; they do not retroactively
change the primary score. A valid executed test that contradicts the prediction is
scientific evidence, whereas a test that cannot be constructed or validated is
unavailable.

**Scope sensitivities** include soma-centered rotation-invariant morphology, native
versus CCFv3 dendrites, the frozen arbor-threshold sensitivity set, and source-family
heterogeneity. Disagreement narrows the coordinate, measurement, or population scope;
it is not by itself proof of a proxy.

Source families are an outcome-independent, non-overlapping cut of the Allen source
ontology with a frozen minimum number of independent groups. The 0.002 regression rule
is a secondary scope flag reported with simultaneous intervals. It cannot rescue a
failed overall test, and it does not veto a successful overall estimand.

All tuning, feature screening, imputation, scaling, calibration, target eligibility,
and ensemble weighting occur inside grouped development folds. Isotonic calibration is
not allowed. One shared Platt calibrator is eligible only when at least eight training
groups are available and each outcome state occurs in at least four independent
training groups; otherwise all compared models use no calibration.

## Development, freeze, and audit

At least 12 biological groups are reserved for development and at least 8 for the
sealed audit, subject to provenance validation. The same role ledger is shared with
EP10 and EP11 because all three episodes use related axonal outcomes. All three role
manifests, outcome rules, target vocabularies, and primary contracts must be frozen
before any shared audit outcome is opened.

Development may compare a bounded grammar of dendritic representations, transforms,
learners, calibrators, and at most a two-member ensemble. The planned budget is 40–96
valid trials, with 20 initial coverage trials and patience 18 at the optimization-only
threshold of 0.001. Resource ceilings remain 1,500 CPU-hours, 120 wall-clock hours,
64 CPU cores, and 1 TB scratch, with no GPU request.

After freeze, the audit is run once. Failed implementation or data-integrity checks may
invalidate a run, but an inconvenient scientific result is not a reason to reopen the
audit or change the model.

## How the paper’s figures answer the question

![EP09: own versus matched-donor dendrite at fixed recipient targets](outputs/ep09_conceptual_question-v3.png)

The conceptual figure carries three scientific questions:

1. **The biological contrast:** context-matched cortical IT neurons can have different
   dendrites and exact distal target vectors.
2. **The primary test:** the correct dendrite must beat coherent matched-donor
   dendrites in held-out biological groups within SEU-A1876.
3. **The explanation:** does a prespecified scale × exact-target
   correct-minus-donor pattern recur across biological groups? The separate
   M1-correct-versus-M0/nuisance benchmarks remain required in the study.

This is a scientific schematic, not an empirical result. Panel A's matched
neurons and target identities are illustrative. In panel B, both copies show
the same recipient's axon and target vector—not the donor's outputs. Only the
dendritic residual changes. Colors, target slots and scale bands encode no
observations or frozen scale definitions. Panel C asks about within-source
group reproducibility, not cross-species transfer. The full interpretation
rules remain in the table below; equivalence does not identify a shared-class
or neighborhood cause. [Imagegen prompt set](outputs/ep09_conceptual_question-v3-prompt.md).

## Possible conclusions

| Result | Conclusion | What happens next |
|---|---|---|
| Correct-over-donor clears `delta_cell`, M1-correct clears its benchmark, and the scale × exact-target template repeats. | **Within-class cell-level residual coupling with reproducible exact-target structure.** | Seek genuinely independent data and finer non-axon class measurements. |
| Correct-over-donor clears `delta_cell` and the benchmark passes, but the valid exact-target template contradicts the prediction. | **Within-class residual coupling without an established scale × target rule.** | Preserve the narrower primary result; treat new structure as future work. |
| M1-correct beats context, but correct-versus-donor equivalence is established. | **No meaningful own-dendrite advantage at the tested scale; consistent with shared organization, without identifying its biological cause.** | Report the benchmark and equivalence bound; do not present either as identification of a shared class or neighborhood mechanism. |
| M1-correct beats context, while correct-versus-donor establishes neither superiority nor equivalence. | **Benchmark confirmed; biological resolution unresolved.** | Increase independent donor-supported groups or stop. |
| Donor support or diagnostics fail before audit. | **Headline primary unavailable.** | Do not loosen matching or substitute M1-over-M0 after seeing outcomes. |
| A positive result fails a common validity gate or the directional nuisance-superiority gate. | **No valid positive incremental-information claim.** | Repair a technical failure prospectively or close the proxy explanation; do not rescue it with a selected subgroup. |
| The upper interval rules out `delta_morph`, every common design-validity gate passes, and synthetic sensitivity is adequate. | **Context is sufficient at the prespecified scale.** | Report an informative null and the detectable bound; nuisance superiority is not required. |
| Coverage, group count, calibration, or sensitivity is inadequate. | **Unresolved.** | Repair the measurement or narrow the target vocabulary using development evidence only; do not interpret absence of significance as absence of biology. |

Scope sensitivity is a modifier, not a competing terminal result. Coordinate,
threshold, warping, or source-family dependence narrows whichever positive conclusion
is otherwise supported.

## What this episode may claim

If every gate passes, EP09 may claim that within independently defined cortical IT
strata and held-out SEU-A1876 biological groups, a neuron's own dendritic residual
predicts its exact reconstructed distal target vector better than coherent
matched-donor dendrites, with a reproducible scale × exact-target organization.

It may not claim synaptic connectivity, causal routing, physiological communication,
developmental mechanism, universal cell type, unique neuronal identity, or independent
replication. It also may not claim novelty from M1-over-M0 prediction alone. The
scientific value rests on separating context, shared class, and within-class
cell-level residual coupling.

## What must be resolved before launch

- verify biological-group provenance and the minimum development/audit counts;
- verify an outcome-independent cortical IT definition and document its lineage;
- document original-release versus newly annotated source lineage;
- verify source × layer × non-axon class × biological-group donor support under the
  frozen soma-position and quality calipers;
- verify exact-target observation and positive/negative support within that IT cohort;
- confirm native dendritic and distal axonal coordinates are in compatible CCFv3 space;
- freeze the arbor-detection rule, observability rule, target vocabulary, and target
  family cut;
- freeze \(n_{\mathrm{coverage}}\), prevalence support, family support, and the
  outcome-independent target and source ontology cuts;
- freeze dendritic scale blocks; conditional location, scale, imputation, and support
  diagnostics; the 20 independent-block nuisance seeds; the coherent matched-donor
  plans; and the 20 valid matched target hierarchies;
- freeze E0/E1, `delta_cell`, `delta_morph`, `eta_matrix`, `rho_matrix`, the
  exact-target entry order and required signs, matrix scaling, and the tail-specific
  primary and benchmark critical values,
  and all equivalence and simultaneous procedures;
- justify `delta_cell`, `delta_morph`, and all small-group and max-\(t\) critical values with separated
  50,000-replicate calibration and validation simulations whose scenarios and seeds
  were frozen in advance;
- confirm that the complete nuisance and explanatory workload fits the resource ceiling;
- complete the focused novelty check;
- freeze the shared EP09/EP10/EP11 role and audit ledger before any protected outcome
  is opened.

Until these items are resolved, the episode is scientifically specified but not ready
for its one-time audit.
