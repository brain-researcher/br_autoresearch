# EP09 paper plan

## Working title

**Within-class dendritic residuals predict exact long-range target vectors in
cortical IT neurons**

The title is earned only if the correct dendrite beats coherent matched-donor
dendritic residuals in held-out biological groups. “Within-class” refers to the
finest outcome-independent strata available here; it does not establish unique
neuronal identity.

## The paper in one sentence

Among independently defined cortical IT neurons matched for biological group,
source, layer, non-axon class, soma position, and quality, test whether the
correct cell's dendrite predicts its exact distal target vector better than
coherent dendritic residuals from matched peers.

## Why this is a paper-level question

The broad question “do dendrites predict axons?” is already occupied.
[Gao et al. 2023](https://doi.org/10.1038/s41593-023-01339-y) established
dendrite–axon correspondence and exceptions in complete single neurons.
[Liu et al. 2026 (online 2025)](https://doi.org/10.1038/s41593-025-02119-6)
linked dendritic microenvironments to long-range organization.
[Sorensen et al. 2026](https://doi.org/10.1038/s41586-026-10424-8)
directly connected local morphology, cortical position, multimodal identity,
and specific projection targets. Therefore, M1 beating M0 is a required
benchmark, not the discovery.

The sharper tension comes from [Peng et al.
2021](https://doi.org/10.1038/s41586-021-03941-1), who reported that within
cortical L2/3, L4, and L5 IT populations, individual target-subset selection
appeared unrelated to soma depth or dendritic morphology. EP09 asks at what
resolution coupling remains after known context and class structure are held
fixed:

> anatomical context, shared class or neighborhood, or within-class cell-level
> residual coupling?

## Source and rhetoric boundary

SEU-A1876 contains 1,876 morphologies. [Liu et al.
2024](https://doi.org/10.1038/s41467-024-54745-6) report that 1,736 are
refinements of previously released neurons and 140 were newly annotated.
Consequently, this is predominantly a prospective adjudicative reanalysis of
the Peng-2021 data lineage.

Whole-group development and audit roles can establish internal grouped
reproducibility within SEU-A1876. They do not establish independent biological
replication. A lineage-disjoint analysis of newly annotated cells is optional
only if those cells independently satisfy the group, exact-target, and donor
support rules.

## The biological population

The intended primary population is cortical L2/3, L4, and L5 IT neurons whose
IT membership comes from genetic, experimental, or source metadata independent
of the cell's axonal target vector. The provider `Projection class` field is
axon-derived and cannot define eligibility.

This is a real feasibility gate. If an outcome-independent IT definition or
the required donor pools do not exist, EP09 does not silently broaden the
cohort or derive class from the outcome. It stops or becomes a benchmark study
without the proposed headline claim.

## Competing explanations

### H1 — Within-class cell-level residual coupling

The correct dendrite beats coherent matched-donor residuals for exact target
vectors, and the correct-minus-donor scale × exact-target template repeats
across held-out biological groups.

### H2 — Shared class or neighborhood organization

M1-correct beats context, but it is equivalent to matched donor dendrites.
This is consistent with reusable class or neighborhood-level information,
without a demonstrated same-cell residual. Equivalence does not identify its
biological mechanism.

### H3 — Benchmark-only dendritic information

M1-correct beats M0, but the correct-versus-donor interval establishes neither
superiority nor equivalence. This confirms broad predictability without
resolving its biological scale. If the primary instead succeeds but the
exact-target template fails a valid test, retain the narrower within-class
residual-coupling result without the proposed structural rule.

### H4 — Context or measurement proxy

The apparent effect fails matching, nuisance, common-pair, calibration,
overlap, or group-influence requirements.

### H5 — Context is sufficient at the tested scale

The benchmark upper bound excludes `delta_morph`, and the locked procedure
has adequate synthetic sensitivity at that margin.

### H6 — Unresolved

Outcome-independent IT labels, independent groups, exact targets, donors,
probability support, or precision are inadequate. Missing support is not a
negative biological result.

## Analysis 1 — Correct dendrite versus matched donor: primary

Development selects and fits one M1 pipeline on correct cell–dendrite
pairings. During held-out prediction, the fitted outcome model and calibrator
remain fixed.

For each recipient, compare:

- **M1-correct:** recipient context plus its own dendritic representation.
- **M1-swap,j:** the identical fitted model after transferring one frozen
  matched peer's complete standardized dendritic residual vector into the
  recipient context and reapplying recipient missingness.

Donors are matched within biological group × source × layer × independently
defined non-axon IT class, with frozen calipers on soma position and
axon-independent quality. A development-frozen dendrite-only neighborhood may
tighten matching. It cannot substitute for any required stratum.

Multiple no-fixed-point donor plans are frozen from development-only
qualification. No audit result may select a donor, loosen a caliper, remove a
recipient, or switch matching schemes.

For group \(g\) and donor plan \(j\):

\[
C_{gj}=B_g(M1_{\mathrm{swap},j})-B_g(M1_{\mathrm{correct}}),
\qquad
C_g=\operatorname{mean}_j C_{gj}.
\]

The primary estimand is

\[
\Delta_{\mathrm{cell}}=\operatorname{mean}_g C_g.
\]

The positive primary claim requires its one-sided lower bound to exceed the
frozen `delta_cell` margin, the required benchmark and nuisance superiority
to pass, acceptable group influence, and every matching, support, calibration,
and common-pair gate. Sensitivity must be adequate at both scientific margins.

Because the donor vector is transported into recipient context, the paper calls
it a **coherent matched-donor dendritic residual**, not a literal untouched
native tree.

## Analysis 2 — Native dendrite versus context: required benchmark

- **M0:** source, layer, independently defined non-axon class, soma location,
  batch, and axon-independent acquisition or dendrite quality.
- **M1-correct:** exactly the same context plus native dendritic blocks.
- **N0:** a frozen bank of independent-block false dendrites used to check
  unequal feature capacity.

For each biological group:

\[
D_g=B_g(M0)-B_g(M1_{\mathrm{correct}}).
\]

The benchmark lower bound must exceed `delta_morph`, M1-correct must beat the
simultaneous nuisance envelope, and the result must survive influence,
calibration, and synthetic-sensitivity checks. This benchmark is necessary for
a positive primary interpretation but is not itself the novelty claim.

## Analysis 3 — Which scales couple to which exact targets?

Fixed E0/E1 models are used only for interpretation:

- **E0:** context only.
- **E1:** context plus every prespecified dendritic scale block.

The decisive matrix retains every exact supported target as a column. For
dendritic block \(b\), exact target \(t\), group \(g\), and frozen donor plan
\(j\):

\[
S_{btg}=
\operatorname{mean}_j\left[
B_{gt}(E1^{[b\leftarrow\mathrm{donor}\ j]})
-B_{gt}(E1_{\mathrm{correct}})
\right].
\]

Development freezes the block basis, exact-target order, robust entry scales,
required signs, minimum matrix norm, and cosine-similarity margin. Audit groups
test that complete template once. Broad target families may summarize the
matrix but cannot replace its exact-target resolution.

E1 and its calibrator are fitted on correct development pairings and remain
fixed for donor substitutions. The donor bank is the same as for the primary;
within each plan, donor identity is coherent across blocks. A structural claim
also requires a positive group-level E1-over-E0 increment, a sufficiently
nonzero template in development and every audit group, the frozen similarity
margin, and simultaneous replication of the required signs.

A positive \(S_{btg}\) is predictive information relative to a matched donor
block. It is not a causal claim about dendritic growth or routing.

## Analysis 4 — Secondary anatomical falsification

One outcome-independent Allen ontology cut provides a readable target-family
summary. Its E1 dendritic increment is compared with 20 matched shuffled
hierarchies that preserve family size and frozen bins for development
prevalence, observation rate, and source-to-target distance.

This test asks whether the exact-target result aligns with meaningful anatomy.
It is secondary: a broad-family result cannot rescue failed exact-target or
correct-over-donor evidence.

## Outcome and score

Each exact cell–target observation is `detected`, `not_detected`, or
`uncertain`. A valid negative requires evidence that the target was
observable. `Uncertain` is missing, never zero.

Within each group, Brier error is averaged over observable cells within exact
target and then equally over targets. Biological groups receive equal weight.
Cells and cell–target pairs do not create extra independent brains.

All target eligibility, tuning, preprocessing, calibration, donor matching,
and template learning occur inside grouped development data. Audit detections
are opened once by the locked evaluator.

## Main figures

### Figure 1 — The biological-resolution question

![EP09 own-versus-donor concept](ep09_conceptual_question-v3.png)

This image-generated schematic shows illustrative matched neurons, fixed-
recipient dendritic substitution and a scale × exact-target question. Panel B
keeps the recipient's target vector in both copies; it never transfers the
donor's axon or outputs. Target identities, colors, slots and scale bands are
not observed data or frozen scale definitions. There is no verdict panel:
equivalence alone cannot identify a biological cause.
[Exact generation/correction prompts](ep09_conceptual_question-v3-prompt.md).

- context-matched cortical IT neurons with different exact target vectors;
- correct dendrite versus coherent matched donor as the primary contrast;
- M1-correct versus M0 retained as a supporting study benchmark;
- correct-minus-donor scale × exact-target structure from development to
  held-out biological groups; and
- the complete decision table below governs interpretation, including no
  meaningful own-dendrite advantage without an identified biological cause.

### Figure 2 — Is the primary cohort actually identifiable?

- outcome-independent IT-label provenance;
- original-release versus newly annotated lineage;
- biological-group counts and role assignment;
- exact-target observability and positive/negative support; and
- matched-donor pool sizes, caliper balance, overlap, and exclusions determined
  before outcome access.

### Figure 3 — Correct versus matched donor

- groupwise \(C_g\) estimates and the `delta_cell` margin;
- all frozen donor-plan results;
- leave-one-group influence;
- donor balance and common-pair diagnostics; and
- probability calibration.

### Figure 4 — Required benchmark

- groupwise M0 and M1-correct Brier scores;
- \(D_g\) and the `delta_morph` margin;
- M1-correct versus the nuisance envelope; and
- synthetic sensitivity at both scientific margins.

### Figure 5 — Scale × exact target

- frozen development template and audit-group matrices;
- groupwise template similarity;
- simultaneous required-sign results;
- exact-target estimates without favorable-target selection; and
- secondary family aggregation and matched-hierarchy falsification.

### Figure 6 — What the evidence permits

| Correct > donor | Benchmark | Exact-target structure | Supported statement |
| --- | --- | --- | --- |
| Pass | Pass | Pass | Within-class cell-level residual coupling with reproducible exact-target structure |
| Pass | Pass | Contradicts | Within-class residual coupling without the proposed structural rule |
| Equivalent | Pass | Any | No meaningful own-dendrite advantage at the tested scale; consistent with shared organization but not identifying its cause |
| Inconclusive | Pass | Any | Benchmark confirmed; biological resolution unresolved |
| Unavailable | Any | Any | Headline primary unavailable; matching is not relaxed |
| Any | Fail validity | Any | No valid positive claim |
| Not needed | Meaningful benchmark gain excluded with adequate sensitivity | Not needed | Context sufficient at the tested scale |

## Claim boundary

The strongest result remains about reconstructed distal axonal-arbor detections
within eligible SEU-A1876 cortical IT populations. It does not establish
synapses, physiological communication, causal wiring, developmental mechanism,
unique neuronal identity, universal cell type, or independent replication.

## Work required before candidate scoring

1. Verify outcome-independent IT membership and its provenance.
2. Document original-release versus newly annotated lineage.
3. Verify biological-group identity and at least 12 development plus 8 audit
   groups.
4. Freeze exact-target observability and support within the IT cohort.
5. Verify donor support in every source × layer × class × group pool under the
   frozen position and quality calipers.
6. Freeze M0, M1, donor plans, nuisance bank, E0/E1, margins, exact-target
   template, and inference procedures.
7. Demonstrate synthetic recovery and acceptable precision at both scientific
   margins.
8. Confirm the complete workload fits the resource ceiling.
9. Freeze the shared EP09/10/EP11 role ledger before any shared audit outcome is
   opened.

Until the first five items pass, the proposed headline primary is a design, not
an executable study.
