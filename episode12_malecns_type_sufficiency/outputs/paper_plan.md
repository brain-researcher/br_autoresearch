# EP12 paper plan: when does a cell-type average hide separate pathways?

Status: proposed study design, 2026-09-24. No real-data result, external
replication, or working follow-up executor is claimed here.

## The intended discovery

The opening question remains: do neurons called the same type share the same
downstream wiring? The paper should then establish whether their differences
organize identifiable input-output pathways that a type average misrepresents.

A possible result would be that cells receiving mainly from one set of
partners preferentially connect to one downstream circuit, while other cells
of the same type connect a different input-output pair. Another possible
result is a continuous spatial organization. Both require a named circuit,
an explanation tested against alternatives, and evidence beyond the discovery
data. Neither result is assumed.

The primary T/U/M round in [GOAL.md](../GOAL.md) tests residual outgoing
structure. The proposed follow-up tests its circuit consequences. These are
different claims with different endpoints. A paper may contain both rounds;
their results, exposure histories and stopping decisions remain separate.

## Why this direction

| Direction | Assessment |
| --- | --- |
| Catalogue additional within-type clusters | Useful discovery screen, but heterogeneity and connectivity-based subdivisions already have close precedents |
| Declare a new cell-type taxonomy | Requires broader cross-specimen and multimodal evidence; a model split alone cannot establish a type |
| Test what averaging loses about a specific circuit | Preferred: gives the discovered variation a concrete structural consequence and a falsifiable prediction |

The proposed contribution is conditional: identify **where and why** a type
average fails, or bound where it is adequate for a defined structural task.
The mere fact that averaging can lose information is mathematical and cannot
be presented as the biological discovery.

## Closest prior work and the novelty question

| Prior work | Already established | What an EP12 candidate must add |
| --- | --- | --- |
| Schlegel et al., Nature 2024, [doi:10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5) | Cross-connectome typing; AOTU063 illustrates a morphology-defined group with two consistent connectivity subdivisions. The paper also discusses simplifying connectome graphs by cell type. | A circuit consequence beyond another split, with an explicit averaging comparator and independent specimen evaluation |
| Cornean et al., Nature Communications 2024, [doi:10.1038/s41467-024-45971-z](https://doi.org/10.1038/s41467-024-45971-z) | Tm9 input heterogeneity, related circuit motifs, and genetic synapse labeling across individuals | A specific new organization or generalization beyond known within-type heterogeneity; output groups alone are insufficient |
| Berg et al., Cell 2026, [doi:10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015), [MaleCNS project](https://male-cns.janelia.org/) | Source connectome and its published annotations and circuit analyses | Check each candidate against the source paper, supplements and existing group/instance distinctions before asserting novelty |

These examples are literature benchmarks, not a predetermined candidate
list. The comparison is a starting review, not an exhaustive novelty search.
After the outgoing result is fixed, record for each proposed case: current
type name, existing subtype assignments, relevant circuit papers, known
partners, proposed new statement, strongest rival explanation, and evidence
that would refute the statement. A known split may still support a new circuit
finding, but must be described as a known split.

## Study sequence

### 1. Establish the outgoing phenomenon

Run the existing whole-type 80/20 T/U/M design with reciprocal transfer,
the unchanged primary endpoint, 36–96 trials, and the required 99 complete
search-null reruns. Resolve its existing numerical decisions and synthetic
qualification before candidate-discriminating development. Do not add
incoming connectivity or external outcomes to that search.

Report all assigned final types and the aggregate result. Preserve valid
negative and unresolved outcomes. The current minimum bilateral eligibility
does not by itself establish power to distinguish two or three populations;
qualification must cover the actual sample sizes and missingness.

### 2. Develop one specific circuit hypothesis

After the outgoing conclusion is fixed, examine previously assigned
development types under a documented follow-up exposure plan. Start with one
focal type or a small anatomically defined family. Permit at most three
candidate cases; the ranking and tie rule must be fixed before incoming
profiles are inspected. Use outgoing reproducibility, annotation-based
support, anatomical interpretability and metadata-only external coverage.
Final-type outcomes cannot choose the cases. Report every selected case,
including failed ones; do not replace them after inspecting incoming data.

The first inexpensive diagnostic is whether outgoing organization predicts
incoming partner preferences beyond anatomy and continuous variation in
development data. Groups are defined from outgoing profiles only. Incoming
profiles are used to fit the explanatory association, never to redefine the
groups. Use qualified grouped or spatially blocked development comparisons
for tuning; they do not constitute new animal replication.

Inspect cell morphology and endpoint completeness for each candidate. A
spatially organized gradient is a possible explanation, not a technical
failure. A mismatch with existing provider groups is not sufficient novelty.
If the association disappears under fair anatomical/continuous controls, or
is already described with the same consequence, stop this paper direction
or state a narrower contribution. Do not compensate with more clusters.

### 3. Freeze and test the input-output prediction

Use a separate fixed-comparison round for the selected cases. Source
development chooses a complete procedure and its controls. Before opening
external connectivity, freeze the source identity, case list, anatomical
support, partner crosswalk, models, scores, meaningful margins, multiplicity,
uncertainty, resource limits, and report. This is not a continuation of the
outgoing adaptive search and does not inherit its null calibration or budget.

For each external neuron, use its outgoing profile and permitted anatomy to
predict its incoming partner composition. Compare the following source-fitted
models on identical cells, bins and weights:

| Model | What predicts incoming composition? | Explanation tested |
| --- | --- | --- |
| Type average | One source-type input template | Within-type differences add no useful association |
| Anatomy | Fixed spatial and anatomical covariates | Wiring association follows location or known organization |
| Continuous | Anatomy plus a source-fitted continuous outgoing coordinate | A gradient explains the association without discrete groups |
| Groups | Anatomy plus probabilities of source-fitted outgoing groups | Reusable groups add useful association beyond the references |

The groups model has a discrete structural interpretation only if supported
by the outgoing evidence. Its predictive improvement alone cannot establish
discrete populations. A continuous winner supports a continuous organization
only under its own fixed decision rule; it cannot replace the outgoing
round's primary hypothesis.

External outgoing profiles are allowed predictor inputs. Their source-fitted
transform and assignment rules are frozen; no target-side refit, clustering,
model selection or incoming-based component alignment is allowed. External
incoming profiles are withheld prediction targets. This is a distinct
readout in the same graph, not independent biological evidence by itself.

Use one external evaluation transaction containing the locked comparisons.
Select one suitable external specimen by coverage before its connectivity is
opened. FlyWire and hemibrain are candidates for assessment, not presently
selected evaluation sets. A male-to-female transfer cannot separate sex from
specimen and reconstruction differences. A new release of the same specimen
is a robustness check. If fresh compatible data are unavailable, retain the
within-male scope and do not claim external confirmation.

## Endpoints with a concrete interpretation

### Primary follow-up endpoint: predict the incoming partners

Let p_i be the incoming partner composition of neuron i, including explicit
unknown and unsupported mass, and let p_hat_i be a model prediction. Use
neuron-equal squared composition error, the Brier form:

    L = mean_i sum_a (p_i[a] - p_hat_i[a])^2

Lower is better. The groups contrast uses the best fair type, anatomy or
continuous reference selected in source development. The continuous contrast
uses the best fair type or anatomy reference. Report all comparisons. Give
directions and prespecified cases equal weight when aggregating, and show
every case. Do not select the reference on external scores. Discrete and
continuous follow-up claims require their separately fixed contrasts and
multiplicity rule; a positive contrast is not enough without the circuit
readout below.

Eligible cells must have nonzero, sufficiently observed incoming and outgoing
totals under a rule fixed before evaluation. A zero-total profile is
unscorable, not a zero composition. Report known and unknown-bin score
contributions without renormalizing the known subset. Improvement explained
only by unknown or technical bins cannot support a wiring-organization claim.

### Circuit readout: which inputs and outputs meet on the same cells?

Let q_i be the outgoing composition on the same neurons. With n equally
weighted neurons of one type and side, define:

    R_cell[a,b] = (1/n) sum_i p_i[a] q_i[b]
    R_type[a,b] = mean_i p_i[a] * mean_i q_i[b]
    R_model[a,b] = (1/n) sum_i p_hat_i[a] q_i[b]
    E_model = (1/2) sum_a,b |R_cell[a,b] - R_model[a,b]|

R_cell measures which upstream and downstream partner types co-occur at a
focal cell. Each R sums to one when endpoint mass is complete, including
unknown bins. R_type is a within-specimen descriptive averaging comparator;
it uses observed marginals and is not the source-trained prediction model.
Never choose a model or partners from its external residuals. R_model uses
the frozen predictions above, and E_model measures their structural error.

This score describes normalized input-output pair mass, not physiological
information flow, synaptic efficacy, transmission probability or behavioral
gating. Because neurons receive equal weight, it is not a raw synapse-count
two-hop path total either. Validate any claim about specific physical paths
against cell-level edges, coverage, compartments and the prespecified edge
rule; do not equate a weak or unobserved edge with a nonexistent path.

R_cell minus R_type is a covariance of input and output preferences. A nonzero
covariance is not, on its own, evidence for discrete groups or a novel
principle. Require a useful improvement beyond matched anatomy/continuous
references, a named partner-pair interpretation, robustness to technical
error, and transfer. Partner-pair examples must be chosen in development;
external evaluation reports their signed deviations and all locked cases.

## Decisive controls and failure interpretations

| Alternative explanation | Test | Consequence if it explains the result |
| --- | --- | --- |
| Generic averaging or overfitting | Qualified null pairing of whole outgoing profiles with incoming profiles within prespecified type/side/anatomy/quality strata; retain each profile and its mass | No evidence of a specific association beyond that reference |
| Continuous or spatial organization | Fair anatomy and continuous models on the same prediction target; report total and adjusted effects | Bound the claim to that organization; do not claim residual discrete groups |
| Known subtype annotation | Post-outgoing novelty comparison with provider groups, instances and published circuits | Report rediscovery unless the circuit consequence is independently new |
| Reconstruction or missing-mass artifact | Full endpoint accounting, matched quality, fixed detection sensitivity, cell-level inspection | Do not interpret an unsupported route as absent or biologically segregated |
| One influential cell or partner | Prespecified cell/partner influence analysis and full case reporting | Restrict or reject the case-level conclusion |
| Dataset, sex or mapping effect | Fixed homology/coverage manifest; no target-driven matching; report transfer failures | No general conserved-organization claim |

The pairing null is a profile-association control, not a physically valid
randomized whole connectome. Its strata need an exchangeability justification
and enough support; sparse strata or unmodelled anatomy can invalidate it.
Qualify it with synthetic independent, spatial-gradient, continuous-coupling,
grouped-coupling, degree/strength-imbalanced and missing-edge scenarios.
If the null cannot preserve the required covariates at the available sample
size, revise the comparison before external access; do not run an unrestricted
shuffle and call it anatomical control.

Freeze meaningful improvements for L and E, the number of null replicates,
joint decision rule, sensitivity and simultaneous intervals before external
outcomes. Calibrate for the complete source-side case/model selection.
Technical subsampling and within-specimen uncertainty describe measurement
or conditional stability, not between-animal uncertainty. Cell types, sides,
neurons and synapses do not increase the number of independent animals.

## When to deepen, narrow or stop

| Evidence | Next action and permitted claim |
| --- | --- |
| Stable outgoing groups but no input-output consequence | Publish/report the bounded outgoing result if useful; the proposed pathway story is unsupported |
| A continuous or anatomical model explains the association | Pursue that specific organization under the follow-up rule; retain the original M result |
| A new, interpretable association survives controls and transfers | Develop the structural circuit claim, reporting the exact types, partners and specimens |
| Type averaging is adequate with useful precision for both follow-up endpoints | State a bounded adequacy result; outgoing adequacy alone cannot establish this |
| Precision, novelty, support or external correspondence is insufficient | Report the limitation, narrow or stop; do not force a paper-level conclusion |

Before running the follow-up, its fixed contract must also specify compute
limits, per-case support, handling of invalid predictions, qualified null
count, stop events and result labels. These are unresolved design values, not
permission to inherit the outgoing settings. No follow-up is automatically
started by SEARCH_POLICY.yaml.

## Paper figures, contingent on the results

| Figure | Claim the evidence must earn | Decisive content |
| --- | --- | --- |
| 1. What an average misses | The phenomenon is identifiable at the available resolution | Existing T/U/M figure, all-type outcomes, sample-size and missingness sensitivity |
| 2. The actual circuit | The candidate corresponds to a specific organization | Named partners, individual cells, morphology, known subtype and anatomical comparisons |
| 3. Competing explanations | Groups or a continuous coordinate explain input-output pairing beyond simple references | Locked prediction contrasts, R maps, conditional nulls and strongest falsifier |
| 4. Another specimen | The stated circuit prediction transfers within its declared scope | Frozen matching, every case, failures, directional effect sizes and uncertainty |
| 5. Consequence and limits | The retained structure changes a concrete circuit inference | Predicted versus observed pair mass, validated cell-level paths, conditions where averaging remains adequate |

Static connectivity can support a strong structural paper if the organization
is new and consequential. A functional claim would additionally require
appropriate physiological or perturbation evidence. A circuit simulation
would be a model-dependent prediction until validated, not a substitute.

The paper's abstract should ultimately name the circuit, the organization,
the competing explanation excluded, the replication scope, and the inference
that changes. Revise this figure sequence to fit the actual evidence. A
completed search or a positive terminal does not guarantee an influential
paper, and no planned figure is an observed finding.
