# Dataset contract — Episode 12

EP12 uses the existing MaleCNS v1.0 flat-connectome release. It creates no new
data asset and grants no new access by itself.

## Data source

| Item | Fixed value |
| --- | --- |
| Dataset | MaleCNS v1.0 flat connectome |
| Shared dataset ID | flyem_male_cns |
| Provider ID | male-cns:v1.0 |
| Release date | 2026-06-08 |
| Read-only local source | /oak/stanford/groups/russpold/data/br_autoresearch_data/flyem_male_cns/gcs-male-cns-v1.0-flat-connectome |
| License | CC BY 4.0 |
| Paper | Berg et al., Cell (2026), 10.1016/j.cell.2026.08.015 |

The primary analysis uses four parts of that release:

| File | Use in EP12 |
| --- | --- |
| body annotations | neuron type, side, anatomy, and eligibility |
| body neurotransmitters | optional annotation and quality-control context |
| body statistics | connection strength and reconstruction/status controls |
| connectome weights | directed outgoing partner profiles |

The full minconf-0.5 weight table is the primary graph input. Traced-only and
significant-only variants are not primary inputs unless their meaning is
independently established before the analysis begins. Point-synapse,
synapse-partner, and t-bar neurotransmitter tables are not required.

## What has already been seen

Annotation layouts and category counts were previously inspected. No
connection weights, body statistics, synapse values, or neurotransmitter
values were reported as inspected for the original EP12 question.

Existing annotation-only feasibility counts include:

- 211,577 annotation rows;
- 164,506 rows with a non-empty type and 11,751 raw type labels;
- provisional left/right counts of 80,785 and 82,932;
- 1,224 types with at least five provisional left and five right neurons;
- 1,184 at that threshold after rejecting types with any non-lateral row; and
- a conservative scenario with 656 fully lateral types and 13,806 typed rows.

These counts do not define the final cohort. They remain exposed feasibility
information, and revising the design cannot make them unobserved.

## Assigning development and final types

Before body statistics or graph weights are opened, apply a fixed
annotation-only eligibility rule. Assign whole provider types—not neurons,
edges, synapses, or sides—so that 80% go to development and 20% go to final
evaluation.

Any balancing may use only annotation fields chosen in advance. The final
assignment is made once and recorded before outcomes are examined. If the
eligible cohort lacks enough bilateral support, the episode stops; thresholds
cannot be loosened after graph access.

Every type assigned to final evaluation remains in the accounting. A support
rule fixed in advance may mark a type unscorable, but that type cannot be
silently removed or replaced.

## Field restrictions

Provider type defines the focal and partner categories, with the caveat that
the published taxonomy was itself informed by connectivity.

Provider instance is side-encoded, and provider group may contain finer
connectivity-informed distinctions. Neither may be used for eligibility,
splitting, initialization, fitting, or model selection. They may be consulted
only after the result to ask whether a reported group is already represented
in existing annotations.

Unknown, untyped, fragment, proofreading, missing-annotation, and
out-of-vocabulary partner mass must remain explicit. The analysis cannot make
its profiles look cleaner by silently discarding those endpoints.

## Who may see what

| Stage | Available | Still hidden |
| --- | --- | --- |
| Type assignment | annotation fields needed for eligibility and the 80/20 split | body statistics, graph weights, synapses, neurotransmitter values |
| Synthetic qualification | generated examples only | all real connectivity values |
| Outgoing development | declared body statistics and outgoing profiles owned by development types | focal profiles and graph-derived summaries for final types |
| Null calibration | the selected development-only T/U generator and synthetic null data | all final-type focal profiles |
| Final evaluation | the chosen T/U/M procedure and final profiles inside one evaluator operation | final profiles and candidate-guiding diagnostics from the search process |
| Incoming scope | incoming profiles after the outgoing result is fixed | any use of incoming results to change the outgoing conclusion |

Outgoing profiles belong to their focal source neuron. Incoming profiles
belong to their focal target neuron. Keep each neuron and all of its
direction-specific connections in the role assigned to its whole type.
Synapses and edges are measurements, not independent biological samples.

## Final cross-side evaluation

For each final type, the evaluator performs both directions in one final
operation:

1. fit the chosen T/U/M procedure to the left-side outgoing profiles and score
   the right side without fitting on the right;
2. fit the same procedure to the right-side profiles and score the left side
   without fitting on the left.

This source-side fitting is part of the final evaluation, not a new round of
development. The vocabulary, model families, tuning rules, thresholds, and
allowed summaries are already fixed. Final profiles and diagnostics that could
guide a new candidate are not returned to the search process.

The evaluator records first access and refuses a second effective opening. If
an acknowledgement is lost, the existing final operation must be reconciled
before any retry.

## Scope and required records

Outgoing composition is the primary analysis. Incoming composition begins only
after the outgoing conclusion is fixed, uses its own choices, and cannot rescue
the outgoing result.

A future run must retain enough information to reconstruct what happened:
the source used, exposure history, whole-type assignment, partner vocabulary,
trial results and failures, resource use, chosen procedure, null results,
every final type and direction, the final-opening count, the detailed outcome,
and the evidence supporting that outcome.

The current EP12 directory contains a study design, not a working
episode-specific executor. None of these requirements claims that a run has
occurred.

## Data needed for the circuit follow-up

The [paper plan](outputs/paper_plan.md) proposes a later input-output study.
It does not expand access during the primary outgoing round. No external
dataset is selected, provisioned, or declared unexposed by this revision.

| Source | Proposed role | Condition before use |
| --- | --- | --- |
| MaleCNS v1.0, previously assigned development types | Discover a small circuit hypothesis after the outgoing conclusion is fixed; relate outgoing preferences to incoming partners | Record the new incoming exposure, select cases without final-type outcomes, and keep the original result fixed |
| MaleCNS annotations, cell morphology and published circuit descriptions | Identify the cells, assess anatomy and prior subtype annotations | Authenticate any additional morphology assets; distinguish descriptions already used by curation from new evidence |
| One suitable external connectome, initially assessing FlyWire or hemibrain | Test the locked circuit prediction in another specimen | Pin version, specimen identity, anatomy, homologous types, partner crosswalk, observation rules, exposure history and evaluation role before connectivity access |

Choose the external source for coverage of the actual candidate circuit,
not for a preferred result. The two candidate resources are not interchangeable:
hemibrain has restricted anatomical coverage, and a male-to-female comparison
is a transfer challenge with sex, specimen, and reconstruction differences.
Additional releases or hemispheres of one specimen do not add animals.
Previously exposed external outcomes remain development information; use a
genuinely new audit source for a new confirmatory round.

### Incoming access can expose outgoing outcomes

An edge from neuron j to neuron i is both an outgoing observation for j and
an incoming observation for i. In particular, an incoming query for a
development neuron can reveal an outgoing edge owned by a final type. The
follow-up must not issue such queries during the outgoing search or while
that final bank remains reserved. A focal-neuron filter alone is insufficient.

Before constructing a follow-up view, a custodian must check source and target
roles for every edge, record any already consumed roles, and preserve unknown
or withheld mass. Shared edges cannot be represented as fresh independent
evidence. Keep both direction-specific profiles of each focal type within the
same specimen-level role for the new study.

### External prediction has a fixed information boundary

Outgoing profiles and approved anatomical covariates may be inputs to a
frozen rule that assigns group probabilities or a continuous coordinate.
External incoming profiles are prediction targets. They may not select the
case, align groups, fit centroids, choose a vocabulary, tune a model, or decide
which external dataset to report. Applying a source-fitted rule is inference;
reclustering the target connectome is fitting and is not permitted in that
evaluation.

Cross-dataset type matching should use authenticated published homologies and
morphology where possible. Record that existing taxonomies may themselves
have used connectivity. Inspecting candidate-specific external connectivity
to resolve a match consumes that evidence; it is not an annotation-only step.

Incoming and outgoing profiles use separately normalized total incident mass,
including unknown, untyped and unsupported partner bins. Exclusions and
coverage limits are fixed before evaluation. A zero in an incomplete or
withheld graph cannot establish the absence of a pathway.
