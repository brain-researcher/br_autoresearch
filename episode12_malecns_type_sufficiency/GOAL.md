# When does a cell-type average distort a circuit's wiring?

[Scope and novelty review, 2026-10-02](outputs/scope_novelty_review_20261002.md).

Neurons assigned to the same cell type are often summarized by one wiring
profile. That summary may hide meaningful differences. Some cells could
prefer one set of downstream partners, while others of the same type prefer
another.

EP12 asks two scientific co-primary questions. **Does within-type wiring
organization repeat, and does retaining it change a concrete circuit
inference?** A reproducible cluster without a circuit consequence is a
candidate annotation refinement, not the intended paper-level discovery.

The first question is tested by the existing outgoing T/U/M round. If neurons
on the left prefer one set of downstream partners, can the same preference
be recovered on the right? Or is the apparent grouping better explained by
continuous variation, anatomy, connection strength, or reconstruction quality?

We start with outgoing partner-type profiles in the pinned MaleCNS v1.0
release. This first test measures bilateral reproducibility in one male fly.
The second question is tested in a separately selected and frozen circuit
round, planned here but not executable or authorized by this revision:

> When does averaging neurons of the same type hide distinct input-output
> pathways?

The intended paper should identify a named circuit organization and quantify
which upstream/downstream partner associations its type average misstates.
Retaining a group, continuous coordinate, or anatomical organization must
improve a held-out structural prediction, not merely yield a nicer partition.
Neither endpoint establishes a new cell type or its function.

The [paper plan](outputs/paper_plan.md) sets out the circuit follow-up,
closest prior work, decisive comparisons, and evidence needed for each claim.
The T/U/M test below remains the unchanged primary **outgoing** round. Circuit
consequence is co-primary for the proposed **paper**, not an added endpoint
of the running outgoing search. The rounds have separate locks, statistical
decisions and results; one cannot rescue or overwrite the other.

## At a glance

| Question | EP12 design |
| --- | --- |
| What varies? | The distribution of a neuron's outgoing connections across downstream partner types. |
| What is compared? | One average profile (T), one flexible continuous population (U), and two or three residual groups (M). |
| What must repeat? | A preference learned on one body side must predict neurons of the same type on the other side. Both directions are required. |
| What is held out? | Twenty percent of whole annotated types are reserved for one final evaluation. |
| What is the outgoing score? | The held-out predictive gain of M over the better fair T/U reference. |
| What can the first round conclude? | Reproducible groups, an adequate single-population description within a fixed margin, or an unresolved result. |
| What is the circuit co-primary question? | In at most three fixed cases, how much same-cell input-output pair mass does type averaging misstate, and can a frozen representation recover it beyond fair anatomy/continuous references? |
| What is its status? | A planned separate statistical round; case support, meaningful margins, calibration and execution are not frozen by this narrative update. |
| What is the transfer test? | After source-side selection and fitting, test the same circuit prediction in one compatible external specimen if available; cross-sex transfer is not same-sex replication. |

## From a wiring difference to a circuit finding

Consider a hypothetical type X. Some X neurons mainly receive from A and
project to B; others receive from C and project to D. A graph that merges all
X neurons can suggest A-to-D and C-to-B paths through X even when those
cell-level chains are weak or absent. This is an illustration, not an EP12
result. Real data must establish the size and reproducibility of the
association, including missing connections and anatomical organization.

The circuit round asks whether such input-output associations exist in a
small, explicitly selected circuit, and whether retaining a reproducible
group or continuous coordinate improves their prediction. Its consequence
endpoint is error in **neuron-equal input-output pair mass**, with incoming
composition prediction as a supporting endpoint. It must also quantify the
distortion from the type average itself. A better incoming score without a
meaningful pair-mass consequence does not earn the proposed circuit claim.
Anatomy, continuous variation, and discrete groups are competing explanations.
A continuous or spatial organization can be biologically informative without
supporting residual modes in the outgoing test.

This is not the first proposal that within-type variation matters for circuits.
[Dombrovski et al. 2023](https://doi.org/10.1038/s41586-022-05562-8)
linked output synaptic gradients to visuomotor transformations;
[Dombrovski et al. 2025](https://doi.org/10.1038/s41586-025-09037-4)
established molecular control of LPLC2 input and output synaptic gradients.
EP12 therefore needs a new named organization, a new consequential inference,
or a calibrated boundary on averaging, rather than rediscovering those
gradients or declaring their continuous alternative a nuisance.

Known within-type heterogeneity and cross-brain connectivity subgroups are
starting points in the literature, not sufficient novelty claims. Each
candidate needs a comparison with existing annotations and circuit papers.
The paper must show what is newly learned about that circuit, or establish
a useful, well-tested boundary on when averaging is adequate.

The circuit round has its own endpoint, source choices, calibration, budget,
selection rule, and held-out evaluation. Reusing the same male or the other
body side provides a diagnostic, not new independent confirmation. A local
case cannot replace a failed type-aggregate outgoing test.

The first circuit contribution is explicitly **within this specimen**. Cases
come only from previously assigned development types, never final-type
outcomes. At most three cases and their partner pairs are selected once,
before incoming inspection; every selected case and failure is retained.
Source-side fitting and evaluation must be separated under a qualified
cell-block or anatomical-block scheme before the new incoming outcomes are
opened. That split supports a conditional prediction test, not animal-level
replication. New incoming summaries can share already exposed outgoing edges;
unless target separation and selection-aware calibration are justified, the
source-side evidence remains descriptive/diagnostic, not fresh confirmation.
External evidence strengthens the scope only if homologous cells,
partner definitions and coverage can be established without target-driven
matching. Without a compatible fresh external source, no conserved or
cross-animal claim is made.

## Three competing explanations

Every analysis compares the same three model families on the same profiles and
evaluation units.

| Model | Scientific explanation |
| --- | --- |
| **T: one type template** | One average downstream partner profile describes the type well enough at the tested resolution. |
| **U: a flexible single population** | Cells vary continuously around that profile, without distinct groups. |
| **M: two or three residual modes** | Cells form distinct partner-preference groups that reproduce across sides after the planned adjustments. |

U is essential. A mixture can imitate continuous variation, so M cannot win
merely by being more flexible. It must predict better than both T and U, and
its groups must be reproducible and interpretable.

## How the test works

1. **Assign whole types.** Before connection values are opened, use
   annotation-only fields to assign 80% of eligible types to development and
   20% to final evaluation.
2. **Compare T, U, and M fairly.** Every trial fits all three explanations
   under the same data roles and comparable capacity.
3. **Choose one complete procedure.** Use a rule fixed in advance and based
   only on development data. M does not have to win during development for a
   scientifically valid comparison to continue.
4. **Transfer across sides.** For each final type, fit the chosen procedure on
   one side and score the other without fitting on the target side; then
   reverse the direction. Both transfers occur in one final evaluation.
5. **Report the answer.** Apply thresholds and uncertainty rules fixed before
   the development analysis that could favor one explanation over another.

A valid comparison is not the same as evidence for M. A well-qualified test
must be allowed to reach final evaluation even when T or U looks better during
development.

## Two-question figure concept

The conceptual figure shows why cell-resolved input-output pairing can differ
from a merged type graph, then separates the bilateral outgoing question from
the selected circuit-consequence round. The latter compares anatomical,
continuous and grouped explanations without depicting any winner.

The schematic below contains no observed values or real circuit assignments.
It is a design aid, not an EP12 result.

![Conceptual EP12 two-question circuit design, no observed data](outputs/ep12_conceptual_main_figure-v2.png)

The earlier [outgoing-only mockup](outputs/ep12_conceptual_main_figure.png)
is retained. Actual outgoing figure examples still use the frozen
development-only rule, and every assigned final type must be reported. The
circuit figure will report all locked cases and partner pairs; external
outcomes may not choose the best-looking example. All illustrated claims
remain hypotheses until supported by observed results.

## The three scientific answers

| Detailed outcome | Evidence required | Permitted interpretation |
| --- | --- | --- |
| **Reproducible residual groups** | A valid final evaluation; M exceeds the meaningful margin in both directions; the groups recur across sides; their partner preferences are identifiable; all required controls pass. | Eligible annotated types in this male contain bilaterally reproducible residual wiring groups. |
| **Single population adequate within margin** | A calibrated T/U reference; both directional upper bounds for M's advantage fall below the meaningful margin; synthetic tests show adequate sensitivity. | Within this analysis, precision, and model family, a single-population description is adequate. |
| **Unresolved** | A valid final evaluation is completed, but intervals are wide, directions disagree, structure is ambiguous, or the evidence supports only a narrower claim. | These data do not decide the prespecified broad question. |

A nonsignificant difference is not evidence that one population is adequate.
An adequacy conclusion requires enough precision to rule out the meaningful
advantage set in advance.

The T-versus-U comparison remains a prespecified explanatory readout. If U
beats T but M adds no meaningful advantage, EP12 may report that one average
profile misses continuous heterogeneity. This cannot replace the primary
M-versus-better-T/U test after results are seen, and it is not a discovery of
new cell types.

### When the test does not reach an answer

| Status | What happened | What may be said |
| --- | --- | --- |
| **No valid comparison** | Development is completed, but no fair and usable T/U/M comparison can continue; final types remain unopened. | A development-stage limitation prevented the primary held-out test. |
| **Incomplete search** | The run stops before the required search is complete. | Search is incomplete; no completed-search conclusion is allowed. |
| **Technical failure** | An unrecoverable execution error, damaged or mismatched input, role leakage, or final-evaluation failure invalidates the analysis. | No scientific interpretation is allowed. |

## What makes a comparison valid

A complete comparison must:

- use the assigned development and final roles without leakage;
- evaluate T, U, and M under fair capacity and calibration rules;
- preserve unknown, untyped, fragment, and missing partner mass;
- pass synthetic tests for average, continuous, grouped, weak-signal, and
  side-artifact scenarios;
- reproduce when rerun under the same settings; and
- record complete outputs, failures, resource use, and the reason for every
  inclusion or exclusion.

Poor real-data performance by M, weak separation, or continuous-looking
variation is a scientific result, not a broken comparison. Numerical failure,
target-side fitting, unfair model capacity, or uncontrolled missing output
makes the comparison invalid.

For each type and transfer direction, the score is held-out log predictive
density per eligible outgoing synapse. Neurons are weighted equally within a
type and direction; types and directions are then weighted equally. Synapses
and edges are not treated as independent biological replicates.

## Data roles and scope

The split is by whole provider type. Neurons, edges, synapses, and body sides
cannot receive roles independently of their focal type. Every assigned final
type stays in the accounting. A rule fixed in advance may mark a type
unscorable, but the type cannot be silently removed or replaced after final
values are opened.

Provider group and instance labels cannot be used to define, initialize, fit,
or select the candidate groups. They may be inspected only later to ask
whether a result merely rediscovers an existing annotation.

Outgoing composition is primary in the current execution contract. Incoming
composition belongs to the planned separate circuit co-primary round. It
cannot rescue the outgoing result and is not independent evidence merely
because the same graph is viewed in the other direction. Source-owned edges
from final types can leak through incoming queries and must remain closed.

Previously inspected annotation inventories remain exposed feasibility
evidence. Revising the study cannot make previously seen information
unobserved. The live exposure and execution state must therefore be checked
before a run begins.

## Controls

The analysis must test whether any apparent groups are explained by:

- connection strength or missing endpoint mass;
- reconstruction status and other technical quality measures;
- anatomical position, neuromere, serial organization, or optic structure;
- a few influential types, partner families, or high-strength neurons;
- arbitrary partner vocabularies, pooling choices, ranks, or nuisance terms;
- shuffled partner identities or side labels; and
- continuous variation that a mixture happens to approximate.

The post-result comparison with provider group, instance, or published
assignments is a novelty check only. A match is a rediscovery, not a new type.

Report the anatomical contribution as well as the adjusted residual result.
An effect explained by location cannot support residual modes, but may
motivate a separately tested hypothesis about spatial wiring organization.
Do not remove an anatomical adjustment after seeing which conclusion wins.

Use a **source-conditional anatomical hierarchy**: qualified spatial or coarse
anatomical fields first; authenticated hemilineage/serial/retinotopic fields
only where present and interpretable; otherwise explicitly state that the
corresponding explanation remains untested. Do not infer missing metadata
from the outcome being explained or relabel provider group/instance as an
independent anatomical control. The current outgoing vocabulary and nuisance
choices remain frozen; this hierarchy guides the new circuit contract.

## First-round search and null calibration

| Item | Limit |
| --- | ---: |
| Valid trials | 36 minimum; 96 maximum |
| Patience | 16 valid trials without improvement, active only after trial 36 |
| Total compute | 1,000 CPU-core-hours |
| Per trial | 32 CPU cores; 192 GiB RAM; 12 wall-hours |
| Overall wall clock | 120 hours |
| GPU | 0 |

If at least one complete comparison is ready for null calibration, the
development data choose a fair T or U generator using a separate rule fixed in
advance. The study then runs exactly 99 synthetic-null searches. Each starts
from the beginning and repeats the same T/U/M search and selection logic, but
does not launch another layer of null searches.

Each fitted-T/U null keeps the development sample sizes, connection strengths,
missingness, and side structure. The observed and null searches use the same
development statistic: M minus the better fair T/U reference after the fixed
selection rule has been applied. Because this revision changes that selection
program, an older null calibration cannot be reused.

The full program therefore has a minimum declared scale of 36 trials × 3
models × 100 observed-or-null searches = 10,800 model-family fits before types
and directions are counted. This is a protocol count, not measured runtime.
A cost study must show that the planned work fits the compute limit before
real connection outcomes are opened.

The null probability is used when deciding whether the final result supports
M. Completing the null program correctly is required for a valid comparison;
failing its scientific threshold does not by itself make the comparison
invalid or prevent an informative T/U result.

## First-round guardrails

Before outcome-revealing development begins:

- The meaningful margin is 0.02 nats per eligible outgoing synapse; whole
  types remain the uncertainty unit and both transfer directions are covered
  together.
- Adequacy requires synthetic evidence that meaningful residual structure
  would not be missed. Otherwise the result is unresolved.
- Choose one complete, fair T/U/M comparison from development only using
  bilateral predictive performance. M need not win during development.
- Choose the fitted T or U null from development only, preferring U when they
  are comparably supported.
- Choose and explain figure examples using development only before final
  access. Final outcomes cannot choose examples; report every final type.

These are safeguards, not a fixed optimizer, resampling count, tie rule, or
figure layout. Record executable details before outcome-revealing development.

Any small real-data diagnostic must use assigned development types only.
Before their connection values are opened, its selection rule, allowed
outputs, and exposure record must be fixed.

If a smaller fixed comparison is chosen instead of the adaptive study, it must
be described as a different study design. Its success cannot be counted as
completion of this search.

## Execution evidence still required

Design-freeze status (2026-09-24): the EP12 directory described the study but
did not yet contain a working episode-specific executor. Before anyone claims
that EP12 can run through to a result, synthetic tests must demonstrate:

- positive, adequate, and unresolved outcomes;
- distinct handling of no-valid-comparison, budget exhaustion, and input
  failure;
- recovery after interruption without losing or double-counting a trial; and
- one final opening even if an acknowledgement is lost and the run resumes.

Launch update (2026-09-25): `outputs/executor/` now implements and tests the
synthetic control plane, and `outputs/executor_qualification/` durably records
all six terminal/recovery scenarios above. The pre-outcome implementation now
also contains an executable fixed T/U/M method, bounded linear and curved U
references, generated biological counterexamples, annotation-only roles, and
measured numerical/cost checks under `outputs/prelaunch/`. Its current
qualification status is failing, not passing. After a later explicit user
instruction, development-only MaleCNS access began: the role firewall
materialized 816 development types while retaining 204 final types behind the
lock, and `outputs/development_run001/` records one complete real T/U/M
development configuration. This is not completion of the adaptive search.
The wider grammar, mode-support gates, full-search null, and one-shot final
evaluation remain incomplete or locked.

Subsequent status (2026-09-26): the 18+9+9 development prefix is complete and
provisionally archived as 36 valid development-only trials. Scientific
qualification and the required post-36 falsifiers remain incomplete; no
procedure is locked, the 99-search null has not run, and final access remains
unauthorized and unopened.

Documentation or a passing structural check is not evidence that the study
ran. A real result requires durable trial records, the selected procedure,
the final-evaluation record, the detailed outcome, the reason, and links to
the supporting outputs.

## First-round claim boundary

The strongest positive wording is **bilaterally reproducible residual outgoing
wiring groups within eligible curated types in this MaleCNS male**. EP12 does
not establish new cell types, cross-animal or cross-sex generalization,
molecular identity, causality, function, or behavior.

The proposed circuit co-primary claim concerns the accuracy of a structural
input-output description in the locked cases. It requires a meaningful
averaging distortion, a calibrated held-out consequence improvement and the
controls in the paper plan. Its external transfer claim, if tested, names
the exact second specimen and correspondence. Static synapse counts alone do
not establish signal transmission, selective gating, molecular mechanism or
behavior. A successful outgoing round does not earn the paper-level claim.

An explicit scientist launch authorizes the bounded episode-managed work
described here, subject to the data roles above. It does not itself establish
formal acceptance, reward, or a shared scientific claim. This 2026-09-30
narrative/figure repair runs no analysis, amends no operative model or threshold,
and opens no incoming, external or final-role connectivity. Recorded execution
history and the active generated-only qualification chain remain in
`outputs/experiment_log.md`.
