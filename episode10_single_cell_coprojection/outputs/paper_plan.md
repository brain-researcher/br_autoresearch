# EP10 paper plan: when does a target average hide different axonal output organizations?

Status: proposed study design, 2026-09-24. No biological result, independent
replication, or paper-ready candidate is claimed here.

The scientific question and active analysis rules remain in
[GOAL.md](../GOAL.md) and [SEARCH_POLICY.yaml](../SEARCH_POLICY.yaml). This plan
organizes possible manuscript claims without changing those contracts.

## The intended discovery

The paper should explain whether neurons from one cortical source show specific
reproducible conditional patterns across projection targets, where those
patterns apply, and what a projection map averaged over the source fails to
show.

The strongest intended result would name a source and a target combination,
show that it reproduces across verified animals within a frozen soma-position domain,
distinguish it from target prevalence, measured spatial routing, and known
population composition, and show how complete axons implement it through
branching and terminal organization.

A particularly informative case is:

> Among neurons that arborize in a shared target A, does the terminal
> distribution inside A differ between an A+B context and an A+C context?

If supported, this would show that the label “projects to A” can hide distinct
organizations inside A. It would not by itself establish distinct synaptic
partners, synapse strengths, functions, developmental programs, or cell types.

A second valid paper route is explanatory rather than residual. The original
combination first reproduces, but measured spatial routing or an independently
labeled population mixture predicts it in unseen final groups and leaves residual
association below a meaningful bound. That result would explain why an average
is misleading without requiring M2 to remain useful.

Neither route is assumed. Internal predictive gain alone is a methods or
preliminary result, not the intended biological paper.

## Why this direction

| Direction | Assessment |
| --- | --- |
| Catalogue more co-projection frequencies | Useful description, but non-random co-projection is already established |
| Report that complete axons are heterogeneous | Too general; morphology and projectome studies already establish extensive diversity |
| Report only that M2 predicts better than M1 | Potentially useful method result, but it does not validate a named biological organization |
| Reproduce a generic co-target/laminar association | Important control or extension, but it overlaps directly with axonal BARseq |
| Explain one named combination and its full-axon implementation | Preferred: connects a reproducible phenomenon, competing explanations, spatial limits, terminal organization, and a concrete inference lost by averaging |
| Show that measured space or known population composition explains the combination | Also paper-worthy if the phenomenon reproduces, the explanation predicts new animals, and the residual is bounded with useful precision |

The proposed contribution is therefore candidate-specific: what organization
is hidden, where it applies, which explanations survive, and how the
interpretation of a named pathway changes.

## Closest prior work and the novelty question

| Prior work | Already established | What an EP10 candidate must add |
| --- | --- | --- |
| Han et al., Nature 2018, [The logic of single-cell projections from visual cortex](https://www.nature.com/articles/nature26159) | Single cortical neurons show diverse, non-random combinations of long-range targets | More than another catalogue: a named conditional prediction, competing explanations, cross-animal scope, and consequential pathway interpretation |
| Yuan et al., Nature Communications 2024, [axonal BARseq](https://www.nature.com/articles/s41467-024-52756-x) | In auditory-cortex IT neurons, co-target status is associated with laminar termination inside a shared cortical target | For a shared-target claim, a frozen A+B/A+C prediction beyond the generic association; for other candidates, a distinct question not already answered by this result |

These are benchmarks, not an exhaustive review. Candidate-specific review must
also cover the closest complete-morphology, projectome, and molecular tracing
studies before novelty is claimed. Before locked evaluation, prepare a one-page
novelty record containing:

- the source and exact target combination, plus A/B/C eligibility rules only
  when the shared-target endpoint is activated;
- prior findings and the unresolved explanation the candidate distinguishes;
- one held-out prediction appropriate to Route A or Route B, together with its
  strongest rival;
- how either result changes interpretation of the pathway; and
- for an anatomical claim, what complete trees add beyond another laminar
  association.

For manuscript planning, describe the candidate as proceed, reframe, or stop.
This label is editorial guidance, not a new search outcome: proceed means a
consequential held-out prediction remains; reframe means only a narrower
replication, full-tree, or boundary contribution remains; stop means prior work
already answers the question or no prediction can be stated before final data.

A striking tree, a significant profile difference, or a new source area is not
sufficient by itself. Candidate ranking and literature review cannot use
locked-final anatomy.

## Study sequence

### 1. Start with data qualification

The first analysis stage asks whether EP10 has a valid estimand and enough
independent biological-group and position support. It checks sample overlap, repeatable
target calls, non-detection versus unknown, independent-label provenance,
detectable effects, and exact-normalization/null cost. If the shared-target
endpoint may be pursued, it checks whether atlas-aligned coordinates and
terminal profiles can be extracted repeatably without selecting A, B, or C or
comparing their contexts.

The output is a brief proceed, revise, or stop decision naming the source,
target resolution, data roles, common-support domain, precision, and compute
requirements. MOp remains provisional; MOp, MOs, and SSp cannot be pooled to
manufacture support. No candidate effects are scanned in this phase.

### 2. Establish the complete-target-set phenomenon

For every eligible neuron, freeze the target vocabulary, laterality,
observation rule, complete detected target set, and detected target count `K`.
Unknown is not absence. M0, M1, and M2 score the same complete sets on the same
neurons with exact normalization conditional on `K`.

| Model | Information included | Question |
| --- | --- | --- |
| **M0** | Target prevalence, source context, verified technical variables, and eligible independent labels | What follows from prevalence and measured source composition? |
| **M1** | M0 plus routing quantities available for every candidate target set | What additional organization follows from measured spatial routing? |
| **M2** | M1 plus regularized residual target associations | Is useful combination structure left after the measured references? |

M2 minus M1 is the unchanged primary predictive endpoint. M1 minus M0 is
secondary. The aggregate score can support residual association, but cannot
validate a particular combination.

The core numerical design remains in force: at least 12 development and 8
locked-final biological groups, 40–88 valid trials, and at least 16 model-search
anchors. It also retains exactly 99 complete target-set search-null reruns, a
0.002 bits-per-cell useful gain and patience threshold, and a 0.01 bits-per-cell
source-family regression guard. The shared-target anatomy question uses the
same groups and does not change those decisions.

### 3. Develop, freeze, and test a named paper candidate

Candidates come only from grouped development data. Freeze the target
combination, effect direction, probability contrast, useful margin,
applicability and common-support domains, uncertainty, multiplicity, novelty
record, and anatomical summaries before opening final groups. The optional
A+B/A+C endpoint receives an eligibility and activation decision at the same
time. Claims use identified probability contrasts, not raw fixed-`K`
coefficients.

Whether A, B, and C are specified independently or selected by the target-set
controller, exact support and endpoint eligibility are assessed only in
development groups. Final groups remain unopened; too few qualifying final
events make the result inconclusive rather than licensing reselection.

Selection made by the target-set controller is replayed in all 99 null runs;
anatomy cannot select A, B, or C. In locked groups, test the frozen contrast in
its predicted direction and domain without redefining the candidate. Show
group- and position-level results, every selected candidate, disagreement,
influence, unsupported positions, and missingness. A failed candidate cannot be
replaced.

For the explanatory route, first reproduce the phenomenon, then test whether
the frozen spatial or independent-label explanation predicts it and whether
uncertainty bounds remaining M2 value below the useful margin. A
non-significant residual is not an adequacy result without demonstrated power.

### 4. Test anatomical implementation and what an average hides

For Route A, after the candidate is frozen, quantify path divergence,
normalized shared path, separate collateral and terminal-tree coverage, and
continuous trees crossing atlas boundaries. Compare group-level distributions
with references matched on target geometry, soma position, source label,
prevalence, `K`, other targets, total morphology, registration, clipping, and
reconstruction quality. Here and below, the group is the animal when verified,
or the most conservative specimen otherwise with a cross-group claim. A common
root is not a finding.

The shared-target contrast is conditional, not required for every candidate.
If activated, freeze A, B, C, exclusivity, `K` and other-target handling, the
coordinate inside A, one primary normalized-profile metric, one falsifiable
profile prediction, one pooling-discrepancy summary, and whether entry route is
an alternative or mediator.

The contract deliberately does not yet choose that metric. After A, B, and C
are chosen in grouped development, freeze the shared-target endpoint's primary
profile estimator from biological rationale and technical repeatability while
B/C context labels are blinded. Reliability may choose among a prespecified set
of coordinates or summaries; the A+B versus A+C contrast cannot choose the
metric or direction. The estimator may be a group-first mean of
neuron-normalized topographies or another justified group-level summary. A
raw or amount-weighted regional projection map answers a different question;
it must be displayed separately and cannot be called equivalent to the
neuron-normalized profile. Any pooling-discrepancy summary needs a frozen null,
meaningful margin, and group-level uncertainty before it can quantify what an
average hides.

Report total arbor in A separately from normalized distribution. Branch or
reconstruction-endpoint profiles, centroid, spread, and focality are supporting
summaries; endpoints are not synapses. Realized morphology is an outcome and
cannot enter M0–M2 or select the candidate. If the endpoint is activated, its
failure remains in the report.

### 5. Test transfer when a comparable resource exists

A held-out subset of one release is internal cross-animal evaluation, not
automatically external replication. Before transfer, document independence,
source and position overlap, target crosswalk, observation differences, animal
support, lineage, and zero-versus-unknown handling.

Freeze either complete parameter transport or marginal-frequency recalibration
with association structure locked. Do not mix both in one claim. If no resource
provides a genuinely comparable estimand, retain the internal scope and report
the comparability failure. Projection-TAGs can test a prespecified population-
mixture explanation only with an overlapping relation and explicit assay model;
it does not enter the morphology likelihood.

## Endpoints with a concrete interpretation

| Endpoint | What supports it | What it can mean |
| --- | --- | --- |
| **Primary target-set endpoint** | Equal-group locked-final M2-minus-M1 gain of at least 0.002 bits/cell, calibration, passed full-search null, no disallowed source-family loss | Residual combination structure improves complete-set prediction at the frozen resolution |
| **Named-combination contrast** | Frozen probability contrast reproduces across final biological groups in its common-support domain with multiplicity control | A conditional target organization recurs; not yet a mechanism or cell type |
| **Explanatory endpoint** | The phenomenon reproduces, M1 or an independent-label block predicts it, and residual M2 value is bounded below the useful margin | Measured routing or known composition explains the pattern within the stated scope |
| **Shared-target terminal endpoint, when activated** | The A+B versus A+C prediction reproduces at group level with amount, registration, boundary, and a prespecified entry-route analysis | Co-target context predicts axonal organization within A |
| **Pooling-discrepancy readout, when activated** | A frozen group-level comparison separates context profiles from the declared pooled-map estimator with a useful margin | The declared regional average omits a quantified conditional organization |
| **Branch implementation for Route A** | Frozen divergence, shared-path, collateral, and terminal-tree summaries pass matched structural comparisons | Complete axons implement the organization reproducibly |

Target identity and morphology from the same reconstruction are complementary
outcomes, not independent replications.

## Decisive controls and failure interpretations

| Alternative explanation | Test | Consequence if it explains the result |
| --- | --- | --- |
| Target prevalence and fixed `K` | M0 and calibrated complete-set predictions | No residual interaction claim beyond that reference |
| Soma position and measured routing | M0 position terms plus M1's incremental routing terms, common-position contrasts, and applicability checks | Support a spatial explanation if the phenomenon reproduces and residual value is bounded |
| Known population composition | Independent-label block, within-label contrasts, and group-position overlap | Support only the measured composition explanation; without eligible labels, mixture remains unresolved |
| Shared-target definition, amount, and entry route | Freeze exclusivity, `K`, other targets, total A arbor, and the route's role as alternative or mediator | Narrow the A+B/A+C claim or reframe it as amount- or route-dependent |
| Target boundary and reconstruction quality | Continuous-tree, registration, clipping, and repeated-boundary sensitivities | Do not make an anatomical organization claim if measurement explains it |
| One group, resource, or striking tree | Leave-one-group and influence checks; report full distributions | Narrow or reject the candidate claim |
| Adaptive selection or overfitting | Complete 99-run target-set search null | No promotion if the frozen null rule fails |
| Prior axonal BARseq overlap | Candidate-specific literature and novelty review | Report replication or extension unless a separate consequential prediction is supported |

The 99 target-set null does not calibrate anatomy. Anatomy uses frozen metrics,
matched references, group-level uncertainty, and its own multiplicity rule.

## Paper-worthy routes

### Route A: residual combination with anatomical implementation

This route requires useful locked-final M2-over-M1 gain; at least one frozen
combination that reproduces within its applicability domain; full-axon evidence
that passes matched structural comparisons; and an eligibility disposition for
the shared-target endpoint. If that endpoint was activated, its prediction must
also reproduce and its failure cannot be discarded. Comparable external
evidence is added when available; otherwise the claim remains internal
cross-group evidence, and is called cross-animal only when animal identities
are verified.

### Route B: spatial-routing or known-population explanation

This route requires the original phenomenon to reproduce, the frozen spatial
or independently labeled explanation to predict new final groups, adequate
calibration, uncertainty that bounds residual M2 value below the meaningful
margin, and simulations demonstrating sensitivity to residual effects of
interest.

This is not a failed version of Route A. It is an explanatory paper if the
phenomenon and the adequacy bound are both convincing.

### Results that are not sufficient

- M2 improves only in development data.
- The aggregate score improves but no named combination reproduces.
- A+B and A+C look different without a frozen prediction.
- A cotarget–laminar association merely repeats axonal BARseq.
- One visually compelling reconstruction supports the story.
- A projection-derived cluster is called an independent cell population.
- A within-release holdout is described as independent external replication.
- A non-significant residual is interpreted as equality without useful power.

These may support a methods result, descriptive report, or later hypothesis,
but are not candidate-ready biological outcomes under the current plan.

## When to deepen, narrow, or stop

| Evidence | Next action and permitted claim |
| --- | --- |
| M2 clears its margin, a named combination reproduces, matched full-axon evidence passes, and any activated within-A or required external-transfer prediction also passes | Develop Route A, retaining exact pathway, biological-group, position, and external scope |
| Combination reproduces but an activated within-A prediction fails | Report the narrower target-set phenomenon; once activated, that endpoint is required and Route A fails for this candidate |
| Space or independent labels explain a reproduced combination | Develop Route B if the residual bound is informative; do not retain a residual-combination narrative |
| Combination reproduces but branch evidence is unstable | Route A fails. Report only the narrower target-set result unless Route B independently passes its explanation and residual-bound requirements |
| No useful M2 gain and no supported explanation | Close without a paper candidate under the current question |
| Development A+B/A+C support is insufficient | Mark the shared-target endpoint ineligible; retain the unchanged target-set primary endpoint without the within-A claim |
| An activated endpoint has too few qualifying final events | Report it as inconclusive, retain it in the results, and do not reselect A, B, or C; Route A is not supported for that candidate |
| Observation or non-detection is indefensible | Stop or reformulate; unknown cannot become absence |
| Animal identity or common-position support is unresolved | Use the conservative specimen scope or stop the cross-animal claim |
| External data are incomparable | Retain internal scope and report failed comparability; do not remap after outcomes |
| Candidate-specific novelty review finds no distinct contribution | Reframe as replication/extension or stop that candidate |
| Search ends without a decision | Label incomplete search, neither scientific success nor negative evidence |

Failures and ineligible endpoints remain visible. They cannot be converted into
evidence that co-projection organization is absent.

## Main figures, contingent on the results

| Figure | Claim the evidence must earn | Proposed content |
| --- | --- | --- |
| **1. The observable population** | One coherent source, target vocabulary, and group/position domain support the analysis | Source and targets; group-by-position coverage; detection/non-detection/unknown examples; exclusions and observation workflow |
| **2. Distinguishing the competing explanations** | The calibrated comparison tests whether residual association, routing, or known composition is adequate | M0/M1/M2 schematic; gain and bounds by group; calibration; full-search null; influence and source-family results |
| **3. One named combination and its explanations** | The frozen relationship reproduces in a defined domain, or one measured explanation accounts for it | Probability contrasts; applicability map; common support; independent labels; residual bound; every frozen candidate |
| **4. Route-contingent explanation** | Route A earns a full-axon implementation claim, or Route B earns a calibrated spatial/population explanation with a useful residual bound | Route A: fixed-rule trees, branch summaries, and the shared-target panel when activated. Route B: the successful explanatory variable, overlap, prediction, residual bound, and failure regions; no positive anatomy panel is required. |
| **5. Transfer, limits, and the corrected pathway view** | The supported rule predicts unseen groups and, when feasible, a comparable resource within a stated scope | Frozen predictions; external result or comparability failure; applicability boundary; pooled versus conditional pathway diagram; retained and rejected interpretations |

If Route B wins, Figures 3–5 center the successful explanation and residual
bound. The shared-target panel is omitted only when inactive or ineligible. An
activated but negative or inconclusive endpoint remains visible; anatomy may
appear as a limitation or sensitivity, not as a required positive result.

## Tables and supplementary record

The main tables should cover biological support and data roles, the candidate
novelty/prediction record, and every frozen candidate's final disposition.
Supplementary material retains the observation rules, trial and null results,
detectable-effect simulations, label and influence analyses, failed candidates,
anatomy definitions and sensitivities, resource comparability, and a claim
table separating observations, interpretations, and unresolved alternatives.

## Boundary with EP12

EP10 concerns regional target combinations and complete axonal geometry:
routes, branch points, collaterals, and terminal distributions. It can show
that neurons with different co-target contexts organize their axons differently
inside a shared target.

EP10 cannot identify presynaptic or postsynaptic partner identities, synapse
number or strength, compartment-specific connectivity, input-output partner
pairings, physiological information flow, or functional consequences. Those
require the synaptic evidence contemplated by EP12 or additional physiological
and perturbational experiments.

Conversely, EP12 cannot substitute connectivity profiles for EP10's questions
about continuous paths, divergence, target boundaries, and terminal topography.
Results from overlapping cells or the same reconstruction are correlated
evidence, not independent replication.

## What the abstract must eventually say

A positive abstract should always name the cortical source, exact phenomenon,
spatial domain, held-out-group and external scope, strongest measured
alternative, and major unresolved alternative. Route A must additionally name
the full-axon organization that changes the pathway interpretation. Route B
instead names the successful spatial or population explanation and the bound
on residual association; it does not require a positive anatomy claim.

If those nouns cannot be filled with a specific validated case, EP10 has not
earned the intended paper claim. A completed search, a positive terminal state,
or an attractive figure does not itself guarantee a paper-worthy discovery.
