# What can closed-loop dopamine stimulation identify about a learning rule?

Two dopamine interventions can produce different acquisition outcomes without
uniquely identifying the algorithm that generated them. EP02 asks how far the
released closed-loop experiment can take us from a difference between assigned
protocols to evidence about a learning rule.

This is a focused, publication-exposed reanalysis, not the discovery that
dopamine can regulate learning rate. [Coddington, Lindo and Dudman
(2023)](https://www.nature.com/articles/s41586-022-05614-z) already introduced
the adaptive-rate ACTR account and interpreted calibrated stimulation results
in that framework. Reproducing its reported direction is a benchmark, not a
new dopamine mechanism.

The useful remaining question is narrower: after alternative rules have been
qualified on control mice, do their frozen predictions differ for the two
closed-loop contingencies, and what does the mouse-level protocol contrast
actually resolve? A good behavioral fit, a positive protocol contrast, and
unique mechanism identification are three different claims.

![EP02 conceptual design](outputs/ep02_conceptual_question.png)

*Conceptual schematic, not evidence: controls support model development;
predictions are frozen before intervention outcomes; the protocol contrast
does not automatically identify a unique learning rule. The drawing represents
trigger rules, not stimulation on every trial.*

The [paper plan](outputs/paper_plan.md) develops this inference boundary; the
[2026-10-02 review](outputs/scope_novelty_review_20261002.md) retains its
conditional reanalysis scope against current neighboring work. The
operative [search policy](SEARCH_POLICY.json) retains the existing endpoint,
model grammar, budgets, access roles, and unbound thresholds. This writing
revision does not authorize a run or change a binding decision rule.

## Scientific question

Can control-qualified learning rules make distinguishable, locked predictions
for reward-time dopamine stimulation contingent on the absence versus presence
of preparatory licking? If they can, how much of that distinction is resolved
by the existing complete-case mouse cohort?

| Explanation in the frozen panel | What must be predicted before intervention outcomes |
| --- | --- |
| Adaptive learning-rate gate | A nonnegative multiplier changes update magnitude, not sign; the locked target predicts higher acquisition gain for `stimLick-` and lower gain for `stimLick+` than its control reference. |
| Signed error / TD-RPE | The implemented signed teaching rule must generate its own arm predictions under the same trigger and task schedule; a family label is not a prediction. |
| Direct reinforcement | The implemented contingent reinforcement rule must predict both protocols, rather than being dismissed because it fits controls differently. |
| Constant-rate nested null | Predict the protocol behavior without dopamine-dependent rate modulation. |
| Amplitude-dependent hybrid | Lock the physiological predictions first; high-amplitude behavior is only a later boundary diagnostic. |

Every model family must freeze its counterfactual predictions before any
intervention outcome is opened. Predictions may overlap. If a rival predicts
the same reported pattern, that pattern cannot distinguish the rival from the
target. Required model-recovery work measures discriminability within the
specified implementations, not structural identifiability of every possible
dopamine theory. [Wilson and Collins
(2019)](https://elifesciences.org/articles/49547) explain why model recovery
and predictive validation are distinct from selecting a best fit.

The live terminal `candidate_ready_rate_gate_consistent` remains a
pattern-consistency classification. It is not a calibrated test rejecting all
rivals, and the present evaluator packet does not provide such a test. The
paper may compare the already-required locked prediction panel descriptively;
a new rival-exclusion statistic or equivalence threshold would require an
explicit, pre-outcome scientific amendment.

## What the intervention contrast means

Both protocols stimulate at reward time using a verified pre-reward 750-ms
window. `stimLick-` triggers when no lick is detected; `stimLick+` triggers when
a lick is detected, subject to its documented 50% session cap. The assigned
objects are complete closed-loop protocols. They are not interchangeable
fixed doses of dopamine, and realized stimulation counts are not baseline
covariates. The primary analysis must not condition or weight on those counts.

The observed endpoint is session-8 minus session-1 preparatory lick
probability. It measures acquisition gain over two epochs, not a latent
learning-rate parameter. A different gain can reflect changes in acquisition,
asymptote, or behavioral expression. The frozen continuous-lick replication,
latency and nuisance falsifiers help constrain interpretation; they cannot
turn a two-epoch gain into a directly measured learning rate.

This distinction is the proposed contribution: separate evidence for a
protocol-dependent behavioral change from evidence that selects a learning
rule. It is a methods/reanalysis contribution only if the eventual prediction
and robustness results establish something beyond the published group
direction. Current documentation alone is not sufficient for a paper.

## Evidence roles

| Source records, one-based | Role | Exposure rule |
| --- | --- | --- |
| `1,2,4,6,9,11,15,16,19` | Development controls | Development-visible after role separation; never confirmation |
| `3,5,10,14,18,20` | Primary audit: calibrated `stimLick-` | Evaluator-only after a valid lock |
| `7,8,12,13,17` | Primary audit: calibrated `stimLick+` | Evaluator-only after a valid lock |
| `21,22,23,24` | Boundary audit: supraphysiological `stim+Lick+` | Optional post-primary stage in the same evaluator transaction; never rescues the primary result |

The mouse is the only inferential and resampling unit. Trials and sessions may
improve a within-mouse estimate but never increase the biological sample size.

Before this specification was frozen, a local structural check exposed two
per-record diagnostics: cumulative stimulated-trial count and a contingency-
consistency indicator. They are treated as exposed regardless of viewing and
are permanently excluded from scoring, selection, audit classification, and
boundary interpretation.

## Primary estimand

For each mouse, the provisional raw endpoint is the change in preparatory lick
probability from the first to the eighth training session on a prospectively
defined eligible-trial set. Let `raw_gain` denote that mouse-level late-minus-
early probability change. The primary estimand is the raw between-arm contrast:

```text
delta_rate_raw = mean(raw_gain | stimLick-) - mean(raw_gain | stimLick+)
```

For the secondary scale-free companion, let `development_scale` be the frozen
sample standard deviation of the nine development-control raw gains and define

```text
delta_rate_std = delta_rate_raw / development_scale
```

The audit must report both quantities with their units. `delta_rate_raw`, in
probability-gain units, remains the primary estimand. `delta_rate_std`, in
development-SD units, is a deterministic companion rather than a second
estimand. Any standardized margin must be the exact transform of the presigned
raw-unit margin. A positive rate-gate-consistency terminal requires that
raw-unit margin, adequate development-scale reliability, the predicted
direction in both arms, and leave-one-mouse-out stability. The arm-direction
guards are raw deviations
from the frozen development center: the `stimLick-` mean must exceed that
center by its presigned arm margin, and the center must exceed the `stimLick+`
mean by the same margin unless separate raw thresholds are explicitly frozen.
Any standardized arm guard or leave-one-mouse-out floor is obtained by dividing
its raw-unit counterpart by `development_scale`.

Before any audit opening, the binding rule must freeze four raw-unit fields:
`practical_margin_raw`, `arm_margin_raw`, `development_scale_floor_raw`, and
`loo_delta_floor_raw`. These names are canonical in both the policy and the
provisional calibration implementation; synonymous threshold fields are not
permitted.

This estimand is not yet identified from the release alone. `trialID`,
`seshID`, `lickState`, eligible-trial denominators, missingness, and the
701-sample time axis need authoritative definitions. Provider-derived summary
fields are parity checks only.

The paper reports randomization within repeated cohorts of two to four mice
and four removals after initial collection for poor signals associated with
mistargeted fibres or insufficient expression. It also reports that the
experimenter was not blinded during data collection. The release omits the
block roster, within-block allocations, and the excluded animals' groups and
timing. Thus the `C(11,6)=462` unblocked enumeration is only a conditional-
exchangeability sensitivity unless the design record is recovered; it is not
presently a design-exact intention-to-treat analysis.

## Adaptive-search contract

The bounded grammar compares adaptive-rate ACTR, its constant-rate nested
null, signed-PE ACTR, on- and off-policy TD/RPE, direct contingent
reinforcement, an amplitude-threshold hybrid, and integrity anchors. Author
code is a specification oracle; empirical analysis requires an independently
qualified implementation.

The policy permits 28–48 valid whole-mouse LOSO trials, including 12 fixed
branch-coverage slots, at least 8 adaptive successors, and at least 8
falsification or ablation trials. Patience is 10 qualifying trials after all
depth gates. Resource ceilings are 800 CPU-core-hours, 0 GPU-hours, and 120
wall-clock hours.

There may be one aggregate evaluator opening after configuration lock. The
optional high-amplitude boundary diagnostic is contained inside that same
transaction. Audit feedback cannot update search or expose individual mice.

The 28–48-trial search is the retained operative specification, not a
justification for broad discovery. It cannot guarantee that the finite panel
exhausts modern dopamine accounts. For example, the [2024 FLEX
study](https://www.nature.com/articles/s41467-024-50205-3) provides a distinct
temporal-representation account, and [Mah, Golden and Constantinople
(2024)](https://doi.org/10.1016/j.celrep.2024.114840) study dopamine and learning
rates in expert rats. [Action-prediction-error work
(2025)](https://www.nature.com/articles/s41586-025-09008-9) studies value-free
stimulus–action learning in tail striatum, a different circuit and task.
These studies delimit generalization; none is silently added to the active
grammar. Results here cannot exclude these accounts or adjudicate circuit,
task, or learning-stage differences.

## Outcomes that would matter

| Eventual outcome | Scientific interpretation |
| --- | --- |
| Robust positive raw contrast and all frozen target guards pass | The protocol pattern is consistent with the locked rate-gate account in this task; novelty depends on the additional discrimination/robustness evidence, not repeating the direction. |
| Target and a qualified rival predict the same pattern | The intervention pattern is not mechanism-unique within that panel, even if the target is the best control predictor. |
| Positive contrast without the two arm-direction or influence guards | A protocol difference only; not the full rate-gate-consistency pattern. |
| A valid reverse-direction result passes the frozen reverse rule | Evidence against the locked target's arm pattern, not proof of a universal signed-error mechanism. |
| Valid low-power nonrejection or overlapping prediction uncertainty | Unresolved, not equivalence, no dopamine effect, or a null mechanism. |
| Endpoint or assignment information cannot be resolved | A bounded identification limit of this release; not biological evidence against a mechanism. |

The development controls are not randomized arm-versus-control confirmation.
Their center is a frozen reference, so the arm guards do not establish two
separate causal effects. The missing randomization and exclusion information
also remains consequential even if the observed contrast is large.

## Portable prospective implementations

[`COHORT_MAP.json`](COHORT_MAP.json) is a sanitized, outcome-free mapping from
published one-based source indices to the four evidence roles. The Phase-0
inventory and role materializer can consume that mapping, but neither their
generated metadata nor an exact implementation is required for readiness.

[`outputs/code/calibrate_small_n.py`](outputs/code/calibrate_small_n.py)
provides a generic whole-mouse stress calibration.

[`outputs/code/calibrate_endpoint_small_n.py`](outputs/code/calibrate_endpoint_small_n.py)
and its unit test implement a reusable outcome-blind, provisional endpoint-
shaped binomial/beta-binomial calibration engine. It models ties, variable
denominators, overdispersion, missingness, one-arm alternatives, a raw-unit
margin, and independent selection and validation streams.

Neither engine is a decision rule. A binding threshold may enter the final
policy only after endpoint semantics, design support, safety/power criteria,
and scientist approval are frozen without inspecting audit outcomes. Any
generated tables and reports stay in runtime storage. No calibration result is
tracked in this episode.

## Candidate-scoring and audit-opening conditions

1. Provision the role-separated, no-reacquisition access boundary in
   [`outputs/firewall/FIREWALL.md`](outputs/firewall/FIREWALL.md).
2. Freeze endpoint, trial, session, missingness, denominator, and time-axis
   semantics from authoritative documentation.
3. Recover the randomization and exclusion roster, or explicitly limit the
   episode to a noncausal unblocked sensitivity analysis.
4. Freeze the raw effect margin and development-scale reliability floor.
5. Independently validate endpoint-faithful calibration; amend the policy only
   if an eligible rule exists and the scientist approves its terminal wording.
6. Qualify the implementation, finite search space, resources, and
   no-partial-output evaluator.
7. Freeze a versioned, write-once final configuration and obtain required
   audit authorization.

Apply each condition only at the stage it governs. Source qualification and
outcome-blind implementation work do not wait for calibration, final-lock, or
audit-opening controls. Development scoring requires the endpoint definitions,
development-role isolation, and the policy's existing pre-development timing
for the four presigned raw thresholds. Audit outcomes remain closed
until all audit-opening conditions pass.

## Claim boundary

Any eventual result is a publication-exposed, role-sequestered reanalysis of
this task, preparation, cohort, and frozen contract—not a literature-blind
replication or a universal claim about dopamine. The existing release supports
at most a complete-case protocol contrast under explicit assumptions until
the assignment/attrition record is recovered. Mechanism-consistent behavior
does not identify a unique neural learning algorithm.

## Current state

No adaptive trial, configuration lock, audit opening, reward, or scientific
result exists. Endpoint semantics, the documented assignment/exclusion roster,
role separation, and the endpoint-specific binding rule remain unresolved.
The four raw threshold fields remain null; audit opening and positive-terminal
authorization remain disabled. Historical tools and dated attempts are kept
as implementation history, not extra launch gates.
