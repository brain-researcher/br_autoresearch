# Episode 02: Causal Identification of Dopamine's Learning-Rate Role

## Status

EP02 is a formal local research specification. No adaptive trial,
configuration lock, audit opening, reward, or scientific result exists. An
explicit scientist task may implement and qualify the episode, while audit
outcomes remain closed until its scientific and isolation requirements pass.
The machine-readable contract is
[`SEARCH_POLICY.json`](SEARCH_POLICY.json).

The scientific readiness questions are whether the source records map to the
declared mice and roles, the endpoint can be scored, the development and audit
roles are separated, and the evaluator can return only the prespecified
aggregate result. Extra checksum, schema, receipt, or attestation artifacts do
not establish readiness and are not required.

The repository retains portable Phase-0, role-materialization, calibration,
and firewall-verification implementations created during specification work.
They are optional legacy helpers, not launch gates; a simpler implementation
may be used if it preserves the same scientific role and access boundaries.

## Scientific question

At physiological amplitude, does phasic mesolimbic dopamine multiply the
magnitude of an otherwise signed policy update—an adaptive learning-rate
gate—or does it itself provide a signed teaching or reinforcement signal?

| Explanation | Prediction before audit |
| --- | --- |
| Adaptive learning-rate gate | Relative to controls, learning is enhanced for `stimLick-` and suppressed for `stimLick+` without dopamine setting update sign. |
| Signed prediction-error / TD signal | Effects follow the sign and timing of a teaching signal rather than a purely multiplicative rate change. |
| Direct reinforcement | Stimulation reinforces the action or state without a bidirectional rate account. |
| Amplitude-dependent hybrid | Physiological and supraphysiological stimulation obey different rules. |

Every model family must freeze its counterfactual predictions before any
intervention outcome is opened.

## Evidence roles

| Source records, one-based | Role | Exposure rule |
| --- | --- | --- |
| `1,2,4,6,9,11,15,16,19` | Development controls | Fully exposed; never confirmation |
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
defined eligible-trial set:

```text
delta_rate = mean(gain | stimLick-) - mean(gain | stimLick+)
```

The audit must report the raw probability-gain contrast as well as any
development-standardized version. A positive mechanistic terminal requires a
presigned raw-unit margin, adequate development-scale reliability, the
predicted direction in both arms, and leave-one-mouse-out stability.

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
audit-opening controls. Development scoring requires the endpoint definitions
and development-role isolation it actually uses. Audit outcomes remain closed
until all audit-opening conditions pass.

## Claim boundary

Any eventual result is a publication-exposed, role-sequestered reanalysis of
this task, preparation, cohort, and frozen contract—not a literature-blind
replication or a universal claim about dopamine.
