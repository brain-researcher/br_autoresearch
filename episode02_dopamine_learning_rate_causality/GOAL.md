# Episode 02: Causal Identification of Dopamine's Learning-Rate Role

## Status

EP02 is a formal local research specification. No adaptive trial,
configuration lock, audit opening, reward, or scientific result exists. An
explicit scientist task may implement and qualify the episode, while audit
outcomes remain closed until its scientific and isolation requirements pass.
The machine-readable contract is
[`SEARCH_POLICY.json`](SEARCH_POLICY.json).

Generated inventories, calibration packets, receipts, signatures, and
checksums are runtime records. Their presence alone is not evidence that an
audit-opening condition passed.

The repository does track portable, fail-closed implementations for Phase-0
inventory, deterministic role materialization, generic and provisional
endpoint-shaped calibration, and external firewall-receipt verification.
Their presence is capability, not evidence that any runtime gate passed.

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
inventory and role materializer consume that tracked contract but write all
inventories, arrays, commitments, and access receipts to runtime storage.

[`outputs/code/calibrate_small_n.py`](outputs/code/calibrate_small_n.py)
provides a generic whole-mouse stress calibration.

[`outputs/code/calibrate_endpoint_small_n.py`](outputs/code/calibrate_endpoint_small_n.py)
and its unit test implement a reusable outcome-blind, provisional endpoint-
shaped binomial/beta-binomial calibration engine. It models ties, variable
denominators, overdispersion, missingness, one-arm alternatives, a raw-unit
margin, and independent selection and validation streams.

Neither engine is a decision rule. Its local prerequisite inventory records
declared states and hashes but deliberately performs no cryptographic
authority verification. Authorized runs write generated tables and reports to
scratch or another runtime store. A binding threshold may enter this policy
only through a separately reviewed, cryptographically verified and signed
amendment after endpoint semantics, design support, safety/power criteria, and
scientist signoff are frozen. No calibration result is tracked in this
episode.

## Candidate-scoring and audit-opening conditions

1. Provision the permission-separated, no-reacquisition firewall in
   [`outputs/firewall/FIREWALL.md`](outputs/firewall/FIREWALL.md).
2. Freeze endpoint, trial, session, missingness, denominator, and time-axis
   semantics from authoritative documentation.
3. Recover the randomization and exclusion roster, or explicitly limit the
   episode to a noncausal unblocked sensitivity analysis.
4. Freeze the raw effect margin and development-scale reliability floor.
5. Independently validate endpoint-faithful calibration; amend the policy only
   if an eligible rule exists and the scientist signs its terminal wording.
6. Qualify the implementation, finite search space, resources, and
   no-partial-output evaluator.
7. Bind the final configuration and obtain required audit authorization.

Until every applicable condition passes, search and audit remain prohibited.

## Claim boundary

Any eventual result is a publication-exposed, role-sequestered reanalysis of
this task, preparation, cohort, and frozen contract—not a literature-blind
replication or a universal claim about dopamine.
