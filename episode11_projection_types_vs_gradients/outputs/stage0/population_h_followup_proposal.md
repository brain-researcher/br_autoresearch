# Prospective H recovery follow-up — single revised attempt submitted

## Current execution — explicit wall-clock reset, 2026-09-30

The user reset the120-hour wall clock from September30,18:12:23 PDT
through October5,18:12:23 PDT. Prior CPU usage and the1500-core-hour
ceiling remain unchanged. The prepared empirical_perturbation_v1 attempt
was submitted once as Slurm job46143347,3CPUs/16GB/72h, with the absolute
deadline enforced by Slurm and automatic requeue disabled. Initial observed
state is PENDING; no new H recovery result exists. Passing initializer
regression is reused. Research question, data boundaries, scientific gates
and numerical caps remain unchanged. Prior gate wording below is history
and is superseded by this explicit amendment.

Clock record: `outputs/stage0/episode_wall_clock_reset_20260930.json`.
Actual submission: `outputs/stage0/population_h_empirical_perturbation_v1_submission.json`.
Next action: follow the same job; preserve and assess its actual result,
or stop on failure/cutoff. No automatic successor or real-outcome access.


The user instructed “continue” after this proposal, authorizing its
implementation and focused regression. The initializer revision is now
implemented and its bounded regression passed; no revised H optimizer run
has been submitted. The user's budget reply, “as much as possible,” is
interpreted within the original ceilings and supplies no clock origin or
verified remaining allowance. It preserves the
Episode 11 question and targets the diagnosed initialization failure of
`population_h_optimizer_recovery`, rather than removing an empirical recovery
challenge. The active Goal is not complete.

## Evidence and prediction

The completed frozen attempt `45736553` failed because its raw empirical
start reached 1,200 iterations and the best-two converged H fits differed by
0.129459 nats/neuron on fresh draws against a 0.002 tolerance. The best
truth-initialized H fit approximated planted truth. The source result is
`population_h_recovery_summary.json`; the scientific report and QC figure
remain the actual evidence. No new observation is claimed here.

The smallest existing-method candidate is to perturb the **empirical** start
using the already-defined initializer helper. Its directional prediction is
that an empirical seed perturbed within the same identified parameter space
may escape the poor trajectory while remaining independent of planted labels
and signatures. That causal explanation and the benefit of the perturbation
are unproven. Another cap failure or fresh best-two gap at least 0.002 would
refute this proposed repair at the tested perturbation.

## Exact candidate change

Keep all three start types in the same order:

1. Exact fitted-C embedding, unchanged.
2. Nonzero truth-based diagnostic start, unchanged.
3. Empirical initialization from the same 16-replicate synthetic training
   prefix, followed by the existing deterministic empirical perturbation.

The existing `make_initial_starts` helper in
`tools/run_mixed_support_full_family_calibration.py` already defines that
perturbation. Prospectively fix its H seed at `20269193` (the existing
base-seed-plus-R rule: `20261001 + 8192`), stream `702`, perturbation number
`1`. Its Gaussian standard deviations are `0.035` for parameter blocks and
`0.009` for `log_sigma`; base-block perturbations are projected through
`pinv(single_design) @ single_design`, then coefficients are clipped to the
existing bounds. No seed or perturbation is selected after new outcomes.

The implementation calls the existing helper with
`count=3`, that fixed seed, and `embedded_h_start=embedded`. It returns the
three desired starts, preserving the empirical recovery challenge. The
runner's legacy default must remain available, with an explicit prospective
attempt identifier such as `empirical_perturbation_v1`. The focused regression
verifies unchanged first/second starts, the helper-defined third start,
and unchanged caps/gates. This focused regression passed and is retained at
`population_h_empirical_perturbation_v1_regression.json`; it used one bounded
synthetic replicate and no optimizer. It is implementation evidence, not a
new recovery experiment or proof that the candidate repairs H recovery.

Retain R=8192, diagnostic amplitude 1.25, the fitted-C input checkpoint,
continuous/group families, final joint likelihood and marginal scoring,
16-replicate initialization prefix, 32-replicate preconditioner prefix,
1,200 main iterations, 600 polish iterations, all physical KKT/boundary
criteria, and all three acceptance thresholds `0.005/0.002/0.010`.
Do not substitute another truth-informed start for the empirical start.
Even a pass would close only this optimizer diagnostic; production recovery,
Panel A, D12-to-A8 transport and real-data access would remain unqualified.

The selected revision freezes verification stream `5602` and exact-anchor
stream `5702` before launch; `5601/5701` have been inspected. Retain the failed
attempt unchanged and preserve a new result under a distinct name, such as
`population_h_empirical_perturbation_v1_summary.json`.

## Resource decision required before execution

The original global ceilings remain 1,500 CPU core-hours, 120 wall hours and
zero GPUs, including qualification and replay. The four scheduler attempts
currently documented here account for 76.8833 active job hours and 230.6444
allocated CPU core-hours; these are partial accounting that excludes earlier
qualification and does not establish the global wall-clock origin.

Do not reuse the prior 72-hour request automatically. If the global wall
ceiling counts summed active runtimes, the known attempts alone leave at
most 43.1167 hours before earlier qualification is deducted. If it counts a
continuous interval, the existing origin must be supplied. The selected
request must fit the confirmed remaining wall/core allowances, retain at
most three CPUs and 16 GB, and remain within the original ceilings. Runtime
improvement is unproven; the failed empirical start took about 53 hours.
There is no justified promise that this candidate will fit a smaller window.

The start revision has been selected and implemented. Launch still requires
the existing global clock origin/counting rule and complete usage, or verified
remaining wall/core allowances. Read-only reconciliation found neither in
the permitted episode records. No safe positive allowance is established;
43.1167 hours remains an upper bound under an assumed summed-runtime rule,
not a permitted duration. No resource amendment is inferred from “as much as
possible.” The actual remaining resource gate cannot be cleared by a passing
initializer regression.

## Additional recorded scheduler accounting — same resource gate

A single bounded own-user allocation query on 2026-09-30 found 30 positive
runtime EP11-attributed records and 15 never-started records. Their totals
are 81.01916667 active job-hours and 239.87916667 allocated core-hours,
superseding the four-job lower bound above. Returned fields and explicit
limitations are retained in `population_h_budget_accounting_20260930.json`.
The conditional summed-runtime wall upper bound is now 38.98083333 hours.
The earliest named EP11 start to the completed H end spans 140.26888889
hours; neither job names nor a later-date filter determines the original
v1/v2 clock origin. A continuous clock may already be exhausted. No duration
is authorized by these bounds, and no new job was submitted.
