# H empirical-initializer revision: implementation evidence only

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


On 2026-09-30 the user instructed continuation of the proposed H initializer
revision. The existing helper now supplies the third, empirical start with
fixed seed 20269193, stream 702 and perturbation number 1. The exact fitted-C
embedding and truth-based start are unchanged. The legacy default remains
available; the new attempt is explicitly selected by
`--start-policy empirical_perturbation_v1` or the batch script's second
argument of the same name.

The focused regression passed using one bounded synthetic replicate and
the preserved actual C checkpoint. It verifies exact agreement of the first
two starts with legacy behavior, exact helper/independently reconstructed
perturbation agreement, repeatability, bounds and nonzero signatures, and
empirical-start independence from a changed truth-based initializer. It also
checks the 1200/600 caps and the existing failed iteration-cap/start-agreement
status paths. Batch syntax validation passed. Existing broad qualification
checks were reused. No H optimization, additional recovery experiment, real
outcome access or new figure ran.

Actual check artifact:
`population_h_empirical_perturbation_v1_regression.json`.

The revision freezes fresh verification/anchor streams 5602/5702, retains
R8192, all three start types, physical checks and thresholds 0.005/0.002/0.010,
and writes `population_h_empirical_perturbation_v1_summary.json` in a fresh
run directory. That recovery result does not yet exist. The original failed
job 45736553 and all its artifacts remain unchanged; the candidate's recovery
benefit is untested.

## Remaining execution gate

The user's resource reply was “as much as possible.” This is treated as
maximum use within the original global 120 wall-hours, 1500 allocated CPU
core-hours and zero GPUs. Permitted records contain no clock origin/counting
rule or complete prior qualification usage. The later single scheduler-accounting query found 30 positive-runtime
EP11-attributed allocations totaling 81.01916667 active job-hours and
239.87916667 allocated core-hours, still lower bounds because job-name
attribution can omit work. The resulting 38.98083333 hours is only an
optimistic upper bound under an unconfirmed summed-runtime interpretation,
not a safe allocation. Earliest named start through completed H end spans
140.26888889 hours; a continuous global clock may already be exhausted.
Returned accounting fields and limitations are preserved in
`population_h_budget_accounting_20260930.json`.

No job was submitted. The next action is to obtain verified remaining
wall/core allowance, then select an explicit duration within it. The batch
script's historical 72-hour header must not be reused automatically. There
is no demonstrated runtime improvement; the previous empirical start took
about 53 hours. Further execution stops at this resource gate, and a future
pass would not itself qualify Panel A or activate real-data analysis.
