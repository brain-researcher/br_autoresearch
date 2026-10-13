# EP12 full-search resource-scale audit

Status date: 2026-09-26. This audit uses policy, executor source, Slurm resource
requests, and explicit cost metadata only. It does not report or use scientific
outcome values.

## Why the earlier 2,750 core-hour estimate is not the full program

The earlier projection used one original complete T/U/M configuration cost of
2,750.224 CPU-seconds, multiplied by 36 ordinary configurations and 100 total
observed-or-null searches:

`2,750.224 seconds × 36 × 100 / 3,600 = 2,750.224 core-hours`.

That arithmetic omitted the mandatory post-36 composite control branches. The
authorization in `outputs/executor/DEVELOPMENT_BUDGET_OVERRIDE.yaml` therefore
authenticates an obsolete planning calculation, not the exact full 99-null
allocation that will be requested after procedure lock.

## Exact protocol counts before the parity correction

The frozen tail reaches its earliest lawful stop at trial ordinal 52 and its
maximum at 96. The active resource planner counted 328 complete T/U/M branches
per minimum search and 730 per maximum search. Across one observed and 99 null
searches, that is 32,800–73,000 complete triplets, or 98,400–219,000
model-family fits.

The previously quoted approximately 99,874–222,280 core-hour range is the
planner's conservative **busy-work** projection using about 10,962 serial
CPU-seconds per complete branch. It is not a finalized Slurm allocated-cost
receipt, is not an upper bound, and is not caused by accidentally multiplying
one-core work by 32. The allocated projection is computed separately from a
32-core parallel benchmark and is not yet available.

## Required observed/null control parity correction

The frozen control specification and the observed `ProductionFalsifierSuite`
implement the binary/vocabulary/rank control as:

- one weighted baseline;
- one binary branch; and
- the complete 3-vocabulary × 4-rank sensitivity set.

That is 14 complete branches. The null runner instead crossed both weighted
and binary modes with all 12 vocabulary/rank pairs, producing 24 branches, and
the resource planner inherited the same overcount. This is an observed/null
parity defect, not a reason to drop a required control.

Aligning the null runner to the frozen observed controller yields 308 complete
triplets at the earliest lawful stop and 690 at the maximum. The complete
100-search protocol then contains 30,800–69,000 triplets, or
92,400–207,000 model-family fits. Applying the same provisional serial-busy
rate gives approximately 93,784–210,100 busy core-hours for the complete
program. These remain planning estimates, not the exact authorization target.

## What must be measured before the 99-search dispatch

The production procedure must benchmark authenticated worker counts while
preserving byte-identical scientific results, then record both busy and
allocated core-hours. The full-null wrapper currently requests 32 CPUs and
192 GiB, but branch-level utilization is not uniformly 32-way: the null runner
parallelizes provider fits within a branch, while several control families are
small or analytic. A fixed 32-core request cannot be treated as efficient
without the required measurement.

The automatic chain must therefore stop at a self-hashed
resource-authorization-pending receipt containing the measured exact request.
It must not dispatch the 99-search array using either the obsolete 2,750-hour
estimate or the provisional busy-work range.

No search count, null replicate, model family, candidate, fold, type,
direction, control family, partner vocabulary, rank, stopping rule, threshold,
or final-data boundary is reduced by this correction.
