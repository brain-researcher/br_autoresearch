# Unused remaining-participant execution plan

This source preparation selects no job, walltime, concurrency or added budget.
Do not use it unless the original complete sub-07 cost job47591674 succeeds and
the coordinator records a finite resource selection from its observed time,
memory and output costs. An incomplete or failed cost does not admit scaling.

Coordinator and independent source reviews found no concrete defect on
October12,19:54Pacific. No shell syntax check or execution has occurred.
This source-readiness finding does not substitute for complete cost evidence.

The existing initialized root is
`outputs/cache/private/core_reference_20261013_001`. Keep its recipe, release,
all30 input binding and completed sub-07 result unchanged. The new unused
`outputs/code/run_core_reference_remaining001.sh` calls the existing participant
and finalizer CLIs; it never initializes again, runs tests or changes model code.

Use one28-element array with exact index0..27 and fixed participant order:
sub-08, sub-09, sub-10, sub-11, sub-12, sub-13, sub-14, sub-20,
sub-21, sub-22, sub-24, sub-25, sub-26, sub-27, sub-28, sub-29,
sub-30, sub-31, sub-32, sub-34, sub-35, sub-38, sub-43, sub-44,
sub-45, sub-46, sub-47, sub-48. A separate substantive parent fits sub-49
after successful completion of the whole array, then serially invokes the
existing all30 finalizer. This preserves a single final writer without a tiny
standalone finalization job, waiting loop, automatic submission or retry.

Every participant retains all36 splits and438 multiresponse fits, the complete
core64/cosine12/five-family recipe, one numerical thread,2CPU/15GiB allocation,
12GiB observed-RSS guard,10GiB recipe array admission,2GiB private output and
128MiB per-record limit. The sub-49 parent reserves60seconds of its scientific
window for finalization and an additional1MiB final record allowance. Both
commands share its actual allocation clock. The all30 calibration remains
private, carries zero search credit and does not release concept, false-group
or audit comparisons.

The coordinator must choose and record these explicit positional arguments:

1. `ALLOCATION_SECONDS`: one common, measured-cost-supported walltime for all29
   substantive parents, matching the explicit Slurm time request.
2. `ARRAY_CONCURRENCY`: an integer1..28, matching `--array=0-27%N`.
3. `OTHER_RESERVED_EP18_CPUS`: concurrently reserved EP18 cores outside this
   continuation; `2*N + other <= 128` must hold.
4. `REMAINING_EP18_CPU_SECONDS`: ledger headroom after existing charges and
   reservations. This29-person plan reserves `29*2*ALLOCATION_SECONDS`, which
   must fit that headroom within the original12000CPUh ceiling. These are
   coordinator snapshots, not claims that a worker queries live concurrency.

The wrapper's full signature is:

```text
run_core_reference_remaining001.sh participant ALLOCATION_SECONDS ARRAY_CONCURRENCY OTHER_RESERVED_EP18_CPUS REMAINING_EP18_CPU_SECONDS none
run_core_reference_remaining001.sh last ALLOCATION_SECONDS ARRAY_CONCURRENCY OTHER_RESERVED_EP18_CPUS REMAINING_EP18_CPU_SECONDS ACTUAL_ARRAY_JOB_ID
```

Submit from the episode directory using the site's discovered partition,
`--nodes=1 --ntasks=1 --cpus-per-task=2 --mem=15G --time=<explicit selection>
--no-requeue`. The first submission also has the selected28-element array;
the second has `--dependency=afterok:<actual array job ID>` and no array.
The coordinator verifies that dependency and records the actual IDs. A held
submission may be used while binding them; this document creates none.
Reuse the observed cost/check records instead of adding another validation run.

Precreate only the operations parent
`outputs/runs/baseline_models/core_reference_remaining_operations_20261013_001`.
Give Slurm stdout/stderr unique paths there using `%A_%a` for the array and
`%j` for the final parent. Do not precreate private participant directories.
Each is exclusively created by the existing CLI; any prior attempt refuses.
The operations directories also include participant and actual job identity.
Only the final parent claims `finalizer_claim`; preserve it on failure.

Every deadline is the minimum of actual StartTime plus the selected walltime
minus60seconds, actual EndTime minus60seconds, and1791927000 minus10seconds.
The existing cooperative deadline plus bounded TERM/KILL grace stays inside
the original global deadline. No increase in walltime, tolerance, model scope
or retries follows automatically from a stop. Actual throughput under shared
I/O and participant variability is not established by source readiness.
