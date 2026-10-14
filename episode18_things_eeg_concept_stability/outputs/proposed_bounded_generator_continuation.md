# Adopted bounded continuation of the frozen false-partition generator

**ADOPTED October 13, 2026 at 21:43:13 UTC.** The scientist answered
"Adopt the bounded continuation (Recommended)" to the concrete one-job,
13-CPU/96-GiB/ten-hour proposal with a new 24-hour search-only window.
The new window ends **2026-10-14 21:43:13 UTC**, epoch `1792014193`.
The original **2026-10-13 21:30 UTC** deadline, epoch `1791927000`, and all
prior charges remain preserved. The [separate authority](config/bounded_generator_continuation_20261013_authority.json)
activates only this checkpoint continuation. No EEG fitting, full-study clock
reset, score release or audit opening follows.

## Recommendation and scientific purpose

Continue the saved `sub-07` generator once, for at most ten allocation hours.
This can resolve substantially more of the first required bank member's
already frozen search budget. It cannot promise a complete control bank or
completion of the thirty-participant study.

The immediate bottleneck is image/context matching. No checkpoint has passed
the necessary mean-matching condition, so the producer has performed zero
full distribution checks. The annealing search's accepted swaps are not
accepted false partitions. Retain all six feature families, all six folds,
eight bank members, sixty-four starts per member, 100,000 legal proposals per
start, seed `20261012`, objective, temperature schedule, acceptance order,
matching tolerances and diversity rule. No person, group, descriptor or
criterion is removed.

## Permitted operational evidence and estimates

The following are public operational reports; no checkpoint array or EEG
score was inspected to prepare this proposal.

| Evidence | Recorded fields |
| --- | --- |
| [First search report](runs/false_partition_preparation/search_operations_20261013_001/initial_47672861/search/report.json) | `total_legal_proposals=478972`; first member remained unaccepted. |
| [Latest search report](runs/false_partition_preparation/search_continuation_operations_20261013_001/segment_47679999/search/report.json) | `total_legal_proposals=1164556`, `total_accepted_proposals=1151838`, `elapsed_seconds=3537.784232854843`, `member=0`, `start=11`, `swaps=64556`, `exact_checks=0`, `status=resource_incomplete`, `checkpoint_available=true`, `process_peak_rss_bytes=74025005056`, `array_admission_bytes=76700852096`. |
| [Latest allocation receipt](runs/false_partition_preparation/search_continuation_operations_20261013_001/segment_47679999/exit.json) | Thirteen allocated CPUs, 96 GiB, one numerical worker, producer exit75, no counter reset or repeated preparation. |
| [Retained first core-grid cost](runs/baseline_models/core_scalar_operations_20261013_001/initial_47622384/operational.json) | Twenty-nine of 793 scalar fits, zero complete penalties or validation sides, deadline stop. This remains a separate, unadmitted complete-study cost dependency. |

The second segment advanced `1164556 - 478972 = 685584` legal proposals in
3,537.8 seconds: approximately **194 proposals/second**, including its startup
and checkpoint overhead. The current first member has
`64 * 100000 - 1164556 = 5235444` proposals remaining before its fixed cap,
approximately **7.5 hours at that observed effective rate**. A ten-hour
allocation with a nine-hour cooperative deadline supplies useful margin.

The unchanged eight-member search permits at most 51.2 million proposals.
Its remaining proposal budget is 50,035,444, approximately **71.7 hours** or
**932 allocated CPU-hours at thirteen CPUs** at the same rate. This is a
worst-case proposal-count extrapolation, **not a guaranteed runtime bound**:
full distribution checks have not yet been reached, and their costs and the
probability of accepting a member are unmeasured. Approximately 98.9% of
proposals being accepted by annealing does not demonstrate matching progress.

An analogous maximum proposal exposure for all thirty development people
would be about **28,600 allocated CPU-hours** at this rate and allocation,
before EEG fitting, exceeding the existing 12,000 CPU-hour envelope. Earlier
acceptance could make actual work much smaller, but the current evidence
does not support a whole-cohort dispatch or a complete-study deadline reset.

## Proposed finite allocation and accounting

- One job: **13 requested CPUs, 96 GiB, ten hours, no GPU**, one numerical
  worker and all numerical thread counts one. Use coordinator-verified
  eligible partitions at submission; prior execution used `russpold,normal`.
  The CPUs chiefly satisfy the observed memory allocation requirement;
  additional cores do not accelerate this serial generator automatically.
- Retain the existing 90-GiB RSS guard and 80-GiB array admission cap. Existing
  process and allocation evidence supports 96 GiB; no lower-memory rewrite or
  repeated cache preparation is proposed.
- Reserve **130 allocated CPU-hours** for thirteen CPUs, or a conservative
  **160 CPU-hours** if the actual allocation is rounded up to sixteen. Refuse
  a larger allocation until its cost is separately admitted. Replace the
  reservation with actual parent allocated CPU-seconds once, including steps.
- Preserve prior recorded consumption **137.231481481 CPU-hours** and the
  conservative shared-parent bound **142.240092592 CPU-hours**. Recorded plus
  thirteen-CPU reservation is 267.231481481 CPU-hours; conservative bound plus
  sixteen-CPU reservation is 302.240092592 CPU-hours. Previous consumed work
  and released reservations are not reset or charged twice.
- The cooperative stop is the earliest of allocation-start plus nine hours,
  allocation-end minus sixty seconds, and the newly adopted absolute deadline
  minus sixty seconds. Retain forty seconds for checkpoint/report cleanup and
  five seconds kill grace inside that tail. No requeue, automatic successor,
  repeated start or RNG reset follows.

The adopted **search-only window is twenty-four hours from adoption**, ending
October 14 at 21:43:13 UTC. It includes queue time and the single bounded job;
if insufficient runtime remains when allocated, do not silently extend it.
Preserve the original deadline in the historical record.
This proposal does not restart the full study's 336-hour clock or authorize
EEG fitting after the original deadline.

## Exact predecessor and implementation consequence

Resume only this preserved checkpoint:

`outputs/runs/false_partition_preparation/search_continuation_operations_20261013_001/segment_47679999/search/checkpoint.npz`

Its adjacent `report.json` must continue to establish the terminal resumable
resource stop. Reuse the unchanged cache at
`outputs/cache/false_partition_sub07_20261012_001` and frozen configuration
`outputs/config/false_partition_sub07_20261012_001.json`. Write a fresh output
segment; preserve every prior artifact.

The actual producer is [outputs/code/false_partition_search.py](code/false_partition_search.py),
and the prior wrapper is
[outputs/code/run_false_partition_search_continue001.sbatch](code/run_false_partition_search_continue001.sbatch).
The latter hardcodes the earlier predecessor47672861 and the original
deadline, so it is **not a ready command for this proposed continuation**.
There is no corresponding root-level `code/` wrapper.

Changing `not_after_unix` in the configuration or editing the frozen producer
would change the identities bound to both cache and checkpoint:
`load_config` includes the producer's source hash; `load_search_cache` and
`restore_checkpoint` require the original complete configuration digest.
Do not rewrite those identities or regenerate the cache to hide a deadline
change.

The reviewed [adapter](proposed/false_partition_continuation/continue_search.py)
and [finite job template](proposed/false_partition_continuation/run_continuation.sbatch)
record new deadline authority separately and call the unchanged producer's
`load_config` and `run_search`. They require a separately adopted authority
outside the proposal directory before importing the producer. That authority
file was created only after the scientist's adoption above. Original source/configuration/cache/checkpoint identities,
source admission checks, complete counters, assignments, accepted members and
RNG state are retained. Execution beyond the original deadline is disclosed
explicitly; the original CLI and configuration remain unchanged.

Independent source review confirms that `run_search` accepts an external budget
check without reading `not_after_unix`, and restores the saved PCG64 state,
assignments, members, starts, proposal counters and rolling moments. The
adapter also preserves the original CLI's compute-node check, all six
one-thread environment checks, memory-cap relationships, fresh output directory,
predecessor validation and terminal resource/checkpoint report finalization.
Calling the numerical function alone would omit these safeguards. This review
establishes source-level consistency, not control qualification or an EEG result.

On October 13 at 21:40 UTC, coordinator-owned compute step `47377474.2` on
`sh02-15n09` passed all three [lifecycle tests](proposed/false_partition_continuation/test_continuation.py)
and the wrapper's shell syntax check. The tests use temporary stand-ins only:
they verify refusal before import, original configuration identity with a new
execution deadline, and distinct resource-stop/exhaustion/error reports. They
also cover small Slurm timestamp skew and rounded CPU allocation while retaining
the nine-hour hard work bound. See [test evidence](runs/software_integration/proposed_continuation001/).
These are software checks, not checkpoint-payload validation, resumed search,
control qualification or EEG evidence. Their cost is included once in the
existing EP21 parent; no new EP18 scientific reservation is charged.

## Terminal conditions and interpretation

1. **Any member exhausts all sixty-four starts without acceptance:** stop with
   `exhausted_unavailable`, as the existing producer does. For the current
   first member this is the earliest prospective definitive generator stop;
   do not reset its starts or proceed as though the missing member existed.
   The registered generator could not supply all eight controls within its
   frozen proposal budget. This supports the policy's underidentified-control
   disposition, not an EEG null or proof that no possible matched grouping
   exists.
2. **All eight image/context candidates complete:** stop this job successfully
   and review the full bank's operational cost. One accepted first member
   alone is not a complete bank. Selected-design spectrum, leverage and
   effective-degrees-of-freedom rechecks remain mandatory; even eight candidates
   are not yet a fully qualified false bank. No automatic EEG execution or
   participant expansion follows.
3. **Time, memory or signal stop before either event:** preserve the checkpoint
   and report `resource_incomplete`. This does not exhaust the generator and
   does not establish scientific underidentification or any EEG effect.
4. **Unexpected error or invalid predecessor:** stop and preserve evidence;
   no automatic retry or tolerance relaxation.

Operative sources: [search policy, false partitions](../SEARCH_POLICY.yaml),
[frozen deadline](config/false_partition_sub07_20261012_001.json),
[latest execution record](experiment_log.md#2026-10-13-1402-pacific--false_partition_preparation--continuation-reached-its-bounded-time-stop),
and producer `run_search` (lines672–733). The generator and scientific criteria
remain unchanged; only the separately adopted execution window differs.
