# EP10 original M2: inactive failure-telemetry preparation

Prepared October 13, 2026. **Unactivated and untested. No execution selected.**
This is a source-only follow-up to
[`proposed_original_m2_reactivation.md`](../../proposed_original_m2_reactivation.md).
It changes neither the adopted incomplete closeout nor any original numerical
source, scientific threshold, model family, data role, resource ledger or ASTRA
recipe. This proposal has no command-line entrypoint or submission wrapper.

## Specific finding

The four preserved failures in
[`runtime.log`](../../stage0/v3_m2_optional_family_qualification_job_46237243_terminal_20261003/runtime.log)
are iteration-cap errors in the same low-rank engine. The rank-8 hybrid fails
in its rho=0.1 low-rank anchor; its alternating solver was not reached. The
other failed low-rank ranks are 4, 6 and 8 at rho=0.001. There is no terminal
gradient, relative objective change, accepted step or iteration history in
those historical error strings, so their numerical cause remains unresolved.

The concrete telemetry loss is in the qualification wrapper:

- [`_run_recovery_case`](../../stage0/tools/qualify_v3_m2_optional.py#L493)
  allows exceptions to leave the child process without saving scalar failure
  state. ProcessPool exception transport does not preserve live Python frames.
- [`_parallel_recovery_gates`](../../stage0/tools/qualify_v3_m2_optional.py#L673)
  saves per-case JSON only for successes, prints failure strings, and continues
  waiting for all workers. Cancellation preserves printed failures but cannot
  produce the full report through its normal success-only write path.
- The low-rank cap error was already enriched on October 3 with the missing
  final scalars. See [`fit_v3_m2_optional.py`](../../stage0/tools/fit_v3_m2_optional.py#L2966)
  and the existing [log](../../experiment_log.md). Re-adding those words is not
  a repair and cannot recover the historical values.

## Proposed minimal mechanical repair

[`failure_telemetry.py`](failure_telemetry.py) wraps the unchanged worker inside
its child process. Successful results are returned untouched. On an ordinary
exception it reads an explicit allowlist from the original optimizer traceback,
writes a fresh diagnostic JSON, then reraises the original exception. It adds
no iteration callback, objective evaluation, trajectory logging, alternative
optimizer, test fixture or retry. It records no state or data arrays and refuses
production-objective frames. Independent review identified that free-form exception
text could bypass this boundary; the helper now records an exception class and
fixed reason code only. Diagnostic-failure notes likewise omit exception text. Diagnostic-writing failure is attached as an error
note and does not turn a failed optimizer into a pass.

The record distinguishes the U and d gradient gates, the relative-objective
gate, current trial step, whether the saved current state is the accepted trial,
and cumulative line-trial count. `active_rho` distinguishes a hybrid anchor
failure from its later optimization. It is neither a resumable checkpoint nor
a qualified result. It does not catch a hard kill or create missing intermediate
states; catching a signal would be a separately selected wrapper behavior.

Future caller integration, **not applied here**:

```python
run_case_with_failure_record(
    original_qualification._run_recovery_case, family, rank,
    optimizer_path=original_optional.__file__,
    failure_path=fresh_directory / f"{family}_rank_{rank}_failure.json",
)
```

The original qualification worker enforces Slurm and one-thread numerical
settings. This wrapper must run in the worker, before exception serialization;
calling it around the parent's `future.result()` is too late. Each parallel
case needs its own fresh filename. Any future check should focus on preservation
of the original returned object/exception and capture of the whitelisted failure
fields; no new recovery campaign is selected by preparing this helper.

## What source review does and does not establish

The edge-to-symmetric gradient adjoint, Stiefel projection, QR sign handling,
coordinate canonicalization and summed-gradient Armijo derivative remain
consistent with the documented objective. No concrete algebra defect or
convergence-preserving acceleration was established in this review.

The unit step and coupled U/d coordinates can converge slowly when scales
differ. At sufficiently small steps floating-point Armijo decrease can round to
zero; the source permits equal rounded objectives even while the gradient gate
fails. These are possible explanations, not diagnoses of the four old runs.
The proposed scalar record can separate a failing gradient block from an
objective-change-only failure or an apparent objective plateau; it cannot alone
prove roundoff stagnation or supply a better nonconvex optimizer. Increasing
iterations or changing the step rule would change the frozen method and is not
part of this telemetry proposal.

## Remaining blocker and original question

The original objective remains equal-group held-out **M2-minus-M1 conditional
recorded-positive-pattern log score**, on the same 194 cells, 12 conservative
groups and 16 target coordinates, with unknown zeros, required families,
40–88 valid comparisons, 16 anchors, falsifiers and all 99 complete controller
nulls. The terminal-profile proposal is not a substitute. The current M1
[independent postflight](../../stage0/v3_m1_real_fit_single_thread_v3_job_46332190/independent_postflight.json)
already passes all 949 fits/91 checkpoints and should not be repeated here.

The preserved [resource diagnostic](../../stage0/v3_m2_resource_lower_bound_job_46425841_isolated_cpu3_attempt002/resource_lower_bound.json)
is explicitly an interrupted 300-second lower-bound observation, with setup
and all other costs omitted: at least 1,900 additional allocated CPU-hours for
the old implementation. It is not a completed timing unit or an upper bound.
The ledger retains 735.711222 CPU-hours spent and 1,064.288778 remaining. Thus
even that lower bound exceeds remaining allowance by 835.711222 CPU-hours;
the standing willingness to increase compute supplies no finite run estimate.
The previous triangular-solve repair left whole-path time effectively unchanged.

No launch is ready: the low-rank cause and feasible complete-search cost remain
unresolved, and original-study reactivation is not granted. No tests, scientific
payload reads, scheduler calls, empirical/synthetic executions or contract edits
were performed in preparing these two files on login node sh04-ln04.

## Preparation review

Independent source review matched the captured locals to the actual optimizer,
confirmed child-side capture and preservation of successful returns/original
exceptions, and found no activation. Root corrected the one identified defect: 
free-form exception serialization. The helper remains untested and inactive;
no successful runtime check or convergence improvement is claimed.
