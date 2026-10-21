"""Preserve synthetic qualification scalars before a worker loses its traceback.

Activated October 20 as a child-worker diagnostic only; no scientific launch.
No CLI, imports of scientific modules, or optimizer patch.
The caller passes the original synthetic qualification worker and optimizer path.
This records failure only; successful calls are returned unchanged.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


FAMILIES = {
    "low_rank_population_recovery",
    "sparse_plus_low_rank_population_recovery",
}


def _finite(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _max_abs(value):
    if value is None:
        return None
    numbers = [float(item) for item in value.flat]
    if not numbers or not all(math.isfinite(item) for item in numbers):
        return None
    return max(abs(item) for item in numbers)


def failure_record(error, *, family, rank, optimizer_path):
    """Read an allowlist of scalar diagnostics from the original failing frame.

    Must be called inside the worker's except block, before ProcessPool serializes
    the exception. No state, features, labels, probabilities, or fixture arrays
    are returned. This is not a resumable checkpoint or a qualification result.
    """
    if family not in FAMILIES or type(rank) is not int or rank not in (1, 2, 3, 4, 6, 8):
        raise ValueError("Require an existing optional-family qualification case")
    optimizer_path = Path(optimizer_path).resolve()
    record = {
        "status": "failed_diagnostic_only",
        "family": family,
        "rank": rank,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "exception_type": type(error).__name__,
        "exception_reason": "optimizer_worker_exception",
        "source": str(optimizer_path),
        "numerical_method_changed": False,
        "qualification_pass": False,
        "resumable_checkpoint": False,
        "low_rank_frame_found": False,
    }
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if (frame.f_code.co_name == "_fit_low_rank_impl"
                and Path(frame.f_code.co_filename).resolve() == optimizer_path):
            local = frame.f_locals
            if local.get("objective_mode") != "synthetic_qualification":
                raise ValueError("Failure telemetry excludes production objective frames")
            current = local.get("current")
            spec = local.get("specification")
            state = local.get("state")
            relative = _finite(local.get("last_relative_change"))
            gradient = _finite(getattr(current, "gradient_infinity", None))
            threshold = _finite(frame.f_globals.get("LOW_RANK_GRADIENT_INFINITY_TOLERANCE"))
            objective_threshold = _finite(frame.f_globals.get("LOW_RANK_RELATIVE_OBJECTIVE_TOLERANCE"))
            record.update(
                low_rank_frame_found=True,
                source_line=traceback.tb_lineno,
                objective_mode=local["objective_mode"],
                active_rank=getattr(spec, "rank", None),
                active_rho=_finite(getattr(spec, "rho", None)),
                active_penalty_strength=_finite(getattr(spec, "strength", None)),
                iteration=local.get("iteration"),
                maximum_iterations=local.get("maximum_iterations"),
                objective=_finite(getattr(current, "objective", None)),
                smooth_objective=_finite(getattr(current, "smooth_objective", None)),
                penalty_value=_finite(getattr(current, "penalty_value", None)),
                gradient_U_infinity=_max_abs(getattr(current, "gradient_U", None)),
                gradient_d_infinity=_max_abs(getattr(current, "gradient_d", None)),
                gradient_infinity=gradient,
                relative_objective_change=relative,
                gradient_tolerance=threshold,
                relative_objective_tolerance=objective_threshold,
                gradient_gate=(None if gradient is None or threshold is None else gradient <= threshold),
                relative_objective_gate=(None if relative is None or objective_threshold is None
                                         else relative <= objective_threshold),
                line_search_step_at_exception=_finite(local.get("step")),
                current_is_accepted_trial=(current is not None
                                           and current is local.get("accepted_evaluation")),
                state_is_accepted_trial=(state is not None and state is local.get("accepted_state")),
                total_line_trials=local.get("total_line_trials"),
                pre_trial_directional_derivative=_finite(local.get("directional_derivative")),
                nonzero_gradient_with_zero_objective_change=(
                    None if gradient is None or relative is None else gradient > 0.0 and relative == 0.0),
            )
            break
        traceback = traceback.tb_next
    return record


def run_case_with_failure_record(worker, family, rank, *, optimizer_path, failure_path):
    """Call the unchanged worker; persist a failure before reraising it.

    Integration invokes this inside each child, not around
    future.result() in the parent. Caller must select a fresh failure path and
    an authorized bounded compute action; this helper selects neither.
    """
    failure_path = Path(failure_path)
    if not failure_path.parent.is_dir() or failure_path.exists():
        raise ValueError("Require an existing output directory and fresh failure path")
    try:
        return worker(family, rank)
    except Exception as error:
        try:
            record = failure_record(error, family=family, rank=rank,
                                    optimizer_path=optimizer_path)
            encoded = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
            with failure_path.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as diagnostic_error:
            error.add_note("Failure telemetry could not be preserved; diagnostic type="
                           + type(diagnostic_error).__name__)
        raise
