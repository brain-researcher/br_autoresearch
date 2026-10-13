#!/usr/bin/env python3
"""Bound the frozen permutation timing cost without completing a slow probe.

This is an aggregate-only preaccess resource diagnostic, never a completed
timing report, fit checkpoint, model trial, or scientific negative result.
The existing synthetic fixture, permutation and fresh-M1 implementation are
unchanged. Fixture/assignment setup is excluded from the measured lower bound.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import time

import benchmark_v3_m2_falsifier_timing_v1 as timing
import validate_v3_m1_numerical_execution_v3 as execution_policy


PREFIX_DEADLINE_SECONDS = 300.0
FULL_CEILING_CORE_HOURS = 1800.0


class PrefixDeadline(BaseException):
    """Abort this diagnostic's incomplete prefix, not a production fit."""


def projected_core_hour_lower_bound(prefix_seconds: float) -> float:
    # Existing run_v3_m2_observed_search_v1._validate_combined_resource_ceiling:
    # cap80, coverage16, 7 Rademachers => 57 remaining slots, each priced at
    # max(permutation, eligible association). Ignore every other positive cost.
    # 99 full replays + observed run, headroom2, allocated controller CPUs2.
    return 2.0 * (57.0 * prefix_seconds + 99.0 * 57.0 * prefix_seconds) * 2.0 / 3600.0


def measure_prefix(fixture, assignment):
    """Return only elapsed-time evidence; never return unfinished coefficients."""
    def interrupt(signum, frame):
        raise PrefixDeadline()

    previous_handler = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("diagnostic cannot replace an active process timer")
    interrupted = False
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink):
            signal.signal(signal.SIGALRM, interrupt)
            started = time.perf_counter()
            try:
                signal.setitimer(signal.ITIMER_REAL, PREFIX_DEADLINE_SECONDS)
                execution = timing.permutation_m1.fit_permuted_outer_m1(
                    fixture.cells,
                    assignment,
                    fixture.candidates,
                    fixture.geometry,
                    identity_prefix="target_free_falsifier_timing_v1",
                )
                elapsed = time.perf_counter() - started
            except PrefixDeadline:
                elapsed = time.perf_counter() - started
                interrupted = True
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, previous_handler)
    if not interrupted:
        timing.validate_permutation_accounting(execution)
    # Exclude the signal setup and dispatch overhead conservatively: use only
    # the 300-second scheduled interval for an interrupted prefix. A completed
    # prefix uses its direct stopwatch measurement, as the benchmark does.
    lower_bound = min(elapsed, PREFIX_DEADLINE_SECONDS) if interrupted else elapsed
    return lower_bound, elapsed, interrupted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    timing.require_slurm_one_thread()
    backend = dict(execution_policy.numeric_backend())
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    timing._write_json_atomic(root / "numeric_backend.json", backend)
    timing._write_json_atomic(root / "progress.json", {
        "phase": "unmeasured_fixture_setup", "status": "incomplete",
        "complete_timing_probe": False,
    })
    fixture = timing.build_synthetic_fixture()
    trial = timing._permutation_trial()
    assignment = timing.permutation.build_observed_permutation_assignment(fixture.cells, trial)
    timing._write_json_atomic(root / "progress.json", {
        "phase": "measured_required_M1_prefix", "status": "incomplete",
        "prefix_deadline_seconds": PREFIX_DEADLINE_SECONDS,
        "complete_timing_probe": False,
    })
    lower_bound, elapsed, interrupted = measure_prefix(fixture, assignment)
    projected = projected_core_hour_lower_bound(lower_bound)
    exceeded = projected > FULL_CEILING_CORE_HOURS
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_kind": "v3_M2_preaccess_resource_lower_bound_diagnostic",
        "status": "resource_ceiling_exceeded_by_lower_bound" if exceeded else "inconclusive_prefix_only",
        "SLURM_job_id": os.environ["SLURM_JOB_ID"],
        "prefix_interrupted_at_prespecified_deadline": interrupted,
        "observed_prefix_interval_seconds": elapsed,
        "required_permutation_unit_wall_seconds_lower_bound": lower_bound,
        "projected_allocated_core_hours_lower_bound": projected,
        "full_lifetime_core_hour_ceiling": FULL_CEILING_CORE_HOURS,
        "full_ceiling_exceeded": exceeded,
        "projection_branch": {"cap": 80, "coverage": 16, "rademachers": 7,
                              "remaining_slots": 57, "full_replays": 99,
                              "headroom_factor": 2, "allocated_controller_cpus": 2},
        "all_other_projected_costs_assumed_zero": True,
        "synthetic_fixture_and_assignment_setup_excluded": True,
        "complete_timing_probe": False,
        "valid_model_trials": 0,
        "real_M2_outcomes_opened": False,
        "real_membership_or_K_opened": False,
        "existing_M0_M1_M2_results_opened": False,
        "SWC_final_unassigned_opened": False,
        "scientific_disposition": "incomplete_search_no_positive_or_negative_biology",
    }
    timing.assert_aggregate_safe(result)
    timing._write_json_atomic(root / "resource_lower_bound.json", result)
    timing._write_json_atomic(root / "progress.json", {
        "phase": "diagnostic_complete", "status": result["status"],
        "complete_timing_probe": False,
    })
    print(result["status"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
