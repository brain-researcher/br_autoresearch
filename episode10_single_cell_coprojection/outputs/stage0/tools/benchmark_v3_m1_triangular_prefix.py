#!/usr/bin/env python3
"""Paired target-free engineering benchmark, not a complete permutation unit.

Run the original and repaired factored solves on the first actual warm path
of the same frozen synthetic permutation. Each stops after that path, or at
300 seconds. Complete fits are compared in memory; no optimizer state is
exported or reused by production and no model-selection result is returned.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import time
from unittest import mock

import numpy as np

import benchmark_v3_m2_falsifier_timing_v1 as timing
import fit_v3_m1 as m1
import validate_v3_m1_numerical_execution_v3 as execution_policy
from v3_m1_triangular_solve import solve_cholesky

PREFIX_SECONDS = 300.0
PREFIX_SCHEDULING_SECONDS = 720.0
COMPARISON_SECONDS = 120.0


class PrefixComplete(BaseException):
    pass


class PrefixDeadline(BaseException):
    pass


def original_factored_solve(factor, rhs):
    """Exactly the two original general solves, retained only in this probe."""
    return np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))


def run_prefix(fixture, assignment, factored_solve, *, maximum_seconds):
    if signal.getitimer(signal.ITIMER_REAL) != (0., 0.):
        raise RuntimeError("prefix benchmark cannot replace an active timer")
    statistics = {"complete_warm_path": False, "deadline_reached": False,
                  "converged_fits": 0, "factor_solve_calls": 0,
                  "factor_solve_seconds": 0., "exact_evaluation_calls": 0,
                  "exact_evaluation_seconds": 0.}
    completed = []
    original_fit = m1.fit_newton
    original_evaluate = m1.evaluate_model
    original_path = timing.permutation_m1._fit_warm_path

    def measured_solve(factor, rhs):
        started = time.perf_counter()
        try:
            return factored_solve(factor, rhs)
        finally:
            statistics["factor_solve_calls"] += 1
            statistics["factor_solve_seconds"] += time.perf_counter() - started

    def measured_evaluate(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_evaluate(*args, **kwargs)
        finally:
            statistics["exact_evaluation_calls"] += 1
            statistics["exact_evaluation_seconds"] += time.perf_counter() - started

    def measured_fit(prepared, library, geometry, ridge_lambda, *args, **kwargs):
        result = original_fit(prepared, library, geometry, ridge_lambda, *args, **kwargs)
        completed.append((float(ridge_lambda), result, prepared))
        statistics["converged_fits"] += 1
        return result

    def first_path(*args, **kwargs):
        original_path(*args, **kwargs)
        statistics["complete_warm_path"] = True
        raise PrefixComplete()

    def interrupt(signum, frame):
        raise PrefixDeadline()

    old_handler = signal.signal(signal.SIGALRM, interrupt)
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        with mock.patch.object(m1, "_cholesky_factored_solve", measured_solve), \
             mock.patch.object(m1, "fit_newton", measured_fit), \
             mock.patch.object(m1, "evaluate_model", measured_evaluate), \
             mock.patch.object(timing.permutation_m1, "_fit_warm_path", first_path), \
             open(os.devnull, "w", encoding="utf-8") as sink, \
             contextlib.redirect_stdout(sink):
            signal.setitimer(signal.ITIMER_REAL, maximum_seconds)
            timing.permutation_m1.fit_permuted_outer_m1(
                fixture.cells, assignment, fixture.candidates, fixture.geometry,
                identity_prefix="target_free_falsifier_timing_v1")
    except PrefixComplete:
        pass
    except PrefixDeadline:
        statistics["deadline_reached"] = True
    except Exception as error:
        statistics["error_type"] = type(error).__name__
        statistics["error"] = str(error)[:400]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, old_handler)
    statistics["wall_seconds"] = time.perf_counter() - started
    statistics["cpu_seconds"] = time.process_time() - cpu_started
    return statistics, completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    timing.require_slurm_one_thread()
    backend = dict(execution_policy.numeric_backend())
    args.output_root.mkdir(parents=True, exist_ok=False)
    timing._write_json_atomic(args.output_root / "numeric_backend.json", backend)
    started = time.perf_counter()
    report = {"report_kind": "M1_triangular_solve_paired_prefix_engineering",
              "status": "incomplete", "per_variant_deadline_seconds": PREFIX_SECONDS,
              "prefix_scheduling_envelope_seconds": PREFIX_SCHEDULING_SECONDS,
              "distribution_comparison_deadline_seconds": COMPARISON_SECONDS,
              "scheduler_wall_bound_seconds": 900,
              "first_warm_path_only": True, "complete_permutation_timing_unit": False,
              "real_M2_outcomes_opened": False, "real_membership_or_K_opened": False,
              "existing_M0_M1_M2_results_opened": False,
              "SWC_final_unassigned_opened": False, "valid_controller_trials": 0,
              "SLURM_job_id": os.environ["SLURM_JOB_ID"]}
    output = args.output_root / "paired_prefix.json"
    timing._write_json_atomic(output, report)
    fixture = timing.build_synthetic_fixture()
    assignment = timing.permutation.build_observed_permutation_assignment(
        fixture.cells, timing._permutation_trial())
    report["fixture_setup_seconds"] = time.perf_counter() - started
    variants = {}
    for label, solve in (("original_general_solve", original_factored_solve),
                         ("repaired_triangular_solve", solve_cholesky)):
        remaining = PREFIX_SCHEDULING_SECONDS - (time.perf_counter() - started)
        if remaining <= 0.:
            report["status"] = "total_deadline_before_paired_completion"
            timing._write_json_atomic(output, report)
            return 2
        statistics, variants[label] = run_prefix(
            fixture, assignment, solve, maximum_seconds=min(PREFIX_SECONDS, remaining))
        report[label] = statistics
        timing._write_json_atomic(output, report)
    left = variants["original_general_solve"]
    right = variants["repaired_triangular_solve"]
    comparisons = []
    for original, repaired in zip(left, right):
        ridge, old, old_prepared = original
        new_ridge, new, new_prepared = repaired
        if (ridge != new_ridge
                or tuple((c.identifier, c.group, c.k, c.observed_index) for c in old_prepared.cells)
                != tuple((c.identifier, c.group, c.k, c.observed_index) for c in new_prepared.cells)
                or any(not np.array_equal(getattr(old_prepared, field), getattr(new_prepared, field))
                       for field in ("x", "weights", "geometry_centers", "geometry_scales"))):
            raise RuntimeError("paired warm-path fixture or lambda changed")
        comparisons.append({
            "ridge_lambda": ridge,
            "objective_absolute_difference": abs(old.objective - new.objective),
            "parameter_maximum_absolute_difference": float(np.max(np.abs(old.theta - new.theta))),
            "original_iterations": old.iterations, "repaired_iterations": new.iterations,
            "original_gradient_infinity": old.gradient_infinity_norm,
            "repaired_gradient_infinity": new.gradient_infinity_norm,
            "original_relative_objective_change": old.relative_objective_change,
            "repaired_relative_objective_change": new.relative_objective_change,
        })
    report["common_completed_lambda_comparisons"] = comparisons
    timing._write_json_atomic(output, report)
    # The last mutually completed lambda is fixed by the common path prefix,
    # never chosen from an agreement value. Compare its full exact candidate
    # distributions once; this is separate from each timed optimizer prefix.
    report["last_common_distribution_comparison"] = {"status": "no_common_completed_fit"}
    if comparisons:
        old = left[len(comparisons) - 1]
        new = right[len(comparisons) - 1]
        old_handler = signal.getsignal(signal.SIGALRM)
        def interrupt_comparison(signum, frame):
            raise PrefixDeadline()
        comparison_started = time.perf_counter()
        signal.signal(signal.SIGALRM, interrupt_comparison)
        try:
            signal.setitimer(signal.ITIMER_REAL, COMPARISON_SECONDS)
            agreement = m1.compare_conditional_distributions(
                old[1].theta, new[1].theta, old[2], fixture.candidates, fixture.geometry)
            report["last_common_distribution_comparison"] = {
                "status": "complete", "ridge_lambda": old[0], **agreement}
        except PrefixDeadline:
            report["last_common_distribution_comparison"] = {
                "status": "deadline_reached", "ridge_lambda": old[0]}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.)
            signal.signal(signal.SIGALRM, old_handler)
        report["last_common_distribution_comparison"]["wall_seconds"] = time.perf_counter() - comparison_started
    complete = all(report[label]["complete_warm_path"]
                   for label in ("original_general_solve", "repaired_triangular_solve"))
    complete = complete and report["last_common_distribution_comparison"]["status"] == "complete"
    report["status"] = "paired_prefix_complete" if complete else "paired_prefix_incomplete"
    if complete:
        old_seconds = report["original_general_solve"]["wall_seconds"]
        new_seconds = report["repaired_triangular_solve"]["wall_seconds"]
        report["descriptive_prefix_speedup"] = old_seconds / new_seconds
    report["selection_note"] = ("Fixed first warm path and original-then-repaired order; "
                                "one pair, not a full controller runtime projection or qualification.")
    report["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    timing.assert_aggregate_safe(report)
    timing._write_json_atomic(output, report)
    print(report["status"], flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
