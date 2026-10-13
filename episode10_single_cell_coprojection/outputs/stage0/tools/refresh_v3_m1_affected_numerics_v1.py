#!/usr/bin/env python3
"""Refresh only M1 numerical evidence affected by the direct-derivative repair.

The existing 36-fit qualification and unchanged rank/geometry evidence are
reused. This producer cannot open target calls, M0 artifacts, or final data.
Its successful result still requires scheduler MaxRSS and resource postflight.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

import fit_v3_m1 as m1
import validate_v3_m1_numerical_execution_v3 as execution_policy


STAGE0 = Path(__file__).resolve().parent.parent
QUALIFICATION_JOB = "46328672"
OUTPUT_PREFIX = "v3_m1_affected_numerics_refresh_v1"
SUMMARY_NAME = "v3_m1_affected_numerics_refresh.json"
EXECUTION_ADDENDUM = STAGE0 / "v3_m1_numerical_execution_addendum_v3.yaml"
BACKEND_KEYS = (
    "cpu_model", "numpy", "openblas_library", "openblas_config",
    "openblas_core", "openblas_threads", "thread_environment",
)
SEALS = {
    "real_target_calls_or_real_K_opened": False,
    "M0_prediction_or_fit_artifact_opened": False,
    "final_or_unassigned_or_SWC_opened": False,
}


class RefreshError(RuntimeError):
    pass


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RefreshError("required JSON must be a regular non-symlink file")
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise RefreshError("required JSON is not an object")
    return result


def validate_qualification(
    summary: Mapping[str, Any], root: Path, backend: Mapping[str, Any]
) -> None:
    """Bind the existing frozen qualification, without running its fits again."""
    phases = ("warm_repeat1", "warm_repeat2", "zero")
    lambdas = tuple(float(value) for value in m1.LAMBDAS_DESCENDING)
    if not (
        summary.get("status") == "pass"
        and summary.get("stage") == "v3_M1_target_free_synthetic_warm_zero_diagnostic_v2"
        and str(summary.get("SLURM_job_id")) == QUALIFICATION_JOB
        and summary.get("output_root") == str(root)
        and summary.get("rows") == 68 and summary.get("groups") == 4
        and summary.get("seed") == 20260928
        and tuple(summary.get("phase_order", ())) == phases
        and tuple(summary.get("lambda_order", ())) == lambdas
        and summary.get("completed_fit_count") == 36
        and summary.get("active_fit") is None
        and summary.get("failure_stage") is None
        and summary.get("post_freeze_single_thread_qualification") is True
        and summary.get("numerical_execution_schema_version")
        == "episode10_v3_m1_numerical_execution_addendum_v3"
        and summary.get("model_or_convergence_threshold_changed") is False
        and summary.get("Wolfe_or_certificate_threshold_changed") is False
        and summary.get("real_input_root_argument_supported") is False
        and summary.get("target_root_argument_supported") is False
        and summary.get("real_target_calls_or_real_K_or_membership_opened") is False
        and all(summary.get(key) is value for key, value in SEALS.items())
    ):
        raise RefreshError("latest repaired-evaluator 36-fit qualification did not pass")
    qualified_backend = summary.get("numeric_backend", {})
    if not isinstance(qualified_backend, Mapping) or any(
        qualified_backend.get(key) != backend.get(key) for key in BACKEND_KEYS
    ):
        raise RefreshError("refresh backend differs from the passed qualification")
    fits = summary.get("fit_attempts", ())
    comparisons = summary.get("warm_zero_comparisons", ())
    expected = [(phase, ordinal, lam) for phase in phases
                for ordinal, lam in enumerate(lambdas, 1)]
    witnesses = {
        "gradient_and_repeated_unchanged_initial_objective",
        "gradient_and_relative_objective_change", "strong_convexity_gap_after_zoom_stall",
    }
    if len(fits) != 36 or len(comparisons) != 12:
        raise RefreshError("qualification fit/comparison counts changed")
    for ordinal, (fit, (phase, phase_ordinal, lam)) in enumerate(zip(fits, expected), 1):
        if not (
            fit.get("fit_ordinal") == ordinal and fit.get("phase") == phase
            and fit.get("phase_fit_ordinal") == phase_ordinal
            and fit.get("lambda") == lam and fit.get("status") == "pass"
            and fit.get("convergence_witness") in witnesses
        ):
            raise RefreshError("qualification fit sequence or witness changed")
    for comparison, lam in zip(comparisons, lambdas):
        objective_tolerance = max(
            1e-10, 1.1 * m1.PARAMETER_DIMENSION * m1.GRADIENT_TOLERANCE**2 / (2 * lam)
        )
        limits = {
            "repeat_objective_absolute_difference": 1e-12,
            "repeat_coefficient_max_absolute_difference": 1e-10,
            "repeat_probability_max_absolute_difference": 1e-10,
            "zero_start_objective_absolute_difference": objective_tolerance,
            "zero_start_probability_max_absolute_difference": 1e-5,
            "zero_start_log_probability_max_absolute_difference": 1e-4,
        }
        if not (
            comparison.get("status") == "pass" and comparison.get("lambda") == lam
            and comparison.get("zero_start_objective_absolute_tolerance") == objective_tolerance
            and all(0.0 <= float(comparison.get(key, math.inf)) <= limit
                    for key, limit in limits.items())
        ):
            raise RefreshError("qualification comparison failed its frozen threshold")


def validate_resources(payload: Mapping[str, Any], allocated_cpus: int) -> Mapping[str, Any]:
    """Require explicit current accounting with this request already reserved."""
    reservation = payload.get("m1_affected_numerics_refresh", {})
    if not (
        payload.get("accounting_epoch") == "wall_reset_20261001"
        and payload.get("no_new_wall_reset") is True
        and float(payload.get("historical_strict_wall_hours_preserved", -1))
        >= 129.83266666666668
        and float(payload.get("historical_allocated_core_hours_floor_preserved", -1))
        >= 302.02566666666667
        and reservation.get("qualification_job_id") == QUALIFICATION_JOB
        and reservation.get("reservation_included") is True
        and reservation.get("request_wall_hours") == 8
        and reservation.get("requested_cpus") == 1
        and reservation.get("conservative_allocated_cpus") == 4
        and reservation.get("memory_GiB") == 16
        and 1 <= allocated_cpus <= int(reservation["conservative_allocated_cpus"])
    ):
        raise RefreshError("current accounting or this refresh reservation is absent")
    for key in ("remaining_unreserved_wall_hours", "remaining_unreserved_allocated_core_hours"):
        value = float(reservation.get(key, math.nan))
        if not math.isfinite(value) or value < 0.0:
            raise RefreshError("remaining unreserved resources are invalid")
    return reservation


def recovery_subset() -> Mapping[str, Any]:
    cells, _, prepared, library, geometry, planted = m1.generate_synthetic_dataset(
        groups=16, rows_per_group=34, seed=20260927
    )
    return m1._synthetic_population_recovery_gate(cells, prepared, library, geometry, planted)


def all_k8_subset(interface_root: Path, geometry_root: Path) -> Mapping[str, Any]:
    """The unchanged all-K8 numerical block of run_real_shape_benchmark.

    The preceding mixed-K 79-fold rank calculation is deliberately reused,
    rather than rerun. No model or optimizer formula is implemented here.
    """
    started = time.perf_counter()
    mixed_cells, pairs = m1.load_target_free_real_shape(interface_root, geometry_root)
    library = m1.CandidateLibrary()
    geometry = m1.GeometryEngine(library, pairs)
    worst_cells = m1.clone_with_k(mixed_cells, 8)
    cache_started = time.perf_counter()
    moments = tuple(geometry.uniform_moments(cell) for cell in worst_cells)
    prepared = m1.prepare_data(worst_cells, moments)
    feature_seconds = time.perf_counter() - cache_started
    unique_states = sum(library.basis(cell.k).count for cell in worst_cells)
    if len(worst_cells) != 194 or len({cell.group for cell in worst_cells}) != 12 or unique_states != 2496780:
        raise RefreshError("frozen all-K8 real-shape fixture changed")
    zero = np.zeros(m1.PARAMETER_DIMENSION, dtype=float)
    ogh_started = time.perf_counter()
    first = m1.evaluate_model(zero, prepared, library, geometry, 1e-4, need_hessian=True)
    ogh_seconds = time.perf_counter() - ogh_started
    repeat = m1.evaluate_model(zero, prepared, library, geometry, 1e-4, need_hessian=True)
    if not (
        first.objective == repeat.objective and np.array_equal(first.gradient, repeat.gradient)
        and first.hessian is not None and repeat.hessian is not None
        and np.array_equal(first.hessian, repeat.hessian)
    ):
        raise RefreshError("all-K8 objective/gradient/Hessian is not deterministic")
    factor_started = time.perf_counter()
    direction, solve = m1._newton_direction(first.hessian, first.gradient, 1e-4)
    factor_seconds = time.perf_counter() - factor_started
    if not np.all(np.isfinite(direction)):
        raise RefreshError("all-K8 Newton direction is nonfinite")
    path_started = time.perf_counter()
    path = m1.fit_path(prepared, library, geometry)
    path_seconds = time.perf_counter() - path_started
    if tuple(path) != tuple(m1.LAMBDAS_DESCENDING) or any(
        result.telemetry.get("status") != "pass" for result in path.values()
    ):
        raise RefreshError("all-K8 complete warm path did not pass")
    return {
        "status": "pass", "rows": 194, "groups": 12,
        "all_K8_unique_candidate_states": unique_states,
        "all_K8_score_computations_per_two_pass_objective": 2 * unique_states,
        "all_K8_feature_moment_build_seconds": feature_seconds,
        "objective_gradient_Hessian_seconds": ogh_seconds,
        "objective_gradient_Hessian_candidate_visits": first.candidate_visits,
        "candidate_sets_scored_per_second": first.candidate_visits / ogh_seconds,
        "factorization_seconds": factor_seconds, "factorization": solve,
        "complete_12_lambda_warm_path_seconds": path_seconds,
        "complete_path_iterations": {format(lam, ".12g"): fit.iterations for lam, fit in path.items()},
        "elapsed_wall_seconds": time.perf_counter() - started,
        **SEALS,
    }


def production_projection(
    benchmark: Mapping[str, Any], reservation: Mapping[str, Any], peak_rss_mib: float,
    refresh_elapsed_seconds: float,
) -> Mapping[str, Any]:
    path_seconds = float(benchmark["complete_12_lambda_warm_path_seconds"])
    projected_hours = 157.0 * path_seconds / 3600.0
    # Scheduler allocation may exceed the one requested numerical CPU. Charge
    # the explicit conservative reservation; never derive BLAS threads from it.
    allocated_cpus = int(reservation["conservative_allocated_cpus"])
    projected_core_hours = allocated_cpus * projected_hours
    passed = (
        math.isfinite(path_seconds) and path_seconds > 0
        and projected_hours < 0.8 * 8.0
        and math.isfinite(refresh_elapsed_seconds)
        and 0 <= refresh_elapsed_seconds < 0.8 * 8 * 3600
        and math.isfinite(peak_rss_mib) and peak_rss_mib < 0.8 * 16 * 1024
        and projected_hours <= float(reservation["remaining_unreserved_wall_hours"])
        and projected_core_hours <= float(reservation["remaining_unreserved_allocated_core_hours"])
    )
    return {
        "status": "pass_pending_scheduler_MaxRSS_and_current_resource_postflight" if passed else "fail",
        "uncached_production_path_equivalents_for_pass": 157,
        "exact_reuse_path_equivalents_report_only": 91,
        "projected_production_wall_hours": projected_hours,
        "projected_allocated_core_hours": projected_core_hours,
        "conservative_production_allocated_cpus": allocated_cpus,
        "frozen_production_request_hours": 8,
        "frozen_production_request_memory_GiB": 16,
        "minimum_headroom_fraction": 0.25,
        "refresh_elapsed_wall_seconds": refresh_elapsed_seconds,
        "frozen_refresh_time_80_percent_seconds": 0.8 * 8 * 3600,
        "process_peak_RSS_MiB_non_authoritative": peak_rss_mib,
        "Slurm_MaxRSS": None,
        "Slurm_MaxRSS_status": "required_post_job_batch_step",
        "remaining_budget_reconciliation_required": True,
        "production_authorized": False,
    }


def run_numerics(
    payload: dict[str, Any], interface_root: Path, geometry_root: Path,
    reservation: Mapping[str, Any], checkpoint: Callable[[], None],
) -> None:
    started = time.perf_counter()
    payload["phase"] = "544_row_recovery_started"
    checkpoint()
    payload["recovery"] = recovery_subset()
    payload["phase"] = "544_row_recovery_completed"
    checkpoint()
    if payload["recovery"].get("status") != "pass":
        raise RefreshError("frozen sampled/population recovery did not pass")
    payload["phase"] = "all_K8_numerical_throughput_started"
    checkpoint()
    payload["all_K8_numerical_throughput"] = all_k8_subset(interface_root, geometry_root)
    payload["resource_projection"] = production_projection(
        payload["all_K8_numerical_throughput"], reservation,
        float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
        time.perf_counter() - started,
    )
    if payload["resource_projection"]["status"] == "fail":
        raise RefreshError("fresh 157-path projection exceeds frozen headroom or remaining resources")
    payload["phase"] = "numerical_refresh_completed"
    payload["status"] = "pass_pending_scheduler_MaxRSS_and_current_resource_postflight"
    checkpoint()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--resource-accounting", required=True, type=Path)
    args = parser.parse_args(argv)
    output_root = m1._prepare_job_output(args.output_root, OUTPUT_PREFIX)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "status": "running", "phase": "required_gates", "analysis_label": "QC",
        "SLURM_job_id": os.environ["SLURM_JOB_ID"], "output_root": str(output_root),
        "qualification_job_reused": QUALIFICATION_JOB,
        "static_rank_evidence_reused": ["45596378", "45602325"],
        "warm_zero_gate_rerun": False, "static_rank_gates_rerun": False,
        "target_root_argument_supported": False, "production_authorized": False,
        **SEALS,
    }

    def checkpoint() -> None:
        payload["elapsed_wall_seconds"] = time.perf_counter() - started
        m1._atomic_json(output_root / SUMMARY_NAME, payload)

    checkpoint()
    try:
        m1.validate_contract(
            STAGE0 / "v3_m1_model_contract.yaml",
            STAGE0 / "v3_m1_synthetic_design_addendum_v1.yaml",
            STAGE0 / "v3_m1_synthetic_recovery_addendum_v2.yaml",
            STAGE0 / "v3_m1_solver_addendum_v2.yaml",
        )
        execution_policy.validate_addendum(EXECUTION_ADDENDUM)
        backend = dict(execution_policy.numeric_backend())
        payload["numeric_backend"] = backend
        scratch = Path(os.environ["SCRATCH"]).resolve() / "br_autoresearch" / "episode10_single_cell_coprojection"
        qualification_root = m1._require_exact_scratch_child(
            scratch / f"v3_m1_synthetic_warm_zero_diagnostic_v2_job_{QUALIFICATION_JOB}",
            f"v3_m1_synthetic_warm_zero_diagnostic_v2_job_{QUALIFICATION_JOB}", "qualification",
        )
        validate_qualification(
            _read_json(qualification_root / "v3_m1_synthetic_warm_zero_diagnostic.json"),
            qualification_root, backend,
        )
        accounting_path = args.resource_accounting.absolute()
        if accounting_path.parent != STAGE0 or not accounting_path.name.startswith("resource_accounting_"):
            raise RefreshError("resource accounting must be an explicit episode stage0 record")
        accounting = _read_json(accounting_path)
        reservation = validate_resources(accounting, int(os.environ["SLURM_CPUS_PER_TASK"]))
        payload["resource_accounting"] = str(accounting_path)
        payload["resource_reservation"] = dict(reservation)
        run_numerics(
            payload, scratch / "v3_model_interface_preflight_v1",
            scratch / "v3_m1_geometry_v1", reservation, checkpoint,
        )
        if time.perf_counter() - started >= 0.8 * 8 * 3600:
            raise RefreshError("whole refresh exceeded frozen 80-percent time limit")
        return 0
    except Exception as error:
        payload.update(status="fail", failure_class=type(error).__name__, failure_message=str(error))
        if isinstance(error, m1.UncertifiedLineSearchFailure):
            payload["optimizer_scalar_context"] = m1._scalar_mapping(error.scalar_context)
        checkpoint()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
