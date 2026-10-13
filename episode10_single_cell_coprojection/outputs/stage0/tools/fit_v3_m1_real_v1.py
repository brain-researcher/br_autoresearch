#!/usr/bin/env python3
"""Fit the frozen EP10 v3 development-only real M1 model.

This production runner is intentionally separate from ``fit_v3_m1.py``.  The
target-free engine keeps no real-target CLI route; this file opens only the
authorized development ID/group/K/membership projection after independently
requiring the passed actual-K gate (job 45602325).  It has no route to M0,
SWCs, final data, or unassigned data.

The runner performs the frozen 12-outer/11-inner LOGO procedure, reuses the 66
mathematically identical unordered leave-two training paths, performs every
selected outer refit from zero, runs the 12 full-development tuning paths, and
performs the full refit from zero.  Checkpoints are atomic and scoped to one
deterministic training set.  The conservative budget basis remains the frozen
157 uncached path equivalents even though exact reuse executes 91 equivalents.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import stat
import sys
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml


SCRIPT_PATH = Path(os.path.abspath(__file__))
STAGE0_DIR = SCRIPT_PATH.parent.parent
ENGINE_PATH = SCRIPT_PATH.parent / "fit_v3_m1.py"
EXECUTION_VALIDATOR_PATH = (
    SCRIPT_PATH.parent / "validate_v3_m1_numerical_execution_v3.py"
)
READINESS_PATH = STAGE0_DIR / "readiness_status.yaml"

SPEC = importlib.util.spec_from_file_location("ep10_fit_v3_m1_real_engine", ENGINE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import frozen M1 engine: {ENGINE_PATH}")
m1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m1
SPEC.loader.exec_module(m1)

EXECUTION_SPEC = importlib.util.spec_from_file_location(
    "ep10_fit_v3_m1_real_execution_policy", EXECUTION_VALIDATOR_PATH
)
if EXECUTION_SPEC is None or EXECUTION_SPEC.loader is None:
    raise RuntimeError(
        f"cannot import M1 numerical execution policy: {EXECUTION_VALIDATOR_PATH}"
    )
execution_policy = importlib.util.module_from_spec(EXECUTION_SPEC)
sys.modules[EXECUTION_SPEC.name] = execution_policy
EXECUTION_SPEC.loader.exec_module(execution_policy)


ACTUAL_K_JOB_ID = "45602325"
PRIOR_MEMBERSHIP_OPENING_JOB_ID = "45609750"
QUALIFICATION_JOB_ID = "46328672"
NUMERICAL_REFRESH_JOB_ID = "46330399"
RESOURCE_AMENDMENT_PATH = STAGE0_DIR / "v3_m1_production_resource_amendment_v4.yaml"
RESOURCE_AMENDMENT_SCHEMA = "episode10_v3_m1_production_resource_amendment_v4"
RESOURCE_ASSESSMENT_PATH = STAGE0_DIR / "resource_accounting_20261002_m1_clean_refit_launch.json"
WALL_AUTHORIZATION_PATH = STAGE0_DIR / "v3_m1_wall_limit_authorization_20261002.json"
HISTORICAL_M0_ACCESS = "true_job_45760290_post_M1_mechanical_comparison_only"
ACTUAL_K_ROOT_BASENAME = f"v3_m1_actual_k_gate_v1_job_{ACTUAL_K_JOB_ID}"
ACTUAL_K_SUMMARY_FILENAME = "v3_m1_actual_k_gate_summary.json"
QUALIFICATION_ROOT_BASENAME = (
    f"v3_m1_synthetic_warm_zero_diagnostic_v2_job_{QUALIFICATION_JOB_ID}"
)
QUALIFICATION_SUMMARY_FILENAME = "v3_m1_synthetic_warm_zero_diagnostic.json"
TARGET_ROOT_BASENAME = "gao_mop_development_targets_v3_ep10_assets"
TARGET_FILENAME = "primary_16_target_calls.csv"
TARGET_GATE_FILENAME = "v3_gate_summary.json"
OUTPUT_PREFIX = "v3_m1_real_fit_single_thread_v3"
CHECKPOINT_DIRECTORY = "checkpoints"
CHECKPOINT_MANIFEST = "v3_m1_real_single_thread_v3_checkpoint_manifest.json"
CHECKPOINT_SCHEMA = "episode10_v3_m1_real_fit_checkpoint_single_thread_v3"
PRODUCTION_STAGE = "v3_M1_development_real_fit_single_thread_v3"
EXECUTION_SCHEMA_VERSION = "episode10_v3_m1_numerical_execution_addendum_v3"

PREDICTION_FILENAME = "v3_m1_cross_fitted_predictions.csv"
SUMMARY_FILENAME = "v3_m1_fit_summary.json"
FULL_REFIT_FILENAME = "v3_m1_full_refit.json"
TELEMETRY_FILENAME = "v3_m1_solver_telemetry.json"
REQUIRED_OUTPUTS = (
    PREDICTION_FILENAME,
    SUMMARY_FILENAME,
    FULL_REFIT_FILENAME,
    TELEMETRY_FILENAME,
)

EXPECTED_TARGET_ROWS = 200
EXPECTED_ROWS = 194
EXPECTED_EXCLUDED_ROWS = 6
EXPECTED_GROUPS = 12
TARGET_FIELD_COUNT = 90
TARGET_IDENTIFIER_INDEX = 0
TARGET_GROUP_INDEX = 2
TARGET_K_INDEX = 8
TARGET_DETECTED_START_INDEX = 26
TARGET_DETECTED_STOP_INDEX = 42
K_PATTERN = re.compile(rb"(?:[0-9]|1[0-6])\Z")
CHECKPOINT_KEY_PATTERN = re.compile(r"[a-z0-9_]+\Z")


class RealM1Error(m1.M1Error):
    """The real M1 runner failed a frozen input, scope, or execution rule."""


class BoundaryReview(RealM1Error):
    """At least six outer folds selected a frozen ridge-grid boundary."""


@dataclass(frozen=True)
class MembershipScanCounts:
    total_rows: int
    matched_rows: int
    extra_rows: int


@dataclass(frozen=True)
class Prediction:
    log_probability_nats: np.ndarray
    marginals: np.ndarray


@dataclass(frozen=True)
class TrainingBundle:
    key: str
    indexes: tuple[int, ...]
    prepared: Any
    fits: Mapping[float, Any]
    checkpoint_reused: bool


@dataclass(frozen=True)
class OuterSelection:
    selected_lambdas: Mapping[str, float]
    details: tuple[Mapping[str, Any], ...]
    boundary_count: int
    warm_cache: Mapping[str, TrainingBundle]


@dataclass(frozen=True)
class CrossFitResult:
    prediction: Prediction
    outer_details: tuple[Mapping[str, Any], ...]
    boundary_count: int
    warm_cache: Mapping[str, TrainingBundle]
    outer_refits: Mapping[str, TrainingBundle]


@dataclass(frozen=True)
class FullFitResult:
    selected_lambda: float
    selection: Mapping[str, Any]
    warm_cache: Mapping[str, TrainingBundle]
    full_refit: TrainingBundle


WarmFactory = Callable[..., TrainingBundle]
ZeroFactory = Callable[..., TrainingBundle]
Predictor = Callable[..., Prediction]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _open_regular_at(directory_fd: int, name: str) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RealM1Error("sealed input is not a regular file")
    return descriptor


def _read_json_at(directory_fd: int, name: str, label: str) -> Mapping[str, Any]:
    descriptor = _open_regular_at(directory_fd, name)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError) as error:
        raise RealM1Error(f"{label} is unreadable") from error
    if not isinstance(payload, dict):
        raise RealM1Error(f"{label} is not a mapping")
    return payload


def validate_real_contract(path: Path) -> Mapping[str, Any]:
    """Validate only the frozen clauses that determine this orchestration."""
    contract = m1.validate_contract(path)
    resampling = contract.get("resampling", {})
    outer = resampling.get("outer", {})
    inner = resampling.get("inner", {})
    fold_fit = resampling.get("fold_fit", {})
    selection = resampling.get("selection", {})
    boundary = resampling.get("boundary_review", {})
    full = resampling.get("full_development_refit", {})
    scoring = contract.get("scoring", {})
    calibration = contract.get("calibration", {})
    required = contract.get("required_outputs_on_execute", {})
    target = contract.get("required_inputs", {}).get("target_root", {})
    join = contract.get("required_inputs", {}).get("join", {})
    m0 = contract.get("required_inputs", {}).get("M0_comparator_root", {})
    if not (
        target.get("required_basename") == TARGET_ROOT_BASENAME
        and target.get("required_file") == TARGET_FILENAME
        and target.get("required_gate_summary") == TARGET_GATE_FILENAME
        and join.get("key") == "provider_neuron_id"
        and join.get("cardinality") == "one_to_one"
        and join.get("exact_identity_set_required") is True
        and join.get("exact_group_match_required") is True
        and int(outer.get("folds", -1)) == 12
        and outer.get("method") == "leave_one_conservative_group_out"
        and int(inner.get("folds_per_outer_split", -1)) == 11
        and inner.get("method") == "leave_one_remaining_group_out"
        and fold_fit.get("same_12_lambda_grid_for_every_fold") is True
        and fold_fit.get("tuning_paths_use_descending_warm_starts") is True
        and fold_fit.get("selected_lambda_outer_refit_initialization")
        == "all_261_parameters_zero"
        and selection.get("criterion")
        == "minimum_equal_group_inner_validation_data_NLL_without_ridge_penalty"
        and selection.get("ties") == "largest_lambda"
        and int(boundary.get("stop_if_boundary_selected_in_at_least_outer_folds", -1))
        == 6
        and full.get("initialization") == "all_261_parameters_zero"
        and scoring.get("denominator")
        == "all_194_cells_including_K_i_obs_0_or_16"
        and calibration.get("bins") == "forbidden"
        and tuple(required.get("files", ())) == REQUIRED_OUTPUTS
        and m0.get("access_timing") is not None
    ):
        raise RealM1Error("frozen real-M1 orchestration clauses changed")
    return contract


def _read_readiness() -> Mapping[str, Any]:
    absolute = Path(os.path.abspath(READINESS_PATH))
    if (
        absolute != READINESS_PATH
        or not absolute.is_file()
        or absolute.is_symlink()
        or absolute.resolve(strict=True) != READINESS_PATH.resolve(strict=True)
    ):
        raise RealM1Error("readiness ledger is not the exact stage0 file")
    descriptor = os.open(absolute, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RealM1Error("readiness ledger is not a regular file")
    with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise RealM1Error("readiness ledger is not a mapping")
    return payload


def _execution_identity(
    addendum: Mapping[str, Any], backend: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return only the numerical invariants that a same-job resume must match."""
    return {
        "numerical_execution_schema_version": addendum["schema_version"],
        "qualification_job_id": QUALIFICATION_JOB_ID,
        "numerical_refresh_job_id": NUMERICAL_REFRESH_JOB_ID,
        "production_resource_amendment_schema_version": RESOURCE_AMENDMENT_SCHEMA,
        "historical_M0_artifact_access": HISTORICAL_M0_ACCESS,
        "prior_membership_opening_job_id": PRIOR_MEMBERSHIP_OPENING_JOB_ID,
        "cpu_model": backend["cpu_model"],
        "numpy": backend["numpy"],
        "openblas_library": backend["openblas_library"],
        "openblas_config": backend["openblas_config"],
        "openblas_core": backend["openblas_core"],
        "openblas_threads": int(backend["openblas_threads"]),
        "thread_environment": dict(backend["thread_environment"]),
    }


def require_single_thread_qualification(
    qualification_root_argument: Path,
) -> Mapping[str, Any]:
    """Require the exact passed synthetic one-thread gate before membership."""
    readiness = _read_readiness()
    gates = readiness.get("gates", {})
    access = readiness.get("access", {})
    qualification_record = str(
        gates.get(
            "gao_development_v3_m1_repaired_evaluator_qualification", ""
        )
    )
    if not (
        qualification_record.startswith(f"pass_job_{QUALIFICATION_JOB_ID}_")
        and int(QUALIFICATION_JOB_ID)
        in readiness.get("launch", {}).get("slurm_job_ids", [])
        and access.get("v3_m1_M0_artifact_opened") == HISTORICAL_M0_ACCESS
        and access.get("final_group_outcomes_opened") is False
    ):
        raise RealM1Error(
            "passed single-thread qualification or sealed readiness boundary is absent"
        )

    root = m1._require_exact_scratch_child(
        qualification_root_argument,
        QUALIFICATION_ROOT_BASENAME,
        "single-thread qualification root",
    )
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        summary = _read_json_at(
            root_fd,
            QUALIFICATION_SUMMARY_FILENAME,
            "single-thread qualification summary",
        )
    finally:
        os.close(root_fd)

    phases = ("warm_repeat1", "warm_repeat2", "zero")
    lambdas = tuple(float(value) for value in m1.LAMBDAS_DESCENDING)
    fit_attempts = summary.get("fit_attempts")
    comparisons = summary.get("warm_zero_comparisons")
    backend = summary.get("numeric_backend")
    if not (
        summary.get("status") == "pass"
        and summary.get("stage")
        == "v3_M1_target_free_synthetic_warm_zero_diagnostic_v2"
        and str(summary.get("SLURM_job_id")) == QUALIFICATION_JOB_ID
        and summary.get("output_root") == str(root)
        and int(summary.get("rows", -1)) == 68
        and int(summary.get("groups", -1)) == 4
        and int(summary.get("seed", -1)) == 20260928
        and tuple(summary.get("phase_order", ())) == phases
        and tuple(float(value) for value in summary.get("lambda_order", ()))
        == lambdas
        and int(summary.get("completed_fit_count", -1)) == 36
        and summary.get("active_fit") is None
        and summary.get("failure_stage") is None
        and summary.get("post_freeze_single_thread_qualification") is True
        and summary.get("numerical_execution_addendum")
        == "v3_m1_numerical_execution_addendum_v3.yaml"
        and summary.get("numerical_execution_schema_version")
        == "episode10_v3_m1_numerical_execution_addendum_v3"
        and summary.get("model_or_convergence_threshold_changed") is False
        and summary.get("Wolfe_or_certificate_threshold_changed") is False
        and summary.get("real_input_root_argument_supported") is False
        and summary.get("target_root_argument_supported") is False
        and summary.get("real_target_calls_or_real_K_opened") is False
        and summary.get("real_target_calls_or_real_K_or_membership_opened")
        is False
        and summary.get("M0_prediction_or_fit_artifact_opened") is False
        and summary.get("final_or_unassigned_or_SWC_opened") is False
        and isinstance(fit_attempts, list)
        and len(fit_attempts) == 36
        and isinstance(comparisons, list)
        and len(comparisons) == 12
        and isinstance(backend, Mapping)
        and backend.get("openblas_core") == "Zen"
        and int(backend.get("openblas_threads", -1)) == 1
        and "AMD EPYC 7502" in str(backend.get("cpu_model", ""))
        and backend.get("thread_environment")
        == {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    ):
        raise RealM1Error("single-thread qualification identity or seal changed")

    valid_witnesses = {
        "gradient_and_repeated_unchanged_initial_objective",
        "gradient_and_relative_objective_change",
        "strong_convexity_gap_after_zoom_stall",
    }
    expected_fits = [
        (phase, ordinal, ridge_lambda)
        for phase in phases
        for ordinal, ridge_lambda in enumerate(lambdas, start=1)
    ]
    for fit_ordinal, (fit, expected) in enumerate(
        zip(fit_attempts, expected_fits, strict=True), start=1
    ):
        phase, phase_fit_ordinal, ridge_lambda = expected
        if not (
            isinstance(fit, Mapping)
            and int(fit.get("fit_ordinal", -1)) == fit_ordinal
            and fit.get("phase") == phase
            and int(fit.get("phase_fit_ordinal", -1)) == phase_fit_ordinal
            and float(fit.get("lambda", math.nan)) == ridge_lambda
            and fit.get("status") == "pass"
            and fit.get("convergence_witness") in valid_witnesses
        ):
            raise RealM1Error("single-thread qualification fit sequence changed")

    for comparison, ridge_lambda in zip(comparisons, lambdas, strict=True):
        objective_tolerance = max(
            1.0e-10,
            1.1
            * m1.PARAMETER_DIMENSION
            * m1.GRADIENT_TOLERANCE**2
            / (2.0 * ridge_lambda),
        )
        if not (
            isinstance(comparison, Mapping)
            and comparison.get("status") == "pass"
            and float(comparison.get("lambda", math.nan)) == ridge_lambda
            and float(
                comparison.get(
                    "repeat_objective_absolute_difference", math.inf
                )
            )
            <= 1.0e-12
            and float(
                comparison.get(
                    "repeat_coefficient_max_absolute_difference", math.inf
                )
            )
            <= 1.0e-10
            and float(
                comparison.get(
                    "repeat_probability_max_absolute_difference", math.inf
                )
            )
            <= 1.0e-10
            and float(
                comparison.get("zero_start_objective_absolute_tolerance", math.nan)
            )
            == objective_tolerance
            and float(
                comparison.get("zero_start_objective_absolute_difference", math.inf)
            )
            <= objective_tolerance
            and float(
                comparison.get(
                    "zero_start_probability_max_absolute_difference", math.inf
                )
            )
            <= 1.0e-5
            and float(
                comparison.get(
                    "zero_start_log_probability_max_absolute_difference", math.inf
                )
            )
            <= 1.0e-4
        ):
            raise RealM1Error("single-thread qualification comparison changed")
    return summary


def require_matching_qualification_backend(
    qualification: Mapping[str, Any], current_backend: Mapping[str, Any]
) -> None:
    """Require production to use the numerical backend qualified in job 46328672."""
    qualified_backend = qualification.get("numeric_backend")
    keys = (
        "cpu_model",
        "numpy",
        "openblas_library",
        "openblas_config",
        "openblas_core",
        "openblas_threads",
        "thread_environment",
    )
    if not (
        isinstance(qualified_backend, Mapping)
        and all(
            current_backend.get(key) == qualified_backend.get(key)
            for key in keys
        )
    ):
        raise RealM1Error(
            "production numerical backend differs from the passed qualification"
        )


def _read_exact_resource_record(path: Path, expected: Path) -> Mapping[str, Any]:
    """Read only the approved episode-local resource amendment/assessment."""
    if Path(os.path.abspath(path)) != expected or path.is_symlink():
        raise RealM1Error("production resource record is not the exact approved file")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RealM1Error("production resource record is not regular")
    with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) if expected.suffix == ".yaml" else json.load(handle)
    if not isinstance(payload, Mapping):
        raise RealM1Error("production resource record is not a mapping")
    return payload


def assess_approved_numerical_refresh(
    refresh: Mapping[str, Any], amendment: Mapping[str, Any],
    assessment: Mapping[str, Any], authorization: Mapping[str, Any],
    numeric_backend: Mapping[str, Any], allocated_cpus: int,
) -> Mapping[str, Any]:
    """Apply the approved 12-hour limit without relabeling the original failure."""
    boundary = amendment.get("production_boundary", {})
    reconciliation = amendment.get("resource_reconciliation", {})
    reservation = assessment.get("m1_clean_refit", {})
    if not (
        amendment.get("schema_version") == RESOURCE_AMENDMENT_SCHEMA
        and amendment.get("status") == "frozen_user_authorized"
        and amendment.get("authorization", {}).get("record")
        == f"outputs/stage0/{WALL_AUTHORIZATION_PATH.name}"
        and amendment.get("qualification_job_id") == QUALIFICATION_JOB_ID
        and amendment.get("numerical_refresh_job_id") == NUMERICAL_REFRESH_JOB_ID
        and authorization.get("status") == "approved"
        and authorization.get("approved_production_wall_time") == "12:00:00"
        and boundary.get("requested_wall_time") == "12:00:00"
        and boundary.get("requested_cpus_per_task") == 1
        and boundary.get("requested_memory") == "16G"
        and boundary.get("conservative_allocated_cpus") == 4
        and reconciliation.get("conservative_path_equivalents") == 157
        and reconciliation.get("minimum_time_and_memory_headroom_fraction") == 0.25
        and reconciliation.get("frozen_twelve_hour_request_80_percent_seconds") == 34560
    ):
        raise RealM1Error("the approved production-only 12-hour amendment is absent")
    if not (
        assessment.get("accounting_epoch") == "wall_reset_20261001"
        and reservation.get("reservation_included") is True
        and reservation.get("request_wall_hours") == 12
        and reservation.get("requested_cpus") == 1
        and reservation.get("conservative_allocated_cpus") == 4
        and reservation.get("memory_GiB") == 16
        and reservation.get("qualification_job_id") == QUALIFICATION_JOB_ID
        and reservation.get("numerical_refresh_job_id") == NUMERICAL_REFRESH_JOB_ID
        and reservation.get("assessed_projection_status") == "PASS"
        and 1 <= allocated_cpus <= 4
    ):
        raise RealM1Error("current clean-production resource reservation is absent")
    for key in ("remaining_unreserved_wall_hours", "remaining_unreserved_allocated_core_hours"):
        remaining = float(reservation.get(key, math.nan))
        if not math.isfinite(remaining) or remaining < 0:
            raise RealM1Error("clean-production reservation exceeds remaining resources")

    recovery = refresh.get("recovery", {})
    sampled = recovery.get("sampled_one_hot_recovery", {})
    population = recovery.get("exact_population_recovery", {})
    throughput = refresh.get("all_K8_numerical_throughput", {})
    seals = ("real_target_calls_or_real_K_opened", "M0_prediction_or_fit_artifact_opened",
             "final_or_unassigned_or_SWC_opened")
    if not (
        str(refresh.get("SLURM_job_id")) == NUMERICAL_REFRESH_JOB_ID
        and refresh.get("qualification_job_reused") == QUALIFICATION_JOB_ID
        and refresh.get("status") == "fail"
        and refresh.get("failure_class") == "RefreshError"
        and refresh.get("failure_message")
        == "fresh 157-path projection exceeds frozen headroom or remaining resources"
        and refresh.get("resource_projection", {}).get("status") == "fail"
        and all(refresh.get(key) is False for key in seals)
        and recovery.get("status") == "pass"
        and recovery.get("rows") == 544 and recovery.get("groups") == 16
        and sampled.get("status") == "complete_descriptive"
        and sampled.get("optimizer", {}).get("status") == "pass"
        and population.get("status") == "pass"
        and population.get("optimizer", {}).get("status") == "pass"
        and population.get("implementation_consistency_witness", {}).get("pass") is True
        and all(check.get("pass") is True for check in population.get("acceptance_checks", {}).values())
        and len(population.get("acceptance_checks", {})) == 3
        and throughput.get("status") == "pass"
        and throughput.get("rows") == 194 and throughput.get("groups") == 12
        and throughput.get("all_K8_unique_candidate_states") == 2496780
        and set(throughput.get("complete_path_iterations", {}))
        == {format(lam, ".12g") for lam in m1.LAMBDAS_DESCENDING}
    ):
        raise RealM1Error("affected numerical refresh did not pass its frozen numerical gates")
    require_matching_qualification_backend(refresh, numeric_backend)
    seconds = float(throughput.get("complete_12_lambda_warm_path_seconds", math.nan))
    projected_hours = 157.0 * seconds / 3600.0
    projected_core_hours = 4.0 * projected_hours
    maxrss_kib = float(reservation.get("producer_batch_MaxRSS_KiB", math.nan))
    producer_seconds = float(reservation.get("producer_elapsed_seconds", math.nan))
    if not (
        math.isfinite(seconds) and seconds > 0 and 157.0 * seconds < 34560.0
        and math.isfinite(maxrss_kib) and 0 < maxrss_kib < 0.8 * 16 * 1024**2
        and math.isfinite(producer_seconds) and 0 < producer_seconds < 0.8 * 8 * 3600
        and float(reservation.get("projected_production_wall_hours", math.nan)) == projected_hours
        and float(reservation.get("projected_allocated_core_hours", math.nan)) == projected_core_hours
    ):
        raise RealM1Error("fresh 157-path projection lacks approved time/memory headroom")
    return {
        "status": "pass_under_user_authorized_12_hour_request",
        "original_refresh_status_preserved": "fail",
        "numerical_refresh_job_id": NUMERICAL_REFRESH_JOB_ID,
        "projected_production_wall_hours": projected_hours,
        "projected_allocated_core_hours": projected_core_hours,
        "producer_batch_MaxRSS_KiB": maxrss_kib,
        "producer_elapsed_seconds": producer_seconds,
        "resource_amendment": RESOURCE_AMENDMENT_PATH.name,
        "resource_assessment": RESOURCE_ASSESSMENT_PATH.name,
    }


def require_approved_numerical_refresh(
    refresh_root_argument: Path, amendment_path: Path, assessment_path: Path,
    numeric_backend: Mapping[str, Any],
) -> Mapping[str, Any]:
    root = m1._require_exact_scratch_child(
        refresh_root_argument,
        f"v3_m1_affected_numerics_refresh_v1_job_{NUMERICAL_REFRESH_JOB_ID}",
        "affected numerical refresh",
    )
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        refresh = _read_json_at(descriptor, "v3_m1_affected_numerics_refresh.json", "affected refresh")
    finally:
        os.close(descriptor)
    if refresh.get("output_root") != str(root):
        raise RealM1Error("affected refresh output-root identity changed")
    return assess_approved_numerical_refresh(
        refresh,
        _read_exact_resource_record(amendment_path, RESOURCE_AMENDMENT_PATH),
        _read_exact_resource_record(assessment_path, RESOURCE_ASSESSMENT_PATH),
        _read_exact_resource_record(WALL_AUTHORIZATION_PATH, WALL_AUTHORIZATION_PATH),
        numeric_backend, int(os.environ["SLURM_CPUS_PER_TASK"]),
    )


def require_actual_k_prerequisite(
    actual_k_root_argument: Path,
    *,
    membership_opened_by_job_id: str | None = None,
) -> Mapping[str, Any]:
    """Require the exact passed actual-K job before membership can be opened."""
    readiness = _read_readiness()
    gates = readiness.get("gates", {})
    access = readiness.get("access", {})
    actual_record = str(gates.get("gao_development_v3_m1_actual_K_gate", ""))
    expected_membership_access: bool | str = False
    if membership_opened_by_job_id is not None:
        if re.fullmatch(r"[0-9]+", membership_opened_by_job_id) is None:
            raise RealM1Error("prior membership-opening job ID is noncanonical")
        expected_membership_access = (
            f"true_job_{membership_opened_by_job_id}_"
            "provider_neuron_id_fMOST_brain_id_K_and_detected16_only"
        )
    if not (
        actual_record.startswith(f"pass_job_{ACTUAL_K_JOB_ID}_")
        and int(ACTUAL_K_JOB_ID) in readiness.get("launch", {}).get("slurm_job_ids", [])
        and access.get("v3_m1_actual_K_opened")
        == f"true_job_{ACTUAL_K_JOB_ID}_provider_neuron_id_and_physical_K_only"
        and access.get("v3_m1_realized_membership_opened")
        == expected_membership_access
        and access.get("v3_m1_M0_artifact_opened") == HISTORICAL_M0_ACCESS
        and access.get("final_group_outcomes_opened") is False
    ):
        raise RealM1Error("passed actual-K job or sealed readiness boundary is absent")

    root = m1._require_exact_scratch_child(
        actual_k_root_argument, ACTUAL_K_ROOT_BASENAME, "actual-K gate root"
    )
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        summary = _read_json_at(root_fd, ACTUAL_K_SUMMARY_FILENAME, "actual-K summary")
    finally:
        os.close(root_fd)
    rank = summary.get("rank_and_condition", {})
    if not (
        summary.get("status") == "pass"
        and str(summary.get("SLURM_job_id")) == ACTUAL_K_JOB_ID
        and summary.get("target_fields_opened")
        == "provider_neuron_id_and_physical_K_only"
        and summary.get("realized_positive_set_membership_opened") is False
        and summary.get("candidate_scores_or_model_fit_computed") is False
        and summary.get("M0_prediction_or_fit_artifact_opened") is False
        and summary.get("SWC_opened") is False
        and summary.get("final_or_unassigned_opened") is False
        and rank.get("status") == "pass"
        and int(rank.get("training_folds_evaluated", -1)) == 79
        and int(rank.get("training_folds_passed", -1)) == 79
        and int(rank.get("minimum_joint_rank", -1)) == m1.PARAMETER_DIMENSION
        and int(rank.get("minimum_geometry_residual_rank", -1))
        == m1.GEOMETRY_DIMENSION
        and float(rank.get("maximum_joint_condition_number_2", math.inf)) <= 1.0e10
    ):
        raise RealM1Error("actual-K gate artifact is not passed and sealed")
    return summary


def _scan_membership_projection(
    handle: Any, expected_ids: set[str]
) -> tuple[dict[str, tuple[str, int, tuple[int, ...]]], MembershipScanCounts]:
    """Project only ID, group, K, and detected16 from the physical 90-field CSV.

    Forbidden fields transit the binary input buffer but are never assembled.
    For the six rows outside the 194-row interface, K and detected16 are only
    checked for emptiness and are not materialized.
    """
    expected = set(expected_ids)
    projected: dict[str, tuple[str, int, tuple[int, ...]]] = {}
    seen_identifiers: set[str] = set()
    record_index = 0
    data_rows = 0
    field_index = 0
    field_buffer = bytearray()
    field_nonempty = False
    field_at_start = True
    record_active = False
    in_quotes = False
    quote_pending = False
    current_identifier: str | None = None
    current_matched = False
    current_group: str | None = None
    current_k: int | None = None
    current_detected: list[int] = []

    selected_indexes = {
        TARGET_IDENTIFIER_INDEX,
        TARGET_GROUP_INDEX,
        TARGET_K_INDEX,
        *range(TARGET_DETECTED_START_INDEX, TARGET_DETECTED_STOP_INDEX),
    }

    def capture_current_field() -> bool:
        if record_index == 0:
            return field_index in selected_indexes
        return field_index == TARGET_IDENTIFIER_INDEX or (
            current_matched
            and (
                field_index in {TARGET_GROUP_INDEX, TARGET_K_INDEX}
                or TARGET_DETECTED_START_INDEX
                <= field_index
                < TARGET_DETECTED_STOP_INDEX
            )
        )

    def append_selected(value: bytes) -> None:
        if capture_current_field():
            field_buffer.extend(value)

    def finish_field() -> None:
        nonlocal field_index, field_nonempty, field_at_start
        nonlocal current_identifier, current_matched, current_group, current_k
        nonlocal current_detected
        value = bytes(field_buffer)
        field_buffer.clear()
        if record_index == 0:
            expected_header: bytes | None = None
            if field_index == TARGET_IDENTIFIER_INDEX:
                expected_header = b"provider_neuron_id"
            elif field_index == TARGET_GROUP_INDEX:
                expected_header = b"fMOST_brain_id"
            elif field_index == TARGET_K_INDEX:
                expected_header = b"K"
            elif TARGET_DETECTED_START_INDEX <= field_index < TARGET_DETECTED_STOP_INDEX:
                panel_index = field_index - TARGET_DETECTED_START_INDEX
                expected_header = f"detected__{m1.PANEL[panel_index]}".encode("ascii")
            if expected_header is not None and value != expected_header:
                raise RealM1Error("physical target membership columns changed")
        elif field_index == TARGET_IDENTIFIER_INDEX:
            try:
                identifier = value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise RealM1Error("target identifier is not UTF-8") from error
            if not identifier or identifier in seen_identifiers:
                raise RealM1Error("target identifiers are empty or duplicated")
            seen_identifiers.add(identifier)
            current_identifier = identifier
            current_matched = identifier in expected
        elif field_index == TARGET_GROUP_INDEX:
            if current_identifier is None:
                raise RealM1Error("target group precedes its identifier")
            if current_matched:
                try:
                    group = value.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise RealM1Error("target group is not UTF-8") from error
                if not group:
                    raise RealM1Error("development membership group is empty")
                current_group = group
        elif field_index == TARGET_K_INDEX:
            if current_identifier is None:
                raise RealM1Error("target K precedes its identifier")
            if current_matched:
                if K_PATTERN.fullmatch(value) is None:
                    raise RealM1Error("development membership contains noncanonical K")
                current_k = int(value)
            elif field_nonempty:
                raise RealM1Error("excluded target row has nonempty K")
        elif TARGET_DETECTED_START_INDEX <= field_index < TARGET_DETECTED_STOP_INDEX:
            if current_identifier is None:
                raise RealM1Error("detected membership precedes its identifier")
            if current_matched:
                if value not in {b"0", b"1"}:
                    raise RealM1Error("development membership is not binary")
                current_detected.append(int(value))
            elif field_nonempty:
                raise RealM1Error("excluded target row has nonempty detected membership")
        field_index += 1
        field_nonempty = False
        field_at_start = True

    def finish_record() -> None:
        nonlocal record_index, data_rows, field_index, record_active
        nonlocal current_identifier, current_matched, current_group, current_k
        nonlocal current_detected
        if field_index != TARGET_FIELD_COUNT:
            raise RealM1Error("development target-call CSV physical width changed")
        if record_index > 0:
            if current_identifier is None:
                raise RealM1Error("development membership row lacks an identifier")
            if current_matched:
                if (
                    current_group is None
                    or current_k is None
                    or len(current_detected) != m1.PANEL_SIZE
                ):
                    raise RealM1Error("development membership row is incomplete")
                if sum(current_detected) != current_k:
                    raise RealM1Error("development K disagrees with detected16")
                projected[current_identifier] = (
                    current_group,
                    current_k,
                    tuple(current_detected),
                )
            data_rows += 1
        record_index += 1
        field_index = 0
        record_active = False
        current_identifier = None
        current_matched = False
        current_group = None
        current_k = None
        current_detected = []

    while True:
        byte = handle.read(1)
        if not byte:
            if in_quotes and not quote_pending:
                raise RealM1Error("development target-call CSV has an unterminated quote")
            if quote_pending:
                in_quotes = False
                quote_pending = False
            if record_active or field_index or field_nonempty or field_buffer:
                finish_field()
                finish_record()
            break

        closed_quote = False
        if in_quotes:
            record_active = True
            if quote_pending:
                if byte == b'"':
                    field_nonempty = True
                    append_selected(b'"')
                    quote_pending = False
                    continue
                in_quotes = False
                quote_pending = False
                closed_quote = True
            elif byte == b'"':
                quote_pending = True
                continue
            else:
                field_nonempty = True
                append_selected(byte)
                continue
        elif field_at_start and byte == b'"':
            in_quotes = True
            field_at_start = False
            record_active = True
            continue
        elif byte == b'"':
            raise RealM1Error("development target-call CSV has a stray quote")

        if byte == b",":
            record_active = True
            finish_field()
        elif byte == b"\n":
            finish_field()
            finish_record()
        elif byte == b"\r":
            if handle.read(1) != b"\n":
                raise RealM1Error("development target-call CSV line ending changed")
            finish_field()
            finish_record()
        elif closed_quote:
            raise RealM1Error("quoted target field has trailing characters")
        else:
            record_active = True
            field_at_start = False
            field_nonempty = True
            append_selected(byte)

    if record_index == 0:
        raise RealM1Error("development target-call CSV is empty")
    return projected, MembershipScanCounts(
        total_rows=data_rows,
        matched_rows=len(projected),
        extra_rows=data_rows - len(projected),
    )


def _read_membership_inputs(
    target_root_argument: Path, expected_ids: set[str]
) -> tuple[dict[str, tuple[str, int, tuple[int, ...]]], MembershipScanCounts]:
    root = m1._require_exact_scratch_child(
        target_root_argument, TARGET_ROOT_BASENAME, "development target root"
    )
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        upstream = _read_json_at(root_fd, TARGET_GATE_FILENAME, "target gate summary")
        if not (
            upstream.get("status") == "pass"
            and int(upstream.get("input_files", -1)) == EXPECTED_TARGET_ROWS
            and int(upstream.get("eligible_cells", -1)) == EXPECTED_ROWS
            and int(upstream.get("excluded_cells", -1)) == EXPECTED_EXCLUDED_ROWS
            and upstream.get("final_or_unassigned_files_opened") is False
        ):
            raise RealM1Error("upstream development target gate is not passed and sealed")
        descriptor = _open_regular_at(root_fd, TARGET_FILENAME)
        with os.fdopen(descriptor, "rb", buffering=65536) as handle:
            projected, counts = _scan_membership_projection(handle, expected_ids)
    finally:
        os.close(root_fd)
    if not (
        counts.total_rows == EXPECTED_TARGET_ROWS
        and counts.matched_rows == EXPECTED_ROWS
        and counts.extra_rows == EXPECTED_EXCLUDED_ROWS
        and set(projected) == expected_ids
    ):
        raise RealM1Error("development membership does not match the 194-row interface")
    return projected, counts


def _align_membership(
    cells: Sequence[Any],
    membership: Mapping[str, tuple[str, int, tuple[int, ...]]],
    library: Any,
) -> tuple[Any, ...]:
    identifiers = tuple(cell.identifier for cell in cells)
    if not (
        len(cells) == EXPECTED_ROWS
        and len(set(identifiers)) == EXPECTED_ROWS
        and set(identifiers) == set(membership)
    ):
        raise RealM1Error("development membership join is not exact")
    aligned: list[Any] = []
    for cell in cells:
        group, k, detected = membership[cell.identifier]
        if group != cell.group:
            raise RealM1Error("development target/interface group mismatch")
        vector = np.asarray(detected, dtype=np.uint8)
        if vector.shape != (m1.PANEL_SIZE,) or int(np.sum(vector)) != k:
            raise RealM1Error("detected16 vector is malformed")
        basis = library.basis(k)
        matches = np.flatnonzero(np.all(basis.membership == vector, axis=1))
        if matches.size != 1:
            raise RealM1Error("observed set lacks one lexicographic candidate index")
        aligned.append(replace(cell, k=k, observed_index=int(matches[0])))
    return tuple(aligned)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    if path.is_symlink() or temporary.is_symlink():
        raise RealM1Error("refusing symlinked JSON output")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o660,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _atomic_csv(
    path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    if path.is_symlink() or temporary.is_symlink():
        raise RealM1Error("refusing symlinked CSV output")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o660,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.is_symlink():
        raise RealM1Error("refusing to overwrite or symlink a fit checkpoint")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o660,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _prepare_output_root(
    path: Path,
    execution_identity: Mapping[str, Any],
) -> tuple[Path, Path, bool, Mapping[str, Any] | None]:
    scratch_value = os.environ.get("SCRATCH")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not scratch_value or not job_id:
        raise RealM1Error("real M1 requires SCRATCH and SLURM_JOB_ID")
    scratch = Path(scratch_value).resolve(strict=True)
    episode = (scratch / "br_autoresearch" / "episode10_single_cell_coprojection").resolve(
        strict=True
    )
    absolute = Path(os.path.abspath(path))
    if (
        absolute.parent != episode
        or absolute.name != f"{OUTPUT_PREFIX}_job_{job_id}"
        or absolute.is_symlink()
    ):
        raise RealM1Error("real-M1 output is not the job-scoped Scratch child")
    resumed = absolute.exists()
    completed_summary: Mapping[str, Any] | None = None
    if resumed:
        if not absolute.is_dir() or absolute.resolve(strict=True) != absolute:
            raise RealM1Error("existing real-M1 output root is not an exact directory")
        manifest_path = absolute / CHECKPOINT_MANIFEST
        if manifest_path.exists():
            if not manifest_path.is_file() or manifest_path.is_symlink():
                raise RealM1Error("existing checkpoint manifest is not an ordinary file")
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if not (
                isinstance(manifest, dict)
                and manifest.get("stage") == PRODUCTION_STAGE
                and str(manifest.get("SLURM_job_id")) == job_id
                and manifest.get("actual_K_gate_job_id") == ACTUAL_K_JOB_ID
                and manifest.get("execution_identity")
                == dict(execution_identity)
                and manifest.get("M0_prediction_or_fit_artifact_opened") is False
                and manifest.get("final_or_unassigned_or_SWC_opened") is False
            ):
                raise RealM1Error("existing checkpoint manifest is not resumable")
        else:
            # A same-job Slurm requeue can occur after mkdir but before the
            # first manifest rename.  Accept only that narrow bootstrap state.
            partial_pattern = re.compile(
                rf"\.{re.escape(CHECKPOINT_MANIFEST)}\.partial\.[0-9]+\Z"
            )
            for child in absolute.iterdir():
                if child.name == CHECKPOINT_DIRECTORY:
                    if (
                        not child.is_dir()
                        or child.is_symlink()
                        or any(child.iterdir())
                    ):
                        raise RealM1Error("manifest-free checkpoint bootstrap is not empty")
                elif partial_pattern.fullmatch(child.name):
                    if not child.is_file() or child.is_symlink():
                        raise RealM1Error("manifest bootstrap partial is not ordinary")
                    child.unlink()
                else:
                    raise RealM1Error("existing output root lacks a safe bootstrap manifest")
        summary_path = absolute / SUMMARY_FILENAME
        if summary_path.exists():
            if not summary_path.is_file() or summary_path.is_symlink():
                raise RealM1Error("existing real-M1 summary is not an ordinary file")
            with summary_path.open("r", encoding="utf-8") as handle:
                prior_summary = json.load(handle)
            if prior_summary.get("status") == "pass":
                if not (
                    prior_summary.get("stage")
                    == PRODUCTION_STAGE
                    and prior_summary.get("actual_K_gate_job_id") == ACTUAL_K_JOB_ID
                    and prior_summary.get("execution_identity")
                    == dict(execution_identity)
                    and int(prior_summary.get("development_cells", -1)) == EXPECTED_ROWS
                    and int(prior_summary.get("development_groups", -1))
                    == EXPECTED_GROUPS
                    and prior_summary.get("M0_prediction_or_fit_artifact_opened")
                    is False
                    and prior_summary.get("SWC_opened") is False
                    and prior_summary.get("final_or_unassigned_opened") is False
                ):
                    raise RealM1Error("completed real-M1 summary identity changed")
                completed_summary = prior_summary
    else:
        absolute.mkdir(mode=0o770)
    checkpoint_root = absolute / CHECKPOINT_DIRECTORY
    if checkpoint_root.exists():
        if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
            raise RealM1Error("checkpoint root is not an ordinary directory")
    else:
        checkpoint_root.mkdir(mode=0o770)
    if completed_summary is not None:
        for name in REQUIRED_OUTPUTS:
            artifact = absolute / name
            if not artifact.is_file() or artifact.is_symlink():
                raise RealM1Error("completed real-M1 artifact set is incomplete")
    return absolute, checkpoint_root, resumed, completed_summary


def _write_checkpoint_manifest(
    output_root: Path,
    *,
    execution_identity: Mapping[str, Any],
    phase: str,
    realized_membership_opened: bool,
    completed_warm_paths: int,
    completed_zero_refits: int,
) -> None:
    _atomic_json(
        output_root / CHECKPOINT_MANIFEST,
        {
            "status": "complete" if phase == "complete" else "in_progress",
            "stage": PRODUCTION_STAGE,
            "phase": phase,
            "SLURM_job_id": os.environ["SLURM_JOB_ID"],
            "actual_K_gate_job_id": ACTUAL_K_JOB_ID,
            "execution_identity": dict(execution_identity),
            "development_cells": EXPECTED_ROWS,
            "development_groups": EXPECTED_GROUPS,
            "completed_warm_paths": completed_warm_paths,
            "completed_zero_refits": completed_zero_refits,
            "realized_development_membership_opened": realized_membership_opened,
            "M0_prediction_or_fit_artifact_opened": False,
            "final_or_unassigned_or_SWC_opened": False,
            "updated_at_unix_seconds": time.time(),
        },
    )


def _optimizer_payload(result: Any) -> dict[str, Any]:
    return {
        "objective": float(result.objective),
        "gradient_infinity_norm": float(result.gradient_infinity_norm),
        "iterations": int(result.iterations),
        "accepted_steps": int(result.accepted_steps),
        "relative_objective_change": (
            float(result.relative_objective_change)
            if math.isfinite(float(result.relative_objective_change))
            else None
        ),
        "telemetry": _jsonable(result.telemetry),
    }


def _validate_optimizer_result(result: Any, ridge_lambda: float) -> None:
    theta = np.asarray(result.theta, dtype=float)
    base_pass = bool(
        theta.shape == (m1.PARAMETER_DIMENSION,)
        and np.all(np.isfinite(theta))
        and math.isfinite(float(result.objective))
        and math.isfinite(float(result.gradient_infinity_norm))
        and float(result.gradient_infinity_norm) <= m1.GRADIENT_TOLERANCE
        and result.telemetry.get("status") == "pass"
        and float(result.telemetry.get("ridge_lambda", math.nan)) == float(ridge_lambda)
    )
    witness = result.telemetry.get("convergence_witness")
    if witness in {
        "gradient_and_relative_objective_change",
        "gradient_and_repeated_unchanged_initial_objective",
    }:
        witness_pass = bool(
            math.isfinite(float(result.relative_objective_change))
            and float(result.relative_objective_change) <= m1.RELATIVE_TOLERANCE
        )
    elif witness == "strong_convexity_gap_after_zoom_stall":
        certificate = result.telemetry.get("certificate", {})
        witness_pass = bool(
            isinstance(certificate, Mapping)
            and certificate.get("certificate_pass") is True
            and certificate.get("gradient_predicate_pass") is True
            and certificate.get("relative_gap_predicate_pass") is True
        )
    else:
        witness_pass = False
    if not (base_pass and witness_pass):
        raise RealM1Error("optimizer result lacks the frozen convergence witness")


def _checkpoint_path(checkpoint_root: Path, kind: str, key: str) -> Path:
    if kind not in {"warm", "zero"} or CHECKPOINT_KEY_PATTERN.fullmatch(key) is None:
        raise RealM1Error("checkpoint kind or key is invalid")
    return checkpoint_root / f"{kind}__{key}.npz"


def _write_fit_checkpoint(
    path: Path,
    *,
    kind: str,
    key: str,
    results: Mapping[float, Any],
    training_cells: int,
    training_groups: Sequence[str],
) -> None:
    lambdas = tuple(float(value) for value in results)
    theta = np.vstack([np.asarray(results[value].theta, dtype=float) for value in lambdas])
    telemetry = np.asarray(
        [
            json.dumps(
                _jsonable(results[value].telemetry),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            for value in lambdas
        ],
        dtype=np.str_,
    )
    _atomic_npz(
        path,
        schema=np.asarray(CHECKPOINT_SCHEMA),
        kind=np.asarray(kind),
        key=np.asarray(key),
        actual_k_job_id=np.asarray(ACTUAL_K_JOB_ID),
        qualification_job_id=np.asarray(QUALIFICATION_JOB_ID),
        prior_membership_opening_job_id=np.asarray(
            PRIOR_MEMBERSHIP_OPENING_JOB_ID
        ),
        numerical_execution_schema_version=np.asarray(EXECUTION_SCHEMA_VERSION),
        producer_slurm_job_id=np.asarray(os.environ["SLURM_JOB_ID"]),
        training_cells=np.asarray(training_cells, dtype=np.int64),
        training_groups=np.asarray(tuple(training_groups), dtype=np.str_),
        lambdas=np.asarray(lambdas, dtype=float),
        theta=theta,
        objective=np.asarray([results[value].objective for value in lambdas], dtype=float),
        gradient=np.asarray(
            [results[value].gradient_infinity_norm for value in lambdas], dtype=float
        ),
        iterations=np.asarray([results[value].iterations for value in lambdas], dtype=np.int64),
        accepted_steps=np.asarray(
            [results[value].accepted_steps for value in lambdas], dtype=np.int64
        ),
        relative_change=np.asarray(
            [results[value].relative_objective_change for value in lambdas], dtype=float
        ),
        telemetry_json=telemetry,
    )


def _load_fit_checkpoint(
    path: Path,
    *,
    kind: str,
    key: str,
    expected_lambdas: Sequence[float],
    training_cells: int,
    training_groups: Sequence[str],
) -> Mapping[float, Any]:
    if not path.is_file() or path.is_symlink():
        raise RealM1Error("fit checkpoint is absent or symlinked")
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "schema",
                "kind",
                "key",
                "actual_k_job_id",
                "qualification_job_id",
                "prior_membership_opening_job_id",
                "numerical_execution_schema_version",
                "producer_slurm_job_id",
                "training_cells",
                "training_groups",
                "lambdas",
                "theta",
                "objective",
                "gradient",
                "iterations",
                "accepted_steps",
                "relative_change",
                "telemetry_json",
            }
            if set(data.files) != required:
                raise RealM1Error("fit checkpoint fields changed")
            lambdas = np.asarray(data["lambdas"], dtype=float)
            expected = np.asarray(tuple(expected_lambdas), dtype=float)
            if not (
                str(data["schema"].item()) == CHECKPOINT_SCHEMA
                and str(data["kind"].item()) == kind
                and str(data["key"].item()) == key
                and str(data["actual_k_job_id"].item()) == ACTUAL_K_JOB_ID
                and str(data["qualification_job_id"].item())
                == QUALIFICATION_JOB_ID
                and str(data["prior_membership_opening_job_id"].item())
                == PRIOR_MEMBERSHIP_OPENING_JOB_ID
                and str(data["numerical_execution_schema_version"].item())
                == EXECUTION_SCHEMA_VERSION
                and str(data["producer_slurm_job_id"].item())
                == os.environ.get("SLURM_JOB_ID")
                and int(data["training_cells"].item()) == training_cells
                and tuple(str(value) for value in data["training_groups"].tolist())
                == tuple(training_groups)
                and np.array_equal(lambdas, expected)
            ):
                raise RealM1Error("fit checkpoint identity changed")
            theta = np.asarray(data["theta"], dtype=float)
            objective = np.asarray(data["objective"], dtype=float)
            gradient = np.asarray(data["gradient"], dtype=float)
            iterations = np.asarray(data["iterations"], dtype=np.int64)
            accepted = np.asarray(data["accepted_steps"], dtype=np.int64)
            relative = np.asarray(data["relative_change"], dtype=float)
            telemetry_json = tuple(str(value) for value in data["telemetry_json"].tolist())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise RealM1Error("fit checkpoint is unreadable") from error
    count = len(expected_lambdas)
    if not (
        theta.shape == (count, m1.PARAMETER_DIMENSION)
        and objective.shape == gradient.shape == relative.shape == (count,)
        and iterations.shape == accepted.shape == (count,)
        and len(telemetry_json) == count
        and np.all(np.isfinite(theta))
        and np.all(np.isfinite(objective))
        and np.all(np.isfinite(gradient))
    ):
        raise RealM1Error("fit checkpoint arrays are malformed")
    restored: dict[float, Any] = {}
    for index, ridge_lambda in enumerate(expected_lambdas):
        telemetry = json.loads(telemetry_json[index])
        result = m1.OptimizerResult(
            theta=theta[index].copy(),
            objective=float(objective[index]),
            gradient_infinity_norm=float(gradient[index]),
            iterations=int(iterations[index]),
            accepted_steps=int(accepted[index]),
            relative_objective_change=float(relative[index]),
            telemetry=telemetry,
        )
        _validate_optimizer_result(result, float(ridge_lambda))
        restored[float(ridge_lambda)] = result
    return restored


def _prepare_training(
    cells: Sequence[Any], moments: Sequence[Any], indexes: Sequence[int]
) -> Any:
    selected = tuple(int(index) for index in indexes)
    if not selected or len(set(selected)) != len(selected):
        raise RealM1Error("training indexes are empty or duplicated")
    return m1.prepare_data(cells, moments, selected)


def _production_warm_factory(
    key: str,
    indexes: Sequence[int],
    cells: Sequence[Any],
    moments: Sequence[Any],
    library: Any,
    geometry: Any,
    checkpoint_root: Path,
) -> TrainingBundle:
    prepared = _prepare_training(cells, moments, indexes)
    groups = tuple(sorted({cell.group for cell in prepared.cells}, key=m1.GROUP_ORDER.index))
    path = _checkpoint_path(checkpoint_root, "warm", key)
    if path.exists():
        fits = _load_fit_checkpoint(
            path,
            kind="warm",
            key=key,
            expected_lambdas=m1.LAMBDAS_DESCENDING,
            training_cells=len(prepared.cells),
            training_groups=groups,
        )
        reused = True
    else:
        try:
            fits = m1.fit_path(prepared, library, geometry)
        except m1.M1Error as error:
            error.production_fit_kind = "warm"
            error.production_fit_key = key
            raise
        if tuple(fits) != m1.LAMBDAS_DESCENDING:
            raise RealM1Error("warm path lambda order changed")
        for ridge_lambda, result in fits.items():
            _validate_optimizer_result(result, ridge_lambda)
        _write_fit_checkpoint(
            path,
            kind="warm",
            key=key,
            results=fits,
            training_cells=len(prepared.cells),
            training_groups=groups,
        )
        reused = False
    return TrainingBundle(key, tuple(int(i) for i in indexes), prepared, fits, reused)


def _production_zero_factory(
    key: str,
    indexes: Sequence[int],
    ridge_lambda: float,
    cells: Sequence[Any],
    moments: Sequence[Any],
    library: Any,
    geometry: Any,
    checkpoint_root: Path,
) -> TrainingBundle:
    prepared = _prepare_training(cells, moments, indexes)
    groups = tuple(sorted({cell.group for cell in prepared.cells}, key=m1.GROUP_ORDER.index))
    path = _checkpoint_path(checkpoint_root, "zero", key)
    if path.exists():
        fits = _load_fit_checkpoint(
            path,
            kind="zero",
            key=key,
            expected_lambdas=(ridge_lambda,),
            training_cells=len(prepared.cells),
            training_groups=groups,
        )
        reused = True
    else:
        try:
            result = m1.fit_newton(
                prepared, library, geometry, float(ridge_lambda), initial=None
            )
        except m1.M1Error as error:
            error.production_fit_kind = "zero"
            error.production_fit_key = key
            error.production_fit_lambda = float(ridge_lambda)
            raise
        _validate_optimizer_result(result, ridge_lambda)
        fits = {float(ridge_lambda): result}
        _write_fit_checkpoint(
            path,
            kind="zero",
            key=key,
            results=fits,
            training_cells=len(prepared.cells),
            training_groups=groups,
        )
        reused = False
    return TrainingBundle(key, tuple(int(i) for i in indexes), prepared, fits, reused)


def _production_predictor(
    bundle: TrainingBundle,
    result: Any,
    validation_indexes: Sequence[int],
    cells: Sequence[Any],
    moments: Sequence[Any],
    library: Any,
    geometry: Any,
) -> Prediction:
    selected = tuple(int(index) for index in validation_indexes)
    held_cells = tuple(cells[index] for index in selected)
    held_moments = tuple(moments[index] for index in selected)
    prepared = m1.apply_fixed_transforms(
        held_cells,
        held_moments,
        bundle.prepared.design_transform,
        bundle.prepared.geometry_scales,
    )
    evaluation = m1.evaluate_model(
        result.theta,
        prepared,
        library,
        geometry,
        0.0,
        need_hessian=False,
        need_predictions=True,
    )
    if evaluation.log_probability_nats is None or evaluation.marginals is None:
        raise RealM1Error("prediction evaluation omitted required arrays")
    log_probability = np.asarray(evaluation.log_probability_nats, dtype=float)
    marginals = np.asarray(evaluation.marginals, dtype=float)
    expected_k = np.asarray([cell.k for cell in held_cells], dtype=float)
    if not (
        log_probability.shape == (len(held_cells),)
        and marginals.shape == (len(held_cells), m1.PANEL_SIZE)
        and np.all(np.isfinite(log_probability))
        and np.all(np.isfinite(marginals))
        and np.all(log_probability <= 1.0e-12)
        and np.all(marginals >= -1.0e-12)
        and np.all(marginals <= 1.0 + 1.0e-12)
        and np.allclose(np.sum(marginals, axis=1), expected_k, rtol=0.0, atol=1.0e-10)
    ):
        raise RealM1Error("exact prediction arrays are malformed")
    return Prediction(log_probability.copy(), marginals.copy())


def _group_indexes(cells: Sequence[Any]) -> Mapping[str, tuple[int, ...]]:
    groups: dict[str, list[int]] = {group: [] for group in m1.GROUP_ORDER}
    for index, cell in enumerate(cells):
        if cell.group not in groups:
            raise RealM1Error("cell contains a group outside the frozen order")
        groups[cell.group].append(index)
    observed = {group: len(indexes) for group, indexes in groups.items()}
    if observed != m1.GROUP_SIZES:
        raise RealM1Error("real-M1 group sizes changed")
    return {group: tuple(indexes) for group, indexes in groups.items()}


def _training_indexes(
    cells: Sequence[Any], excluded_groups: set[str]
) -> tuple[int, ...]:
    return tuple(
        index for index, cell in enumerate(cells) if cell.group not in excluded_groups
    )


def select_lambda_one_se(
    validation_nll_by_lambda: Mapping[float, Sequence[float]],
    expected_folds: int,
) -> tuple[float, dict[str, Any]]:
    summaries: dict[float, tuple[float, float]] = {}
    for ridge_lambda in m1.LAMBDAS_DESCENDING:
        values = np.asarray(validation_nll_by_lambda[ridge_lambda], dtype=float)
        if values.shape != (expected_folds,) or not np.all(np.isfinite(values)):
            raise RealM1Error("one-SE selection lacks the exact finite group losses")
        mean = float(np.mean(values))
        standard_error = float(np.std(values, ddof=1) / math.sqrt(expected_folds))
        summaries[ridge_lambda] = (mean, standard_error)
    minimum_lambda = min(
        m1.LAMBDAS_DESCENDING,
        key=lambda value: (summaries[value][0], -value),
    )
    threshold = summaries[minimum_lambda][0] + summaries[minimum_lambda][1]
    qualifying = [
        value
        for value in m1.LAMBDAS_DESCENDING
        if summaries[value][0] <= threshold
    ]
    if not qualifying:
        raise RealM1Error("one-SE selection has no qualifying lambda")
    selected = max(qualifying)
    return selected, {
        "minimum_mean_lambda": minimum_lambda,
        "minimum_mean_nll_nats": summaries[minimum_lambda][0],
        "standard_error_at_minimum_nats": summaries[minimum_lambda][1],
        "one_se_threshold_nats": threshold,
        "selected_largest_qualifying_lambda": selected,
        "grid": [
            {
                "lambda": value,
                "mean_group_nll_nats": summaries[value][0],
                "group_standard_error_nats": summaries[value][1],
            }
            for value in m1.LAMBDAS_DESCENDING
        ],
    }


def select_outer_lambdas(
    cells: Sequence[Any],
    moments: Sequence[Any],
    library: Any,
    geometry: Any,
    checkpoint_root: Path,
    *,
    warm_factory: WarmFactory = _production_warm_factory,
    predictor: Predictor = _production_predictor,
) -> OuterSelection:
    by_group = _group_indexes(cells)
    warm_cache: dict[str, TrainingBundle] = {}
    selections: dict[str, float] = {}
    details: list[Mapping[str, Any]] = []
    for outer_group in m1.GROUP_ORDER:
        validation: dict[float, list[float]] = {
            value: [] for value in m1.LAMBDAS_DESCENDING
        }
        for inner_group in m1.GROUP_ORDER:
            if inner_group == outer_group:
                continue
            pair = tuple(
                sorted(
                    (outer_group, inner_group),
                    key=m1.GROUP_ORDER.index,
                )
            )
            key = f"leave_two__{pair[0]}__{pair[1]}"
            if key not in warm_cache:
                indexes = _training_indexes(cells, set(pair))
                warm_cache[key] = warm_factory(
                    key,
                    indexes,
                    cells,
                    moments,
                    library,
                    geometry,
                    checkpoint_root,
                )
            bundle = warm_cache[key]
            for ridge_lambda in m1.LAMBDAS_DESCENDING:
                predicted = predictor(
                    bundle,
                    bundle.fits[ridge_lambda],
                    by_group[inner_group],
                    cells,
                    moments,
                    library,
                    geometry,
                )
                validation[ridge_lambda].append(
                    float(-np.mean(predicted.log_probability_nats))
                )
        selected, selection = select_lambda_one_se(validation, 11)
        selections[outer_group] = selected
        boundary = selected in {
            min(m1.LAMBDAS_DESCENDING),
            max(m1.LAMBDAS_DESCENDING),
        }
        details.append(
            {
                "held_out_group": outer_group,
                "held_out_cells": len(by_group[outer_group]),
                "training_groups": 11,
                "inner_folds": 11,
                "selected_lambda": selected,
                "selected_lambda_is_grid_boundary": boundary,
                "inner_selection": selection,
            }
        )
        print(
            json.dumps(
                {
                    "event": "M1_outer_tuning_complete",
                    "held_out_group": outer_group,
                    "selected_lambda": selected,
                    "boundary": boundary,
                    "unique_leave_two_paths_complete": len(warm_cache),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if len(warm_cache) != math.comb(EXPECTED_GROUPS, 2):
        raise RealM1Error("unordered leave-two path reuse did not yield exactly 66 paths")
    boundary_count = sum(
        bool(detail["selected_lambda_is_grid_boundary"]) for detail in details
    )
    return OuterSelection(selections, tuple(details), boundary_count, warm_cache)


def refit_outer_and_predict(
    cells: Sequence[Any],
    moments: Sequence[Any],
    library: Any,
    geometry: Any,
    checkpoint_root: Path,
    selection: OuterSelection,
    *,
    zero_factory: ZeroFactory = _production_zero_factory,
    predictor: Predictor = _production_predictor,
) -> CrossFitResult:
    if selection.boundary_count >= 6:
        raise BoundaryReview("ridge-grid boundary selected in at least six outer folds")
    by_group = _group_indexes(cells)
    log_probability = np.full(len(cells), np.nan, dtype=float)
    marginals = np.full((len(cells), m1.PANEL_SIZE), np.nan, dtype=float)
    refits: dict[str, TrainingBundle] = {}
    details: list[Mapping[str, Any]] = []
    for prior_detail in selection.details:
        group = str(prior_detail["held_out_group"])
        ridge_lambda = float(selection.selected_lambdas[group])
        key = f"outer_refit__{group}"
        indexes = _training_indexes(cells, {group})
        bundle = zero_factory(
            key,
            indexes,
            ridge_lambda,
            cells,
            moments,
            library,
            geometry,
            checkpoint_root,
        )
        result = bundle.fits[ridge_lambda]
        predicted = predictor(
            bundle,
            result,
            by_group[group],
            cells,
            moments,
            library,
            geometry,
        )
        log_probability[list(by_group[group])] = predicted.log_probability_nats
        marginals[list(by_group[group])] = predicted.marginals
        refits[group] = bundle
        detail = dict(prior_detail)
        detail["outer_refit_initialization"] = "all_261_parameters_zero"
        detail["outer_refit_optimizer"] = {
            key: value
            for key, value in _optimizer_payload(result).items()
            if key != "telemetry"
        }
        details.append(detail)
        print(
            json.dumps(
                {
                    "event": "M1_outer_refit_complete",
                    "held_out_group": group,
                    "held_out_cells": len(by_group[group]),
                    "selected_lambda": ridge_lambda,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not (
        len(refits) == EXPECTED_GROUPS
        and np.all(np.isfinite(log_probability))
        and np.all(np.isfinite(marginals))
    ):
        raise RealM1Error("outer cross-fit left a development cell unpredicted")
    return CrossFitResult(
        Prediction(log_probability, marginals),
        tuple(details),
        selection.boundary_count,
        selection.warm_cache,
        refits,
    )


def tune_and_refit_full_development(
    cells: Sequence[Any],
    moments: Sequence[Any],
    library: Any,
    geometry: Any,
    checkpoint_root: Path,
    existing_warm_cache: Mapping[str, TrainingBundle],
    *,
    warm_factory: WarmFactory = _production_warm_factory,
    zero_factory: ZeroFactory = _production_zero_factory,
    predictor: Predictor = _production_predictor,
) -> FullFitResult:
    by_group = _group_indexes(cells)
    cache = dict(existing_warm_cache)
    validation: dict[float, list[float]] = {
        value: [] for value in m1.LAMBDAS_DESCENDING
    }
    for held_group in m1.GROUP_ORDER:
        key = f"leave_one__{held_group}"
        indexes = _training_indexes(cells, {held_group})
        if key not in cache:
            cache[key] = warm_factory(
                key,
                indexes,
                cells,
                moments,
                library,
                geometry,
                checkpoint_root,
            )
        bundle = cache[key]
        for ridge_lambda in m1.LAMBDAS_DESCENDING:
            predicted = predictor(
                bundle,
                bundle.fits[ridge_lambda],
                by_group[held_group],
                cells,
                moments,
                library,
                geometry,
            )
            validation[ridge_lambda].append(
                float(-np.mean(predicted.log_probability_nats))
            )
        print(
            json.dumps(
                {
                    "event": "M1_full_tuning_fold_complete",
                    "held_out_group": held_group,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if len(cache) != math.comb(EXPECTED_GROUPS, 2) + EXPECTED_GROUPS:
        raise RealM1Error("exact training-set reuse did not yield 78 warm paths")
    selected, selection = select_lambda_one_se(validation, EXPECTED_GROUPS)
    full_indexes = tuple(range(len(cells)))
    full_refit = zero_factory(
        "full_refit",
        full_indexes,
        selected,
        cells,
        moments,
        library,
        geometry,
        checkpoint_root,
    )
    return FullFitResult(selected, selection, cache, full_refit)


def _observed_matrix(cells: Sequence[Any], library: Any) -> np.ndarray:
    observed = np.empty((len(cells), m1.PANEL_SIZE), dtype=float)
    for index, cell in enumerate(cells):
        basis = library.basis(cell.k)
        if not 0 <= cell.observed_index < basis.count:
            raise RealM1Error("observed candidate index is outside its fixed-K space")
        observed[index] = basis.membership[cell.observed_index]
    return observed


def _equal_group_mean(values: np.ndarray, cells: Sequence[Any]) -> float:
    array = np.asarray(values, dtype=float)
    if array.shape != (len(cells),) or not np.all(np.isfinite(array)):
        raise RealM1Error("equal-group aggregation input is malformed")
    weights = m1.equal_group_weights([cell.group for cell in cells])
    return float(np.dot(weights, array))


def score_and_calibration(
    cells: Sequence[Any], library: Any, prediction: Prediction
) -> Mapping[str, Any]:
    observed = _observed_matrix(cells, library)
    log2_probability = prediction.log_probability_nats / m1.LN2
    uniform = np.asarray(
        [-math.log2(math.comb(m1.PANEL_SIZE, cell.k)) for cell in cells],
        dtype=float,
    )
    gain = log2_probability - uniform
    per_group: dict[str, Any] = {}
    for group in m1.GROUP_ORDER:
        indexes = np.asarray([cell.group == group for cell in cells], dtype=bool)
        residual = np.mean(observed[indexes] - prediction.marginals[indexes], axis=0)
        k_values = np.asarray([cell.k for cell in cells if cell.group == group], dtype=int)
        k_counts = Counter(int(value) for value in k_values)
        per_group[group] = {
            "cells": int(np.sum(indexes)),
            "mean_M1_log2_probability": float(np.mean(log2_probability[indexes])),
            "uniform_fixed_K_mean_log2_probability": float(np.mean(uniform[indexes])),
            "paired_M1_minus_uniform_gain_bits_per_cell": float(np.mean(gain[indexes])),
            "marginal_Brier_M1": float(
                np.mean((observed[indexes] - prediction.marginals[indexes]) ** 2)
            ),
            "observed_minus_expected_by_target": {
                target: float(residual[target_index])
                for target_index, target in enumerate(m1.PANEL)
            },
            "degenerate_K_counts": {
                "K_0": int(np.sum(k_values == 0)),
                "K_16": int(np.sum(k_values == 16)),
            },
            "K_i_obs_count_by_value": {
                str(value): int(k_counts.get(value, 0))
                for value in range(m1.PANEL_SIZE + 1)
            },
            "pair_informative_cells_K_2_through_15": int(
                np.sum((k_values >= 2) & (k_values <= 15))
            ),
        }
    return {
        "scoring_scale": "bits",
        "aggregate_equal_group_mean_M1_log2_probability": _equal_group_mean(
            log2_probability, cells
        ),
        "aggregate_equal_group_uniform_fixed_K_mean_log2_probability": _equal_group_mean(
            uniform, cells
        ),
        "aggregate_equal_group_paired_M1_minus_uniform_gain_bits_per_cell": (
            _equal_group_mean(gain, cells)
        ),
        "M1_minus_M0_secondary_gain": {
            "status": "pending_separate_post_M1_comparator",
            "M0_prediction_or_fit_artifact_opened": False,
        },
        "per_group": per_group,
        "degenerate_K_counts": {
            "K_0": int(sum(cell.k == 0 for cell in cells)),
            "K_16": int(sum(cell.k == 16 for cell in cells)),
        },
        "pair_informative_cells_K_2_through_15": int(
            sum(2 <= cell.k <= 15 for cell in cells)
        ),
        "calibration": {
            "prediction_type": "outer_cross_fitted_exact_inclusion_marginals",
            "bins_used": False,
            "group_by_target_residual_definition": (
                "group_mean(observed_recorded_indicator-exact_inclusion_marginal)"
            ),
            "per_group_marginal_Brier_reported": True,
        },
    }


def _write_predictions(
    output_root: Path, cells: Sequence[Any], prediction: Prediction
) -> None:
    log2_probability = prediction.log_probability_nats / m1.LN2
    uniform = np.asarray(
        [-math.log2(math.comb(m1.PANEL_SIZE, cell.k)) for cell in cells], dtype=float
    )
    fields = [
        "provider_neuron_id",
        "fMOST_brain_id",
        "K",
        "M1_outer_cross_fitted_log2_probability",
        "uniform_fixed_K_log2_probability",
        "M1_minus_uniform_gain_bits",
        *[f"exact_inclusion_marginal__{target}" for target in m1.PANEL],
    ]
    rows: list[dict[str, object]] = []
    for index, cell in enumerate(cells):
        row: dict[str, object] = {
            "provider_neuron_id": cell.identifier,
            "fMOST_brain_id": cell.group,
            "K": cell.k,
            "M1_outer_cross_fitted_log2_probability": format(
                log2_probability[index], ".17g"
            ),
            "uniform_fixed_K_log2_probability": format(uniform[index], ".17g"),
            "M1_minus_uniform_gain_bits": format(
                log2_probability[index] - uniform[index], ".17g"
            ),
        }
        for target_index, target in enumerate(m1.PANEL):
            row[f"exact_inclusion_marginal__{target}"] = format(
                prediction.marginals[index, target_index], ".17g"
            )
        rows.append(row)
    _atomic_csv(output_root / PREDICTION_FILENAME, fields, rows)


def _solver_telemetry(
    cross_fit: CrossFitResult, full: FullFitResult
) -> Mapping[str, Any]:
    warm_paths: list[Mapping[str, Any]] = []
    for key in sorted(full.warm_cache):
        bundle = full.warm_cache[key]
        warm_paths.append(
            {
                "training_set": key,
                "training_cells": len(bundle.prepared.cells),
                "training_groups": sorted(
                    {cell.group for cell in bundle.prepared.cells},
                    key=m1.GROUP_ORDER.index,
                ),
                "checkpoint_reused": bundle.checkpoint_reused,
                "fits": [
                    {
                        "lambda": ridge_lambda,
                        **_optimizer_payload(bundle.fits[ridge_lambda]),
                    }
                    for ridge_lambda in m1.LAMBDAS_DESCENDING
                ],
            }
        )
    zero_refits: list[Mapping[str, Any]] = []
    for group in m1.GROUP_ORDER:
        bundle = cross_fit.outer_refits[group]
        ridge_lambda = next(iter(bundle.fits))
        zero_refits.append(
            {
                "refit": bundle.key,
                "held_out_group": group,
                "initialization": "all_261_parameters_zero",
                "lambda": ridge_lambda,
                "checkpoint_reused": bundle.checkpoint_reused,
                **_optimizer_payload(bundle.fits[ridge_lambda]),
            }
        )
    full_lambda = full.selected_lambda
    zero_refits.append(
        {
            "refit": "full_refit",
            "initialization": "all_261_parameters_zero",
            "lambda": full_lambda,
            "checkpoint_reused": full.full_refit.checkpoint_reused,
            **_optimizer_payload(full.full_refit.fits[full_lambda]),
        }
    )
    return {
        "status": "pass",
        "stage": "v3_M1_development_real_fit_solver_telemetry_v1",
        "warm_training_paths": warm_paths,
        "zero_start_refits": zero_refits,
        "unique_unordered_leave_two_warm_paths": 66,
        "leave_one_full_tuning_warm_paths": 12,
        "zero_start_refits_count": 13,
        "executed_conservative_path_equivalents_with_exact_reuse": 91,
        "frozen_uncached_budget_basis_path_equivalents": 157,
        "budget_pass_uses_uncached_157_basis": True,
        "M0_prediction_or_fit_artifact_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
    }


def _full_refit_payload(full: FullFitResult) -> Mapping[str, Any]:
    result = full.full_refit.fits[full.selected_lambda]
    b_matrix, gamma = m1._theta_blocks(result.theta)
    prepared = full.full_refit.prepared
    return {
        "status": "pass",
        "purpose": "full_development_coefficient_artifact_not_performance_estimate",
        "selected_lambda": full.selected_lambda,
        "selection": full.selection,
        "optimizer": {
            key: value
            for key, value in _optimizer_payload(result).items()
            if key != "telemetry"
        },
        "ordered_target_contrast_definition": {
            "ordered_targets": list(m1.PANEL),
            "Helmert_C_16_by_15": m1.TARGET_CONTRASTS.tolist(),
        },
        "ordered_M0_covariates": list(m1.COVARIATE_NAMES),
        "full_development_transforms": {
            "soma_monomial_mean": prepared.design_transform.soma_mean.tolist(),
            "soma_monomial_population_sd": prepared.design_transform.soma_scale.tolist(),
        },
        "ordered_six_geometry_features": list(m1.FEATURE_NAMES),
        "full_development_geometry_RMS_scales": prepared.geometry_scales.tolist(),
        "B_17_by_15": b_matrix.tolist(),
        "gamma_length_6": gamma.tolist(),
        "development_cells": EXPECTED_ROWS,
        "development_groups": EXPECTED_GROUPS,
        "M0_prediction_or_fit_artifact_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
    }


def run_real(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    validate_real_contract(args.contract)
    execution_addendum = execution_policy.validate_addendum(
        args.execution_addendum
    )
    numeric_backend = execution_policy.numeric_backend()
    qualification = require_single_thread_qualification(args.qualification_root)
    require_matching_qualification_backend(qualification, numeric_backend)
    resource_assessment = require_approved_numerical_refresh(
        args.affected_refresh_root, args.resource_amendment, args.resource_assessment,
        numeric_backend,
    )
    execution_identity = {
        **_execution_identity(execution_addendum, numeric_backend),
        "production_resource_assessment": dict(resource_assessment),
    }
    require_actual_k_prerequisite(
        args.actual_k_root,
        membership_opened_by_job_id=PRIOR_MEMBERSHIP_OPENING_JOB_ID,
    )
    output_root, checkpoint_root, resumed, completed_summary = _prepare_output_root(
        args.output_root, execution_identity
    )
    if completed_summary is not None:
        _write_checkpoint_manifest(
            output_root,
            execution_identity=execution_identity,
            phase="complete",
            realized_membership_opened=True,
            completed_warm_paths=78,
            completed_zero_refits=13,
        )
        return completed_summary
    _write_checkpoint_manifest(
        output_root,
        execution_identity=execution_identity,
        phase="prerequisites_passed_before_membership",
        realized_membership_opened=False,
        completed_warm_paths=0,
        completed_zero_refits=0,
    )
    membership_access_started = False
    try:
        target_blind_cells, pairs = m1.load_target_free_real_shape(
            args.interface_root, args.geometry_root
        )
        expected_ids = {cell.identifier for cell in target_blind_cells}
        membership_access_started = True
        _write_checkpoint_manifest(
            output_root,
            execution_identity=execution_identity,
            phase="membership_access_started",
            realized_membership_opened=True,
            completed_warm_paths=0,
            completed_zero_refits=0,
        )
        membership, counts = _read_membership_inputs(args.target_root, expected_ids)
        library = m1.CandidateLibrary()
        cells = _align_membership(target_blind_cells, membership, library)
        del membership
        geometry = m1.GeometryEngine(library, pairs)
        moments = tuple(geometry.uniform_moments(cell) for cell in cells)
        _write_checkpoint_manifest(
            output_root,
            execution_identity=execution_identity,
            phase="membership_loaded_outer_tuning",
            realized_membership_opened=True,
            completed_warm_paths=0,
            completed_zero_refits=0,
        )

        outer_selection = select_outer_lambdas(
            cells, moments, library, geometry, checkpoint_root
        )
        _write_checkpoint_manifest(
            output_root,
            execution_identity=execution_identity,
            phase="outer_tuning_complete",
            realized_membership_opened=True,
            completed_warm_paths=len(outer_selection.warm_cache),
            completed_zero_refits=0,
        )
        if outer_selection.boundary_count >= 6:
            review = {
                "status": "stop_for_contract_review",
                "reason": "ridge_grid_boundary_selected_in_at_least_6_outer_folds",
                "stage": PRODUCTION_STAGE,
                "outer_boundary_fold_count": outer_selection.boundary_count,
                "outer_folds": list(outer_selection.details),
                "development_cells": EXPECTED_ROWS,
                "development_groups": EXPECTED_GROUPS,
                "actual_K_gate_job_id": ACTUAL_K_JOB_ID,
                "execution_identity": dict(execution_identity),
                "numeric_backend": dict(numeric_backend),
                "realized_development_membership_opened": True,
                "M0_prediction_or_fit_artifact_opened": False,
                "final_or_unassigned_or_SWC_opened": False,
            }
            _atomic_json(output_root / SUMMARY_FILENAME, review)
            raise BoundaryReview(
                f"ridge-grid boundary selected in {outer_selection.boundary_count} outer folds"
            )

        cross_fit = refit_outer_and_predict(
            cells,
            moments,
            library,
            geometry,
            checkpoint_root,
            outer_selection,
        )
        _write_predictions(output_root, cells, cross_fit.prediction)
        _write_checkpoint_manifest(
            output_root,
            execution_identity=execution_identity,
            phase="outer_predictions_complete_full_tuning",
            realized_membership_opened=True,
            completed_warm_paths=len(cross_fit.warm_cache),
            completed_zero_refits=len(cross_fit.outer_refits),
        )

        full = tune_and_refit_full_development(
            cells,
            moments,
            library,
            geometry,
            checkpoint_root,
            cross_fit.warm_cache,
        )
        score = score_and_calibration(cells, library, cross_fit.prediction)
        _atomic_json(output_root / FULL_REFIT_FILENAME, _full_refit_payload(full))
        _atomic_json(output_root / TELEMETRY_FILENAME, _solver_telemetry(cross_fit, full))
        summary = {
            "status": "pass",
            "stage": PRODUCTION_STAGE,
            "analysis_label": "exploratory",
            "estimand": (
                "mapped_positive_terminal_observation_pattern_conditional_on_recorded_K"
            ),
            "development_cells": len(cells),
            "development_groups": len({cell.group for cell in cells}),
            "target_panel_size": m1.PANEL_SIZE,
            "actual_K_gate_job_id": ACTUAL_K_JOB_ID,
            "execution_identity": dict(execution_identity),
            "numeric_backend": dict(numeric_backend),
            "target_fields_opened": (
                "provider_neuron_id_fMOST_brain_id_K_and_detected16_only"
            ),
            "target_projection_rows": counts.total_rows,
            "modeled_interface_rows": counts.matched_rows,
            "unmatched_target_projection_rows": counts.extra_rows,
            "outer_folds": list(cross_fit.outer_details),
            "outer_boundary_fold_count": cross_fit.boundary_count,
            "full_development_selected_lambda": full.selected_lambda,
            "full_development_selection": full.selection,
            "scores_and_calibration": score,
            "work_accounting": {
                "uncached_production_path_equivalents_budget_basis": 157,
                "exact_training_set_reuse_path_equivalents_executed": 91,
                "budget_pass_remains_based_on_uncached_157": True,
            },
            "required_artifacts": list(REQUIRED_OUTPUTS),
            "checkpoint_resume_used": resumed,
            "elapsed_wall_seconds": time.perf_counter() - started,
            "zero_semantics": (
                "no_mapped_positive_terminal_observation_recorded_not_absence"
            ),
            "M0_prediction_or_fit_artifact_opened": False,
            "SWC_opened": False,
            "final_or_unassigned_opened": False,
            "full_refit_used_as_performance_estimate": False,
        }
        _atomic_json(output_root / SUMMARY_FILENAME, summary)
        _write_checkpoint_manifest(
            output_root,
            execution_identity=execution_identity,
            phase="complete",
            realized_membership_opened=True,
            completed_warm_paths=len(full.warm_cache),
            completed_zero_refits=len(cross_fit.outer_refits) + 1,
        )
        return summary
    except BoundaryReview:
        raise
    except Exception as error:
        failure_path = output_root / SUMMARY_FILENAME
        _atomic_json(
            failure_path,
            {
                "status": "fail",
                "stage": PRODUCTION_STAGE,
                "failure_class": type(error).__name__,
                "failure_message": str(error),
                "failure_code": "real_M1_failed_closed",
                "SLURM_job_id": os.environ["SLURM_JOB_ID"],
                "actual_K_gate_job_id": ACTUAL_K_JOB_ID,
                "execution_identity": dict(execution_identity),
                "numeric_backend": dict(numeric_backend),
                "production_fit_kind": getattr(
                    error, "production_fit_kind", None
                ),
                "production_fit_key": getattr(
                    error, "production_fit_key", None
                ),
                "production_fit_lambda": getattr(
                    error, "production_fit_lambda", None
                ),
                "optimizer_scalar_context": (
                    _jsonable(error.scalar_context)
                    if isinstance(getattr(error, "scalar_context", None), Mapping)
                    else None
                ),
                "realized_development_membership_opened": (
                    membership_access_started
                ),
                "M0_prediction_or_fit_artifact_opened": False,
                "final_or_unassigned_or_SWC_opened": False,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--interface-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--actual-k-root", type=Path, required=True)
    parser.add_argument("--execution-addendum", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--affected-refresh-root", type=Path, required=True)
    parser.add_argument("--resource-amendment", type=Path, required=True)
    parser.add_argument("--resource-assessment", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_real(args)
    except BoundaryReview:
        print("ERROR: real M1 stopped for frozen boundary review", file=sys.stderr)
        return 3
    except Exception as error:
        print(f"ERROR: real M1 failed closed ({type(error).__name__})", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "stage": result["stage"],
                "development_cells": result["development_cells"],
                "development_groups": result["development_groups"],
                "outer_boundary_fold_count": result["outer_boundary_fold_count"],
                "M0_prediction_or_fit_artifact_opened": False,
                "final_or_unassigned_opened": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
