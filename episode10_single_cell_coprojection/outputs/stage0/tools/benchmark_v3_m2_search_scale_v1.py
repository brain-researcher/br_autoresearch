#!/usr/bin/env python3
"""Sealed, membership-blind scale benchmark for the EP10 v3 M2 search.

This program is scaffolding for a *fresh synthetic* benchmark.  It may read
only the frozen target-blind shape plus the four-field actual-K projection
already authorized by the M2 actual-K gate.  It never reads an existing M0
fit, M0 telemetry, a realized positive-set identity, final/unassigned data, or
SWCs.  Reports contain aggregate timings and resource projections only.

Every measured unit must be exposed by the fixed local hooks.  A missing unit
produces ``blocked_missing_execution_units``; it is never imputed or treated
as a pass.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import stat
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
STAGE0_DIR = TOOLS_DIR.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

EXPECTED_ROWS = 194
EXPECTED_GROUPS = 12
EXPECTED_FOLDS = 12
EXPECTED_PANEL_SIZE = 16
EXPECTED_PAIR_DIMENSION = 120
EXPECTED_RESIDUAL_RANK = 104
EXPECTED_NULL_REPLAYS = 99
EXPECTED_MAXIMUM_VALID_TRIALS = 80
EXPECTED_MAXIMUM_CONCURRENT_REPLAYS = 32
EXPECTED_REQUESTED_CPUS_PER_TASK = 1
# Sherlock raises an 8 GiB / one-thread request to two physically allocated
# CPUs.  This was observed prospectively on jobs 45862377 and 45870818:
# ReqTRES cpu=1, but ReqCPUS/AllocCPUS=2 and CPUTimeRAW=2*Elapsed.  Numerical
# threads remain one; resource ceilings conservatively count both allocations.
EXPECTED_ALLOCATED_CPUS_PER_TASK = 2
EXPECTED_MEMORY_GIB_PER_TASK = 8.0
TIMING_SAFETY_FACTOR = 2.0
SYNTHETIC_SEED = 2026092901

OUTPUT_PREFIX = "v3_m2_search_scale_benchmark_v1"
SUMMARY_FILENAME = "v3_m2_search_scale_benchmark_summary.json"
CONTRACT_FILENAME = "v3_m2_search_contract.yaml"
ADDENDUM_FILENAME = "v3_m2_numerical_falsifier_addendum_v1.yaml"
HYBRID_CLARIFICATION_FILENAME = "v3_m2_numerical_falsifier_clarification_v2.yaml"
HYBRID_RANK_CLARIFICATION_FILENAME = (
    "v3_m2_numerical_falsifier_clarification_v3.yaml"
)
INVENTORY_FILENAME = "v3_m2_frozen_cell_inventory.tsv"
CORE_REPORT_FILENAME = "v3_m2_core_contract_qualification.json"
ACTUAL_K_REPORT_FILENAME = "v3_m2_actual_k_gate_summary.json"
ADEQUACY_REPORT_FILENAME = "v3_m2_m1_adequacy.json"
OPTIONAL_REPORT_FILENAME = "v3_m2_optional_family_qualification.json"

EXECUTOR_MODULE_NAME = "v3_m2_trial_executor_v1"
OPTIONAL_MODULE_NAME = "fit_v3_m2_optional"
BRIDGE_MODULE_NAME = "v3_m2_scale_benchmark_units_v1"
BENCHMARK_HOOK_NAME = "benchmark_scale_units"

RESOURCE_CEILINGS = {
    "cpu_core_hours": 1800.0,
    "gpu_hours": 0.0,
    "wall_clock_hours": 144.0,
    "parallel_cores": 64,
    "aggregate_memory_GiB": 512.0,
    "scratch_GiB": 1536.0,
}

# One measured unit is one complete operation with the exact scope named here.
# In particular, every trial unit includes all 12 LOGO folds and the selected
# refit unit includes both contractually required full-development repeats.
UNIT_SYNTHETIC = "synthetic_membership_generation"
UNIT_M0 = "fresh_M0_replay_refit"
UNIT_M1 = "fresh_M1_replay_refit"
UNIT_SETUP = "shared_12_fold_transform_and_offset_cache_setup"
UNIT_REFERENCE = "reference_12_fold_trial"
UNIT_CORE = "core_12_fold_trial"
UNIT_LOW_RANK = "low_rank_12_fold_trial"
UNIT_HYBRID = "hybrid_12_fold_trial"
UNIT_DIAGNOSTIC = "optional_trigger_diagnostic"
UNIT_REFIT = "selected_full_development_refit_pair"
UNIT_REPORT = "controller_selection_and_aggregate_report"

EXECUTOR_UNITS = (
    UNIT_SYNTHETIC,
    UNIT_M0,
    UNIT_M1,
    UNIT_SETUP,
    UNIT_REFERENCE,
    UNIT_CORE,
    UNIT_DIAGNOSTIC,
    UNIT_REFIT,
    UNIT_REPORT,
)
OPTIONAL_UNITS = (UNIT_LOW_RANK, UNIT_HYBRID)
REQUIRED_UNITS = EXECUTOR_UNITS + OPTIONAL_UNITS
MEASUREMENT_FIELDS = (
    "wall_seconds",
    "cpu_seconds",
    "peak_rss_gib",
    "scratch_gib",
)

PROHIBITED_OUTPUT_KEYS = {
    "cell_key",
    "provider_neuron_id",
    "identifier",
    "K_i_obs",
    "k_i_obs",
    "observed_index",
    "candidate_index",
    "membership",
    "membership_matrix",
    "realized_positive_set",
    "realized_set",
    "candidate_probabilities",
    "fold_rows",
}


class BenchmarkError(RuntimeError):
    """The sealed scale benchmark failed closed."""


@dataclass(frozen=True)
class SyntheticOutcome:
    """In-memory synthetic outcome; only ``witness`` may enter a report."""

    cells: tuple[Any, ...]
    membership_matrix: np.ndarray
    witness: Mapping[str, Any]


@dataclass(frozen=True)
class BenchmarkContext:
    """In-memory arguments supplied to fixed local benchmark hooks."""

    cells: tuple[Any, ...]
    membership_matrix: np.ndarray
    pair_geometry: Any
    synthetic_seed: int
    synthetic_generation_timing: Any
    scale_benchmark_session: object
    expected_folds: int = EXPECTED_FOLDS
    expected_residual_rank: int = EXPECTED_RESIDUAL_RANK


def _open_regular(path: Path, mode: str = "r") -> Any:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise BenchmarkError(f"required input is not a regular file: {path.name}")
    return os.fdopen(
        descriptor,
        mode,
        encoding=None if "b" in mode else "utf-8",
    )


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        with _open_regular(path) as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise BenchmarkError(f"{label} root is not an object")
    return value


def _exact_stage0_file(path: Path, basename: str, label: str) -> Path:
    expected = STAGE0_DIR / basename
    absolute = Path(os.path.abspath(path))
    try:
        resolved = absolute.resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
    except OSError as error:
        raise BenchmarkError(f"{label} is unavailable") from error
    if (
        absolute.name != basename
        or absolute.is_symlink()
        or resolved != expected_resolved
        or not absolute.is_file()
    ):
        raise BenchmarkError(f"{label} must be the exact frozen stage0 file")
    return absolute


def _scratch_episode_root() -> Path:
    scratch_value = os.environ.get("SCRATCH")
    if not scratch_value:
        raise BenchmarkError("SCRATCH is unset")
    try:
        scratch = Path(scratch_value).resolve(strict=True)
        episode = (
            scratch / "br_autoresearch" / "episode10_single_cell_coprojection"
        ).resolve(strict=True)
    except OSError as error:
        raise BenchmarkError("exact EP10 Scratch root is unavailable") from error
    if not episode.is_dir() or episode.is_symlink():
        raise BenchmarkError("exact EP10 Scratch root is not a real directory")
    return episode


def _sealed_report_path(
    path: Path,
    root_prefix: str,
    filename: str,
    label: str,
) -> tuple[Path, str]:
    absolute = Path(os.path.abspath(path))
    match = re.fullmatch(rf"{re.escape(root_prefix)}_job_([0-9]+)", absolute.parent.name)
    if match is None:
        raise BenchmarkError(f"{label} job-root name changed")
    expected = _scratch_episode_root() / absolute.parent.name / filename
    try:
        parent_resolved = absolute.parent.resolve(strict=True)
        expected_parent = expected.parent.resolve(strict=True)
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise BenchmarkError(f"{label} path is unavailable") from error
    if (
        absolute != expected
        or parent_resolved != expected_parent
        or resolved != absolute
        or absolute.is_symlink()
        or not absolute.is_file()
    ):
        raise BenchmarkError(f"{label} is not the exact unsymlinked Scratch report")
    return absolute, match.group(1)


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise BenchmarkError(f"{label} is not numeric") from error
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        raise BenchmarkError(f"{label} is not a permitted finite measurement")
    return result


def validate_frozen_contracts(
    contract_argument: Path,
    addendum_argument: Path,
    hybrid_clarification_argument: Path,
    hybrid_rank_clarification_argument: Path,
) -> Mapping[str, Any]:
    """Validate the exact trial, replay, request, and ceiling clauses."""
    contract_path = _exact_stage0_file(
        contract_argument, CONTRACT_FILENAME, "M2 search contract"
    )
    addendum_path = _exact_stage0_file(
        addendum_argument, ADDENDUM_FILENAME, "M2 numerical addendum"
    )
    hybrid_clarification_path = _exact_stage0_file(
        hybrid_clarification_argument,
        HYBRID_CLARIFICATION_FILENAME,
        "M2 hybrid coordinate clarification",
    )
    hybrid_rank_clarification_path = _exact_stage0_file(
        hybrid_rank_clarification_argument,
        HYBRID_RANK_CLARIFICATION_FILENAME,
        "M2 hybrid rank-initialization clarification",
    )
    gate = importlib.import_module("gate_v3_m2_actual_k")
    gate.validate_gate_contract(contract_path, addendum_path)
    contract = gate._read_yaml_mapping(contract_path, "M2 search contract")
    addendum = gate._read_yaml_mapping(addendum_path, "M2 numerical addendum")
    clarification = gate._read_yaml_mapping(
        hybrid_clarification_path, "M2 hybrid coordinate clarification"
    )
    rank_clarification = gate._read_yaml_mapping(
        hybrid_rank_clarification_path,
        "M2 hybrid rank-initialization clarification",
    )

    controller = contract.get("controller", {})
    schedule = controller.get("minimum_schedule", {})
    anchors = schedule.get("trials_1_through_16", [])
    extensions = schedule.get("triggered_extension_anchors_insert_before_postcoverage", {})
    low_rank = extensions.get("low_rank_if_triggered", {})
    hybrid = extensions.get("sparse_plus_low_rank_if_triggered", {})
    full_null = contract.get("full_controller_null", {})
    ceilings = contract.get("resource_ceilings", {})
    requests = addendum.get("resource_starting_requests", {})
    null_request = requests.get("null_array_initial", {})
    completion = addendum.get("access_and_completion_boundary", {})
    clarification_scope = clarification.get("authority_and_scope", {})
    hybrid_coordinates = clarification.get(
        "sparse_plus_low_rank_coordinate_clarification", {}
    )
    effective = hybrid_coordinates.get("effective_coefficient", {})
    clarification_objective = hybrid_coordinates.get("objective", {})
    sparse_execution = clarification.get("sparse_block_execution", {})
    degeneracy = clarification.get(
        "low_rank_initialization_numerical_degeneracy", {}
    )
    joint_execution = clarification.get("joint_execution_and_gates", {})
    clarification_access = clarification.get("access_boundary", {})
    rank_scope = rank_clarification.get("authority_and_scope", {})
    rank_definitions = rank_clarification.get("frozen_definitions", {})
    rank_initialization = rank_clarification.get(
        "hybrid_low_rank_initialization", {}
    )
    rank_refit = rank_clarification.get("selected_full_development_refit", {})
    rank_access = rank_clarification.get("access_boundary", {})

    expected_ceilings = dict(RESOURCE_CEILINGS)
    observed_ceilings = {
        key: float(ceilings.get(key, math.nan)) for key in expected_ceilings
    }
    if (
        contract.get("contract_id") != "v3_m2_search_v1"
        or addendum.get("addendum_id") != "v3_m2_numerical_falsifier_v1"
        or contract.get("status")
        != "frozen_prospectively_before_any_M2_outcome_access"
        or addendum.get("status")
        != "frozen_prospectively_before_any_M2_outcome_access"
        or int(controller.get("valid_trial_cap", -1))
        != EXPECTED_MAXIMUM_VALID_TRIALS
        or len(anchors) != 16
        or [int(item.get("trial", -1)) for item in anchors] != list(range(1, 17))
        or int(low_rank.get("count", -1)) != 8
        or int(hybrid.get("count", -1)) != 4
        or int(full_null.get("replicates", -1)) != EXPECTED_NULL_REPLAYS
        or full_null.get("RNG") != "PCG64"
        or full_null.get("generator_seeds")
        != "2026093001_through_2026093099_inclusive"
        or ceilings.get("apply_to_controller_falsifiers_refits_and_all_99_null_replays")
        is not True
        or observed_ceilings != expected_ceilings
        or int(null_request.get("tasks", -1)) != EXPECTED_NULL_REPLAYS
        or int(null_request.get("maximum_concurrent_tasks", -1))
        != EXPECTED_MAXIMUM_CONCURRENT_REPLAYS
        or int(null_request.get("cpus_per_task", -1))
        != EXPECTED_REQUESTED_CPUS_PER_TASK
        or float(null_request.get("memory_GiB_per_task", math.nan))
        != EXPECTED_MEMORY_GIB_PER_TASK
        or null_request.get("time_limit_per_task") != "24:00:00"
        or requests.get("benchmark_before_observed_or_null_scale_up") != "required"
        or requests.get("benchmark_must_project_below_all_parent_resource_ceilings")
        is not True
        or requests.get("GPU") != "none"
        or "benchmark_projects_observed_and_99_null_work_below_all_resource_ceilings"
        not in completion.get("first_real_M2_outcome_access_requires", [])
        or clarification.get("clarification_id")
        != "v3_m2_numerical_falsifier_clarification_v2"
        or clarification.get("status")
        != "frozen_prospectively_before_any_M2_outcome_access"
        or clarification_scope.get("parent_contract")
        != "outputs/stage0/v3_m2_search_contract.yaml"
        or clarification_scope.get("numerical_addendum")
        != "outputs/stage0/v3_m2_numerical_falsifier_addendum_v1.yaml"
        or clarification_scope.get("candidate_pair_effect_or_M2_score_used") is not False
        or clarification_scope.get("real_M2_outcome_access_before_freeze") is not False
        or clarification_scope.get("final_unassigned_or_SWC_access") != "forbidden"
        or clarification_scope.get("M0_full_refit_or_telemetry_access") != "forbidden"
        or len(clarification_scope.get("supersedes_only", [])) != 2
        or int(hybrid_coordinates.get("named_edge_set", {}).get("maximum_edges", -1))
        != 16
        or effective.get("formula")
        != "beta = P * (offdiag(U diag(d) U_transpose) + E_C a)"
        or clarification_objective.get("low_rank_penalty")
        != "rho_low_times_lambda_reference_times_sum_d_squared_over_2"
        or clarification_objective.get("sparse_penalty")
        != "rho_sparse_times_core_lasso_lambda_max_times_sum_absolute_a"
        or sparse_execution.get("algorithm")
        != "deterministic_monotone_proximal_gradient_with_backtracking_on_a"
        or sparse_execution.get("v1_sparse_step_disposition")
        != "explicitly_superseded_by_this_clarification"
        or sparse_execution.get("initialization") != "exact_zero"
        or sparse_execution.get("L_state")
        != "initialize_once_then_carry_monotonically_across_sparse_blocks"
        or int(sparse_execution.get("maximum_line_trials_per_sparse_block", -1))
        != 80
        or degeneracy.get("scope")
        != "deterministic_null_gradient_initialization_only"
        or degeneracy.get("interval_radius")
        != "per_eigenpair_residual_plus_rounding_guard"
        or "math.fsum" not in str(degeneracy.get("cluster_initial_eigenvalue", ""))
        or degeneracy.get("rank_cut_inside_cluster")
        != "retain_the_first_required_canonical_cluster_vectors"
        or set(degeneracy.get("certificates", {}))
        != {
            "orthogonality_maximum_absolute_entry",
            "projector_reconstruction_maximum_absolute_entry",
            "common_value_residual_l2_maximum",
        }
        or joint_execution.get("block_order") != ["low_rank", "sparse"]
        or int(joint_execution.get("maximum_sweeps", -1)) != 20000
        or float(joint_execution.get("joint_stationarity_infinity_tolerance", math.nan))
        != 1.0e-6
        or float(joint_execution.get("relative_objective_tolerance", math.nan))
        != 1.0e-9
        or joint_execution.get("convergence_requires_both") is not True
        or "full-sweep objective change"
        not in str(joint_execution.get("relative_objective_definition", ""))
        or clarification_access.get("may_implement_and_qualify_target_free_now") is not True
        or clarification_access.get("first_real_M2_outcome_access")
        != "remains_governed_by_every_v1_preaccess_gate"
        or rank_clarification.get("clarification_id")
        != "v3_m2_numerical_falsifier_clarification_v3"
        or rank_clarification.get("status")
        != "frozen_prospectively_before_any_M2_outcome_access"
        or rank_scope.get("parent_contract")
        != "outputs/stage0/v3_m2_search_contract.yaml"
        or rank_scope.get("numerical_addendum")
        != "outputs/stage0/v3_m2_numerical_falsifier_addendum_v1.yaml"
        or rank_scope.get("prior_clarification")
        != "outputs/stage0/v3_m2_numerical_falsifier_clarification_v2.yaml"
        or rank_scope.get(
            "candidate_pair_effect_or_M2_score_used_to_freeze_this_clarification"
        )
        is not False
        or rank_scope.get("real_M2_outcome_access_before_freeze") is not False
        or rank_scope.get("final_unassigned_or_SWC_access") != "forbidden"
        or rank_scope.get("M0_full_refit_or_telemetry_access") != "forbidden"
        or rank_definitions.get("ordered_rank_lattice") != [1, 2, 3, 4, 6, 8]
        or rank_definitions.get("mandatory_low_rank_coverage_inventory", {}).get(
            "completion_requirement"
        )
        != "all_eight_exact_nodes_are_valid_and_score_bearing"
        or rank_initialization.get("sparse_block")
        != "exact_zero_at_every_cold_hybrid_fit"
        or rank_refit.get("selection_is_not_reopened") is not True
        or rank_access.get("may_implement_and_qualify_target_free_now") is not True
        or rank_access.get("first_real_M2_outcome_access")
        != "remains_governed_by_every_v1_v2_and_parent_preaccess_gate"
    ):
        raise BenchmarkError("frozen M2 scale-benchmark clauses changed")
    return {
        "status": "pass",
        "contract_id": "v3_m2_search_v1",
        "addendum_id": "v3_m2_numerical_falsifier_v1",
        "hybrid_clarification_id": "v3_m2_numerical_falsifier_clarification_v2",
        "hybrid_rank_initialization_clarification_id": (
            "v3_m2_numerical_falsifier_clarification_v3"
        ),
        "core_coverage_trials": 16,
        "worst_triggered_coverage_trials": 28,
        "branch_minimum_trials": 52,
        "maximum_valid_trials": 80,
        "null_replays": 99,
        "resource_ceilings": expected_ceilings,
    }


def _validate_core_payload(payload: Mapping[str, Any], job_id: str) -> None:
    execution = payload.get("execution", {})
    inventory = payload.get("gate_inventory", {})
    if not (
        payload.get("status") == "pass"
        and payload.get("overall_gate_status") == "pass"
        and payload.get("mode") == "full"
        and payload.get("stage")
        == "v3_M2_target_free_frozen_core_contract_qualification"
        and str(execution.get("slurm_job_id")) == job_id
        and inventory.get("all_required_gates_present_and_passed") is True
        and int(inventory.get("required_gate_count", -1)) == 8
        and payload.get("real_target_membership_or_M2_effects_opened") is False
        and payload.get("actual_K_opened") is False
        and payload.get("M0_full_refit_or_telemetry_opened") is False
        and payload.get("final_or_unassigned_or_SWC_opened") is False
        and payload.get("controller_replay_qualification_included") is False
    ):
        raise BenchmarkError("core prerequisite is not a full passed sealed report")


def _validate_actual_k_payload(payload: Mapping[str, Any], job_id: str) -> str:
    rank_gate = payload.get("rank_gate", {})
    core_job_id = str(payload.get("qualified_core_job_id", ""))
    if not (
        payload.get("status") == "pass"
        and payload.get("stage") == "v3_M2_actual_K_only_residual_rank_gate_v1"
        and str(payload.get("SLURM_job_id")) == job_id
        and int(payload.get("rows", -1)) == EXPECTED_ROWS
        and int(payload.get("groups", -1)) == EXPECTED_GROUPS
        and payload.get("actual_K_opened") is True
        and payload.get("cell_ids_groups_segment_flags_or_K_values_emitted") is False
        and payload.get("realized_positive_set_membership_opened") is False
        and payload.get("candidate_scores_or_M2_fit_computed") is False
        and payload.get("M0_full_refit_or_telemetry_opened") is False
        and payload.get("SWC_opened") is False
        and payload.get("final_or_unassigned_opened") is False
        and rank_gate.get("status") == "pass"
        and int(rank_gate.get("training_folds_evaluated", -1)) == EXPECTED_FOLDS
        and int(rank_gate.get("training_folds_passed", -1)) == EXPECTED_FOLDS
        and int(rank_gate.get("expected_rank", -1)) == EXPECTED_RESIDUAL_RANK
        and rank_gate.get("computed_rank_at_tau_over_10_unique")
        == [EXPECTED_RESIDUAL_RANK]
        and rank_gate.get("computed_rank_at_tau_unique") == [EXPECTED_RESIDUAL_RANK]
        and rank_gate.get("computed_rank_at_ten_tau_unique")
        == [EXPECTED_RESIDUAL_RANK]
        and rank_gate.get("rank_stable_in_every_fold") is True
        and rank_gate.get("rank_computed_not_hard_coded") is True
        and rank_gate.get("fold_level_rows_emitted") is False
        and rank_gate.get("cell_ids_groups_or_K_values_emitted") is False
        and core_job_id.isdigit()
    ):
        raise BenchmarkError("actual-K prerequisite is not a passed sealed rank report")
    return core_job_id


def _validate_adequacy_payload(payload: Mapping[str, Any]) -> None:
    if not (
        payload.get("status") == "pass"
        and payload.get("gate") == "v3_M2_preaccess_M1_adequacy"
        and int(payload.get("development_cells", -1)) == EXPECTED_ROWS
        and int(payload.get("development_groups", -1)) == EXPECTED_GROUPS
        and int(payload.get("target_panel_size", -1)) == EXPECTED_PANEL_SIZE
        and payload.get("M1_summary_and_outer_predictions_opened") is True
        and payload.get("real_M2_outcomes_opened") is False
        and payload.get("M0_full_refit_or_telemetry_opened") is False
        and payload.get("SWC_opened") is False
        and payload.get("final_or_unassigned_opened") is False
    ):
        raise BenchmarkError("M1 adequacy prerequisite is not passed and sealed")
    for key in (
        "equal_group_M1_minus_uniform_log_score_gain_bits_per_cell",
        "equal_group_marginal_Brier_M1",
        "equal_group_marginal_Brier_fixed_K_uniform",
        "maximum_absolute_marginal_sum_K_error",
    ):
        _finite(payload.get(key), f"M1 adequacy {key}")


def _validate_optional_payload(payload: Mapping[str, Any], job_id: str) -> None:
    execution = payload.get("execution", {})
    inventory = payload.get("gate_inventory", {})
    required = {
        "optional_trigger_diagnostic",
        "low_rank_population_recovery",
        "sparse_plus_low_rank_population_recovery",
    }
    gates = payload.get("gates", {})
    if not (
        payload.get("status") == "pass"
        and payload.get("overall_gate_status") == "pass"
        and payload.get("stage")
        == "v3_M2_target_free_optional_family_qualification_v1"
        and str(execution.get("slurm_job_id")) == job_id
        and inventory.get("all_required_gates_present_and_passed") is True
        and int(inventory.get("required_gate_count", -1)) == len(required)
        and set(inventory.get("required", ())) == required
        and set(inventory.get("passing", ())) == required
        and isinstance(gates, Mapping)
        and set(gates) == required
        and all(
            isinstance(gates[name], Mapping)
            and gates[name].get("status") == "pass"
            for name in required
        )
        and payload.get("real_target_membership_or_M2_effects_opened") is False
        and payload.get("actual_K_opened") is False
        and payload.get("M0_full_refit_or_telemetry_opened") is False
        and payload.get("final_or_unassigned_or_SWC_opened") is False
    ):
        raise BenchmarkError(
            "optional-family prerequisite is not a full passed sealed report"
        )


def validate_prerequisite_reports(
    core_report: Path,
    actual_k_report: Path,
    adequacy_report: Path,
    optional_report: Path,
) -> Mapping[str, Any]:
    core_path, core_job_id = _sealed_report_path(
        core_report,
        "v3_m2_core_contract_qualification",
        CORE_REPORT_FILENAME,
        "core prerequisite",
    )
    actual_path, actual_job_id = _sealed_report_path(
        actual_k_report,
        "v3_m2_actual_k_gate_v1",
        ACTUAL_K_REPORT_FILENAME,
        "actual-K prerequisite",
    )
    adequacy_path, adequacy_job_id = _sealed_report_path(
        adequacy_report,
        "v3_m2_m1_adequacy_gate",
        ADEQUACY_REPORT_FILENAME,
        "M1 adequacy prerequisite",
    )
    optional_path, optional_job_id = _sealed_report_path(
        optional_report,
        "v3_m2_optional_family_qualification",
        OPTIONAL_REPORT_FILENAME,
        "optional-family prerequisite",
    )
    core_payload = _read_json(core_path, "core prerequisite")
    actual_payload = _read_json(actual_path, "actual-K prerequisite")
    adequacy_payload = _read_json(adequacy_path, "M1 adequacy prerequisite")
    optional_payload = _read_json(optional_path, "optional-family prerequisite")
    _validate_core_payload(core_payload, core_job_id)
    linked_core_job_id = _validate_actual_k_payload(actual_payload, actual_job_id)
    _validate_adequacy_payload(adequacy_payload)
    _validate_optional_payload(optional_payload, optional_job_id)
    if linked_core_job_id != core_job_id:
        raise BenchmarkError("actual-K prerequisite does not link the supplied core report")
    return {
        "status": "pass",
        "core_job_id": core_job_id,
        "actual_K_job_id": actual_job_id,
        "M1_adequacy_job_id": adequacy_job_id,
        "optional_family_job_id": optional_job_id,
        "all_prerequisites_passed_and_cross_linked": True,
    }


def load_actual_k_only_shape(
    interface_root: Path,
    geometry_root: Path,
    target_root: Path,
    inventory_argument: Path,
) -> tuple[tuple[Any, ...], Any, Mapping[str, Any]]:
    """Load exact K without ever materializing a real positive-set identity."""
    inventory_path = _exact_stage0_file(
        inventory_argument, INVENTORY_FILENAME, "frozen M2 cell inventory"
    )
    m1 = importlib.import_module("fit_v3_m1")
    actual_gate = importlib.import_module("gate_v3_m2_actual_k")
    sealed_k = importlib.import_module("gate_v3_m1_actual_k")
    inventory = actual_gate._read_inventory(inventory_path)
    target_blind_cells, pair_geometry = m1.load_target_free_real_shape(
        interface_root, geometry_root
    )
    actual_gate._validate_interface_inventory(target_blind_cells, inventory)
    projected, counts = sealed_k._read_target_inputs(target_root, set(inventory))
    if (
        int(counts.total_rows) != int(sealed_k.EXPECTED_TARGET_ROWS)
        or int(counts.matched_rows) != EXPECTED_ROWS
        or int(counts.extra_rows) != int(sealed_k.EXPECTED_EXCLUDED_ROWS)
    ):
        raise BenchmarkError("actual-K projection counts changed")
    cells = sealed_k._align_actual_k(target_blind_cells, projected)
    actual_gate._validate_interface_inventory(cells, inventory)
    if any(
        int(cell.observed_index) != math.comb(EXPECTED_PANEL_SIZE, int(cell.k))
        for cell in cells
    ):
        raise BenchmarkError("actual-K cells do not retain the inert one-past-end seal")
    return cells, pair_geometry, {
        "rows": EXPECTED_ROWS,
        "groups": EXPECTED_GROUPS,
        "actual_K_opened": True,
        "real_membership_opened": False,
        "cell_ids_or_K_values_emitted": False,
    }


def construct_synthetic_memberships(
    cells: Sequence[Any],
    pair_geometry: Any,
    *,
    seed: int = SYNTHETIC_SEED,
    m1_module: ModuleType | Any | None = None,
) -> SyntheticOutcome:
    """Draw one deterministic synthetic candidate at every exact actual K.

    The input ``observed_index`` must be the one-past-end poison installed by
    the actual-K gate.  The output index and 16-vector are generated only from
    the frozen planted M1 distribution and a PCG64/SeedSequence stream.
    """
    m1 = m1_module if m1_module is not None else importlib.import_module("fit_v3_m1")
    if len(cells) != EXPECTED_ROWS:
        raise BenchmarkError("synthetic construction requires exactly 194 cells")
    library = m1.CandidateLibrary()
    geometry = m1.GeometryEngine(library, pair_geometry)
    for cell in cells:
        basis = library.basis(int(cell.k))
        if int(cell.observed_index) != int(basis.count):
            raise BenchmarkError("synthetic construction input is not membership-blind")
    moments = tuple(geometry.uniform_moments(cell) for cell in cells)
    prepared = m1.prepare_data(cells, moments)
    theta = m1.planted_parameters()
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed))))
    uniforms = rng.random(len(cells))
    synthetic_cells: list[Any] = []
    memberships: list[np.ndarray] = []
    for index, (cell, uniform) in enumerate(zip(cells, uniforms, strict=True)):
        basis = library.basis(int(cell.k))
        candidate_index = int(
            m1.sample_candidate_index(
                theta,
                prepared,
                index,
                library,
                geometry,
                float(uniform),
            )
        )
        if not 0 <= candidate_index < int(basis.count):
            raise BenchmarkError("synthetic candidate index is outside exact Omega_i")
        membership = np.asarray(basis.membership[candidate_index], dtype=np.uint8)
        if (
            membership.shape != (EXPECTED_PANEL_SIZE,)
            or not np.all((membership == 0) | (membership == 1))
            or int(np.sum(membership, dtype=np.int64)) != int(cell.k)
        ):
            raise BenchmarkError("synthetic candidate does not preserve exact actual K")
        synthetic_cells.append(
            replace(
                cell,
                observed_index=candidate_index,
                sampling_uniform=float(uniform),
            )
        )
        memberships.append(membership.copy())
    matrix = np.stack(memberships, axis=0)
    matrix.setflags(write=False)
    return SyntheticOutcome(
        cells=tuple(synthetic_cells),
        membership_matrix=matrix,
        witness={
            "status": "pass",
            "generator": "numpy_SeedSequence_PCG64",
            "seed": int(seed),
            "rows": EXPECTED_ROWS,
            "panel_size": EXPECTED_PANEL_SIZE,
            "every_synthetic_membership_has_exact_actual_K": True,
            "input_real_membership_used": False,
            "cell_ids_candidate_indexes_K_values_or_memberships_emitted": False,
        },
    )


def construct_benchmark_context(
    cells: Sequence[Any],
    pair_geometry: Any,
) -> tuple[SyntheticOutcome, BenchmarkContext]:
    """Time the exact constructor and seal its outcome into one context."""

    bridge = _require_local_module(BRIDGE_MODULE_NAME)
    session = bridge.issue_scale_benchmark_session()
    outcome, generation_timing = bridge.time_synthetic_constructor(
        session,
        pair_geometry,
        lambda: construct_synthetic_memberships(cells, pair_geometry),
    )
    context = BenchmarkContext(
        cells=outcome.cells,
        membership_matrix=outcome.membership_matrix,
        pair_geometry=pair_geometry,
        synthetic_seed=SYNTHETIC_SEED,
        synthetic_generation_timing=generation_timing,
        scale_benchmark_session=session,
    )
    bridge.bind_scale_benchmark_context(context)
    return outcome, context


def _require_local_module(name: str) -> ModuleType:
    try:
        module = importlib.import_module(name)
    except Exception as error:
        raise BenchmarkError(f"required local module is unavailable: {name}") from error
    expected = TOOLS_DIR / f"{name}.py"
    module_path = Path(os.path.abspath(str(getattr(module, "__file__", ""))))
    try:
        valid = (
            module_path == expected
            and not module_path.is_symlink()
            and module_path.resolve(strict=True) == expected.resolve(strict=True)
        )
    except OSError:
        valid = False
    if not valid:
        raise BenchmarkError(f"required module did not resolve to the sealed tools file: {name}")
    return module


def discover_timing_hooks() -> tuple[Mapping[str, ModuleType], Mapping[str, Any]]:
    """Import only the two fixed local modules and inventory benchmark hooks."""
    modules: dict[str, ModuleType] = {}
    missing_modules: list[str] = []
    for name in (EXECUTOR_MODULE_NAME, OPTIONAL_MODULE_NAME):
        try:
            modules[name] = _require_local_module(name)
        except BenchmarkError:
            missing_modules.append(name)

    executor = modules.get(EXECUTOR_MODULE_NAME)
    optional = modules.get(OPTIONAL_MODULE_NAME)
    executor_api = (
        "logo_fold_specs",
        "prepare_trial_cache",
        "CoreTrialExecutor",
    )
    optional_api = (
        "fold_low_rank_scales",
        "initialize_low_rank",
        "fit_low_rank",
        "fit_low_rank_synthetic_qualification",
        "refit_low_rank_from_converged",
        "refit_low_rank_from_converged_synthetic_qualification",
        "evaluate_low_rank",
        "low_rank_candidate_distribution",
    )
    hybrid_api = (
        "fit_sparse_plus_low_rank",
        "fit_sparse_plus_low_rank_synthetic_qualification",
        "refit_sparse_plus_low_rank_from_converged",
        "refit_sparse_plus_low_rank_from_converged_synthetic_qualification",
    )
    executor_api_present = bool(executor) and all(
        hasattr(executor, name) for name in executor_api
    )
    optional_low_rank_api_present = bool(optional) and all(
        hasattr(optional, name) for name in optional_api
    )
    executor_hook_present = bool(executor) and callable(
        getattr(executor, BENCHMARK_HOOK_NAME, None)
    )
    optional_hook_present = bool(optional) and callable(
        getattr(optional, BENCHMARK_HOOK_NAME, None)
    )
    missing_units: list[str] = []
    if not executor_hook_present:
        missing_units.extend(EXECUTOR_UNITS)
    if not optional_hook_present:
        missing_units.extend(OPTIONAL_UNITS)
    return modules, {
        "status": "available" if not missing_units else "blocked_missing_execution_units",
        "fixed_module_names": [EXECUTOR_MODULE_NAME, OPTIONAL_MODULE_NAME],
        "missing_modules": missing_modules,
        "executor_core_public_API_present": executor_api_present,
        "optional_low_rank_public_API_present": optional_low_rank_api_present,
        "hybrid_public_API_present": bool(optional)
        and all(hasattr(optional, name) for name in hybrid_api),
        "benchmark_hook_name": BENCHMARK_HOOK_NAME,
        "executor_benchmark_hook_present": executor_hook_present,
        "optional_benchmark_hook_present": optional_hook_present,
        "missing_measured_units": list(dict.fromkeys(missing_units)),
        "no_missing_unit_imputed": True,
    }


def normalize_measured_units(units: Mapping[str, Any]) -> Mapping[str, Mapping[str, float]]:
    """Whitelist and validate aggregate resource measurements."""
    if set(units) != set(REQUIRED_UNITS):
        missing = sorted(set(REQUIRED_UNITS) - set(units))
        extra = sorted(set(units) - set(REQUIRED_UNITS))
        raise BenchmarkError(f"measured-unit inventory changed; missing={missing}, extra={extra}")
    normalized: dict[str, Mapping[str, float]] = {}
    for name in REQUIRED_UNITS:
        item = units[name]
        if not isinstance(item, Mapping) or item.get("status") != "pass":
            raise BenchmarkError(f"measured unit is unavailable or nonpassing: {name}")
        if any(flag is not False for flag in (
            item.get("real_membership_opened", False),
            item.get("existing_M0_artifact_or_telemetry_opened", False),
            item.get("final_or_unassigned_or_SWC_opened", False),
        )):
            raise BenchmarkError(f"measured unit violated a protected input seal: {name}")
        normalized[name] = {
            "wall_seconds": _finite(
                item.get("wall_seconds"), f"{name}.wall_seconds", positive=True
            ),
            "cpu_seconds": _finite(item.get("cpu_seconds"), f"{name}.cpu_seconds"),
            "peak_rss_gib": _finite(
                item.get("peak_rss_gib"), f"{name}.peak_rss_gib", positive=True
            ),
            "scratch_gib": _finite(item.get("scratch_gib"), f"{name}.scratch_gib"),
        }
    return normalized


def collect_measured_units(
    context: BenchmarkContext,
    modules: Mapping[str, ModuleType],
) -> Mapping[str, Mapping[str, float]]:
    """Call the fixed hook on each fixed module; accept no partial inventory."""
    executor = modules.get(EXECUTOR_MODULE_NAME)
    optional = modules.get(OPTIONAL_MODULE_NAME)
    if executor is None or optional is None:
        raise BenchmarkError("timing modules are incomplete")
    executor_hook = getattr(executor, BENCHMARK_HOOK_NAME, None)
    optional_hook = getattr(optional, BENCHMARK_HOOK_NAME, None)
    if not callable(executor_hook) or not callable(optional_hook):
        raise BenchmarkError("one or more required benchmark hooks are unavailable")
    executor_units = executor_hook(context)
    optional_units = optional_hook(context)
    if not isinstance(executor_units, Mapping) or not isinstance(optional_units, Mapping):
        raise BenchmarkError("benchmark hook did not return a measured-unit mapping")
    if set(executor_units) != set(EXECUTOR_UNITS):
        raise BenchmarkError("executor benchmark hook unit inventory changed")
    if set(optional_units) != set(OPTIONAL_UNITS):
        raise BenchmarkError("optional benchmark hook unit inventory changed")
    return normalize_measured_units({**executor_units, **optional_units})


def trial_unit_counts(valid_trial_cap: int) -> Mapping[str, int]:
    """Return the exact worst fully-triggered 52- or 80-trial composition."""
    if int(valid_trial_cap) not in (52, 80):
        raise BenchmarkError("projection supports only the contracted 52 and 80 trials")
    counts = {
        UNIT_REFERENCE: 1,
        UNIT_CORE: 15,
        UNIT_LOW_RANK: 8,
        UNIT_HYBRID: 4,
        "adaptive_worst_trial": 24 + (int(valid_trial_cap) - 52),
        UNIT_DIAGNOSTIC: 1,
        UNIT_REFIT: 1,
        UNIT_REPORT: 1,
    }
    if sum(counts[name] for name in (
        UNIT_REFERENCE,
        UNIT_CORE,
        UNIT_LOW_RANK,
        UNIT_HYBRID,
        "adaptive_worst_trial",
    )) != int(valid_trial_cap):
        raise BenchmarkError("internal controller trial accounting changed")
    return counts


def _metric_controller_total(
    units: Mapping[str, Mapping[str, float]],
    valid_trial_cap: int,
    field: str,
) -> float:
    counts = trial_unit_counts(valid_trial_cap)
    adaptive = max(
        float(units[UNIT_CORE][field]),
        float(units[UNIT_LOW_RANK][field]),
        float(units[UNIT_HYBRID][field]),
    )
    return math.fsum(
        (
            float(units[UNIT_SETUP][field]),
            counts[UNIT_REFERENCE] * float(units[UNIT_REFERENCE][field]),
            counts[UNIT_CORE] * float(units[UNIT_CORE][field]),
            counts[UNIT_LOW_RANK] * float(units[UNIT_LOW_RANK][field]),
            counts[UNIT_HYBRID] * float(units[UNIT_HYBRID][field]),
            counts["adaptive_worst_trial"] * adaptive,
            float(units[UNIT_DIAGNOSTIC][field]),
            float(units[UNIT_REFIT][field]),
            float(units[UNIT_REPORT][field]),
        )
    )


def project_search_resources(
    measured_units: Mapping[str, Any],
    valid_trial_cap: int,
) -> Mapping[str, Any]:
    """Project one observed controller and all 99 complete fresh replays.

    CPU core-hours conservatively use Sherlock's observed two-CPU allocation
    for each requested one-thread/8-GiB task, while the separately reported
    process-CPU projection is diagnostic only.  Calendar wall time is observed
    time plus four worst-case replay waves (ceil(99/32)).  Scratch assumes every
    measured unit's aggregate scratch contribution is retained; no deletion
    credit is taken.
    """
    units = normalize_measured_units(measured_units)
    cap = int(valid_trial_cap)
    counts = trial_unit_counts(cap)
    controller_wall = _metric_controller_total(units, cap, "wall_seconds")
    controller_cpu = _metric_controller_total(units, cap, "cpu_seconds")
    controller_scratch = _metric_controller_total(units, cap, "scratch_gib")
    replay_prefix_wall = math.fsum(
        float(units[name]["wall_seconds"])
        for name in (UNIT_SYNTHETIC, UNIT_M0, UNIT_M1)
    )
    replay_prefix_cpu = math.fsum(
        float(units[name]["cpu_seconds"])
        for name in (UNIT_SYNTHETIC, UNIT_M0, UNIT_M1)
    )
    replay_prefix_scratch = math.fsum(
        float(units[name]["scratch_gib"])
        for name in (UNIT_SYNTHETIC, UNIT_M0, UNIT_M1)
    )
    replay_wall = replay_prefix_wall + controller_wall
    replay_cpu = replay_prefix_cpu + controller_cpu
    replay_scratch = replay_prefix_scratch + controller_scratch
    aggregate_allocated_seconds_unbuffered = (
        controller_wall + EXPECTED_NULL_REPLAYS * replay_wall
    ) * EXPECTED_ALLOCATED_CPUS_PER_TASK
    aggregate_process_cpu_seconds = controller_cpu + EXPECTED_NULL_REPLAYS * replay_cpu
    replay_waves = math.ceil(
        EXPECTED_NULL_REPLAYS / EXPECTED_MAXIMUM_CONCURRENT_REPLAYS
    )
    calendar_seconds_unbuffered = controller_wall + replay_waves * replay_wall
    aggregate_allocated_seconds = (
        TIMING_SAFETY_FACTOR * aggregate_allocated_seconds_unbuffered
    )
    calendar_seconds = TIMING_SAFETY_FACTOR * calendar_seconds_unbuffered
    scratch_gib = controller_scratch + EXPECTED_NULL_REPLAYS * replay_scratch
    observed_peak_rss = max(float(item["peak_rss_gib"]) for item in units.values())
    replay_peak_rss = observed_peak_rss
    memory_per_task = max(EXPECTED_MEMORY_GIB_PER_TASK, replay_peak_rss)
    aggregate_memory = EXPECTED_MAXIMUM_CONCURRENT_REPLAYS * memory_per_task
    parallel_cores = (
        EXPECTED_MAXIMUM_CONCURRENT_REPLAYS
        * EXPECTED_ALLOCATED_CPUS_PER_TASK
    )
    projected = {
        "valid_trial_cap": cap,
        "exact_trial_unit_counts": dict(counts),
        "null_replays": EXPECTED_NULL_REPLAYS,
        "maximum_concurrent_replays": EXPECTED_MAXIMUM_CONCURRENT_REPLAYS,
        "replay_waves": replay_waves,
        "timing_safety_factor": TIMING_SAFETY_FACTOR,
        "observed_controller_wall_hours": controller_wall / 3600.0,
        "one_fresh_replay_wall_hours": replay_wall / 3600.0,
        "observed_plus_99_replays_allocated_cpu_core_hours": (
            aggregate_allocated_seconds / 3600.0
        ),
        "observed_plus_99_replays_process_cpu_hours_diagnostic": (
            aggregate_process_cpu_seconds / 3600.0
        ),
        "observed_plus_four_replay_waves_wall_clock_hours": (
            calendar_seconds / 3600.0
        ),
        "parallel_cores": parallel_cores,
        "aggregate_memory_GiB": aggregate_memory,
        "measured_peak_RSS_GiB_per_task": replay_peak_rss,
        "requested_CPUs_per_task": EXPECTED_REQUESTED_CPUS_PER_TASK,
        "effective_numeric_threads_per_task": EXPECTED_REQUESTED_CPUS_PER_TASK,
        "scheduler_allocated_CPUs_per_task": EXPECTED_ALLOCATED_CPUS_PER_TASK,
        "requested_memory_GiB_per_task": EXPECTED_MEMORY_GIB_PER_TASK,
        "projected_scratch_GiB_no_deletion_credit": scratch_gib,
        "gpu_hours": 0.0,
        "formulas": {
            "controller_trials": (
                "1*reference + 15*core + 8*low_rank + 4*hybrid + "
                f"{24 + cap - 52}*max(core,low_rank,hybrid)"
            ),
            "controller_fixed_units": (
                "1*shared_setup + 1*diagnostic + 1*selected_refit_pair + 1*report"
            ),
            "fresh_replay": "synthetic_generation + fresh_M0 + fresh_M1 + controller",
            "allocated_core_hours": (
                "2.0*(observed + 99*fresh_replay)*2_allocated_CPUs/3600"
            ),
            "calendar_wall_hours": (
                "2.0*(observed + ceil(99/32)*fresh_replay)/3600"
            ),
            "scratch": "observed + 99*fresh_replay; no deletion credit",
        },
    }
    gates = {
        "cpu_core_hours": projected[
            "observed_plus_99_replays_allocated_cpu_core_hours"
        ]
        <= RESOURCE_CEILINGS["cpu_core_hours"],
        "gpu_hours": projected["gpu_hours"] <= RESOURCE_CEILINGS["gpu_hours"],
        "wall_clock_hours": projected[
            "observed_plus_four_replay_waves_wall_clock_hours"
        ]
        <= RESOURCE_CEILINGS["wall_clock_hours"],
        "parallel_cores": projected["parallel_cores"]
        <= RESOURCE_CEILINGS["parallel_cores"],
        "aggregate_memory_GiB": projected["aggregate_memory_GiB"]
        <= RESOURCE_CEILINGS["aggregate_memory_GiB"],
        "scratch_GiB": projected["projected_scratch_GiB_no_deletion_credit"]
        <= RESOURCE_CEILINGS["scratch_GiB"],
        "per_task_measured_RSS_within_initial_request": replay_peak_rss
        <= EXPECTED_MEMORY_GIB_PER_TASK,
    }
    projected["ceiling_gates"] = gates
    projected["all_ceiling_gates_pass"] = all(gates.values())
    return projected


def project_52_and_80(
    measured_units: Mapping[str, Any],
) -> Mapping[str, Mapping[str, Any]]:
    return {
        "52_trial_fully_triggered_branch": project_search_resources(measured_units, 52),
        "80_trial_cap": project_search_resources(measured_units, 80),
    }


def validate_aggregate_report(payload: Mapping[str, Any]) -> None:
    """Reject cell-level identifiers/outcomes and non-JSON values recursively."""
    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if not isinstance(key, str):
                    raise BenchmarkError("aggregate report contains a non-string key")
                if key in PROHIBITED_OUTPUT_KEYS:
                    raise BenchmarkError(f"aggregate report contains prohibited key: {key}")
                visit(nested, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                visit(nested, f"{path}[{index}]")
        elif value is None or isinstance(value, (str, bool, int)):
            return
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise BenchmarkError(f"aggregate report contains nonfinite value at {path}")
        else:
            raise BenchmarkError(f"aggregate report contains non-JSON value at {path}")

    visit(payload, "report")


def _prepare_output_root(path: Path) -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise BenchmarkError("benchmark requires a numeric SLURM_JOB_ID")
    episode = _scratch_episode_root()
    absolute = Path(os.path.abspath(path))
    if (
        absolute.parent != episode
        or absolute.name != f"{OUTPUT_PREFIX}_job_{job_id}"
        or absolute.is_symlink()
    ):
        raise BenchmarkError("benchmark output is not the exact job-scoped Scratch child")
    try:
        absolute.mkdir(mode=0o770)
    except FileExistsError as error:
        raise BenchmarkError("benchmark output root already exists") from error
    return absolute


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    validate_aggregate_report(payload)
    directory_fd = os.open(
        path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    temporary_name = f".{path.name}.tmp.{os.getpid()}"
    descriptor = -1
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o660,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


def _validate_runtime() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise BenchmarkError("scale benchmark must run inside a Slurm job")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            raise BenchmarkError(f"{name} must equal 1")


def run_benchmark(args: argparse.Namespace) -> Mapping[str, Any]:
    _validate_runtime()
    contract_witness = validate_frozen_contracts(
        args.contract,
        args.addendum,
        args.hybrid_clarification,
        args.hybrid_rank_clarification,
    )
    prerequisite_witness = validate_prerequisite_reports(
        args.core_report,
        args.actual_k_report,
        args.adequacy_report,
        args.optional_report,
    )
    output_root = _prepare_output_root(args.output_root)
    summary_path = output_root / SUMMARY_FILENAME
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running",
        "stage": "v3_M2_membership_blind_full_search_scale_benchmark_v1",
        "analysis_label": "exploratory_development_only",
        "SLURM_job_id": os.environ["SLURM_JOB_ID"],
        "contract_validation": contract_witness,
        "prerequisite_validation": prerequisite_witness,
        "actual_K_opened": False,
        "synthetic_membership_constructed": False,
        "real_target_membership_or_M2_effects_opened": False,
        "existing_M0_full_refit_or_telemetry_opened": False,
        "final_or_unassigned_or_SWC_opened": False,
        "cell_ids_candidate_indexes_K_values_or_memberships_emitted": False,
        "aggregate_only_report": True,
        "final_scale_authorization_claimed": False,
        "optional_timing_workload_scope": {
            "fixed_8_low_rank_and_4_hybrid_units_are_worst_case_target_free_timing_probes": True,
            "probe_execution_is_coverage_evidence_authorization": False,
            "production_optional_activation_rank_and_edges_remain_outcome_bound": True,
            "rank_neighbor_full_refit_r6_to_r8_is_target_free_timing_only": True,
            "observed_session_cap80_bookkeeping_excludes_numerical_fits": True,
        },
    }
    try:
        modules, availability = discover_timing_hooks()
        report["execution_unit_availability"] = availability
        if availability["status"] != "available":
            report.update(
                {
                    "status": "blocked_missing_execution_units",
                    "policy_ceiling_projection_available": False,
                    "policy_ceiling_projection_pass": False,
                    "missing_unit_timings_imputed": False,
                    "elapsed_benchmark_wall_seconds": time.perf_counter() - started,
                }
            )
            _atomic_json(summary_path, report)
            return report

        cells, pair_geometry, actual_k_witness = load_actual_k_only_shape(
            args.interface_root,
            args.geometry_root,
            args.target_root,
            args.inventory,
        )
        report["actual_K_opened"] = True
        report["actual_K_input_witness"] = actual_k_witness
        outcome, context = construct_benchmark_context(cells, pair_geometry)
        report["synthetic_membership_constructed"] = True
        report["synthetic_outcome_witness"] = outcome.witness
        measured = collect_measured_units(context, modules)
        projections = project_52_and_80(measured)
        all_pass = all(
            projection["all_ceiling_gates_pass"]
            for projection in projections.values()
        )
        report["measured_units"] = measured
        report["resource_projections"] = projections
        report["policy_ceiling_projection_available"] = True
        report["policy_ceiling_projection_pass"] = all_pass
        report["missing_unit_timings_imputed"] = False
        report["elapsed_benchmark_wall_seconds"] = time.perf_counter() - started
        report["status"] = (
            "pass_pending_sacct_and_remaining_budget_reconciliation"
            if all_pass
            else "fail_projected_resource_ceiling"
        )
        _atomic_json(summary_path, report)
        return report
    except Exception as error:
        report.update(
            {
                "status": "fail",
                "failure_class": type(error).__name__,
                "failure_code": "v3_M2_scale_benchmark_failed_closed",
                "policy_ceiling_projection_pass": False,
                "elapsed_benchmark_wall_seconds": time.perf_counter() - started,
            }
        )
        _atomic_json(summary_path, report)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--addendum", type=Path, required=True)
    parser.add_argument("--hybrid-clarification", type=Path, required=True)
    parser.add_argument("--hybrid-rank-clarification", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--interface-root", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--core-report", type=Path, required=True)
    parser.add_argument("--actual-k-report", type=Path, required=True)
    parser.add_argument("--adequacy-report", type=Path, required=True)
    parser.add_argument("--optional-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_benchmark(args)
    except Exception as error:
        print(
            f"ERROR: M2 scale benchmark failed closed ({type(error).__name__})",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "policy_ceiling_projection_pass": result[
                    "policy_ceiling_projection_pass"
                ],
                "aggregate_only_report": result["aggregate_only_report"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"].startswith("pass_pending_") else 3


if __name__ == "__main__":
    raise SystemExit(main())
