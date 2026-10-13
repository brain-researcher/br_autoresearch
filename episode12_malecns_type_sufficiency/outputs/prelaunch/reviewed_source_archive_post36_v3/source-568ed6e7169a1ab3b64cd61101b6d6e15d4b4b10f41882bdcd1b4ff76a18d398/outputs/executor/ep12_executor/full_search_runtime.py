"""Authenticated production runtime provider for EP12 fitted-null searches.

There is deliberately no final-connectivity loader here. Loading reconstructs
and verifies each design, generator, schedule, seed, callback, and lock.
"""

from __future__ import annotations

import errno
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from .adaptive_search import validate_development_configuration
from .fitted_null import (
    FittedNullDesign,
    TypePooledExpansionPlan,
    freeze_null_seed_manifest,
    null_seed_binding,
)
from .full_search_null import (
    FullSearchNullRuntime,
    freeze_procedure_lock,
    verify_null_controller_program,
)
from .model_artifacts import (
    ModelArtifactError,
    atomic_json,
    atomic_npy,
    load_model_state,
    write_model_state,
)
from .null_trial_runner import (
    NullTrialInputs,
    ProductionFullSearchCallback,
    SELECTION_PROGRAM_SHA256,
    _collapse,
)
from .policy import EpisodePolicy, digest_bytes, digest_object
from .procedure_lock import ProcedureLockError, verify_procedure_lock
from .scientific_models import FittedPredictiveModel, fit_triplet, seed_from


BUNDLE_SCHEMA = "ep12.full_search_null_runtime_bundle.v4"
CALLBACK_DATA_SCHEMA = "ep12.full_search_null_callback_data.v1"
SELECTION_ATTESTATION_SCHEMA = "ep12.selected_development_fit_attestation.v2"
RESOURCE_PLAN_SCHEMA = "ep12.full_search_null_resource_plan.v4"
RUNTIME_BENCHMARK_SCHEMA = "ep12.full_search_null_runtime_benchmark.v3"
TIMING_CORPUS_SCHEMA = "ep12.authenticated_observed_timing_corpus.v1"
SCRATCH_STAGE_SCHEMA = "ep12.full_search_null_scratch_stage.v1"
RUNTIME_BENCHMARK_SCOPE = (
    "three_deterministic_fitted_null_seeds_one_authenticated_configuration_"
    "one_complete_T_U_M_branch_per_seed"
)
RUNTIME_BENCHMARK_ROLE = "authenticated_planning_estimate"
PER_TRIAL_RUNTIME_ENFORCEMENT = (
    "isolated_callback_timeout_at_minimum_of_12h_and_remaining_cumulative_wall"
)
CUMULATIVE_RUNTIME_ENFORCEMENT = (
    "sum_authenticated_persisted_trial_wall_seconds_at_trial_boundaries_"
    "across_checkpoint_requeue"
)
WALL_BUDGET_EXHAUSTION_HANDLING = (
    "durable_self_hashed_incomplete_search_receipt_fail_closed_no_finalize_or_aggregate"
)
EPISODE_ROOT = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency"
)
EPISODE_OUTPUTS_ROOT = EPISODE_ROOT / "outputs"
EPISODE_SCRATCH_ROOT = Path(
    "/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency"
)
_TRIAL_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?$")


class FullSearchRuntimeError(RuntimeError):
    """A runtime bundle is incomplete, changed, or crosses a data boundary."""


def _scoped_path(
    path: Path,
    *,
    root: Path,
    label: str,
    allow_root: bool = False,
    require_directory: bool = False,
    require_file: bool = False,
) -> Path:
    """Return a canonical path below one exact EP12 root, rejecting aliases."""

    raw = Path(path)
    if not raw.is_absolute() or ".." in raw.parts:
        raise FullSearchRuntimeError(
            f"{label} must be an absolute traversal-free EP12 path"
        )
    lexical = Path(os.path.abspath(os.fspath(raw)))
    resolved = raw.resolve(strict=False)
    root_resolved = Path(root).resolve(strict=False)
    if lexical != resolved:
        raise FullSearchRuntimeError(f"{label} may not traverse a symbolic link")
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as error:
        raise FullSearchRuntimeError(
            f"{label} must stay under exact EP12 root {root_resolved}"
        ) from error
    if not allow_root and not relative.parts:
        raise FullSearchRuntimeError(f"{label} may not equal the EP12 root")
    if require_directory and not resolved.is_dir():
        raise FullSearchRuntimeError(f"{label} is not an existing directory")
    if require_file and not resolved.is_file():
        raise FullSearchRuntimeError(f"{label} is not an existing file")
    return resolved


def _runtime_bundle_path(path: Path, *, label: str) -> Path:
    """Accept an immutable runtime bundle from durable Outputs or EP12 Scratch."""

    errors: list[str] = []
    for root in (EPISODE_OUTPUTS_ROOT, EPISODE_SCRATCH_ROOT):
        try:
            return _scoped_path(
                path,
                root=root,
                label=label,
                require_directory=True,
            )
        except FullSearchRuntimeError as error:
            errors.append(str(error))
    raise FullSearchRuntimeError(
        f"{label} must be an authenticated EP12 Outputs or Scratch directory: "
        + " | ".join(errors)
    )


def _trial_id(value: Any, label: str = "trial ID") -> str:
    result = str(value)
    if _TRIAL_ID_RE.fullmatch(result) is None:
        raise FullSearchRuntimeError(f"{label} is not a safe canonical identifier")
    return result


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ))
    except (TypeError, ValueError) as error:
        raise FullSearchRuntimeError("runtime metadata is not canonical JSON") from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise FullSearchRuntimeError(f"{label} may not be a symbolic link")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FullSearchRuntimeError(f"missing {label}: {path}") from error
    except (OSError, UnicodeError) as error:
        raise FullSearchRuntimeError(f"unreadable {label}: {path}") from error
    except json.JSONDecodeError as error:
        raise FullSearchRuntimeError(f"malformed {label}: {path}") from error
    if not isinstance(value, dict):
        raise FullSearchRuntimeError(f"{label} must be a JSON object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FullSearchRuntimeError(f"{label} must be a JSON object")
    return value


def _sha(value: Any, label: str) -> str:
    result = str(value)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise FullSearchRuntimeError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    core = {key: item for key, item in value.items() if key != field}
    if value.get(field) != digest_object(core):
        raise FullSearchRuntimeError(f"{label} self-hash mismatch")


def verify_runtime_benchmark_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Verify timing-only evidence from three production-path null fits."""

    required = {
        "schema_version", "method", "worker_count",
        "benchmark_replicate_indices", "benchmark_replicate_ids",
        "benchmark_replicate_seeds_sha256", "max_timing_trial_id",
        "max_timing_configuration_sha256", "design_sha256",
        "generator_manifest_sha256", "expansion_plan_sha256",
        "callback_identity_sha256", "elapsed_wall_seconds",
        "measured_parallel_mean_wall_seconds",
        "measured_parallel_max_wall_seconds",
        "complete_T_U_M_branches_measured",
        "benchmark_scope", "benchmark_role", "projection_is_upper_bound",
        "observed_timing_corpus_sha256",
        "observed_timing_corpus_trial_count",
        "scientific_scores_or_objectives_persisted", "development_only",
        "final_connectivity_accessed", "receipt_sha256",
    }
    receipt = _json_copy(value)
    if set(receipt) != required:
        raise FullSearchRuntimeError("runtime-benchmark receipt schema differs")
    _verify_self_hash(receipt, "receipt_sha256", "runtime benchmark receipt")
    elapsed = receipt["elapsed_wall_seconds"]
    if not isinstance(elapsed, list) or len(elapsed) != 3:
        raise FullSearchRuntimeError("runtime benchmark must contain three timings")
    seconds = [float(value) for value in elapsed]
    mean_seconds = float(receipt["measured_parallel_mean_wall_seconds"])
    max_seconds = float(receipt["measured_parallel_max_wall_seconds"])
    if not (
        receipt["schema_version"] == RUNTIME_BENCHMARK_SCHEMA
        and receipt["method"]
        == "production_fitted_null_sampler_and_T_U_M_callback_base_fit"
        and int(receipt["worker_count"]) == 32
        and receipt["benchmark_replicate_indices"] == [0, 1, 2]
        and receipt["benchmark_replicate_ids"]
        == ["null_001", "null_002", "null_003"]
        and all(math.isfinite(value) and value > 0.0 for value in seconds)
        and math.isclose(mean_seconds, sum(seconds) / len(seconds))
        and math.isclose(max_seconds, max(seconds))
        and int(receipt["complete_T_U_M_branches_measured"]) == 3
        and receipt["benchmark_scope"] == RUNTIME_BENCHMARK_SCOPE
        and receipt["benchmark_role"] == RUNTIME_BENCHMARK_ROLE
        and receipt["projection_is_upper_bound"] is False
        and int(receipt["observed_timing_corpus_trial_count"]) == 36
        and receipt["scientific_scores_or_objectives_persisted"] is False
        and receipt["development_only"] is True
        and receipt["final_connectivity_accessed"] is False
    ):
        raise FullSearchRuntimeError("runtime benchmark violates its frozen protocol")
    for field in (
        "benchmark_replicate_seeds_sha256",
        "max_timing_configuration_sha256",
        "design_sha256",
        "generator_manifest_sha256",
        "expansion_plan_sha256",
        "callback_identity_sha256",
        "observed_timing_corpus_sha256",
    ):
        _sha(receipt[field], field.replace("_", " "))
    _trial_id(receipt["max_timing_trial_id"], "max-timing trial ID")
    return receipt


def verify_resource_plan(
    value: Mapping[str, Any], *, policy: EpisodePolicy
) -> dict[str, Any]:
    """Authenticate only engineering capacity, never scientific pass criteria."""

    required = {
        "schema_version", "policy_sha256", "controller_program_sha256",
        "requested_cpus_per_replicate", "provider_type_workers",
        "provider_type_parallelism", "control_branch_execution",
        "branch_fit_inventory", "minimum_search_last_trial_ordinal",
        "maximum_search_last_trial_ordinal", "timing_basis",
        "complete_T_U_M_fit_count_scope",
        "minimum_complete_T_U_M_fit_count", "maximum_complete_T_U_M_fit_count",
        "minimum_model_family_fit_count", "maximum_model_family_fit_count",
        "conservative_seconds_per_complete_T_U_M_branch",
        "projected_minimum_replicate_wall_hours",
        "projected_maximum_replicate_wall_hours",
        "projected_minimum_replicate_allocated_core_hours",
        "projected_maximum_replicate_allocated_core_hours",
        "projected_minimum_replicate_busy_core_hours",
        "projected_maximum_replicate_busy_core_hours",
        "program_search_count", "observed_search_count",
        "synthetic_null_search_count",
        "projected_minimum_program_allocated_core_hours",
        "projected_maximum_program_allocated_core_hours",
        "projected_minimum_program_busy_core_hours",
        "projected_maximum_program_busy_core_hours",
        "maximum_complete_T_U_M_branches_in_one_control_trial",
        "projected_maximum_control_trial_wall_hours",
        "per_trial_wall_hours_limit", "overall_wall_hours_limit",
        "runtime_benchmark_scope", "runtime_benchmark_role",
        "projection_is_upper_bound",
        "actual_per_trial_runtime_enforcement",
        "actual_cumulative_runtime_enforcement",
        "wall_budget_exhaustion_handling",
        "cpu_core_hour_ceiling_enforced", "checkpoint_resume",
        "array_replicates", "gpu_count", "scientific_thresholds_added",
        "development_only", "final_connectivity_accessed", "plan_sha256",
    }
    plan = _json_copy(value)
    if set(plan) != required:
        raise FullSearchRuntimeError("full-null resource-plan schema differs")
    _verify_self_hash(plan, "plan_sha256", "full-null resource plan")
    workers = int(plan["provider_type_workers"])
    requested = int(plan["requested_cpus_per_replicate"])
    minimum_fits = int(plan["minimum_complete_T_U_M_fit_count"])
    maximum_fits = int(plan["maximum_complete_T_U_M_fit_count"])
    seconds_per_branch = float(
        plan["conservative_seconds_per_complete_T_U_M_branch"]
    )
    minimum_wall = float(plan["projected_minimum_replicate_wall_hours"])
    maximum_wall = float(plan["projected_maximum_replicate_wall_hours"])
    minimum_allocated = float(
        plan["projected_minimum_replicate_allocated_core_hours"]
    )
    maximum_allocated = float(
        plan["projected_maximum_replicate_allocated_core_hours"]
    )
    minimum_busy = float(plan["projected_minimum_replicate_busy_core_hours"])
    maximum_busy = float(plan["projected_maximum_replicate_busy_core_hours"])
    program_search_count = int(plan["program_search_count"])
    timing = plan["timing_basis"]
    if not isinstance(timing, dict) or set(timing) != {
        "method", "observed_timing_corpus_schema",
        "observed_timing_corpus_trial_count",
        "observed_timing_corpus_sha256",
        "observed_attestation_sha256",
        "observed_attestation_manifest_file_sha256",
        "observed_serial_mean_wall_seconds",
        "observed_serial_max_wall_seconds",
        "observed_serial_mean_cpu_seconds_per_branch",
        "observed_serial_max_cpu_seconds_per_branch",
        "max_timing_trial_id", "max_busy_timing_trial_id",
        "legacy_cost_projection_file_sha256",
        "legacy_cost_projection_used_for_projection",
        "max_timing_report_file_sha256",
        "max_busy_timing_report_file_sha256", "parallelism_scope",
        "control_branch_execution", "scientific_outcomes_persisted",
        "runtime_benchmark_receipt",
    }:
        raise FullSearchRuntimeError("full-null timing basis schema differs")
    benchmark = verify_runtime_benchmark_receipt(
        _mapping(timing["runtime_benchmark_receipt"], "runtime benchmark")
    )
    serial_mean = float(timing["observed_serial_mean_wall_seconds"])
    serial_max = float(timing["observed_serial_max_wall_seconds"])
    serial_cpu_mean = float(
        timing["observed_serial_mean_cpu_seconds_per_branch"]
    )
    serial_cpu_max = float(
        timing["observed_serial_max_cpu_seconds_per_branch"]
    )
    expected_seconds = max(
        float(benchmark["measured_parallel_max_wall_seconds"]),
        serial_max / workers,
    )
    if not (
        plan["schema_version"] == RESOURCE_PLAN_SCHEMA
        and plan["policy_sha256"] == policy.policy_hash
        and workers == requested == policy.cpu_cores_per_trial_maximum == 32
        and plan["provider_type_parallelism"]
        == "ordered_deterministic_thread_pool_within_complete_branch"
        and plan["control_branch_execution"] == "sequential_no_nested_pool"
        and int(plan["minimum_search_last_trial_ordinal"]) == 52
        and int(plan["maximum_search_last_trial_ordinal"])
        == policy.maximum_valid_trials == 96
        and plan["complete_T_U_M_fit_count_scope"]
        == "per_observed_or_synthetic_null_search"
        and minimum_fits == 308
        and maximum_fits == 690
        and int(plan["minimum_model_family_fit_count"])
        == len(policy.model_triplet) * minimum_fits
        and int(plan["maximum_model_family_fit_count"])
        == len(policy.model_triplet) * maximum_fits
        and math.isfinite(serial_mean)
        and math.isfinite(serial_max)
        and math.isfinite(serial_cpu_mean)
        and math.isfinite(serial_cpu_max)
        and 0.0 < serial_mean <= serial_max
        and 0.0 < serial_cpu_mean <= serial_cpu_max
        and timing["observed_timing_corpus_schema"] == TIMING_CORPUS_SCHEMA
        and int(timing["observed_timing_corpus_trial_count"])
        == policy.minimum_valid_trials == 36
        and benchmark["observed_timing_corpus_sha256"]
        == timing["observed_timing_corpus_sha256"]
        and int(benchmark["observed_timing_corpus_trial_count"])
        == int(timing["observed_timing_corpus_trial_count"])
        and benchmark["max_timing_trial_id"] == timing["max_timing_trial_id"]
        and timing["method"]
        == (
            "production_parallel_benchmark_with_authenticated_"
            "observed_timing_corpus_maxima"
        )
        and timing["parallelism_scope"]
        == "provider_types_within_one_complete_T_U_M_branch"
        and timing["control_branch_execution"] == "sequential"
        and math.isclose(seconds_per_branch, expected_seconds)
        and math.isclose(minimum_wall, minimum_fits * expected_seconds / 3600.0)
        and math.isclose(maximum_wall, maximum_fits * expected_seconds / 3600.0)
        and math.isclose(minimum_allocated, minimum_wall * requested)
        and math.isclose(maximum_allocated, maximum_wall * requested)
        and math.isclose(minimum_busy, minimum_fits * serial_cpu_max / 3600.0)
        and math.isclose(maximum_busy, maximum_fits * serial_cpu_max / 3600.0)
        and 0.0 < minimum_busy <= minimum_allocated
        and minimum_busy <= maximum_busy <= maximum_allocated
        and int(plan["observed_search_count"]) == 1
        and int(plan["synthetic_null_search_count"]) == policy.null_replicates == 99
        and program_search_count == 1 + policy.null_replicates == 100
        and math.isclose(
            float(plan["projected_minimum_program_allocated_core_hours"]),
            minimum_allocated * program_search_count,
        )
        and math.isclose(
            float(plan["projected_maximum_program_allocated_core_hours"]),
            maximum_allocated * program_search_count,
        )
        and math.isclose(
            float(plan["projected_minimum_program_busy_core_hours"]),
            minimum_busy * program_search_count,
        )
        and math.isclose(
            float(plan["projected_maximum_program_busy_core_hours"]),
            maximum_busy * program_search_count,
        )
        and float(plan["projected_maximum_control_trial_wall_hours"])
        <= policy.wall_hours_per_trial_maximum
        and float(plan["per_trial_wall_hours_limit"])
        == policy.wall_hours_per_trial_maximum
        and maximum_wall
        <= float(plan["overall_wall_hours_limit"])
        == float(policy.raw["budgets"]["wall_clock_hour_ceiling"])
        and plan["runtime_benchmark_scope"] == RUNTIME_BENCHMARK_SCOPE
        and plan["runtime_benchmark_scope"] == benchmark["benchmark_scope"]
        and plan["runtime_benchmark_role"] == RUNTIME_BENCHMARK_ROLE
        and plan["runtime_benchmark_role"] == benchmark["benchmark_role"]
        and plan["projection_is_upper_bound"] is False
        and benchmark["projection_is_upper_bound"] is False
        and plan["actual_per_trial_runtime_enforcement"]
        == PER_TRIAL_RUNTIME_ENFORCEMENT
        and plan["actual_cumulative_runtime_enforcement"]
        == CUMULATIVE_RUNTIME_ENFORCEMENT
        and plan["wall_budget_exhaustion_handling"]
        == WALL_BUDGET_EXHAUSTION_HANDLING
        and plan["cpu_core_hour_ceiling_enforced"] is False
        and plan["checkpoint_resume"]
        == "immutable_per-step_hash_chain_and_idempotent_replay"
        and int(plan["array_replicates"]) == policy.null_replicates
        and int(plan["gpu_count"]) == 0
        and plan["scientific_thresholds_added"] is False
        and timing["legacy_cost_projection_used_for_projection"] is False
        and timing["scientific_outcomes_persisted"] is False
        and plan["development_only"] is True
        and plan["final_connectivity_accessed"] is False
    ):
        raise FullSearchRuntimeError("full-null resource plan violates frozen limits")
    inventory = plan["branch_fit_inventory"]
    if not isinstance(inventory, dict) or set(inventory) != {
        "strength_and_margin_preserving_graph_randomization",
        "cross_side_component_alignment_permutation",
        "shuffled_partner_and_valid_side_controls",
        "binary_weighted_vocabulary_and_rank_sensitivities",
        "nuisance_and_partner_block_ablations",
        "capacity_matched_noise_controls",
        "leave_one_development_type_and_partner_family_influence",
        "few_type_and_high_strength_neuron_concentration_check",
        "status_endpoint_anatomy_and_missingness_checks",
    } or any(int(count) < 1 for count in inventory.values()):
        raise FullSearchRuntimeError("resource plan omits a control branch inventory")
    maximum_control_branches = max(int(count) for count in inventory.values())
    if not (
        int(plan["maximum_complete_T_U_M_branches_in_one_control_trial"])
        == maximum_control_branches
        and math.isclose(
            float(plan["projected_maximum_control_trial_wall_hours"]),
            maximum_control_branches * seconds_per_branch / 3600.0,
        )
    ):
        raise FullSearchRuntimeError("resource plan control-branch projection differs")
    for field in (
        "observed_timing_corpus_sha256",
        "observed_attestation_sha256",
        "observed_attestation_manifest_file_sha256",
        "legacy_cost_projection_file_sha256",
        "max_timing_report_file_sha256",
        "max_busy_timing_report_file_sha256",
    ):
        _sha(timing[field], field.replace("_", " "))
    _trial_id(timing["max_timing_trial_id"], "maximum-wall timing trial ID")
    _trial_id(timing["max_busy_timing_trial_id"], "maximum-CPU timing trial ID")
    return plan


def freeze_selected_fit_attestation(
    *,
    controller_program: Mapping[str, Any],
    selected_trial_id: str,
    selected_configuration: Mapping[str, Any],
    development_role_manifest_sha256: str,
    source_selection_artifact_sha256: str,
    upstream_procedure_lock_sha256: str,
    null_generator_selection_sha256: str,
) -> dict[str, Any]:
    """Bind the observed winner/generators without placing it in a null schedule."""

    selected_trial_id = _trial_id(selected_trial_id, "selected trial ID")
    configuration = _json_copy(selected_configuration)
    if configuration.get("trial_id") != selected_trial_id:
        raise FullSearchRuntimeError("selected fit/configuration identity differs")
    core = {
        "schema_version": SELECTION_ATTESTATION_SCHEMA,
        "controller_program_sha256": _sha(
            controller_program.get("program_sha256"), "controller program"
        ),
        "upstream_procedure_lock_sha256": _sha(
            upstream_procedure_lock_sha256, "upstream procedure lock"
        ),
        "selected_trial_id": selected_trial_id,
        "selected_configuration": configuration,
        "selected_configuration_sha256": _sha(
            digest_object(configuration), "selected configuration"
        ),
        "null_generator_selection_sha256": _sha(
            null_generator_selection_sha256, "null-generator selection"
        ),
        "development_role_manifest_sha256": _sha(
            development_role_manifest_sha256, "development role manifest"
        ),
        "selection_program_sha256": SELECTION_PROGRAM_SHA256,
        "source_selection_artifact_sha256": _sha(
            source_selection_artifact_sha256, "source selection artifact"
        ),
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "attestation_sha256": digest_object(core)}


def _verify_attestation(
    value: Mapping[str, Any],
    *,
    controller_program: Mapping[str, Any],
    role_sha256: str,
    upstream_procedure_lock_sha256: str,
    null_generator_selection_sha256: str,
    policy: EpisodePolicy,
) -> Mapping[str, Any]:
    required = {
        "schema_version", "controller_program_sha256",
        "upstream_procedure_lock_sha256", "selected_trial_id",
        "selected_configuration", "selected_configuration_sha256",
        "null_generator_selection_sha256", "development_role_manifest_sha256",
        "selection_program_sha256", "source_selection_artifact_sha256",
        "development_only", "final_connectivity_accessed",
        "final_connectivity_access_authorized", "attestation_sha256",
    }
    if set(value) != required:
        raise FullSearchRuntimeError("selected-fit attestation schema differs")
    _verify_self_hash(value, "attestation_sha256", "selected-fit attestation")
    _trial_id(value.get("selected_trial_id"), "attested selected trial ID")
    if not (
        value["schema_version"] == SELECTION_ATTESTATION_SCHEMA
        and value["controller_program_sha256"]
        == controller_program["program_sha256"]
        and value["upstream_procedure_lock_sha256"]
        == _sha(upstream_procedure_lock_sha256, "upstream procedure lock")
        and value["null_generator_selection_sha256"]
        == _sha(null_generator_selection_sha256, "null-generator selection")
        and value["development_role_manifest_sha256"] == _sha(role_sha256, "role")
        and value["selection_program_sha256"] == SELECTION_PROGRAM_SHA256
        and value["development_only"] is True
        and value["final_connectivity_accessed"] is False
        and value["final_connectivity_access_authorized"] is False
    ):
        raise FullSearchRuntimeError("selected-fit attestation differs from lock")
    configuration = _mapping(value["selected_configuration"], "selected configuration")
    validate_development_configuration(policy, configuration)
    if (
        configuration.get("trial_id") != value["selected_trial_id"]
        or digest_object(configuration) != value["selected_configuration_sha256"]
    ):
        raise FullSearchRuntimeError("selected-fit attestation configuration changed")
    _sha(value["source_selection_artifact_sha256"], "selection artifact")
    return configuration


def build_selected_fitted_null_state(
    *,
    policy: EpisodePolicy,
    neuron_ids: Sequence[int | str],
    focal_types: Sequence[str],
    sides: Sequence[str],
    raw_counts: np.ndarray,
    raw_vocabulary: Sequence[str],
    partner_hierarchy: Mapping[str, Mapping[str, Any]],
    covariates_by_spline_df: Mapping[int, np.ndarray],
    selected_configuration: Mapping[str, Any],
    fit_settings: Mapping[str, Any],
) -> tuple[
    FittedNullDesign,
    dict[tuple[str, str], FittedPredictiveModel],
    dict[int, np.ndarray],
]:
    """Refit exact selected source-side T/U generators from development arrays."""

    config = dict(selected_configuration)
    validate_development_configuration(policy, config)
    if set(fit_settings) != {"integration_draws", "source_cv_folds"}:
        raise FullSearchRuntimeError("selected-fit settings differ from schema")
    covariates = {
        int(key): np.asarray(value, dtype=np.float64)
        for key, value in covariates_by_spline_df.items()
    }
    expected_df = set(map(int, policy.grammar_choices["nuisance_spline_df"]))
    if set(covariates) != expected_df:
        raise FullSearchRuntimeError("selected-fit covariate branches are incomplete")
    counts = np.asarray(raw_counts, dtype=np.int64)
    types = np.asarray(list(map(str, focal_types)), dtype=object)
    side_values = np.asarray(list(map(str, sides)), dtype=object)
    ids = np.asarray(list(neuron_ids), dtype=object)
    if (
        counts.shape != (len(ids), len(raw_vocabulary))
        or len(types) != len(ids) or len(side_values) != len(ids)
        or any(value.shape[0] != len(ids) for value in covariates.values())
    ):
        raise FullSearchRuntimeError("selected-fit development arrays disagree")
    collapsed, vocabulary, raw_to_collapsed = _collapse(
        counts, types, raw_vocabulary, config, partner_hierarchy
    )
    selected_covariates = covariates[int(config["nuisance_spline_df"])]
    minimum = int(config["component_count"]) * int(fit_settings["source_cv_folds"])
    scorable: list[str] = []
    for provider_type in sorted(set(map(str, types))):
        mask = types == provider_type
        left = mask & (side_values == "L")
        right = mask & (side_values == "R")
        if (
            int(left.sum()) >= minimum and int(right.sum()) >= minimum
            and np.all(collapsed[left].sum(1) > 0)
            and np.all(collapsed[right].sum(1) > 0)
        ):
            scorable.append(provider_type)
    if not scorable:
        raise FullSearchRuntimeError("selected configuration has no scorable type")
    keep = np.isin(types, np.asarray(scorable, dtype=object))
    typed_bins = sorted({
        vocabulary[int(raw_to_collapsed[index])]
        for index, key in enumerate(raw_vocabulary)
        if str(key).startswith("typed::")
    })
    design = FittedNullDesign.create(
        neuron_ids=ids[keep].tolist(),
        focal_types=types[keep].tolist(),
        sides=side_values[keep].tolist(),
        covariates=selected_covariates[keep],
        raw_counts=counts[keep],
        raw_vocabulary=list(map(str, raw_vocabulary)),
        collapsed_vocabulary=vocabulary,
        raw_to_collapsed=raw_to_collapsed,
        collapsed_typed_bins=typed_bins,
    )
    options = {
        "integration_draws": int(fit_settings["integration_draws"]),
        "folds": int(fit_settings["source_cv_folds"]),
        "latent_rank": int(config["rank"]),
        "ridge": float(config["regularization"]),
        "pseudocount": float(config["composition_pseudocount"]),
        "representation": str(config["representation"]),
        "covariance_mode": str(config["covariance"]),
        "covariance_rank": int(config["covariance_rank"]),
        "component_candidates": (int(config["component_count"]),),
    }
    generators: dict[tuple[str, str], FittedPredictiveModel] = {}
    for provider_type in scorable:
        for side, label in (("L", "left-source"), ("R", "right-source")):
            mask = (types == provider_type) & (side_values == side)
            base = seed_from("ep12-development", config["trial_id"], provider_type)
            fitted = fit_triplet(
                collapsed[mask], selected_covariates[mask],
                seed=seed_from(base, label), **options,
            )
            generators[(provider_type, side)] = (
                fitted.T if fitted.null_generator_model_id == "T" else fitted.U
            )
    filtered = {
        key: np.ascontiguousarray(value[keep], dtype=np.float64)
        for key, value in covariates.items()
    }
    return design, generators, filtered

def _write_callback_data(
    directory: Path,
    *,
    design: FittedNullDesign,
    covariates: Mapping[int, np.ndarray],
    hierarchy: Mapping[str, Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for spline_df, source in sorted(covariates.items()):
        value = np.ascontiguousarray(source, dtype=np.float64)
        if value.ndim != 2 or value.shape[0] != len(design.neuron_ids):
            raise FullSearchRuntimeError("callback covariates differ from design")
        name = f"covariates_spline_df_{int(spline_df)}.npy"
        atomic_npy(directory / name, value)
        rows.append({
            "spline_df": int(spline_df),
            "file": name,
            "dtype": "float64",
            "shape": list(value.shape),
            "file_sha256": digest_bytes((directory / name).read_bytes()),
        })
    core = {
        "schema_version": CALLBACK_DATA_SCHEMA,
        "design_sha256": design.design_sha256,
        "raw_vocabulary": list(design.raw_vocabulary),
        "partner_hierarchy": _json_copy(hierarchy),
        "fit_settings": _json_copy(settings),
        "covariates": rows,
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    result = {**core, "callback_data_sha256": digest_object(core)}
    atomic_json(directory / "callback_data.json", result)
    return result


def _load_callback_data(
    directory: Path,
    *,
    design: FittedNullDesign,
    policy: EpisodePolicy,
    artifact_root: Path,
) -> tuple[NullTrialInputs, dict[str, Any]]:
    metadata = _read_json(directory / "callback_data.json", "callback data")
    required = {
        "schema_version", "design_sha256", "raw_vocabulary",
        "partner_hierarchy", "fit_settings", "covariates", "development_only",
        "final_connectivity_accessed", "final_connectivity_access_authorized",
        "callback_data_sha256",
    }
    if set(metadata) != required:
        raise FullSearchRuntimeError("callback-data schema differs")
    _verify_self_hash(metadata, "callback_data_sha256", "callback data")
    if not (
        metadata["schema_version"] == CALLBACK_DATA_SCHEMA
        and metadata["design_sha256"] == design.design_sha256
        and tuple(metadata["raw_vocabulary"]) == design.raw_vocabulary
        and metadata["development_only"] is True
        and metadata["final_connectivity_accessed"] is False
        and metadata["final_connectivity_access_authorized"] is False
    ):
        raise FullSearchRuntimeError("callback data differs from design or firewall")
    arrays: dict[int, np.ndarray] = {}
    rows = metadata["covariates"]
    if not isinstance(rows, list):
        raise FullSearchRuntimeError("callback covariate table is malformed")
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "spline_df", "file", "dtype", "shape", "file_sha256"
        }:
            raise FullSearchRuntimeError("callback covariate row differs from schema")
        relative = Path(str(row["file"]))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise FullSearchRuntimeError("callback covariate path is unsafe")
        path = directory / relative
        if path.is_symlink() or digest_bytes(path.read_bytes()) != row["file_sha256"]:
            raise FullSearchRuntimeError("callback covariate digest mismatch")
        try:
            value = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise FullSearchRuntimeError("callback covariate is unreadable") from error
        if (
            value.dtype != np.dtype(np.float64)
            or list(value.shape) != row["shape"]
            or value.ndim != 2
            or value.shape[0] != len(design.neuron_ids)
        ):
            raise FullSearchRuntimeError("callback covariate differs from metadata")
        spline_df = int(row["spline_df"])
        if spline_df in arrays:
            raise FullSearchRuntimeError("callback spline branch is duplicated")
        arrays[spline_df] = np.ascontiguousarray(value)
    if set(arrays) != set(map(int, policy.grammar_choices["nuisance_spline_df"])):
        raise FullSearchRuntimeError("callback spline branches are incomplete")
    settings = metadata["fit_settings"]
    if not isinstance(settings, dict) or set(settings) != {
        "integration_draws", "source_cv_folds"
    }:
        raise FullSearchRuntimeError("callback settings differ from schema")
    hierarchy = metadata["partner_hierarchy"]
    if not isinstance(hierarchy, dict) or not hierarchy:
        raise FullSearchRuntimeError("callback hierarchy is unavailable")
    inputs = NullTrialInputs(
        raw_vocabulary=design.raw_vocabulary,
        covariates_by_spline_df=arrays,
        partner_hierarchy=hierarchy,
        fit_settings=settings,
        artifact_root=artifact_root.resolve(),
        callback_data_sha256=str(metadata["callback_data_sha256"]),
    )
    return inputs, metadata


def _file_inventory(directory: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise FullSearchRuntimeError("runtime bundle may not contain symlinks")
        if path.is_file() and path.name != "runtime_bundle.json":
            result[path.relative_to(directory).as_posix()] = digest_bytes(path.read_bytes())
    return result


def materialize_runtime_bundle(
    *,
    output_dir: Path,
    policy: EpisodePolicy,
    design: FittedNullDesign,
    generators: Mapping[tuple[str, str], FittedPredictiveModel],
    covariates_by_spline_df: Mapping[int, np.ndarray],
    partner_hierarchy: Mapping[str, Mapping[str, Any]],
    fit_settings: Mapping[str, Any],
    controller_program: Mapping[str, Any],
    selection_attestation: Mapping[str, Any],
    null_generator_selection_sha256: str,
    development_role_manifest_sha256: str,
    upstream_procedure_lock_sha256: str,
    upstream_procedure_lock_file_sha256: str,
    resource_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically emit the complete provider bundle and procedure lock."""

    output_dir = _scoped_path(
        output_dir,
        root=EPISODE_OUTPUTS_ROOT,
        label="runtime bundle output",
    )
    upstream_procedure_lock_sha256 = _sha(
        upstream_procedure_lock_sha256, "upstream procedure lock"
    )
    upstream_procedure_lock_file_sha256 = _sha(
        upstream_procedure_lock_file_sha256, "upstream procedure-lock file"
    )
    if output_dir.exists():
        raise FullSearchRuntimeError(f"refusing to overwrite runtime bundle: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.with_name(f".{output_dir.name}.staging.{os.getpid()}")
    if stage.exists():
        raise FullSearchRuntimeError(f"stale runtime staging directory: {stage}")
    stage.mkdir()
    verify_null_controller_program(controller_program, policy=policy)
    verified_resource_plan = verify_resource_plan(resource_plan, policy=policy)
    if (
        verified_resource_plan["controller_program_sha256"]
        != controller_program["program_sha256"]
    ):
        raise FullSearchRuntimeError("resource plan belongs to another controller")
    null_generator_selection_sha256 = _sha(
        null_generator_selection_sha256, "null-generator selection"
    )
    selected = _verify_attestation(
        selection_attestation,
        controller_program=controller_program,
        role_sha256=development_role_manifest_sha256,
        upstream_procedure_lock_sha256=upstream_procedure_lock_sha256,
        null_generator_selection_sha256=null_generator_selection_sha256,
        policy=policy,
    )
    if int(selected["nuisance_spline_df"]) not in covariates_by_spline_df:
        raise FullSearchRuntimeError("selected fit lacks its covariate branch")
    model_manifest = write_model_state(
        stage / "model_state", design=design, generators=generators
    )
    callback_metadata = _write_callback_data(
        stage / "callback_data",
        design=design,
        covariates=covariates_by_spline_df,
        hierarchy=partner_hierarchy,
        settings=fit_settings,
    )
    inputs, _ = _load_callback_data(
        stage / "callback_data",
        design=design,
        policy=policy,
        artifact_root=output_dir.parent / "runtime_artifacts",
    )
    callback = ProductionFullSearchCallback(policy=policy, inputs=inputs)
    expansion = TypePooledExpansionPlan.build(design)
    binding = null_seed_binding(
        design, generators, expansion,
        selection_sha256=SELECTION_PROGRAM_SHA256,
    )
    benchmark_receipt = verified_resource_plan["timing_basis"][
        "runtime_benchmark_receipt"
    ]
    if not (
        benchmark_receipt["design_sha256"] == design.design_sha256
        and benchmark_receipt["generator_manifest_sha256"]
        == binding["generator_manifest_sha256"]
        and benchmark_receipt["expansion_plan_sha256"] == expansion.plan_sha256
        and benchmark_receipt["callback_identity_sha256"]
        == digest_object(callback.identity())
    ):
        raise FullSearchRuntimeError(
            "runtime benchmark does not describe the bundled production path"
        )
    seed_material = digest_object({
        "protocol": "ep12.full_search_null_master_seed.v1",
        "binding": binding,
        "controller_program_sha256": controller_program["program_sha256"],
        "attestation_sha256": selection_attestation["attestation_sha256"],
        "upstream_procedure_lock_sha256": upstream_procedure_lock_sha256,
        "upstream_procedure_lock_file_sha256": (
            upstream_procedure_lock_file_sha256
        ),
    })
    seed_manifest = freeze_null_seed_manifest(
        master_seed=int.from_bytes(bytes.fromhex(seed_material)[:8], "big"),
        binding=binding,
    )
    procedure_lock = freeze_procedure_lock(
        policy=policy,
        controller_program=controller_program,
        design=design,
        generators=generators,
        expansion_plan=expansion,
        seed_manifest=seed_manifest,
        selection_sha256=SELECTION_PROGRAM_SHA256,
        development_role_manifest_sha256=development_role_manifest_sha256,
        callback_identity=callback.identity(),
    )
    atomic_json(stage / "null_controller_program.json", dict(controller_program))
    atomic_json(stage / "selected_fit_attestation.json", dict(selection_attestation))
    atomic_json(stage / "null_seed_manifest.json", seed_manifest)
    atomic_json(stage / "procedure_lock.json", procedure_lock)
    atomic_json(stage / "resource_plan.json", verified_resource_plan)
    atomic_json(
        stage / "runtime_benchmark_receipt.json",
        verified_resource_plan["timing_basis"]["runtime_benchmark_receipt"],
    )
    core = {
        "schema_version": BUNDLE_SCHEMA,
        "policy_sha256": policy.policy_hash,
        "upstream_procedure_lock_sha256": upstream_procedure_lock_sha256,
        "upstream_procedure_lock_file_sha256": (
            upstream_procedure_lock_file_sha256
        ),
        "development_role_manifest_sha256": _sha(
            development_role_manifest_sha256, "role manifest"
        ),
        "selection_program_sha256": SELECTION_PROGRAM_SHA256,
        "selected_configuration_sha256": selection_attestation[
            "selected_configuration_sha256"
        ],
        "model_state_sha256": model_manifest["state_sha256"],
        "callback_data_sha256": callback_metadata["callback_data_sha256"],
        "expansion_plan_sha256": expansion.plan_sha256,
        "controller_program_sha256": controller_program["program_sha256"],
        "null_generator_selection_sha256": null_generator_selection_sha256,
        "seed_manifest_sha256": digest_object(seed_manifest),
        "procedure_lock_sha256": procedure_lock["procedure_lock_sha256"],
        "resource_plan_sha256": verified_resource_plan["plan_sha256"],
        "files": _file_inventory(stage),
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    manifest = {**core, "bundle_sha256": digest_object(core)}
    atomic_json(stage / "runtime_bundle.json", manifest)
    os.replace(stage, output_dir)
    descriptor = os.open(str(output_dir.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest

def load_runtime_bundle(
    bundle_dir: Path,
    *,
    policy: EpisodePolicy,
    artifact_root: Path,
    expected_bundle_sha256: str,
    upstream_procedure_lock_sha256: str,
    upstream_procedure_lock_file_sha256: str,
) -> FullSearchNullRuntime:
    """Authenticate a bundle and return the only production runtime object."""

    bundle_dir = _runtime_bundle_path(bundle_dir, label="runtime bundle")
    artifact_root = _scoped_path(
        artifact_root,
        root=EPISODE_SCRATCH_ROOT,
        label="full-null work root",
    )
    expected_bundle_sha256 = _sha(expected_bundle_sha256, "expected bundle")
    upstream_procedure_lock_sha256 = _sha(
        upstream_procedure_lock_sha256, "upstream procedure lock"
    )
    upstream_procedure_lock_file_sha256 = _sha(
        upstream_procedure_lock_file_sha256, "upstream procedure-lock file"
    )
    manifest = _read_json(bundle_dir / "runtime_bundle.json", "runtime bundle")
    required = {
        "schema_version", "policy_sha256", "upstream_procedure_lock_sha256",
        "upstream_procedure_lock_file_sha256",
        "development_role_manifest_sha256",
        "selection_program_sha256", "selected_configuration_sha256",
        "model_state_sha256", "callback_data_sha256", "expansion_plan_sha256",
        "controller_program_sha256", "null_generator_selection_sha256",
        "seed_manifest_sha256", "procedure_lock_sha256",
        "resource_plan_sha256",
        "files", "development_only", "final_connectivity_accessed",
        "final_connectivity_access_authorized", "bundle_sha256",
    }
    if set(manifest) != required:
        raise FullSearchRuntimeError("runtime-bundle schema differs")
    _verify_self_hash(manifest, "bundle_sha256", "runtime bundle")
    if not (
        manifest["schema_version"] == BUNDLE_SCHEMA
        and manifest["policy_sha256"] == policy.policy_hash
        and manifest["bundle_sha256"] == expected_bundle_sha256
        and manifest["upstream_procedure_lock_sha256"]
        == upstream_procedure_lock_sha256
        and manifest["upstream_procedure_lock_file_sha256"]
        == upstream_procedure_lock_file_sha256
        and manifest["selection_program_sha256"] == SELECTION_PROGRAM_SHA256
        and manifest["development_only"] is True
        and manifest["final_connectivity_accessed"] is False
        and manifest["final_connectivity_access_authorized"] is False
    ):
        raise FullSearchRuntimeError("runtime bundle differs from policy or firewall")
    if manifest["files"] != _file_inventory(bundle_dir):
        raise FullSearchRuntimeError("runtime bundle file inventory changed")
    try:
        design, generators, model_manifest = load_model_state(
            bundle_dir / "model_state"
        )
    except ModelArtifactError as error:
        raise FullSearchRuntimeError("fitted model state failed authentication") from error
    if model_manifest["state_sha256"] != manifest["model_state_sha256"]:
        raise FullSearchRuntimeError("model-state identity differs from bundle")
    controller_program = _read_json(
        bundle_dir / "null_controller_program.json", "null controller program"
    )
    verify_null_controller_program(controller_program, policy=policy)
    if controller_program["program_sha256"] != manifest["controller_program_sha256"]:
        raise FullSearchRuntimeError("null controller program differs from bundle")
    resource_plan = verify_resource_plan(
        _read_json(bundle_dir / "resource_plan.json", "resource plan"),
        policy=policy,
    )
    benchmark_receipt = verify_runtime_benchmark_receipt(
        _read_json(
            bundle_dir / "runtime_benchmark_receipt.json",
            "runtime benchmark receipt",
        )
    )
    if not (
        resource_plan["plan_sha256"] == manifest["resource_plan_sha256"]
        and resource_plan["controller_program_sha256"]
        == controller_program["program_sha256"]
        and benchmark_receipt
        == resource_plan["timing_basis"]["runtime_benchmark_receipt"]
    ):
        raise FullSearchRuntimeError("resource plan differs from runtime bundle")
    role_sha256 = _sha(
        manifest["development_role_manifest_sha256"], "role manifest"
    )
    attestation = _read_json(
        bundle_dir / "selected_fit_attestation.json", "selected-fit attestation"
    )
    _verify_attestation(
        attestation,
        controller_program=controller_program,
        role_sha256=role_sha256,
        upstream_procedure_lock_sha256=upstream_procedure_lock_sha256,
        null_generator_selection_sha256=manifest[
            "null_generator_selection_sha256"
        ],
        policy=policy,
    )
    if (
        attestation["selected_configuration_sha256"]
        != manifest["selected_configuration_sha256"]
    ):
        raise FullSearchRuntimeError("selected configuration differs from bundle")
    inputs, callback_metadata = _load_callback_data(
        bundle_dir / "callback_data",
        design=design,
        policy=policy,
        artifact_root=artifact_root,
    )
    if callback_metadata["callback_data_sha256"] != manifest["callback_data_sha256"]:
        raise FullSearchRuntimeError("callback-data identity differs from bundle")
    callback = ProductionFullSearchCallback(policy=policy, inputs=inputs)
    expansion = TypePooledExpansionPlan.build(design)
    if expansion.plan_sha256 != manifest["expansion_plan_sha256"]:
        raise FullSearchRuntimeError("reconstructed expansion plan differs from bundle")
    seed_manifest = _read_json(
        bundle_dir / "null_seed_manifest.json", "null-seed manifest"
    )
    if digest_object(seed_manifest) != manifest["seed_manifest_sha256"]:
        raise FullSearchRuntimeError("null-seed manifest differs from bundle")
    binding = null_seed_binding(
        design,
        generators,
        expansion,
        selection_sha256=SELECTION_PROGRAM_SHA256,
    )
    if not (
        benchmark_receipt["design_sha256"] == design.design_sha256
        and benchmark_receipt["generator_manifest_sha256"]
        == binding["generator_manifest_sha256"]
        and benchmark_receipt["expansion_plan_sha256"] == expansion.plan_sha256
        and benchmark_receipt["callback_identity_sha256"]
        == digest_object(callback.identity())
    ):
        raise FullSearchRuntimeError(
            "runtime benchmark does not describe the loaded production path"
        )
    seed_material = digest_object({
        "protocol": "ep12.full_search_null_master_seed.v1",
        "binding": binding,
        "controller_program_sha256": controller_program["program_sha256"],
        "attestation_sha256": attestation["attestation_sha256"],
        "upstream_procedure_lock_sha256": upstream_procedure_lock_sha256,
        "upstream_procedure_lock_file_sha256": (
            upstream_procedure_lock_file_sha256
        ),
    })
    expected_seed_manifest = freeze_null_seed_manifest(
        master_seed=int.from_bytes(bytes.fromhex(seed_material)[:8], "big"),
        binding=binding,
    )
    if seed_manifest != expected_seed_manifest:
        raise FullSearchRuntimeError(
            "null-seed manifest does not derive from the locked controller program"
        )
    procedure_lock = _read_json(bundle_dir / "procedure_lock.json", "procedure lock")
    recomputed = freeze_procedure_lock(
        policy=policy,
        controller_program=controller_program,
        design=design,
        generators=generators,
        expansion_plan=expansion,
        seed_manifest=seed_manifest,
        selection_sha256=SELECTION_PROGRAM_SHA256,
        development_role_manifest_sha256=role_sha256,
        callback_identity=callback.identity(),
    )
    if (
        procedure_lock != recomputed
        or procedure_lock["procedure_lock_sha256"]
        != manifest["procedure_lock_sha256"]
    ):
        raise FullSearchRuntimeError("procedure lock is incomplete or changed")
    return FullSearchNullRuntime(
        design=design,
        generators=generators,
        expansion_plan=expansion,
        seed_manifest=seed_manifest,
        selection_sha256=SELECTION_PROGRAM_SHA256,
        development_role_manifest_sha256=role_sha256,
        controller_program=controller_program,
        procedure_lock=procedure_lock,
        callback=callback,
    )


def verify_scratch_stage_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the deterministic receipt for one durable-to-Scratch copy."""

    required = {
        "schema_version", "durable_bundle_dir", "scratch_bundle_dir",
        "bundle_sha256", "durable_manifest_file_sha256",
        "scratch_manifest_file_sha256", "file_inventory_sha256",
        "file_count", "operation", "development_only",
        "final_connectivity_accessed", "receipt_sha256",
    }
    receipt = _json_copy(value)
    if set(receipt) != required:
        raise FullSearchRuntimeError("scratch-stage receipt schema differs")
    _verify_self_hash(receipt, "receipt_sha256", "scratch-stage receipt")
    for field in (
        "bundle_sha256", "durable_manifest_file_sha256",
        "scratch_manifest_file_sha256", "file_inventory_sha256",
    ):
        _sha(receipt[field], field.replace("_", " "))
    if not (
        receipt["schema_version"] == SCRATCH_STAGE_SCHEMA
        and Path(str(receipt["durable_bundle_dir"])).is_absolute()
        and Path(str(receipt["scratch_bundle_dir"])).is_absolute()
        and int(receipt["file_count"]) > 0
        and receipt["operation"]
        == "atomic_copy_once_or_verify_identical_existing_stage"
        and receipt["development_only"] is True
        and receipt["final_connectivity_accessed"] is False
        and receipt["durable_manifest_file_sha256"]
        == receipt["scratch_manifest_file_sha256"]
    ):
        raise FullSearchRuntimeError("scratch-stage receipt violates its protocol")
    return receipt


def stage_runtime_bundle(
    *,
    durable_bundle_dir: Path,
    scratch_bundle_dir: Path,
    policy: EpisodePolicy,
    expected_bundle_sha256: str,
    upstream_procedure_lock_sha256: str,
    upstream_procedure_lock_file_sha256: str,
) -> dict[str, Any]:
    """Atomically copy one authenticated executable bundle from Oak to Scratch."""

    durable = _scoped_path(
        durable_bundle_dir,
        root=EPISODE_OUTPUTS_ROOT,
        label="durable runtime bundle",
        require_directory=True,
    )
    scratch = _scoped_path(
        scratch_bundle_dir,
        root=EPISODE_SCRATCH_ROOT,
        label="scratch runtime bundle",
    )
    expected_bundle_sha256 = _sha(expected_bundle_sha256, "expected bundle")
    semantic_lock_sha256 = _sha(
        upstream_procedure_lock_sha256, "upstream procedure lock"
    )
    lock_file_sha256 = _sha(
        upstream_procedure_lock_file_sha256, "upstream procedure-lock file"
    )
    durable_manifest_path = durable / "runtime_bundle.json"
    durable_manifest = _read_json(durable_manifest_path, "durable runtime bundle")
    _verify_self_hash(durable_manifest, "bundle_sha256", "durable runtime bundle")
    if not (
        durable_manifest.get("schema_version") == BUNDLE_SCHEMA
        and durable_manifest.get("bundle_sha256") == expected_bundle_sha256
        and durable_manifest.get("upstream_procedure_lock_sha256")
        == semantic_lock_sha256
        and durable_manifest.get("upstream_procedure_lock_file_sha256")
        == lock_file_sha256
        and durable_manifest.get("development_only") is True
        and durable_manifest.get("final_connectivity_accessed") is False
    ):
        raise FullSearchRuntimeError("durable runtime bundle differs from staging lock")
    if durable_manifest.get("files") != _file_inventory(durable):
        raise FullSearchRuntimeError("durable runtime bundle inventory changed")

    def verify_stage(path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            raise FullSearchRuntimeError(
                "scratch runtime stage is not an ordinary directory"
            )
        staged_manifest_path = path / "runtime_bundle.json"
        if staged_manifest_path.read_bytes() != durable_manifest_path.read_bytes():
            raise FullSearchRuntimeError("scratch runtime manifest differs from durable")
        if _file_inventory(path) != durable_manifest["files"]:
            raise FullSearchRuntimeError("scratch runtime file inventory differs")
        verification_artifacts = scratch.parent / (
            f".{scratch.name}.stage_verification_artifacts"
        )
        load_runtime_bundle(
            path,
            policy=policy,
            artifact_root=verification_artifacts,
            expected_bundle_sha256=expected_bundle_sha256,
            upstream_procedure_lock_sha256=semantic_lock_sha256,
            upstream_procedure_lock_file_sha256=lock_file_sha256,
        )

    if scratch.exists():
        verify_stage(scratch)
    else:
        scratch.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        owned_identity: tuple[int, int] | None = None
        for _ in range(8):
            candidate = scratch.with_name(
                f".{scratch.name}.staging.{expected_bundle_sha256[:16]}."
                f"{os.getpid()}.{secrets.token_hex(16)}"
            )
            try:
                os.mkdir(candidate, 0o770)
            except FileExistsError:
                continue
            state = os.lstat(candidate)
            temporary = candidate
            owned_identity = (int(state.st_dev), int(state.st_ino))
            break
        if temporary is None or owned_identity is None:
            raise FullSearchRuntimeError(
                "cannot allocate a collision-resistant scratch staging directory"
            )
        try:
            shutil.copytree(
                durable, temporary, symlinks=True, dirs_exist_ok=True
            )
            verify_stage(temporary)
            try:
                os.rename(temporary, scratch)
            except OSError as error:
                if error.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not (
                    scratch.exists() or scratch.is_symlink()
                ):
                    raise
                verify_stage(scratch)
            else:
                temporary = None
                owned_identity = None
            descriptor = os.open(str(scratch.parent), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            if temporary is not None and os.path.lexists(os.fspath(temporary)):
                state = os.lstat(temporary)
                current_identity = (int(state.st_dev), int(state.st_ino))
                if (
                    temporary.is_symlink()
                    or not temporary.is_dir()
                    or current_identity != owned_identity
                ):
                    raise FullSearchRuntimeError(
                        "scratch staging ownership changed; refusing cleanup"
                    )
                shutil.rmtree(temporary)
    durable_manifest_sha256 = digest_bytes(durable_manifest_path.read_bytes())
    scratch_manifest_sha256 = digest_bytes(
        (scratch / "runtime_bundle.json").read_bytes()
    )
    core = {
        "schema_version": SCRATCH_STAGE_SCHEMA,
        "durable_bundle_dir": str(durable),
        "scratch_bundle_dir": str(scratch),
        "bundle_sha256": expected_bundle_sha256,
        "durable_manifest_file_sha256": durable_manifest_sha256,
        "scratch_manifest_file_sha256": scratch_manifest_sha256,
        "file_inventory_sha256": digest_object(durable_manifest["files"]),
        "file_count": len(durable_manifest["files"]),
        "operation": "atomic_copy_once_or_verify_identical_existing_stage",
        "development_only": True,
        "final_connectivity_accessed": False,
    }
    return verify_scratch_stage_receipt(
        {**core, "receipt_sha256": digest_object(core)}
    )


def load_runtime_from_environment() -> FullSearchNullRuntime:
    """Strict environment provider used by the single-task batch wrapper."""

    names = (
        "EP12_FULL_NULL_BUNDLE",
        "EP12_FULL_NULL_BUNDLE_SHA256",
        "EP12_FULL_NULL_WORK_ROOT",
        "EP12_FULL_NULL_DURABLE_ROOT",
        "EP12_PRE_NULL_PROCEDURE_LOCK_PATH",
        "EP12_PRE_NULL_PROCEDURE_LOCK_SHA256",
        "EP12_PRE_NULL_PROCEDURE_LOCK_FILE_SHA256",
    )
    values = {name: os.environ.get(name) for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise FullSearchRuntimeError(
            "required runtime environment is missing: " + ", ".join(missing)
        )
    actual_episode_root = Path(__file__).resolve().parents[3]
    if actual_episode_root != EPISODE_ROOT.resolve(strict=False):
        raise FullSearchRuntimeError("runtime provider is outside exact EP12 root")
    expected_policy = (EPISODE_ROOT / "SEARCH_POLICY.yaml").resolve()
    raw_policy = Path(os.environ.get("EP12_SEARCH_POLICY", str(expected_policy)))
    policy_path = _scoped_path(
        raw_policy,
        root=EPISODE_ROOT,
        label="search policy",
        require_file=True,
    )
    if policy_path != expected_policy:
        raise FullSearchRuntimeError(
            "runtime provider accepts only EP12 SEARCH_POLICY.yaml"
        )
    policy = EpisodePolicy.load(policy_path)
    semantic_lock_sha256 = _sha(
        values["EP12_PRE_NULL_PROCEDURE_LOCK_SHA256"],
        "environment upstream procedure lock",
    )
    lock_file_sha256 = _sha(
        values["EP12_PRE_NULL_PROCEDURE_LOCK_FILE_SHA256"],
        "environment upstream procedure-lock file",
    )
    lock_path = _scoped_path(
        Path(str(values["EP12_PRE_NULL_PROCEDURE_LOCK_PATH"])),
        root=EPISODE_OUTPUTS_ROOT,
        label="upstream procedure-lock path",
        require_file=True,
    )
    try:
        lock_bytes = lock_path.read_bytes()
    except OSError as error:
        raise FullSearchRuntimeError("upstream procedure-lock file is unreadable") from error
    if digest_bytes(lock_bytes) != lock_file_sha256:
        raise FullSearchRuntimeError("upstream procedure-lock file digest mismatch")
    upstream_lock = _read_json(lock_path, "upstream procedure lock")
    try:
        verify_procedure_lock(upstream_lock, policy=policy)
    except ProcedureLockError as error:
        raise FullSearchRuntimeError("upstream procedure lock is invalid") from error
    if upstream_lock.get("procedure_lock_sha256") != semantic_lock_sha256:
        raise FullSearchRuntimeError("upstream procedure-lock semantic digest mismatch")
    if not (
        upstream_lock.get("procedure_locked") is True
        and upstream_lock.get("null_eligible") is True
        and upstream_lock.get("final_connectivity_accessed") is False
        and upstream_lock.get("final_connectivity_access_authorized") is False
    ):
        raise FullSearchRuntimeError("upstream procedure lock does not authorize null replay")
    bundle_path = _runtime_bundle_path(
        Path(str(values["EP12_FULL_NULL_BUNDLE"])), label="runtime bundle"
    )
    work_root = _scoped_path(
        Path(str(values["EP12_FULL_NULL_WORK_ROOT"])),
        root=EPISODE_SCRATCH_ROOT,
        label="full-null work root",
    )
    durable_root = _scoped_path(
        Path(str(values["EP12_FULL_NULL_DURABLE_ROOT"])),
        root=EPISODE_OUTPUTS_ROOT,
        label="full-null durable root",
    )
    if (
        durable_root == bundle_path
        or durable_root in bundle_path.parents
        or bundle_path in durable_root.parents
        or durable_root == lock_path
    ):
        raise FullSearchRuntimeError(
            "full-null durable root overlaps an immutable runtime input"
        )
    bundle_manifest = _read_json(
        bundle_path / "runtime_bundle.json", "runtime bundle"
    )
    upstream_binding = _mapping(
        upstream_lock.get("pre_null_input_binding"), "pre-null input binding"
    )
    selected = _mapping(
        upstream_lock.get("selected_comparison"), "selected comparison"
    )
    if not (
        bundle_manifest.get("upstream_procedure_lock_sha256")
        == semantic_lock_sha256
        and bundle_manifest.get("upstream_procedure_lock_file_sha256")
        == lock_file_sha256
        and bundle_manifest.get("controller_program_sha256")
        == upstream_lock.get("null_controller_program_sha256")
        and bundle_manifest.get("null_generator_selection_sha256")
        == upstream_lock.get("null_generator_selection_sha256")
        and bundle_manifest.get("selected_configuration_sha256")
        == selected.get("configuration_sha256")
        and bundle_manifest.get("callback_data_sha256")
        == upstream_binding.get("callback_data_sha256")
        and bundle_manifest.get("model_state_sha256")
        == upstream_binding.get("model_state_sha256")
    ):
        raise FullSearchRuntimeError(
            "runtime bundle is not semantically derived from the upstream lock"
        )
    return replace(
        load_runtime_bundle(
            bundle_path,
            policy=policy,
            artifact_root=work_root,
            expected_bundle_sha256=str(values["EP12_FULL_NULL_BUNDLE_SHA256"]),
            upstream_procedure_lock_sha256=semantic_lock_sha256,
            upstream_procedure_lock_file_sha256=lock_file_sha256,
        ),
        durable_output_root=durable_root,
    )
