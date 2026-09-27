"""Authenticate the pre-null lock and materialize the executable null bundle.

This is the sole bridge from the durable pre-null input bundle to the
development-only fitted-null runtime.  It never opens final-role connectivity
and refuses any controller that replays the observed realized schedule.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .adaptive_search import validate_development_configuration
from .falsifier_controls import (
    endpoint_branch_inventory,
    nuisance_block_columns,
    partner_family_inventory,
)
from .fitted_null import (
    TypePooledExpansionPlan,
    freeze_null_seed_manifest,
    null_seed_binding,
    sample_fitted_null,
)
from .full_search_runtime import (
    CUMULATIVE_RUNTIME_ENFORCEMENT,
    EPISODE_OUTPUTS_ROOT,
    EPISODE_SCRATCH_ROOT,
    PER_TRIAL_RUNTIME_ENFORCEMENT,
    RUNTIME_BENCHMARK_ROLE,
    RUNTIME_BENCHMARK_SCOPE,
    TIMING_CORPUS_KIND,
    WALL_BUDGET_EXHAUSTION_HANDLING,
    _load_callback_data,
    build_selected_fit_binding,
    load_runtime_bundle,
    materialize_runtime_bundle,
    stage_runtime_bundle,
    validate_runtime_benchmark,
    verify_resource_plan,
)
from .model_artifacts import load_model_state
from .null_controller import verify_null_controller_program
from .null_trial_runner import ProductionFullSearchCallback, SELECTION_PROGRAM_SHA256
from .policy import EpisodePolicy, digest_object
from .pre_null_materializer import load_observed_trial_status, verify_pre_null_inputs
from .procedure_lock import verify_procedure_lock


class PostLockRuntimeMaterializerError(RuntimeError):
    """The post-lock runtime cannot be derived from authenticated inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostLockRuntimeMaterializerError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PostLockRuntimeMaterializerError(
            f"cannot read {label}: {path}"
        ) from error
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _inside(path: Path, root: Path, label: str, *, exists: bool = True) -> Path:
    raw = Path(path)
    _require(raw.is_absolute() and ".." not in raw.parts, f"unsafe {label} path")
    lexical = Path(raw.absolute())
    resolved = raw.resolve(strict=False)
    root_resolved = root.resolve(strict=False)
    _require(lexical == resolved, f"{label} may not traverse a symbolic link")
    try:
        relative = resolved.relative_to(root_resolved)
    except ValueError as error:
        raise PostLockRuntimeMaterializerError(
            f"{label} must stay under {root_resolved}"
        ) from error
    _require(bool(relative.parts), f"{label} may not equal its root")
    if exists:
        _require(resolved.exists(), f"{label} does not exist: {resolved}")
    return resolved


def _selected_record(
    archive: Mapping[str, Any], lock: Mapping[str, Any]
) -> dict[str, Any]:
    selected = lock.get("selected_comparison")
    _require(isinstance(selected, Mapping), "procedure lock lacks selected comparison")
    matches = [
        dict(record)
        for record in archive.get("records", [])
        if isinstance(record, Mapping)
        and record.get("trial_id") == selected.get("trial_id")
    ]
    _require(len(matches) == 1, "selected comparison is absent or duplicated")
    record = matches[0]
    _require(
        record.get("configuration") == selected.get("configuration")
        and record.get("configuration_sha256")
        == selected.get("configuration_sha256")
        and record.get("scientific_signature")
        == selected.get("scientific_signature")
        and record.get("result_table_sha256")
        == selected.get("result_table_sha256"),
        "selected comparison differs between archive and procedure lock",
    )
    return record


def _development_role_identity(path: Path) -> str:
    """Validate and identify the frozen whole-type development/final split."""

    manifest = _read_json(path, "development-role manifest")
    rows = manifest.get("rows")
    _require(isinstance(rows, list), "development-role rows are unavailable")
    compact_rows: list[dict[str, Any]] = []
    provider_types: set[str] = set()
    role_counts = {"development": 0, "final": 0}
    for raw in rows:
        _require(isinstance(raw, Mapping), "development-role row is malformed")
        provider_type = str(raw.get("provider_type", ""))
        role = str(raw.get("role", ""))
        _require(
            provider_type
            and provider_type not in provider_types
            and role in role_counts,
            "development-role split duplicates or mislabels a provider type",
        )
        provider_types.add(provider_type)
        role_counts[role] += 1
        compact_rows.append(
            {
                "provider_type": provider_type,
                "role": role,
                "left_neurons": int(raw.get("left_neurons", -1)),
                "right_neurons": int(raw.get("right_neurons", -1)),
                "minimum_side_count": int(raw.get("minimum_side_count", -1)),
                "minimum_side_count_decile": int(
                    raw.get("minimum_side_count_decile", -1)
                ),
            }
        )
    _require(
        manifest.get("assignment_unit") == "whole_provider_type"
        and manifest.get("real_connectivity_used") is False
        and int(manifest.get("eligible_type_count", -1)) == len(rows) == 1020
        and int(manifest.get("development_type_count", -1))
        == role_counts["development"] == 816
        and int(manifest.get("final_type_count", -1))
        == role_counts["final"] == 204,
        "development-role manifest does not preserve the frozen 816/204 split",
    )
    return digest_object(
        {
            "assignment_unit": manifest["assignment_unit"],
            "eligibility": manifest.get("eligibility"),
            "split": manifest.get("split"),
            "rows": compact_rows,
        }
    )


def freeze_resource_plan(
    *,
    policy: EpisodePolicy,
    controller_program: Mapping[str, Any],
    raw_vocabulary: Sequence[str],
    raw_counts: Any,
    partner_hierarchy: Mapping[str, Mapping[str, Any]],
    covariates_by_spline_df: Mapping[int, Any],
    timing_basis: Mapping[str, Any],
    workers: int = 32,
) -> dict[str, Any]:
    """Record bounded branches and measured engineering projections."""

    program = verify_null_controller_program(controller_program, policy=policy)
    _require(
        workers == policy.cpu_cores_per_trial_maximum == 32,
        "resource worker count violates the frozen per-trial CPU limit",
    )
    families = partner_family_inventory(partner_hierarchy, raw_vocabulary)
    endpoints = endpoint_branch_inventory(raw_vocabulary, raw_counts)
    block_counts = []
    for spline_df, covariates in covariates_by_spline_df.items():
        block_counts.append(
            len(
                nuisance_block_columns(
                    spline_df=int(spline_df),
                    design_width=int(covariates.shape[1]),
                )
            )
        )
    _require(block_counts, "resource plan lacks covariate branches")
    family_count = len(families)
    endpoint_count = len(endpoints)
    inventory = {
        "strength_and_margin_preserving_graph_randomization": 1,
        "cross_side_component_alignment_permutation": 1,
        "shuffled_partner_and_valid_side_controls": 1,
        "binary_weighted_vocabulary_and_rank_sensitivities": (
            2
            + len(policy.grammar_choices["partner_vocabulary"])
            * len(policy.grammar_choices["rank"])
        ),
        "nuisance_and_partner_block_ablations": (
            1 + len(policy.nuisance_terms) + family_count
        ),
        "capacity_matched_noise_controls": 2,
        "leave_one_development_type_and_partner_family_influence": (
            1 + family_count
        ),
        "few_type_and_high_strength_neuron_concentration_check": 1,
        "status_endpoint_anatomy_and_missingness_checks": (
            1 + max(block_counts) + endpoint_count
        ),
    }

    def fit_total(last_ordinal: int) -> int:
        total = policy.minimum_valid_trials
        for slot in program["post36_slots"][: last_ordinal - policy.minimum_valid_trials]:
            falsifier = slot["falsifier_id"]
            total += 1 if falsifier is None else inventory[str(falsifier)]
        return total

    minimum_fits = fit_total(52)
    maximum_fits = fit_total(policy.maximum_valid_trials)
    _require(
        minimum_fits == 308 and maximum_fits == 690,
        "controller branch inventory differs from the exact 308/690 search",
    )
    benchmark = validate_runtime_benchmark(timing_basis["runtime_benchmark"])
    mean_seconds = float(timing_basis["observed_serial_mean_wall_seconds"])
    maximum_seconds = float(timing_basis["observed_serial_max_wall_seconds"])
    mean_busy_seconds = float(
        timing_basis["observed_serial_mean_cpu_seconds_per_branch"]
    )
    busy_seconds = float(
        timing_basis["observed_serial_max_cpu_seconds_per_branch"]
    )
    _require(
        math.isfinite(mean_seconds)
        and math.isfinite(maximum_seconds)
        and math.isfinite(mean_busy_seconds)
        and math.isfinite(busy_seconds)
        and 0.0 < mean_seconds <= maximum_seconds,
        "resource timing basis is invalid",
    )
    _require(
        0.0 < mean_busy_seconds <= busy_seconds,
        "resource busy-time basis is invalid",
    )
    conservative_seconds = max(
        float(benchmark["measured_parallel_max_wall_seconds"]),
        maximum_seconds / workers,
    )
    maximum_control_branches = max(map(int, inventory.values()))
    minimum_wall_hours = minimum_fits * conservative_seconds / 3600.0
    maximum_wall_hours = maximum_fits * conservative_seconds / 3600.0
    minimum_busy_hours = minimum_fits * busy_seconds / 3600.0
    maximum_busy_hours = maximum_fits * busy_seconds / 3600.0
    program_search_count = 1 + policy.null_replicates
    core = {
        "policy_sha256": policy.policy_hash,
        "controller_program_sha256": program["program_sha256"],
        "requested_cpus_per_replicate": workers,
        "provider_type_workers": workers,
        "provider_type_parallelism": (
            "ordered_deterministic_thread_pool_within_complete_branch"
        ),
        "control_branch_execution": "sequential_no_nested_pool",
        "branch_fit_inventory": inventory,
        "minimum_search_last_trial_ordinal": 52,
        "maximum_search_last_trial_ordinal": policy.maximum_valid_trials,
        "complete_T_U_M_fit_count_scope": (
            "per_observed_or_synthetic_null_search"
        ),
        "minimum_complete_T_U_M_fit_count": minimum_fits,
        "maximum_complete_T_U_M_fit_count": maximum_fits,
        "minimum_model_family_fit_count": minimum_fits * len(policy.model_triplet),
        "maximum_model_family_fit_count": maximum_fits * len(policy.model_triplet),
        "timing_basis": dict(timing_basis),
        "conservative_seconds_per_complete_T_U_M_branch": conservative_seconds,
        "projected_minimum_replicate_wall_hours": minimum_wall_hours,
        "projected_maximum_replicate_wall_hours": maximum_wall_hours,
        "projected_minimum_replicate_allocated_core_hours": (
            minimum_wall_hours * workers
        ),
        "projected_maximum_replicate_allocated_core_hours": (
            maximum_wall_hours * workers
        ),
        "projected_minimum_replicate_busy_core_hours": minimum_busy_hours,
        "projected_maximum_replicate_busy_core_hours": maximum_busy_hours,
        "program_search_count": program_search_count,
        "observed_search_count": 1,
        "synthetic_null_search_count": policy.null_replicates,
        "projected_minimum_program_allocated_core_hours": (
            minimum_wall_hours * workers * program_search_count
        ),
        "projected_maximum_program_allocated_core_hours": (
            maximum_wall_hours * workers * program_search_count
        ),
        "projected_minimum_program_busy_core_hours": (
            minimum_busy_hours * program_search_count
        ),
        "projected_maximum_program_busy_core_hours": (
            maximum_busy_hours * program_search_count
        ),
        "maximum_complete_T_U_M_branches_in_one_control_trial": (
            maximum_control_branches
        ),
        "projected_maximum_control_trial_wall_hours": (
            maximum_control_branches * conservative_seconds / 3600.0
        ),
        "per_trial_wall_hours_limit": policy.wall_hours_per_trial_maximum,
        "overall_wall_hours_limit": float(
            policy.raw["budgets"]["wall_clock_hour_ceiling"]
        ),
        "runtime_benchmark_scope": RUNTIME_BENCHMARK_SCOPE,
        "runtime_benchmark_role": RUNTIME_BENCHMARK_ROLE,
        "projection_is_upper_bound": False,
        "actual_per_trial_runtime_enforcement": (
            PER_TRIAL_RUNTIME_ENFORCEMENT
        ),
        "actual_cumulative_runtime_enforcement": (
            CUMULATIVE_RUNTIME_ENFORCEMENT
        ),
        "wall_budget_exhaustion_handling": (
            WALL_BUDGET_EXHAUSTION_HANDLING
        ),
        "cpu_core_hour_ceiling_enforced": False,
        "checkpoint_resume": "write_once_checkpoint_and_idempotent_resume",
        "array_replicates": policy.null_replicates,
        "gpu_count": 0,
        "scientific_thresholds_added": False,
        "development_only": True,
        "final_connectivity_accessed": False,
    }
    return verify_resource_plan(core, policy=policy)


def _timing_evidence(
    max_timing_path: Path,
    *,
    run_root: Path,
    expected_trial_ids: Sequence[str],
    policy: EpisodePolicy,
    controller_program: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read all 36 structurally complete observed timings without outcomes."""

    run_root = _inside(run_root, EPISODE_OUTPUTS_ROOT, "observed run root")
    max_timing_path = _inside(
        max_timing_path,
        run_root,
        "maximum timing report",
    )
    observed_status = load_observed_trial_status(
        run_root=run_root,
        policy=policy,
        outputs_root=EPISODE_OUTPUTS_ROOT,
    )
    rows_by_id = observed_status.get("rows_by_id")
    _require(
        isinstance(rows_by_id, Mapping),
        "observed timing corpus lacks structurally complete trial rows",
    )
    trial_ids = [str(value) for value in expected_trial_ids]
    _require(
        len(trial_ids) == len(set(trial_ids)) == policy.minimum_valid_trials == 36
        and set(rows_by_id) == set(trial_ids),
        "observed timing corpus differs from the exact initial trial archive",
    )

    def positive_seconds(value: Any, label: str) -> float:
        try:
            seconds = float(value)
        except (TypeError, ValueError) as error:
            raise PostLockRuntimeMaterializerError(
                f"{label} is not numeric"
            ) from error
        _require(
            math.isfinite(seconds) and seconds > 0.0,
            f"{label} must be positive and finite",
        )
        return seconds

    timing_rows: list[dict[str, Any]] = []
    report_paths: dict[str, Path] = {}
    for trial_id in sorted(trial_ids):
        row = rows_by_id[trial_id]
        _require(isinstance(row, Mapping), f"invalid observed row: {trial_id}")
        report_path = _inside(
            Path(str(row.get("trial_report_path", ""))),
            run_root,
            f"observed trial report {trial_id}",
        )
        timing_rows.append(
            {
                "trial_id": trial_id,
                "wall_seconds": positive_seconds(
                    row.get("wall_seconds"), f"{trial_id} wall seconds"
                ),
                "cpu_seconds": positive_seconds(
                    row.get("cpu_seconds"), f"{trial_id} CPU seconds"
                ),
            }
        )
        report_paths[trial_id] = report_path

    max_wall = min(
        timing_rows,
        key=lambda row: (-float(row["wall_seconds"]), str(row["trial_id"])),
    )
    max_cpu = min(
        timing_rows,
        key=lambda row: (-float(row["cpu_seconds"]), str(row["trial_id"])),
    )
    _require(
        max_timing_path == report_paths[str(max_wall["trial_id"])],
        "selected max timing report is not the complete corpus wall maximum",
    )
    timing = _read_json(max_timing_path, "maximum timing report")
    program = verify_null_controller_program(controller_program, policy=policy)
    configuration_keys = set(program["coverage_configurations"][0])
    raw_configuration = timing.get("configuration")
    _require(
        isinstance(raw_configuration, Mapping)
        and configuration_keys <= set(raw_configuration),
        "max timing report lacks its locked development configuration",
    )
    configuration = {
        key: raw_configuration[key] for key in sorted(configuration_keys)
    }
    validate_development_configuration(policy, configuration)
    configuration_sha256 = digest_object(configuration)
    _require(
        raw_configuration.get("configuration_sha256") == configuration_sha256
        and timing.get("trial_id") == configuration["trial_id"]
        and timing.get("final_connectivity_accessed") is False,
        "max timing report configuration or data role changed",
    )
    corpus_core = {
        "kind": TIMING_CORPUS_KIND,
        "trial_count": len(timing_rows),
        "trials": timing_rows,
        "all_trials_complete_T_U_M": True,
        "development_only": True,
        "scientific_outcomes_persisted": False,
        "final_connectivity_accessed": False,
    }
    evidence = {
        "method": (
            "production_parallel_benchmark_with_complete_"
            "observed_timing_corpus_maxima"
        ),
        "observed_timing_corpus_kind": TIMING_CORPUS_KIND,
        "observed_timing_corpus_trial_count": len(timing_rows),
        "observed_timing_corpus_sha256": digest_object(corpus_core),
        "observed_serial_mean_wall_seconds": (
            math.fsum(float(row["wall_seconds"]) for row in timing_rows)
            / len(timing_rows)
        ),
        "observed_serial_max_wall_seconds": float(max_wall["wall_seconds"]),
        "observed_serial_mean_cpu_seconds_per_branch": (
            math.fsum(float(row["cpu_seconds"]) for row in timing_rows)
            / len(timing_rows)
        ),
        "observed_serial_max_cpu_seconds_per_branch": float(
            max_cpu["cpu_seconds"]
        ),
        "max_timing_trial_id": str(max_wall["trial_id"]),
        "max_busy_timing_trial_id": str(max_cpu["trial_id"]),
        "legacy_cost_projection_superseded": True,
        "legacy_cost_projection_used_for_projection": False,
        "parallelism_scope": "provider_types_within_one_complete_T_U_M_branch",
        "control_branch_execution": "sequential",
        "scientific_outcomes_persisted": False,
    }
    return evidence, configuration


def _measure_runtime_benchmark(
    *,
    policy: EpisodePolicy,
    design: Any,
    generators: Mapping[tuple[str, str], Any],
    inputs: Any,
    max_timing_configuration: Mapping[str, Any],
    observed_timing_corpus_sha256: str,
    observed_timing_corpus_trial_count: int,
) -> dict[str, Any]:
    """Run three generated-null samples through the production T/U/M fitter."""

    try:
        allocated_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    except ValueError as error:
        raise PostLockRuntimeMaterializerError(
            "SLURM_CPUS_PER_TASK is not an integer"
        ) from error
    _require(
        allocated_workers == policy.cpu_cores_per_trial_maximum == 32,
        "runtime benchmark requires the exact 32-core production allocation",
    )
    configuration = dict(max_timing_configuration)
    validate_development_configuration(policy, configuration)
    configuration_sha256 = digest_object(configuration)
    expansion = TypePooledExpansionPlan.build(design)
    binding = null_seed_binding(
        design,
        generators,
        expansion,
        selection_sha256=SELECTION_PROGRAM_SHA256,
    )
    seed_material = digest_object(
        {
            "protocol": "ep12.runtime_benchmark_master_seed.v1",
            "binding": binding,
            "max_timing_configuration_sha256": configuration_sha256,
            "observed_timing_corpus_sha256": observed_timing_corpus_sha256,
        }
    )
    seed_manifest = freeze_null_seed_manifest(
        master_seed=int.from_bytes(bytes.fromhex(seed_material)[:8], "big"),
        binding=binding,
    )
    callback = ProductionFullSearchCallback(policy=policy, inputs=inputs)
    trial = {
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
    }
    elapsed: list[float] = []
    samples = []
    for replicate_index in range(3):
        sample = sample_fitted_null(
            design,
            generators,
            expansion,
            seed_manifest=seed_manifest,
            replicate_index=replicate_index,
            selection_sha256=SELECTION_PROGRAM_SHA256,
        )
        started = time.perf_counter()
        fit_result = callback._base_fit(sample, trial)
        duration = time.perf_counter() - started
        _require(
            math.isfinite(duration)
            and duration > 0.0
            and fit_result.get("complete_T_U_M_result") is True,
            "production runtime benchmark did not complete a T/U/M branch",
        )
        elapsed.append(duration)
        samples.append(sample)
        del fit_result
    seeds = [int(seed_manifest["replicates"][index]["seed"]) for index in range(3)]
    benchmark = {
        "method": "production_fitted_null_sampler_and_T_U_M_callback_base_fit",
        "worker_count": allocated_workers,
        "benchmark_replicate_indices": [0, 1, 2],
        "benchmark_replicate_ids": [sample.replicate_id for sample in samples],
        "benchmark_replicate_seeds_sha256": digest_object(seeds),
        "max_timing_trial_id": str(configuration["trial_id"]),
        "max_timing_configuration_sha256": configuration_sha256,
        "design_sha256": str(design.design_sha256),
        "generator_manifest_sha256": str(binding["generator_manifest_sha256"]),
        "expansion_plan_sha256": str(expansion.plan_sha256),
        "callback_identity_sha256": digest_object(callback.identity()),
        "elapsed_wall_seconds": elapsed,
        "measured_parallel_mean_wall_seconds": sum(elapsed) / len(elapsed),
        "measured_parallel_max_wall_seconds": max(elapsed),
        "complete_T_U_M_branches_measured": len(elapsed),
        "benchmark_scope": RUNTIME_BENCHMARK_SCOPE,
        "benchmark_role": RUNTIME_BENCHMARK_ROLE,
        "projection_is_upper_bound": False,
        "observed_timing_corpus_sha256": observed_timing_corpus_sha256,
        "observed_timing_corpus_trial_count": observed_timing_corpus_trial_count,
        "scientific_scores_or_objectives_persisted": False,
        "development_only": True,
        "final_connectivity_accessed": False,
    }
    return validate_runtime_benchmark(benchmark)


def materialize_post_lock_runtime(
    *,
    pre_null_dir: Path,
    procedure_lock_path: Path,
    output_dir: Path,
    full_null_work_root: Path,
    policy_path: Path,
    development_role_manifest_path: Path,
    max_timing_path: Path,
    scratch_bundle_dir: Path | None = None,
) -> dict[str, Any]:
    """Cross-bind every input, then create and reload-verify the runtime."""

    pre_null_dir = _inside(pre_null_dir, EPISODE_OUTPUTS_ROOT, "pre-null bundle")
    procedure_lock_path = _inside(
        procedure_lock_path, EPISODE_OUTPUTS_ROOT, "procedure lock"
    )
    output_dir = _inside(
        output_dir, EPISODE_OUTPUTS_ROOT, "runtime bundle", exists=False
    )
    full_null_work_root = _inside(
        full_null_work_root, EPISODE_SCRATCH_ROOT, "full-null work", exists=False
    )
    development_role_manifest_path = _inside(
        development_role_manifest_path,
        EPISODE_OUTPUTS_ROOT,
        "development-role manifest",
    )
    policy = EpisodePolicy.load(policy_path)
    pre_manifest = verify_pre_null_inputs(pre_null_dir, policy_path=policy_path)
    lock = _read_json(procedure_lock_path, "procedure lock")
    verify_procedure_lock(lock, policy=policy)
    _require(
        lock.get("procedure_locked") is True
        and lock.get("null_eligible") is True
        and lock.get("final_connectivity_accessed") is False,
        "only an eligible, role-safe procedure lock may materialize a runtime",
    )
    lock_binding = lock["pre_null_input_binding"]
    _require(
        lock_binding["bundle_sha256"] == pre_manifest["bundle_sha256"]
        and lock_binding["callback_data_sha256"]
        == pre_manifest["callback_data_sha256"]
        and lock_binding["model_state_sha256"]
        == pre_manifest["model_state_sha256"],
        "procedure lock is not bound to these scientific pre-null inputs",
    )
    archive = _read_json(
        pre_null_dir / "complete_comparison_archive.json", "comparison archive"
    )
    selection = _read_json(
        pre_null_dir / "null_generator_selection.json", "generator selection"
    )
    program = _read_json(
        pre_null_dir / "null_controller_program.json", "null controller program"
    )
    source_binding = _read_json(
        pre_null_dir / "runtime_source_binding.json", "runtime source binding"
    )
    selected = _selected_record(archive, lock)
    _require(
        selection.get("selected_trial_id") == selected["trial_id"]
        and selection.get("selected_configuration_sha256")
        == selected["configuration_sha256"]
        and digest_object(selection) == lock["null_generator_selection_sha256"]
        and program.get("program_sha256")
        == lock["null_controller_program_sha256"]
        and source_binding.get("source_binding_sha256")
        == lock_binding["runtime_source_binding_sha256"],
        "selected generator/controller/source binding differs from procedure lock",
    )
    design, generators, state_manifest = load_model_state(
        pre_null_dir / "selected_fitted_null_state"
    )
    _require(
        state_manifest["state_sha256"] == lock_binding["model_state_sha256"]
        and design.design_sha256 == selection["design_sha256"],
        "selected fitted-null state differs from the lock",
    )
    inputs, callback_metadata = _load_callback_data(
        pre_null_dir / "runtime_callback_data",
        design=design,
        policy=policy,
        artifact_root=full_null_work_root,
    )
    _require(
        callback_metadata["callback_data_sha256"]
        == lock_binding["callback_data_sha256"],
        "runtime callback data differs from the lock",
    )
    role_identity_sha256 = _development_role_identity(
        development_role_manifest_path
    )
    selected_fit_binding = build_selected_fit_binding(
        controller_program=program,
        selected_trial_id=str(selected["trial_id"]),
        selected_configuration=selected["configuration"],
        development_role_manifest_sha256=role_identity_sha256,
        upstream_procedure_lock_sha256=str(lock["procedure_lock_sha256"]),
        null_generator_selection_sha256=digest_object(selection),
    )
    timing_evidence, max_timing_configuration = _timing_evidence(
        max_timing_path,
        run_root=pre_null_dir.parent,
        expected_trial_ids=archive["initial_trial_ids"],
        policy=policy,
        controller_program=program,
    )
    if output_dir.exists():
        manifest = _read_json(output_dir / "runtime_bundle.json", "runtime bundle")
        resource_plan = verify_resource_plan(
            _read_json(output_dir / "resource_plan.json", "resource plan"),
            policy=policy,
        )
        recorded_timing = resource_plan["timing_basis"]
        _require(
            all(recorded_timing.get(key) == value for key, value in timing_evidence.items())
            and recorded_timing["runtime_benchmark"][
                "max_timing_configuration_sha256"
            ]
            == digest_object(max_timing_configuration),
            "existing runtime benchmark differs from current timing evidence",
        )
    else:
        benchmark = _measure_runtime_benchmark(
            policy=policy,
            design=design,
            generators=generators,
            inputs=inputs,
            max_timing_configuration=max_timing_configuration,
            observed_timing_corpus_sha256=str(
                timing_evidence["observed_timing_corpus_sha256"]
            ),
            observed_timing_corpus_trial_count=int(
                timing_evidence["observed_timing_corpus_trial_count"]
            ),
        )
        resource_plan = freeze_resource_plan(
            policy=policy,
            controller_program=program,
            raw_vocabulary=design.raw_vocabulary,
            raw_counts=design.raw_counts,
            partner_hierarchy=inputs.partner_hierarchy,
            covariates_by_spline_df=inputs.covariates_by_spline_df,
            timing_basis={
                **timing_evidence,
                "runtime_benchmark": benchmark,
            },
        )
        manifest = materialize_runtime_bundle(
            output_dir=output_dir,
            policy=policy,
            design=design,
            generators=generators,
            covariates_by_spline_df=inputs.covariates_by_spline_df,
            partner_hierarchy=inputs.partner_hierarchy,
            fit_settings=inputs.fit_settings,
            controller_program=program,
            selected_fit_binding=selected_fit_binding,
            null_generator_selection_sha256=digest_object(selection),
            development_role_manifest_sha256=role_identity_sha256,
            upstream_procedure_lock_sha256=str(lock["procedure_lock_sha256"]),
            resource_plan=resource_plan,
        )
    full_null_work_root.mkdir(parents=True, exist_ok=True)
    load_runtime_bundle(
        output_dir,
        policy=policy,
        artifact_root=full_null_work_root,
        expected_bundle_sha256=str(manifest["bundle_sha256"]),
        upstream_procedure_lock_sha256=str(lock["procedure_lock_sha256"]),
    )
    _require(
        manifest.get("selected_configuration_sha256")
        == selected["configuration_sha256"]
        and manifest.get("controller_program_sha256") == program["program_sha256"]
        and manifest.get("null_generator_selection_sha256")
        == digest_object(selection)
        and manifest.get("development_role_manifest_sha256")
        == role_identity_sha256,
        "reloaded runtime manifest differs from locked inputs",
    )
    result = dict(manifest)
    if scratch_bundle_dir is not None:
        scratch_bundle_dir = _inside(
            scratch_bundle_dir,
            EPISODE_SCRATCH_ROOT,
            "scratch runtime bundle",
            exists=False,
        )
        _require(
            scratch_bundle_dir != full_null_work_root
            and scratch_bundle_dir not in full_null_work_root.parents
            and full_null_work_root not in scratch_bundle_dir.parents,
            "scratch runtime bundle and full-null work root may not overlap",
        )
        stage_result = stage_runtime_bundle(
            durable_bundle_dir=output_dir,
            scratch_bundle_dir=scratch_bundle_dir,
            policy=policy,
            expected_bundle_sha256=str(manifest["bundle_sha256"]),
            upstream_procedure_lock_sha256=str(lock["procedure_lock_sha256"]),
        )
        result["scratch_execution_stage"] = stage_result
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-null-dir", type=Path, required=True)
    parser.add_argument("--procedure-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-null-work-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--development-role-manifest", type=Path, required=True)
    parser.add_argument("--max-timing-report", type=Path, required=True)
    parser.add_argument("--scratch-bundle-dir", type=Path)
    args = parser.parse_args(argv)
    manifest = materialize_post_lock_runtime(
        pre_null_dir=args.pre_null_dir,
        procedure_lock_path=args.procedure_lock,
        output_dir=args.output_dir,
        full_null_work_root=args.full_null_work_root,
        policy_path=args.policy,
        development_role_manifest_path=args.development_role_manifest,
        max_timing_path=args.max_timing_report,
        scratch_bundle_dir=args.scratch_bundle_dir,
    )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PostLockRuntimeMaterializerError",
    "freeze_resource_plan",
    "materialize_post_lock_runtime",
]
