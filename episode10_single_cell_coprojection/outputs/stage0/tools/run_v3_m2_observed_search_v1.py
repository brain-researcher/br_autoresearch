#!/usr/bin/env python3
"""Run one sealed EP10 v3 observed M2 controller session.

The public entry point accepts already-loaded development objects.  It never
loads target membership, final/unassigned data, SWCs, or M0 full-refit
artifacts.  The only paths it reads are the six aggregate prerequisite reports
and the controller-owned target-blind execution inventory.

The controller session is intentionally not serializable.  A job therefore
writes one atomic aggregate log record and one separate sealed numerical
execution record after every committed trial, then refuses to resume from a
nonempty output directory: after process loss there is no unambiguous way to
reconstruct the live one-shot authority ledger.  Numerical records contain
development scores needed for deterministic downstream reconstruction; they
never contain opaque authorities or capabilities.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


SCRIPT_PATH = Path(os.path.abspath(__file__))
TOOLS_DIR = SCRIPT_PATH.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import fit_v3_m1 as m1  # noqa: E402
import fit_v3_m2 as m2  # noqa: E402
import fit_v3_m2_optional as optional_model  # noqa: E402
import load_v3_m2_observed_inputs_v1 as input_loader  # noqa: E402
import v3_m2_observed_session_v1 as observed  # noqa: E402
import v3_m2_optional_adapter_v1 as optional_adapter  # noqa: E402
import v3_m2_permutation_production_adapter_v1 as permutation_adapter  # noqa: E402
import v3_m2_rademacher_production_adapter_v1 as rademacher_adapter  # noqa: E402
import v3_m2_trial_executor_v1 as executor  # noqa: E402
import v3_search_controller_v1 as controller  # noqa: E402


EXPECTED_ROWS = 194
EXPECTED_GROUPS = 12
EXPECTED_FOLDS = 12
EXPECTED_RANK = 104
EXPECTED_PANEL_SIZE = 16

OUTPUT_PREFIX = "v3_m2_observed_search_v1_job_"
POSTSELECTION_RESULT_FILENAME = "selected_refit_report.json"
POSTSELECTION_REQUIRED_GATES = (
    "repeat_objective",
    "repeat_candidate_probability",
    "repeat_identified_pair_contrast",
    "no_single_group_dominance",
    "selected_pair_direction_stability",
    "MOp_source_family",
)
REQUIRED_PREREQUISITES = (
    "core",
    "actual_k",
    "m1_adequacy",
    "optional",
    "scale",
    "falsifier_timing",
)

SCALE_REQUIRED_UNITS = (
    "synthetic_membership_generation",
    "fresh_M0_replay_refit",
    "fresh_M1_replay_refit",
    "shared_12_fold_transform_and_offset_cache_setup",
    "reference_12_fold_trial",
    "core_12_fold_trial",
    "optional_trigger_diagnostic",
    "selected_full_development_refit_pair",
    "controller_selection_and_aggregate_report",
    "low_rank_12_fold_trial",
    "hybrid_12_fold_trial",
)
FALSIFIER_REQUIRED_UNITS = (
    "Rademacher_core",
    "Rademacher_low_rank",
    "Rademacher_hybrid_including_internal_precursor",
    "one_outer_only_permutation_M1_plus_configured_M2_fit",
)
RESOURCE_CEILINGS = MappingProxyType(
    {
        "cpu_core_hours": 1800.0,
        "gpu_hours": 0.0,
        "wall_clock_hours": 144.0,
        "parallel_cores": 64.0,
        "aggregate_memory_GiB": 512.0,
        "scratch_GiB": 1536.0,
    }
)
RESOURCE_PROJECTIONS = MappingProxyType(
    {
        "cpu_core_hours": "observed_plus_99_replays_allocated_cpu_core_hours",
        "gpu_hours": "gpu_hours",
        "wall_clock_hours": "observed_plus_four_replay_waves_wall_clock_hours",
        "parallel_cores": "parallel_cores",
        "aggregate_memory_GiB": "aggregate_memory_GiB",
        "scratch_GiB": "projected_scratch_GiB_no_deletion_credit",
    }
)

_FORBIDDEN_ROUTE_COMPONENTS = frozenset(
    {
        "final",
        "final_groups",
        "unassigned",
        "unassigned_groups",
        "swc",
        "swcs",
        "m0_full",
        "m0_full_refit",
        "m0_full_telemetry",
    }
)
_FORBIDDEN_TRUE_FLAGS = frozenset(
    {
        "real_target_membership_or_m2_effects_opened",
        "realized_positive_set_membership_opened",
        "real_m2_outcomes_opened",
        "real_outcomes_opened",
        "final_or_unassigned_or_swc_opened",
        "final_or_unassigned_opened",
        "protected_final_unassigned_or_swc_opened",
        "swc_opened",
        "m0_full_refit_or_telemetry_opened",
        "existing_m0_full_refit_or_telemetry_opened",
        "existing_m0_artifact_or_telemetry_opened",
        "existing_m0_m1_or_m2_result_opened",
    }
)


class ObservedSearchRunnerError(RuntimeError):
    """The observed-search orchestration failed closed."""


@dataclass(frozen=True)
class ValidatedPrerequisiteReports:
    """Six minimally validated aggregate reports and their source kinds."""

    payloads: Mapping[str, Mapping[str, Any]]
    source_kinds: Mapping[str, str]


@dataclass(frozen=True)
class ObservedSearchComponents:
    """Strong references required by the weakly-bound live authority graph."""

    owner: executor.CoreTrialExecutor
    session: observed.ControllerRunSession
    optional: optional_adapter.OptionalFamilyAdapter
    permutation: permutation_adapter.WholeSetPermutationAdapter
    rademacher: rademacher_adapter.RademacherReferenceAdapter
    optional_authority: observed.AdapterAuthority
    permutation_authority: observed.AdapterAuthority
    rademacher_authority: observed.AdapterAuthority


@dataclass(frozen=True)
class PostselectionContext:
    """Exact live objects and fixed output route supplied to postselection."""

    components: ObservedSearchComponents
    terminal: observed.TerminalSelectionAuthorization
    selected_refit_authority: observed.SelectedRefitProcedureAuthorization
    authorized_winner: controller.TrialResult
    candidate_selection: Mapping[str, Any]
    position_strata: object
    output_root: Path


@dataclass(frozen=True)
class ObservedSearchResult:
    """Live terminal authorization plus its aggregate durable summary."""

    terminal: observed.TerminalSelectionAuthorization
    summary_path: Path
    manifest_path: Path
    attempt_paths: tuple[Path, ...]
    execution_paths: tuple[Path, ...]
    postselection_callback_invoked: bool
    postselection_result_linkage: str | None


PostselectionCallback = Callable[[PostselectionContext], str]


def _validate_position_strata(
    value: object,
    cells: Sequence[m1.Cell],
) -> object:
    """Reauthenticate only the loader-issued pre-target strata object."""

    try:
        return input_loader._require_position_strata(value, tuple(cells))
    except input_loader.ObservedInputError as error:
        raise ObservedSearchRunnerError(
            "pre-target position strata are malformed or misaligned"
        ) from error


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ObservedSearchRunnerError(f"{label} is not a mapping")
    return value


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ObservedSearchRunnerError(f"{label} is not a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ObservedSearchRunnerError(f"{label} is not a finite number") from error
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise ObservedSearchRunnerError(f"{label} is outside its numeric domain")
    return number


def _reject_forbidden_route(path: Path, label: str) -> None:
    """Reject named protected route components without substring matching."""

    for component in path.parts:
        normalized = component.casefold().replace("-", "_")
        if normalized in _FORBIDDEN_ROUTE_COMPONENTS:
            raise ObservedSearchRunnerError(f"{label} enters a forbidden data route")


def _read_report_path(path_like: str | os.PathLike[str], label: str) -> Mapping[str, Any]:
    path = Path(os.path.abspath(os.fspath(path_like)))
    _reject_forbidden_route(path, label)
    try:
        info = path.lstat()
    except OSError as error:
        raise ObservedSearchRunnerError(f"{label} path is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ObservedSearchRunnerError(f"{label} must be a regular nonsymlink file")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise ObservedSearchRunnerError(f"{label} cannot be opened safely") from error
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ObservedSearchRunnerError(f"{label} is not readable aggregate JSON") from error
    return _mapping(payload, label)


def _load_report(
    name: str,
    source: Mapping[str, Any] | str | os.PathLike[str],
) -> tuple[Mapping[str, Any], str]:
    if isinstance(source, Mapping):
        return source, "mapping"
    if isinstance(source, (str, os.PathLike)):
        return _read_report_path(source, f"{name} prerequisite"), "regular_json_path"
    raise ObservedSearchRunnerError(
        f"{name} prerequisite must be a mapping or regular JSON path"
    )


def _reject_declared_protected_access(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() in _FORBIDDEN_TRUE_FLAGS:
                if child is not False:
                    raise ObservedSearchRunnerError(
                        "a prerequisite declares forbidden protected-data access"
                    )
            _reject_declared_protected_access(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _reject_declared_protected_access(child)


def _validate_core(payload: Mapping[str, Any]) -> None:
    inventory = _mapping(payload.get("gate_inventory"), "core gate inventory")
    if not (
        payload.get("status") == "pass"
        and payload.get("overall_gate_status") == "pass"
        and payload.get("mode") == "full"
        and payload.get("stage")
        == "v3_M2_target_free_frozen_core_contract_qualification"
        and inventory.get("all_required_gates_present_and_passed") is True
        and int(inventory.get("required_gate_count", -1)) == 8
        and payload.get("actual_K_opened") is False
    ):
        raise ObservedSearchRunnerError("core prerequisite is not the passed full gate")


def _validate_actual_k(payload: Mapping[str, Any]) -> None:
    rank = _mapping(payload.get("rank_gate"), "actual-K rank gate")
    if not (
        payload.get("status") == "pass"
        and payload.get("stage") == "v3_M2_actual_K_only_residual_rank_gate_v1"
        and int(payload.get("rows", -1)) == EXPECTED_ROWS
        and int(payload.get("groups", -1)) == EXPECTED_GROUPS
        and payload.get("actual_K_opened") is True
        and rank.get("status") == "pass"
        and int(rank.get("training_folds_evaluated", -1)) == EXPECTED_FOLDS
        and int(rank.get("training_folds_passed", -1)) == EXPECTED_FOLDS
        and int(rank.get("expected_rank", -1)) == EXPECTED_RANK
        and rank.get("rank_stable_in_every_fold") is True
    ):
        raise ObservedSearchRunnerError("actual-K prerequisite is not the rank-104 pass")


def _validate_m1_adequacy(payload: Mapping[str, Any]) -> None:
    if not (
        payload.get("status") == "pass"
        and payload.get("gate") == "v3_M2_preaccess_M1_adequacy"
        and int(payload.get("development_cells", -1)) == EXPECTED_ROWS
        and int(payload.get("development_groups", -1)) == EXPECTED_GROUPS
        and int(payload.get("target_panel_size", -1)) == EXPECTED_PANEL_SIZE
        and payload.get("M1_summary_and_outer_predictions_opened") is True
    ):
        raise ObservedSearchRunnerError("M1 adequacy prerequisite is not passed")
    for key in (
        "equal_group_M1_minus_uniform_log_score_gain_bits_per_cell",
        "equal_group_marginal_Brier_M1",
        "equal_group_marginal_Brier_fixed_K_uniform",
        "maximum_absolute_marginal_sum_K_error",
    ):
        _finite(payload.get(key), f"M1 adequacy {key}")


def _validate_optional(payload: Mapping[str, Any]) -> None:
    required = {
        "optional_trigger_diagnostic",
        "low_rank_population_recovery",
        "sparse_plus_low_rank_population_recovery",
    }
    inventory = _mapping(payload.get("gate_inventory"), "optional gate inventory")
    gates = _mapping(payload.get("gates"), "optional gates")
    if not (
        payload.get("status") == "pass"
        and payload.get("overall_gate_status") == "pass"
        and payload.get("stage")
        == "v3_M2_target_free_optional_family_qualification_v1"
        and inventory.get("all_required_gates_present_and_passed") is True
        and int(inventory.get("required_gate_count", -1)) == len(required)
        and set(gates) == required
        and all(
            isinstance(gates[name], Mapping) and gates[name].get("status") == "pass"
            for name in required
        )
    ):
        raise ObservedSearchRunnerError("optional-family prerequisite is not passed")


def _validate_measured_units(
    value: object,
    expected: Sequence[str],
    label: str,
    *,
    require_status: bool,
) -> None:
    units = _mapping(value, f"{label} measured units")
    if set(units) != set(expected):
        raise ObservedSearchRunnerError(f"{label} measured-unit inventory is incomplete")
    for name in expected:
        unit = _mapping(units[name], f"{label} unit {name}")
        if require_status and unit.get("status") != "pass":
            raise ObservedSearchRunnerError(f"{label} unit is not passed: {name}")
        for metric in ("wall_seconds", "cpu_seconds", "peak_rss_gib", "scratch_gib"):
            _finite(unit.get(metric), f"{label} {name} {metric}", nonnegative=True)


def _validate_scale(payload: Mapping[str, Any]) -> None:
    if not (
        payload.get("status")
        in {"pass", "pass_pending_sacct_and_remaining_budget_reconciliation"}
        and payload.get("stage")
        == "v3_M2_membership_blind_full_search_scale_benchmark_v1"
        and payload.get("policy_ceiling_projection_available") is True
        and payload.get("policy_ceiling_projection_pass") is True
        and payload.get("missing_unit_timings_imputed") is False
        and payload.get("aggregate_only_report") is True
    ):
        raise ObservedSearchRunnerError("scale prerequisite is not the passed projection")
    # The scale benchmark stores the already-normalized unit mapping; status
    # was consumed before serialization.  The supplement retains per-unit PASS.
    _validate_measured_units(
        payload.get("measured_units"),
        SCALE_REQUIRED_UNITS,
        "scale",
        require_status=False,
    )
    projections = _mapping(payload.get("resource_projections"), "scale projections")
    for name in ("52_trial_fully_triggered_branch", "80_trial_cap"):
        projection = _mapping(projections.get(name), f"scale projection {name}")
        gates = _mapping(projection.get("ceiling_gates"), f"scale gates {name}")
        if projection.get("all_ceiling_gates_pass") is not True:
            raise ObservedSearchRunnerError(f"scale projection did not pass: {name}")
        for resource, field in RESOURCE_PROJECTIONS.items():
            value = _finite(
                projection.get(field),
                f"scale projection {name} {field}",
                nonnegative=True,
            )
            if value > RESOURCE_CEILINGS[resource] or gates.get(resource) is not True:
                raise ObservedSearchRunnerError(
                    f"scale projection exceeds the frozen {resource} ceiling"
                )
        if gates.get("per_task_measured_RSS_within_initial_request") is not True:
            raise ObservedSearchRunnerError("scale per-task memory request is insufficient")


def _validate_falsifier_timing(payload: Mapping[str, Any]) -> None:
    fixture = _mapping(payload.get("fixture"), "falsifier timing fixture")
    if not (
        payload.get("status") == "pass"
        and payload.get("report_kind")
        == "v3_m2_v4_target_free_falsifier_timing_supplement_v1"
        and payload.get("aggregate_only") is True
        and fixture.get("source") == "deterministic_in_memory_synthetic"
        and int(fixture.get("rows", -1)) == EXPECTED_ROWS
        and int(fixture.get("groups", -1)) == EXPECTED_GROUPS
        and int(fixture.get("folds", -1)) == EXPECTED_FOLDS
        and int(fixture.get("expected_residual_rank", -1)) == EXPECTED_RANK
    ):
        raise ObservedSearchRunnerError("falsifier timing prerequisite is not passed")
    _validate_measured_units(
        payload.get("measured_units"),
        FALSIFIER_REQUIRED_UNITS,
        "falsifier timing",
        require_status=True,
    )
    rademacher = _mapping(payload.get("rademacher_witness"), "Rademacher witness")
    permutation = _mapping(payload.get("permutation_witness"), "permutation witness")
    if not (
        int(rademacher.get("frozen_seed_count", -1)) == 7
        and int(rademacher.get("folds_per_probe", -1)) == EXPECTED_FOLDS
        and rademacher.get("one_dictionary_shared_across_each_probe") is True
        and int(permutation.get("mapping_count", -1)) == 1
        and int(permutation.get("configured_M2_folds", -1)) == EXPECTED_FOLDS
        and permutation.get("own_fresh_M1_baseline_used") is True
        and permutation.get("full_development_M1_fit_run") is False
    ):
        raise ObservedSearchRunnerError("falsifier timing unit accounting is incomplete")


def _validate_combined_resource_ceiling(
    scale_payload: Mapping[str, Any],
    falsifier_payload: Mapping[str, Any],
) -> None:
    """Reconcile the supplement into the benchmark's frozen ceiling formulas."""

    scale_units = _mapping(scale_payload.get("measured_units"), "scale measured units")
    falsifier_units = _mapping(
        falsifier_payload.get("measured_units"),
        "falsifier measured units",
    )

    def metric(unit: Mapping[str, Any], field: str) -> float:
        return _finite(unit.get(field), f"combined resource {field}", nonnegative=True)

    fixed_names = (
        "shared_12_fold_transform_and_offset_cache_setup",
        "optional_trigger_diagnostic",
        "selected_full_development_refit_pair",
        "controller_selection_and_aggregate_report",
    )
    prefix_names = (
        "synthetic_membership_generation",
        "fresh_M0_replay_refit",
        "fresh_M1_replay_refit",
    )
    all_units = tuple(scale_units.values()) + tuple(falsifier_units.values())
    peak_rss = max(metric(_mapping(unit, "resource unit"), "peak_rss_gib") for unit in all_units)

    # The frozen controller has three possible coverage compositions.  Each
    # postcoverage branch has seven Rademacher slots eligible only for the
    # families reached during coverage.  Every other slot through the cap is
    # an eligible association or a whole-set permutation (including the
    # exhaustion path).  Evaluate every composition rather than assuming the
    # longest optional-coverage branch.
    coverage_branches = (
        (16, (), ("core_12_fold_trial",), ("Rademacher_core",)),
        (
            24,
            (("low_rank_12_fold_trial", 8.0),),
            ("core_12_fold_trial", "low_rank_12_fold_trial"),
            ("Rademacher_core", "Rademacher_low_rank"),
        ),
        (
            28,
            (
                ("low_rank_12_fold_trial", 8.0),
                ("hybrid_12_fold_trial", 4.0),
            ),
            (
                "core_12_fold_trial",
                "low_rank_12_fold_trial",
                "hybrid_12_fold_trial",
            ),
            (
                "Rademacher_core",
                "Rademacher_low_rank",
                "Rademacher_hybrid_including_internal_precursor",
            ),
        ),
    )
    permutation_name = "one_outer_only_permutation_M1_plus_configured_M2_fit"
    for cap in (52, 80):
        for (
            coverage_length,
            optional_coverage,
            eligible_trials,
            eligible_rademacher,
        ) in coverage_branches:
            remaining_postcoverage = cap - coverage_length - 7
            if remaining_postcoverage < 0:
                raise ObservedSearchRunnerError(
                    "combined resource branch exceeds the frozen trial cap"
                )
            totals: dict[str, float] = {}
            replay_prefix: dict[str, float] = {}
            for field in ("wall_seconds", "cpu_seconds", "scratch_gib"):
                rademacher = max(
                    metric(_mapping(falsifier_units[name], name), field)
                    for name in eligible_rademacher
                )
                association_or_permutation = max(
                    metric(_mapping(falsifier_units[permutation_name], permutation_name), field),
                    *(
                        metric(_mapping(scale_units[name], name), field)
                        for name in eligible_trials
                    ),
                )
                totals[field] = (
                    sum(
                        metric(_mapping(scale_units[name], name), field)
                        for name in fixed_names
                    )
                    + metric(
                        _mapping(scale_units["reference_12_fold_trial"], "reference"),
                        field,
                    )
                    + 15.0
                    * metric(
                        _mapping(scale_units["core_12_fold_trial"], "core"),
                        field,
                    )
                    + sum(
                        count * metric(_mapping(scale_units[name], name), field)
                        for name, count in optional_coverage
                    )
                    + 7.0 * rademacher
                    + float(remaining_postcoverage) * association_or_permutation
                )
                replay_prefix[field] = sum(
                    metric(_mapping(scale_units[name], name), field)
                    for name in prefix_names
                )
            replay_wall = replay_prefix["wall_seconds"] + totals["wall_seconds"]
            replay_scratch = replay_prefix["scratch_gib"] + totals["scratch_gib"]
            projected = {
                "cpu_core_hours": (
                    2.0
                    * (totals["wall_seconds"] + 99.0 * replay_wall)
                    * 2.0
                    / 3600.0
                ),
                "gpu_hours": 0.0,
                "wall_clock_hours": (
                    2.0 * (totals["wall_seconds"] + 4.0 * replay_wall) / 3600.0
                ),
                "parallel_cores": 64.0,
                "aggregate_memory_GiB": 32.0 * max(8.0, peak_rss),
                "scratch_GiB": totals["scratch_gib"] + 99.0 * replay_scratch,
            }
            exceeded = tuple(
                resource
                for resource, value in projected.items()
                if value > RESOURCE_CEILINGS[resource]
            )
            if exceeded:
                raise ObservedSearchRunnerError(
                    "combined scale/falsifier projection at cap "
                    f"{cap}, coverage {coverage_length} exceeds "
                    + ", ".join(exceeded)
                )


_PREREQUISITE_VALIDATORS = MappingProxyType(
    {
        "core": _validate_core,
        "actual_k": _validate_actual_k,
        "m1_adequacy": _validate_m1_adequacy,
        "optional": _validate_optional,
        "scale": _validate_scale,
        "falsifier_timing": _validate_falsifier_timing,
    }
)


def validate_prerequisite_reports(
    reports: Mapping[
        str,
        Mapping[str, Any] | str | os.PathLike[str],
    ],
) -> ValidatedPrerequisiteReports:
    """Validate only existing PASS, scope, unit, and resource fields."""

    if not isinstance(reports, Mapping) or set(reports) != set(REQUIRED_PREREQUISITES):
        raise ObservedSearchRunnerError("the six prerequisite reports are required exactly")
    payloads: dict[str, Mapping[str, Any]] = {}
    source_kinds: dict[str, str] = {}
    for name in REQUIRED_PREREQUISITES:
        payload, source_kind = _load_report(name, reports[name])
        _reject_declared_protected_access(payload)
        _PREREQUISITE_VALIDATORS[name](payload)
        payloads[name] = payload
        source_kinds[name] = source_kind
    _validate_combined_resource_ceiling(
        payloads["scale"],
        payloads["falsifier_timing"],
    )
    return ValidatedPrerequisiteReports(
        payloads=MappingProxyType(payloads),
        source_kinds=MappingProxyType(source_kinds),
    )


def _normalize_offsets(value: object) -> Mapping[str, executor.FixedM1Offset]:
    offsets = value if isinstance(value, Mapping) else getattr(value, "offsets", None)
    if not isinstance(offsets, Mapping) or set(offsets) != set(m1.GROUP_ORDER):
        raise ObservedSearchRunnerError("fixed M1 offsets must contain the exact 12 groups")
    if not isinstance(value, Mapping):
        imported_mode = getattr(value, "mode", None)
        imported_rank = getattr(value, "expected_rank", None)
        if (
            imported_mode is not None
            and imported_mode != executor.SEALED_PRODUCTION_MODE
        ) or (imported_rank is not None and imported_rank != EXPECTED_RANK):
            raise ObservedSearchRunnerError(
                "imported M1 offsets do not have the sealed rank-104 scope"
            )
    return offsets


def build_observed_search(
    *,
    cells: Sequence[m1.Cell],
    moments: Sequence[m1.UniformCellMoments],
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    pairs: m2.PairFeatureLibrary,
    fixed_m1_offsets: Mapping[str, executor.FixedM1Offset] | object,
    execution_inventory: executor.FrozenExecutionInventory | None = None,
) -> ObservedSearchComponents:
    """Build and bind all named production components before the first attempt."""

    rows = tuple(cells)
    uniform_moments = tuple(moments)
    if len(rows) != EXPECTED_ROWS or len(uniform_moments) != EXPECTED_ROWS:
        raise ObservedSearchRunnerError("observed execution requires exactly 194 aligned rows")
    inventory = (
        executor.FrozenExecutionInventory.from_tsv()
        if execution_inventory is None
        else execution_inventory
    )
    offsets = _normalize_offsets(fixed_m1_offsets)
    cache = executor.prepare_trial_cache(
        rows,
        uniform_moments,
        candidates,
        geometry,
        pairs,
        offsets,
        mode=executor.SEALED_PRODUCTION_MODE,
        execution_inventory=inventory,
        expected_rank=EXPECTED_RANK,
    )
    owner = executor.CoreTrialExecutor(cache)

    # Adapter construction precedes the session claim.  All three authorities
    # are then issued while the session is still pristine.
    optional = optional_adapter.create_optional_family_adapter(owner)
    permutation = permutation_adapter.create_whole_set_permutation_adapter(owner)
    rademacher = rademacher_adapter.create_rademacher_reference_adapter(
        owner,
        rows,
        uniform_moments,
    )
    session = observed.start_observed_session(
        owner,
        selected_refit_inputs=(
            rows,
            uniform_moments,
            candidates,
            geometry,
            pairs,
        ),
    )
    optional_authority = observed.issue_optional_family_authority(session, optional)
    permutation_authority = observed.issue_whole_set_permutation_authority(
        session,
        permutation,
    )
    rademacher_authority = observed.issue_rademacher_reference_authority(
        session,
        rademacher,
    )
    return ObservedSearchComponents(
        owner=owner,
        session=session,
        optional=optional,
        permutation=permutation,
        rademacher=rademacher,
        optional_authority=optional_authority,
        permutation_authority=permutation_authority,
        rademacher_authority=rademacher_authority,
    )


def _configuration_record(configuration: object | None) -> Mapping[str, Any] | None:
    if configuration is None:
        return None
    if type(configuration) is not controller.Configuration:
        raise ObservedSearchRunnerError("execution configuration has an unsupported type")
    return {
        "key": configuration.key,
        "family": configuration.family,
        "rho": float(configuration.rho),
        "alpha": None if configuration.alpha is None else float(configuration.alpha),
        "rank": configuration.rank,
        "sparse_rho": (
            None if configuration.sparse_rho is None else float(configuration.sparse_rho)
        ),
        "sparse_edges": list(configuration.sparse_edges),
    }


def _trial_identity_record(trial: object) -> Mapping[str, Any]:
    if type(trial) is not controller.Trial:
        raise ObservedSearchRunnerError("execution detail has an unsupported trial type")
    return {
        "valid_index": int(trial.valid_index),
        "role": trial.role,
        "kind": trial.kind,
        "phase": trial.phase,
        "falsifier": bool(trial.falsifier),
        "seed": trial.seed,
        "cycle": trial.cycle,
        "configuration": _configuration_record(trial.configuration),
        "matched_configuration": _configuration_record(trial.matched_configuration),
        "coverage_binding_present": trial.coverage_binding is not None,
    }


def _result_record(result: object) -> Mapping[str, Any]:
    if type(result) is not controller.TrialResult:
        raise ObservedSearchRunnerError("execution detail has an unsupported result type")
    return {
        "valid": bool(result.valid),
        "gain_bits_per_cell": float(result.gain_bits_per_cell),
        "active_pair_degrees": int(result.active_pair_degrees),
        "coverage_binding_present": result.coverage_binding is not None,
    }


def _fold_spec_record(spec: object) -> Mapping[str, Any]:
    if type(spec) is not executor.FoldSpec:
        raise ObservedSearchRunnerError("execution detail has an unsupported fold spec")
    return {
        "ordinal": int(spec.ordinal),
        "held_out_group": spec.held_out_group,
        "training_indexes": list(spec.training_indexes),
        "validation_indexes": list(spec.validation_indexes),
    }


def _fixed_m1_record(offset: object) -> Mapping[str, Any]:
    if type(offset) is not executor.FixedM1Offset:
        raise ObservedSearchRunnerError("execution detail has an unsupported M1 offset")
    return {
        "held_out_group": offset.held_out_group,
        "fit_identity": offset.fit_identity,
        "training_indexes": list(offset.training_indexes),
    }


def _penalty_record(penalty: object | None) -> Mapping[str, Any] | None:
    if penalty is None:
        return None
    if type(penalty) is not executor.PenaltyMapping:
        raise ObservedSearchRunnerError("execution detail has an unsupported core penalty")
    maximum = penalty.lambda_maximum
    return {
        "family": penalty.family,
        "penalty_kind": penalty.penalty_kind,
        "l1_ratio": float(penalty.l1_ratio),
        "rho": float(penalty.rho),
        "lambda_maximum": {
            "value": float(maximum.value),
            "lower_bracket": float(maximum.lower_bracket),
            "upper_bracket": float(maximum.upper_bracket),
            "zero_kkt_residual": float(maximum.zero_kkt_residual),
            "bisections": int(maximum.bisections),
        },
        "strength": float(penalty.specification.strength),
    }


def _optional_specification_record(specification: object) -> Mapping[str, Any]:
    if type(specification) is optional_model.LowRankSpec:
        return {
            "type": "low_rank",
            "rank": int(specification.rank),
            "rho": float(specification.rho),
            "lambda_reference": float(specification.lambda_reference),
            "strength": float(specification.strength),
        }
    if type(specification) is optional_model.HybridSpec:
        return {
            "type": "sparse_plus_low_rank",
            "rank": int(specification.rank),
            "rho_low": float(specification.rho_low),
            "rho_sparse": float(specification.rho_sparse),
            "lambda_reference": float(specification.lambda_reference),
            "core_lasso_lambda_maximum": float(
                specification.core_lasso_lambda_maximum
            ),
            "low_strength": float(specification.low_strength),
            "sparse_strength": float(specification.sparse_strength),
            "named_edges": [list(edge) for edge in specification.named_edges],
            "edge_indexes": list(specification.edge_indexes),
        }
    raise ObservedSearchRunnerError("execution detail has an unsupported optional spec")


def _fit_record(fit: object | None) -> Mapping[str, Any] | None:
    if fit is None:
        return None
    if type(fit) is m2.M2OptimizerResult:
        return {
            "type": "core",
            "status": fit.status,
            "objective": float(fit.objective),
            "smooth_objective": float(fit.smooth_objective),
            "penalty_value": float(fit.penalty_value),
            "iterations": int(fit.iterations),
            "kkt_residual": float(fit.kkt_residual),
            "relative_objective_change": float(fit.relative_objective_change),
            "range_residual": float(fit.range_residual),
            "coefficient_count": int(len(fit.beta)),
        }
    if type(fit) is optional_model.LowRankOptimizerResult:
        return {
            "type": "low_rank",
            "status": fit.status,
            "objective": float(fit.objective),
            "smooth_objective": float(fit.smooth_objective),
            "penalty_value": float(fit.penalty_value),
            "iterations": int(fit.iterations),
            "gradient_infinity": float(fit.gradient_infinity),
            "relative_objective_change": float(fit.relative_objective_change),
            "coefficient_count": int(len(fit.beta)),
        }
    if type(fit) is optional_model.HybridOptimizerResult:
        return {
            "type": "sparse_plus_low_rank",
            "status": fit.status,
            "objective": float(fit.objective),
            "smooth_objective": float(fit.smooth_objective),
            "low_rank_penalty_value": float(fit.low_rank_penalty_value),
            "sparse_penalty_value": float(fit.sparse_penalty_value),
            "sweeps": int(fit.sweeps),
            "joint_stationarity_infinity": float(fit.joint_stationarity_infinity),
            "relative_objective_change": float(fit.relative_objective_change),
            "coefficient_count": int(len(fit.beta)),
        }
    raise ObservedSearchRunnerError("execution detail has an unsupported optimizer result")


def _cell_score_record(cell: object) -> Mapping[str, Any]:
    if type(cell) is executor.CellTrialResult:
        return {
            "global_index": int(cell.global_index),
            "cell_key": cell.cell_key,
            "group": cell.group,
            "k": int(cell.k),
            "m1_log_probability_nats": float(cell.m1_log_probability_nats),
            "m2_log_probability_nats": float(cell.m2_log_probability_nats),
        }
    if type(cell) is permutation_adapter.PermutationCellScore:
        return {
            "global_index": int(cell.global_index),
            "cell_key": cell.cell_key,
            "group": cell.group,
            "k": int(cell.k),
            "m1_log_probability_nats": float(cell.m1_log_probability_nats),
            "m2_log_probability_nats": float(cell.m2_log_probability_nats),
            "gain_bits": float(cell.gain_bits),
        }
    if type(cell) is rademacher_adapter.RademacherCellScore:
        return {
            "global_index": int(cell.global_index),
            "cell_key": cell.cell_key,
            "group": cell.group,
            "k": int(cell.k),
            "m1_log_probability_nats": float(cell.m1_log_probability_nats),
            "m2_log_probability_nats": float(cell.m2_log_probability_nats),
            "gain_bits": float(cell.gain_bits),
        }
    raise ObservedSearchRunnerError("execution detail has an unsupported cell score")


def _pair_row_record(row: object) -> Mapping[str, Any]:
    try:
        controller._require_canonical_pair_contrast_row(row)
    except (controller.ControllerError, TypeError, ValueError) as error:
        raise ObservedSearchRunnerError("execution pair row lost controller authority") from error
    return {
        "cell_key": row.cell_key,
        "group": row.group,
        "k": int(row.k),
        "values": list(row.values),
    }


def _validate_serialized_fold_inventory(folds: Sequence[object]) -> None:
    if len(folds) != EXPECTED_FOLDS:
        raise ObservedSearchRunnerError("execution detail does not contain 12 folds")
    cells = tuple(cell for fold in folds for cell in fold.cells)
    if len(cells) != EXPECTED_ROWS or len({cell.cell_key for cell in cells}) != EXPECTED_ROWS:
        raise ObservedSearchRunnerError("execution detail does not contain 194 unique cells")


def _authenticated_execution(
    components: ObservedSearchComponents,
    result: controller.TrialResult,
) -> object:
    state = observed._require_session(components.session)
    index = result.trial.valid_index - 1
    if (
        not 0 <= index < len(state.history)
        or index >= len(state.results)
        or state.results[index] is not result
    ):
        raise ObservedSearchRunnerError("result is not a committed session entry")
    entry = state.history[index]
    if type(entry) is executor.TrialExecution:
        owned_result = executor._require_owned_execution(components.owner, entry)
        if owned_result is not result:
            evidence = state.coverage_evidence
            if not (
                0 <= index < controller.COVERAGE_TRIALS
                and type(evidence) is controller.CoverageEvidence
                and evidence.core_results[index] is result
                and result.trial is state.schedule[index]
                and owned_result.trial is result.trial
                and owned_result.valid is result.valid
                and owned_result.gain_bits_per_cell == result.gain_bits_per_cell
                and owned_result.active_pair_degrees == result.active_pair_degrees
                and owned_result.coverage_binding is None
                and (
                    index != controller.COVERAGE_TRIALS - 1
                    or result.coverage_binding is evidence.binding
                )
            ):
                raise ObservedSearchRunnerError(
                    "core execution/result identity changed outside coverage binding"
                )
        return entry
    if type(entry) is not executor.ExternalExecutionReceipt:
        raise ObservedSearchRunnerError("committed history has an unsupported entry type")
    if executor._require_external_execution_receipt(components.owner, entry) is not result:
        raise ObservedSearchRunnerError("external execution/result identity changed")
    trial = result.trial
    if trial.kind == "permutation":
        return permutation_adapter._execution_from_receipt(components.permutation, entry)
    if trial.kind == "rademacher_noise":
        return rademacher_adapter._execution_from_receipt(components.rademacher, entry)
    configuration = trial.configuration
    if configuration is not None and configuration.family in {
        "low_rank",
        "sparse_plus_low_rank",
    }:
        return optional_adapter._execution_from_receipt(components.optional, entry)
    raise ObservedSearchRunnerError("external execution has no supported named adapter")


def _core_fold_record(
    owner: executor.CoreTrialExecutor,
    fold: executor.FoldTrialResult,
) -> Mapping[str, Any]:
    """Serialize one authenticated core fold with its prepared M1 offset."""

    if type(fold) is not executor.FoldTrialResult:
        raise ObservedSearchRunnerError("core execution contains an unsupported fold")
    prepared_matches = tuple(
        prepared_fold
        for prepared_fold in owner.prepared.folds
        if prepared_fold.spec is fold.spec
        and prepared_fold.spec.held_out_group == fold.spec.held_out_group
    )
    if len(prepared_matches) != 1:
        raise ObservedSearchRunnerError(
            "core execution fold does not map to one exact prepared fold"
        )
    prepared_fold = prepared_matches[0]
    if (
        prepared_fold.fixed_m1.held_out_group != fold.spec.held_out_group
        or prepared_fold.fixed_m1.training_indexes != fold.spec.training_indexes
    ):
        raise ObservedSearchRunnerError(
            "core prepared fold has a mismatched fixed M1 offset"
        )
    return {
        "spec": _fold_spec_record(fold.spec),
        "fixed_m1": _fixed_m1_record(prepared_fold.fixed_m1),
        "penalty": _penalty_record(fold.penalty),
        "fit": _fit_record(fold.fit),
        "gain_bits_per_cell": float(fold.gain_bits_per_cell),
        "active_pair_degrees": int(fold.active_pair_degrees),
        "cells": [_cell_score_record(cell) for cell in fold.cells],
    }


def _serialize_execution_detail(
    components: ObservedSearchComponents,
    result: controller.TrialResult,
) -> Mapping[str, Any]:
    """Reauthenticate and serialize one committed result without capabilities."""

    execution = _authenticated_execution(components, result)
    trial = result.trial
    common: dict[str, Any] = {
        "record_kind": "v3_m2_committed_numeric_execution_v1",
        "trial": _trial_identity_record(trial),
        "result": _result_record(result),
        "contains_development_outcome_detail": True,
        "final_unassigned_swc_or_m0_full_opened": False,
    }
    if type(execution) is executor.TrialExecution:
        _validate_serialized_fold_inventory(execution.folds)
        if len(execution.pair_contrast_rows) != EXPECTED_ROWS:
            raise ObservedSearchRunnerError("core execution lacks 194 pair rows")
        common.update(
            {
                "execution_kind": "core",
                "selector_eligible": True,
                "folds": [
                    _core_fold_record(components.owner, fold)
                    for fold in execution.folds
                ],
                "pair_contrast_rows": [
                    _pair_row_record(row) for row in execution.pair_contrast_rows
                ],
                "equal_group_pair_contrasts": list(
                    execution.equal_group_pair_contrasts
                ),
            }
        )
        return common
    if type(execution) is optional_adapter.OptionalTrialExecution:
        _validate_serialized_fold_inventory(execution.folds)
        if len(execution.pair_contrast_rows) != EXPECTED_ROWS:
            raise ObservedSearchRunnerError("optional execution lacks 194 pair rows")
        common.update(
            {
                "execution_kind": "optional",
                "selector_eligible": True,
                "adapter_attempt_ordinal": int(execution.attempt_ordinal),
                "mode": execution.mode,
                "folds": [
                    {
                        "spec": _fold_spec_record(fold.spec),
                        "specification": _optional_specification_record(
                            fold.specification
                        ),
                        "initializer_configuration": _configuration_record(
                            fold.initializer_configuration
                        ),
                        "fit": _fit_record(fold.fit),
                        "gain_bits_per_cell": float(fold.gain_bits_per_cell),
                        "active_pair_degrees": int(fold.active_pair_degrees),
                        "cells": [_cell_score_record(cell) for cell in fold.cells],
                    }
                    for fold in execution.folds
                ],
                "pair_contrast_rows": [
                    _pair_row_record(row) for row in execution.pair_contrast_rows
                ],
                "equal_group_pair_contrasts": list(
                    execution.equal_group_pair_contrasts
                ),
            }
        )
        return common
    if type(execution) is permutation_adapter.PermutationTrialExecution:
        _validate_serialized_fold_inventory(execution.folds)
        common.update(
            {
                "execution_kind": "whole_set_permutation",
                "selector_eligible": False,
                "configuration": _configuration_record(execution.configuration),
                "mode": execution.mode,
                "adapter_attempt_ordinal": int(execution.attempt_ordinal),
                "session_attempt_ordinal": int(execution.session_attempt_ordinal),
                "null_replicate_index": execution.null_replicate_index,
                "recipient_to_donor": list(execution.recipient_to_donor),
                "m1_fit_identities": [list(value) for value in execution.m1_fit_identities],
                "folds": [
                    {
                        "spec": _fold_spec_record(fold.spec),
                        "fixed_m1": _fixed_m1_record(fold.fixed_m1),
                        "penalty": _penalty_record(fold.penalty),
                        "fit": _fit_record(fold.fit),
                        "gain_bits_per_cell": float(fold.gain_bits_per_cell),
                        "active_pair_degrees": int(fold.active_pair_degrees),
                        "cells": [_cell_score_record(cell) for cell in fold.cells],
                    }
                    for fold in execution.folds
                ],
                "pair_contrast_rows": None,
                "pair_contrast_reason": "selector_ineligible_permutation_reference",
            }
        )
        return common
    if type(execution) is rademacher_adapter.RademacherTrialExecution:
        _validate_serialized_fold_inventory(execution.folds)
        route = execution.seed_route
        common.update(
            {
                "execution_kind": "rademacher_reference",
                "selector_eligible": False,
                "matched_configuration": _configuration_record(
                    execution.matched_configuration
                ),
                "seed_route": {
                    "mode": route.mode,
                    "trial_seed": int(route.trial_seed),
                    "null_replicate_seed": route.null_replicate_seed,
                    "valid_index": route.valid_index,
                },
                "mode": execution.mode,
                "adapter_attempt_ordinal": int(execution.attempt_ordinal),
                "session_attempt_ordinal": int(execution.session_attempt_ordinal),
                "null_replicate_index": execution.null_replicate_index,
                "folds": [
                    {
                        "spec": _fold_spec_record(fold.spec),
                        "fixed_m1": _fixed_m1_record(fold.fixed_m1),
                        "specification": (
                            _penalty_record(fold.specification)
                            if type(fold.specification) is executor.PenaltyMapping
                            else _optional_specification_record(fold.specification)
                        ),
                        "precursor_configuration": _configuration_record(
                            fold.precursor_configuration
                        ),
                        "precursor_specification": (
                            None
                            if fold.precursor_specification is None
                            else _optional_specification_record(
                                fold.precursor_specification
                            )
                        ),
                        "fit": _fit_record(fold.fit),
                        "gain_bits_per_cell": float(fold.gain_bits_per_cell),
                        "active_pair_degrees": int(fold.active_pair_degrees),
                        "cells": [_cell_score_record(cell) for cell in fold.cells],
                    }
                    for fold in execution.folds
                ],
                "pair_contrast_rows": None,
                "pair_contrast_reason": "selector_ineligible_rademacher_reference",
            }
        )
        return common
    raise ObservedSearchRunnerError("committed execution detail type is unsupported")


def serialize_committed_execution_detail(
    components: ObservedSearchComponents,
    result: controller.TrialResult,
) -> Mapping[str, Any]:
    """Public narrow serializer for an exact live committed session result."""

    return _serialize_execution_detail(components, result)


def _select_winner_pair(
    components: ObservedSearchComponents,
    winner: controller.TrialResult,
) -> Mapping[str, Any]:
    execution = _authenticated_execution(components, winner)
    if type(execution) not in {
        executor.TrialExecution,
        optional_adapter.OptionalTrialExecution,
    }:
        raise ObservedSearchRunnerError("selected model lacks selector-eligible pair rows")
    if len(execution.pair_contrast_rows) != EXPECTED_ROWS:
        raise ObservedSearchRunnerError("selected model lacks the exact pair-row inventory")
    try:
        selected = controller.select_pair_contrast(execution.pair_contrast_rows)
    except (controller.ControllerError, TypeError, ValueError) as error:
        raise ObservedSearchRunnerError("frozen candidate-pair selector failed") from error
    return dict(selected)


def _coverage_rehydration_record(
    components: ObservedSearchComponents,
) -> Mapping[str, Any]:
    state = observed._require_session(components.session)
    evidence = state.coverage_evidence
    if type(evidence) is not controller.CoverageEvidence:
        raise ObservedSearchRunnerError("terminal session lacks coverage evidence")
    diagnostic = evidence.diagnostic
    localized = evidence.localized
    activation = evidence.activation
    return {
        "core_trial_count": controller.COVERAGE_TRIALS,
        "coverage_length": int(state.coverage_length),
        "diagnostic": {
            "leading_energy": float(diagnostic.leading_energy),
            "leading_edge_participation_ratio": float(
                diagnostic.leading_edge_participation_ratio
            ),
            "median_leave_one_group_cosine": float(
                diagnostic.median_leave_one_group_cosine
            ),
            "leave_one_group_cosines_at_least_0_70": int(
                diagnostic.leave_one_group_cosines_at_least_0_70
            ),
        },
        "core_incumbent_equal_group_pair_contrasts": list(
            evidence.equal_group_pair_contrasts
        ),
        "localized_sparse_selection": {
            "eligible": bool(localized.eligible),
            "edges": list(localized.edges),
            "total_absolute_contrast": float(localized.total_absolute_contrast),
            "captured_fraction": float(localized.captured_fraction),
        },
        "optional_activation": {
            "low_rank": bool(activation.low_rank),
            "sparse_plus_low_rank": bool(activation.sparse_plus_low_rank),
        },
        "rehydration_rule": (
            "rebuild trials 1..16, then call bind_coverage_evidence with the "
            "stored core results, diagnostic, and equal-group contrasts"
        ),
    }


def _require_job_id(job_id: str | int | None) -> str:
    environment = os.environ.get("SLURM_JOB_ID", "")
    supplied = environment if job_id is None else str(job_id)
    if not supplied.isdigit() or not environment or supplied != environment:
        raise ObservedSearchRunnerError("observed search requires its exact live Slurm job ID")
    return supplied


def _claim_output_root(path_like: str | os.PathLike[str], job_id: str) -> Path:
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise ObservedSearchRunnerError("SCRATCH is unavailable")
    try:
        scratch_root = Path(scratch).resolve(strict=True)
    except OSError as error:
        raise ObservedSearchRunnerError("SCRATCH cannot be resolved") from error
    episode_root = scratch_root / "br_autoresearch" / "episode10_single_cell_coprojection"
    episode_root.mkdir(parents=True, exist_ok=True)
    episode_root = episode_root.resolve(strict=True)
    expected = episode_root / f"{OUTPUT_PREFIX}{job_id}"
    supplied = Path(os.path.abspath(os.fspath(path_like)))
    _reject_forbidden_route(supplied, "observed output")
    if supplied != expected:
        raise ObservedSearchRunnerError("observed output is not the exact job-scoped path")
    if supplied.exists():
        if supplied.is_symlink() or not supplied.is_dir():
            raise ObservedSearchRunnerError("observed output is not a regular directory")
        try:
            if any(supplied.iterdir()):
                raise ObservedSearchRunnerError(
                    "nonempty output makes observed-session resume ambiguous"
                )
        except OSError as error:
            raise ObservedSearchRunnerError("observed output cannot be inspected") from error
    else:
        supplied.mkdir(mode=0o700)
    claim = supplied / "run_claim.json"
    encoded = json.dumps(
        {"job_id": job_id, "status": "claimed_fresh_session"},
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    try:
        descriptor = os.open(
            claim,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise ObservedSearchRunnerError("observed output claim failed closed") from error
    return supplied


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        raise ObservedSearchRunnerError("atomic job-scoped record write failed") from error
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass


def _dispatch_next(components: ObservedSearchComponents) -> Any:
    session = components.session
    trial = session.next_trial
    if trial is None:
        raise ObservedSearchRunnerError("dispatch requested after terminal state")
    if trial.kind == "permutation":
        return session.run_next_falsifier(components.permutation_authority)
    if trial.kind == "rademacher_noise":
        return session.run_next_falsifier(components.rademacher_authority)
    configuration = trial.configuration
    if configuration is not None and configuration.family in {
        "low_rank",
        "sparse_plus_low_rank",
    }:
        return session.run_next_optional(components.optional_authority)
    if trial.kind not in {"reference_anchor", "anchor", "successor"}:
        raise ObservedSearchRunnerError("next trial has no named execution route")
    return session.run_next_core()


def _trial_record(
    attempt_ordinal: int,
    trial: Any,
    result: Any,
    session: observed.ControllerRunSession,
) -> Mapping[str, Any]:
    configuration = trial.configuration
    return {
        "status": "attempt_complete",
        "attempt_ordinal": attempt_ordinal,
        "trial": {
            "valid_index": int(trial.valid_index),
            "role": str(trial.role),
            "kind": str(trial.kind),
            "phase": str(trial.phase),
            "falsifier": bool(trial.falsifier),
            "cycle": trial.cycle,
            "seed": trial.seed,
            "configuration_family": (
                None if configuration is None else configuration.family
            ),
        },
        "result": {
            "valid": bool(result.valid),
            "gain_bits_per_cell": float(result.gain_bits_per_cell),
            "active_pair_degrees": int(result.active_pair_degrees),
        },
        "session": {
            "status": session.status,
            "valid_trial_count": int(session.valid_trial_count),
            "invalid_attempt_count": int(session.invalid_attempt_count),
        },
        "aggregate_only": True,
    }


def _invalid_attempt_record(
    attempt_ordinal: int,
    trial: Any,
    failure_class: str,
    session: observed.ControllerRunSession,
) -> Mapping[str, Any]:
    configuration = trial.configuration
    return {
        "status": "attempt_invalid",
        "attempt_ordinal": attempt_ordinal,
        "trial": {
            "valid_index": int(trial.valid_index),
            "role": str(trial.role),
            "kind": str(trial.kind),
            "phase": str(trial.phase),
            "falsifier": bool(trial.falsifier),
            "cycle": trial.cycle,
            "seed": trial.seed,
            "configuration_family": (
                None if configuration is None else configuration.family
            ),
        },
        "failure_class": failure_class,
        "session": {
            "status": session.status,
            "valid_trial_count": int(session.valid_trial_count),
            "invalid_attempt_count": int(session.invalid_attempt_count),
        },
        "aggregate_only": True,
    }


def _invalid_numeric_attempt_record(
    attempt_ordinal: int,
    trial: controller.Trial,
    failure_class: str,
    session: observed.ControllerRunSession,
) -> Mapping[str, Any]:
    return {
        "record_kind": "v3_m2_committed_invalid_attempt_v1",
        "outcome": "invalid",
        "attempt_ordinal": attempt_ordinal,
        "trial": _trial_identity_record(trial),
        "failure_class": failure_class,
        "numerical_execution_detail_available": False,
        "session": {
            "status": session.status,
            "valid_trial_count": int(session.valid_trial_count),
            "invalid_attempt_count": int(session.invalid_attempt_count),
        },
        "opaque_authorities_or_capabilities_serialized": False,
        "final_unassigned_swc_or_m0_full_opened": False,
    }


def _resource_attempt_record(
    attempt_ordinal: int,
    trial: Any,
    failure_class: str,
    session: observed.ControllerRunSession,
) -> Mapping[str, Any]:
    configuration = trial.configuration
    return {
        "status": "attempt_resource_stop",
        "attempt_ordinal": attempt_ordinal,
        "trial": {
            "valid_index": int(trial.valid_index),
            "role": str(trial.role),
            "kind": str(trial.kind),
            "phase": str(trial.phase),
            "falsifier": bool(trial.falsifier),
            "cycle": trial.cycle,
            "seed": trial.seed,
            "configuration_family": (
                None if configuration is None else configuration.family
            ),
        },
        "failure_class": failure_class,
        "session": {
            "status": session.status,
            "valid_trial_count": int(session.valid_trial_count),
            "invalid_attempt_count": int(session.invalid_attempt_count),
        },
        "aggregate_only": True,
    }


def _resource_numeric_attempt_record(
    attempt_ordinal: int,
    trial: controller.Trial,
    failure_class: str,
    session: observed.ControllerRunSession,
) -> Mapping[str, Any]:
    return {
        "record_kind": "v3_m2_committed_resource_attempt_v1",
        "outcome": "resource",
        "attempt_ordinal": attempt_ordinal,
        "trial": _trial_identity_record(trial),
        "failure_class": failure_class,
        "numerical_execution_detail_available": False,
        "session": {
            "status": session.status,
            "valid_trial_count": int(session.valid_trial_count),
            "invalid_attempt_count": int(session.invalid_attempt_count),
        },
        "opaque_authorities_or_capabilities_serialized": False,
        "final_unassigned_swc_or_m0_full_opened": False,
    }


def _committed_failed_attempt(
    session: observed.ControllerRunSession,
    trial: controller.Trial,
    attempt_ordinal: int,
    invalid_attempts_before: int,
) -> tuple[str, str] | None:
    """Return an authenticated invalid/resource ledger commit, or no commit.

    ObservedSessionError is also used for preflight, capability, and atomic
    commit integrity failures.  Only the authenticated session attempt ledger
    distinguishes those zero-commit failures from numerical invalids and
    committed resource stops.
    """

    state = observed._require_session(session, deep_cache=False)
    invalid_attempts_after = int(state.invalid_attempts)
    invalid_delta = invalid_attempts_after - invalid_attempts_before
    ledger_count = len(state.attempts)
    if ledger_count == attempt_ordinal - 1:
        if invalid_delta != 0:
            raise ObservedSearchRunnerError(
                "session invalid count changed without an attempt-ledger commit"
            )
        return None
    if ledger_count != attempt_ordinal:
        raise ObservedSearchRunnerError(
            "session attempt ledger changed by an impossible delta"
        )
    attempt = state.attempts[-1]
    if (
        attempt.trial is not trial
        or int(attempt.invalid_attempts_after) != invalid_attempts_after
        or type(attempt.error_type) is not str
        or not attempt.error_type
    ):
        raise ObservedSearchRunnerError(
            "session invalid-attempt ledger does not match the dispatched trial"
        )
    if attempt.outcome == "invalid" and invalid_delta == 1:
        return "invalid", attempt.error_type
    if attempt.outcome == "resource" and invalid_delta == 0:
        return "resource", attempt.error_type
    raise ObservedSearchRunnerError(
        "session failed-attempt outcome disagrees with its invalid count"
    )


def _callback_state_record(
    *,
    requested: bool,
    invoked: bool,
    status: str,
    result_linkage: str | None,
) -> Mapping[str, Any]:
    if (requested, invoked, status) not in {
        (False, False, "not_requested"),
        (True, False, "pending"),
        (True, True, "complete"),
    }:
        raise ObservedSearchRunnerError("postselection callback state is inconsistent")
    if (status == "complete") != (result_linkage == POSTSELECTION_RESULT_FILENAME):
        raise ObservedSearchRunnerError(
            "postselection result linkage is inconsistent with callback state"
        )
    return {
        "postselection_callback_requested": requested,
        "postselection_callback_invoked": invoked,
        "postselection_callback_status": status,
        "postselection_result_linkage": result_linkage,
    }


def _validate_postselection_result_linkage(
    root: Path,
    linkage: object,
    terminal: observed.TerminalSelectionAuthorization,
    candidate_selection: Mapping[str, Any],
) -> str:
    """Require the fixed mode-0600 selected-refit report before completion."""

    if type(linkage) is not str or linkage != POSTSELECTION_RESULT_FILENAME:
        raise ObservedSearchRunnerError(
            "postselection callback did not return the fixed selected-refit path"
        )
    path = root / linkage
    if path.parent != root or path.is_symlink():
        raise ObservedSearchRunnerError("postselection result path escaped its job root")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ObservedSearchRunnerError(
                "postselection result is not an exact mode-0600 regular file"
            )
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            payload = json.load(handle)
    except ObservedSearchRunnerError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ObservedSearchRunnerError(
            "postselection result linkage is unreadable"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, Mapping):
        raise ObservedSearchRunnerError("postselection result is not a mapping")
    gates = payload.get("completion_gates")
    configuration = terminal.selected_configuration
    if (
        payload.get("status") != "pass"
        or payload.get("terminal_authorization_reauthenticated") is not True
        or payload.get("terminal_winner_valid_index")
        != terminal.winner_result.trial.valid_index
        or payload.get("selected_configuration_key") != configuration.key
        or payload.get("selected_pair_index") != candidate_selection.get("pair_index")
        or not isinstance(gates, Mapping)
        or tuple(sorted(gates)) != tuple(sorted(POSTSELECTION_REQUIRED_GATES))
        or any(gates.get(name) is not True for name in POSTSELECTION_REQUIRED_GATES)
        or payload.get("opaque_authority_serialized") is not False
        or payload.get("final_unassigned_swc_or_m0_full_opened") is not False
    ):
        raise ObservedSearchRunnerError(
            "postselection result does not prove selected-refit completion"
        )
    return linkage


def _terminal_summary(
    terminal: observed.TerminalSelectionAuthorization,
    *,
    job_id: str,
    attempt_count: int,
    prerequisites: ValidatedPrerequisiteReports,
    candidate_selection: Mapping[str, Any],
    selected_refit_authorized: bool,
    callback_requested: bool,
    callback_invoked: bool,
    callback_status: str,
    callback_result_linkage: str | None,
) -> Mapping[str, Any]:
    callback_state = _callback_state_record(
        requested=callback_requested,
        invoked=callback_invoked,
        status=callback_status,
        result_linkage=callback_result_linkage,
    )
    winner = terminal.winner_result
    configuration = terminal.selected_configuration
    return {
        "status": (
            "ordinary_terminal"
            if selected_refit_authorized
            else "incomplete_terminal"
        ),
        "job_id": job_id,
        "disposition": terminal.disposition,
        "scientific_conclusion": terminal.scientific_conclusion,
        "attempt_count": attempt_count,
        "valid_trial_count": int(terminal.valid_trial_count),
        "invalid_attempt_count": int(terminal.invalid_attempt_count),
        "selected": {
            "valid_index": int(winner.trial.valid_index),
            "configuration_family": configuration.family,
            "gain_bits_per_cell": float(winner.gain_bits_per_cell),
            "active_pair_degrees": int(winner.active_pair_degrees),
        },
        "candidate_selection_status": (
            str(candidate_selection.get("status"))
            if "status" in candidate_selection
            else "selected_in_sealed_terminal_manifest"
        ),
        "selected_refit_authorized": selected_refit_authorized,
        **callback_state,
        "terminal_manifest": "terminal_manifest.json",
        "prerequisites": {
            name: {
                "status": "pass",
                "source_kind": prerequisites.source_kinds[name],
            }
            for name in REQUIRED_PREREQUISITES
        },
        "expected_rows": EXPECTED_ROWS,
        "expected_groups": EXPECTED_GROUPS,
        "expected_residual_rank": EXPECTED_RANK,
        "aggregate_only": True,
        "final_unassigned_swc_or_m0_full_opened": False,
    }


def _terminal_manifest(
    terminal: observed.TerminalSelectionAuthorization,
    *,
    job_id: str,
    attempt_paths: Sequence[Path],
    execution_entries: Sequence[Mapping[str, Any]],
    invalid_execution_entries: Sequence[Mapping[str, Any]],
    coverage_rehydration: Mapping[str, Any],
    candidate_selection: Mapping[str, Any],
    selected_refit_authorized: bool,
    callback_requested: bool,
    callback_invoked: bool,
    callback_status: str,
    callback_result_linkage: str | None,
) -> Mapping[str, Any]:
    callback_state = _callback_state_record(
        requested=callback_requested,
        invoked=callback_invoked,
        status=callback_status,
        result_linkage=callback_result_linkage,
    )
    winner = terminal.winner_result
    ordered = tuple(sorted(execution_entries, key=lambda item: int(item["valid_index"])))
    if tuple(int(item["valid_index"]) for item in ordered) != tuple(
        range(1, terminal.valid_trial_count + 1)
    ):
        raise ObservedSearchRunnerError("terminal numeric execution inventory is incomplete")
    if len(attempt_paths) != (
        terminal.valid_trial_count + terminal.invalid_attempt_count
    ):
        raise ObservedSearchRunnerError("terminal attempt-log inventory is incomplete")
    if len(invalid_execution_entries) != terminal.invalid_attempt_count:
        raise ObservedSearchRunnerError(
            "terminal invalid-attempt numeric inventory is incomplete"
        )
    selected = tuple(
        item for item in ordered if int(item["valid_index"]) == winner.trial.valid_index
    )
    if len(selected) != 1:
        raise ObservedSearchRunnerError("terminal winner numeric execution is unavailable")
    return {
        "record_kind": "v3_m2_observed_terminal_manifest_v1",
        "status": (
            "ordinary_terminal"
            if selected_refit_authorized
            else "incomplete_terminal"
        ),
        "job_id": job_id,
        "disposition": terminal.disposition,
        "scientific_conclusion": terminal.scientific_conclusion,
        "valid_trial_count": int(terminal.valid_trial_count),
        "invalid_attempt_count": int(terminal.invalid_attempt_count),
        "attempt_logs": [path.name for path in attempt_paths],
        "committed_executions": [dict(item) for item in ordered],
        "committed_invalid_attempts": [
            dict(item)
            for item in sorted(
                invalid_execution_entries,
                key=lambda item: int(item["attempt_ordinal"]),
            )
        ],
        "winner_valid_index": int(winner.trial.valid_index),
        "winner_execution_file": selected[0]["numeric_execution_file"],
        "selected_configuration": _configuration_record(
            terminal.selected_configuration
        ),
        "coverage_rehydration": dict(coverage_rehydration),
        "candidate_selection": dict(candidate_selection),
        "selected_refit_authorized": selected_refit_authorized,
        **callback_state,
        "opaque_authorities_or_capabilities_serialized": False,
        "final_unassigned_swc_or_m0_full_opened": False,
    }


def run_observed_search(
    *,
    cells: Sequence[m1.Cell],
    moments: Sequence[m1.UniformCellMoments],
    candidates: m1.CandidateLibrary,
    geometry: m1.GeometryEngine,
    pairs: m2.PairFeatureLibrary,
    position_strata: object,
    fixed_m1_offsets: Mapping[str, executor.FixedM1Offset] | object,
    prerequisite_reports: Mapping[
        str,
        Mapping[str, Any] | str | os.PathLike[str],
    ],
    output_root: str | os.PathLike[str],
    job_id: str | int | None = None,
    execution_inventory: executor.FrozenExecutionInventory | None = None,
    postselection_callback: PostselectionCallback | None = None,
) -> ObservedSearchResult:
    """Run the serial controller with aggregate logs and sealed numeric detail."""

    frozen_position_strata = _validate_position_strata(position_strata, cells)
    if not callable(postselection_callback):
        raise ObservedSearchRunnerError(
            "observed production search requires the named in-process "
            "selected-refit callback"
        )
    exact_job_id = _require_job_id(job_id)
    prerequisites = validate_prerequisite_reports(prerequisite_reports)
    root = _claim_output_root(output_root, exact_job_id)
    _atomic_json(
        root / "started.json",
        {
            "status": "running",
            "job_id": exact_job_id,
            "expected_rows": EXPECTED_ROWS,
            "expected_groups": EXPECTED_GROUPS,
            "expected_residual_rank": EXPECTED_RANK,
            "prerequisites": list(REQUIRED_PREREQUISITES),
            "aggregate_only": True,
        },
    )

    components: ObservedSearchComponents | None = None
    stage = "component_build"
    attempt_paths: list[Path] = []
    execution_paths: list[Path] = []
    execution_entries: list[Mapping[str, Any]] = []
    invalid_execution_entries: list[Mapping[str, Any]] = []
    attempt_count = 0
    callback_requested = postselection_callback is not None
    callback_invoked = False
    callback_status = "pending" if callback_requested else "not_requested"
    callback_result_linkage: str | None = None
    selected_refit_authority: (
        observed.SelectedRefitProcedureAuthorization | None
    ) = None
    try:
        numeric_root = root / "numeric"
        numeric_root.mkdir(mode=0o700)
        components = build_observed_search(
            cells=cells,
            moments=moments,
            candidates=candidates,
            geometry=geometry,
            pairs=pairs,
            fixed_m1_offsets=fixed_m1_offsets,
            execution_inventory=execution_inventory,
        )
        while components.session.next_trial is not None:
            attempt_count += 1
            stage = f"attempt_{attempt_count:04d}"
            trial = components.session.next_trial
            assert trial is not None
            attempt_path = root / f"attempt_{attempt_count:04d}.json"
            invalid_attempts_before = int(
                components.session.invalid_attempt_count
            )
            try:
                result = _dispatch_next(components)
            except BaseException as error:
                failed_attempt = _committed_failed_attempt(
                    components.session,
                    trial,
                    attempt_count,
                    invalid_attempts_before,
                )
                if failed_attempt is None:
                    raise
                outcome, failure_class = failed_attempt
                if outcome == "resource":
                    resource_numeric_path = (
                        numeric_root / f"resource_attempt_{attempt_count:04d}.json"
                    )
                    _atomic_json(
                        resource_numeric_path,
                        _resource_numeric_attempt_record(
                            attempt_count,
                            trial,
                            failure_class,
                            components.session,
                        ),
                    )
                    execution_paths.append(resource_numeric_path)
                    _atomic_json(
                        attempt_path,
                        _resource_attempt_record(
                            attempt_count,
                            trial,
                            failure_class,
                            components.session,
                        ),
                    )
                    attempt_paths.append(attempt_path)
                    raise
                invalid_numeric_path = (
                    numeric_root / f"invalid_attempt_{attempt_count:04d}.json"
                )
                _atomic_json(
                    invalid_numeric_path,
                    _invalid_numeric_attempt_record(
                        attempt_count,
                        trial,
                        failure_class,
                        components.session,
                    ),
                )
                execution_paths.append(invalid_numeric_path)
                invalid_execution_entries.append(
                    {
                        "attempt_ordinal": attempt_count,
                        "proposed_valid_index": int(trial.valid_index),
                        "trial_kind": trial.kind,
                        "numeric_attempt_file": str(
                            invalid_numeric_path.relative_to(root)
                        ),
                        "aggregate_attempt_log": attempt_path.name,
                    }
                )
                _atomic_json(
                    attempt_path,
                    _invalid_attempt_record(
                        attempt_count,
                        trial,
                        failure_class,
                        components.session,
                    ),
                )
                attempt_paths.append(attempt_path)
                if not isinstance(error, Exception):
                    raise
                if components.session.status == observed.RUNNING:
                    continue
                raise
            numeric_path = (
                numeric_root / f"committed_trial_{int(result.trial.valid_index):04d}.json"
            )
            _atomic_json(
                numeric_path,
                serialize_committed_execution_detail(components, result),
            )
            execution_paths.append(numeric_path)
            _atomic_json(
                attempt_path,
                _trial_record(attempt_count, trial, result, components.session),
            )
            attempt_paths.append(attempt_path)
            execution_entries.append(
                {
                    "valid_index": int(result.trial.valid_index),
                    "attempt_ordinal": attempt_count,
                    "trial_kind": result.trial.kind,
                    "configuration_key": (
                        None
                        if result.trial.configuration is None
                        else result.trial.configuration.key
                    ),
                    "numeric_execution_file": str(
                        numeric_path.relative_to(root)
                    ),
                    "aggregate_attempt_log": attempt_path.name,
                }
            )

        stage = "terminal_selection"
        terminal = observed.authorize_terminal_selection(components.session)
        winner = terminal.winner_result
        ordinary = (
            terminal.disposition == observed.ORDINARY_PATIENCE_STOP
            and terminal.scientific_conclusion == observed.ORDINARY_CONCLUSION
        )
        callback_requested = postselection_callback is not None and ordinary
        callback_status = "pending" if callback_requested else "not_requested"
        selected_refit_authorized = False
        if ordinary:
            authorized_winner = observed.require_selected_refit_authorization(
                components.owner,
                terminal,
            )
            if authorized_winner is not winner:
                raise ObservedSearchRunnerError(
                    "selected-refit authorization changed the terminal winner"
                )
            selected_refit_authorized = True
            stage = "candidate_selection"
            candidate_selection = _select_winner_pair(components, winner)
        else:
            candidate_selection = {
                "status": "not_run_incomplete_terminal",
                "reason": "candidate selection is downstream of an ordinary stop",
            }

        stage = "terminal_manifest"
        coverage_rehydration = _coverage_rehydration_record(components)
        manifest_path = root / "terminal_manifest.json"
        _atomic_json(
            manifest_path,
            _terminal_manifest(
                terminal,
                job_id=exact_job_id,
                attempt_paths=attempt_paths,
                execution_entries=execution_entries,
                invalid_execution_entries=invalid_execution_entries,
                coverage_rehydration=coverage_rehydration,
                candidate_selection=candidate_selection,
                selected_refit_authorized=selected_refit_authorized,
                callback_requested=callback_requested,
                callback_invoked=callback_invoked,
                callback_status=callback_status,
                callback_result_linkage=callback_result_linkage,
            ),
        )
        stage = "terminal_summary"
        summary_path = root / "terminal_summary.json"
        _atomic_json(
            summary_path,
            _terminal_summary(
                terminal,
                job_id=exact_job_id,
                attempt_count=attempt_count,
                prerequisites=prerequisites,
                candidate_selection=candidate_selection,
                selected_refit_authorized=selected_refit_authorized,
                callback_requested=callback_requested,
                callback_invoked=callback_invoked,
                callback_status=callback_status,
                callback_result_linkage=callback_result_linkage,
            ),
        )
        if not ordinary:
            stage = "incomplete_terminal"
            raise ObservedSearchRunnerError(
                "observed search ended without an ordinary patience stop"
            )
        if callback_requested:
            if not callable(postselection_callback):
                stage = "postselection_callback"
                raise ObservedSearchRunnerError("postselection callback is not callable")
            stage = "selected_refit_authority"
            selected_refit_authority = observed.authorize_selected_refit_procedure(
                components.owner,
                terminal,
            )
            stage = "postselection_callback"
            linkage = postselection_callback(
                PostselectionContext(
                    components=components,
                    terminal=terminal,
                    selected_refit_authority=selected_refit_authority,
                    authorized_winner=authorized_winner,
                    candidate_selection=MappingProxyType(dict(candidate_selection)),
                    position_strata=frozen_position_strata,
                    output_root=root,
                )
            )
            callback_result_linkage = _validate_postselection_result_linkage(
                root,
                linkage,
                terminal,
                candidate_selection,
            )
            observed._complete_selected_refit_procedure(
                selected_refit_authority,
                callback_result_linkage,
            )
            callback_invoked = True
            callback_status = "complete"
            stage = "terminal_callback_complete"
            _atomic_json(
                manifest_path,
                _terminal_manifest(
                    terminal,
                    job_id=exact_job_id,
                    attempt_paths=attempt_paths,
                    execution_entries=execution_entries,
                    invalid_execution_entries=invalid_execution_entries,
                    coverage_rehydration=coverage_rehydration,
                    candidate_selection=candidate_selection,
                    selected_refit_authorized=selected_refit_authorized,
                    callback_requested=callback_requested,
                    callback_invoked=callback_invoked,
                    callback_status=callback_status,
                    callback_result_linkage=callback_result_linkage,
                ),
            )
            _atomic_json(
                summary_path,
                _terminal_summary(
                    terminal,
                    job_id=exact_job_id,
                    attempt_count=attempt_count,
                    prerequisites=prerequisites,
                    candidate_selection=candidate_selection,
                    selected_refit_authorized=selected_refit_authorized,
                    callback_requested=callback_requested,
                    callback_invoked=callback_invoked,
                    callback_status=callback_status,
                    callback_result_linkage=callback_result_linkage,
                ),
            )
        return ObservedSearchResult(
            terminal=terminal,
            summary_path=summary_path,
            manifest_path=manifest_path,
            attempt_paths=tuple(attempt_paths),
            execution_paths=tuple(execution_paths),
            postselection_callback_invoked=callback_invoked,
            postselection_result_linkage=callback_result_linkage,
        )
    except BaseException as error:
        if selected_refit_authority is not None:
            try:
                observed._burn_selected_refit_procedure(
                    selected_refit_authority
                )
            except BaseException:
                pass
        failure: dict[str, Any] = {
            "status": "failed_closed",
            "job_id": exact_job_id,
            "stage": stage,
            "exception_type": type(error).__name__,
            "completed_attempt_records": len(attempt_paths),
            "completed_numeric_execution_records": len(execution_paths),
            "aggregate_only": True,
        }
        if components is not None:
            failure["session"] = {
                "status": components.session.status,
                "valid_trial_count": int(components.session.valid_trial_count),
                "invalid_attempt_count": int(components.session.invalid_attempt_count),
            }
        try:
            _atomic_json(root / "failure.json", failure)
        except ObservedSearchRunnerError:
            pass
        raise


__all__ = (
    "ObservedSearchComponents",
    "ObservedSearchResult",
    "ObservedSearchRunnerError",
    "ValidatedPrerequisiteReports",
    "build_observed_search",
    "run_observed_search",
    "serialize_committed_execution_detail",
    "validate_prerequisite_reports",
)
