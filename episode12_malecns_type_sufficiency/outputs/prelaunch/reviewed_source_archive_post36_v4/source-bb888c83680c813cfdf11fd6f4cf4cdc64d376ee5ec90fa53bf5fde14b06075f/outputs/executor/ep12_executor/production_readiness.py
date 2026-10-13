"""Outcome-blind production-readiness gate for the EP12 observed tail.

This module requires a separate attestation root.  The attester authenticates
all source bytes and extracts only schema/status metadata; neither layer emits
or compares objectives, scores, deltas, intervals, or scientific selections.

The gate is intentionally narrower than a procedure lock.  Passing it means
that the deterministic post-36 observed tail may be launched.  A falsifier
*plan* is not falsifier execution, so this gate never authorizes a procedure
lock, a full-search null, or final-connectivity access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import observed_tail_plan as frozen_tail_plan

from .attest_observed_trials import verify_attestation_root
from .policy import EpisodePolicy, canonical_json, digest_bytes, digest_object
from .qualification_launch import (
    QUALIFICATION_HASH_TARGETS,
    QualificationLaunchError,
    read_qualification_launch_receipt,
    secure_qualification_path,
)
from .scientific_transition_binding import (
    ScientificTransitionBindingError,
    verify_scientific_contract_transition,
)
from .yaml_subset import load_yaml_subset


EPISODE_ROOT = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency"
)
OUTPUTS_ROOT = EPISODE_ROOT / "outputs"

QUALIFICATION_SCHEMA = "ep12.scientific_qualification_report.v2"
FALSIFIER_PLAN_SCHEMA = "ep12.observed_post36_tail_plan.v1"
READINESS_SCHEMA = "ep12.production_readiness_gate.v2"
READINESS_MANIFEST_SCHEMA = "ep12.production_readiness_gate_manifest.v2"
EXPECTED_BUDGET_OVERRIDE_SHA256 = (
    "adac5f1c7f7497450881ca757e9084b4770ba0b72b07d162781f740802fc9e7a"
)
AUTHORIZED_PLANNING_CORE_HOURS = 2750.224361392
AUTHORIZED_ROUNDED_CORE_HOURS = 2750

_QUALIFICATION_HASHED_FILES = {
    "report_sha256": "qualification_report.json",
    "receipts_sha256": "fit_and_score_receipts.jsonl",
    "result_table_csv_sha256": "result_table.csv",
    "result_table_markdown_sha256": "result_table.md",
    "cost_profile_sha256": "cost_profile.json",
    "environment_lock_sha256": "environment_lock.json",
    "integration_stability_sha256": "integration_stability.json",
}

_CHAIN_SOURCES = (
    (
        "coverage",
        18,
        Path("search_plan/coverage_configurations.json"),
        Path("observed"),
        None,
    ),
    (
        "adaptive_cycle_1",
        9,
        Path("adaptive/cycle1/plan/successor_configurations.json"),
        Path("adaptive/cycle1/observed"),
        1,
    ),
    (
        "adaptive_cycle_2",
        9,
        Path("adaptive/cycle2/plan/successor_configurations.json"),
        Path("adaptive/cycle2/observed"),
        2,
    ),
)


class ReadinessError(ValueError):
    """A readiness input is missing, malformed, or not qualified."""


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    boundary = root.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ReadinessError(f"{label} must stay under {boundary}: {resolved}")
    return resolved


def _read_json(path: Path) -> Any:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError as error:
        raise ReadinessError(f"missing JSON artifact: {path}") from error
    except OSError as error:
        raise ReadinessError(f"cannot open JSON artifact {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"JSON artifact is not regular: {path}")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            try:
                return json.load(handle)
            except json.JSONDecodeError as error:
                raise ReadinessError(f"malformed JSON artifact: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReadinessError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"hashed artifact is not regular: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError as error:
        raise ReadinessError(f"missing hashed artifact: {path}") from error
    except OSError as error:
        raise ReadinessError(f"cannot hash artifact {path}: {error}") from error
    finally:
        if "descriptor" in locals() and descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def _verify_hash(path: Path, expected: object, *, label: str) -> str:
    expected_text = str(expected)
    actual = _sha256_file(path)
    _require(actual == expected_text, f"{label} SHA-256 mismatch")
    return actual


def _check(
    check_id: str,
    function: Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    try:
        evidence = function(*args, **kwargs)
    except (ReadinessError, OSError, TypeError, ValueError, KeyError) as error:
        return {
            "check_id": check_id,
            "passed": False,
            "reason": str(error),
            "evidence": {},
        }
    return {
        "check_id": check_id,
        "passed": True,
        "reason": None,
        "evidence": evidence,
    }


def _same_number(observed: Any, expected: float) -> bool:
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        return False
    return math.isfinite(float(observed)) and math.isclose(
        float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
    )


def _qualification_cell_id(
    scenario: str,
    neurons: int,
    count_total: int,
    strength: float,
    missing_fraction: float,
) -> str:
    strength_id = str(float(strength)).replace(".", "p")
    missing_id = str(float(missing_fraction)).replace(".", "p")
    return (
        f"{scenario}__n{neurons}__N{count_total}__"
        f"s{strength_id}__m{missing_id}"
    )


def _expected_qualification_cells(
    spec: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Rebuild the exact frozen qualification grid without scientific imports."""

    generators = _mapping(spec.get("generators"), label="qualification generators")
    run_tiers = _mapping(spec.get("run_tiers"), label="qualification run tiers")
    resolution = _mapping(spec.get("resolution_grid"), label="qualification resolution grid")
    strata_fields = (
        run_tiers.get("qualification_neurons_per_side"),
        run_tiers.get("qualification_count_totals"),
        run_tiers.get("qualification_generator_strengths"),
        run_tiers.get("qualification_missing_fractions"),
    )
    _require(
        all(isinstance(field, list) for field in strata_fields)
        and len({len(field) for field in strata_fields}) == 1,
        "frozen qualification strata are malformed",
    )
    cells: dict[str, dict[str, Any]] = {}

    def add(
        scenario: str,
        neurons: Any,
        count_total: Any,
        strength: Any,
        missing_fraction: Any,
        acceptance_role: str,
    ) -> None:
        _require(scenario in generators, f"unknown qualification scenario: {scenario}")
        generator = _mapping(generators[scenario], label=f"generator {scenario}")
        record = {
            "scenario": scenario,
            "neurons_per_side": int(neurons),
            "count_total": int(count_total),
            "strength": float(strength),
            "missing_fraction": float(missing_fraction),
            "acceptance_role": acceptance_role,
            "truth_family": str(generator.get("truth")),
        }
        cell_id = _qualification_cell_id(
            scenario,
            record["neurons_per_side"],
            record["count_total"],
            record["strength"],
            record["missing_fraction"],
        )
        cells[cell_id] = record

    for scenario in generators:
        for neurons, count_total, strength, missing_fraction in zip(*strata_fields):
            add(
                str(scenario),
                neurons,
                count_total,
                strength,
                missing_fraction,
                "core_operating_characteristic",
            )
    _require(
        run_tiers.get("qualification_adds_group_strength_resolution_grid") is True,
        "frozen qualification omits the group-strength resolution grid",
    )
    for strength in resolution.get("group_logit_strength", ()):
        add(
            "residual_groups_within_continuum",
            19,
            100,
            strength,
            0.1,
            "resolution_diagnostic_only",
        )
    _require(
        run_tiers.get("qualification_adds_side_loss_fraction_grid") is True,
        "frozen qualification omits the side-loss resolution grid",
    )
    for missing_fraction in resolution.get("side_loss_fraction", ()):
        add(
            "side_specific_endpoint_loss",
            19,
            100,
            1.8,
            missing_fraction,
            "resolution_diagnostic_only",
        )
    _require(len(cells) == 35, "frozen qualification grid must contain exactly 35 cells")
    return cells


def verify_full_synthetic_qualification(
    *,
    launch_path: Path,
    qualification_dir: Path,
    outputs_root: Path = OUTPUTS_ROOT,
    executor_root: Path | None = None,
) -> dict[str, Any]:
    """Verify a full generated-data qualification and all manifest hashes."""

    _require(outputs_root.is_absolute(), "qualification outputs root must be absolute")
    episode_root = outputs_root.parent
    try:
        secured_outputs_root = secure_qualification_path(
            outputs_root,
            episode_root=episode_root,
            label="qualification outputs root",
        )
        if secured_outputs_root != episode_root / "outputs":
            raise QualificationLaunchError("qualification outputs root path changed")
        launch_path = secure_qualification_path(
            launch_path,
            episode_root=episode_root,
            label="qualification launch",
        )
        qualification_dir = secure_qualification_path(
            qualification_dir,
            episode_root=episode_root,
            label="qualification output directory",
        )
    except QualificationLaunchError as error:
        raise ReadinessError(str(error)) from error

    nested_launch_path = (
        episode_root
        / "outputs/prelaunch/post36_downstream_launch/stages/"
        "scientific_qualification/receipt.json"
    )
    launch_hash_targets: dict[str, Path]
    if launch_path == nested_launch_path:
        _require(
            executor_root is not None,
            "archive-bound qualification requires an authenticated executor root",
        )
        from .archive_nested_qualification import (
            ArchiveNestedQualificationError,
            qualification_launch_view,
            verify_archive_nested_qualification_launch,
        )

        try:
            nested_evidence = verify_archive_nested_qualification_launch(
                episode_root=episode_root,
                executor_root=executor_root,
                launch_path=launch_path,
            )
            launch = qualification_launch_view(nested_evidence)
            source_paths = _mapping(
                nested_evidence.get("source_paths"),
                label="archive-bound qualification source paths",
            )
            launch_hash_targets = {
                field: Path(str(source_paths[field]))
                for field in QUALIFICATION_HASH_TARGETS
            }
        except (ArchiveNestedQualificationError, KeyError) as error:
            raise ReadinessError(str(error)) from error
    else:
        try:
            launch = read_qualification_launch_receipt(
                launch_path, episode_root=episode_root
            )
        except QualificationLaunchError as error:
            raise ReadinessError(str(error)) from error
        launch_hash_targets = {}
        for field, relative in QUALIFICATION_HASH_TARGETS.items():
            try:
                launch_hash_targets[field] = secure_qualification_path(
                    episode_root / relative,
                    episode_root=episode_root,
                    label=f"qualification source {field}",
                )
            except QualificationLaunchError as error:
                raise ReadinessError(str(error)) from error
    _require(
        Path(str(launch["output_dir"])) == qualification_dir,
        "qualification directory does not match the frozen launch receipt",
    )
    repetitions = int(launch.get("replicates_per_core_error_cell", 0))
    transition_identity = _mapping(
        launch.get("scientific_contract_transition"),
        label="qualification launch scientific transition",
    )

    for field, path in launch_hash_targets.items():
        _verify_hash(path, launch.get(field), label=f"launch-bound {field}")
    spec_path = launch_hash_targets["qualification_spec_sha256"]
    _require(
        _sha256_file(spec_path)
        == transition_identity.get("active_qualification_spec_sha256"),
        "qualification spec differs from the active post-36 transition",
    )
    spec = _mapping(load_yaml_subset(spec_path), label="qualification specification")
    _require(
        spec.get("schema_version") == "ep12.synthetic_scientific_qualification.v2"
        and spec.get("status")
        == "frozen_before_post36_scientific_qualification_run",
        "qualification specification is not the frozen post-36 EP12 schema",
    )
    _require(
        spec.get("uses_actual_fitter_and_scorer") is True
        and spec.get("fixture_routing_as_evidence") == "forbidden"
        and spec.get("real_connectivity_values_allowed") is False,
        "qualification specification permits invalid evidence or real connectivity",
    )
    run_tiers = _mapping(spec.get("run_tiers"), label="qualification run tiers")
    expected_repetitions = int(
        run_tiers.get("qualification_replicates_per_core_error_cell", 0)
    )
    expected_whole_types = int(run_tiers.get("whole_types_per_replicate", 0))
    _require(
        repetitions == expected_repetitions == 72,
        "qualification launch does not run the frozen 72 replicates per cell",
    )
    expected_cells = _expected_qualification_cells(spec)

    manifest_path = qualification_dir / "manifest.json"
    report_path = qualification_dir / "qualification_report.json"
    manifest = _mapping(_read_json(manifest_path), label="qualification manifest")
    for field, filename in _QUALIFICATION_HASHED_FILES.items():
        _require(field in manifest, f"qualification manifest is missing {field}")
        _verify_hash(
            qualification_dir / filename,
            manifest[field],
            label=f"qualification {filename}",
        )
    report = _mapping(_read_json(report_path), label="qualification report")
    _require(report.get("schema_version") == QUALIFICATION_SCHEMA, "unsupported qualification report schema")
    _require(report.get("tier") == "qualification", "report is not the full qualification tier")
    _require(report.get("qualification_scope") == "actual_T_U_M_generated_data_only", "qualification scope is not the actual generated-data T/U/M program")
    _require(report.get("scientific_contract_enforced") is True, "scientific contract was not enforced")
    _require(report.get("contract_configuration") is True, "qualification did not use the contract configuration")
    _require(report.get("real_connectivity_accessed") is False, "qualification report records real connectivity access")
    _require(manifest.get("real_connectivity_accessed") is False, "qualification manifest records real connectivity access")
    _require(report.get("full_search_null_executed") is False, "generated-data qualification must not masquerade as a full-search null")
    _require(
        report.get("scientific_contract_transition") == transition_identity,
        "qualification report transition identity differs from its launch",
    )
    _require(
        report.get("observed_prefix_contract_sha256")
        == transition_identity.get("observed_prefix_contract_sha256")
        and report.get("active_contract_sha256")
        == transition_identity.get("active_contract_sha256")
        and report.get("active_qualification_spec_sha256")
        == transition_identity.get("active_qualification_spec_sha256"),
        "qualification report conflates observed-prefix and active identities",
    )
    settings = _mapping(report.get("settings"), label="qualification settings")
    _require(int(settings.get("repetitions_per_cell", 0)) == repetitions, "qualification repetition count differs from the launch receipt")
    _require(
        int(settings.get("whole_types_per_replicate", 0)) == expected_whole_types == 8,
        "qualification did not use the frozen whole-type replication count",
    )
    cells = report.get("cells")
    _require(isinstance(cells, list), "qualification cells must be a list")
    observed_cells: dict[str, Mapping[str, Any]] = {}
    for raw_cell in cells:
        cell = _mapping(raw_cell, label="qualification cell")
        cell_id = cell.get("cell_id")
        _require(isinstance(cell_id, str) and cell_id, "qualification cell lacks an ID")
        _require(cell_id not in observed_cells, f"duplicate qualification cell: {cell_id}")
        observed_cells[cell_id] = cell
    _require(
        set(observed_cells) == set(expected_cells),
        "qualification report does not contain the exact frozen 35-cell grid",
    )
    terminal_names = {
        "M_supported_before_full_search_null",
        "single_population_adequate_before_sensitivity_gate",
        "unresolved",
        "numerical_failure",
    }
    for cell_id, expected in expected_cells.items():
        cell = observed_cells[cell_id]
        for field in (
            "scenario",
            "neurons_per_side",
            "count_total",
            "acceptance_role",
            "truth_family",
        ):
            _require(
                cell.get(field) == expected[field],
                f"qualification cell {cell_id} changes frozen field {field}",
            )
        for field in ("strength", "missing_fraction"):
            _require(
                _same_number(cell.get(field), expected[field]),
                f"qualification cell {cell_id} changes frozen field {field}",
            )
        _require(
            int(cell.get("replicates", 0)) == repetitions,
            f"qualification cell {cell_id} has incomplete replication",
        )
        _require(
            cell.get("passed") is True and cell.get("failure_reasons") == [],
            f"qualification cell {cell_id} failed its frozen acceptance rule",
        )
        terminal_counts = _mapping(
            cell.get("terminal_counts"), label=f"terminal counts {cell_id}"
        )
        _require(
            set(terminal_counts) == terminal_names,
            f"qualification cell {cell_id} has malformed terminal counts",
        )
        integer_counts = []
        for name in terminal_names:
            value = terminal_counts[name]
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"qualification cell {cell_id} has invalid terminal count {name}",
            )
            integer_counts.append(value)
        _require(
            sum(integer_counts) == repetitions
            and terminal_counts["numerical_failure"] == 0,
            f"qualification cell {cell_id} has incomplete or failed outcomes",
        )
    cost_profile = _mapping(
        _read_json(qualification_dir / "cost_profile.json"),
        label="qualification cost profile",
    )
    stability = _mapping(
        _read_json(qualification_dir / "integration_stability.json"),
        label="qualification integration stability",
    )
    _require(report.get("cost_profile") == cost_profile, "report/cost-profile contents disagree")
    _require(report.get("integration_stability") == stability, "report/stability contents disagree")
    _require(stability.get("passed") is True, "integration stability did not pass")
    legacy_cost_passed = cost_profile.get("cost_gate_passed") is True
    _require(
        manifest.get("cost_gate_passed") is legacy_cost_passed,
        "manifest/cost-profile gate status disagrees",
    )
    # The authenticated user override is checked separately.  The original
    # qualification report may therefore be false solely because its legacy
    # 1000-core-hour ceiling was exceeded; scientific failures remain fatal.
    expected_legacy_pass = legacy_cost_passed
    _require(
        report.get("qualification_passed") is expected_legacy_pass
        and manifest.get("qualification_passed") is expected_legacy_pass,
        "qualification aggregate status is inconsistent with scientific and legacy-cost gates",
    )
    _require(report.get("contract_sha256") == launch.get("scientific_contract_sha256"), "report contract hash differs from launch")
    _require(report.get("spec_sha256") == launch.get("qualification_spec_sha256"), "report qualification-spec hash differs from launch")
    _require(report.get("receipt_sha256") == manifest.get("receipts_sha256"), "report/manifest receipt hashes disagree")
    _require(
        manifest.get("transition_manifest_file_sha256")
        == transition_identity.get("transition_manifest_file_sha256")
        and manifest.get("transition_receipt_sha256")
        == transition_identity.get("transition_receipt_sha256")
        and manifest.get("observed_prefix_contract_sha256")
        == transition_identity.get("observed_prefix_contract_sha256")
        and manifest.get("active_contract_sha256")
        == transition_identity.get("active_contract_sha256"),
        "qualification manifest does not bind the scientific transition",
    )
    return {
        "launch_path": str(launch_path),
        "qualification_dir": str(qualification_dir),
        "manifest_sha256": _sha256_file(manifest_path),
        "report_sha256": _sha256_file(report_path),
        "tier": "qualification",
        "repetitions_per_cell": repetitions,
        "cell_count": len(cells),
        "exact_frozen_grid_verified": True,
        "scientific_qualification_passed": True,
        "legacy_cost_gate_passed": legacy_cost_passed,
        "scientific_contract_transition": dict(transition_identity),
        "observed_prefix_contract_sha256": transition_identity[
            "observed_prefix_contract_sha256"
        ],
        "active_contract_sha256": transition_identity["active_contract_sha256"],
        "real_connectivity_accessed": False,
    }


def verify_scientific_contract_transition_gate(
    *,
    transition_manifest_path: Path,
    transition_manifest_file_sha256: str,
    transition_receipt_sha256: str,
    observed_run_root: Path,
    observed_attestation_dir: Path,
    observed_finalization_dir: Path,
    policy: EpisodePolicy,
    executor_root: Path,
    outputs_root: Path = OUTPUTS_ROOT,
) -> dict[str, Any]:
    """Authenticate the transition and require the exact rebuilt attestation."""

    episode_root = outputs_root.parent
    attestation = verify_attestation_root(
        run_root=observed_run_root,
        attestation_dir=observed_attestation_dir,
        policy=policy,
        outputs_root=outputs_root,
    )
    try:
        evidence = verify_scientific_contract_transition(
            episode_root=episode_root,
            executor_root=executor_root,
            manifest_path=transition_manifest_path,
            expected_manifest_file_sha256=transition_manifest_file_sha256,
            expected_transition_receipt_sha256=transition_receipt_sha256,
            observed_attestation_dir=observed_attestation_dir,
            observed_finalization_dir=observed_finalization_dir,
        )
    except ScientificTransitionBindingError as exc:
        raise ReadinessError(str(exc)) from exc
    _require(
        evidence.get("observed_attestation_manifest_file_sha256")
        == attestation.get("attestation_manifest_sha256")
        and evidence.get("observed_attestation_artifact_file_sha256")
        == attestation.get("attestation_artifact_file_sha256")
        and evidence.get("observed_attestation_sha256")
        == attestation.get("attestation_sha256")
        and evidence.get("observed_trial_count")
        == attestation.get("verified_trial_count")
        == 36
        and evidence.get("exact_36_attestation_set_matches_transition_evidence")
        is True,
        "scientific transition and rebuilt observed attestation differ",
    )
    return evidence


def verify_development_budget_override(
    *,
    override_path: Path,
    qualification_dir: Path,
    policy: EpisodePolicy,
    outputs_root: Path = OUTPUTS_ROOT,
    configuration_root: Path | None = None,
) -> dict[str, Any]:
    """Authenticate the explicit 2750-core-hour minimum-search authorization."""

    source_root = configuration_root or outputs_root
    override_path = _inside(
        override_path, source_root, label="development budget override"
    )
    qualification_dir = _inside(
        qualification_dir, outputs_root, label="qualification directory"
    )
    actual_hash = _sha256_file(override_path)
    _require(
        actual_hash == EXPECTED_BUDGET_OVERRIDE_SHA256,
        "development budget override hash differs from the authenticated authorization",
    )
    override = _mapping(load_yaml_subset(override_path), label="development budget override")
    _require(
        override.get("schema_version") == "ep12.development_budget_override.v1"
        and override.get("episode_id") == "ep12",
        "development budget override schema or episode mismatch",
    )
    _require(
        override.get("scope")
        == "minimum_full_development_search_including_observed_and_required_99_null_searches",
        "development budget override has the wrong scope",
    )
    _require(
        _same_number(
            override.get("measured_planning_core_hours"),
            AUTHORIZED_PLANNING_CORE_HOURS,
        )
        and override.get("user_rounded_planning_core_hours")
        == AUTHORIZED_ROUNDED_CORE_HOURS,
        "development budget override does not authenticate 2750 core-hours",
    )
    _require(
        _same_number(
            override.get("original_policy_cpu_core_hour_ceiling"),
            policy.cpu_core_hour_ceiling,
        )
        and override.get("original_cpu_ceiling_enforced_for_this_scope") is False
        and override.get("cpu_budget_stop_disabled_until_minimum_full_search_complete")
        is True
        and override.get("new_hard_cpu_ceiling") is None,
        "development budget override does not disable only the original CPU ceiling",
    )
    _require(
        override.get("final_connectivity_access_authorized") is False,
        "development budget override authorizes final connectivity",
    )
    unchanged = _mapping(override.get("unchanged_limits"), label="unchanged budget limits")
    expected_unchanged = {
        "minimum_valid_trials": policy.minimum_valid_trials,
        "maximum_valid_trials": policy.maximum_valid_trials,
        "cpu_cores_per_trial_max": policy.cpu_cores_per_trial_maximum,
        "memory_gib_per_trial_max": policy.memory_gib_per_trial_maximum,
        "wall_hours_per_trial_max": policy.wall_hours_per_trial_maximum,
        "wall_clock_hour_ceiling": 120,
        "gpu_hour_ceiling": policy.gpu_hour_ceiling,
    }
    for name, expected in expected_unchanged.items():
        _require(
            _same_number(unchanged.get(name), float(expected)),
            f"development budget override changes frozen limit {name}",
        )
    cost_profile = _mapping(
        _read_json(qualification_dir / "cost_profile.json"),
        label="qualification cost profile",
    )
    try:
        measured_projection = float(cost_profile["projected_cpu_core_hours"])
    except (KeyError, TypeError, ValueError) as error:
        raise ReadinessError(
            "qualification cost projection is missing or nonnumeric"
        ) from error
    _require(
        math.isfinite(measured_projection)
        and measured_projection > 0.0
        and cost_profile.get("scope")
        == "generated_data_measured_projection_not_real_runtime",
        "qualification cost profile is not a positive generated-data measurement",
    )
    _require(
        _same_number(
            cost_profile.get("policy_cpu_core_hour_ceiling"),
            policy.cpu_core_hour_ceiling,
        )
        and cost_profile.get("cost_gate_passed")
        is (measured_projection <= policy.cpu_core_hour_ceiling),
        "qualification legacy cost status is inconsistent with its measured projection",
    )
    return {
        "override_path": str(override_path),
        "override_sha256": actual_hash,
        "authorized_planning_core_hours": AUTHORIZED_PLANNING_CORE_HOURS,
        "user_rounded_planning_core_hours": AUTHORIZED_ROUNDED_CORE_HOURS,
        "qualification_projected_cpu_core_hours": measured_projection,
        "qualification_projection_recorded_not_used_as_hard_ceiling": True,
        "legacy_cost_gate_passed": bool(cost_profile["cost_gate_passed"]),
        "scope": override["scope"],
        "minimum_full_search_cpu_stop_disabled": True,
        "final_connectivity_accessed": False,
    }


def _load_configuration_payload(path: Path, *, expected_cycle: int | None) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if expected_cycle is None:
        _require(isinstance(payload, list), f"coverage configuration file has the wrong schema: {path}")
        configurations = payload
        hashes = [digest_object(value) for value in configurations]
    else:
        container = _mapping(payload, label=f"adaptive cycle {expected_cycle} configurations")
        _require(container.get("schema_version") == "ep12.adaptive_successor_configurations.v1", f"adaptive cycle {expected_cycle} configuration schema mismatch")
        _require(int(container.get("cycle", -1)) == expected_cycle, f"adaptive cycle {expected_cycle} identity mismatch")
        _require(container.get("final_connectivity_accessed") is False, f"adaptive cycle {expected_cycle} records final-connectivity access")
        configurations = container.get("configurations")
        hashes = container.get("configuration_sha256")
        _require(isinstance(configurations, list), f"adaptive cycle {expected_cycle} configurations are missing")
        _require(isinstance(hashes, list) and len(hashes) == len(configurations), f"adaptive cycle {expected_cycle} configuration hashes are incomplete")
    normalized: list[dict[str, Any]] = []
    for index, (configuration, expected_hash) in enumerate(zip(configurations, hashes)):
        _require(isinstance(configuration, Mapping), f"configuration {index} is not an object")
        record = dict(configuration)
        _require(digest_object(record) == str(expected_hash), f"configuration {record.get('trial_id', index)} hash mismatch")
        normalized.append(record)
    return normalized


def _verify_plan_manifest(plan_dir: Path, configuration_name: str) -> str:
    manifest_path = plan_dir / "manifest.json"
    manifest = _mapping(_read_json(manifest_path), label=f"plan manifest {plan_dir}")
    _require(manifest.get("final_connectivity_accessed") is False, f"plan manifest records final-connectivity access: {plan_dir}")
    files = _mapping(manifest.get("files"), label=f"plan manifest files {plan_dir}")
    _require(configuration_name in files, f"plan manifest does not bind {configuration_name}")
    for name, expected in files.items():
        _verify_hash(plan_dir / str(name), expected, label=f"plan artifact {plan_dir / str(name)}")
    return _sha256_file(manifest_path)


def _validate_configuration_shape(
    configuration: Mapping[str, Any],
    *,
    policy: EpisodePolicy,
    stage: str,
) -> None:
    trial_id = configuration.get("trial_id")
    _require(isinstance(trial_id, str) and trial_id, "configuration lacks a trial_id")
    _require(configuration.get("data_mode") == "development", f"{trial_id} is not a development-only configuration")
    _require(tuple(configuration.get("models", ())) == policy.model_triplet, f"{trial_id} does not contain the complete T/U/M triplet")
    _require(tuple(configuration.get("nuisance", ())) == policy.nuisance_terms, f"{trial_id} changes the prespecified nuisance families")
    for name, choices in policy.grammar_choices.items():
        _require(configuration.get(name) in choices, f"{trial_id} has an out-of-grammar value for {name}")
    if stage == "coverage":
        _require(configuration.get("stage") == "coverage", f"{trial_id} is not a coverage-stage configuration")
        _require(trial_id.startswith("coverage_"), f"coverage trial has a noncoverage identifier: {trial_id}")
    else:
        cycle = "1" if stage.endswith("1") else "2"
        _require(trial_id.startswith(f"adaptive_c{cycle}_"), f"{stage} trial identity mismatch: {trial_id}")


def scientific_configuration_sha256(configuration: Mapping[str, Any]) -> str:
    """Hash scientific choices while excluding only trial identity and stage."""

    if not isinstance(configuration, Mapping):
        raise ReadinessError("scientific configuration must be an object")
    return digest_object(
        {
            key: value
            for key, value in configuration.items()
            if key not in {"trial_id", "stage"}
        }
    )


def _verify_attested_trial(
    configuration: Mapping[str, Any],
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a declared configuration to one preverified status-only row."""

    trial_id = str(configuration["trial_id"])
    configuration_sha256 = digest_object(configuration)
    scientific_sha256 = scientific_configuration_sha256(configuration)
    _require(attestation.get("trial_id") == trial_id, f"attestation identity mismatch for {trial_id}")
    _require(
        attestation.get("configuration_sha256") == configuration_sha256,
        f"attestation/configuration hash mismatch for {trial_id}",
    )
    _require(
        attestation.get("scientific_configuration_sha256") == scientific_sha256,
        f"attestation scientific signature mismatch for {trial_id}",
    )
    _require(
        attestation.get("status") == "valid_complete_T_U_M_triplet"
        and attestation.get("complete_T_U_M_triplet") is True
        and attestation.get("technical_failure_count") == 0
        and attestation.get("data_role") == "development_only"
        and attestation.get("final_connectivity_accessed") is False,
        f"attestation structural/status gate failed for {trial_id}",
    )
    return {
        "trial_id": trial_id,
        "configuration_sha256": configuration_sha256,
        "scientific_configuration_sha256": scientific_sha256,
        "complete_T_U_M_triplet": True,
        "technical_failure_count": 0,
        "source_manifest_sha256": attestation["source_manifest_sha256"],
    }


def verify_observed_provisional_chain(
    *,
    run_root: Path,
    observed_finalization_dir: Path,
    observed_attestation_dir: Path,
    policy: EpisodePolicy,
    outputs_root: Path = OUTPUTS_ROOT,
) -> dict[str, Any]:
    """Authenticate the 18+9+9 chain without reading scientific outcomes."""

    run_root = _inside(run_root, outputs_root, label="observed run root")
    observed_finalization_dir = _inside(
        observed_finalization_dir, outputs_root, label="observed finalization directory"
    )
    launch = _mapping(
        _read_json(run_root / "adaptive_dag_launch.json"), label="adaptive DAG launch"
    )
    _require(launch.get("schema_version") == "ep12.adaptive_dag_launch.v1", "adaptive DAG launch schema mismatch")
    _require(int(launch.get("minimum_observed_trial_count_after_success", -1)) == policy.minimum_valid_trials, "adaptive DAG minimum differs from policy")
    _require(launch.get("final_connectivity_access_authorized") is False, "adaptive DAG authorizes final connectivity")

    attestation_evidence = verify_attestation_root(
        run_root=run_root,
        attestation_dir=observed_attestation_dir,
        policy=policy,
        outputs_root=outputs_root,
    )
    attestation_rows = attestation_evidence.pop("_trial_rows_by_id")

    trial_ids: list[str] = []
    configuration_hashes: list[str] = []
    scientific_configuration_hashes: list[str] = []
    stage_evidence: list[dict[str, Any]] = []
    for stage, expected_count, relative_config, relative_results, expected_cycle in _CHAIN_SOURCES:
        configuration_path = run_root / relative_config
        plan_manifest_sha256 = _verify_plan_manifest(
            configuration_path.parent, configuration_path.name
        )
        configurations = _load_configuration_payload(
            configuration_path, expected_cycle=expected_cycle
        )
        _require(len(configurations) == expected_count, f"{stage} expected {expected_count} configurations, found {len(configurations)}")
        stage_trials: list[dict[str, Any]] = []
        for configuration in configurations:
            _validate_configuration_shape(configuration, policy=policy, stage=stage)
            trial_id = str(configuration["trial_id"])
            configuration_hash = digest_object(configuration)
            trial_ids.append(trial_id)
            configuration_hashes.append(configuration_hash)
            _require(
                trial_id in attestation_rows,
                f"separate attestation omits declared trial {trial_id}",
            )
            trial_evidence = _verify_attested_trial(
                configuration,
                _mapping(attestation_rows[trial_id], label=f"attestation row {trial_id}"),
            )
            _require(
                trial_evidence["configuration_sha256"] == configuration_hash,
                f"trial {trial_id} attestation does not bind its declared configuration",
            )
            scientific_configuration_hashes.append(
                str(trial_evidence["scientific_configuration_sha256"])
            )
            stage_trials.append(trial_evidence)
        stage_evidence.append(
            {
                "stage": stage,
                "configuration_path": str(configuration_path),
                "configuration_file_sha256": _sha256_file(configuration_path),
                "plan_manifest_sha256": plan_manifest_sha256,
                "trial_count": len(stage_trials),
                "trial_manifest_sha256": {
                    item["trial_id"]: item["source_manifest_sha256"]
                    for item in stage_trials
                },
            }
        )
    _require(len(trial_ids) == policy.minimum_valid_trials, "observed provisional chain does not contain the policy minimum")
    _require(len(set(trial_ids)) == len(trial_ids), "observed provisional chain repeats a trial ID")
    _require(len(set(configuration_hashes)) == len(configuration_hashes), "observed provisional chain repeats a scientific configuration")
    _require(
        len(set(scientific_configuration_hashes))
        == len(scientific_configuration_hashes),
        "observed provisional chain repeats a scientific configuration after excluding identity/stage",
    )

    finalization_manifest_path = observed_finalization_dir / "manifest.json"
    finalization_manifest = _mapping(
        _read_json(finalization_manifest_path), label="observed finalization manifest"
    )
    _require(finalization_manifest.get("schema_version") == "ep12.observed_minimum_finalization_manifest.v1", "observed finalization manifest schema mismatch")
    _require(finalization_manifest.get("final_connectivity_accessed") is False, "observed finalization records final-connectivity access")
    finalization_files = _mapping(finalization_manifest.get("files"), label="observed finalization files")
    _require(set(finalization_files) == {"observed_search_archive.json"}, "observed finalization manifest has an unexpected file set")
    _verify_hash(
        observed_finalization_dir / "observed_search_archive.json",
        finalization_files["observed_search_archive.json"],
        label="opaque observed-search archive",
    )
    # The archive contains observed objectives.  Its bytes are authenticated
    # above but are intentionally not decoded here.
    return {
        "run_root": str(run_root),
        "verified_trial_count": len(trial_ids),
        "coverage_trial_count": stage_evidence[0]["trial_count"],
        "adaptive_cycle_1_trial_count": stage_evidence[1]["trial_count"],
        "adaptive_cycle_2_trial_count": stage_evidence[2]["trial_count"],
        "unique_trial_ids": True,
        "unique_configuration_hashes": True,
        "unique_scientific_configuration_hashes": True,
        "trial_attestations_verified": True,
        "separate_observed_attestation": attestation_evidence,
        "source_status_metadata_parsed": True,
        "outcome_values_read": False,
        "outcome_values_compared": False,
        "outcome_values_emitted": False,
        "observed_search_archive_parsed": False,
        "observed_finalization_manifest_sha256": _sha256_file(
            finalization_manifest_path
        ),
        "stages": stage_evidence,
        "provisional_only": True,
        "required_falsifiers_completed": False,
        "full_search_null_completed": False,
        "procedure_locked": False,
        "final_connectivity_accessed": False,
    }


def _normalized_plan_hash(payload: Mapping[str, Any]) -> str:
    material = {key: value for key, value in payload.items() if key != "plan_sha256"}
    return digest_object(material)


def _normalized_slot_hash(slot: Mapping[str, Any]) -> str:
    material = {key: value for key, value in slot.items() if key != "slot_sha256"}
    return digest_object(material)


def verify_falsifier_plan(
    *,
    plan_path: Path,
    policy: EpisodePolicy,
    outputs_root: Path = OUTPUTS_ROOT,
    configuration_root: Path | None = None,
) -> dict[str, Any]:
    """Verify presence, identity, coverage, and deterministic slot metadata."""

    source_root = configuration_root or outputs_root
    plan_path = _inside(plan_path, source_root, label="falsifier plan")
    payload = _mapping(_read_json(plan_path), label="falsifier plan")
    frozen_evidence = frozen_tail_plan.validate_plan(
        payload, policy_path=policy.path
    )
    _require(
        payload.get("plan_sha256") == frozen_tail_plan.EXPECTED_PLAN_SHA256,
        "falsifier plan hash differs from the frozen expected hash",
    )
    _require(payload.get("schema_version") == FALSIFIER_PLAN_SCHEMA, "unsupported falsifier-plan schema")
    _require(payload.get("plan_sha256") == _normalized_plan_hash(payload), "falsifier plan self-hash mismatch")
    identity = _mapping(payload.get("policy_identity"), label="falsifier plan policy identity")
    _require(
        identity.get("path") == "SEARCH_POLICY.yaml",
        "falsifier plan does not identify the canonical EP12 search policy",
    )
    if "policy_id" in identity:
        _require(
            identity.get("policy_id") == policy.policy_id,
            "falsifier plan policy ID mismatch",
        )
    policy_hash = identity.get(
        "sha256", identity.get("policy_sha256", identity.get("policy_hash"))
    )
    _require(policy_hash == policy.policy_hash, "falsifier plan policy hash mismatch")
    slots = payload.get("all_slots")
    _require(isinstance(slots, list) and slots, "falsifier plan has no executable slots")
    trial_ids: list[str] = []
    ordinals: list[int] = []
    falsifier_ids: list[str] = []
    for raw_slot in slots:
        slot = _mapping(raw_slot, label="falsifier plan slot")
        _require(slot.get("slot_sha256") == _normalized_slot_hash(slot), f"falsifier slot self-hash mismatch: {slot.get('trial_id')}")
        trial_id = slot.get("trial_id")
        _require(isinstance(trial_id, str) and trial_id, "falsifier slot lacks a trial ID")
        ordinal = int(slot.get("valid_trial_ordinal", 0))
        _require(ordinal > policy.minimum_valid_trials, f"post-36 slot has invalid ordinal: {ordinal}")
        selector = slot.get("selector")
        _require(selector not in (None, "", {}), f"falsifier slot lacks a deterministic selector: {trial_id}")
        serialized_selector = canonical_json(selector).lower()
        _require("manual" not in serialized_selector and "human" not in serialized_selector, f"falsifier slot requires manual selection: {trial_id}")
        evidence = slot.get("required_realization_evidence")
        _require(isinstance(evidence, list) and evidence, f"falsifier slot has no realization evidence contract: {trial_id}")
        falsifier_id = slot.get("falsifier_id")
        if falsifier_id is not None:
            _require(isinstance(falsifier_id, str) and falsifier_id, f"invalid falsifier ID in {trial_id}")
            falsifier_ids.append(falsifier_id)
        trial_ids.append(trial_id)
        ordinals.append(ordinal)
    _require(len(set(trial_ids)) == len(trial_ids), "falsifier plan repeats a trial ID")
    _require(len(set(ordinals)) == len(ordinals), "falsifier plan repeats a valid-trial ordinal")
    _require(ordinals == sorted(ordinals), "falsifier plan ordinals are not ordered")
    non_trial = payload.get("non_trial_required_gates")
    _require(isinstance(non_trial, list), "falsifier plan omits non-trial required gates")
    non_trial_ids = {
        str(item.get("falsifier_id"))
        for item in non_trial
        if isinstance(item, Mapping) and item.get("falsifier_id")
    }
    classification = _mapping(
        payload.get("required_falsifier_classification"),
        label="falsifier classification",
    )
    classified = classification.get("all_policy_ids")
    _require(
        isinstance(classified, list)
        and set(map(str, classified)) == set(policy.required_falsifiers),
        "falsifier classification differs from SEARCH_POLICY",
    )
    covered_ids = set(falsifier_ids) | non_trial_ids
    missing = sorted(set(policy.required_falsifiers) - covered_ids)
    _require(not missing, f"falsifier plan omits required policy controls: {', '.join(missing)}")
    arithmetic = _mapping(payload.get("policy_arithmetic"), label="falsifier plan arithmetic")
    fraction = arithmetic.get("post_coverage_falsifier_fraction")
    _require(
        isinstance(fraction, Mapping),
        "falsifier plan omits exact post-coverage fraction arithmetic",
    )
    numerator = int(fraction.get("numerator", -1))
    denominator = int(fraction.get("denominator", 0))
    _require(
        denominator > 0
        and numerator / denominator == policy.falsifier_fraction_minimum,
        "falsifier plan fraction differs from policy",
    )
    _require(
        int(arithmetic.get("minimum_valid_trials", -1))
        == policy.minimum_valid_trials
        and int(arithmetic.get("maximum_valid_trials", -1))
        == policy.maximum_valid_trials
        and int(arithmetic.get("patience_valid_trials", -1))
        == policy.patience_valid_trials,
        "falsifier plan trial or patience arithmetic differs from policy",
    )
    _require(
        ordinals
        == list(range(policy.minimum_valid_trials + 1, policy.maximum_valid_trials + 1)),
        "falsifier plan does not cover every reserve ordinal through the policy maximum",
    )
    execution_contract = _mapping(
        payload.get("execution_contract"), label="falsifier execution contract"
    )
    _require(
        execution_contract.get("final_connectivity_accessed") is False
        and execution_contract.get("final_connectivity_access_authorized") is False,
        "falsifier plan crosses the final-connectivity firewall",
    )
    return {
        "plan_path": str(plan_path),
        "plan_sha256": _sha256_file(plan_path),
        "self_hash": payload["plan_sha256"],
        "frozen_validator": frozen_evidence,
        "slot_count": len(slots),
        "planned_falsifier_slot_count": len(falsifier_ids),
        "required_falsifier_ids_covered": sorted(covered_ids),
        "deterministic_selectors_present": True,
        "realization_evidence_contracts_present": True,
        "plan_only_not_execution": True,
        "final_connectivity_accessed": False,
    }


def evaluate_production_readiness(
    *,
    qualification_launch_path: Path,
    qualification_dir: Path,
    observed_run_root: Path,
    observed_finalization_dir: Path,
    observed_attestation_dir: Path,
    transition_manifest_path: Path,
    transition_manifest_file_sha256: str,
    transition_receipt_sha256: str,
    falsifier_plan_path: Path,
    budget_override_path: Path = OUTPUTS_ROOT
    / "executor"
    / "DEVELOPMENT_BUDGET_OVERRIDE.yaml",
    policy_path: Path = EPISODE_ROOT / "SEARCH_POLICY.yaml",
    outputs_root: Path = OUTPUTS_ROOT,
    executor_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the five outcome-blind gates and emit strict claim guards."""

    policy = EpisodePolicy.load(policy_path)
    frozen_executor_root = (
        executor_root or Path(__file__).resolve().parents[1]
    ).resolve()
    configuration_root = outputs_root
    if executor_root is not None:
        expected_policy_path = frozen_executor_root.parents[1] / "SEARCH_POLICY.yaml"
        _require(
            Path(policy_path).resolve() == expected_policy_path.resolve(),
            "policy path is not paired with the authenticated frozen executor root",
        )
        configuration_root = frozen_executor_root
    checks = [
        _check(
            "scientific_contract_transition",
            verify_scientific_contract_transition_gate,
            transition_manifest_path=transition_manifest_path,
            transition_manifest_file_sha256=transition_manifest_file_sha256,
            transition_receipt_sha256=transition_receipt_sha256,
            observed_run_root=observed_run_root,
            observed_attestation_dir=observed_attestation_dir,
            observed_finalization_dir=observed_finalization_dir,
            policy=policy,
            executor_root=frozen_executor_root,
            outputs_root=outputs_root,
        ),
        _check(
            "full_synthetic_qualification",
            verify_full_synthetic_qualification,
            launch_path=qualification_launch_path,
            qualification_dir=qualification_dir,
            outputs_root=outputs_root,
            executor_root=frozen_executor_root,
        ),
        _check(
            "observed_36_trial_provisional_chain",
            verify_observed_provisional_chain,
            run_root=observed_run_root,
            observed_finalization_dir=observed_finalization_dir,
            observed_attestation_dir=observed_attestation_dir,
            policy=policy,
            outputs_root=outputs_root,
        ),
        _check(
            "post36_falsifier_plan",
            verify_falsifier_plan,
            plan_path=falsifier_plan_path,
            policy=policy,
            outputs_root=outputs_root,
            configuration_root=configuration_root,
        ),
        _check(
            "development_budget_override",
            verify_development_budget_override,
            override_path=budget_override_path,
            qualification_dir=qualification_dir,
            policy=policy,
            outputs_root=outputs_root,
            configuration_root=configuration_root,
        ),
    ]
    failed = [str(check["check_id"]) for check in checks if not check["passed"]]
    tail_permitted = not failed
    transition_check = checks[0]
    transition_evidence = (
        dict(transition_check["evidence"])
        if transition_check["passed"]
        else {}
    )
    unavailable_after_plan = [
        "post36_falsifier_trials_executed_and_authenticated",
        "patience_and_stop_rule_realized",
        "mode_support_and_all_required_controls_passed",
        "one_complete_comparison_selected_and_locked",
    ]
    report: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA,
        "episode_id": "ep12",
        "policy_id": policy.policy_id,
        "policy_sha256": policy.policy_hash,
        "gate_scope": "outcome_blind_readiness_for_observed_post36_tail_only",
        "checks": checks,
        "failed_foundational_checks": failed,
        "production_readiness_passed": tail_permitted,
        "observed_prefix_contract_sha256": transition_evidence.get(
            "observed_prefix_contract_sha256"
        ),
        "active_contract_sha256": transition_evidence.get(
            "active_contract_sha256"
        ),
        "transition_manifest_file_sha256": transition_evidence.get(
            "transition_manifest_file_sha256"
        ),
        "transition_receipt_file_sha256": transition_evidence.get(
            "transition_receipt_file_sha256"
        ),
        "transition_receipt_sha256": transition_evidence.get(
            "transition_receipt_sha256"
        ),
        "claim_guards": {
            "observed_post36_tail_launch": {
                "permitted": tail_permitted,
                "missing": failed,
            },
            "procedure_lock": {
                "permitted": False,
                "missing": failed + unavailable_after_plan,
            },
            "full_search_null_launch": {
                "permitted": False,
                "missing": failed
                + unavailable_after_plan
                + ["conditional_null_eligibility_decision_and_null_runner_lock"],
            },
            "final_connectivity_access": {
                "permitted": False,
                "missing": failed
                + unavailable_after_plan
                + [
                    "completed_full_search_null_when_required",
                    "one_shot_final_evaluator_lock",
                    "explicit_final_access_authorization",
                ],
            },
        },
        "observed_outcome_values_read": False,
        "falsifier_plan_counts_as_execution": False,
        "procedure_locked": False,
        "full_search_null_completed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return report


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _json_artifact_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_or_verify_exact(path: Path, content: bytes, *, label: str) -> None:
    if path.exists():
        _require(path.is_file(), f"existing {label} is not a regular file: {path}")
        _require(path.read_bytes() == content, f"existing {label} differs from retry payload")
        return
    _atomic_write_bytes(path, content)


def write_readiness_artifacts(output_dir: Path, report: Mapping[str, Any], *, outputs_root: Path = OUTPUTS_ROOT) -> None:
    """Atomically create or recover the two-file readiness transaction."""

    output_dir = _inside(output_dir, outputs_root, label="readiness output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "production_readiness.json"
    report_bytes = _json_artifact_bytes(report)
    _write_or_verify_exact(report_path, report_bytes, label="readiness report")
    manifest = {
        "schema_version": READINESS_MANIFEST_SCHEMA,
        "files": {"production_readiness.json": digest_bytes(report_bytes)},
        "production_readiness_passed": report.get("production_readiness_passed") is True,
        "observed_prefix_contract_sha256": report.get(
            "observed_prefix_contract_sha256"
        ),
        "active_contract_sha256": report.get("active_contract_sha256"),
        "transition_manifest_file_sha256": report.get(
            "transition_manifest_file_sha256"
        ),
        "transition_receipt_file_sha256": report.get(
            "transition_receipt_file_sha256"
        ),
        "transition_receipt_sha256": report.get("transition_receipt_sha256"),
        "procedure_locked": False,
        "full_search_null_completed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_bytes = _json_artifact_bytes(manifest)
    _write_or_verify_exact(manifest_path, manifest_bytes, label="readiness manifest")


def claim_is_permitted(report: Mapping[str, Any], claim: str) -> bool:
    aliases = {
        "observed_tail": "observed_post36_tail_launch",
        "procedure_lock": "procedure_lock",
        "full_search_null": "full_search_null_launch",
        "final_connectivity": "final_connectivity_access",
    }
    key = aliases[claim]
    guards = _mapping(report.get("claim_guards"), label="claim guards")
    guard = _mapping(guards.get(key), label=f"claim guard {key}")
    return guard.get("permitted") is True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ep12-production-readiness")
    parser.add_argument("--qualification-launch", type=Path, required=True)
    parser.add_argument("--qualification-dir", type=Path, required=True)
    parser.add_argument("--observed-run-root", type=Path, required=True)
    parser.add_argument("--observed-finalization-dir", type=Path, required=True)
    parser.add_argument("--observed-attestation-dir", type=Path, required=True)
    parser.add_argument("--transition-manifest", type=Path, required=True)
    parser.add_argument("--transition-manifest-sha256", required=True)
    parser.add_argument("--transition-receipt-sha256", required=True)
    parser.add_argument("--falsifier-plan", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--executor-root", type=Path, required=True)
    parser.add_argument(
        "--budget-override",
        type=Path,
        default=OUTPUTS_ROOT / "executor" / "DEVELOPMENT_BUDGET_OVERRIDE.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-for",
        choices=("observed_tail", "procedure_lock", "full_search_null", "final_connectivity"),
        default="observed_tail",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = evaluate_production_readiness(
        qualification_launch_path=arguments.qualification_launch,
        qualification_dir=arguments.qualification_dir,
        observed_run_root=arguments.observed_run_root,
        observed_finalization_dir=arguments.observed_finalization_dir,
        observed_attestation_dir=arguments.observed_attestation_dir,
        transition_manifest_path=arguments.transition_manifest,
        transition_manifest_file_sha256=arguments.transition_manifest_sha256,
        transition_receipt_sha256=arguments.transition_receipt_sha256,
        falsifier_plan_path=arguments.falsifier_plan,
        budget_override_path=arguments.budget_override,
        policy_path=arguments.policy,
        executor_root=arguments.executor_root,
    )
    write_readiness_artifacts(arguments.output_dir, report)
    permitted = claim_is_permitted(report, arguments.require_for)
    print(
        json.dumps(
            {
                "production_readiness_passed": report["production_readiness_passed"],
                "required_claim": arguments.require_for,
                "required_claim_permitted": permitted,
                "failed_foundational_checks": report["failed_foundational_checks"],
                "procedure_locked": False,
                "full_search_null_completed": False,
                "final_connectivity_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0 if permitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
