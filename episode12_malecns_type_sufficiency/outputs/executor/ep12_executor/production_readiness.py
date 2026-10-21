"""Lean, outcome-blind readiness checks for the EP12 observed tail.

Readiness is a scientific checklist, not a provenance transaction. This
module checks the completed development prefix, synthetic qualification,
frozen falsifier plan, contract roles, and resource authorization without
creating hashes, manifests, receipts, schemas, or environment attestations.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import observed_tail_plan as frozen_tail_plan

from .adaptive_search import validate_development_configuration
from .canonical_target import DECODER_PROTOCOL
from .contract_roles import RANK_RULE, load_contract_provenance
from .policy import EpisodePolicy
from .runtime import EPISODE_ROOT, atomic_json
from .scientific_qualification import _accept_cell, _cell_summary, qualification_cells
from .yaml_subset import load_yaml_subset


OUTPUTS_ROOT = EPISODE_ROOT / "outputs"
QUALIFICATION_SPEC = (
    OUTPUTS_ROOT / "executor" / "SYNTHETIC_QUALIFICATION_SPEC_POST36_V3.yaml"
)
QUALIFICATION_ARTIFACTS = (
    "replicate_results.jsonl",
    "cost_profile.json",
    "integration_stability.json",
    "result_table.csv",
    "result_table.md",
    "qualification_report.json",
)
QUALIFICATION_NONFAILURE_TERMINALS = frozenset(
    {
        "M_supported_before_full_search_null",
        "single_population_adequate_before_sensitivity_gate",
        "unresolved",
    }
)
REQUIRED_READINESS_CHECK_IDS = (
    "scientific_contract_roles",
    "full_synthetic_qualification",
    "canonical_pipeline_qualification",
    "observed_36_trial_provisional_chain",
    "post36_falsifier_plan",
    "development_resource_authorization",
)
_CHAIN_SOURCES = (
    ("coverage", 18, Path("search_plan/coverage_configurations.json"), Path("observed")),
    (
        "adaptive_cycle_1",
        9,
        Path("adaptive/cycle1/plan/successor_configurations.json"),
        Path("adaptive/cycle1/observed"),
    ),
    (
        "adaptive_cycle_2",
        9,
        Path("adaptive/cycle2/plan/successor_configurations.json"),
        Path("adaptive/cycle2/observed"),
    ),
)


class ReadinessError(ValueError):
    """A required scientific readiness fact is missing or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    boundary = root.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ReadinessError(f"{label} must stay under {boundary}: {resolved}")
    return resolved


def _read_json(path: Path, *, label: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReadinessError(f"cannot read {label}: {path}: {error}") from error


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return dict(value)


def _check(
    check_id: str, function: Callable[..., Mapping[str, Any]], **kwargs: Any
) -> dict[str, Any]:
    try:
        evidence = dict(function(**kwargs))
    except (ReadinessError, ValueError, OSError, KeyError, TypeError) as error:
        return {"check_id": check_id, "passed": False, "reason": str(error)}
    return {"check_id": check_id, "passed": True, "evidence": evidence}


def verify_contract_roles(
    *, prefix_contract_path: Path, active_contract_path: Path
) -> dict[str, Any]:
    roles = load_contract_provenance(
        prefix_contract_path=prefix_contract_path,
        active_contract_path=active_contract_path,
    )
    prefix = load_yaml_subset(prefix_contract_path)
    active = load_yaml_subset(active_contract_path)
    for label, contract in (("prefix", prefix), ("active", active)):
        _require(
            contract.get("real_connectivity_access_authorized") == "development_only",
            f"{label} contract is not development-only",
        )
        _require(
            contract.get("final_connectivity_access_authorized", False) is False,
            f"{label} contract authorizes final connectivity",
        )
    _require(
        active.get("mode_support", {}).get(
            "prevalence_parameter_rank_definition"
        )
        == RANK_RULE,
        "active contract changed the frozen prevalence-rank definition",
    )
    return {
        "prefix_contract_id": prefix.get("contract_id"),
        "active_contract_id": active.get("contract_id"),
        "prevalence_parameter_rank_definition": RANK_RULE,
        "development_roles_only": True,
        "final_connectivity_accessed": False,
        "contract_roles": roles,
    }


def verify_full_synthetic_qualification(
    *,
    qualification_dir: Path,
    qualification_spec_path: Path = QUALIFICATION_SPEC,
    outputs_root: Path = OUTPUTS_ROOT,
    **_: Any,
) -> dict[str, Any]:
    directory = _inside(qualification_dir, outputs_root, label="qualification directory")
    artifacts = {name: directory / name for name in QUALIFICATION_ARTIFACTS}
    for name, path in artifacts.items():
        _require(path.is_file() and not path.is_symlink(), f"missing qualification artifact: {name}")

    spec = load_yaml_subset(qualification_spec_path)
    expected_cells = qualification_cells("qualification", spec)
    expected_cell_ids = {cell.cell_id for cell in expected_cells}
    expected_repetitions = int(
        spec["run_tiers"]["qualification_replicates_per_core_error_cell"]
    )
    expected_whole_types = int(spec["run_tiers"]["whole_types_per_replicate"])
    expected_record_count = len(expected_cells) * expected_repetitions
    report = _mapping(
        _read_json(artifacts["qualification_report.json"], label="qualification report"),
        label="qualification report",
    )
    oracle_path_value = report.get("stochastic_oracle_aggregate_path")
    _require(
        isinstance(oracle_path_value, str) and oracle_path_value,
        "qualification report lacks its stochastic-oracle aggregate",
    )
    oracle_path = _inside(
        Path(oracle_path_value), outputs_root, label="stochastic-oracle aggregate"
    )
    oracle_report = _mapping(
        _read_json(oracle_path, label="stochastic-oracle aggregate"),
        label="stochastic-oracle aggregate",
    )
    from .coverage_successor import coverage_evidence_by_cell

    oracle_by_cell, coverage_error_budget = coverage_evidence_by_cell(
        oracle_report,
        expected_cell_ids=sorted(expected_cell_ids),
    )
    _require(report.get("tier") == "qualification", "qualification tier is not full")
    _require(
        report.get("qualification_scope") == "actual_T_U_M_generated_data_only",
        "qualification scope is not the actual generated T/U/M fitter",
    )
    _require(
        report.get("contract_configuration") is True
        and report.get("scientific_contract_enforced") is True,
        "qualification did not use the frozen scientific contract configuration",
    )
    cells = report.get("cells")
    _require(isinstance(cells, list), "qualification cells must be a list")
    _require(
        len(cells) == len(expected_cells),
        f"qualification has {len(cells)} cells; expected {len(expected_cells)}",
    )
    observed_cell_ids = [
        str(cell.get("cell_id", "")) if isinstance(cell, Mapping) else ""
        for cell in cells
    ]
    _require(
        len(set(observed_cell_ids)) == len(observed_cell_ids)
        and set(observed_cell_ids) == expected_cell_ids,
        "qualification cell inventory differs from the frozen full tier",
    )
    _require(
        report.get("qualification_passed") is True,
        "scientific qualification has not passed",
    )
    _require(
        all(isinstance(cell, Mapping) and cell.get("passed") is True for cell in cells),
        "one or more scientific qualification cells failed",
    )
    _require(
        all(int(cell.get("replicates", -1)) == expected_repetitions for cell in cells),
        "one or more qualification cells has the wrong replicate count",
    )
    settings = _mapping(report.get("settings"), label="qualification settings")
    _require(
        int(settings.get("repetitions_per_cell", -1)) == expected_repetitions
        and int(settings.get("whole_types_per_replicate", -1)) == expected_whole_types,
        "qualification settings differ from the frozen full tier",
    )

    cost_profile = _mapping(report.get("cost_profile"), label="qualification cost profile")
    _require(cost_profile.get("cost_gate_passed") is True, "qualification cost gate failed")
    _require(
        int(cost_profile.get("measured_type_direction_triplets", -1))
        == expected_record_count * expected_whole_types * 2,
        "qualification cost profile has the wrong measured triplet count",
    )
    _require(
        _read_json(artifacts["cost_profile.json"], label="cost profile artifact")
        == cost_profile,
        "cost profile artifact differs from the qualification report",
    )
    stability = _mapping(
        report.get("integration_stability"), label="integration stability"
    )
    _require(stability.get("passed") is True, "integration stability failed")
    _require(
        _read_json(
            artifacts["integration_stability.json"],
            label="integration stability artifact",
        )
        == stability,
        "integration stability artifact differs from the qualification report",
    )

    seen_records: set[tuple[str, int]] = set()
    records_by_cell: dict[str, list[dict[str, Any]]] = {
        cell.cell_id: [] for cell in expected_cells
    }
    try:
        with artifacts["replicate_results.jsonl"].open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = _mapping(json.loads(line), label=f"qualification record {line_number}")
                cell_id = str(record.get("cell_id", ""))
                replicate_index = int(record.get("replicate_index", -1))
                key = (cell_id, replicate_index)
                _require(cell_id in expected_cell_ids, "qualification record has an unknown cell")
                _require(
                    0 <= replicate_index < expected_repetitions,
                    "qualification record has an invalid replicate index",
                )
                _require(key not in seen_records, "qualification replicate record is duplicated")
                _require(
                    record.get("terminal") in QUALIFICATION_NONFAILURE_TERMINALS,
                    "qualification replicate lacks a valid nonfailure terminal",
                )
                _mapping(
                    record.get("intervals"),
                    label="qualification replicate intervals",
                )
                _mapping(
                    record.get("controls"),
                    label="qualification replicate controls",
                )
                type_records = record.get("type_records")
                _require(
                    isinstance(type_records, list)
                    and len(type_records) == expected_whole_types,
                    "qualification replicate has the wrong whole-type count",
                )
                type_indices: set[int] = set()
                for type_record_number, type_record_value in enumerate(type_records):
                    type_record = _mapping(
                        type_record_value,
                        label=(
                            "qualification type record "
                            f"{line_number}:{type_record_number}"
                        ),
                    )
                    type_index = type_record.get("synthetic_type_index")
                    _require(
                        isinstance(type_index, int) and not isinstance(type_index, bool),
                        "qualification type record has an invalid type index",
                    )
                    type_indices.add(type_index)
                    deltas = type_record.get("deltas")
                    _require(
                        isinstance(deltas, list) and len(deltas) == 2,
                        "qualification type record must contain two directional deltas",
                    )
                    _mapping(
                        type_record.get("source_fit_metadata"),
                        label="qualification type record source fit metadata",
                    )
                _require(
                    type_indices == set(range(expected_whole_types)),
                    "qualification replicate type indices are incomplete or duplicated",
                )
                _require(
                    record.get("numerical_failures") == [],
                    "qualification replicate contains a numerical failure",
                )
                seen_records.add(key)
                records_by_cell[cell_id].append(record)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        if isinstance(error, ReadinessError):
            raise
        raise ReadinessError(f"cannot validate qualification records: {error}") from error
    _require(
        len(seen_records) == expected_record_count,
        f"qualification has {len(seen_records)} replicate records; expected {expected_record_count}",
    )

    reported_cells = {
        str(cell["cell_id"]): dict(cell)
        for cell in cells
        if isinstance(cell, Mapping) and "cell_id" in cell
    }
    recomputed_cells: dict[str, dict[str, Any]] = {}
    try:
        for cell in expected_cells:
            summary = _cell_summary(
                cell,
                records_by_cell[cell.cell_id],
                oracle_enclosure=oracle_by_cell[cell.cell_id],
                binomial_confidence_level=(
                    coverage_error_budget.binomial_readout_confidence_level
                ),
            )
            passed, reasons = _accept_cell(summary, spec["acceptance"])
            summary["passed"] = passed
            summary["failure_reasons"] = reasons
            recomputed_cells[cell.cell_id] = summary
    except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as error:
        raise ReadinessError(
            f"cannot recompute qualification operating characteristics: {error}"
        ) from error
    failed_recomputed = [
        cell_id
        for cell_id, summary in recomputed_cells.items()
        if summary.get("passed") is not True
    ]
    _require(
        not failed_recomputed,
        "recomputed qualification acceptance failed for cells: "
        + ", ".join(failed_recomputed[:3]),
    )
    summaries_match = all(
        reported_cells[cell_id] == summary
        for cell_id, summary in recomputed_cells.items()
    )
    _require(
        summaries_match,
        "qualification report cell summaries differ from recomputed replicate results",
    )

    try:
        with artifacts["result_table.csv"].open("r", encoding="utf-8", newline="") as handle:
            table_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ReadinessError(f"cannot read qualification result table: {error}") from error
    _require(
        {str(row.get("cell_id", "")) for row in table_rows} == expected_cell_ids
        and len(table_rows) == len(expected_cells),
        "qualification result table differs from the frozen cell inventory",
    )
    for row in table_rows:
        try:
            table_repetitions = int(row.get("replicates", ""))
        except (TypeError, ValueError) as error:
            raise ReadinessError(
                "qualification result table has an invalid replicate count"
            ) from error
        _require(
            table_repetitions == expected_repetitions,
            "qualification result table has the wrong replicate count",
        )
        _require(
            str(row.get("passed", "")).strip().lower() == "true",
            "qualification result table contains a failed cell",
        )
    _require(
        artifacts["result_table.md"].stat().st_size > 0,
        "qualification markdown result table is empty",
    )
    _require(
        report.get("real_connectivity_accessed") is False,
        "qualification accessed real connectivity",
    )
    _require(
        report.get("real_connectivity_access_authorized") == "development_only",
        "qualification authorization differs from the frozen development-only contract",
    )
    return {
        "tier": "qualification",
        "scientific_cell_count": len(cells),
        "replicates_per_cell": expected_repetitions,
        "replicate_record_count": len(seen_records),
        "qualification_passed": True,
        "real_connectivity_accessed": False,
    }


def _configurations(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path, label="development configuration plan")
    if isinstance(payload, Mapping):
        for key in ("configurations", "successors", "trials"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    _require(isinstance(payload, list), f"configuration plan is not a list: {path}")
    return [_mapping(value, label="development configuration") for value in payload]


def verify_observed_provisional_chain(
    *,
    run_root: Path,
    observed_finalization_dir: Path,
    policy: EpisodePolicy,
    outputs_root: Path = OUTPUTS_ROOT,
    expected_contract_id: str = "ep12_outgoing_tum_post36_v3",
    **_: Any,
) -> dict[str, Any]:
    root = _inside(run_root, outputs_root, label="observed run root")
    finalization = _inside(
        observed_finalization_dir, outputs_root, label="observed finalization directory"
    )
    trial_ids: list[str] = []
    source_counts: dict[str, int] = {}
    for source, expected_count, relative_plan, relative_results in _CHAIN_SOURCES:
        configurations = _configurations(root / relative_plan)
        _require(
            len(configurations) == expected_count,
            f"{source} has {len(configurations)} configurations; expected {expected_count}",
        )
        for configuration in configurations:
            validate_development_configuration(policy, configuration)
            trial_id = str(configuration.get("trial_id", ""))
            _require(trial_id, f"{source} has a configuration without trial_id")
            report = _mapping(
                _read_json(
                    root / relative_results / trial_id / "trial_report.json",
                    label=f"development report {trial_id}",
                ),
                label=f"development report {trial_id}",
            )
            _require(
                report.get("status") == "completed",
                f"development trial is not complete: {trial_id}",
            )
            _require(
                report.get("scientific_contract_id") == expected_contract_id
                and report.get("score_target_protocol") == DECODER_PROTOCOL,
                f"development trial requires corrected canonical V3 replay: {trial_id}",
            )
            _require(
                report.get("data_role") == "development_only"
                and report.get("final_connectivity_accessed") is False,
                f"development/final isolation failed: {trial_id}",
            )
            aggregate = _mapping(report.get("aggregate"), label=f"aggregate {trial_id}")
            _require(
                int(aggregate.get("technical_failure_count", -1)) == 0,
                f"development trial has technical failures: {trial_id}",
            )
            trial_ids.append(trial_id)
        source_counts[source] = len(configurations)
    _require(len(trial_ids) == policy.minimum_valid_trials == 36, "prefix is not 36 trials")
    _require(len(set(trial_ids)) == len(trial_ids), "development trial IDs are not unique")

    archive = _mapping(
        _read_json(
            finalization / "observed_search_archive.json",
            label="observed finalization archive",
        ),
        label="observed finalization archive",
    )
    _require(
        int(archive.get("verified_valid_trial_count", -1)) == len(trial_ids),
        "observed finalization does not cover all 36 development trials",
    )
    _require(
        archive.get("final_connectivity_accessed") is False
        and archive.get("final_connectivity_access_authorized") is False,
        "observed finalization crossed the final-data boundary",
    )
    return {
        "independent_trial_count": len(trial_ids),
        "scientific_contract_id": expected_contract_id,
        "score_target_protocol": DECODER_PROTOCOL,
        "source_counts": source_counts,
        "whole_provider_type_assignment": True,
        "development_roles_only": True,
        "final_connectivity_accessed": False,
    }


def verify_falsifier_plan(
    *,
    plan_path: Path,
    policy: EpisodePolicy,
    policy_path: Path,
    outputs_root: Path = OUTPUTS_ROOT,
    **_: Any,
) -> dict[str, Any]:
    path = _inside(plan_path, outputs_root, label="falsifier plan")
    plan = _mapping(_read_json(path, label="falsifier plan"), label="falsifier plan")
    frozen_tail_plan.validate_plan(plan, policy_path=policy_path)
    slots = plan.get("all_slots")
    _require(isinstance(slots, list) and slots, "falsifier plan has no slots")
    required = set(policy.required_falsifiers)
    planned = {
        str(slot.get("falsifier_id"))
        for slot in slots
        if isinstance(slot, Mapping) and slot.get("falsifier_id")
    }
    planned.update(
        str(gate.get("falsifier_id"))
        for gate in plan.get("non_trial_required_gates", [])
        if isinstance(gate, Mapping) and gate.get("falsifier_id")
    )
    _require(required <= planned, "falsifier plan omits a required control")
    _require(
        all(
            isinstance(slot, Mapping)
            and slot.get("selector", {}).get("target_data_role") == "development_only"
            and slot.get("selector", {}).get("final_type_information_allowed") is False
            for slot in slots
        ),
        "falsifier plan does not preserve development/final isolation",
    )
    return {
        "planned_slot_count": len(slots),
        "required_falsifier_ids": sorted(required),
        "bounded_search": True,
        "plan_only_not_execution": True,
        "final_connectivity_accessed": False,
    }


def verify_development_budget_override(
    *, override_path: Path, policy: EpisodePolicy, outputs_root: Path = OUTPUTS_ROOT, **_: Any
) -> dict[str, Any]:
    path = _inside(override_path, outputs_root, label="development resource authorization")
    override = load_yaml_subset(path)
    _require(
        override.get("scope")
        == "minimum_full_development_search_including_observed_and_required_99_null_searches",
        "resource authorization has the wrong scientific scope",
    )
    _require(
        override.get("cpu_budget_stop_disabled_until_minimum_full_search_complete") is True,
        "minimum full development search is not resource-authorized",
    )
    _require(
        override.get("final_connectivity_access_authorized") is False,
        "resource authorization crossed the final-data boundary",
    )
    limits = _mapping(override.get("unchanged_limits"), label="unchanged limits")
    _require(
        int(limits.get("minimum_valid_trials", -1)) == policy.minimum_valid_trials
        and int(limits.get("maximum_valid_trials", -1)) == policy.maximum_valid_trials,
        "resource authorization changed the bounded-search trial limits",
    )
    return {
        "scope": override.get("scope"),
        "authorized_planning_core_hours": override.get(
            "user_rounded_planning_core_hours"
        ),
        "maximum_valid_trials": policy.maximum_valid_trials,
        "wall_clock_hour_ceiling": limits.get("wall_clock_hour_ceiling"),
        "final_connectivity_accessed": False,
    }


def evaluate_production_readiness(
    *,
    qualification_dir: Path,
    observed_run_root: Path,
    observed_finalization_dir: Path,
    falsifier_plan_path: Path,
    budget_override_path: Path = OUTPUTS_ROOT / "executor" / "DEVELOPMENT_BUDGET_OVERRIDE.yaml",
    policy_path: Path = EPISODE_ROOT / "SEARCH_POLICY.yaml",
    prefix_contract_path: Path = OUTPUTS_ROOT / "executor" / "SCIENTIFIC_CONTRACT_POST36_V3.yaml",
    active_contract_path: Path = OUTPUTS_ROOT / "executor" / "SCIENTIFIC_CONTRACT_POST36_V3.yaml",
    outputs_root: Path = OUTPUTS_ROOT,
    **_: Any,
) -> dict[str, Any]:
    """Evaluate only the facts needed to decide whether the tail can start."""

    policy = EpisodePolicy.load(policy_path)
    checks = [
        _check(
            "scientific_contract_roles",
            verify_contract_roles,
            prefix_contract_path=prefix_contract_path,
            active_contract_path=active_contract_path,
        ),
        _check(
            "full_synthetic_qualification",
            verify_full_synthetic_qualification,
            qualification_dir=qualification_dir,
            outputs_root=outputs_root,
        ),
        {
            "check_id": "canonical_pipeline_qualification",
            "passed": False,
            "reason": (
                "The affected pooling/canonical-decoder/CV/reference/null procedure "
                "has no completed operating-characteristic qualification. The fixed "
                "11-category generated qualifier and focused arithmetic tests do not "
                "cover that boundary. No corrected procedure may inherit its pass."
            ),
        },
        _check(
            "observed_36_trial_provisional_chain",
            verify_observed_provisional_chain,
            run_root=observed_run_root,
            observed_finalization_dir=observed_finalization_dir,
            policy=policy,
            outputs_root=outputs_root,
            expected_contract_id=load_yaml_subset(active_contract_path)["contract_id"],
        ),
        _check(
            "post36_falsifier_plan",
            verify_falsifier_plan,
            plan_path=falsifier_plan_path,
            policy=policy,
            policy_path=policy_path,
            outputs_root=outputs_root,
        ),
        _check(
            "development_resource_authorization",
            verify_development_budget_override,
            override_path=budget_override_path,
            policy=policy,
            outputs_root=outputs_root,
        ),
    ]
    failed = [str(check["check_id"]) for check in checks if not check["passed"]]
    tail_permitted = not failed
    unavailable_after_plan = [
        "post36_falsifier_trials_completed",
        "patience_and_stop_rule_realized",
        "one_complete_comparison_selected_and_locked",
    ]
    return {
        "status": "pass" if tail_permitted else "revise",
        "episode_id": "ep12",
        "policy_id": policy.policy_id,
        "gate_scope": "readiness_for_observed_post36_tail_only",
        "checks": checks,
        "failed_foundational_checks": failed,
        "production_readiness_passed": tail_permitted,
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
                + ["conditional_null_eligibility_decision"],
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


def write_readiness_artifacts(
    output_dir: Path, report: Mapping[str, Any], *, outputs_root: Path = OUTPUTS_ROOT
) -> None:
    """Publish one readiness decision with a single monotone revise-to-pass step."""

    directory = _inside(output_dir, outputs_root, label="readiness output directory")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "production_readiness.json"
    candidate = dict(report)
    candidate_state = _validate_readiness_publication(candidate)
    lease_path = directory / ".production_readiness.lock"
    with lease_path.open("a+b") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        try:
            if path.is_symlink() or path.exists():
                _require(
                    path.is_file() and not path.is_symlink(),
                    "existing readiness decision is not an ordinary file",
                )
                current = _mapping(
                    _read_json(path, label="existing readiness report"),
                    label="existing readiness report",
                )
                current_state = _validate_readiness_publication(current)
                if current == candidate:
                    return
                _require(
                    current_state["status"] == "revise"
                    and candidate_state["status"] == "pass",
                    "readiness decision may only advance from revise to pass",
                )
                _require(
                    current_state["identity"] == candidate_state["identity"],
                    "readiness revise-to-pass identity differs",
                )
                atomic_json(path, candidate)
                return
            atomic_json(path, candidate)
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


def _validate_readiness_publication(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the semantic shape used to authorize a monotone publication."""

    _require(
        report.get("episode_id") == "ep12", "readiness episode identity is invalid"
    )
    policy_id = report.get("policy_id")
    _require(
        isinstance(policy_id, str) and policy_id,
        "readiness policy identity is missing",
    )
    _require(
        report.get("gate_scope") == "readiness_for_observed_post36_tail_only",
        "readiness gate scope is invalid",
    )
    checks = report.get("checks")
    _require(
        isinstance(checks, list) and checks,
        "readiness checks must be a nonempty list",
    )
    check_ids: list[str] = []
    failed: list[str] = []
    for value in checks:
        check = _mapping(value, label="readiness check")
        check_id = check.get("check_id")
        _require(isinstance(check_id, str) and check_id, "readiness check ID is missing")
        check_ids.append(check_id)
        passed = check.get("passed")
        _require(
            isinstance(passed, bool),
            f"readiness check result is invalid: {check_id}",
        )
        if passed:
            _mapping(check.get("evidence"), label=f"readiness check evidence {check_id}")
        else:
            _require(
                isinstance(check.get("reason"), str) and bool(check.get("reason")),
                f"failed readiness check lacks a reason: {check_id}",
            )
            failed.append(check_id)
    _require(
        len(check_ids) == len(set(check_ids))
        and set(check_ids) == set(REQUIRED_READINESS_CHECK_IDS),
        "readiness check inventory differs from the required gate set",
    )
    reported_failed = report.get("failed_foundational_checks")
    _require(
        isinstance(reported_failed, list)
        and len(reported_failed) == len(set(map(str, reported_failed)))
        and set(map(str, reported_failed)) == set(failed),
        "readiness failed-check inventory is inconsistent",
    )
    passed = not failed
    _require(
        report.get("status") == ("pass" if passed else "revise")
        and report.get("production_readiness_passed") is passed,
        "readiness status is inconsistent with its checks",
    )
    guards = _mapping(report.get("claim_guards"), label="readiness claim guards")
    observed_guard = _mapping(
        guards.get("observed_post36_tail_launch"),
        label="observed-tail readiness guard",
    )
    missing = observed_guard.get("missing")
    _require(
        observed_guard.get("permitted") is passed
        and isinstance(missing, list)
        and set(map(str, missing)) == set(failed),
        "observed-tail readiness guard is inconsistent",
    )
    _require(
        report.get("observed_outcome_values_read") is False,
        "readiness publication decoded observed outcomes",
    )
    for key in (
        "procedure_locked",
        "full_search_null_completed",
        "final_connectivity_accessed",
        "final_connectivity_access_authorized",
    ):
        _require(
            report.get(key) is False,
            f"readiness publication unexpectedly sets {key}",
        )
    return {
        "status": "pass" if passed else "revise",
        "identity": {
            "episode_id": report["episode_id"],
            "policy_id": policy_id,
            "gate_scope": report["gate_scope"],
        },
    }


def claim_is_permitted(report: Mapping[str, Any], claim: str) -> bool:
    aliases = {
        "observed_tail": "observed_post36_tail_launch",
        "procedure_lock": "procedure_lock",
        "full_search_null": "full_search_null_launch",
        "final_connectivity": "final_connectivity_access",
    }
    guards = _mapping(report.get("claim_guards"), label="claim guards")
    guard = _mapping(guards.get(aliases[claim]), label=f"claim guard {claim}")
    return guard.get("permitted") is True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ep12-production-readiness")
    parser.add_argument("--qualification-dir", type=Path, required=True)
    parser.add_argument("--observed-run-root", type=Path, required=True)
    parser.add_argument("--observed-finalization-dir", type=Path, required=True)
    parser.add_argument("--falsifier-plan", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--prefix-contract", type=Path, required=True)
    parser.add_argument("--active-contract", type=Path, required=True)
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
        qualification_dir=arguments.qualification_dir,
        observed_run_root=arguments.observed_run_root,
        observed_finalization_dir=arguments.observed_finalization_dir,
        falsifier_plan_path=arguments.falsifier_plan,
        budget_override_path=arguments.budget_override,
        policy_path=arguments.policy,
        prefix_contract_path=arguments.prefix_contract,
        active_contract_path=arguments.active_contract,
    )
    write_readiness_artifacts(arguments.output_dir, report)
    permitted = claim_is_permitted(report, arguments.require_for)
    print(
        json.dumps(
            {
                "status": report["status"],
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
