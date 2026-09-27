"""Lean, outcome-blind readiness checks for the EP12 observed tail.

Readiness is a scientific checklist, not a provenance transaction. This
module checks the completed development prefix, synthetic qualification,
frozen falsifier plan, contract roles, and resource authorization without
creating hashes, manifests, receipts, schemas, or environment attestations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import observed_tail_plan as frozen_tail_plan

from .adaptive_search import validate_development_configuration
from .contract_roles import RANK_RULE, load_contract_provenance
from .policy import EpisodePolicy
from .runtime import EPISODE_ROOT, atomic_json
from .yaml_subset import load_yaml_subset


OUTPUTS_ROOT = EPISODE_ROOT / "outputs"
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
    *, qualification_dir: Path, outputs_root: Path = OUTPUTS_ROOT, **_: Any
) -> dict[str, Any]:
    directory = _inside(qualification_dir, outputs_root, label="qualification directory")
    report = _mapping(
        _read_json(directory / "qualification_report.json", label="qualification report"),
        label="qualification report",
    )
    cells = report.get("cells")
    _require(isinstance(cells, list) and cells, "qualification has no scientific cells")
    _require(
        report.get("qualification_passed") is True,
        "scientific qualification has not passed",
    )
    _require(
        all(isinstance(cell, Mapping) and cell.get("passed") is True for cell in cells),
        "one or more scientific qualification cells failed",
    )
    stability = report.get("integration_stability")
    if isinstance(stability, Mapping) and "passed" in stability:
        _require(stability.get("passed") is True, "integration stability failed")
    _require(
        report.get("real_connectivity_accessed") is False,
        "qualification accessed real connectivity",
    )
    _require(
        report.get("real_connectivity_access_authorized") is False,
        "qualification authorized real connectivity",
    )
    return {
        "tier": report.get("tier"),
        "scientific_cell_count": len(cells),
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
    prefix_contract_path: Path = OUTPUTS_ROOT / "executor" / "SCIENTIFIC_CONTRACT.yaml",
    active_contract_path: Path = OUTPUTS_ROOT / "executor" / "SCIENTIFIC_CONTRACT_POST36.yaml",
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
        _check(
            "observed_36_trial_provisional_chain",
            verify_observed_provisional_chain,
            run_root=observed_run_root,
            observed_finalization_dir=observed_finalization_dir,
            policy=policy,
            outputs_root=outputs_root,
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
    """Write the single readiness decision; retries must be identical."""

    directory = _inside(output_dir, outputs_root, label="readiness output directory")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "production_readiness.json"
    if path.exists():
        current = _mapping(
            _read_json(path, label="existing readiness report"),
            label="existing readiness report",
        )
        _require(current == dict(report), "existing readiness decision differs")
        return
    atomic_json(path, dict(report))


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
