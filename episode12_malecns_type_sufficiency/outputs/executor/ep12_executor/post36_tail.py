"""Bounded controller for the EP12 observed post-36 scientific tail.

The frozen tail plan is declarative.  This module supplies the missing
execution state machine while preserving three boundaries:

* only development-role snapshots may enter a trial;
* a slot advances its valid ordinal only after a complete T/U/M result and
  every slot-specific scientific result is durably recorded; and
* a declared control is never replaced by a merely similar transform.

The controller is event sourced through :class:`EventJournal`. A
proposal is committed before execution, a completed bundle can be recovered
after an interrupted acknowledgement, and promotion/control dispositions are
separate records.  Scientific outcome values are consumed mechanically by the
controller, but the command-line summary deliberately exposes only execution
state and gate counts.

The production capability registry binds every declared control to its exact
implementation and frozen branch contract.  A whole-plan capability audit
runs before the first real tail proposal, so an unavailable branch cannot turn
a partial run into apparent completion.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .adaptive_search import validate_development_configuration
from .contract_roles import (
    active_record_fields,
    load_contract_provenance,
    prefix_record_fields,
    require_record_fields,
    validate_contract_provenance,
)
from .adaptive_successors import (
    _ranking_key,
    _read_configuration_payload,
    _scientific_signature,
    _successors,
    _verify_trial,
)
from .journal import EventJournal
from .policy import EpisodePolicy, digest_object
from .production_readiness import claim_is_permitted
from .runtime import EPISODE_ROOT, SCRATCH_ROOT, atomic_json



# This is the authenticated Slurm start of the observed coverage search.  The
# deadline is exactly the already-frozen 120-hour overall wall-clock ceiling;
# neither timestamp depends on a connectivity outcome.
OBSERVED_SEARCH_STARTED_UTC = "2026-09-26T05:57:34Z"
OBSERVED_SEARCH_WALL_DEADLINE_UTC = "2026-10-01T05:57:34Z"
WALL_DEADLINE_STOP_REASON = (
    "overall_wall_deadline_cannot_accommodate_next_tail_slot"
)

OUTPUTS_ROOT = EPISODE_ROOT / "outputs"
DEFAULT_PLAN_PATH = OUTPUTS_ROOT / "executor" / "OBSERVED_POST36_TAIL_PLAN.json"
DEFAULT_RUN_ROOT = OUTPUTS_ROOT / "development_run002"
DEFAULT_READINESS_DIR = DEFAULT_RUN_ROOT / "production_readiness"
DEFAULT_TAIL_ROOT = DEFAULT_RUN_ROOT / "post36_tail"
DEFAULT_SNAPSHOT_DIR = SCRATCH_ROOT / "development_snapshot_v1"
DEFAULT_TAIL_SCRATCH = SCRATCH_ROOT / "development_run002" / "post36_tail"

_SNAPSHOT_FILES = (
    "development_outgoing_counts.npz",
    "development_focal_neurons.parquet",
    "development_body_statistics.parquet",
    "raw_partner_vocabulary.json",
)


class TailControllerError(RuntimeError):
    """The tail state, input, or requested transition is invalid."""


class TailCapabilityError(TailControllerError):
    """The frozen plan requests a control that is not honestly executable."""


class TailExecutionError(TailControllerError):
    """One slot attempt failed before a valid ordinal could be committed."""

    def __init__(self, message: str, *, evidence: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.evidence = dict(evidence or {})


class TailControllerBusy(TailControllerError):
    """A second process tried to drive the same tail concurrently."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    boundary = root.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise TailControllerError(f"{label} must stay under {boundary}: {resolved}")
    return resolved


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TailControllerError(f"missing {label}: {path}") from error
    except json.JSONDecodeError as error:
        raise TailControllerError(f"malformed {label}: {path}") from error


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TailControllerError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError as error:
        raise TailControllerError(f"missing hashed artifact: {path}") from error
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TailControllerError(message)


def _parse_utc(value: str, *, label: str) -> datetime:
    _require(isinstance(value, str) and value.endswith("Z"), f"{label} is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise TailControllerError(f"{label} is not a valid UTC timestamp") from error
    _require(parsed.tzinfo is not None, f"{label} lacks a UTC offset")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime, *, label: str) -> str:
    _require(value.tzinfo is not None, f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def observed_search_wall_budget(policy: EpisodePolicy) -> dict[str, Any]:
    """Bind the authenticated start to the policy's existing wall limits."""

    budgets = _mapping(policy.raw.get("budgets"), label="policy budgets")
    try:
        ceiling_hours = float(budgets["wall_clock_hour_ceiling"])
        per_trial_hours = float(policy.wall_hours_per_trial_maximum)
    except (KeyError, TypeError, ValueError) as error:
        raise TailControllerError("policy wall-clock budget is malformed") from error
    _require(
        math.isfinite(ceiling_hours) and ceiling_hours > 0.0,
        "policy wall-clock ceiling is invalid",
    )
    _require(
        math.isfinite(per_trial_hours) and per_trial_hours > 0.0,
        "policy per-trial wall limit is invalid",
    )
    started = _parse_utc(
        OBSERVED_SEARCH_STARTED_UTC, label="observed search start"
    )
    deadline = _parse_utc(
        OBSERVED_SEARCH_WALL_DEADLINE_UTC, label="observed search wall deadline"
    )
    _require(
        (deadline - started).total_seconds() == ceiling_hours * 3600.0,
        "authenticated observed-search deadline differs from the policy ceiling",
    )
    return {
        "anchor_kind": "authenticated_observed_coverage_slurm_start",
        "observed_search_started_utc": OBSERVED_SEARCH_STARTED_UTC,
        "overall_wall_deadline_utc": OBSERVED_SEARCH_WALL_DEADLINE_UTC,
        "wall_clock_hour_ceiling": ceiling_hours,
        "wall_hours_per_trial_maximum": per_trial_hours,
        "decision_scope": "outcome_blind_tail_slot_admission",
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }


def validate_observed_search_wall_budget(
    policy: EpisodePolicy, value: Mapping[str, Any]
) -> dict[str, Any]:
    expected = observed_search_wall_budget(policy)
    _require(dict(value) == expected, "controller wall budget differs from policy")
    return expected


def assess_observed_search_wall_budget(
    *,
    policy: EpisodePolicy,
    wall_budget: Mapping[str, Any],
    now_utc: datetime,
) -> dict[str, Any]:
    """Decide whether one maximum-duration tail slot still fits.

    The decision uses only authenticated timestamps and frozen policy limits.
    It neither reads nor summarizes any connectivity outcome.
    """

    budget = validate_observed_search_wall_budget(policy, wall_budget)
    _require(now_utc.tzinfo is not None, "wall-budget clock must be timezone-aware")
    now = now_utc.astimezone(timezone.utc)
    deadline = _parse_utc(
        str(budget["overall_wall_deadline_utc"]),
        label="observed search wall deadline",
    )
    maximum_duration = timedelta(
        hours=float(budget["wall_hours_per_trial_maximum"])
    )
    latest_start = deadline - maximum_duration
    projected_completion = now + maximum_duration
    admitted = projected_completion <= deadline
    return {
        "observed_search_started_utc": budget["observed_search_started_utc"],
        "overall_wall_deadline_utc": budget["overall_wall_deadline_utc"],
        "evaluated_at_utc": _format_utc(now, label="wall-budget evaluation time"),
        "next_slot_latest_start_utc": _format_utc(
            latest_start, label="next-slot latest start"
        ),
        "projected_next_slot_completion_utc": _format_utc(
            projected_completion, label="projected next-slot completion"
        ),
        "remaining_wall_seconds": (deadline - now).total_seconds(),
        "next_slot_maximum_duration_seconds": maximum_duration.total_seconds(),
        "can_start_next_tail_slot": admitted,
        "deadline_would_be_violated_by_next_maximum_duration_slot": not admitted,
        "decision_basis": (
            "evaluation_time_plus_policy_wall_hours_per_trial_maximum_lte_"
            "authenticated_overall_wall_deadline"
        ),
        "outcome_values_read_for_decision": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }


def _objective(record: Mapping[str, Any]) -> dict[str, Any]:
    value = _mapping(record.get("objective"), label="trial objective")
    numeric_names = (
        "left_to_right",
        "right_to_left",
        "bilateral",
        "bilateral_floor",
        "direction_disagreement",
    )
    normalized: dict[str, Any] = {}
    for name in numeric_names:
        try:
            number = float(value[name])
        except (KeyError, TypeError, ValueError) as error:
            raise TailControllerError(f"objective field is not numeric: {name}") from error
        _require(math.isfinite(number), f"objective field is nonfinite: {name}")
        normalized[name] = number
    try:
        normalized["types_both_above_margin"] = int(
            value["types_both_above_margin"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TailControllerError(
            "objective types_both_above_margin is not an integer"
        ) from error
    return normalized


def _primary_record(
    *,
    trial_id: str,
    configuration: Mapping[str, Any],
    objective: Mapping[str, Any],
    valid_type_count: int,
    unscorable_type_count: int,
    configuration_sha256: str | None = None,
    evidence: Mapping[str, Any] | None = None,
    contract_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(configuration)
    normalized = {
        "trial_id": str(trial_id),
        "configuration": config,
        "configuration_sha256": configuration_sha256 or digest_object(config),
        "objective": dict(objective),
        "valid_type_count": int(valid_type_count),
        "unscorable_type_count": int(unscorable_type_count),
        "evidence": dict(evidence or {}),
        **dict(contract_fields or {}),
    }
    _require(
        normalized["configuration_sha256"] == digest_object(config),
        f"configuration hash mismatch: {trial_id}",
    )
    normalized["objective"] = _objective(normalized)
    _require(normalized["valid_type_count"] > 0, f"trial has no valid types: {trial_id}")
    _require(
        normalized["unscorable_type_count"] >= 0,
        f"negative unscorable count: {trial_id}",
    )
    return normalized


def promotion_decision(
    *,
    incumbent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    promotion_eligible: bool,
) -> dict[str, Any]:
    """Apply the frozen lexicographic archive ordering.

    A control may be arbitrarily strong and still cannot replace the primary
    incumbent.  For an eligible candidate, the existing ranking key includes
    the frozen configuration-hash tie rule, so no human tie resolution enters.
    """

    incumbent_key = _ranking_key(dict(incumbent))
    candidate_key = _ranking_key(dict(candidate))
    improved = bool(promotion_eligible and candidate_key < incumbent_key)
    selected = candidate if improved else incumbent
    return {
        "promotion_eligible": bool(promotion_eligible),
        "improved": improved,
        "disposition": "incumbent" if improved else "retired",
        "incumbent_trial_id": str(selected["trial_id"]),
        "incumbent_configuration_sha256": str(selected["configuration_sha256"]),
        "ranking_rule": (
            "bilateral directional floor, bilateral mean, direction agreement, "
            "count above margin, then configuration SHA-256"
        ),
    }


def production_control_capabilities() -> dict[str, dict[str, Any]]:
    """Return the evidence-bearing registry for exact production controls."""

    transform_path = OUTPUTS_ROOT / "executor" / "ep12_executor" / "falsifier_transforms.py"
    fitter_path = OUTPUTS_ROOT / "executor" / "ep12_executor" / "development_trial.py"
    composite_evidence = [
        str(OUTPUTS_ROOT / "executor" / "ep12_executor" / "falsifier_controls.py"),
        str(OUTPUTS_ROOT / "executor" / "ep12_executor" / "control_trial.py"),
        str(
            OUTPUTS_ROOT
            / "executor"
            / "ep12_executor"
            / "production_falsifier_suite.py"
        ),
        str(OUTPUTS_ROOT / "executor" / "FALSIFIER_CONTROL_IMPLEMENTATION_SPEC.json"),
        str(OUTPUTS_ROOT / "executor" / "tests" / "test_falsifier_controls.py"),
        str(
            OUTPUTS_ROOT
            / "executor"
            / "tests"
            / "test_production_falsifier_suite.py"
        ),
    ]
    return {
        "strength_and_margin_preserving_graph_randomization": {
            "supported": True,
            "implementation": "margin_preserving_switch_randomization",
            "evidence": [str(transform_path), str(fitter_path)],
        },
        "shuffled_partner_and_valid_side_controls": {
            "supported": True,
            "implementation": (
                "shuffle_partner_identities_on_side plus "
                "shuffle_side_membership, both conservation checked"
            ),
            "evidence": [str(transform_path), str(fitter_path)],
        },
        "cross_side_component_alignment_permutation": {
            "supported": True,
            "implementation": "ProductionFalsifierSuite::cross_side_component_alignment_permutation",
            "evidence": composite_evidence,
        },
        "binary_weighted_vocabulary_and_rank_sensitivities": {
            "supported": True,
            "implementation": "ProductionFalsifierSuite::binary_weighted_vocabulary_and_rank_sensitivities",
            "evidence": composite_evidence,
        },
        "nuisance_and_partner_block_ablations": {
            "supported": True,
            "implementation": "ProductionFalsifierSuite::nuisance_and_partner_block_ablations",
            "evidence": composite_evidence,
        },
        "capacity_matched_noise_controls": {
            "supported": True,
            "implementation": "ProductionFalsifierSuite::capacity_matched_noise_controls",
            "evidence": composite_evidence,
        },
        "leave_one_development_type_and_partner_family_influence": {
            "supported": True,
            "implementation": "ProductionFalsifierSuite::leave_one_development_type_and_partner_family_influence",
            "evidence": composite_evidence,
        },
        "few_type_and_high_strength_neuron_concentration_check": {
            "supported": True,
            "implementation": "ProductionFalsifierSuite::few_type_and_high_strength_neuron_concentration_check",
            "evidence": composite_evidence,
        },
        "status_endpoint_anatomy_and_missingness_checks": {
            "supported": True,
            "implementation": "ProductionFalsifierSuite::status_endpoint_anatomy_and_missingness_checks",
            "evidence": composite_evidence,
        },
    }


def capability_report(
    plan: Mapping[str, Any],
    capabilities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    classification = _mapping(
        plan.get("required_falsifier_classification"), label="falsifier classification"
    )
    required = [str(value) for value in classification.get("countable_observed_trial_ids", [])]
    missing: list[dict[str, Any]] = []
    supported: list[str] = []
    for falsifier_id in required:
        record = capabilities.get(falsifier_id)
        if isinstance(record, Mapping) and record.get("supported") is True:
            supported.append(falsifier_id)
        else:
            missing.append(
                {
                    "falsifier_id": falsifier_id,
                    "reason": (
                        str(record.get("reason"))
                        if isinstance(record, Mapping) and record.get("reason")
                        else "no registered production capability"
                    ),
                    "missing_evidence": (
                        list(record.get("missing_evidence", []))
                        if isinstance(record, Mapping)
                        else ["registered implementation and passed control result"]
                    ),
                }
            )
    return {
        "plan_sha256": plan.get("plan_sha256"),
        "whole_plan_executable": not missing,
        "required_countable_falsifier_ids": required,
        "supported_falsifier_ids": supported,
        "unsupported_falsifiers": missing,
        "unsupported_slot_ids": [
            str(slot["trial_id"])
            for slot in plan.get("all_slots", [])
            if slot.get("falsifier_id") in {item["falsifier_id"] for item in missing}
        ],
        "substitution_allowed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }


def load_frozen_plan(
    plan_path: Path,
    *,
    policy_path: Path = EPISODE_ROOT / "SEARCH_POLICY.yaml",
) -> dict[str, Any]:
    """Load the artifact only if it exactly matches the frozen generator."""

    import observed_tail_plan

    payload = _mapping(_read_json(plan_path, label="post-36 tail plan"), label="tail plan")
    observed_tail_plan.validate_plan(payload, policy_path=policy_path)
    return dict(payload)


def verify_readiness_directory(
    readiness_dir: Path,
    *,
    outputs_root: Path = OUTPUTS_ROOT,
) -> dict[str, Any]:
    """Read the single outcome-blind scientific readiness decision."""

    directory = _inside(readiness_dir, outputs_root, label="readiness directory")
    report_path = directory / "production_readiness.json"
    report = _mapping(_read_json(report_path, label="readiness report"), label="readiness report")
    _require(report.get("production_readiness_passed") is True, "readiness did not pass")
    _require(claim_is_permitted(report, "observed_tail"), "readiness does not permit the observed tail")
    _require(
        all(
            isinstance(check, Mapping) and check.get("passed") is True
            for check in report.get("checks", [])
        ),
        "one or more foundational readiness checks failed",
    )
    _require(
        report.get("observed_outcome_values_read") is False,
        "readiness handoff decoded observed outcomes",
    )
    for value, label in (
        (report.get("procedure_locked"), "procedure lock"),
        (report.get("full_search_null_completed"), "full-search null"),
        (report.get("final_connectivity_accessed"), "final connectivity access"),
        (report.get("final_connectivity_access_authorized"), "final connectivity authorization"),
    ):
        _require(value is False, f"readiness unexpectedly records {label}")
    return {
        "status": report.get("status", "pass"),
        "policy_id": report.get("policy_id"),
        "checks": report.get("checks", []),
        "observed_outcome_values_read": False,
        "final_connectivity_accessed": False,
    }


def load_production_prefix(
    *,
    run_root: Path,
    finalization_dir: Path,
    policy: EpisodePolicy,
    contract_provenance: Mapping[str, Any],
    outputs_root: Path = OUTPUTS_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify and decode the 36 development results inside the controller.

    This function is called only by the production compute job after the
    outcome-blind readiness handoff.  It never reads final-role connectivity.
    """

    root = _inside(run_root, outputs_root, label="observed run root")
    finalization = _inside(
        finalization_dir, outputs_root, label="observed finalization directory"
    )
    sources = (
        (
            root / "search_plan" / "coverage_configurations.json",
            root / "observed",
        ),
        (
            root / "adaptive" / "cycle1" / "plan" / "successor_configurations.json",
            root / "adaptive" / "cycle1" / "observed",
        ),
        (
            root / "adaptive" / "cycle2" / "plan" / "successor_configurations.json",
            root / "adaptive" / "cycle2" / "observed",
        ),
    )
    provenance = validate_contract_provenance(contract_provenance)
    prefix_fields = prefix_record_fields(provenance)
    records: list[dict[str, Any]] = []
    source_counts: list[dict[str, Any]] = []
    for configuration_path, result_root in sources:
        configurations, hashes = _read_configuration_payload(configuration_path)
        for configuration, configuration_hash in zip(configurations, hashes):
            validate_development_configuration(policy, configuration)
            verified = _verify_trial(
                result_root=result_root,
                configuration=configuration,
                configuration_hash=configuration_hash,
            )
            report_path = result_root / str(verified["trial_id"]) / "trial_report.json"
            report = _mapping(
                _read_json(report_path, label=f"prefix trial report {verified['trial_id']}"),
                label=f"prefix trial report {verified['trial_id']}",
            )
            _require(
                report.get("data_role") == "development_only"
                and report.get("final_connectivity_accessed") is False,
                f"prefix trial crossed the final-data boundary: {verified['trial_id']}",
            )
            records.append(
                _primary_record(
                    trial_id=str(verified["trial_id"]),
                    configuration=verified["configuration"],
                    configuration_sha256=str(verified["configuration_sha256"]),
                    objective=verified["objective"],
                    valid_type_count=int(verified["valid_type_count"]),
                    unscorable_type_count=int(verified["unscorable_type_count"]),
                    evidence={
                        "trial_report_sha256": verified["trial_report_sha256"],
                        "manifest_sha256": verified["manifest_sha256"],
                        "scientific_contract_id": "ep12_outgoing_tum_v1",
                    },
                    contract_fields=prefix_fields,
                )
            )
        source_counts.append(
            {
                "configuration_path": str(configuration_path),
                "trial_count": len(configurations),
            }
        )
    _require(
        len(records) == policy.minimum_valid_trials == 36,
        f"post-36 prefix requires 36 valid trials, found {len(records)}",
    )
    _require(
        len({_scientific_signature(record["configuration"]) for record in records})
        == len(records),
        "post-36 prefix repeats a scientific configuration",
    )

    archive_path = finalization / "observed_search_archive.json"
    archive = _mapping(
        _read_json(archive_path, label="observed search archive"),
        label="observed search archive",
    )
    _require(
        int(archive.get("verified_valid_trial_count", -1)) == 36,
        "observed search archive is not the full prefix",
    )
    archive_rows = archive.get("archive")
    _require(isinstance(archive_rows, list), "observed search archive rows are missing")
    by_id = {str(record["trial_id"]): record for record in records}
    _require(len(archive_rows) == len(by_id), "observed search archive length mismatch")
    for row in archive_rows:
        row = _mapping(row, label="observed archive row")
        trial_id = str(row.get("trial_id", ""))
        _require(trial_id in by_id, f"unknown trial in observed archive: {trial_id}")
        verified = by_id[trial_id]
        _require(
            row.get("configuration_sha256") == verified["configuration_sha256"],
            f"archive configuration hash mismatch: {trial_id}",
        )
        _require(
            dict(_mapping(row.get("objective"), label=f"archive objective {trial_id}"))
            == verified["objective"],
            f"archive objective mismatch: {trial_id}",
        )
    ranked = sorted(records, key=_ranking_key)
    _require(
        archive.get("provisional_incumbent_trial_id") == ranked[0]["trial_id"]
        and archive.get("provisional_incumbent_configuration_sha256")
        == ranked[0]["configuration_sha256"],
        "observed archive provisional incumbent is inconsistent",
    )
    _require(
        archive.get("final_connectivity_accessed") is False
        and archive.get("final_connectivity_access_authorized") is False,
        "observed archive crosses the final-connectivity boundary",
    )
    return records, {
        "source_configuration_counts": source_counts,
        "observed_finalization_path": str(archive_path),
        "contract_provenance": provenance,
        "final_connectivity_accessed": False,
    }


def snapshot_identity(snapshot_dir: Path, *, scratch_root: Path = SCRATCH_ROOT) -> dict[str, Any]:
    directory = _inside(snapshot_dir, scratch_root, label="development snapshot")
    files: dict[str, str] = {}
    for name in _SNAPSHOT_FILES:
        files[name] = _sha256_file(directory / name)
    hierarchy = directory / "partner_type_hierarchy.json"
    if hierarchy.is_file():
        files[hierarchy.name] = _sha256_file(hierarchy)
    return {
        "snapshot_dir": str(directory),
        "files": files,
        "development_roles_only": True,
        "final_connectivity_accessed": False,
    }


class SlotExecutor(Protocol):
    def capabilities(self) -> Mapping[str, Mapping[str, Any]]:
        ...

    def recover(self, proposal: Mapping[str, Any]) -> Mapping[str, Any] | None:
        ...

    def execute(self, proposal: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class TailReplay:
    initialization: Mapping[str, Any]
    proposals: Mapping[str, Mapping[str, Any]]
    scored: Mapping[str, Mapping[str, Any]]
    dispositions: tuple[Mapping[str, Any], ...]
    failures: tuple[Mapping[str, Any], ...]
    stop_record: Mapping[str, Any] | None
    primary_archive: tuple[Mapping[str, Any], ...]
    incumbent: Mapping[str, Any]
    current_valid_ordinal: int
    counted_falsifiers: int
    completed_falsifier_ids: frozenset[str]
    last_promotion_improvement_ordinal: int
    incumbent_decision_count: int


class Post36TailController:
    """Replayable mechanical controller over a validated frozen plan."""

    def __init__(
        self,
        *,
        policy: EpisodePolicy,
        plan: Mapping[str, Any],
        journal_path: Path,
        status_path: Path,
        executor: SlotExecutor,
        identity: Mapping[str, Any],
        now_utc: Callable[[], datetime] | None = None,
    ):
        self.policy = policy
        self.plan = dict(plan)
        self.journal = EventJournal(journal_path)
        self.status_path = status_path.resolve()
        self.executor = executor
        self.identity = dict(identity)
        self.wall_budget = validate_observed_search_wall_budget(
            self.policy,
            _mapping(
                self.identity.get("observed_search_wall_budget"),
                label="observed search wall budget",
            ),
        )
        self._now_utc = now_utc or (lambda: datetime.now(timezone.utc))
        self.contract_provenance = validate_contract_provenance(
            _mapping(self.identity.get("contract_provenance"), label="contract provenance")
        )
        self.active_contract_fields = active_record_fields(self.contract_provenance)
        self.slots = [dict(value) for value in self.plan.get("all_slots", [])]
        _require(len(self.slots) > 0, "tail plan has no slots")
        self.slot_by_id = {str(slot["trial_id"]): slot for slot in self.slots}
        _require(len(self.slot_by_id) == len(self.slots), "tail plan repeats a slot ID")

    def initialize(
        self,
        *,
        prefix_records: Sequence[Mapping[str, Any]],
        prefix_evidence: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        normalized: list[dict[str, Any]] = []
        for ordinal, raw in enumerate(prefix_records, start=1):
            record = _primary_record(
                trial_id=str(raw["trial_id"]),
                configuration=_mapping(raw["configuration"], label="prefix configuration"),
                configuration_sha256=str(raw["configuration_sha256"]),
                objective=_mapping(raw["objective"], label="prefix objective"),
                valid_type_count=int(raw["valid_type_count"]),
                unscorable_type_count=int(raw["unscorable_type_count"]),
                evidence=_mapping(raw.get("evidence", {}), label="prefix evidence"),
                contract_fields=prefix_record_fields(self.contract_provenance),
            )
            validate_development_configuration(self.policy, record["configuration"])
            record["valid_trial_ordinal"] = ordinal
            record["promotion_eligible"] = True
            normalized.append(record)
        expected = int(self.plan["prefix_contract"]["pre_tail_valid_trial_count"])
        _require(len(normalized) == expected == 36, "tail prefix must contain exactly 36 trials")
        _require(
            len({_scientific_signature(record["configuration"]) for record in normalized})
            == len(normalized),
            "tail prefix repeats a scientific configuration",
        )
        cycle_ids = sorted(
            {
                str(record["trial_id"]).split("_")[1]
                for record in normalized
                if str(record["trial_id"]).startswith("adaptive_c")
            }
        )
        _require(cycle_ids == ["c1", "c2"], "tail prefix lacks both adaptive successor cycles")
        payload = {
            "record_kind": "controller_initialization",
            "identity": self.identity,
            "contract_provenance": self.contract_provenance,
            "prefix_records": normalized,
            "prefix_archive_sha256": digest_object(normalized),
            "prefix_evidence": dict(prefix_evidence),
            "prefix_valid_trial_count": len(normalized),
            "prefix_adaptive_successor_cycles": cycle_ids,
            "prefix_incumbent_decision_count": max(0, len(normalized) - 1),
            "patience_epoch_start_valid_ordinal": 36,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        result = self.journal.append(
            event_id="controller:initialize", timestamp_utc=_utc_now(), payload=payload
        )
        if not result.appended:
            existing = result.record
            _require(
                existing.get("identity") == self.identity,
                "controller identity changed on resume",
            )
            _require(
                existing.get("prefix_archive_sha256") == digest_object(normalized),
                "prefix archive changed on resume",
            )
        self.replay()
        return result.record

    def replay(self) -> TailReplay:
        records = self.journal.read()
        _require(records, "tail controller has not been initialized")
        initialization = records[0]
        _require(
            initialization.get("record_kind") == "controller_initialization",
            "first tail record is not controller initialization",
        )
        _require(initialization.get("identity") == self.identity, "tail identity mismatch")
        prefix = initialization.get("prefix_records")
        _require(isinstance(prefix, list) and len(prefix) == 36, "tail prefix is malformed")
        for record in prefix:
            require_record_fields(
                _mapping(record, label="prefix record"), self.contract_provenance,
                active=False, label="prefix record",
            )
        primary_archive: list[Mapping[str, Any]] = [dict(value) for value in prefix]
        incumbent: Mapping[str, Any] = sorted(primary_archive, key=_ranking_key)[0]
        proposals: dict[str, Mapping[str, Any]] = {}
        scored: dict[str, Mapping[str, Any]] = {}
        dispositions: list[Mapping[str, Any]] = []
        failures: list[Mapping[str, Any]] = []
        stop_record: Mapping[str, Any] | None = None
        expected_ordinal = 37
        counted_falsifiers = 0
        completed_falsifier_ids: set[str] = set()
        last_improvement = int(
            initialization.get("patience_epoch_start_valid_ordinal", 36)
        )
        decision_count = int(initialization.get("prefix_incumbent_decision_count", 0))
        event_position: dict[tuple[str, str], int] = {}
        for position, record in enumerate(records[1:], start=1):
            require_record_fields(
                record, self.contract_provenance, active=True,
                label=f"tail ledger record {position}",
            )
            kind = str(record.get("record_kind", ""))
            trial_id = str(record.get("trial_id", ""))
            if stop_record is not None:
                raise TailControllerError("tail journal contains events after its stop record")
            if kind == "slot_proposal":
                _require(trial_id not in proposals, f"duplicate proposal: {trial_id}")
                slot = self.slot_by_id.get(trial_id)
                _require(slot is not None, f"proposal names unknown plan slot: {trial_id}")
                _require(
                    int(slot["valid_trial_ordinal"]) == expected_ordinal,
                    f"proposal skips valid ordinal {expected_ordinal}: {trial_id}",
                )
                _require(
                    record.get("slot_sha256") == slot.get("slot_sha256"),
                    f"proposal slot hash mismatch: {trial_id}",
                )
                proposals[trial_id] = record
                event_position[(trial_id, "proposal")] = position
            elif kind == "slot_attempt_failure":
                _require(trial_id in proposals, f"failure lacks proposal: {trial_id}")
                _require(trial_id not in scored, f"failure follows scored result: {trial_id}")
                failures.append(record)
            elif kind == "slot_scored":
                _require(trial_id in proposals, f"score lacks proposal: {trial_id}")
                _require(trial_id not in scored, f"duplicate score: {trial_id}")
                _require(
                    int(record.get("valid_trial_ordinal", -1)) == expected_ordinal,
                    f"score has wrong valid ordinal: {trial_id}",
                )
                scored[trial_id] = record
                event_position[(trial_id, "scored")] = position
            elif kind == "slot_disposition":
                _require(trial_id in scored, f"disposition lacks score: {trial_id}")
                _require(
                    event_position[(trial_id, "proposal")]
                    < event_position[(trial_id, "scored")]
                    < position,
                    f"slot event ordering is invalid: {trial_id}",
                )
                _require(
                    int(record.get("valid_trial_ordinal", -1)) == expected_ordinal,
                    f"disposition has wrong valid ordinal: {trial_id}",
                )
                slot = self.slot_by_id[trial_id]
                candidate = _primary_record(
                    trial_id=trial_id,
                    configuration=_mapping(
                        proposals[trial_id]["configuration"], label="proposed configuration"
                    ),
                    configuration_sha256=str(
                        proposals[trial_id]["configuration_sha256"]
                    ),
                    objective=_mapping(scored[trial_id]["objective"], label="scored objective"),
                    valid_type_count=int(scored[trial_id]["valid_type_count"]),
                    unscorable_type_count=int(scored[trial_id]["unscorable_type_count"]),
                    evidence=_mapping(scored[trial_id]["evidence"], label="score evidence"),
                    contract_fields=self.active_contract_fields,
                )
                decision = promotion_decision(
                    incumbent=incumbent,
                    candidate=candidate,
                    promotion_eligible=slot.get("promotion_eligible") is True,
                )
                for name in (
                    "promotion_eligible",
                    "improved",
                    "disposition",
                    "incumbent_trial_id",
                    "incumbent_configuration_sha256",
                ):
                    _require(
                        record.get(name) == decision[name],
                        f"promotion disposition mismatch for {trial_id}: {name}",
                    )
                if decision["improved"]:
                    incumbent = candidate
                    last_improvement = expected_ordinal
                if slot.get("promotion_eligible") is True:
                    primary_archive.append(candidate)
                    decision_count += 1
                else:
                    _require(
                        record.get("registered_falsifier_id") == slot.get("falsifier_id"),
                        f"falsifier identity mismatch: {trial_id}",
                    )
                    _require(
                        record.get("control_passed") is True,
                        f"failed control was counted as valid: {trial_id}",
                    )
                    counted_falsifiers += 1
                    completed_falsifier_ids.add(str(slot["falsifier_id"]))
                dispositions.append(record)
                expected_ordinal += 1
            elif kind == "controller_stop":
                _require(stop_record is None, "tail journal repeats its stop record")
                stop_record = record
            else:
                raise TailControllerError(f"unknown tail journal record kind: {kind}")

        current_valid_ordinal = expected_ordinal - 1
        return TailReplay(
            initialization=initialization,
            proposals=proposals,
            scored=scored,
            dispositions=tuple(dispositions),
            failures=tuple(failures),
            stop_record=stop_record,
            primary_archive=tuple(primary_archive),
            incumbent=incumbent,
            current_valid_ordinal=current_valid_ordinal,
            counted_falsifiers=counted_falsifiers,
            completed_falsifier_ids=frozenset(completed_falsifier_ids),
            last_promotion_improvement_ordinal=last_improvement,
            incumbent_decision_count=decision_count,
        )

    def _patience_status(self, state: TailReplay) -> dict[str, Any]:
        import observed_tail_plan

        return observed_tail_plan.assess_patience(
            valid_trial_ordinal=state.current_valid_ordinal,
            counted_post_coverage_falsifiers=state.counted_falsifiers,
            completed_countable_falsifier_ids=state.completed_falsifier_ids,
            last_promotion_eligible_improvement_ordinal=(
                state.last_promotion_improvement_ordinal
            ),
        )

    def stop_status(self, state: TailReplay | None = None) -> dict[str, Any]:
        state = state or self.replay()
        patience = self._patience_status(state)
        maximum = int(self.plan["policy_arithmetic"]["maximum_valid_trials"])
        maximum_reached = state.current_valid_ordinal >= maximum
        fraction_gate = bool(patience["falsifier_fraction_gate"])
        ids_gate = bool(patience["all_countable_falsifier_ids_gate"])
        cycles = set(state.initialization.get("prefix_adaptive_successor_cycles", []))
        promotion_gates = {
            "branch_coverage_verified": True,
            "minimum_adaptive_successor_cycles": (
                len(cycles) >= self.policy.minimum_adaptive_cycles
            ),
            "minimum_incumbent_challenger_decisions": (
                state.incumbent_decision_count >= self.policy.minimum_incumbent_decisions
            ),
            "post_coverage_falsifier_fraction": fraction_gate,
            "all_countable_falsifier_ids": ids_gate,
            "valid_primary_incumbent": bool(state.incumbent),
        }
        gates_pass = all(promotion_gates.values())
        if patience["patience_actionable"]:
            action = "stop"
            reason = "patience_exhausted"
        elif maximum_reached:
            action = "stop"
            reason = "valid_trials_max_reached"
        else:
            action = "continue"
            reason = None
        return {
            "action": action,
            "reason": reason,
            "current_valid_ordinal": state.current_valid_ordinal,
            "patience": patience,
            "maximum_valid_trials_reached": maximum_reached,
            "promotion_gates": promotion_gates,
            "completed_promotable_search": bool(action == "stop" and gates_pass),
            "maximum_with_failed_promotion_gates_is_complete": False,
        }

    def _wall_budget_status(self) -> dict[str, Any]:
        return assess_observed_search_wall_budget(
            policy=self.policy,
            wall_budget=self.wall_budget,
            now_utc=self._now_utc(),
        )

    def _wall_deadline_stop_status(
        self,
        state: TailReplay,
        wall_budget: Mapping[str, Any],
    ) -> dict[str, Any]:
        _require(
            wall_budget.get("can_start_next_tail_slot") is False,
            "cannot record a wall-deadline stop while another slot fits",
        )
        status = self.stop_status(state)
        _require(
            status.get("action") == "continue",
            "scientific stop must take precedence over wall-deadline stop",
        )
        return {
            **status,
            "action": "stop",
            "reason": WALL_DEADLINE_STOP_REASON,
            "stop_class": "hard_overall_wall_clock_ceiling",
            "wall_budget": dict(wall_budget),
            "wall_deadline_pending": True,
            "completed_promotable_search": False,
            "next_tail_slot_submission_permitted": False,
            "procedure_lock_submission_permitted": False,
        }

    def _append_stop(self, state: TailReplay, status: Mapping[str, Any]) -> Mapping[str, Any]:
        _require(status.get("action") == "stop", "cannot record an inactive stop")
        payload = {
            "record_kind": "controller_stop",
            **self.active_contract_fields,
            "trial_id": "__post36_stop__",
            "valid_trial_ordinal": state.current_valid_ordinal,
            "reason": status.get("reason"),
            "stop_class": status.get("stop_class", "scientific_search_stop"),
            "patience": status.get("patience"),
            "promotion_gates": status.get("promotion_gates"),
            "completed_promotable_search": status.get("completed_promotable_search"),
            "wall_budget": dict(
                _mapping(status.get("wall_budget", {}), label="stop wall budget")
            ),
            "wall_deadline_pending": status.get("wall_deadline_pending") is True,
            "next_tail_slot_submission_permitted": status.get(
                "next_tail_slot_submission_permitted", False
            )
            is True,
            "procedure_lock_submission_permitted": status.get(
                "procedure_lock_submission_permitted", True
            )
            is True,
            "incumbent_trial_id": state.incumbent["trial_id"],
            "incumbent_configuration_sha256": state.incumbent[
                "configuration_sha256"
            ],
            "non_trial_required_gates_remain_required": True,
            "procedure_locked": False,
            "full_search_null_completed": False,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        return self.journal.append(
            event_id="controller:stop", timestamp_utc=_utc_now(), payload=payload
        ).record

    def _write_wall_deadline_decision(
        self, state: TailReplay
    ) -> Mapping[str, Any]:
        stop = _mapping(state.stop_record, label="wall-deadline stop record")
        _require(
            stop.get("reason") == WALL_DEADLINE_STOP_REASON
            and stop.get("wall_deadline_pending") is True,
            "wall-deadline decision requires the durable deadline stop",
        )
        wall_budget = _mapping(stop.get("wall_budget"), label="stop wall budget")
        _require(
            wall_budget.get("can_start_next_tail_slot") is False,
            "wall-deadline decision cannot permit a next tail slot",
        )
        next_slot = self._next_slot(state)
        journal_status = self.journal.status()
        decision = {
            **self.active_contract_fields,
            "action": "wall_deadline_pending",
            "reason": WALL_DEADLINE_STOP_REASON,
            "policy_id": self.identity["policy_id"],
            "plan_sha256": self.plan["plan_sha256"],
            "controller_stop_event_id": stop["event_id"],
            "ledger_record_count": journal_status["record_count"],
            "current_valid_ordinal": state.current_valid_ordinal,
            "next_tail_slot_id": next_slot["trial_id"],
            "next_tail_slot_valid_ordinal": int(next_slot["valid_trial_ordinal"]),
            "wall_budget": dict(wall_budget),
            "valid_ordinal_advanced_by_deadline_decision": False,
            "next_tail_slot_submission_permitted": False,
            "procedure_lock_submission_permitted": False,
            "completed_promotable_search": False,
            "outcome_values_read_for_deadline_decision": False,
            "development_roles_only": True,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        atomic_json(self.status_path.parent / "wall_deadline_stop.json", decision)
        return decision

    def _next_slot(self, state: TailReplay) -> dict[str, Any]:
        index = state.current_valid_ordinal - 36
        _require(0 <= index < len(self.slots), "tail plan is exhausted without a stop")
        slot = self.slots[index]
        _require(
            int(slot["valid_trial_ordinal"]) == state.current_valid_ordinal + 1,
            "tail slot ordinal is not contiguous",
        )
        return slot

    def _configuration_for_slot(
        self, state: TailReplay, slot: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        trial_id = str(slot["trial_id"])
        if slot.get("slot_kind") == "mandatory_falsifier":
            configuration = dict(state.incumbent["configuration"])
            configuration["trial_id"] = trial_id
            configuration["stage"] = "adaptive"
            validate_development_configuration(self.policy, configuration)
            lineage = {
                "parent_trial_id": state.incumbent["trial_id"],
                "parent_configuration_sha256": state.incumbent[
                    "configuration_sha256"
                ],
                "selection": "best promotion-eligible complete comparison",
                "registered_falsifier_id": slot.get("falsifier_id"),
            }
            return configuration, lineage
        _require(
            slot.get("slot_kind") == "adaptive_challenger",
            f"unknown slot kind: {slot.get('slot_kind')}",
        )
        ranked = sorted((dict(value) for value in state.primary_archive), key=_ranking_key)
        configurations, lineage_rows = _successors(
            policy=self.policy,
            cycle=int(slot["valid_trial_ordinal"]),
            ranked_parents=ranked,
            seen_configurations=[dict(value["configuration"]) for value in state.primary_archive],
            count=1,
        )
        configuration = dict(configurations[0])
        configuration["trial_id"] = trial_id
        configuration["stage"] = "adaptive"
        validate_development_configuration(self.policy, configuration)
        return configuration, dict(lineage_rows[0])

    def ensure_proposal(self, state: TailReplay | None = None) -> Mapping[str, Any]:
        state = state or self.replay()
        slot = self._next_slot(state)
        trial_id = str(slot["trial_id"])
        existing = state.proposals.get(trial_id)
        if existing is not None:
            return existing
        capability = None
        if slot.get("slot_kind") == "mandatory_falsifier":
            capability = self.executor.capabilities().get(str(slot.get("falsifier_id")))
            if not isinstance(capability, Mapping) or capability.get("supported") is not True:
                reason = (
                    capability.get("reason")
                    if isinstance(capability, Mapping)
                    else "no registered production capability"
                )
                raise TailCapabilityError(
                    f"unsupported falsifier {slot.get('falsifier_id')}: {reason}"
                )
        proposal_context = self.journal.position()
        configuration, lineage = self._configuration_for_slot(state, slot)
        payload = {
            "record_kind": "slot_proposal",
            **self.active_contract_fields,
            "trial_id": trial_id,
            "valid_trial_ordinal": int(slot["valid_trial_ordinal"]),
            "slot_kind": slot["slot_kind"],
            "slot_sha256": slot["slot_sha256"],
            "plan_sha256": self.plan["plan_sha256"],
            "proposal_context": proposal_context,
            "visible_ledger_prefix_record_count": len(self.journal.read()),
            "configuration": configuration,
            "configuration_sha256": digest_object(configuration),
            "scientific_signature": _scientific_signature(configuration),
            "lineage": lineage,
            "outcome_evidence_refs": [lineage["parent_trial_id"]],
            "registered_falsifier_id": slot.get("falsifier_id"),
            "promotion_eligible": slot.get("promotion_eligible") is True,
            "required_realization_evidence": list(
                slot.get("required_realization_evidence", [])
            ),
            "control_capability": dict(capability or {}),
            "transform_seed": int(slot["slot_sha256"][:16], 16) % (2**32),
            "development_roles_only": True,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        return self.journal.append(
            event_id=f"slot:{trial_id}:proposal",
            timestamp_utc=_utc_now(),
            payload=payload,
        ).record

    def _validate_realization(
        self,
        *,
        proposal: Mapping[str, Any],
        realization: Mapping[str, Any],
    ) -> dict[str, Any]:
        trial_id = str(proposal["trial_id"])
        slot = self.slot_by_id[trial_id]
        require_record_fields(
            realization, self.contract_provenance, active=True,
            label=f"realization {trial_id}",
        )
        for name in ("trial_id", "slot_sha256", "configuration_sha256"):
            _require(
                realization.get(name) == proposal.get(name),
                f"realization {name} mismatch: {trial_id}",
            )
        _require(
            realization.get("proposal_context") == proposal.get("proposal_context"),
            f"realization proposal context mismatch: {trial_id}",
        )
        _require(
            realization.get("status") == "completed"
            and realization.get("valid_complete_T_U_M_triplet") is True,
            f"realization is not a complete T/U/M triplet: {trial_id}",
        )
        _require(
            realization.get("data_role") == "development_only"
            and realization.get("final_connectivity_accessed") is False,
            f"realization crossed a data-role boundary: {trial_id}",
        )
        _require(
            int(realization.get("technical_failure_count", -1)) == 0,
            f"realization has technical failures: {trial_id}",
        )
        valid_type_count = int(realization.get("valid_type_count", 0))
        unscorable_type_count = int(realization.get("unscorable_type_count", -1))
        _require(valid_type_count > 0, f"realization has no valid types: {trial_id}")
        _require(unscorable_type_count >= 0, f"invalid unscorable count: {trial_id}")
        objective = _objective({"objective": realization.get("objective")})
        evidence = _mapping(realization.get("evidence"), label="realization evidence")
        _require(evidence, f"realization lacks scientific evidence: {trial_id}")
        control = _mapping(realization.get("control_result"), label="control result")
        require_record_fields(
            control, self.contract_provenance, active=True,
            label=f"control result {trial_id}",
        )
        if slot.get("slot_kind") == "mandatory_falsifier":
            _require(
                control.get("falsifier_id") == slot.get("falsifier_id"),
                f"control result identity mismatch: {trial_id}",
            )
            _require(
                control.get("control_passed") is True
                and control.get("applicability_checks_passed") is True
                and control.get("conservation_checks_passed") is True,
                f"control result did not pass: {trial_id}",
            )
            _require(
                isinstance(control.get("transforms"), list) and control["transforms"],
                f"control result has no transforms: {trial_id}",
            )
        else:
            _require(
                control.get("falsifier_id") is None
                and control.get("control_passed") is True,
                f"primary challenger has an invalid control result: {trial_id}",
            )
        return {
            "objective": objective,
            "valid_type_count": valid_type_count,
            "unscorable_type_count": unscorable_type_count,
            "evidence": dict(evidence),
            "control_result": dict(control),
            "resources": dict(
                _mapping(realization.get("resources", {}), label="realization resources")
            ),
        }

    def _append_attempt_failure(
        self,
        *,
        proposal: Mapping[str, Any],
        error: Exception,
    ) -> Mapping[str, Any]:
        state = self.replay()
        trial_id = str(proposal["trial_id"])
        attempt_index = (
            sum(str(value.get("trial_id")) == trial_id for value in state.failures) + 1
        )
        evidence = error.evidence if isinstance(error, TailExecutionError) else {}
        payload = {
            "record_kind": "slot_attempt_failure",
            **self.active_contract_fields,
            "trial_id": trial_id,
            "valid_trial_ordinal": int(proposal["valid_trial_ordinal"]),
            "attempt_index": attempt_index,
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "evidence": evidence,
            "valid_ordinal_advanced": False,
            "retry_requires_identical_proposal": True,
            "final_connectivity_accessed": False,
        }
        return self.journal.append(
            event_id=f"slot:{trial_id}:attempt:{attempt_index:03d}:failed",
            timestamp_utc=_utc_now(),
            payload=payload,
        ).record

    def _append_score_and_disposition(
        self,
        *,
        proposal: Mapping[str, Any],
        normalized: Mapping[str, Any],
    ) -> TailReplay:
        trial_id = str(proposal["trial_id"])
        slot = self.slot_by_id[trial_id]
        scored_payload = {
            "record_kind": "slot_scored",
            **self.active_contract_fields,
            "trial_id": trial_id,
            "valid_trial_ordinal": int(proposal["valid_trial_ordinal"]),
            "slot_sha256": proposal["slot_sha256"],
            "configuration_sha256": proposal["configuration_sha256"],
            "objective": normalized["objective"],
            "valid_type_count": normalized["valid_type_count"],
            "unscorable_type_count": normalized["unscorable_type_count"],
            "evidence": normalized["evidence"],
            "control_result": normalized["control_result"],
            "resources": normalized["resources"],
            "valid_complete_T_U_M_triplet": True,
            "technical_failure_count": 0,
            "development_roles_only": True,
            "final_connectivity_accessed": False,
        }
        self.journal.append(
            event_id=f"slot:{trial_id}:scored",
            timestamp_utc=_utc_now(),
            payload=scored_payload,
        )
        state = self.replay()
        candidate = _primary_record(
            trial_id=trial_id,
            configuration=_mapping(proposal["configuration"], label="proposed configuration"),
            configuration_sha256=str(proposal["configuration_sha256"]),
            objective=normalized["objective"],
            valid_type_count=int(normalized["valid_type_count"]),
            unscorable_type_count=int(normalized["unscorable_type_count"]),
            evidence=normalized["evidence"],
            contract_fields=self.active_contract_fields,
        )
        decision = promotion_decision(
            incumbent=state.incumbent,
            candidate=candidate,
            promotion_eligible=slot.get("promotion_eligible") is True,
        )
        disposition_payload = {
            "record_kind": "slot_disposition",
            **self.active_contract_fields,
            "trial_id": trial_id,
            "valid_trial_ordinal": int(proposal["valid_trial_ordinal"]),
            "slot_sha256": proposal["slot_sha256"],
            "configuration_sha256": proposal["configuration_sha256"],
            **decision,
            "registered_falsifier_id": slot.get("falsifier_id"),
            "control_passed": normalized["control_result"].get("control_passed")
            is True,
            "counts_toward_post_coverage_denominator": True,
            "counts_toward_post_coverage_falsifier_numerator": slot.get(
                "counts_toward_post_coverage_falsifier_numerator"
            )
            is True,
            "valid_ordinal_advanced": True,
            "final_connectivity_accessed": False,
        }
        self.journal.append(
            event_id=f"slot:{trial_id}:disposition",
            timestamp_utc=_utc_now(),
            payload=disposition_payload,
        )
        return self.replay()

    def _status_payload(
        self,
        *,
        state: TailReplay,
        action: str,
        reason: str | None,
        failure: Mapping[str, Any] | None = None,
        wall_budget: Mapping[str, Any] | None = None,
        wall_deadline_decision: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        patience = self._patience_status(state)
        next_slot = None
        maximum = int(self.plan["policy_arithmetic"]["maximum_valid_trials"])
        if action == "continue" and state.current_valid_ordinal < maximum:
            next_slot = self._next_slot(state)["trial_id"]
        return {
            **self.active_contract_fields,
            "action": action,
            "reason": reason,
            "current_valid_ordinal": state.current_valid_ordinal,
            "committed_post36_slot_count": len(state.dispositions),
            "counted_post_coverage_falsifiers": state.counted_falsifiers,
            "completed_countable_falsifier_ids": sorted(
                state.completed_falsifier_ids
            ),
            "falsifier_fraction_gate": patience["falsifier_fraction_gate"],
            "all_countable_falsifier_ids_gate": patience[
                "all_countable_falsifier_ids_gate"
            ],
            "trailing_valid_trials_without_eligible_improvement": patience[
                "trailing_valid_trials_without_eligible_improvement"
            ],
            "patience_actionable": patience["patience_actionable"],
            "incumbent_trial_id": state.incumbent["trial_id"],
            "incumbent_configuration_sha256": state.incumbent[
                "configuration_sha256"
            ],
            "next_slot_id": next_slot,
            "failure": dict(failure or {}),
            "wall_budget": dict(wall_budget or {}),
            "wall_deadline_pending": action == "wall_deadline_pending",
            "wall_deadline_decision": dict(wall_deadline_decision or {}),
            "next_tail_slot_submission_permitted": action == "continue",
            "ledger": self.journal.status(),
            "procedure_locked": False,
            "full_search_null_completed": False,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }

    def _write_status(self, payload: Mapping[str, Any]) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.status_path, dict(payload))

    def _write_realized_schedule(self, state: TailReplay) -> None:
        """Freeze the realized scientific schedule after a valid stop."""

        _require(state.stop_record is not None, "realized schedule requires a stop record")
        realized_slots: list[dict[str, Any]] = []
        for disposition in state.dispositions:
            trial_id = str(disposition["trial_id"])
            proposal = state.proposals[trial_id]
            score = state.scored[trial_id]
            realized_slots.append(
                {
                    **self.active_contract_fields,
                    "trial_id": trial_id,
                    "valid_trial_ordinal": int(disposition["valid_trial_ordinal"]),
                    "slot_sha256": proposal["slot_sha256"],
                    "slot_kind": proposal["slot_kind"],
                    "configuration": proposal["configuration"],
                    "configuration_sha256": proposal["configuration_sha256"],
                    "scientific_signature": proposal["scientific_signature"],
                    "proposal_context": proposal["proposal_context"],
                    "lineage": proposal["lineage"],
                    "registered_falsifier_id": proposal[
                        "registered_falsifier_id"
                    ],
                    "promotion_eligible": disposition["promotion_eligible"],
                    "improved": disposition["improved"],
                    "disposition": disposition["disposition"],
                    "control_result": score["control_result"],
                }
            )
        journal_status = self.journal.status()
        schedule = {
            **self.active_contract_fields,
            "contract_provenance": self.contract_provenance,
            "policy_id": self.identity["policy_id"],
            "plan_sha256": self.plan["plan_sha256"],
            "stop_reason": state.stop_record["reason"],
            "stop_class": state.stop_record.get("stop_class"),
            "wall_budget": state.stop_record.get("wall_budget", {}),
            "wall_deadline_pending": state.stop_record.get(
                "wall_deadline_pending"
            )
            is True,
            "next_tail_slot_submission_permitted": state.stop_record.get(
                "next_tail_slot_submission_permitted"
            )
            is True,
            "procedure_lock_submission_permitted": state.stop_record.get(
                "procedure_lock_submission_permitted"
            )
            is True,
            "stop_valid_trial_ordinal": state.current_valid_ordinal,
            "completed_promotable_search": state.stop_record[
                "completed_promotable_search"
            ],
            "promotion_gates": state.stop_record["promotion_gates"],
            "incumbent_trial_id": state.incumbent["trial_id"],
            "incumbent_configuration_sha256": state.incumbent[
                "configuration_sha256"
            ],
            "realized_slots": realized_slots,
            "realized_slot_count": len(realized_slots),
            "realized_controller_program_sha256": digest_object(realized_slots),
            "ledger_record_count": journal_status["record_count"],
            "non_trial_required_gates": self.plan["non_trial_required_gates"],
            "non_trial_required_gates_completed": False,
            "procedure_locked": False,
            "full_search_null_completed": False,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        root = self.status_path.parent
        schedule_path = root / "realized_schedule.json"
        atomic_json(schedule_path, schedule)

    def _wall_deadline_pending_payload(self, state: TailReplay) -> dict[str, Any]:
        _require(
            state.stop_record is not None
            and state.stop_record.get("reason") == WALL_DEADLINE_STOP_REASON,
            "wall-deadline pending status requires its durable stop record",
        )
        self._write_realized_schedule(state)
        decision = self._write_wall_deadline_decision(state)
        payload = self._status_payload(
            state=state,
            action="wall_deadline_pending",
            reason=WALL_DEADLINE_STOP_REASON,
            wall_budget=_mapping(
                state.stop_record.get("wall_budget"), label="stop wall budget"
            ),
            wall_deadline_decision={
                "path": str(self.status_path.parent / "wall_deadline_stop.json"),
                "controller_stop_event_id": decision["controller_stop_event_id"],
            },
        )
        self._write_status(payload)
        return payload

    def run_one(self) -> dict[str, Any]:
        """Run or recover at most one slot under an exclusive controller lease."""

        lease_path = self.journal.path.with_suffix(self.journal.path.suffix + ".controller")
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        with lease_path.open("a+b") as lease:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise TailControllerBusy("another post-36 controller holds the lease") from error
            try:
                state = self.replay()
                if state.stop_record is not None:
                    if state.stop_record.get("reason") == WALL_DEADLINE_STOP_REASON:
                        return self._wall_deadline_pending_payload(state)
                    self._write_realized_schedule(state)
                    payload = self._status_payload(
                        state=state,
                        action="stopped",
                        reason=str(state.stop_record.get("reason")),
                    )
                    self._write_status(payload)
                    return payload
                stop = self.stop_status(state)
                if stop["action"] == "stop":
                    self._append_stop(state, stop)
                    state = self.replay()
                    self._write_realized_schedule(state)
                    payload = self._status_payload(
                        state=state, action="stopped", reason=str(stop["reason"])
                    )
                    self._write_status(payload)
                    return payload
                wall_budget = self._wall_budget_status()
                if wall_budget["can_start_next_tail_slot"] is not True:
                    wall_stop = self._wall_deadline_stop_status(state, wall_budget)
                    self._append_stop(state, wall_stop)
                    return self._wall_deadline_pending_payload(self.replay())
                proposal = self.ensure_proposal(state)
                try:
                    realization = self.executor.recover(proposal)
                    if realization is None:
                        realization = self.executor.execute(proposal)
                    normalized = self._validate_realization(
                        proposal=proposal,
                        realization=_mapping(realization, label="slot realization"),
                    )
                except Exception as error:
                    failure = self._append_attempt_failure(proposal=proposal, error=error)
                    state = self.replay()
                    payload = self._status_payload(
                        state=state,
                        action="retry_required",
                        reason="slot_attempt_failed_without_advancing_valid_ordinal",
                        wall_budget=wall_budget,
                        failure={
                            "trial_id": failure["trial_id"],
                            "attempt_index": failure["attempt_index"],
                            "failure_type": failure["failure_type"],
                            "failure_message": failure["failure_message"],
                            "evidence": failure["evidence"],
                        },
                    )
                    self._write_status(payload)
                    return payload
                state = self._append_score_and_disposition(
                    proposal=proposal, normalized=normalized
                )
                stop = self.stop_status(state)
                if stop["action"] == "stop":
                    self._append_stop(state, stop)
                    state = self.replay()
                    self._write_realized_schedule(state)
                    action = "stopped"
                    reason = str(stop["reason"])
                else:
                    wall_budget = self._wall_budget_status()
                    if wall_budget["can_start_next_tail_slot"] is not True:
                        wall_stop = self._wall_deadline_stop_status(
                            state, wall_budget
                        )
                        self._append_stop(state, wall_stop)
                        return self._wall_deadline_pending_payload(self.replay())
                    action = "continue"
                    reason = None
                payload = self._status_payload(
                    state=state,
                    action=action,
                    reason=reason,
                    wall_budget=(wall_budget if action == "continue" else None),
                )
                self._write_status(payload)
                return payload
            finally:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


class ProductionSlotExecutor:
    """Wire supported development transforms into the existing T/U/M fitter."""

    def __init__(
        self,
        *,
        tail_root: Path,
        scratch_root: Path,
        snapshot_dir: Path,
        policy: EpisodePolicy,
        active_contract_path: Path,
        contract_provenance: Mapping[str, Any],
    ):
        self.tail_root = _inside(tail_root, OUTPUTS_ROOT, label="tail output root")
        self.scratch_root = _inside(
            scratch_root, SCRATCH_ROOT, label="tail scratch root"
        )
        self.snapshot_dir = _inside(
            snapshot_dir, SCRATCH_ROOT, label="development snapshot"
        )
        self.policy = policy
        self.contract_provenance = validate_contract_provenance(contract_provenance)
        self.active_contract_fields = active_record_fields(self.contract_provenance)
        self.active_contract_path = active_contract_path.resolve(strict=True)
        from .yaml_subset import load_yaml_subset

        contract = load_yaml_subset(self.active_contract_path)
        _require(
            contract.get("contract_id") == "ep12_outgoing_tum_post36_v2"
            and contract.get("real_connectivity_access_authorized")
            == "development_only"
            and contract.get("final_connectivity_access_authorized", False) is False,
            "active contract does not preserve the post-36 development-only role",
        )

    def capabilities(self) -> Mapping[str, Mapping[str, Any]]:
        return production_control_capabilities()

    def _slot_dir(self, proposal: Mapping[str, Any]) -> Path:
        return self.tail_root / "slots" / str(proposal["trial_id"])

    def _verify_bundle(
        self, directory: Path, proposal: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        realization = dict(
            _mapping(
                _read_json(directory / "realization.json", label="slot realization"),
                label="slot realization",
            )
        )
        realization["evidence"] = dict(
            _mapping(realization.get("evidence"), label="realization evidence")
        )
        _require(
            realization.get("trial_id") == proposal.get("trial_id")
            and realization.get("slot_sha256") == proposal.get("slot_sha256")
            and realization.get("configuration_sha256")
            == proposal.get("configuration_sha256"),
            "slot realization identity mismatch",
        )
        require_record_fields(
            realization, self.contract_provenance, active=True,
            label="slot realization",
        )
        return realization

    def recover(self, proposal: Mapping[str, Any]) -> Mapping[str, Any] | None:
        directory = self._slot_dir(proposal)
        if not directory.exists():
            return None
        if not directory.is_dir():
            raise TailExecutionError(
                "durable slot path is not a directory",
                evidence={"path": str(directory)},
            )
        return self._verify_bundle(directory, proposal)

    def _copy_snapshot(self, target: Path) -> None:
        target.mkdir(parents=True, exist_ok=False)
        for name in _SNAPSHOT_FILES:
            shutil.copy2(self.snapshot_dir / name, target / name)
        hierarchy = self.snapshot_dir / "partner_type_hierarchy.json"
        if hierarchy.is_file():
            shutil.copy2(hierarchy, target / hierarchy.name)

    def _transform_snapshot(
        self,
        *,
        proposal: Mapping[str, Any],
        target: Path,
    ) -> dict[str, Any]:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as parquet
        from scipy import sparse

        from .falsifier_transforms import (
            margin_preserving_switch_randomization,
            shuffle_partner_identities_on_side,
            shuffle_side_membership,
        )

        self._copy_snapshot(target)
        falsifier_id = str(proposal["registered_falsifier_id"])
        seed = int(proposal["transform_seed"])
        matrix_path = target / "development_outgoing_counts.npz"
        focal_path = target / "development_focal_neurons.parquet"
        matrix = sparse.load_npz(matrix_path).tocsr()
        focal = parquet.read_table(focal_path).to_pandas()
        counts = matrix.toarray().astype(np.int64, copy=False)
        transforms: list[dict[str, Any]] = []
        applicability = True
        conservation = True
        if falsifier_id == "strength_and_margin_preserving_graph_randomization":
            strata = (
                focal["type"].astype(str) + "::" + focal["somaSide"].astype(str)
            ).to_numpy(object)
            transformed, receipt = margin_preserving_switch_randomization(
                counts, strata, seed=seed
            )
            transforms.append(receipt.as_dict())
            applicability = (
                int(receipt.details.get("completed_switches", 0)) > 0
                and receipt.input_sha256 != receipt.output_sha256
            )
            conservation = bool(
                receipt.row_totals_preserved
                and receipt.column_totals_preserved
                and receipt.total_mass_preserved
            )
        elif falsifier_id == "shuffled_partner_and_valid_side_controls":
            vocabulary = json.loads(
                (target / "raw_partner_vocabulary.json").read_text(encoding="utf-8")
            )
            typed_columns = [
                index
                for index, value in enumerate(vocabulary)
                if str(value).startswith("typed::")
            ]
            partner_shuffled, partner_receipt = shuffle_partner_identities_on_side(
                counts,
                focal["somaSide"].astype(str).to_numpy(object),
                typed_columns,
                shuffled_side="R",
                seed=seed,
            )
            shuffled_sides, side_receipt = shuffle_side_membership(
                focal["somaSide"].astype(str).to_numpy(object),
                focal["type"].astype(str).to_numpy(object),
                seed=(seed + 1) % (2**32),
            )
            focal = focal.copy()
            focal["somaSide"] = shuffled_sides
            parquet.write_table(
                pa.Table.from_pandas(focal, preserve_index=False), focal_path
            )
            transformed = partner_shuffled
            transforms.extend([partner_receipt.as_dict(), side_receipt])
            applicability = bool(
                partner_receipt.input_sha256 != partner_receipt.output_sha256
                and int(side_receipt.get("changed_row_count", 0)) > 0
            )
            conservation = bool(
                partner_receipt.row_totals_preserved
                and partner_receipt.total_mass_preserved
                and side_receipt.get("stratum_side_counts_before")
                == side_receipt.get("stratum_side_counts_after")
            )
        else:
            capability = self.capabilities().get(falsifier_id, {})
            raise TailCapabilityError(
                f"unsupported falsifier {falsifier_id}: "
                f"{capability.get('reason', 'no registered implementation')}"
            )
        sparse.save_npz(matrix_path, sparse.csr_matrix(transformed), compressed=True)
        return {
            **self.active_contract_fields,
            "falsifier_id": falsifier_id,
            "implementation": self.capabilities()[falsifier_id]["implementation"],
            "transforms": transforms,
            "applicability_checks_passed": applicability,
            "conservation_checks_passed": conservation,
            "control_passed": bool(applicability and conservation),
            "development_roles_only": True,
            "final_connectivity_accessed": False,
        }

    def execute(self, proposal: Mapping[str, Any]) -> Mapping[str, Any]:
        from .control_trial import verify_development_trial_artifacts
        from .development_trial import run_development_trial
        from .falsifier_controls import COUNTABLE_COMPOSITE_CONTROLS
        from .production_falsifier_suite import ProductionFalsifierSuite

        trial_id = str(proposal["trial_id"])
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        restart_count = os.environ.get("SLURM_RESTART_COUNT", "0")
        if not restart_count.isdigit():
            raise TailExecutionError(
                "Slurm restart count is not a nonnegative integer",
                evidence={"slurm_restart_count": restart_count},
            )
        attempt_token = (
            f"{slurm_job_id}.restart{restart_count}"
            if slurm_job_id is not None
            else f"pid{os.getpid()}"
        )
        attempt_dir = self.scratch_root / "attempts" / f"{trial_id}.{attempt_token}"
        if attempt_dir.exists() and any(attempt_dir.iterdir()):
            raise TailExecutionError(
                "refusing to reuse a nonempty slot attempt directory",
                evidence={"attempt_dir": str(attempt_dir)},
            )
        attempt_dir.mkdir(parents=True, exist_ok=True)
        proposal_path = attempt_dir / "proposal.json"
        atomic_json(proposal_path, dict(proposal))
        falsifier_id = str(proposal.get("registered_falsifier_id") or "")
        composite = bool(
            proposal.get("slot_kind") == "mandatory_falsifier"
            and falsifier_id in COUNTABLE_COMPOSITE_CONTROLS
        )
        artifact_roots: list[Path] = []
        if composite:
            suite = ProductionFalsifierSuite(
                snapshot_dir=self.snapshot_dir,
                policy=self.policy,
                contract_path=self.active_contract_path,
                contract_provenance=self.contract_provenance,
            )
            suite_result = suite.execute(proposal=proposal, attempt_dir=attempt_dir)
            control_result = {
                **dict(suite_result["control_receipt"]),
                **self.active_contract_fields,
            }
            representative = suite_result["representative"]
            report = representative["report"]
            verified = representative["verified"]
            resource_totals = suite_result["resource_totals"]
            artifact_roots.append(Path(suite_result["branch_root"]))
        elif proposal.get("slot_kind") == "mandatory_falsifier":
            transformed_snapshot = attempt_dir / "transformed_snapshot"
            control_result = self._transform_snapshot(
                proposal=proposal, target=transformed_snapshot
            )
            snapshot_dir = transformed_snapshot
        else:
            control_result = {
                **self.active_contract_fields,
                "falsifier_id": None,
                "implementation": "untransformed primary development challenger",
                "transforms": [
                    {
                        "transform": "identity_primary_development_snapshot",
                        "snapshot_identity": snapshot_identity(self.snapshot_dir),
                    }
                ],
                "applicability_checks_passed": True,
                "conservation_checks_passed": True,
                "control_passed": True,
                "development_roles_only": True,
                "final_connectivity_accessed": False,
            }
            snapshot_dir = self.snapshot_dir
        if not composite:
            result_root = attempt_dir / "result"
            trial_dir = result_root / trial_id
            report = run_development_trial(
                output_dir=trial_dir,
                snapshot_dir=snapshot_dir,
                configuration=dict(proposal["configuration"]),
                contract_path=self.active_contract_path,
                contract_provenance=self.contract_provenance,
            )
            verified = verify_development_trial_artifacts(
                result_root=result_root,
                configuration=dict(proposal["configuration"]),
                configuration_hash=str(proposal["configuration_sha256"]),
            )
            resource_totals = {
                "wall_seconds": float(report.get("wall_seconds", 0.0)),
                "cpu_seconds": float(report.get("cpu_seconds", 0.0)),
            }
            artifact_roots.append(result_root)
        if control_result.get("control_passed") is not True:
            raise TailExecutionError(
                "falsifier transform failed applicability or conservation checks",
                evidence={"control_result": control_result},
            )
        require_record_fields(
            report, self.contract_provenance, active=True,
            label=f"post36 trial report {trial_id}",
        )
        require_record_fields(
            control_result, self.contract_provenance, active=True,
            label=f"post36 control result {trial_id}",
        )
        control_path = attempt_dir / "control_result.json"
        atomic_json(control_path, control_result)
        representative_trial_dir = (
            Path(representative["trial_dir"]) if composite else trial_dir
        )
        result_table_relative = (
            representative_trial_dir / "result_table.csv"
        ).relative_to(attempt_dir).as_posix()
        realization = {
            **self.active_contract_fields,
            "trial_id": trial_id,
            "slot_sha256": proposal["slot_sha256"],
            "configuration_sha256": proposal["configuration_sha256"],
            "proposal_context": proposal["proposal_context"],
            "status": "completed",
            "valid_complete_T_U_M_triplet": True,
            "data_role": "development_only",
            "final_connectivity_accessed": False,
            "technical_failure_count": int(
                report["aggregate"]["technical_failure_count"]
            ),
            "valid_type_count": int(verified["valid_type_count"]),
            "unscorable_type_count": int(verified["unscorable_type_count"]),
            "objective": verified["objective"],
            "control_result": control_result,
            "resources": {
                "wall_hours": float(resource_totals["wall_seconds"]) / 3600.0,
                "cpu_core_hours": float(resource_totals["cpu_seconds"]) / 3600.0,
                "gpu_hours": 0.0,
            },
            "evidence": {
                "result_table": result_table_relative,
                "control_result": "control_result.json",
                "proposal": "proposal.json",
            },
        }
        realization_path = attempt_dir / "realization.json"
        atomic_json(realization_path, realization)
        final_dir = self._slot_dir(proposal)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = final_dir.with_name(f".{final_dir.name}.staging-{attempt_token}")
        if final_dir.exists() or staging.exists():
            raise TailExecutionError(
                "refusing to overwrite an existing durable slot bundle",
                evidence={"final_dir": str(final_dir), "staging": str(staging)},
            )
        # The transformed development snapshot is transient job input.  Keep it
        # on scratch and copy only the compact authenticated result bundle to
        # durable episode outputs.
        shutil.copytree(
            attempt_dir,
            staging,
            ignore=shutil.ignore_patterns("transformed_snapshot"),
        )
        os.replace(staging, final_dir)
        return self._verify_bundle(final_dir, proposal)


def _write_capability_artifacts(
    *, tail_root: Path, report: Mapping[str, Any]
) -> None:
    root = _inside(tail_root, OUTPUTS_ROOT, label="tail output root")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "capability_report.json", dict(report))
    if report.get("whole_plan_executable") is not True:
        atomic_json(
            root / "controller_status.json",
            {
                "action": "blocked",
                "reason": "unsupported_frozen_falsifier_controls",
                "unsupported_falsifiers": report.get("unsupported_falsifiers"),
                "valid_ordinal_advanced": False,
                "procedure_locked": False,
                "full_search_null_completed": False,
                "final_connectivity_accessed": False,
                "final_connectivity_access_authorized": False,
            },
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ep12-post36-tail")
    subcommands = parser.add_subparsers(dest="command", required=True)
    capabilities = subcommands.add_parser(
        "capabilities", help="audit every frozen control without reading outcomes"
    )
    capabilities.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    capabilities.add_argument("--tail-root", type=Path, default=DEFAULT_TAIL_ROOT)
    capabilities.add_argument("--policy", type=Path, required=True)

    run = subcommands.add_parser(
        "run-one", help="run or recover one planned valid slot and checkpoint"
    )
    run.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    run.add_argument("--policy", type=Path, required=True)
    run.add_argument("--readiness-dir", type=Path, default=DEFAULT_READINESS_DIR)
    run.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    run.add_argument(
        "--observed-finalization-dir",
        type=Path,
        default=DEFAULT_RUN_ROOT / "observed_finalization",
    )
    run.add_argument("--tail-root", type=Path, default=DEFAULT_TAIL_ROOT)
    run.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    run.add_argument("--scratch-root", type=Path, default=DEFAULT_TAIL_SCRATCH)
    run.add_argument("--prefix-contract", type=Path, required=True)
    run.add_argument("--active-contract", type=Path, required=True)
    return parser


def _sanitized_summary(status: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action": status.get("action"),
        "reason": status.get("reason"),
        "current_valid_ordinal": status.get("current_valid_ordinal"),
        "committed_post36_slot_count": status.get("committed_post36_slot_count"),
        "counted_post_coverage_falsifiers": status.get(
            "counted_post_coverage_falsifiers"
        ),
        "falsifier_fraction_gate": status.get("falsifier_fraction_gate"),
        "all_countable_falsifier_ids_gate": status.get(
            "all_countable_falsifier_ids_gate"
        ),
        "patience_actionable": status.get("patience_actionable"),
        "next_slot_id": status.get("next_slot_id"),
        "wall_deadline_pending": status.get("wall_deadline_pending") is True,
        "next_tail_slot_submission_permitted": status.get(
            "next_tail_slot_submission_permitted"
        )
        is True,
        "procedure_locked": False,
        "full_search_null_completed": False,
        "final_connectivity_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    policy = EpisodePolicy.load(arguments.policy)
    plan = load_frozen_plan(arguments.plan, policy_path=arguments.policy)
    report = capability_report(plan, production_control_capabilities())
    _write_capability_artifacts(tail_root=arguments.tail_root, report=report)
    if arguments.command == "capabilities":
        print(
            json.dumps(
                {
                    "whole_plan_executable": report["whole_plan_executable"],
                    "supported_falsifier_ids": report["supported_falsifier_ids"],
                    "unsupported_falsifier_ids": [
                        value["falsifier_id"]
                        for value in report["unsupported_falsifiers"]
                    ],
                    "final_connectivity_accessed": False,
                },
                sort_keys=True,
            )
        )
        return 0 if report["whole_plan_executable"] else 4
    if report["whole_plan_executable"] is not True:
        print(
            json.dumps(
                {
                    "action": "blocked",
                    "reason": "unsupported_frozen_falsifier_controls",
                    "unsupported_falsifier_ids": [
                        value["falsifier_id"]
                        for value in report["unsupported_falsifiers"]
                    ],
                    "valid_ordinal_advanced": False,
                    "final_connectivity_accessed": False,
                },
                sort_keys=True,
            )
        )
        return 4

    contract_provenance = load_contract_provenance(
        prefix_contract_path=arguments.prefix_contract,
        active_contract_path=arguments.active_contract,
    )

    readiness = verify_readiness_directory(arguments.readiness_dir)
    _require(
        readiness.get("policy_id") in {None, policy.policy_id},
        "readiness policy differs from the active scientific policy",
    )
    prefix_records, prefix_evidence = load_production_prefix(
        run_root=arguments.run_root,
        finalization_dir=arguments.observed_finalization_dir,
        policy=policy,
        contract_provenance=contract_provenance,
    )
    snapshot = snapshot_identity(arguments.snapshot_dir)
    identity = {
        "policy_id": policy.policy_id,
        "plan_sha256": plan["plan_sha256"],
        "observed_search_wall_budget": observed_search_wall_budget(policy),
        "readiness": readiness,
        "snapshot": snapshot,
        "contract_provenance": contract_provenance,
        "final_connectivity_accessed": False,
    }
    tail_root = _inside(arguments.tail_root, OUTPUTS_ROOT, label="tail output root")
    tail_root.mkdir(parents=True, exist_ok=True)
    executor = ProductionSlotExecutor(
        tail_root=tail_root,
        scratch_root=arguments.scratch_root,
        snapshot_dir=arguments.snapshot_dir,
        policy=policy,
        active_contract_path=arguments.active_contract,
        contract_provenance=contract_provenance,
    )
    controller = Post36TailController(
        policy=policy,
        plan=plan,
        journal_path=tail_root / "ledger.jsonl",
        status_path=tail_root / "controller_status.json",
        executor=executor,
        identity=identity,
    )
    controller.initialize(
        prefix_records=prefix_records,
        prefix_evidence={**prefix_evidence, "readiness": readiness, "snapshot": snapshot},
    )
    status = controller.run_one()
    print(json.dumps(_sanitized_summary(status), sort_keys=True))
    if status["action"] == "continue":
        return 10
    if status["action"] == "stopped":
        return 0
    if status["action"] == "wall_deadline_pending":
        return 12
    if status["action"] == "retry_required":
        return 3
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
