"""Authenticate EP12's unique development comparison and pre-null decision.

The lock consumes only explicitly supplied development artifacts.  It performs
no discovery and never opens connectivity inputs.  It binds the complete
minimum comparison archive, every realized post-36 slot, every registered
control result, the frozen mode-support inference, and the selected fitted-null
generator plan.  The resulting digest is the handoff a full-search-null runtime
must bind before it may execute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .adaptive_successors import _ranking_key
from .contract_provenance import ACTIVE_EPOCH, PREFIX_EPOCH
from .mode_support_inference import (
    BOOTSTRAP_REPLICATES,
    ModeSupportInferenceError,
    verify_mode_support_result,
)
from .policy import EpisodePolicy, canonical_json, digest_object


SCHEMA_VERSION = "ep12.pre_null_procedure_lock.v4"
MANIFEST_SCHEMA_VERSION = "ep12.pre_null_procedure_lock_manifest.v4"
ARCHIVE_SCHEMA_VERSION = "ep12.complete_development_comparison_archive.v2"
CONTROL_INVENTORY_SCHEMA_VERSION = "ep12.required_control_result_inventory.v1"
PRE_NULL_MANIFEST_SCHEMA_VERSION = "ep12.pre_null_inputs_manifest.v3"
RUNTIME_SOURCE_BINDING_SCHEMA_VERSION = (
    "ep12.full_search_runtime_source_binding.v3"
)
FITTED_NULL_FALSIFIER_ID = "fitted_T_U_null_full_revised_search_rerun"
NOVELTY_AUDIT_FALSIFIER_ID = "forbidden_group_instance_post_result_novelty_audit"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProcedureLockError(ValueError):
    """An input cannot safely support a frozen pre-null decision."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProcedureLockError(message)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcedureLockError(f"{label} must be an object")
    return value


def _sequence(value: Any, *, label: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProcedureLockError(f"{label} must be an array")
    return list(value)


def _sha(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ProcedureLockError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProcedureLockError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ProcedureLockError(f"{name} must be finite")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ledger_object_sha256(path: Path) -> str:
    records: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                records.append(
                    _mapping(value, label=f"ledger record {line_number}")
                )
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcedureLockError(f"cannot authenticate ledger JSONL: {exc}") from exc
    _require(records, "realized ledger is empty")
    return digest_object(records)


def _scientific_signature(configuration: Mapping[str, Any]) -> str:
    return digest_object(
        {
            key: value
            for key, value in configuration.items()
            if key not in {"trial_id", "stage"}
        }
    )


def _validate_schedule(
    schedule: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    policy: EpisodePolicy,
    schedule_file_sha256: str,
    ledger_file_sha256: str,
    ledger_object_sha256: str,
    active_contract_fields: Mapping[str, str],
) -> list[Mapping[str, Any]]:
    _require(
        schedule.get("schema_version") == "ep12.post36_realized_schedule.v2",
        "realized schedule schema mismatch",
    )
    for key, value in active_contract_fields.items():
        _require(schedule.get(key) == value,
                 f"realized schedule contract provenance mismatch: {key}")
    _require(schedule.get("policy_sha256") == policy.policy_hash, "schedule policy mismatch")
    _sha("schedule plan_sha256", schedule.get("plan_sha256"))
    _sha("schedule controller_identity_sha256", schedule.get("controller_identity_sha256"))
    _require(
        schedule.get("completed_promotable_search") is True,
        "post-36 promotable search is incomplete",
    )
    promotion_gates = _mapping(schedule.get("promotion_gates"), label="promotion gates")
    _require(promotion_gates and all(value is True for value in promotion_gates.values()),
             "not every frozen promotion gate passed")
    _require(schedule.get("procedure_locked") is False, "schedule was already procedure-locked")
    _require(schedule.get("full_search_null_completed") is False,
             "schedule claims a prior full-search null")
    _require(schedule.get("final_connectivity_accessed") is False,
             "schedule records final-connectivity access")
    _require(schedule.get("final_connectivity_access_authorized") is False,
             "schedule authorizes final-connectivity access")
    _require(schedule.get("non_trial_required_gates_completed") is False,
             "schedule unexpectedly claims completion of non-trial gates")
    slots = [
        _mapping(item, label="realized schedule slot")
        for item in _sequence(schedule.get("realized_slots"), label="realized slots")
    ]
    _require(int(schedule.get("realized_slot_count", -1)) == len(slots),
             "realized slot count mismatch")
    _require(
        schedule.get("realized_controller_program_sha256") == digest_object(slots),
        "realized controller program digest mismatch",
    )
    trial_ids: set[str] = set()
    for slot in slots:
        trial_id = slot.get("trial_id")
        _require(isinstance(trial_id, str) and trial_id and trial_id not in trial_ids,
                 "realized schedule repeats or omits a trial ID")
        trial_ids.add(trial_id)
        for key, value in active_contract_fields.items():
            _require(slot.get(key) == value,
                     f"realized slot contract provenance mismatch: {trial_id}/{key}")
        configuration = _mapping(slot.get("configuration"), label=f"configuration {trial_id}")
        _require(configuration.get("trial_id") == trial_id,
                 f"slot configuration identity mismatch: {trial_id}")
        _require(digest_object(configuration) == slot.get("configuration_sha256"),
                 f"slot configuration digest mismatch: {trial_id}")
        _require(_scientific_signature(configuration) == slot.get("scientific_signature"),
                 f"slot scientific signature mismatch: {trial_id}")
        for name in (
            "slot_sha256",
            "configuration_sha256",
            "scientific_signature",
            "proposal_context_hash",
            "control_receipt_sha256",
            "durable_bundle_manifest_sha256",
        ):
            _sha(f"slot {trial_id} {name}", slot.get(name))

    _require(
        manifest.get("schema_version")
        == "ep12.post36_realized_schedule_manifest.v2",
        "realized schedule manifest schema mismatch",
    )
    files = _mapping(manifest.get("files"), label="realized schedule manifest files")
    _require(set(files) == {"realized_schedule.json", "ledger.jsonl"},
             "realized schedule manifest file set mismatch")
    _require(files.get("realized_schedule.json") == _sha(
        "schedule_file_sha256", schedule_file_sha256),
        "realized schedule file hash mismatch")
    _require(files.get("ledger.jsonl") == _sha("ledger_file_sha256", ledger_file_sha256),
             "realized ledger file hash mismatch")
    _require(
        schedule.get("ledger_hash")
        == _sha("ledger_object_sha256", ledger_object_sha256),
        "schedule/ledger semantic digest mismatch",
    )
    _require(manifest.get("ledger_head_hash") == schedule.get("ledger_head_hash"),
             "schedule/manifest ledger head mismatch")
    _require(manifest.get("procedure_locked") is False,
             "schedule manifest was already procedure-locked")
    _require(manifest.get("full_search_null_completed") is False,
             "schedule manifest claims a prior full-search null")
    _require(manifest.get("final_connectivity_accessed") is False,
             "schedule manifest records final-connectivity access")
    for key, value in active_contract_fields.items():
        _require(manifest.get(key) == value,
                 f"schedule manifest contract provenance mismatch: {key}")
    return slots


def _validate_comparison_record(record: Mapping[str, Any]) -> dict[str, Any]:
    trial_id = record.get("trial_id")
    _require(isinstance(trial_id, str) and trial_id, "comparison lacks trial_id")
    configuration = dict(
        _mapping(record.get("configuration"), label=f"comparison configuration {trial_id}")
    )
    _require(configuration.get("trial_id") == trial_id,
             f"comparison/configuration identity mismatch: {trial_id}")
    configuration_sha256 = _sha(
        f"comparison {trial_id} configuration_sha256",
        record.get("configuration_sha256"),
    )
    _require(digest_object(configuration) == configuration_sha256,
             f"comparison configuration digest mismatch: {trial_id}")
    signature = _sha(
        f"comparison {trial_id} scientific_signature",
        record.get("scientific_signature"),
    )
    _require(_scientific_signature(configuration) == signature,
             f"comparison scientific signature mismatch: {trial_id}")
    _require(record.get("status") == "completed", f"comparison is incomplete: {trial_id}")
    _require(record.get("valid_complete_T_U_M_triplet") is True,
             f"comparison lacks a complete T/U/M triplet: {trial_id}")
    _require(record.get("data_role") == "development_only",
             f"comparison is not development-only: {trial_id}")
    _require(record.get("final_connectivity_accessed") is False,
             f"comparison crossed the final-data boundary: {trial_id}")
    _require(int(record.get("technical_failure_count", -1)) == 0,
             f"comparison has technical failures: {trial_id}")
    _require(isinstance(record.get("selection_eligible"), bool),
             f"comparison selection eligibility is missing: {trial_id}")
    objective = dict(_mapping(record.get("objective"), label=f"objective {trial_id}"))
    for key in (
        "bilateral_floor",
        "bilateral",
        "direction_disagreement",
        "types_both_above_margin",
    ):
        _finite(f"objective {trial_id}.{key}", objective.get(key))
    result_table_sha256 = _sha(
        f"comparison {trial_id} result_table_sha256", record.get("result_table_sha256")
    )
    contract_epoch = record.get("contract_epoch")
    _require(contract_epoch in {PREFIX_EPOCH, ACTIVE_EPOCH},
             f"comparison contract epoch is invalid: {trial_id}")
    scientific_contract_sha256 = _sha(
        f"comparison {trial_id} scientific_contract_sha256",
        record.get("scientific_contract_sha256"),
    )
    transition_semantic = record.get("contract_transition_semantic_sha256")
    transition_file = record.get("contract_transition_file_sha256")
    if contract_epoch == ACTIVE_EPOCH:
        transition_semantic = _sha(
            f"comparison {trial_id} transition semantic sha256",
            transition_semantic,
        )
        transition_file = _sha(
            f"comparison {trial_id} transition file sha256", transition_file
        )
    else:
        _require(transition_semantic is None and transition_file is None,
                 f"legacy comparison carries ambiguous transition fields: {trial_id}")
    return {
        "trial_id": trial_id,
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "scientific_signature": signature,
        "source_stage": record.get("source_stage"),
        "selection_eligible": record["selection_eligible"],
        "status": "completed",
        "valid_complete_T_U_M_triplet": True,
        "data_role": "development_only",
        "final_connectivity_accessed": False,
        "technical_failure_count": 0,
        "objective": objective,
        "result_table_sha256": result_table_sha256,
        "contract_epoch": contract_epoch,
        "scientific_contract_sha256": scientific_contract_sha256,
        **(
            {
                "contract_transition_semantic_sha256": transition_semantic,
                "contract_transition_file_sha256": transition_file,
            }
            if contract_epoch == ACTIVE_EPOCH
            else {}
        ),
    }


def _select_comparison(
    archive: Mapping[str, Any],
    *,
    policy: EpisodePolicy,
    slots: Sequence[Mapping[str, Any]],
    observed_attestation_sha256: str,
    prefix_contract_sha256: str,
    active_contract_sha256: str,
    contract_transition_semantic_sha256: str,
    contract_transition_file_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    _require(archive.get("schema_version") == ARCHIVE_SCHEMA_VERSION,
             "comparison archive schema mismatch")
    _require(archive.get("policy_sha256") == policy.policy_hash,
             "comparison archive policy mismatch")
    _require(archive.get("observed_attestation_sha256") == observed_attestation_sha256,
             "comparison archive attestation binding mismatch")
    _require(archive.get("complete") is True, "comparison archive is not complete")
    _require(archive.get("final_connectivity_accessed") is False,
             "comparison archive records final-connectivity access")
    initial_ids = [str(value) for value in _sequence(
        archive.get("initial_trial_ids"), label="initial trial IDs"
    )]
    _require(len(initial_ids) == policy.minimum_valid_trials,
             "comparison archive does not contain the frozen 18+9+9 minimum")
    _require(len(set(initial_ids)) == len(initial_ids),
             "initial comparison archive repeats a trial ID")
    tail_ids = [str(slot["trial_id"]) for slot in slots]
    _require(set(initial_ids).isdisjoint(tail_ids),
             "initial and post-36 trial identities overlap")
    partition = _mapping(archive.get("contract_partition"), label="contract partition")
    prefix_partition = _mapping(
        partition.get("initial_prefix"), label="initial contract partition"
    )
    tail_partition = _mapping(
        partition.get("post36_tail"), label="tail contract partition"
    )
    _require(
        prefix_partition.get("contract_epoch") == PREFIX_EPOCH
        and prefix_partition.get("scientific_contract_sha256")
        == prefix_contract_sha256
        and list(prefix_partition.get("trial_ids", [])) == initial_ids,
        "initial-v1 contract partition is not exact",
    )
    _require(
        tail_partition.get("contract_epoch") == ACTIVE_EPOCH
        and tail_partition.get("scientific_contract_sha256")
        == active_contract_sha256
        and list(tail_partition.get("trial_ids", [])) == tail_ids,
        "tail-v2 contract partition is not exact",
    )
    _require(
        partition.get("contract_transition_semantic_sha256")
        == contract_transition_semantic_sha256
        and partition.get("contract_transition_file_sha256")
        == contract_transition_file_sha256,
        "archive contract-transition identity mismatch",
    )
    records = [
        _validate_comparison_record(_mapping(value, label="comparison record"))
        for value in _sequence(archive.get("records"), label="comparison records")
    ]
    by_id = {record["trial_id"]: record for record in records}
    _require(len(by_id) == len(records), "comparison archive repeats a trial ID")
    _require(set(by_id) == set(initial_ids) | set(tail_ids),
             "comparison archive does not exactly cover initial and realized-tail trials")
    _require(
        len({record["configuration_sha256"] for record in records}) == len(records),
        "comparison archive repeats a full configuration",
    )
    for trial_id in initial_ids:
        record = by_id[trial_id]
        _require(record["source_stage"] == "initial_minimum",
                 f"initial comparison has wrong source stage: {trial_id}")
        _require(record["selection_eligible"] is True,
                 f"initial comparison is incorrectly excluded from ranking: {trial_id}")
        _require(record["contract_epoch"] == PREFIX_EPOCH
                 and record["scientific_contract_sha256"] == prefix_contract_sha256,
                 f"initial comparison has wrong contract provenance: {trial_id}")
    for slot in slots:
        trial_id = str(slot["trial_id"])
        record = by_id[trial_id]
        _require(record["source_stage"] == "post36_tail",
                 f"tail comparison has wrong source stage: {trial_id}")
        for key in ("configuration_sha256", "scientific_signature"):
            _require(record[key] == slot[key],
                     f"tail archive/schedule {key} mismatch: {trial_id}")
        _require(record["selection_eligible"] is (slot.get("promotion_eligible") is True),
                 f"tail selection eligibility mismatch: {trial_id}")
        _require(record["contract_epoch"] == ACTIVE_EPOCH
                 and record["scientific_contract_sha256"] == active_contract_sha256
                 and record["contract_transition_semantic_sha256"]
                 == contract_transition_semantic_sha256
                 and record["contract_transition_file_sha256"]
                 == contract_transition_file_sha256,
                 f"tail comparison has wrong contract provenance: {trial_id}")
    candidates = [record for record in records if record["selection_eligible"]]
    _require(candidates, "comparison archive has no selection-eligible comparison")
    ranked = sorted(candidates, key=_ranking_key)
    selected = ranked[0]
    _require(selected.get("trial_id"), "selected comparison lacks identity")
    _require(
        archive.get("selected_trial_id") == selected["trial_id"]
        and archive.get("selected_configuration_sha256")
        == selected["configuration_sha256"],
        "comparison archive selected identity disagrees with frozen ranking",
    )
    _require(
        archive.get("selected_source_contract_epoch") == selected["contract_epoch"]
        and archive.get("selected_source_contract_sha256")
        == selected["scientific_contract_sha256"],
        "selected comparison source-contract provenance mismatch",
    )
    source_evidence = _mapping(
        archive.get("source_evidence"), label="comparison archive source evidence"
    )
    _require(source_evidence, "comparison archive source evidence is empty")
    archive_sha256 = digest_object(archive)
    return selected, records, archive_sha256


def _normalize_controls(
    inventory: Mapping[str, Any],
    *,
    policy: EpisodePolicy,
    slots: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], bool]:
    _require(inventory.get("schema_version") == CONTROL_INVENTORY_SCHEMA_VERSION,
             "control inventory schema mismatch")
    _require(inventory.get("policy_sha256") == policy.policy_hash,
             "control inventory policy mismatch")
    _require(inventory.get("final_connectivity_accessed") is False,
             "control inventory records final-connectivity access")
    supplied: dict[str, Mapping[str, Any]] = {}
    for value in _sequence(inventory.get("controls"), label="control results"):
        record = _mapping(value, label="control result")
        falsifier_id = record.get("falsifier_id")
        _require(isinstance(falsifier_id, str) and falsifier_id,
                 "control result lacks falsifier_id")
        _require(falsifier_id not in supplied, f"duplicate control result: {falsifier_id}")
        supplied[falsifier_id] = record
    required = list(policy.required_falsifiers)
    _require(FITTED_NULL_FALSIFIER_ID in required, "policy omits fitted-null falsifier")
    _require(NOVELTY_AUDIT_FALSIFIER_ID in required, "policy omits novelty audit")
    extra = sorted(set(supplied) - set(required))
    _require(not extra, f"control inventory contains unregistered controls: {extra}")
    observed_ids = set(required) - {
        FITTED_NULL_FALSIFIER_ID,
        NOVELTY_AUDIT_FALSIFIER_ID,
    }
    scheduled_hashes: dict[str, list[str]] = {name: [] for name in observed_ids}
    for slot in slots:
        falsifier_id = slot.get("registered_falsifier_id")
        if falsifier_id is None:
            continue
        _require(falsifier_id in observed_ids,
                 f"tail schedule contains unregistered observed control: {falsifier_id}")
        scheduled_hashes[str(falsifier_id)].append(str(slot["control_receipt_sha256"]))

    missing: list[str] = []
    normalized: list[dict[str, Any]] = []
    observed_passed = True
    for falsifier_id in required:
        if falsifier_id not in supplied:
            missing.append(falsifier_id)
            normalized.append(
                {
                    "falsifier_id": falsifier_id,
                    "status": "missing",
                    "applicable": None,
                    "control_passed": False,
                    "evidence_sha256s": [],
                    "final_connectivity_accessed": False,
                }
            )
            if falsifier_id in observed_ids:
                observed_passed = False
            continue
        record = supplied[falsifier_id]
        _require(record.get("final_connectivity_accessed") is False,
                 f"control records final-connectivity access: {falsifier_id}")
        hashes = [
            _sha(f"control {falsifier_id} evidence digest", value)
            for value in _sequence(
                record.get("evidence_sha256s"), label=f"control evidence {falsifier_id}"
            )
        ]
        _require(hashes and len(set(hashes)) == len(hashes),
                 f"control evidence is empty or duplicated: {falsifier_id}")
        status = record.get("status")
        applicable = record.get("applicable")
        passed = record.get("control_passed")
        if falsifier_id in observed_ids:
            expected = sorted(scheduled_hashes[falsifier_id])
            _require(expected, f"realized tail omits required control: {falsifier_id}")
            _require(sorted(hashes) == expected,
                     f"control inventory/schedule evidence mismatch: {falsifier_id}")
            this_passed = status == "passed" and applicable is True and passed is True
            observed_passed = observed_passed and this_passed
        elif falsifier_id == FITTED_NULL_FALSIFIER_ID:
            _require(status == "pending_full_search_null" and applicable is None
                     and passed is None,
                     "fitted-null control must be registered as pending")
        else:
            _require(status == "pending_post_result_novelty_audit" and applicable is None
                     and passed is None,
                     "novelty audit must be registered as pending post-result")
        normalized.append(
            {
                "falsifier_id": falsifier_id,
                "status": status,
                "applicable": applicable,
                "control_passed": passed,
                "evidence_sha256s": sorted(hashes),
                "final_connectivity_accessed": False,
            }
        )
    return normalized, missing, observed_passed


def _validate_generator_selection(selection: Mapping[str, Any]) -> str:
    _require(isinstance(selection.get("schema_version"), str),
             "null-generator selection lacks schema")
    _require(selection.get("complete") is True,
             "null-generator selection is incomplete")
    _require(selection.get("development_only") is True,
             "null-generator selection is not development-only")
    _require(selection.get("final_connectivity_accessed") is False,
             "null-generator selection records final-connectivity access")
    choices = _mapping(
        selection.get("selected_generator_by_type_side"),
        label="selected fitted-null generators",
    )
    _require(choices, "null-generator selection is empty")
    for key, value in choices.items():
        _require(isinstance(key, str) and key and value in {"T", "U"},
                 "null-generator selection must choose T or U for every type-side key")
    return digest_object(selection)


def build_procedure_lock(
    *,
    policy: EpisodePolicy,
    prefix_contract_sha256: str,
    active_contract_sha256: str,
    contract_transition_semantic_sha256: str,
    contract_transition_file_sha256: str,
    observed_attestation_sha256: str,
    development_role_manifest_sha256: str,
    comparison_archive: Mapping[str, Any],
    mode_support_result: Mapping[str, Any],
    realized_schedule: Mapping[str, Any],
    realized_schedule_manifest: Mapping[str, Any],
    realized_schedule_file_sha256: str,
    realized_ledger_file_sha256: str,
    realized_ledger_object_sha256: str,
    control_result_inventory: Mapping[str, Any],
    null_generator_selection: Mapping[str, Any],
    null_controller_program: Mapping[str, Any],
    null_controller_program_file_sha256: str,
    pre_null_manifest: Mapping[str, Any],
    pre_null_manifest_file_sha256: str,
    runtime_source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the one authenticated, development-only pre-null procedure lock."""

    prefix_contract_sha256 = _sha(
        "prefix_contract_sha256", prefix_contract_sha256
    )
    active_contract_sha256 = _sha(
        "active_contract_sha256", active_contract_sha256
    )
    contract_transition_semantic_sha256 = _sha(
        "contract_transition_semantic_sha256",
        contract_transition_semantic_sha256,
    )
    contract_transition_file_sha256 = _sha(
        "contract_transition_file_sha256", contract_transition_file_sha256
    )
    _require(prefix_contract_sha256 != active_contract_sha256,
             "prefix and active contracts must be distinct")
    active_fields = {
        "contract_epoch": ACTIVE_EPOCH,
        "scientific_contract_sha256": active_contract_sha256,
        "contract_transition_semantic_sha256": contract_transition_semantic_sha256,
        "contract_transition_file_sha256": contract_transition_file_sha256,
    }
    observed_attestation_sha256 = _sha(
        "observed_attestation_sha256", observed_attestation_sha256
    )
    development_role_manifest_sha256 = _sha(
        "development_role_manifest_sha256", development_role_manifest_sha256
    )
    schedule_file_sha256 = _sha(
        "realized_schedule_file_sha256", realized_schedule_file_sha256
    )
    ledger_file_sha256 = _sha(
        "realized_ledger_file_sha256", realized_ledger_file_sha256
    )
    slots = _validate_schedule(
        realized_schedule,
        realized_schedule_manifest,
        policy=policy,
        schedule_file_sha256=schedule_file_sha256,
        ledger_file_sha256=ledger_file_sha256,
        ledger_object_sha256=realized_ledger_object_sha256,
        active_contract_fields=active_fields,
    )
    selected, records, archive_sha256 = _select_comparison(
        comparison_archive,
        policy=policy,
        slots=slots,
        observed_attestation_sha256=observed_attestation_sha256,
        prefix_contract_sha256=prefix_contract_sha256,
        active_contract_sha256=active_contract_sha256,
        contract_transition_semantic_sha256=contract_transition_semantic_sha256,
        contract_transition_file_sha256=contract_transition_file_sha256,
    )
    _require(realized_schedule.get("incumbent_trial_id") == selected["trial_id"],
             "frozen ranking disagrees with realized-schedule incumbent")
    _require(realized_schedule.get("incumbent_configuration_sha256")
             == selected["configuration_sha256"],
             "selected configuration disagrees with realized-schedule incumbent")

    try:
        verify_mode_support_result(mode_support_result)
    except ModeSupportInferenceError as exc:
        raise ProcedureLockError(str(exc)) from exc
    _require(mode_support_result.get("comparison_trial_id") == selected["trial_id"],
             "mode-support result is for a different trial")
    _require(mode_support_result.get("configuration_sha256")
             == selected["configuration_sha256"],
             "mode-support result is for a different configuration")
    _require(mode_support_result.get("result_table_sha256")
             == selected["result_table_sha256"],
             "mode-support result is for a different result table")
    _require(mode_support_result.get("contract_sha256") == active_contract_sha256,
             "mode-support result is not an active-v2 refit")
    _require(
        mode_support_result.get("selected_source_contract_epoch")
        == selected["contract_epoch"]
        and mode_support_result.get("selected_source_contract_sha256")
        == selected["scientific_contract_sha256"]
        and mode_support_result.get("refit_contract_epoch") == ACTIVE_EPOCH
        and mode_support_result.get("contract_transition_semantic_sha256")
        == contract_transition_semantic_sha256
        and mode_support_result.get("contract_transition_file_sha256")
        == contract_transition_file_sha256,
        "mode-support result lacks selected-source/active-refit provenance",
    )
    _require(float(mode_support_result.get("meaningful_margin"))
             == policy.meaningful_margin,
             "mode-support result uses a different meaningful margin")
    if mode_support_result.get("complete_input"):
        _require(mode_support_result["bootstrap"]["replicates"] == BOOTSTRAP_REPLICATES,
                 "mode-support result uses an unfrozen bootstrap count")

    controls, missing_controls, observed_controls_passed = _normalize_controls(
        control_result_inventory,
        policy=policy,
        slots=slots,
    )
    generator_selection_sha256 = _validate_generator_selection(
        null_generator_selection
    )
    _require(
        null_generator_selection.get("selected_source_contract_epoch")
        == selected["contract_epoch"]
        and null_generator_selection.get("selected_source_contract_sha256")
        == selected["scientific_contract_sha256"]
        and null_generator_selection.get("refit_contract_epoch") == ACTIVE_EPOCH
        and null_generator_selection.get("refit_contract_sha256")
        == active_contract_sha256
        and null_generator_selection.get("contract_transition_semantic_sha256")
        == contract_transition_semantic_sha256
        and null_generator_selection.get("contract_transition_file_sha256")
        == contract_transition_file_sha256,
        "fitted-null selection lacks selected-source/active-refit provenance",
    )
    try:
        from .full_search_null import verify_null_controller_program

        verified_controller = verify_null_controller_program(
            null_controller_program, policy=policy
        )
    except Exception as exc:
        raise ProcedureLockError(
            "null controller program failed authentication"
        ) from exc
    controller_file_sha256 = _sha(
        "null_controller_program_file_sha256",
        null_controller_program_file_sha256,
    )
    pre_manifest = _mapping(pre_null_manifest, label="pre-null manifest")
    pre_core = {
        key: value for key, value in pre_manifest.items() if key != "bundle_sha256"
    }
    _require(
        pre_manifest.get("schema_version") == PRE_NULL_MANIFEST_SCHEMA_VERSION
        and pre_manifest.get("bundle_sha256") == digest_object(pre_core)
        and pre_manifest.get("policy_sha256") == policy.policy_hash
        and pre_manifest.get("development_only") is True
        and pre_manifest.get("final_connectivity_accessed") is False
        and pre_manifest.get("final_connectivity_access_authorized") is False,
        "pre-null manifest is unauthenticated or crosses the role firewall",
    )
    source_binding = _mapping(
        runtime_source_binding, label="runtime source binding"
    )
    source_core = {
        key: value
        for key, value in source_binding.items()
        if key != "source_binding_sha256"
    }
    _require(
        source_binding.get("schema_version")
        == RUNTIME_SOURCE_BINDING_SCHEMA_VERSION
        and source_binding.get("source_binding_sha256")
        == digest_object(source_core)
        and source_binding.get("policy_sha256") == policy.policy_hash
        and source_binding.get("development_only") is True
        and source_binding.get("final_connectivity_accessed") is False
        and source_binding.get("final_connectivity_access_authorized") is False,
        "runtime source binding is unauthenticated or unsafe",
    )
    _require(
        pre_manifest.get("selected_trial_id") == selected["trial_id"]
        == source_binding.get("selected_trial_id")
        and pre_manifest.get("selected_configuration_sha256")
        == selected["configuration_sha256"]
        == source_binding.get("selected_configuration_sha256")
        and pre_manifest.get("comparison_archive_sha256") == archive_sha256
        == source_binding.get("comparison_archive_sha256")
        and pre_manifest.get("generator_selection_sha256")
        == generator_selection_sha256
        == source_binding.get("null_generator_selection_sha256")
        and pre_manifest.get("null_controller_program_sha256")
        == verified_controller["program_sha256"]
        == source_binding.get("null_controller_program_sha256")
        and pre_manifest.get("null_controller_program_file_sha256")
        == controller_file_sha256
        and pre_manifest.get("runtime_source_binding_sha256")
        == source_binding.get("source_binding_sha256"),
        "pre-null bundle/source identities differ from locked scientific inputs",
    )
    _require(
        pre_manifest.get("prefix_contract_sha256") == prefix_contract_sha256
        == source_binding.get("prefix_contract_sha256")
        and pre_manifest.get("active_contract_sha256") == active_contract_sha256
        == source_binding.get("active_contract_sha256")
        and pre_manifest.get("contract_transition_semantic_sha256")
        == contract_transition_semantic_sha256
        == source_binding.get("contract_transition_semantic_sha256")
        and pre_manifest.get("contract_transition_file_sha256")
        == contract_transition_file_sha256
        == source_binding.get("contract_transition_file_sha256")
        and pre_manifest.get("selected_source_contract_epoch")
        == selected["contract_epoch"]
        == source_binding.get("selected_source_contract_epoch")
        and pre_manifest.get("selected_source_contract_sha256")
        == selected["scientific_contract_sha256"]
        == source_binding.get("selected_source_contract_sha256")
        and pre_manifest.get("refit_contract_epoch") == ACTIVE_EPOCH
        == source_binding.get("refit_contract_epoch")
        and pre_manifest.get("mode_support_inference_sha256")
        == mode_support_result.get("inference_sha256")
        and pre_manifest.get("control_inventory_sha256")
        == digest_object(control_result_inventory),
        "pre-null manifest differs from supplied contract, mode, or controls",
    )
    pre_null_binding = {
        "bundle_sha256": _sha(
            "pre-null bundle_sha256", pre_manifest.get("bundle_sha256")
        ),
        "manifest_file_sha256": _sha(
            "pre-null manifest file SHA-256", pre_null_manifest_file_sha256
        ),
        "runtime_source_binding_sha256": _sha(
            "runtime source-binding SHA-256",
            source_binding.get("source_binding_sha256"),
        ),
        "callback_data_sha256": _sha(
            "callback-data SHA-256", pre_manifest.get("callback_data_sha256")
        ),
        "model_state_sha256": _sha(
            "model-state SHA-256", pre_manifest.get("model_state_sha256")
        ),
        "snapshot_binding_sha256": _sha(
            "snapshot-binding SHA-256",
            pre_manifest.get("snapshot_binding_sha256"),
        ),
    }
    _require(
        source_binding.get("callback_data_sha256")
        == pre_null_binding["callback_data_sha256"],
        "runtime source binding differs from callback data",
    )
    registration = _mapping(
        control_result_inventory.get("pending_registrations"),
        label="pending control registrations",
    ).get(FITTED_NULL_FALSIFIER_ID)
    _require(
        isinstance(registration, Mapping)
        and registration.get("null_controller_program_sha256")
        == verified_controller["program_sha256"]
        and registration.get("observed_schedule_replayed") is False,
        "fitted-null registration does not bind the outcome-blind controller",
    )
    mode_complete = mode_support_result.get("complete_input") is True
    mode_supported = mode_support_result.get("scientific_mode_support") is True
    # The mode report is authenticated evidence for later interpretation, but
    # neither weak support nor an unresolved mode diagnostic invalidates the
    # complete T/U/M comparison or excuses the required fitted-null rerun.
    procedure_locked = not missing_controls
    # SEARCH_POLICY makes a complete, mechanically qualified T/U/M comparison
    # valid even when the observed structure is weak or continuous.  Mode
    # support is a later interpretation layer whose fitted-null probability is
    # itself still outstanding here; using its observed point/interval gates to
    # skip that null would make the required falsifier circular.
    scientific_support = procedure_locked and observed_controls_passed and mode_supported
    null_eligible = bool(procedure_locked and observed_controls_passed)
    unresolved_reasons: list[str] = []
    if missing_controls:
        unresolved_reasons.append("required_control_results_missing")
    if not mode_complete:
        unresolved_reasons.append("mode_support_inputs_incomplete")
    if not observed_controls_passed:
        unresolved_reasons.append("observed_control_support_not_established")
    if mode_complete and not mode_supported:
        unresolved_reasons.append("simultaneous_mode_support_not_established")

    core = {
        "schema_version": SCHEMA_VERSION,
        "policy_sha256": policy.policy_hash,
        "prefix_contract_sha256": prefix_contract_sha256,
        "active_contract_sha256": active_contract_sha256,
        "contract_transition_semantic_sha256": contract_transition_semantic_sha256,
        "contract_transition_file_sha256": contract_transition_file_sha256,
        "observed_attestation_sha256": observed_attestation_sha256,
        "development_role_manifest_sha256": development_role_manifest_sha256,
        "comparison_archive_sha256": archive_sha256,
        "comparison_archive_record_count": len(records),
        "ranking_rule": {
            "name": "frozen_development_only_bilateral_predictive_ranking",
            "keys": [
                "bilateral_floor_descending",
                "bilateral_descending",
                "direction_disagreement_ascending",
                "types_both_above_margin_descending",
                "configuration_sha256_ascending",
            ],
            "selection_eligible_only": True,
        },
        "selected_comparison": {
            "trial_id": selected["trial_id"],
            "configuration": selected["configuration"],
            "configuration_sha256": selected["configuration_sha256"],
            "scientific_signature": selected["scientific_signature"],
            "result_table_sha256": selected["result_table_sha256"],
            "source_contract_epoch": selected["contract_epoch"],
            "source_contract_sha256": selected["scientific_contract_sha256"],
            "deterministic_refit_contract_epoch": ACTIVE_EPOCH,
            "deterministic_refit_contract_sha256": active_contract_sha256,
        },
        "mode_support_inference_sha256": mode_support_result["inference_sha256"],
        "mode_support_state": mode_support_result["support_state"],
        "mode_support_gates": mode_support_result["gates"],
        "required_control_results": controls,
        "missing_required_control_results": missing_controls,
        "observed_controls_passed": observed_controls_passed,
        "realized_schedule": {
            "schedule_file_sha256": schedule_file_sha256,
            "schedule_object_sha256": digest_object(realized_schedule),
            "schedule_manifest_sha256": digest_object(realized_schedule_manifest),
            "ledger_file_sha256": ledger_file_sha256,
            "plan_sha256": realized_schedule["plan_sha256"],
            "controller_identity_sha256": realized_schedule[
                "controller_identity_sha256"
            ],
            "realized_controller_program_sha256": realized_schedule[
                "realized_controller_program_sha256"
            ],
            "stop_reason": realized_schedule["stop_reason"],
            "stop_valid_trial_ordinal": realized_schedule[
                "stop_valid_trial_ordinal"
            ],
            **active_fields,
        },
        "null_generator_selection_sha256": generator_selection_sha256,
        "null_controller_program_sha256": verified_controller[
            "program_sha256"
        ],
        "null_controller_program_file_sha256": controller_file_sha256,
        "null_controller_code_identity_sha256": verified_controller[
            "controller_code_identity_sha256"
        ],
        "pre_null_input_binding": pre_null_binding,
        "procedure_locked": procedure_locked,
        "scientific_support": scientific_support,
        "scientific_support_state": "supported" if scientific_support else "unresolved",
        "null_eligible": null_eligible,
        "null_eligibility_basis": (
            "at_least_one_valid_complete_comparison_and_all_mechanical_"
            "qualification_controls; observed_mode_support_is_interpretive_only"
        ),
        "null_eligibility_decision": (
            "eligible_run_frozen_full_search_null"
            if null_eligible
            else "not_eligible_mechanical_qualification_incomplete"
        ),
        "unresolved_reasons": unresolved_reasons,
        "pending_required_controls": [
            FITTED_NULL_FALSIFIER_ID,
            NOVELTY_AUDIT_FALSIFIER_ID,
        ],
        "full_search_null_contract": {
            "required_replicates": policy.null_replicates,
            "pass_threshold": policy.null_pass_threshold,
            "rerun_complete_frozen_controller": True,
            "runtime_must_bind_this_procedure_lock_sha256": True,
            "runtime_must_bind_selected_comparison": True,
            "runtime_must_bind_null_generator_selection_sha256": True,
            "runtime_must_bind_null_controller_program_sha256": True,
            "observed_realized_schedule_may_be_replayed": False,
        },
        "full_search_null_started": False,
        "full_search_null_completed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    result = dict(core)
    result["procedure_lock_sha256"] = digest_object(core)
    return result


def verify_procedure_lock(lock: Mapping[str, Any], *, policy: EpisodePolicy) -> None:
    """Verify the lock's self-authentication and frozen safety fields."""

    _require(lock.get("schema_version") == SCHEMA_VERSION,
             "procedure-lock schema mismatch")
    claimed = _sha("procedure_lock_sha256", lock.get("procedure_lock_sha256"))
    core = {key: value for key, value in lock.items() if key != "procedure_lock_sha256"}
    _require(digest_object(core) == claimed, "procedure-lock digest mismatch")
    _require(lock.get("policy_sha256") == policy.policy_hash,
             "procedure-lock policy mismatch")
    for name in (
        "prefix_contract_sha256",
        "active_contract_sha256",
        "contract_transition_semantic_sha256",
        "contract_transition_file_sha256",
    ):
        _sha(f"procedure lock {name}", lock.get(name))
    _require(
        lock.get("prefix_contract_sha256") != lock.get("active_contract_sha256"),
        "procedure lock collapses prefix and active contracts",
    )
    _require("contract_sha256" not in lock,
             "procedure lock retains an ambiguous single contract hash")
    _require(lock.get("final_connectivity_accessed") is False,
             "procedure lock records final-connectivity access")
    _require(lock.get("final_connectivity_access_authorized") is False,
             "procedure lock authorizes final-connectivity access")
    null_contract = _mapping(lock.get("full_search_null_contract"), label="null contract")
    _require(null_contract.get("required_replicates") == policy.null_replicates,
             "procedure lock changes the null replicate count")
    _require(float(null_contract.get("pass_threshold")) == policy.null_pass_threshold,
             "procedure lock changes the null pass threshold")
    _require(
        null_contract.get("runtime_must_bind_null_controller_program_sha256")
        is True
        and null_contract.get("observed_realized_schedule_may_be_replayed")
        is False,
        "procedure lock does not require outcome-blind adaptive null execution",
    )
    pre_null = _mapping(lock.get("pre_null_input_binding"), label="pre-null binding")
    _require(
        set(pre_null)
        == {
            "bundle_sha256", "manifest_file_sha256",
            "runtime_source_binding_sha256", "callback_data_sha256",
            "model_state_sha256", "snapshot_binding_sha256",
        }
        and all(_SHA256_RE.fullmatch(str(value)) for value in pre_null.values()),
        "procedure lock lacks complete pre-null input identities",
    )
    _require(lock.get("null_eligible") is True
             or lock.get("full_search_null_started") is False,
             "an ineligible lock claims the full-search null started")


def write_procedure_lock(output_dir: Path, lock: Mapping[str, Any]) -> Mapping[str, Any]:
    """Atomically create an immutable lock bundle, with idempotent recovery."""

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "procedure_lock.json"
    manifest_path = output_dir / "manifest.json"
    encoded = (canonical_json(lock) + "\n").encode("utf-8")
    lock_file_sha256 = hashlib.sha256(encoded).hexdigest()
    manifest_core = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "procedure_lock_sha256": lock["procedure_lock_sha256"],
        "files": {"procedure_lock.json": lock_file_sha256},
        "procedure_locked": lock["procedure_locked"],
        "null_eligible": lock["null_eligible"],
        "full_search_null_completed": False,
        "final_connectivity_accessed": False,
    }
    manifest = dict(manifest_core)
    manifest["manifest_sha256"] = digest_object(manifest_core)
    manifest_encoded = (canonical_json(manifest) + "\n").encode("utf-8")

    if lock_path.exists():
        _require(
            lock_path.is_file() and lock_path.read_bytes() == encoded,
            "refusing to overwrite a different procedure lock",
        )
    if manifest_path.exists():
        _require(
            manifest_path.is_file()
            and manifest_path.read_bytes() == manifest_encoded,
            "refusing to overwrite a different procedure-lock manifest",
        )
    if lock_path.exists() and manifest_path.exists():
        return manifest
    token = f"{os.getpid()}-{hashlib.sha256(encoded).hexdigest()[:12]}"
    lock_tmp = output_dir / f".procedure_lock.{token}.tmp"
    manifest_tmp = output_dir / f".manifest.{token}.tmp"
    try:
        pending = []
        if not lock_path.exists():
            pending.append((lock_tmp, lock_path, encoded))
        if not manifest_path.exists():
            pending.append((manifest_tmp, manifest_path, manifest_encoded))
        for path, _, payload in pending:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
            try:
                view = memoryview(payload)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise ProcedureLockError(f"short write while creating {path}")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for temporary, target, _ in pending:
            os.replace(temporary, target)
        directory_descriptor = os.open(output_dir, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        for path in (lock_tmp, manifest_tmp):
            if path.exists():
                path.unlink()
    return manifest


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcedureLockError(f"cannot read JSON artifact {path}: {exc}") from exc
    return _mapping(value, label=str(path))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--prefix-contract-sha256", required=True)
    parser.add_argument("--active-contract-sha256", required=True)
    parser.add_argument("--contract-transition-semantic-sha256", required=True)
    parser.add_argument("--contract-transition-file-sha256", required=True)
    parser.add_argument("--observed-attestation-sha256", required=True)
    parser.add_argument("--development-role-manifest-sha256", required=True)
    parser.add_argument("--comparison-archive", type=Path, required=True)
    parser.add_argument("--mode-support", type=Path, required=True)
    parser.add_argument("--realized-schedule", type=Path, required=True)
    parser.add_argument("--realized-schedule-manifest", type=Path, required=True)
    parser.add_argument("--realized-ledger", type=Path, required=True)
    parser.add_argument("--control-results", type=Path, required=True)
    parser.add_argument("--null-generator-selection", type=Path, required=True)
    parser.add_argument("--null-controller-program", type=Path, required=True)
    parser.add_argument("--pre-null-manifest", type=Path, required=True)
    parser.add_argument("--runtime-source-binding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    policy = EpisodePolicy.load(args.policy)
    schedule = _read_json(args.realized_schedule)
    lock = build_procedure_lock(
        policy=policy,
        prefix_contract_sha256=args.prefix_contract_sha256,
        active_contract_sha256=args.active_contract_sha256,
        contract_transition_semantic_sha256=(
            args.contract_transition_semantic_sha256
        ),
        contract_transition_file_sha256=args.contract_transition_file_sha256,
        observed_attestation_sha256=args.observed_attestation_sha256,
        development_role_manifest_sha256=args.development_role_manifest_sha256,
        comparison_archive=_read_json(args.comparison_archive),
        mode_support_result=_read_json(args.mode_support),
        realized_schedule=schedule,
        realized_schedule_manifest=_read_json(args.realized_schedule_manifest),
        realized_schedule_file_sha256=_file_sha256(args.realized_schedule),
        realized_ledger_file_sha256=_file_sha256(args.realized_ledger),
        realized_ledger_object_sha256=_ledger_object_sha256(
            args.realized_ledger
        ),
        control_result_inventory=_read_json(args.control_results),
        null_generator_selection=_read_json(args.null_generator_selection),
        null_controller_program=_read_json(args.null_controller_program),
        null_controller_program_file_sha256=_file_sha256(
            args.null_controller_program
        ),
        pre_null_manifest=_read_json(args.pre_null_manifest),
        pre_null_manifest_file_sha256=_file_sha256(args.pre_null_manifest),
        runtime_source_binding=_read_json(args.runtime_source_binding),
    )
    verify_procedure_lock(lock, policy=policy)
    manifest = write_procedure_lock(args.output_dir, lock)
    print(canonical_json({
        "procedure_lock_sha256": lock["procedure_lock_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "procedure_locked": lock["procedure_locked"],
        "null_eligible": lock["null_eligible"],
        "final_connectivity_accessed": False,
    }))
    return 0 if lock["procedure_locked"] else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_SCHEMA_VERSION",
    "CONTROL_INVENTORY_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "ProcedureLockError",
    "SCHEMA_VERSION",
    "build_procedure_lock",
    "verify_procedure_lock",
    "write_procedure_lock",
]
