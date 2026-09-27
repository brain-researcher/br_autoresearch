"""Build the one immutable scientific procedure lock for EP12.

The lock binds the selected development comparison, fitted-null generators,
controller program, and role firewall. It deliberately does not bind source
archives, transition receipts, environment inventories, or completion
manifests.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adaptive_successors import _ranking_key
from .contract_roles import ACTIVE_EPOCH, PREFIX_EPOCH, RANK_RULE
from .mode_support_inference import (
    BOOTSTRAP_REPLICATES,
    ModeSupportInferenceError,
    verify_mode_support_result,
)
from .policy import EpisodePolicy, canonical_json, digest_object
from .runtime import publish_once
from .yaml_subset import load_yaml_subset


FITTED_NULL_FALSIFIER_ID = "fitted_T_U_null_full_revised_search_rerun"
NOVELTY_AUDIT_FALSIFIER_ID = (
    "forbidden_group_instance_post_result_novelty_audit"
)
PREFIX_CONTRACT_ID = "ep12_outgoing_tum_v1"
ACTIVE_CONTRACT_ID = "ep12_outgoing_tum_post36_v2"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProcedureLockError(ValueError):
    """A scientific input cannot support the immutable procedure lock."""


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
    except (TypeError, ValueError) as error:
        raise ProcedureLockError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ProcedureLockError(f"{name} must be finite")
    return result


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
    *,
    policy: EpisodePolicy,
    active_contract_fields: Mapping[str, str] | None = None,
) -> list[Mapping[str, Any]]:
    """Validate the scientific schedule semantics."""

    expected_fields = dict(active_contract_fields or {"contract_epoch": ACTIVE_EPOCH})
    for key, value in expected_fields.items():
        _require(
            schedule.get(key) == value,
            f"realized schedule contract role mismatch: {key}",
        )
    _require(
        schedule.get("policy_id") in {None, policy.policy_id}
        and schedule.get("policy_sha256") in {None, policy.policy_hash},
        "realized schedule uses another scientific policy",
    )
    _require(
        schedule.get("completed_promotable_search") is True,
        "post-36 promotable search is incomplete",
    )
    gates = _mapping(schedule.get("promotion_gates"), label="promotion gates")
    _require(gates and all(value is True for value in gates.values()),
             "not every frozen promotion gate passed")
    _require(
        schedule.get("procedure_locked") is False
        and schedule.get("full_search_null_completed") is False
        and schedule.get("final_connectivity_accessed") is False
        and schedule.get("final_connectivity_access_authorized") is False,
        "realized schedule crosses a downstream or final-data boundary",
    )
    slots = [
        _mapping(item, label="realized schedule slot")
        for item in _sequence(schedule.get("realized_slots"), label="realized slots")
    ]
    _require(
        int(schedule.get("realized_slot_count", -1)) == len(slots),
        "realized slot count mismatch",
    )
    trial_ids: set[str] = set()
    for slot in slots:
        trial_id = slot.get("trial_id")
        _require(
            isinstance(trial_id, str) and trial_id and trial_id not in trial_ids,
            "realized schedule repeats or omits a trial ID",
        )
        trial_ids.add(trial_id)
        for key, value in expected_fields.items():
            _require(slot.get(key) == value, f"slot contract role mismatch: {trial_id}/{key}")
        configuration = _mapping(
            slot.get("configuration"), label=f"configuration {trial_id}"
        )
        _require(
            configuration.get("trial_id") == trial_id,
            f"slot configuration identity mismatch: {trial_id}",
        )
        _require(
            digest_object(configuration) == slot.get("configuration_sha256")
            and _scientific_signature(configuration) == slot.get("scientific_signature"),
            f"slot scientific configuration changed: {trial_id}",
        )
        _sha(f"slot {trial_id} identity", slot.get("slot_sha256"))
        context = slot.get("proposal_context")
        if context is not None:
            _require(isinstance(context, Mapping), f"slot proposal context is invalid: {trial_id}")
    return slots


def _validate_comparison_record(record: Mapping[str, Any]) -> dict[str, Any]:
    trial_id = record.get("trial_id")
    _require(isinstance(trial_id, str) and trial_id, "comparison lacks trial_id")
    configuration = dict(
        _mapping(record.get("configuration"), label=f"configuration {trial_id}")
    )
    configuration_sha256 = _sha(
        f"comparison {trial_id} configuration", record.get("configuration_sha256")
    )
    _require(
        configuration.get("trial_id") == trial_id
        and digest_object(configuration) == configuration_sha256,
        f"comparison configuration changed: {trial_id}",
    )
    signature = _sha(
        f"comparison {trial_id} signature", record.get("scientific_signature")
    )
    _require(
        signature == _scientific_signature(configuration),
        f"comparison scientific signature changed: {trial_id}",
    )
    _require(
        record.get("status") == "completed"
        and record.get("valid_complete_T_U_M_triplet") is True
        and record.get("technical_failure_count") == 0,
        f"comparison is not a complete failure-free T/U/M trial: {trial_id}",
    )
    _require(
        record.get("data_role") == "development_only"
        and record.get("final_connectivity_accessed") is False,
        f"comparison crossed the final-data boundary: {trial_id}",
    )
    objective = dict(_mapping(record.get("objective"), label=f"objective {trial_id}"))
    for key in (
        "bilateral_floor",
        "bilateral",
        "direction_disagreement",
        "types_both_above_margin",
    ):
        _finite(f"objective {trial_id}.{key}", objective.get(key))
    epoch = record.get("contract_epoch")
    contract_id = record.get("scientific_contract_id")
    _require(
        (epoch, contract_id)
        in {
            (PREFIX_EPOCH, PREFIX_CONTRACT_ID),
            (ACTIVE_EPOCH, ACTIVE_CONTRACT_ID),
        },
        f"comparison contract role is invalid: {trial_id}",
    )
    return {
        **dict(record),
        "trial_id": trial_id,
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "scientific_signature": signature,
        "objective": objective,
        "result_table_sha256": _sha(
            f"comparison {trial_id} result table", record.get("result_table_sha256")
        ),
        "contract_epoch": epoch,
        "scientific_contract_id": contract_id,
    }


def _select_comparison(
    archive: Mapping[str, Any],
    *,
    policy: EpisodePolicy,
    slots: Sequence[Mapping[str, Any]],
    **_: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    _require(
        archive.get("policy_sha256") in {None, policy.policy_hash},
        "comparison archive uses another scientific policy",
    )
    _require(
        archive.get("complete") is True
        and archive.get("final_connectivity_accessed") is False,
        "comparison archive is incomplete or crosses the final-data boundary",
    )
    initial_ids = [
        str(value)
        for value in _sequence(archive.get("initial_trial_ids"), label="initial trial IDs")
    ]
    _require(
        len(initial_ids) == policy.minimum_valid_trials == 36
        and len(set(initial_ids)) == len(initial_ids),
        "comparison archive does not contain the unique 18+9+9 prefix",
    )
    tail_ids = [str(slot["trial_id"]) for slot in slots]
    _require(set(initial_ids).isdisjoint(tail_ids), "prefix and tail trial IDs overlap")
    partition = _mapping(archive.get("contract_partition"), label="contract partition")
    prefix = _mapping(partition.get("initial_prefix"), label="prefix partition")
    tail = _mapping(partition.get("post36_tail"), label="tail partition")
    _require(
        prefix.get("contract_epoch") == PREFIX_EPOCH
        and prefix.get("scientific_contract_id") == PREFIX_CONTRACT_ID
        and list(prefix.get("trial_ids", [])) == initial_ids,
        "prefix contract partition is ambiguous",
    )
    _require(
        tail.get("contract_epoch") == ACTIVE_EPOCH
        and tail.get("scientific_contract_id") == ACTIVE_CONTRACT_ID
        and list(tail.get("trial_ids", [])) == tail_ids
        and partition.get("prevalence_parameter_rank_definition") == RANK_RULE,
        "post-36 contract partition is ambiguous",
    )
    records = [
        _validate_comparison_record(_mapping(value, label="comparison record"))
        for value in _sequence(archive.get("records"), label="comparison records")
    ]
    by_id = {record["trial_id"]: record for record in records}
    _require(
        len(by_id) == len(records)
        and set(by_id) == set(initial_ids) | set(tail_ids),
        "comparison archive does not exactly cover prefix and tail trials",
    )
    _require(
        len({record["configuration_sha256"] for record in records}) == len(records),
        "comparison archive repeats a scientific configuration",
    )
    for trial_id in initial_ids:
        record = by_id[trial_id]
        _require(
            record.get("source_stage") == "initial_minimum"
            and record.get("selection_eligible") is True
            and record.get("contract_epoch") == PREFIX_EPOCH,
            f"prefix comparison has the wrong role: {trial_id}",
        )
    for slot in slots:
        trial_id = str(slot["trial_id"])
        record = by_id[trial_id]
        _require(
            record.get("source_stage") == "post36_tail"
            and record.get("contract_epoch") == ACTIVE_EPOCH
            and record.get("configuration_sha256") == slot.get("configuration_sha256")
            and record.get("scientific_signature") == slot.get("scientific_signature")
            and record.get("selection_eligible")
            is (slot.get("promotion_eligible") is True),
            f"tail comparison differs from the realized schedule: {trial_id}",
        )
    candidates = [record for record in records if record.get("selection_eligible") is True]
    _require(candidates, "comparison archive has no selection-eligible comparison")
    selected = sorted(candidates, key=_ranking_key)[0]
    _require(
        archive.get("selected_trial_id") == selected["trial_id"]
        and archive.get("selected_configuration_sha256")
        == selected["configuration_sha256"]
        and archive.get("selected_source_contract_epoch")
        == selected["contract_epoch"]
        and archive.get("selected_source_contract_id")
        == selected["scientific_contract_id"],
        "comparison archive selection disagrees with the frozen ranking",
    )
    return selected, records, digest_object(archive)


def _normalize_controls(
    inventory: Mapping[str, Any],
    *,
    policy: EpisodePolicy,
    slots: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], bool]:
    _require(
        inventory.get("policy_sha256") in {None, policy.policy_hash}
        and inventory.get("final_connectivity_accessed") is False,
        "control inventory uses another policy or final data",
    )
    supplied: dict[str, Mapping[str, Any]] = {}
    for value in _sequence(inventory.get("controls"), label="control results"):
        record = _mapping(value, label="control result")
        identifier = record.get("falsifier_id")
        _require(
            isinstance(identifier, str) and identifier and identifier not in supplied,
            "control inventory repeats or omits a falsifier ID",
        )
        supplied[identifier] = record
    required = list(policy.required_falsifiers)
    extra = sorted(set(supplied) - set(required))
    _require(not extra, f"control inventory contains unregistered controls: {extra}")
    observed_ids = set(required) - {
        FITTED_NULL_FALSIFIER_ID,
        NOVELTY_AUDIT_FALSIFIER_ID,
    }
    missing: list[str] = []
    normalized: list[dict[str, Any]] = []
    observed_passed = True
    for identifier in required:
        record = supplied.get(identifier)
        if record is None:
            missing.append(identifier)
            if identifier in observed_ids:
                observed_passed = False
            continue
        _require(
            record.get("final_connectivity_accessed") is False,
            f"control crossed the final-data boundary: {identifier}",
        )
        status = record.get("status")
        if identifier in observed_ids:
            passed = (
                status == "passed"
                and record.get("applicable") is True
                and record.get("control_passed") is True
            )
            observed_passed = observed_passed and passed
        elif identifier == FITTED_NULL_FALSIFIER_ID:
            _require(status == "pending_full_search_null", "fitted-null control is not pending")
        else:
            _require(
                status == "pending_post_result_novelty_audit",
                "novelty audit is not pending post-result",
            )
        normalized.append(dict(record))
    return normalized, missing, observed_passed


def _validate_generator_selection(selection: Mapping[str, Any]) -> str:
    _require(
        selection.get("complete") is True
        and selection.get("development_only") is True
        and selection.get("final_connectivity_accessed") is False,
        "fitted-null generator selection is incomplete or unsafe",
    )
    choices = _mapping(
        selection.get("selected_generator_by_type_side"),
        label="selected fitted-null generators",
    )
    _require(choices, "fitted-null generator selection is empty")
    _require(
        all(isinstance(key, str) and value in {"T", "U"} for key, value in choices.items()),
        "every type-side generator must be T or U",
    )
    return digest_object(selection)


def build_procedure_lock(
    *,
    policy: EpisodePolicy,
    comparison_archive: Mapping[str, Any],
    mode_support_result: Mapping[str, Any],
    realized_schedule: Mapping[str, Any],
    control_result_inventory: Mapping[str, Any],
    null_generator_selection: Mapping[str, Any],
    null_controller_program: Mapping[str, Any],
    pre_null_manifest: Mapping[str, Any],
    runtime_source_binding: Mapping[str, Any],
    prefix_contract_id: str = PREFIX_CONTRACT_ID,
    active_contract_id: str = ACTIVE_CONTRACT_ID,
    **_: Any,
) -> dict[str, Any]:
    """Build a development-only lock from scientific inputs."""

    _require(
        prefix_contract_id == PREFIX_CONTRACT_ID
        and active_contract_id == ACTIVE_CONTRACT_ID,
        "procedure lock received unexpected scientific contracts",
    )
    active_fields = {
        "contract_epoch": ACTIVE_EPOCH,
        "prevalence_parameter_rank_definition": RANK_RULE,
    }
    slots = _validate_schedule(
        realized_schedule,
        policy=policy,
        active_contract_fields=active_fields,
    )
    selected, records, archive_sha256 = _select_comparison(
        comparison_archive, policy=policy, slots=slots
    )
    _require(
        realized_schedule.get("incumbent_trial_id") == selected["trial_id"]
        and realized_schedule.get("incumbent_configuration_sha256")
        == selected["configuration_sha256"],
        "realized schedule and frozen ranking select different comparisons",
    )
    try:
        verify_mode_support_result(mode_support_result)
    except ModeSupportInferenceError as error:
        raise ProcedureLockError(str(error)) from error
    _require(
        mode_support_result.get("comparison_trial_id") == selected["trial_id"]
        and mode_support_result.get("configuration_sha256")
        == selected["configuration_sha256"]
        and mode_support_result.get("result_table_sha256")
        == selected["result_table_sha256"]
        and mode_support_result.get("selected_source_contract_epoch")
        == selected["contract_epoch"]
        and mode_support_result.get("selected_source_contract_id")
        == selected["scientific_contract_id"]
        and mode_support_result.get("refit_contract_epoch") == ACTIVE_EPOCH
        and mode_support_result.get("refit_contract_id") == ACTIVE_CONTRACT_ID,
        "mode-support result belongs to another comparison or contract role",
    )
    _require(
        float(mode_support_result.get("meaningful_margin")) == policy.meaningful_margin,
        "mode-support result changes the meaningful margin",
    )
    if mode_support_result.get("complete_input") is True:
        _require(
            mode_support_result.get("bootstrap", {}).get("replicates")
            == BOOTSTRAP_REPLICATES,
            "mode-support result changes the bootstrap count",
        )
    controls, missing_controls, observed_controls_passed = _normalize_controls(
        control_result_inventory, policy=policy, slots=slots
    )
    generator_selection_sha256 = _validate_generator_selection(
        null_generator_selection
    )
    _require(
        null_generator_selection.get("selected_trial_id") == selected["trial_id"]
        and null_generator_selection.get("selected_configuration_sha256")
        == selected["configuration_sha256"]
        and null_generator_selection.get("refit_contract_epoch") == ACTIVE_EPOCH
        and null_generator_selection.get("refit_contract_id") == ACTIVE_CONTRACT_ID,
        "fitted-null generators belong to another comparison or contract role",
    )
    try:
        from .full_search_null import verify_null_controller_program

        verified_controller = verify_null_controller_program(
            null_controller_program, policy=policy
        )
    except Exception as error:
        raise ProcedureLockError("null controller program is invalid") from error

    for artifact, label in (
        (pre_null_manifest, "pre-null inputs"),
        (runtime_source_binding, "runtime source binding"),
    ):
        _require(
            artifact.get("prefix_contract_id") == PREFIX_CONTRACT_ID
            and artifact.get("active_contract_id") == ACTIVE_CONTRACT_ID
            and artifact.get("prevalence_parameter_rank_definition") == RANK_RULE
            and artifact.get("selected_trial_id") == selected["trial_id"]
            and artifact.get("selected_configuration_sha256")
            == selected["configuration_sha256"]
            and artifact.get("development_only") is True
            and artifact.get("final_connectivity_accessed") is False
            and artifact.get("final_connectivity_access_authorized") is False,
            f"{label} differs from the selected development-only procedure",
        )
    _require(
        pre_null_manifest.get("comparison_archive_sha256") == archive_sha256
        == runtime_source_binding.get("comparison_archive_sha256")
        and pre_null_manifest.get("generator_selection_sha256")
        == generator_selection_sha256
        == runtime_source_binding.get("null_generator_selection_sha256")
        and pre_null_manifest.get("null_controller_program_sha256")
        == verified_controller["program_sha256"]
        == runtime_source_binding.get("null_controller_program_sha256"),
        "pre-null scientific inputs are internally inconsistent",
    )
    pre_null_binding = {
        "bundle_sha256": _sha(
            "pre-null scientific bundle", pre_null_manifest.get("bundle_sha256")
        ),
        "runtime_source_binding_sha256": _sha(
            "runtime source binding",
            runtime_source_binding.get("source_binding_sha256"),
        ),
        "callback_data_sha256": _sha(
            "runtime callback data", pre_null_manifest.get("callback_data_sha256")
        ),
        "model_state_sha256": _sha(
            "selected model state", pre_null_manifest.get("model_state_sha256")
        ),
        "snapshot_binding_sha256": _sha(
            "development snapshot binding",
            pre_null_manifest.get("snapshot_binding_sha256"),
        ),
    }
    procedure_locked = not missing_controls
    null_eligible = bool(procedure_locked and observed_controls_passed)
    mode_complete = mode_support_result.get("complete_input") is True
    mode_supported = mode_support_result.get("scientific_mode_support") is True
    unresolved: list[str] = []
    if missing_controls:
        unresolved.append("required_control_results_missing")
    if not observed_controls_passed:
        unresolved.append("observed_control_support_not_established")
    if not mode_complete:
        unresolved.append("mode_support_inputs_incomplete")
    if mode_complete and not mode_supported:
        unresolved.append("simultaneous_mode_support_not_established")

    core = {
        "policy_id": policy.policy_id,
        "policy_sha256": policy.policy_hash,
        "prefix_contract_id": PREFIX_CONTRACT_ID,
        "active_contract_id": ACTIVE_CONTRACT_ID,
        "prefix_contract_epoch": PREFIX_EPOCH,
        "active_contract_epoch": ACTIVE_EPOCH,
        "prevalence_parameter_rank_definition": RANK_RULE,
        "comparison_archive_record_count": len(records),
        "selected_comparison": {
            "trial_id": selected["trial_id"],
            "configuration": selected["configuration"],
            "configuration_sha256": selected["configuration_sha256"],
            "scientific_signature": selected["scientific_signature"],
            "objective": selected["objective"],
            "result_table_sha256": selected["result_table_sha256"],
            "source_contract_epoch": selected["contract_epoch"],
            "source_contract_id": selected["scientific_contract_id"],
            "deterministic_refit_contract_epoch": ACTIVE_EPOCH,
            "deterministic_refit_contract_id": ACTIVE_CONTRACT_ID,
        },
        "mode_support_inference_sha256": mode_support_result["inference_sha256"],
        "mode_support_state": mode_support_result["support_state"],
        "mode_support_gates": mode_support_result["gates"],
        "required_control_results": controls,
        "missing_required_control_results": missing_controls,
        "observed_controls_passed": observed_controls_passed,
        "realized_schedule": {
            "plan_sha256": realized_schedule.get("plan_sha256"),
            "stop_reason": realized_schedule.get("stop_reason"),
            "stop_valid_trial_ordinal": realized_schedule.get(
                "stop_valid_trial_ordinal"
            ),
            "realized_controller_program_sha256": digest_object(
                realized_schedule.get("realized_slots", [])
            ),
            **active_fields,
        },
        "null_generator_selection_sha256": generator_selection_sha256,
        "null_controller_program_sha256": verified_controller["program_sha256"],
        "pre_null_input_binding": pre_null_binding,
        "procedure_locked": procedure_locked,
        "scientific_support": bool(
            procedure_locked and observed_controls_passed and mode_supported
        ),
        "scientific_support_state": (
            "supported"
            if procedure_locked and observed_controls_passed and mode_supported
            else "unresolved"
        ),
        "null_eligible": null_eligible,
        "null_eligibility_decision": (
            "eligible_run_frozen_full_search_null"
            if null_eligible
            else "not_eligible_mechanical_qualification_incomplete"
        ),
        "unresolved_reasons": unresolved,
        "pending_required_controls": [
            FITTED_NULL_FALSIFIER_ID,
            NOVELTY_AUDIT_FALSIFIER_ID,
        ],
        "full_search_null_contract": {
            "required_replicates": policy.null_replicates,
            "pass_threshold": policy.null_pass_threshold,
            "rerun_complete_frozen_controller": True,
            "runtime_must_bind_this_procedure_lock": True,
            "runtime_must_bind_selected_comparison": True,
            "observed_realized_schedule_may_be_replayed": False,
        },
        "full_search_null_started": False,
        "full_search_null_completed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "procedure_lock_sha256": digest_object(core)}


def verify_procedure_lock(lock: Mapping[str, Any], *, policy: EpisodePolicy) -> None:
    """Verify the immutable scientific lock and its safety boundaries."""

    claimed = _sha("procedure lock", lock.get("procedure_lock_sha256"))
    core = {key: value for key, value in lock.items() if key != "procedure_lock_sha256"}
    _require(digest_object(core) == claimed, "procedure lock changed")
    _require(
        lock.get("policy_id") == policy.policy_id
        and lock.get("policy_sha256") == policy.policy_hash,
        "procedure lock uses another scientific policy",
    )
    _require(
        lock.get("prefix_contract_id") == PREFIX_CONTRACT_ID
        and lock.get("active_contract_id") == ACTIVE_CONTRACT_ID
        and lock.get("prefix_contract_epoch") == PREFIX_EPOCH
        and lock.get("active_contract_epoch") == ACTIVE_EPOCH
        and lock.get("prevalence_parameter_rank_definition") == RANK_RULE,
        "procedure lock has ambiguous contract roles",
    )
    _require(
        lock.get("final_connectivity_accessed") is False
        and lock.get("final_connectivity_access_authorized") is False,
        "procedure lock crosses the final-data boundary",
    )
    null_contract = _mapping(
        lock.get("full_search_null_contract"), label="full-search-null contract"
    )
    _require(
        null_contract.get("required_replicates") == policy.null_replicates
        and float(null_contract.get("pass_threshold"))
        == policy.null_pass_threshold
        and null_contract.get("rerun_complete_frozen_controller") is True
        and null_contract.get("observed_realized_schedule_may_be_replayed") is False,
        "procedure lock changes the frozen full-search-null program",
    )
    binding = _mapping(lock.get("pre_null_input_binding"), label="pre-null binding")
    required = {
        "bundle_sha256",
        "runtime_source_binding_sha256",
        "callback_data_sha256",
        "model_state_sha256",
        "snapshot_binding_sha256",
    }
    _require(
        set(binding) == required
        and all(_SHA256_RE.fullmatch(str(binding[key])) for key in required),
        "procedure lock lacks a complete scientific input binding",
    )
    _require(
        lock.get("null_eligible") is not True
        or lock.get("procedure_locked") is True,
        "an unlocked procedure claims null eligibility",
    )


def write_procedure_lock(output_dir: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Create the one write-once procedure-lock file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "procedure_lock.json"
    encoded = (canonical_json(lock) + "\n").encode("utf-8")
    try:
        publish_once(path, encoded)
    except (FileExistsError, OSError) as error:
        raise ProcedureLockError(
            f"refusing unsafe or different procedure lock: {error}"
        ) from error
    return {
        "procedure_lock_path": str(path),
        "procedure_lock_sha256": lock["procedure_lock_sha256"],
        "procedure_locked": lock["procedure_locked"],
        "null_eligible": lock["null_eligible"],
        "final_connectivity_accessed": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProcedureLockError(f"cannot read {path}: {error}") from error
    return dict(_mapping(value, label=str(path)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--prefix-contract", type=Path, required=True)
    parser.add_argument("--active-contract", type=Path, required=True)
    parser.add_argument("--comparison-archive", type=Path, required=True)
    parser.add_argument("--mode-support", type=Path, required=True)
    parser.add_argument("--realized-schedule", type=Path, required=True)
    parser.add_argument("--control-results", type=Path, required=True)
    parser.add_argument("--null-generator-selection", type=Path, required=True)
    parser.add_argument("--null-controller-program", type=Path, required=True)
    parser.add_argument("--pre-null-manifest", type=Path, required=True)
    parser.add_argument("--runtime-source-binding", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    policy = EpisodePolicy.load(arguments.policy)
    prefix = load_yaml_subset(arguments.prefix_contract)
    active = load_yaml_subset(arguments.active_contract)
    lock = build_procedure_lock(
        policy=policy,
        prefix_contract_id=str(prefix.get("contract_id")),
        active_contract_id=str(active.get("contract_id")),
        comparison_archive=_read_json(arguments.comparison_archive),
        mode_support_result=_read_json(arguments.mode_support),
        realized_schedule=_read_json(arguments.realized_schedule),
        control_result_inventory=_read_json(arguments.control_results),
        null_generator_selection=_read_json(arguments.null_generator_selection),
        null_controller_program=_read_json(arguments.null_controller_program),
        pre_null_manifest=_read_json(arguments.pre_null_manifest),
        runtime_source_binding=_read_json(arguments.runtime_source_binding),
    )
    verify_procedure_lock(lock, policy=policy)
    summary = write_procedure_lock(arguments.output_dir, lock)
    print(canonical_json(summary))
    return 0 if lock["procedure_locked"] else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FITTED_NULL_FALSIFIER_ID",
    "NOVELTY_AUDIT_FALSIFIER_ID",
    "ProcedureLockError",
    "_validate_schedule",
    "build_procedure_lock",
    "verify_procedure_lock",
    "write_procedure_lock",
]
