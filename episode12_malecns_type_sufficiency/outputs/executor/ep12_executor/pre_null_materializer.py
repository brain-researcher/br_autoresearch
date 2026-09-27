"""Produce EP12's authenticated production pre-null input bundle.

The materializer is the only bridge from completed development trials to the
mode-support/procedure-lock layer.  It authenticates the 18+9+9 prefix and every
realized tail bundle, applies the already-frozen ranking, then deterministically
refits only that winner to recover model state which legacy trial artifacts did
not persist.  Refit prediction gains must replay the durable selected result
table within the contract tolerance; aggregate scores are never substituted for
whole-provider-type evidence.

There is no final-connectivity loader or final-role path in this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as parquet
from scipy import sparse

from .adaptive_search import validate_development_configuration
from .contract_roles import (
    ACTIVE_EPOCH,
    PREFIX_EPOCH,
    RANK_RULE,
    active_record_fields,
    load_contract_provenance,
    prefix_record_fields,
    require_record_fields,
    validate_contract_provenance,
)
from .adaptive_successors import _ranking_key, _scientific_signature
from .development_trial import _fixed_covariates
from .fitted_null import FittedNullDesign, fitted_generator_manifest
from .mode_support import (
    component_partner_contrasts,
    match_component_contrasts,
    point_mode_diagnostics,
)
from .mode_support_inference import infer_mode_support, verify_mode_support_result
from .full_search_runtime import (
    _load_callback_data,
    _write_callback_data,
)
from .model_artifacts import (
    atomic_json,
    load_model_state,
    write_model_state,
)
from .null_trial_runner import _collapse, _fit_options
from .policy import EpisodePolicy, digest_object
from .procedure_lock import (
    FITTED_NULL_FALSIFIER_ID,
    NOVELTY_AUDIT_FALSIFIER_ID,
    _validate_schedule,
)
from .scientific_models import _fit_common, fit_triplet, seed_from
from .scientific_qualification import _contract_settings
from .yaml_subset import load_yaml_subset


EPISODE_ROOT = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency"
)
OUTPUTS_ROOT = EPISODE_ROOT / "outputs"
SCRATCH_ROOT = Path(
    "/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency"
)
DEFAULT_POST36_PLAN = OUTPUTS_ROOT / "executor" / "OBSERVED_POST36_TAIL_PLAN.json"
PREFIX_CONTRACT_ID = "ep12_outgoing_tum_v1"
ACTIVE_CONTRACT_ID = "ep12_outgoing_tum_post36_v2"


class PreNullMaterializerError(RuntimeError):
    """A development artifact cannot support an authenticated pre-null input."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreNullMaterializerError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreNullMaterializerError(f"{label} must be an object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PreNullMaterializerError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PreNullMaterializerError(f"malformed {label}: {path}") from exc
    return dict(_mapping(value, label))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError as exc:
        raise PreNullMaterializerError(f"missing authenticated artifact: {path}") from exc
    return digest.hexdigest()


def _inside(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    boundary = root.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise PreNullMaterializerError(f"{label} must stay under {boundary}: {resolved}")
    return resolved


def _initial_result_dir(run_root: Path, trial_id: str) -> Path:
    if trial_id.startswith("coverage_"):
        return run_root / "observed" / trial_id
    if trial_id.startswith("adaptive_c1_"):
        return run_root / "adaptive" / "cycle1" / "observed" / trial_id
    if trial_id.startswith("adaptive_c2_"):
        return run_root / "adaptive" / "cycle2" / "observed" / trial_id
    raise PreNullMaterializerError(f"unrecognized initial trial identity: {trial_id}")


def _read_json_value(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PreNullMaterializerError(f"missing {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PreNullMaterializerError(f"malformed {label}: {path}") from exc


def _trial_objective(report: Mapping[str, Any], trial_id: str) -> dict[str, Any]:
    aggregate = _mapping(report.get("aggregate"), f"aggregate {trial_id}")
    try:
        left_to_right = float(aggregate["mean_left_to_right_delta"])
        right_to_left = float(aggregate["mean_right_to_left_delta"])
        bilateral = float(aggregate["mean_bilateral_delta"])
        above_margin = int(aggregate["types_with_both_point_estimates_above_margin"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreNullMaterializerError(
            f"trial {trial_id} has an incomplete scientific objective"
        ) from exc
    _require(
        all(math.isfinite(value) for value in (left_to_right, right_to_left, bilateral)),
        f"trial {trial_id} has a nonfinite scientific objective",
    )
    return {
        "left_to_right": left_to_right,
        "right_to_left": right_to_left,
        "bilateral": bilateral,
        "bilateral_floor": min(left_to_right, right_to_left),
        "direction_disagreement": abs(left_to_right - right_to_left),
        "types_both_above_margin": above_margin,
    }


def _validate_trial_outputs(
    *,
    trial_dir: Path,
    trial_id: str,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Check scientific completeness directly, without a hash attestation."""

    required_files = {
        "trial_report.json",
        "result_table.csv",
        "type_direction_receipts.jsonl",
    }
    manifest = _read_json(trial_dir / "manifest.json", f"trial manifest {trial_id}")
    declared_files = _mapping(manifest.get("files"), f"trial files {trial_id}")
    _require(
        required_files <= set(map(str, declared_files)),
        f"trial {trial_id} lacks a required scientific output",
    )
    _require(
        manifest.get("final_connectivity_accessed") is False,
        f"trial {trial_id} crossed the final-connectivity boundary",
    )
    for name in sorted(required_files):
        path = trial_dir / name
        _require(
            path.is_file() and not path.is_symlink(),
            f"trial {trial_id} lacks safe output {name}",
        )

    report_path = trial_dir / "trial_report.json"
    report = _read_json(report_path, f"trial report {trial_id}")
    report_configuration = _mapping(
        report.get("configuration"), f"trial configuration {trial_id}"
    )
    _require(
        report.get("trial_id") == trial_id
        and report.get("status") == "completed"
        and report.get("data_role") == "development_only"
        and report.get("final_connectivity_accessed") is False,
        f"trial {trial_id} is not a completed development-only result",
    )
    _require(
        all(report_configuration.get(key) == value for key, value in configuration.items()),
        f"trial {trial_id} does not match its planned configuration",
    )
    configuration_sha256 = digest_object(dict(configuration))
    _require(
        report_configuration.get("configuration_sha256") == configuration_sha256,
        f"trial {trial_id} configuration identity changed",
    )
    aggregate = _mapping(report.get("aggregate"), f"aggregate {trial_id}")
    try:
        technical_failures = int(aggregate["technical_failure_count"])
        valid_type_count = int(aggregate["valid_type_count"])
        unscorable_type_count = int(aggregate["unscorable_type_count"])
        wall_seconds = float(report["wall_seconds"])
        cpu_seconds = float(report["cpu_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreNullMaterializerError(
            f"trial {trial_id} has incomplete status or timing fields"
        ) from exc
    _require(
        technical_failures == 0
        and valid_type_count > 0
        and unscorable_type_count >= 0
        and report.get("failures") == [],
        f"trial {trial_id} is not a failure-free complete comparison",
    )
    _require(
        math.isfinite(wall_seconds)
        and wall_seconds > 0.0
        and math.isfinite(cpu_seconds)
        and cpu_seconds > 0.0,
        f"trial {trial_id} lacks positive finite timing",
    )

    valid_receipt_types: set[str] = set()
    unscorable_receipt_types: set[str] = set()
    receipt_path = trial_dir / "type_direction_receipts.jsonl"
    try:
        receipt_lines = receipt_path.read_text(encoding="utf-8").splitlines()
        receipts = [json.loads(line) for line in receipt_lines if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreNullMaterializerError(
            f"trial {trial_id} has malformed type-direction results"
        ) from exc
    for raw in receipts:
        receipt = _mapping(raw, f"type-direction result {trial_id}")
        provider_type = str(receipt.get("provider_type", ""))
        _require(
            provider_type
            and provider_type not in valid_receipt_types
            and provider_type not in unscorable_receipt_types,
            f"trial {trial_id} repeats or omits a provider type",
        )
        if receipt.get("status") == "valid_complete_T_U_M_triplet":
            for direction in ("left_to_right_scores", "right_to_left_scores"):
                scores = _mapping(receipt.get(direction), f"{direction} {trial_id}")
                _require(
                    set(scores) == {"T", "U", "M"}
                    and all(math.isfinite(float(scores[key])) for key in ("T", "U", "M")),
                    f"trial {trial_id} lacks a complete T/U/M score triplet",
                )
            valid_receipt_types.add(provider_type)
        elif receipt.get("status") == "unscorable":
            _require(
                bool(receipt.get("reason_code")),
                f"trial {trial_id} has an unexplained unscorable provider type",
            )
            unscorable_receipt_types.add(provider_type)
        else:
            raise PreNullMaterializerError(
                f"trial {trial_id} has an unknown provider-type status"
            )
    _require(
        len(valid_receipt_types) == valid_type_count
        and len(unscorable_receipt_types) == unscorable_type_count,
        f"trial {trial_id} provider-type status counts are incomplete",
    )
    raw_unscorable = report.get("unscorable")
    _require(
        isinstance(raw_unscorable, list)
        and {
            str(_mapping(value, f"unscorable row {trial_id}").get("provider_type", ""))
            for value in raw_unscorable
        }
        == unscorable_receipt_types,
        f"trial {trial_id} unscorable status rows disagree",
    )

    result_types: set[str] = set()
    try:
        with (trial_dir / "result_table.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            reader = csv.DictReader(stream)
            required_columns = {
                "provider_type",
                "left_to_right_delta",
                "right_to_left_delta",
            }
            _require(
                reader.fieldnames is not None
                and required_columns <= set(reader.fieldnames),
                f"trial {trial_id} result table lacks scientific columns",
            )
            for row in reader:
                provider_type = str(row.get("provider_type", ""))
                _require(
                    provider_type and provider_type not in result_types,
                    f"trial {trial_id} result table repeats a provider type",
                )
                _require(
                    all(
                        math.isfinite(float(row[name]))
                        for name in ("left_to_right_delta", "right_to_left_delta")
                    ),
                    f"trial {trial_id} result table contains a nonfinite gain",
                )
                result_types.add(provider_type)
    except (OSError, TypeError, ValueError) as exc:
        raise PreNullMaterializerError(
            f"trial {trial_id} has a malformed scientific result table"
        ) from exc
    _require(
        result_types == valid_receipt_types,
        f"trial {trial_id} result table does not match its complete T/U/M rows",
    )
    return {
        "trial_id": trial_id,
        "configuration": dict(configuration),
        "configuration_sha256": configuration_sha256,
        "objective": _trial_objective(report, trial_id),
        "valid_type_count": valid_type_count,
        "unscorable_type_count": unscorable_type_count,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "trial_report_path": str(report_path),
        "status": "completed",
        "valid_complete_T_U_M_triplet": True,
        "data_role": "development_only",
        "final_connectivity_accessed": False,
    }


def load_observed_trial_status(
    *,
    run_root: Path,
    policy: EpisodePolicy,
    outputs_root: Path = OUTPUTS_ROOT,
) -> dict[str, Any]:
    """Load the exact 18+9+9 development prefix by scientific structure."""

    root = _inside(run_root, outputs_root, "observed run root")
    sources = (
        (
            "coverage",
            root / "search_plan" / "coverage_configurations.json",
            root / "observed",
            [f"coverage_{index:03d}" for index in range(1, 19)],
        ),
        (
            "adaptive_cycle1",
            root / "adaptive" / "cycle1" / "plan" / "successor_configurations.json",
            root / "adaptive" / "cycle1" / "observed",
            [f"adaptive_c1_{index:03d}" for index in range(1, 10)],
        ),
        (
            "adaptive_cycle2",
            root / "adaptive" / "cycle2" / "plan" / "successor_configurations.json",
            root / "adaptive" / "cycle2" / "observed",
            [f"adaptive_c2_{index:03d}" for index in range(1, 10)],
        ),
    )
    rows: list[dict[str, Any]] = []
    signatures: set[str] = set()
    for source_stage, configuration_path, result_root, expected_ids in sources:
        payload = _read_json_value(configuration_path, f"{source_stage} configurations")
        if source_stage == "coverage":
            configurations = payload
        else:
            configurations = _mapping(
                payload, f"{source_stage} configuration plan"
            ).get("configurations")
        _require(
            isinstance(configurations, list),
            f"{source_stage} configurations must be a list",
        )
        trial_ids = [
            str(_mapping(value, f"{source_stage} configuration").get("trial_id", ""))
            for value in configurations
        ]
        _require(
            trial_ids == expected_ids,
            f"{source_stage} does not contain its exact frozen trial sequence",
        )
        for raw_configuration in configurations:
            configuration = dict(
                _mapping(raw_configuration, f"{source_stage} configuration")
            )
            validate_development_configuration(policy, configuration)
            expected_stage = "coverage" if source_stage == "coverage" else "adaptive"
            _require(
                configuration.get("stage") == expected_stage
                and configuration.get("models")
                == ["T_type_template", "U_flexible_unimodal", "M_residual_modes"]
                and configuration.get("data_mode") == "development",
                f"{configuration.get('trial_id')} is not a frozen development T/U/M comparison",
            )
            signature = _scientific_signature(configuration)
            _require(
                signature not in signatures,
                f"development prefix repeats a scientific configuration: {configuration['trial_id']}",
            )
            signatures.add(signature)
            row = _validate_trial_outputs(
                trial_dir=result_root / str(configuration["trial_id"]),
                trial_id=str(configuration["trial_id"]),
                configuration=configuration,
            )
            row["source_stage"] = source_stage
            rows.append(row)
    _require(
        len(rows) == policy.minimum_valid_trials == 36,
        "development prefix is not the frozen 18+9+9 minimum",
    )
    return {
        "trial_count": len(rows),
        "coverage_trial_count": 18,
        "adaptive_cycle_trial_counts": {"cycle1": 9, "cycle2": 9},
        "trial_ids": [str(row["trial_id"]) for row in rows],
        "rows_by_id": {str(row["trial_id"]): row for row in rows},
        "independent_scientific_configuration_count": len(signatures),
        "all_trials_complete_T_U_M": True,
        "development_only": True,
        "final_connectivity_accessed": False,
    }


def _validate_observed_finalization(
    *,
    finalization_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check the frozen prefix summary semantically, without receipt chaining."""

    archive = _read_json(
        finalization_dir / "observed_search_archive.json",
        "observed search archive",
    )
    archive_rows = archive.get("archive")
    _require(isinstance(archive_rows, list), "observed search archive lacks rows")
    by_id = {str(row["trial_id"]): row for row in rows}
    _require(
        len(by_id) == len(rows) == 36
        and int(archive.get("verified_valid_trial_count", -1)) == 36
        and len(archive_rows) == 36,
        "observed search archive is not the complete 18+9+9 prefix",
    )
    archived_ids: set[str] = set()
    for raw in archive_rows:
        archived = _mapping(raw, "observed search archive row")
        trial_id = str(archived.get("trial_id", ""))
        _require(
            trial_id in by_id and trial_id not in archived_ids,
            f"observed archive has an unknown or duplicate trial {trial_id}",
        )
        archived_ids.add(trial_id)
        direct = by_id[trial_id]
        _require(
            archived.get("configuration_sha256") == direct["configuration_sha256"]
            and archived.get("objective") == direct["objective"],
            f"observed archive differs from direct scientific result {trial_id}",
        )
    _require(
        archived_ids == set(by_id),
        "observed search archive omits a development trial",
    )
    ranked = sorted((dict(row) for row in rows), key=_ranking_key)
    _require(
        archive.get("provisional_incumbent_trial_id") == ranked[0]["trial_id"]
        and archive.get("provisional_incumbent_configuration_sha256")
        == ranked[0]["configuration_sha256"],
        "observed archive provisional incumbent is inconsistent",
    )
    _require(
        archive.get("final_connectivity_accessed") is False
        and archive.get("final_connectivity_access_authorized") is False,
        "observed search archive crosses the final-connectivity boundary",
    )
    return {
        "verified_valid_trial_count": 36,
        "initial_search_complete": True,
        "provisional_incumbent_trial_id": ranked[0]["trial_id"],
        "final_connectivity_accessed": False,
    }


def _archive_record(
    record: Mapping[str, Any],
    *,
    source_stage: str,
    selection_eligible: bool,
    result_table_sha256: str,
    contract_fields: Mapping[str, Any],
) -> dict[str, Any]:
    configuration = dict(_mapping(record.get("configuration"), "configuration"))
    trial_id = str(record.get("trial_id", ""))
    _require(configuration.get("trial_id") == trial_id,
             f"configuration identity mismatch: {trial_id}")
    configuration_sha256 = digest_object(configuration)
    _require(record.get("configuration_sha256") == configuration_sha256,
             f"configuration digest mismatch: {trial_id}")
    return {
        "trial_id": trial_id,
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "scientific_signature": _scientific_signature(configuration),
        "source_stage": source_stage,
        "selection_eligible": bool(selection_eligible),
        "status": "completed",
        "valid_complete_T_U_M_triplet": True,
        "data_role": "development_only",
        "final_connectivity_accessed": False,
        "technical_failure_count": 0,
        "objective": dict(_mapping(record.get("objective"), f"objective {trial_id}")),
        "valid_type_count": int(record["valid_type_count"]),
        "unscorable_type_count": int(record["unscorable_type_count"]),
        "result_table_sha256": result_table_sha256,
        **dict(contract_fields),
    }


def _load_tail_bundle(
    tail_root: Path,
    slot: Mapping[str, Any],
    contract_provenance: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any], Path]:
    trial_id = str(slot["trial_id"])
    directory = tail_root / "slots" / trial_id
    realization = _read_json(directory / "realization.json", f"tail realization {trial_id}")
    require_record_fields(realization, contract_provenance, active=True,
                          label=f"tail realization {trial_id}")
    _require(realization.get("trial_id") == trial_id
             and realization.get("slot_sha256") == slot.get("slot_sha256")
             and realization.get("configuration_sha256")
             == slot.get("configuration_sha256"),
             f"tail realization identity mismatch: {trial_id}")
    _require(realization.get("status") == "completed"
             and realization.get("valid_complete_T_U_M_triplet") is True
             and realization.get("technical_failure_count") == 0,
             f"tail realization is not a valid complete T/U/M result: {trial_id}")
    _require(realization.get("data_role") == "development_only"
             and realization.get("final_connectivity_accessed") is False,
             f"tail realization crossed the data-role boundary: {trial_id}")
    control = _read_json(directory / "control_result.json", f"control result {trial_id}")
    require_record_fields(control, contract_provenance, active=True,
                          label=f"tail control {trial_id}")
    _require(realization.get("control_result") == control,
             f"tail realization embeds a different control result: {trial_id}")
    evidence = _mapping(realization.get("evidence"), f"tail evidence {trial_id}")
    relative_result = Path(str(evidence.get("result_table", "")))
    _require(
        relative_result.parts and not relative_result.is_absolute()
        and ".." not in relative_result.parts,
        f"tail result-table path is invalid: {trial_id}",
    )
    result_table = (directory / relative_result).resolve(strict=True)
    _require(
        directory.resolve() in result_table.parents and result_table.is_file(),
        f"tail result table is missing: {trial_id}",
    )
    normalized = {
        "trial_id": trial_id,
        "configuration": dict(_mapping(slot.get("configuration"), "tail configuration")),
        "configuration_sha256": slot["configuration_sha256"],
        "objective": dict(_mapping(realization.get("objective"), "tail objective")),
        "valid_type_count": int(realization["valid_type_count"]),
        "unscorable_type_count": int(realization["unscorable_type_count"]),
        **active_record_fields(contract_provenance),
    }
    return normalized, control, result_table


def build_archive_and_controls(
    *,
    run_root: Path,
    observed_finalization_dir: Path,
    tail_root: Path,
    policy: EpisodePolicy,
    contract_provenance: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
]:
    """Authenticate and merge the complete initial and realized-tail archive."""

    provenance = validate_contract_provenance(contract_provenance)
    prefix_fields = {
        **prefix_record_fields(provenance),
        "scientific_contract_id": PREFIX_CONTRACT_ID,
    }
    active_role_fields = active_record_fields(provenance)
    active_fields = {
        **active_role_fields,
        "scientific_contract_id": ACTIVE_CONTRACT_ID,
    }
    run_root = _inside(run_root, OUTPUTS_ROOT, "run root")
    observed_finalization_dir = _inside(
        observed_finalization_dir, OUTPUTS_ROOT, "observed finalization"
    )
    tail_root = _inside(tail_root, OUTPUTS_ROOT, "tail root")
    initial_status = load_observed_trial_status(
        run_root=run_root,
        policy=policy,
        outputs_root=OUTPUTS_ROOT,
    )
    initial_ids = [str(value) for value in initial_status["trial_ids"]]
    rows_by_id = _mapping(initial_status.get("rows_by_id"), "initial trial rows")
    prefix = [
        dict(_mapping(rows_by_id[trial_id], f"initial trial {trial_id}"))
        for trial_id in initial_ids
    ]
    prefix_evidence = _validate_observed_finalization(
        finalization_dir=observed_finalization_dir,
        rows=prefix,
    )
    records: list[dict[str, Any]] = []
    result_paths: dict[str, Path] = {}
    for record in prefix:
        trial_id = str(record["trial_id"])
        result_path = _initial_result_dir(run_root, trial_id) / "result_table.csv"
        result_paths[trial_id] = result_path
        records.append(
            _archive_record(
                record,
                source_stage="initial_minimum",
                selection_eligible=True,
                result_table_sha256=_sha256_file(result_path),
                contract_fields=prefix_fields,
            )
        )

    schedule_path = tail_root / "realized_schedule.json"
    schedule = _read_json(schedule_path, "realized schedule")
    slots = _validate_schedule(
        schedule,
        policy=policy,
        active_contract_fields=active_role_fields,
    )
    observed_control_hashes: dict[str, list[str]] = {
        identifier: []
        for identifier in policy.required_falsifiers
        if identifier not in {
            FITTED_NULL_FALSIFIER_ID,
            NOVELTY_AUDIT_FALSIFIER_ID,
        }
    }
    for slot in slots:
        normalized, control, result_path = _load_tail_bundle(
            tail_root, slot, provenance
        )
        trial_id = str(slot["trial_id"])
        result_paths[trial_id] = result_path
        records.append(
            _archive_record(
                normalized,
                source_stage="post36_tail",
                selection_eligible=slot.get("promotion_eligible") is True,
                result_table_sha256=_sha256_file(result_path),
                contract_fields=active_fields,
            )
        )
        falsifier_id = slot.get("registered_falsifier_id")
        if falsifier_id is not None:
            _require(falsifier_id in observed_control_hashes,
                     f"tail contains an unregistered observed control: {falsifier_id}")
            _require(control.get("falsifier_id") == falsifier_id
                     and control.get("control_passed") is True
                     and control.get("applicability_checks_passed") is True
                     and control.get("conservation_checks_passed") is True
                     and control.get("final_connectivity_accessed") is False,
                     f"observed control did not pass: {falsifier_id}")
            observed_control_hashes[str(falsifier_id)].append(
                digest_object(control)
            )

    eligible = [record for record in records if record["selection_eligible"]]
    _require(eligible, "complete comparison archive has no eligible comparison")
    selected = sorted(eligible, key=_ranking_key)[0]
    _require(schedule.get("incumbent_trial_id") == selected["trial_id"]
             and schedule.get("incumbent_configuration_sha256")
             == selected["configuration_sha256"],
             "complete archive ranking disagrees with realized schedule incumbent")
    archive = {
        "policy_sha256": policy.policy_hash,
        "initial_trial_count": len(initial_ids),
        "initial_trial_status": "complete_development_T_U_M",
        "initial_trial_ids": initial_ids,
        "records": records,
        "selected_trial_id": selected["trial_id"],
        "selected_configuration_sha256": selected["configuration_sha256"],
        "selected_source_contract_epoch": selected["contract_epoch"],
        "selected_source_contract_id": selected["scientific_contract_id"],
        "contract_partition": {
            "initial_prefix": {
                "contract_epoch": PREFIX_EPOCH,
                "scientific_contract_id": PREFIX_CONTRACT_ID,
                "trial_ids": initial_ids,
            },
            "post36_tail": {
                "contract_epoch": ACTIVE_EPOCH,
                "scientific_contract_id": ACTIVE_CONTRACT_ID,
                "trial_ids": [str(slot["trial_id"]) for slot in slots],
            },
            "prevalence_parameter_rank_definition": RANK_RULE,
        },
        "source_evidence": {
            **prefix_evidence,
            "coverage_trial_count": initial_status["coverage_trial_count"],
            "adaptive_cycle_trial_counts": initial_status[
                "adaptive_cycle_trial_counts"
            ],
            "independent_scientific_configuration_count": initial_status[
                "independent_scientific_configuration_count"
            ],
            "all_trials_complete_T_U_M": True,
            "development_only": True,
        },
        "complete": True,
        "final_connectivity_accessed": False,
    }
    registrations = {
        FITTED_NULL_FALSIFIER_ID: {
            "status": "pending_full_search_null",
            "policy_sha256": policy.policy_hash,
            "realized_controller_program_sha256": schedule[
                "realized_controller_program_sha256"
            ],
        },
        NOVELTY_AUDIT_FALSIFIER_ID: {
            "status": "pending_post_result_novelty_audit",
            "policy_sha256": policy.policy_hash,
            "selected_configuration_sha256": selected["configuration_sha256"],
        },
    }
    controls: list[dict[str, Any]] = []
    for identifier in policy.required_falsifiers:
        if identifier == FITTED_NULL_FALSIFIER_ID:
            status = "pending_full_search_null"
            applicable = passed = None
            evidence = [digest_object(registrations[identifier])]
        elif identifier == NOVELTY_AUDIT_FALSIFIER_ID:
            status = "pending_post_result_novelty_audit"
            applicable = passed = None
            evidence = [digest_object(registrations[identifier])]
        else:
            evidence = sorted(observed_control_hashes[identifier])
            _require(evidence, f"realized schedule omits required control: {identifier}")
            status = "passed"
            applicable = passed = True
        controls.append(
            {
                "falsifier_id": identifier,
                "status": status,
                "applicable": applicable,
                "control_passed": passed,
                "evidence_sha256s": evidence,
                "final_connectivity_accessed": False,
            }
        )
    inventory = {
        "policy_sha256": policy.policy_hash,
        "controls": controls,
        "pending_registrations": registrations,
        "complete_required_falsifier_inventory": True,
        "final_connectivity_accessed": False,
    }
    return (
        archive,
        inventory,
        selected,
        result_paths[str(selected["trial_id"])],
    )


def build_null_controller_program(
    *,
    policy: EpisodePolicy,
    comparison_archive: Mapping[str, Any],
    post36_plan_path: Path,
) -> dict[str, Any]:
    """Freeze only coverage and the outcome-blind adaptive program."""

    from .full_search_null import (
        freeze_null_controller_program,
        verify_null_controller_program,
    )

    post36_plan_path = _inside(
        post36_plan_path, OUTPUTS_ROOT, "outcome-blind post-36 plan"
    )
    plan = _read_json(post36_plan_path, "outcome-blind post-36 plan")
    records = {
        str(record["trial_id"]): record
        for record in _mapping_sequence(
            comparison_archive.get("records"), "comparison archive records"
        )
    }
    initial_ids = [
        str(value)
        for value in _plain_sequence(
            comparison_archive.get("initial_trial_ids"), "initial trial IDs"
        )
    ]
    _require(
        len(initial_ids) == policy.minimum_valid_trials
        and policy.coverage_minimum == 18,
        "comparison archive does not contain the frozen 18+9+9 prefix",
    )
    coverage_configurations: list[dict[str, Any]] = []
    for trial_id in initial_ids[: policy.coverage_minimum]:
        _require(trial_id in records, f"coverage comparison is absent: {trial_id}")
        record = records[trial_id]
        configuration = dict(
            _mapping(record.get("configuration"), f"coverage configuration {trial_id}")
        )
        _require(
            record.get("source_stage") == "initial_minimum"
            and configuration.get("stage") == "coverage"
            and configuration.get("trial_id") == trial_id
            and digest_object(configuration) == record.get("configuration_sha256"),
            f"coverage comparison cannot seed null controller: {trial_id}",
        )
        coverage_configurations.append(configuration)
    classification = _mapping(
        plan.get("required_falsifier_classification"),
        "post-36 falsifier classification",
    )
    countable = [
        str(value)
        for value in _plain_sequence(
            classification.get("countable_observed_trial_ids"),
            "countable observed falsifiers",
        )
    ]
    non_trial: list[str] = []
    for raw in _mapping_sequence(
        plan.get("non_trial_required_gates"), "non-trial required gates"
    ):
        identifier = str(raw.get("falsifier_id", ""))
        _require(identifier and identifier not in non_trial,
                 "non-trial falsifier classification is empty or duplicated")
        non_trial.append(identifier)
    _require(
        set(countable) | set(non_trial) == set(policy.required_falsifiers)
        and not (set(countable) & set(non_trial)),
        "outcome-blind plan falsifier classes do not partition SEARCH_POLICY",
    )
    program = freeze_null_controller_program(
        policy=policy,
        coverage_configurations=coverage_configurations,
        post36_tail_plan=plan,
        countable_falsifier_ids=countable,
        non_trial_required_falsifier_ids=non_trial,
    )
    verify_null_controller_program(program, policy=policy)
    return program




def _plain_sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PreNullMaterializerError(f"{label} must be a list")
    return list(value)


def _mapping_sequence(value: Any, label: str) -> list[Mapping[str, Any]]:
    return [_mapping(item, label) for item in _plain_sequence(value, label)]


def _read_result_table(path: Path) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                provider_type = str(row["provider_type"])
                _require(provider_type and provider_type not in result,
                         f"selected result table repeats provider type {provider_type!r}")
                values = (
                    float(row["left_to_right_delta"]),
                    float(row["right_to_left_delta"]),
                )
                _require(all(math.isfinite(value) for value in values),
                         f"selected result table contains nonfinite gain: {provider_type}")
                result[provider_type] = values
    except (KeyError, ValueError) as exc:
        raise PreNullMaterializerError(
            f"selected result table has an invalid schema: {path}"
        ) from exc
    _require(result, "selected result table contains no provider types")
    return result


def _component_rank(
    *,
    rule: str,
    fitted: Any,
    source_neuron_count: int,
    configuration: Mapping[str, Any],
) -> int:
    dimension = int(np.asarray(fitted.M.prior_means).shape[1])
    _require(dimension > 0 and source_neuron_count > 1,
             "component-rank calculation lacks fitted dimension or neurons")
    if rule != "representation_latent_effective_rank":
        raise PreNullMaterializerError(
            "prevalence parameter-rank rule is not the frozen post-36 choice: "
            "representation_latent_effective_rank"
        )
    diagnostics = _mapping(fitted.M.diagnostics, "fitted M diagnostics")
    _require(
        "representation_rank" in diagnostics,
        "fitted M lacks the required representation-rank diagnostic",
    )
    raw_value = diagnostics["representation_rank"]
    _require(
        isinstance(raw_value, (int, np.integer)) and not isinstance(raw_value, bool),
        "fitted representation-rank diagnostic is not an integer",
    )
    value = int(raw_value)
    maximum = min(dimension, source_neuron_count - 1)
    representation = str(diagnostics.get("representation", ""))
    _require(
        representation == str(configuration["representation"]),
        "fitted representation diagnostic differs from configuration",
    )
    if representation == "count_aware_log_ratio":
        expected = maximum
    elif representation in {"low_rank_composition", "low_rank_count_model"}:
        expected = max(1, min(int(configuration["rank"]), maximum))
    else:
        raise PreNullMaterializerError(
            f"unsupported fitted representation for rank definition: {representation!r}"
        )
    _require(
        value == expected,
        "fitted representation-rank diagnostic differs from the effective representation rank",
    )
    return value


def fit_selected_mode_and_generators(
    *,
    policy: EpisodePolicy,
    selected: Mapping[str, Any],
    selected_result_table: Path,
    active_contract_id: str,
    contract_provenance: Mapping[str, Any],
    numerical_replay_tolerance: float,
    prevalence_parameter_rank_definition: str,
    neuron_ids: Sequence[int | str],
    focal_types: Sequence[str],
    sides: Sequence[str],
    raw_counts: np.ndarray,
    raw_vocabulary: Sequence[str],
    partner_hierarchy: Mapping[str, Mapping[str, Any]],
    covariates_by_spline_df: Mapping[int, np.ndarray],
    fit_settings: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    FittedNullDesign,
    dict[tuple[str, str], Any],
]:
    """Refit one frozen winner and emit whole-type diagnostics plus generators."""

    provenance = validate_contract_provenance(contract_provenance)
    _require(
        active_contract_id == ACTIVE_CONTRACT_ID
        and provenance["active_epoch"] == ACTIVE_EPOCH
        and provenance["prevalence_parameter_rank_definition"] == RANK_RULE,
        "selected refit must use the active post-36 scientific role",
    )
    configuration = dict(_mapping(selected.get("configuration"), "selected configuration"))
    validate_development_configuration(policy, configuration)
    _require(digest_object(configuration) == selected.get("configuration_sha256"),
             "selected configuration digest mismatch")
    selected_epoch = selected.get("contract_epoch")
    expected_source_contract_id = {
        PREFIX_EPOCH: PREFIX_CONTRACT_ID,
        ACTIVE_EPOCH: ACTIVE_CONTRACT_ID,
    }.get(selected_epoch)
    _require(
        expected_source_contract_id is not None
        and selected.get("scientific_contract_id") == expected_source_contract_id,
        "selected comparison lacks a recognized scientific contract role",
    )
    mode_support_contract_seed = digest_object(
        {
            "contract_id": active_contract_id,
            "contract_epoch": ACTIVE_EPOCH,
            "prevalence_parameter_rank_definition": RANK_RULE,
        }
    )
    ids = np.asarray(list(neuron_ids), dtype=object)
    types = np.asarray(list(map(str, focal_types)), dtype=object)
    side_values = np.asarray(list(map(str, sides)), dtype=object)
    raw = np.asarray(raw_counts, dtype=np.int64)
    _require(raw.ndim == 2 and raw.shape == (len(ids), len(raw_vocabulary)),
             "selected refit raw-count dimensions disagree")
    _require(len(types) == len(ids) and len(side_values) == len(ids),
             "selected refit neuron metadata dimensions disagree")
    _require(np.all(raw >= 0) and set(map(str, side_values)) <= {"L", "R"},
             "selected refit has negative counts or invalid sides")
    spline_df = int(configuration["nuisance_spline_df"])
    _require(spline_df in covariates_by_spline_df,
             "selected refit lacks its nuisance-spline covariates")
    covariates = np.asarray(covariates_by_spline_df[spline_df], dtype=np.float64)
    _require(covariates.ndim == 2 and covariates.shape[0] == len(ids)
             and np.all(np.isfinite(covariates)),
             "selected refit covariates are malformed")
    collapsed, collapsed_vocabulary, raw_to_collapsed = _collapse(
        raw,
        types,
        list(map(str, raw_vocabulary)),
        configuration,
        partner_hierarchy,
    )
    options = _fit_options(configuration, fit_settings)
    minimum = int(configuration["component_count"]) * int(
        fit_settings["source_cv_folds"]
    )
    scorable: list[str] = []
    for provider_type in sorted(set(map(str, types))):
        mask = types == provider_type
        left = mask & (side_values == "L")
        right = mask & (side_values == "R")
        if (
            int(left.sum()) >= minimum
            and int(right.sum()) >= minimum
            and np.all(collapsed[left].sum(axis=1) > 0)
            and np.all(collapsed[right].sum(axis=1) > 0)
        ):
            scorable.append(provider_type)
    durable_gains = _read_result_table(selected_result_table)
    _require(set(scorable) == set(durable_gains),
             "selected deterministic refit scorable types differ from durable result table")

    prediction_gain: list[list[float]] = []
    symmetric_kl: list[list[list[float]]] = []
    matched_cosine: list[list[float]] = []
    partner_contrasts: list[list[list[float]]] = []
    memberships: list[list[list[float]]] = []
    parameter_ranks: list[list[list[int]]] = []
    generators: dict[tuple[str, str], Any] = {}
    coordinate_metadata: list[dict[str, Any]] = []
    for provider_type in scorable:
        type_mask = types == provider_type
        left = type_mask & (side_values == "L")
        right = type_mask & (side_values == "R")
        base_seed = seed_from("ep12-development", configuration["trial_id"], provider_type)
        left_fit = fit_triplet(
            collapsed[left],
            covariates[left],
            seed=seed_from(base_seed, "left-source"),
            **options,
        )
        right_fit = fit_triplet(
            collapsed[right],
            covariates[right],
            seed=seed_from(base_seed, "right-source"),
            **options,
        )
        left_score = float(
            left_fit.score_target(collapsed[right], covariates[right])[
                "mean_delta_M_minus_reference"
            ]
        )
        right_score = float(
            right_fit.score_target(collapsed[left], covariates[left])[
                "mean_delta_M_minus_reference"
            ]
        )
        replayed = np.asarray([left_score, right_score], dtype=float)
        durable = np.asarray(durable_gains[provider_type], dtype=float)
        _require(np.all(np.abs(replayed - durable) <= numerical_replay_tolerance),
                 f"selected refit does not replay durable gains: {provider_type}")

        source_diagnostics: list[dict[str, Any]] = []
        source_ranks: list[list[int]] = []
        for mask, fitted in ((left, left_fit), (right, right_fit)):
            basis, coefficients, residuals, _ = _fit_common(
                collapsed[mask],
                covariates[mask],
                ridge=float(configuration["regularization"]),
                pseudocount=float(configuration["composition_pseudocount"]),
                representation=str(configuration["representation"]),
                latent_rank=int(configuration["rank"]),
            )
            _require(np.allclose(basis, fitted.M.basis, atol=1e-12, rtol=0.0)
                     and np.allclose(
                         coefficients,
                         fitted.M.nuisance_coefficients,
                         atol=1e-12,
                         rtol=0.0,
                     ),
                     f"source residual replay differs from fitted M: {provider_type}")
            rank = _component_rank(
                rule=prevalence_parameter_rank_definition,
                fitted=fitted,
                source_neuron_count=int(mask.sum()),
                configuration=configuration,
            )
            diagnostic = point_mode_diagnostics(
                fitted.M,
                residuals,
                fitted_component_parameter_rank=rank,
            )
            source_diagnostics.append(diagnostic)
            source_ranks.append([rank] * int(configuration["component_count"]))

        left_contrast = np.asarray(
            source_diagnostics[0]["component_partner_contrasts"], dtype=float
        )
        right_contrast = np.asarray(
            source_diagnostics[1]["component_partner_contrasts"], dtype=float
        )
        matching = match_component_contrasts(left_contrast, right_contrast)
        left_order = np.asarray(matching["left_component_indices"], dtype=int)
        right_order = np.asarray(matching["right_component_indices"], dtype=int)
        left_aligned = left_contrast[left_order]
        right_aligned = right_contrast[right_order]
        pair_indices = np.triu_indices(int(configuration["component_count"]), k=1)
        left_kl = np.asarray(
            source_diagnostics[0]["pairwise_symmetric_kl"], dtype=float
        )[pair_indices]
        right_kl = np.asarray(
            source_diagnostics[1]["pairwise_symmetric_kl"], dtype=float
        )[pair_indices]

        prediction_gain.append(replayed.tolist())
        symmetric_kl.append([left_kl.tolist(), right_kl.tolist()])
        matched_cosine.append(list(map(float, matching["matched_cosines"])))
        partner_contrasts.append(
            [left_aligned.reshape(-1).tolist(), right_aligned.reshape(-1).tolist()]
        )
        memberships.append(
            [
                list(map(float, source_diagnostics[0]["effective_component_memberships"])),
                list(map(float, source_diagnostics[1]["effective_component_memberships"])),
            ]
        )
        parameter_ranks.append(source_ranks)
        generators[(provider_type, "L")] = (
            left_fit.T if left_fit.null_generator_model_id == "T" else left_fit.U
        )
        generators[(provider_type, "R")] = (
            right_fit.T if right_fit.null_generator_model_id == "T" else right_fit.U
        )
        coordinate_metadata.append(
            {
                "provider_type": provider_type,
                "matched_left_component_indices": left_order.tolist(),
                "matched_right_component_indices": right_order.tolist(),
            }
        )

    keep = np.isin(types, np.asarray(scorable, dtype=object))
    typed_bins = sorted(
        {
            collapsed_vocabulary[int(raw_to_collapsed[index])]
            for index, key in enumerate(raw_vocabulary)
            if str(key).startswith("typed::")
        }
    )
    design = FittedNullDesign.create(
        neuron_ids=ids[keep].tolist(),
        focal_types=types[keep].tolist(),
        sides=side_values[keep].tolist(),
        covariates=covariates[keep],
        raw_counts=raw[keep],
        raw_vocabulary=list(map(str, raw_vocabulary)),
        collapsed_vocabulary=list(collapsed_vocabulary),
        raw_to_collapsed=raw_to_collapsed,
        collapsed_typed_bins=typed_bins,
    )
    generator_rows, generator_manifest_sha256 = fitted_generator_manifest(
        generators, design
    )
    selected_generator_by_type_side = {
        json.dumps([row["focal_type"], row["side"]], separators=(",", ":")): (
            "T" if row["model_id"] == "T" else "U"
        )
        for row in generator_rows
    }
    generator_selection = {
        "selected_trial_id": selected["trial_id"],
        "selected_configuration_sha256": selected["configuration_sha256"],
        "selected_result_table_sha256": _sha256_file(selected_result_table),
        "selected_source_contract_epoch": selected["contract_epoch"],
        "selected_source_contract_id": selected["scientific_contract_id"],
        "refit_contract_epoch": ACTIVE_EPOCH,
        "refit_contract_id": active_contract_id,
        "prevalence_parameter_rank_definition": RANK_RULE,
        "design_sha256": design.design_sha256,
        "generator_unit": "focal_provider_type_by_side",
        "selected_generator_by_type_side": selected_generator_by_type_side,
        "generators": list(generator_rows),
        "generator_manifest_sha256": generator_manifest_sha256,
        "null_generator_rule": (
            "U when source-CV U is within the frozen comparability tolerance of T; "
            "otherwise T"
        ),
        "complete": True,
        "development_only": True,
        "final_connectivity_accessed": False,
    }
    mode = infer_mode_support(
        comparison_trial_id=str(selected["trial_id"]),
        configuration_sha256=str(selected["configuration_sha256"]),
        contract_sha256=mode_support_contract_seed,
        result_table_sha256=_sha256_file(selected_result_table),
        meaningful_margin=policy.meaningful_margin,
        prevalence_parameter_rank_definition=prevalence_parameter_rank_definition,
        whole_provider_type_ids=scorable,
        prediction_gain_by_type_direction=np.asarray(prediction_gain),
        component_symmetric_kl_by_type_direction=np.asarray(symmetric_kl),
        matched_contrast_cosine_by_type=np.asarray(matched_cosine),
        partner_contrast_by_type_direction=np.asarray(partner_contrasts),
        effective_memberships_by_direction_component=np.asarray(memberships),
        fitted_parameter_ranks_by_direction_component=np.asarray(parameter_ranks),
    )
    verify_mode_support_result(mode)
    mode_core = {key: value for key, value in mode.items() if key != "inference_sha256"}
    mode_core["coordinate_metadata"] = {
        "collapsed_partner_vocabulary": list(collapsed_vocabulary),
        "partner_term_layout": "matched_component_major_then_collapsed_partner",
        "provider_type_component_matches": coordinate_metadata,
    }
    mode_core.update(
        {
            "selected_source_contract_epoch": selected["contract_epoch"],
            "selected_source_contract_id": selected["scientific_contract_id"],
            "refit_contract_epoch": ACTIVE_EPOCH,
            "refit_contract_id": active_contract_id,
        }
    )
    mode = dict(mode_core)
    mode["inference_sha256"] = digest_object(mode_core)
    verify_mode_support_result(mode)
    return mode, generator_selection, design, generators


def load_authenticated_development_snapshot(
    *,
    snapshot_dir: Path,
    materialization_dir: Path,
    preparation_dir: Path,
    policy: EpisodePolicy,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate and load the existing development-only snapshot."""

    snapshot_dir = _inside(snapshot_dir, SCRATCH_ROOT, "development snapshot")
    materialization_dir = _inside(
        materialization_dir, OUTPUTS_ROOT, "development materialization"
    )
    preparation_dir = _inside(preparation_dir, OUTPUTS_ROOT, "hierarchy preparation")
    material_manifest_path = materialization_dir / "manifest.json"
    material_manifest = _read_json(
        material_manifest_path, "development materialization manifest"
    )
    _require(
        material_manifest.get("final_focal_connectivity_materialized_or_summarized")
        is False,
        "development materialization manifest records final-role access",
    )
    scratch_hashes = _mapping(
        material_manifest.get("scratch_artifact_sha256"), "snapshot hashes"
    )
    expected_snapshot_files = {
        "development_outgoing_counts.npz",
        "development_focal_neurons.parquet",
        "development_body_statistics.parquet",
        "raw_partner_vocabulary.json",
    }
    _require(set(scratch_hashes) == expected_snapshot_files,
             "development snapshot inventory differs from frozen materialization")
    for name, expected in scratch_hashes.items():
        _require(_sha256_file(snapshot_dir / str(name)) == str(expected),
                 f"development snapshot digest mismatch: {name}")

    preparation_manifest_path = preparation_dir / "manifest.json"
    preparation_manifest = _read_json(
        preparation_manifest_path, "partner hierarchy manifest"
    )
    _require(preparation_manifest.get("final_connectivity_accessed") is False,
             "partner hierarchy manifest records final-connectivity access")
    hierarchy_path = snapshot_dir / "partner_type_hierarchy.json"
    _require(
        _sha256_file(hierarchy_path)
        == preparation_manifest.get("scratch_artifact_sha256"),
        "partner hierarchy scratch digest mismatch",
    )

    matrix = sparse.load_npz(snapshot_dir / "development_outgoing_counts.npz").tocsr()
    focal = parquet.read_table(
        snapshot_dir / "development_focal_neurons.parquet"
    ).to_pandas()
    stats = parquet.read_table(
        snapshot_dir / "development_body_statistics.parquet"
    ).to_pandas()
    raw_vocabulary = json.loads(
        (snapshot_dir / "raw_partner_vocabulary.json").read_text(encoding="utf-8")
    )
    hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))
    _require(isinstance(raw_vocabulary, list) and isinstance(hierarchy, Mapping),
             "development vocabulary or hierarchy is malformed")
    _require(matrix.shape == (len(focal), len(raw_vocabulary)),
             "development snapshot matrix dimensions disagree")
    _require(matrix.data.size == 0 or np.all(matrix.data >= 0),
             "development snapshot contains negative counts")
    covariates = {
        int(spline_df): _fixed_covariates(focal, stats, spline_df=int(spline_df))
        for spline_df in policy.grammar_choices["nuisance_spline_df"]
    }
    arrays = {
        "neuron_ids": focal["bodyId"].tolist(),
        "focal_types": focal["type"].astype(str).tolist(),
        "sides": focal["somaSide"].astype(str).tolist(),
        "raw_counts": matrix.toarray().astype(np.int64, copy=False),
        "raw_vocabulary": list(map(str, raw_vocabulary)),
        "partner_hierarchy": {
            str(key): dict(_mapping(value, f"hierarchy row {key}"))
            for key, value in hierarchy.items()
        },
        "covariates_by_spline_df": covariates,
    }
    binding_core = {
        "scratch_artifact_sha256": {
            **{str(key): str(value) for key, value in scratch_hashes.items()},
            "partner_type_hierarchy.json": _sha256_file(hierarchy_path),
        },
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    binding = dict(binding_core)
    binding["snapshot_binding_sha256"] = digest_object(binding_core)
    return arrays, binding


def verify_pre_null_inputs(
    output_dir: Path, *, policy_path: Path
) -> dict[str, Any]:
    """Authenticate an existing materialized input bundle for retry recovery."""

    directory = _inside(output_dir, OUTPUTS_ROOT, "pre-null input bundle")
    manifest = _read_json(directory / "manifest.json", "pre-null input manifest")
    core = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    _require(manifest.get("bundle_sha256") == digest_object(core),
             "pre-null scientific bundle identity changed")
    _require(manifest.get("development_only") is True
             and manifest.get("final_connectivity_accessed") is False
             and manifest.get("final_connectivity_access_authorized") is False,
             "pre-null input bundle crosses the final-connectivity boundary")
    mode = _read_json(directory / "mode_support.json", "mode-support result")
    verify_mode_support_result(mode)
    archive = _read_json(
        directory / "complete_comparison_archive.json", "comparison archive"
    )
    controls = _read_json(
        directory / "required_control_results.json", "control inventory"
    )
    selection = _read_json(
        directory / "null_generator_selection.json", "generator selection"
    )
    controller_program = _read_json(
        directory / "null_controller_program.json", "null controller program"
    )
    policy = EpisodePolicy.load(policy_path)
    try:
        from .full_search_null import verify_null_controller_program

        verify_null_controller_program(controller_program, policy=policy)
    except Exception as exc:
        raise PreNullMaterializerError(
            "null controller program failed authentication"
        ) from exc
    source_binding = _read_json(
        directory / "runtime_source_binding.json", "runtime source binding"
    )
    source_core = {
        key: value
        for key, value in source_binding.items()
        if key != "source_binding_sha256"
    }
    _require(
        source_binding.get("source_binding_sha256") == digest_object(source_core),
        "runtime source-binding digest mismatch",
    )
    _require(archive.get("complete") is True
             and archive.get("final_connectivity_accessed") is False,
             "comparison archive is incomplete or unsafe")
    _require(controls.get("complete_required_falsifier_inventory") is True
             and controls.get("final_connectivity_accessed") is False,
             "control inventory is incomplete or unsafe")
    _require(selection.get("complete") is True
             and selection.get("development_only") is True
             and selection.get("final_connectivity_accessed") is False,
             "generator selection is incomplete or unsafe")
    archive_records = _mapping_sequence(
        archive.get("records"), "comparison archive records"
    )
    archive_by_id = {
        str(record.get("trial_id", "")): record for record in archive_records
    }
    initial_ids = [
        str(value)
        for value in _plain_sequence(
            archive.get("initial_trial_ids"), "initial trial IDs"
        )
    ]
    _require(
        len(archive_by_id) == len(archive_records)
        and len(initial_ids) == len(set(initial_ids)) == policy.minimum_valid_trials == 36
        and set(initial_ids) <= set(archive_by_id)
        and len(
            {
                _scientific_signature(
                    dict(_mapping(archive_by_id[trial_id].get("configuration"), "configuration"))
                )
                for trial_id in initial_ids
            }
        )
        == len(initial_ids),
        "comparison archive loses trial identity or initial-prefix independence",
    )
    for trial_id, record in archive_by_id.items():
        epoch = record.get("contract_epoch")
        expected_contract_id = {
            PREFIX_EPOCH: PREFIX_CONTRACT_ID,
            ACTIVE_EPOCH: ACTIVE_CONTRACT_ID,
        }.get(epoch)
        _require(
            trial_id
            and expected_contract_id is not None
            and record.get("scientific_contract_id") == expected_contract_id
            and (
                epoch != ACTIVE_EPOCH
                or record.get("prevalence_parameter_rank_definition") == RANK_RULE
            )
            and record.get("valid_complete_T_U_M_triplet") is True
            and record.get("technical_failure_count") == 0
            and record.get("data_role") == "development_only"
            and record.get("final_connectivity_accessed") is False,
            f"comparison archive record is incomplete or role-ambiguous: {trial_id}",
        )
    selected_trial_id = str(archive.get("selected_trial_id", ""))
    _require(
        selected_trial_id in archive_by_id
        and archive_by_id[selected_trial_id].get("configuration_sha256")
        == archive.get("selected_configuration_sha256")
        and archive_by_id[selected_trial_id].get("contract_epoch")
        == archive.get("selected_source_contract_epoch")
        and archive_by_id[selected_trial_id].get("scientific_contract_id")
        == archive.get("selected_source_contract_id"),
        "comparison archive selected result is inconsistent",
    )
    control_rows = _mapping_sequence(controls.get("controls"), "falsifier controls")
    controls_by_id = {
        str(record.get("falsifier_id", "")): record for record in control_rows
    }
    _require(
        len(controls_by_id) == len(control_rows)
        and set(controls_by_id) == set(policy.required_falsifiers),
        "control inventory does not contain each frozen falsifier exactly once",
    )
    for identifier, record in controls_by_id.items():
        if identifier == FITTED_NULL_FALSIFIER_ID:
            passed = record.get("status") == "pending_full_search_null"
        elif identifier == NOVELTY_AUDIT_FALSIFIER_ID:
            passed = record.get("status") == "pending_post_result_novelty_audit"
        else:
            passed = (
                record.get("status") == "passed"
                and record.get("applicable") is True
                and record.get("control_passed") is True
            )
        _require(
            passed and record.get("final_connectivity_accessed") is False,
            f"frozen falsifier gate is incomplete: {identifier}",
        )
    _require(manifest.get("mode_support_inference_sha256")
             == mode.get("inference_sha256"),
             "manifest/mode-support digest mismatch")
    _require(manifest.get("comparison_archive_sha256") == digest_object(archive),
             "manifest/comparison archive digest mismatch")
    _require(manifest.get("control_inventory_sha256") == digest_object(controls),
             "manifest/control inventory digest mismatch")
    _require(manifest.get("generator_selection_sha256") == digest_object(selection),
             "manifest/generator selection digest mismatch")
    _require(
        manifest.get("runtime_source_binding_sha256")
        == source_binding.get("source_binding_sha256"),
        "manifest/runtime source-binding digest mismatch",
    )
    _require(
        source_binding.get("comparison_archive_sha256") == digest_object(archive)
        and source_binding.get("null_generator_selection_sha256")
        == digest_object(selection),
        "runtime source binding differs from pre-null scientific inputs",
    )
    partition = _mapping(archive.get("contract_partition"), "contract partition")
    prefix_partition = _mapping(partition.get("initial_prefix"), "prefix partition")
    active_partition = _mapping(partition.get("post36_tail"), "tail partition")
    _require(
        prefix_partition.get("contract_epoch") == PREFIX_EPOCH
        and prefix_partition.get("scientific_contract_id") == PREFIX_CONTRACT_ID
        and active_partition.get("contract_epoch") == ACTIVE_EPOCH
        and active_partition.get("scientific_contract_id") == ACTIVE_CONTRACT_ID
        and partition.get("prevalence_parameter_rank_definition") == RANK_RULE,
        "comparison archive contract epochs are ambiguous",
    )
    for artifact, label in (
        (manifest, "pre-null manifest"),
        (source_binding, "runtime source binding"),
    ):
        _require(
            artifact.get("prefix_contract_id") == PREFIX_CONTRACT_ID
            and artifact.get("active_contract_id") == ACTIVE_CONTRACT_ID
            and artifact.get("prevalence_parameter_rank_definition") == RANK_RULE
            and artifact.get("selected_source_contract_epoch")
            == archive.get("selected_source_contract_epoch")
            and artifact.get("selected_source_contract_id")
            == archive.get("selected_source_contract_id")
            and artifact.get("refit_contract_epoch") == ACTIVE_EPOCH
            and artifact.get("refit_contract_id") == ACTIVE_CONTRACT_ID,
            f"{label} contract roles differ from the mixed archive",
        )
    mode_support_contract_seed = digest_object(
        {
            "contract_id": ACTIVE_CONTRACT_ID,
            "contract_epoch": ACTIVE_EPOCH,
            "prevalence_parameter_rank_definition": RANK_RULE,
        }
    )
    _require(
        mode.get("contract_sha256") == mode_support_contract_seed
        and mode.get("selected_source_contract_epoch")
        == archive.get("selected_source_contract_epoch")
        and mode.get("selected_source_contract_id")
        == archive.get("selected_source_contract_id")
        and mode.get("refit_contract_epoch") == ACTIVE_EPOCH
        and mode.get("refit_contract_id") == ACTIVE_CONTRACT_ID,
        "mode-support result does not bind the active post-36 refit role",
    )
    _require(
        selection.get("selected_source_contract_epoch")
        == archive.get("selected_source_contract_epoch")
        and selection.get("selected_source_contract_id")
        == archive.get("selected_source_contract_id")
        and selection.get("refit_contract_epoch") == ACTIVE_EPOCH
        and selection.get("refit_contract_id") == ACTIVE_CONTRACT_ID
        and selection.get("prevalence_parameter_rank_definition") == RANK_RULE,
        "fitted-null selection does not bind the active post-36 refit role",
    )
    _require(
        manifest.get("null_controller_program_sha256")
        == controller_program.get("program_sha256")
        == source_binding.get("null_controller_program_sha256"),
        "pre-null controller program identity changed",
    )
    try:
        design, generators, state_manifest = load_model_state(
            directory / "selected_fitted_null_state"
        )
    except Exception as exc:
        raise PreNullMaterializerError(
            "selected fitted-null state failed authentication"
        ) from exc
    generator_rows, generator_manifest_sha256 = fitted_generator_manifest(
        generators, design
    )
    _require(
        state_manifest.get("state_sha256") == manifest.get("model_state_sha256")
        == selection.get("model_state_sha256")
        and design.design_sha256 == selection.get("design_sha256")
        and generator_manifest_sha256
        == selection.get("generator_manifest_sha256")
        and list(generator_rows) == selection.get("generators"),
        "selected fitted-null state differs from generator selection",
    )
    try:
        _, callback_metadata = _load_callback_data(
            directory / "runtime_callback_data",
            design=design,
            policy=policy,
            artifact_root=SCRATCH_ROOT / "pre_null_callback_verification",
        )
    except Exception as exc:
        raise PreNullMaterializerError(
            "runtime callback source failed authentication"
        ) from exc
    _require(
        callback_metadata.get("callback_data_sha256")
        == manifest.get("callback_data_sha256")
        == source_binding.get("callback_data_sha256"),
        "runtime callback source identity changed",
    )
    registration = controls.get("pending_registrations", {}).get(
        FITTED_NULL_FALSIFIER_ID, {}
    )
    _require(
        registration.get("null_controller_program_sha256")
        == controller_program["program_sha256"]
        and registration.get("observed_schedule_replayed") is False,
        "fitted-null control registration does not bind the null controller",
    )
    return manifest


def materialize_pre_null_inputs(
    *,
    output_dir: Path,
    run_root: Path,
    observed_finalization_dir: Path,
    tail_root: Path,
    snapshot_dir: Path,
    materialization_dir: Path,
    preparation_dir: Path,
    policy_path: Path,
    contract_path: Path,
    prefix_contract_path: Path,
    post36_plan_path: Path = DEFAULT_POST36_PLAN,
) -> dict[str, Any]:
    """Atomically materialize all four production pre-null inputs."""

    output_dir = _inside(output_dir, OUTPUTS_ROOT, "pre-null output")
    if output_dir.exists():
        return verify_pre_null_inputs(output_dir, policy_path=policy_path)
    policy = EpisodePolicy.load(policy_path)
    contract_provenance = load_contract_provenance(
        prefix_contract_path=prefix_contract_path,
        active_contract_path=contract_path,
    )
    prefix_contract = load_yaml_subset(prefix_contract_path)
    contract = load_yaml_subset(contract_path)
    _require(
        prefix_contract.get("contract_id") == PREFIX_CONTRACT_ID
        and contract.get("contract_id") == ACTIVE_CONTRACT_ID,
        "scientific contract roles do not match the frozen prefix/post-36 split",
    )
    for document, label in (
        (prefix_contract, "observed-prefix contract"),
        (contract, "active post-36 contract"),
    ):
        _require(
            document.get("real_connectivity_access_authorized") == "development_only"
            and document.get("final_connectivity_access_authorized", False) is False,
            f"{label} does not preserve the role firewall",
        )
    settings = _contract_settings(contract)
    rank_rule = contract.get("mode_support", {}).get(
        "prevalence_parameter_rank_definition"
    )
    _require(rank_rule == RANK_RULE,
             "active post36 contract must freeze representation_latent_effective_rank")
    replay_tolerance = float(
        contract["uncertainty"]["numerical_replay_tolerance_absolute"]
    )
    _require(math.isfinite(replay_tolerance) and replay_tolerance >= 0.0,
             "contract numerical replay tolerance is invalid")
    archive, controls, selected, result_table = (
        build_archive_and_controls(
            run_root=run_root,
            observed_finalization_dir=observed_finalization_dir,
            tail_root=tail_root,
            policy=policy,
            contract_provenance=contract_provenance,
        )
    )
    controller_program = build_null_controller_program(
        policy=policy,
        comparison_archive=archive,
        post36_plan_path=post36_plan_path,
    )
    controls = dict(controls)
    registrations = dict(controls["pending_registrations"])
    registrations[FITTED_NULL_FALSIFIER_ID] = {
        "status": "pending_full_search_null",
        "policy_sha256": policy.policy_hash,
        "null_controller_program_sha256": controller_program["program_sha256"],
        "source_post36_plan_sha256": controller_program[
            "source_post36_plan_sha256"
        ],
        "observed_schedule_replayed": False,
    }
    controls["pending_registrations"] = registrations
    normalized_controls = []
    for raw in controls["controls"]:
        record = dict(raw)
        if record["falsifier_id"] == FITTED_NULL_FALSIFIER_ID:
            record["evidence_sha256s"] = [
                digest_object(registrations[FITTED_NULL_FALSIFIER_ID])
            ]
        normalized_controls.append(record)
    controls["controls"] = normalized_controls
    arrays, snapshot_binding = load_authenticated_development_snapshot(
        snapshot_dir=snapshot_dir,
        materialization_dir=materialization_dir,
        preparation_dir=preparation_dir,
        policy=policy,
    )
    mode, generator_selection, design, generators = fit_selected_mode_and_generators(
        policy=policy,
        selected=selected,
        selected_result_table=result_table,
        active_contract_id=ACTIVE_CONTRACT_ID,
        contract_provenance=contract_provenance,
        numerical_replay_tolerance=replay_tolerance,
        prevalence_parameter_rank_definition=str(rank_rule),
        fit_settings={
            "integration_draws": int(settings["integration_draws"]),
            "source_cv_folds": int(settings["source_cv_folds"]),
        },
        **arrays,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}")
    _require(not stage.exists(), f"stale pre-null staging directory exists: {stage}")
    stage.mkdir()
    try:
        model_manifest = write_model_state(
            stage / "selected_fitted_null_state",
            design=design,
            generators=generators,
        )
        generator_selection = dict(generator_selection)
        generator_selection["model_state_sha256"] = model_manifest["state_sha256"]
        neuron_row = {
            str(value): index for index, value in enumerate(arrays["neuron_ids"])
        }
        _require(
            len(neuron_row) == len(arrays["neuron_ids"]),
            "development snapshot repeats neuron IDs",
        )
        try:
            keep = [neuron_row[str(value)] for value in design.neuron_ids]
        except KeyError as exc:
            raise PreNullMaterializerError(
                "selected fitted-null design is not a snapshot row subset"
            ) from exc
        filtered_covariates = {
            int(spline_df): np.ascontiguousarray(value[keep], dtype=np.float64)
            for spline_df, value in arrays["covariates_by_spline_df"].items()
        }
        callback_metadata = _write_callback_data(
            stage / "runtime_callback_data",
            design=design,
            covariates=filtered_covariates,
            hierarchy=arrays["partner_hierarchy"],
            settings={
                "integration_draws": int(settings["integration_draws"]),
                "source_cv_folds": int(settings["source_cv_folds"]),
            },
        )
        source_core = {
            "policy_sha256": policy.policy_hash,
            "prefix_contract_id": PREFIX_CONTRACT_ID,
            "active_contract_id": ACTIVE_CONTRACT_ID,
            "prevalence_parameter_rank_definition": RANK_RULE,
            "selected_source_contract_epoch": selected["contract_epoch"],
            "selected_source_contract_id": selected["scientific_contract_id"],
            "refit_contract_epoch": ACTIVE_EPOCH,
            "refit_contract_id": ACTIVE_CONTRACT_ID,
            "comparison_archive_sha256": digest_object(archive),
            "null_generator_selection_sha256": digest_object(
                generator_selection
            ),
            "null_controller_program_sha256": controller_program[
                "program_sha256"
            ],
            "callback_data_sha256": callback_metadata["callback_data_sha256"],
            "selected_trial_id": selected["trial_id"],
            "selected_configuration_sha256": selected[
                "configuration_sha256"
            ],
            "development_only": True,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        source_binding = dict(source_core)
        source_binding["source_binding_sha256"] = digest_object(source_core)
        atomic_json(stage / "complete_comparison_archive.json", archive)
        atomic_json(stage / "mode_support.json", mode)
        atomic_json(stage / "required_control_results.json", controls)
        atomic_json(stage / "null_generator_selection.json", generator_selection)
        atomic_json(stage / "null_controller_program.json", controller_program)
        atomic_json(stage / "runtime_source_binding.json", source_binding)
        snapshot_record_core = {
            **snapshot_binding,
            "policy_sha256": policy.policy_hash,
            "prefix_contract_id": PREFIX_CONTRACT_ID,
            "active_contract_id": ACTIVE_CONTRACT_ID,
            "selected_source_contract_epoch": selected["contract_epoch"],
            "selected_source_contract_id": selected["scientific_contract_id"],
            "refit_contract_epoch": ACTIVE_EPOCH,
            "refit_contract_id": ACTIVE_CONTRACT_ID,
            "prevalence_parameter_rank_definition": rank_rule,
            "selected_trial_id": selected["trial_id"],
            "selected_configuration_sha256": selected["configuration_sha256"],
        }
        snapshot_record_core.pop("snapshot_binding_sha256", None)
        snapshot_record = dict(snapshot_record_core)
        snapshot_record["snapshot_binding_sha256"] = digest_object(
            snapshot_record_core
        )
        atomic_json(stage / "snapshot_binding.json", snapshot_record)
        core = {
            "policy_sha256": policy.policy_hash,
            "prefix_contract_id": PREFIX_CONTRACT_ID,
            "active_contract_id": ACTIVE_CONTRACT_ID,
            "prevalence_parameter_rank_definition": RANK_RULE,
            "selected_source_contract_epoch": selected["contract_epoch"],
            "selected_source_contract_id": selected["scientific_contract_id"],
            "refit_contract_epoch": ACTIVE_EPOCH,
            "refit_contract_id": ACTIVE_CONTRACT_ID,
            "selected_trial_id": selected["trial_id"],
            "selected_configuration_sha256": selected["configuration_sha256"],
            "comparison_archive_sha256": digest_object(archive),
            "mode_support_inference_sha256": mode["inference_sha256"],
            "control_inventory_sha256": digest_object(controls),
            "generator_selection_sha256": digest_object(generator_selection),
            "null_controller_program_sha256": controller_program[
                "program_sha256"
            ],
            "callback_data_sha256": callback_metadata["callback_data_sha256"],
            "model_state_sha256": model_manifest["state_sha256"],
            "snapshot_binding_sha256": snapshot_record[
                "snapshot_binding_sha256"
            ],
            "runtime_source_binding_sha256": source_binding[
                "source_binding_sha256"
            ],
            "development_only": True,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        manifest = dict(core)
        manifest["bundle_sha256"] = digest_object(core)
        atomic_json(stage / "manifest.json", manifest)
        os.replace(stage, output_dir)
        descriptor = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return verify_pre_null_inputs(output_dir, policy_path=policy_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--observed-finalization-dir", type=Path, required=True)
    parser.add_argument("--tail-root", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--materialization-dir", type=Path, required=True)
    parser.add_argument("--preparation-dir", type=Path, required=True)
    parser.add_argument(
        "--policy", type=Path, required=True
    )
    parser.add_argument(
        "--contract",
        type=Path,
        required=True,
    )
    parser.add_argument("--prefix-contract", type=Path, required=True)
    parser.add_argument("--post36-plan", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = materialize_pre_null_inputs(
        output_dir=args.output_dir,
        run_root=args.run_root,
        observed_finalization_dir=args.observed_finalization_dir,
        tail_root=args.tail_root,
        snapshot_dir=args.snapshot_dir,
        materialization_dir=args.materialization_dir,
        preparation_dir=args.preparation_dir,
        policy_path=args.policy,
        contract_path=args.contract,
        prefix_contract_path=args.prefix_contract,
        post36_plan_path=args.post36_plan,
    )
    print(json.dumps({
        "bundle_sha256": manifest["bundle_sha256"],
        "selected_trial_id": manifest["selected_trial_id"],
        "mode_support_inference_sha256": manifest[
            "mode_support_inference_sha256"
        ],
        "final_connectivity_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PreNullMaterializerError",
    "build_archive_and_controls",
    "fit_selected_mode_and_generators",
    "load_authenticated_development_snapshot",
    "load_observed_trial_status",
    "materialize_pre_null_inputs",
    "verify_pre_null_inputs",
]
