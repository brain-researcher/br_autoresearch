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
from .contract_snapshot import capture_yaml_contract
from .contract_provenance import (
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
from .attest_observed_trials import verify_attestation_root
from .development_trial import _fixed_covariates
from .fitted_null import FittedNullDesign, fitted_generator_manifest
from .mode_support import (
    component_partner_contrasts,
    match_component_contrasts,
    point_mode_diagnostics,
)
from .mode_support_inference import infer_mode_support, verify_mode_support_result
from .full_search_null import (
    freeze_realized_controller_schedule,
    verify_realized_controller_schedule,
)
from .full_search_runtime import (
    _load_callback_data,
    _write_callback_data,
    freeze_selected_fit_attestation,
    load_runtime_bundle,
    materialize_runtime_bundle,
)
from .model_artifacts import (
    atomic_json,
    load_model_state,
    write_model_state,
)
from .null_trial_runner import _collapse, _fit_options
from .policy import EpisodePolicy, digest_bytes, digest_object
from .post36_tail import (
    TAIL_BUNDLE_MANIFEST_SCHEMA,
    TAIL_REALIZATION_SCHEMA,
    load_production_prefix,
)
from .procedure_lock import (
    ARCHIVE_SCHEMA_VERSION,
    CONTROL_INVENTORY_SCHEMA_VERSION,
    FITTED_NULL_FALSIFIER_ID,
    NOVELTY_AUDIT_FALSIFIER_ID,
    PRE_NULL_MANIFEST_SCHEMA_VERSION,
    RUNTIME_SOURCE_BINDING_SCHEMA_VERSION,
    _ledger_object_sha256,
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
MANIFEST_SCHEMA_VERSION = PRE_NULL_MANIFEST_SCHEMA_VERSION
GENERATOR_SELECTION_SCHEMA_VERSION = "ep12.fitted_null_generator_selection.v1"
SNAPSHOT_BINDING_SCHEMA_VERSION = "ep12.selected_fit_snapshot_binding.v1"
DEFAULT_POST36_PLAN = OUTPUTS_ROOT / "executor" / "OBSERVED_POST36_TAIL_PLAN.json"


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


def _verify_files(directory: Path, files: Mapping[str, Any], label: str) -> None:
    for relative, expected in files.items():
        child = Path(str(relative))
        _require(not child.is_absolute() and ".." not in child.parts,
                 f"unsafe path in {label}: {relative}")
        path = directory / child
        _require(path.is_file() and not path.is_symlink(),
                 f"missing or unsafe {label} file: {relative}")
        _require(_sha256_file(path) == str(expected),
                 f"{label} digest mismatch: {relative}")


def _initial_result_dir(run_root: Path, trial_id: str) -> Path:
    if trial_id.startswith("coverage_"):
        return run_root / "observed" / trial_id
    if trial_id.startswith("adaptive_c1_"):
        return run_root / "adaptive" / "cycle1" / "observed" / trial_id
    if trial_id.startswith("adaptive_c2_"):
        return run_root / "adaptive" / "cycle2" / "observed" / trial_id
    raise PreNullMaterializerError(f"unrecognized initial trial identity: {trial_id}")


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
    manifest_path = directory / "bundle_manifest.json"
    manifest = _read_json(manifest_path, f"tail bundle manifest {trial_id}")
    _require(manifest.get("schema_version") == TAIL_BUNDLE_MANIFEST_SCHEMA,
             f"tail bundle schema mismatch: {trial_id}")
    require_record_fields(manifest, contract_provenance, active=True,
                          label=f"tail bundle {trial_id}")
    _require(_sha256_file(manifest_path) == slot.get("durable_bundle_manifest_sha256"),
             f"tail bundle identity mismatch: {trial_id}")
    _require(manifest.get("trial_id") == trial_id
             and manifest.get("slot_sha256") == slot.get("slot_sha256")
             and manifest.get("configuration_sha256")
             == slot.get("configuration_sha256"),
             f"tail bundle identity fields mismatch: {trial_id}")
    files = _mapping(manifest.get("files"), f"tail bundle files {trial_id}")
    _verify_files(directory, files, f"tail bundle {trial_id}")
    realization = _read_json(directory / "realization.json", f"tail realization {trial_id}")
    _require(realization.get("schema_version") == TAIL_REALIZATION_SCHEMA,
             f"tail realization schema mismatch: {trial_id}")
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
    control = _read_json(directory / "control_receipt.json", f"control receipt {trial_id}")
    require_record_fields(control, contract_provenance, active=True,
                          label=f"tail control {trial_id}")
    _require(digest_object(control) == slot.get("control_receipt_sha256"),
             f"tail control receipt digest mismatch: {trial_id}")
    _require(realization.get("control_receipt") == control,
             f"tail realization embeds a different control receipt: {trial_id}")
    result_table = directory / "result" / trial_id / "result_table.csv"
    _require(result_table.relative_to(directory).as_posix() in files,
             f"tail bundle omits result table from manifest: {trial_id}")
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
    observed_attestation_dir: Path,
    tail_root: Path,
    policy: EpisodePolicy,
    contract_provenance: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
    dict[str, Any],
]:
    """Authenticate and merge the complete initial and realized-tail archive."""

    provenance = validate_contract_provenance(contract_provenance)
    prefix_fields = prefix_record_fields(provenance)
    active_fields = active_record_fields(provenance)
    run_root = _inside(run_root, OUTPUTS_ROOT, "run root")
    observed_finalization_dir = _inside(
        observed_finalization_dir, OUTPUTS_ROOT, "observed finalization"
    )
    observed_attestation_dir = _inside(
        observed_attestation_dir, OUTPUTS_ROOT, "observed attestation"
    )
    tail_root = _inside(tail_root, OUTPUTS_ROOT, "tail root")
    # This parses only the separate structural/status attestation, not outcomes.
    verify_attestation_root(
        run_root=run_root,
        attestation_dir=observed_attestation_dir,
        policy=policy,
        outputs_root=OUTPUTS_ROOT,
    )
    attestation_manifest_path = observed_attestation_dir / "manifest.json"
    observed_attestation_sha256 = _sha256_file(attestation_manifest_path)
    prefix, prefix_evidence = load_production_prefix(
        run_root=run_root,
        finalization_dir=observed_finalization_dir,
        policy=policy,
        outputs_root=OUTPUTS_ROOT,
        contract_provenance=provenance,
    )
    initial_ids = [str(record["trial_id"]) for record in prefix]
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
    schedule_manifest_path = tail_root / "realized_schedule_manifest.json"
    ledger_path = tail_root / "ledger.jsonl"
    schedule = _read_json(schedule_path, "realized schedule")
    schedule_manifest = _read_json(schedule_manifest_path, "realized schedule manifest")
    slots = _validate_schedule(
        schedule,
        schedule_manifest,
        policy=policy,
        schedule_file_sha256=_sha256_file(schedule_path),
        ledger_file_sha256=_sha256_file(ledger_path),
        ledger_object_sha256=_ledger_object_sha256(ledger_path),
        active_contract_fields=active_fields,
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
                str(slot["control_receipt_sha256"])
            )

    eligible = [record for record in records if record["selection_eligible"]]
    _require(eligible, "complete comparison archive has no eligible comparison")
    selected = sorted(eligible, key=_ranking_key)[0]
    _require(schedule.get("incumbent_trial_id") == selected["trial_id"]
             and schedule.get("incumbent_configuration_sha256")
             == selected["configuration_sha256"],
             "complete archive ranking disagrees with realized schedule incumbent")
    archive = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "policy_sha256": policy.policy_hash,
        "observed_attestation_sha256": observed_attestation_sha256,
        "initial_trial_ids": initial_ids,
        "records": records,
        "selected_trial_id": selected["trial_id"],
        "selected_configuration_sha256": selected["configuration_sha256"],
        "selected_source_contract_epoch": selected["contract_epoch"],
        "selected_source_contract_sha256": selected[
            "scientific_contract_sha256"
        ],
        "contract_partition": {
            "initial_prefix": {
                "contract_epoch": PREFIX_EPOCH,
                "scientific_contract_sha256": prefix_fields[
                    "scientific_contract_sha256"
                ],
                "trial_ids": initial_ids,
            },
            "post36_tail": {
                "contract_epoch": ACTIVE_EPOCH,
                "scientific_contract_sha256": active_fields[
                    "scientific_contract_sha256"
                ],
                "trial_ids": [str(slot["trial_id"]) for slot in slots],
            },
            "contract_transition_semantic_sha256": active_fields[
                "contract_transition_semantic_sha256"
            ],
            "contract_transition_file_sha256": active_fields[
                "contract_transition_file_sha256"
            ],
        },
        "source_evidence": prefix_evidence,
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
        "schema_version": CONTROL_INVENTORY_SCHEMA_VERSION,
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
        {
            "schedule": schedule,
            "schedule_manifest": schedule_manifest,
            "schedule_file_sha256": _sha256_file(schedule_path),
            "schedule_manifest_file_sha256": _sha256_file(
                schedule_manifest_path
            ),
            "ledger_file_sha256": _sha256_file(ledger_path),
            "observed_attestation_sha256": observed_attestation_sha256,
        },
    )


def build_null_controller_program(
    *,
    policy: EpisodePolicy,
    comparison_archive: Mapping[str, Any],
    post36_plan_path: Path,
) -> tuple[dict[str, Any], str]:
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
    return program, _sha256_file(post36_plan_path)


def translate_realized_controller_schedule(
    *,
    policy: EpisodePolicy,
    comparison_archive: Mapping[str, Any],
    post36_schedule: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate the authenticated prefix+tail into the exact null schedule.

    This is a schema translation, not a new search decision.  Promotion
    improvements are replayed with the frozen ranking and must agree with every
    post-36 disposition.  Trial identities/configurations are preserved byte-
    for-byte at the JSON-object level because the null callback uses them for
    deterministic seeds and artifact paths.
    """

    raise PreNullMaterializerError(
        "observed realized slots must never be replayed as a null schedule; "
        "use the frozen outcome-blind null-controller program"
    )
    records = {
        str(record["trial_id"]): dict(record)
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
    _require(len(initial_ids) == policy.minimum_valid_trials,
             "runtime schedule lacks the complete initial minimum")
    slots = [
        dict(value)
        for value in _mapping_sequence(
            post36_schedule.get("realized_slots"), "post-36 realized slots"
        )
    ]
    ordinals = [int(slot.get("valid_trial_ordinal", -1)) for slot in slots]
    _require(
        ordinals
        == list(
            range(
                policy.minimum_valid_trials + 1,
                policy.minimum_valid_trials + 1 + len(slots),
            )
        ),
        "post-36 realized slots are not a contiguous valid-trial suffix",
    )

    raw_non_trial = _plain_sequence(
        post36_schedule.get("non_trial_required_gates"),
        "non-trial required gates",
    )
    non_trial: list[str] = []
    for value in raw_non_trial:
        if isinstance(value, Mapping):
            identifier = str(value.get("falsifier_id", ""))
        else:
            identifier = str(value)
        _require(identifier and identifier not in non_trial,
                 "non-trial falsifier classification is empty or duplicated")
        non_trial.append(identifier)
    countable = [
        identifier
        for identifier in policy.required_falsifiers
        if identifier not in set(non_trial)
    ]
    _require(
        set(countable) | set(non_trial) == set(policy.required_falsifiers)
        and not (set(countable) & set(non_trial)),
        "tail falsifier classes do not partition SEARCH_POLICY",
    )

    trials: list[dict[str, Any]] = []
    incumbent: Mapping[str, Any] | None = None

    def append_trial(
        *,
        ordinal: int,
        record: Mapping[str, Any],
        slot_kind: str,
        falsifier_id: str | None,
        promotion_eligible: bool,
        declared_improvement: Any | None,
    ) -> None:
        nonlocal incumbent
        _require(str(record.get("trial_id", "")) in records,
                 "runtime schedule references an unknown comparison")
        improved = bool(
            promotion_eligible
            and (incumbent is None or _ranking_key(dict(record)) < _ranking_key(dict(incumbent)))
        )
        if declared_improvement is not None:
            _require(isinstance(declared_improvement, bool)
                     and declared_improvement is improved,
                     f"tail promotion disposition does not replay: {record['trial_id']}")
        if improved:
            incumbent = record
        trials.append(
            {
                "valid_trial_ordinal": ordinal,
                "trial_id": str(record["trial_id"]),
                "configuration": dict(_mapping(
                    record.get("configuration"),
                    f"runtime configuration {record['trial_id']}",
                )),
                "configuration_sha256": str(record["configuration_sha256"]),
                "slot_kind": slot_kind,
                "falsifier_id": falsifier_id,
                "promotion_eligible": promotion_eligible,
                "observed_promotion_eligible_improvement": improved,
                "counts_toward_post_coverage_denominator": (
                    ordinal > policy.coverage_minimum
                ),
                "counts_toward_post_coverage_falsifier_numerator": (
                    slot_kind == "mandatory_falsifier"
                ),
            }
        )

    for ordinal, trial_id in enumerate(initial_ids, start=1):
        _require(trial_id in records, f"initial comparison is absent: {trial_id}")
        append_trial(
            ordinal=ordinal,
            record=records[trial_id],
            slot_kind=(
                "coverage"
                if ordinal <= policy.coverage_minimum
                else "adaptive_challenger"
            ),
            falsifier_id=None,
            promotion_eligible=True,
            declared_improvement=None,
        )
    for slot in slots:
        trial_id = str(slot.get("trial_id", ""))
        _require(trial_id in records, f"tail comparison is absent: {trial_id}")
        kind = str(slot.get("slot_kind", ""))
        _require(kind in {"adaptive_challenger", "mandatory_falsifier"},
                 f"unsupported tail slot kind: {kind}")
        eligible = slot.get("promotion_eligible") is True
        falsifier = slot.get("registered_falsifier_id")
        _require(
            (kind == "mandatory_falsifier" and not eligible
             and isinstance(falsifier, str) and falsifier in countable)
            or (kind == "adaptive_challenger" and eligible and falsifier is None),
            f"tail slot classification cannot enter runtime schedule: {trial_id}",
        )
        append_trial(
            ordinal=int(slot["valid_trial_ordinal"]),
            record=records[trial_id],
            slot_kind=kind,
            falsifier_id=str(falsifier) if falsifier is not None else None,
            promotion_eligible=eligible,
            declared_improvement=slot.get("improved"),
        )

    _require(incumbent is not None, "runtime schedule has no promotion-eligible fit")
    _require(
        incumbent["trial_id"] == comparison_archive.get("selected_trial_id")
        == post36_schedule.get("incumbent_trial_id")
        and incumbent["configuration_sha256"]
        == comparison_archive.get("selected_configuration_sha256")
        == post36_schedule.get("incumbent_configuration_sha256"),
        "translated runtime incumbent differs from the authenticated winner",
    )
    schedule = freeze_realized_controller_schedule(
        policy=policy,
        trials=trials,
        stop_reason=str(post36_schedule.get("stop_reason", "")),
        countable_falsifier_ids=countable,
        non_trial_required_falsifier_ids=non_trial,
    )
    verify_realized_controller_schedule(schedule, policy=policy)
    return schedule


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
    contract_sha256: str,
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
    active_fields = active_record_fields(provenance)
    _require(contract_sha256 == active_fields["scientific_contract_sha256"],
             "selected refit must bind the active post36 contract")
    configuration = dict(_mapping(selected.get("configuration"), "selected configuration"))
    validate_development_configuration(policy, configuration)
    _require(digest_object(configuration) == selected.get("configuration_sha256"),
             "selected configuration digest mismatch")
    _require(
        selected.get("contract_epoch") in {PREFIX_EPOCH, ACTIVE_EPOCH}
        and selected.get("scientific_contract_sha256")
        in {
            provenance["prefix_contract"]["sha256"],
            provenance["active_contract"]["sha256"],
        },
        "selected comparison lacks a recognized source contract epoch",
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
        "schema_version": GENERATOR_SELECTION_SCHEMA_VERSION,
        "selected_trial_id": selected["trial_id"],
        "selected_configuration_sha256": selected["configuration_sha256"],
        "selected_result_table_sha256": _sha256_file(selected_result_table),
        "selected_source_contract_epoch": selected["contract_epoch"],
        "selected_source_contract_sha256": selected[
            "scientific_contract_sha256"
        ],
        "refit_contract_epoch": ACTIVE_EPOCH,
        "refit_contract_sha256": contract_sha256,
        "contract_transition_semantic_sha256": active_fields[
            "contract_transition_semantic_sha256"
        ],
        "contract_transition_file_sha256": active_fields[
            "contract_transition_file_sha256"
        ],
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
        contract_sha256=contract_sha256,
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
            "selected_source_contract_sha256": selected[
                "scientific_contract_sha256"
            ],
            "refit_contract_epoch": ACTIVE_EPOCH,
            "contract_transition_semantic_sha256": active_fields[
                "contract_transition_semantic_sha256"
            ],
            "contract_transition_file_sha256": active_fields[
                "contract_transition_file_sha256"
            ],
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
        material_manifest.get("schema_version")
        == "ep12.development_materialization_manifest.v1",
        "development materialization manifest schema mismatch",
    )
    _require(
        material_manifest.get("final_focal_connectivity_materialized_or_summarized")
        is False,
        "development materialization manifest records final-role access",
    )
    _verify_files(
        materialization_dir,
        _mapping(material_manifest.get("files"), "materialization files"),
        "development materialization",
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
    _require(
        preparation_manifest.get("schema_version")
        == "ep12.partner_type_hierarchy_manifest.v1",
        "partner hierarchy manifest schema mismatch",
    )
    _require(preparation_manifest.get("final_connectivity_accessed") is False,
             "partner hierarchy manifest records final-connectivity access")
    _verify_files(
        preparation_dir,
        _mapping(preparation_manifest.get("files"), "hierarchy files"),
        "partner hierarchy",
    )
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
        "schema_version": SNAPSHOT_BINDING_SCHEMA_VERSION,
        "materialization_manifest_sha256": _sha256_file(material_manifest_path),
        "hierarchy_manifest_sha256": _sha256_file(preparation_manifest_path),
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


def _inventory(directory: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise PreNullMaterializerError(f"pre-null bundle contains symlink: {path}")
        if path.is_file() and path != directory / "manifest.json":
            result[path.relative_to(directory).as_posix()] = _sha256_file(path)
    return result


def verify_pre_null_inputs(
    output_dir: Path, *, policy_path: Path
) -> dict[str, Any]:
    """Authenticate an existing materialized input bundle for retry recovery."""

    directory = _inside(output_dir, OUTPUTS_ROOT, "pre-null input bundle")
    manifest = _read_json(directory / "manifest.json", "pre-null input manifest")
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION,
             "pre-null input manifest schema mismatch")
    core = {key: value for key, value in manifest.items() if key != "bundle_sha256"}
    _require(manifest.get("bundle_sha256") == digest_object(core),
             "pre-null input manifest self-hash mismatch")
    _require(manifest.get("files") == _inventory(directory),
             "pre-null input bundle file inventory changed")
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
    _require(
        source_binding.get("schema_version")
        == RUNTIME_SOURCE_BINDING_SCHEMA_VERSION,
        "runtime source-binding schema mismatch",
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
    _require(archive.get("schema_version") == ARCHIVE_SCHEMA_VERSION
             and archive.get("complete") is True
             and archive.get("final_connectivity_accessed") is False,
             "comparison archive is incomplete or unsafe")
    _require(controls.get("schema_version") == CONTROL_INVENTORY_SCHEMA_VERSION
             and controls.get("complete_required_falsifier_inventory") is True
             and controls.get("final_connectivity_accessed") is False,
             "control inventory is incomplete or unsafe")
    _require(selection.get("schema_version") == GENERATOR_SELECTION_SCHEMA_VERSION
             and selection.get("complete") is True
             and selection.get("development_only") is True
             and selection.get("final_connectivity_accessed") is False,
             "generator selection is incomplete or unsafe")
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
        and active_partition.get("contract_epoch") == ACTIVE_EPOCH,
        "comparison archive contract epochs are ambiguous",
    )
    for artifact, label in (
        (manifest, "pre-null manifest"),
        (source_binding, "runtime source binding"),
    ):
        _require(
            artifact.get("prefix_contract_sha256")
            == prefix_partition.get("scientific_contract_sha256")
            and artifact.get("active_contract_sha256")
            == active_partition.get("scientific_contract_sha256")
            and artifact.get("contract_transition_semantic_sha256")
            == partition.get("contract_transition_semantic_sha256")
            and artifact.get("contract_transition_file_sha256")
            == partition.get("contract_transition_file_sha256")
            and artifact.get("selected_source_contract_epoch")
            == archive.get("selected_source_contract_epoch")
            and artifact.get("selected_source_contract_sha256")
            == archive.get("selected_source_contract_sha256")
            and artifact.get("refit_contract_epoch") == ACTIVE_EPOCH,
            f"{label} contract provenance differs from the mixed archive",
        )
    _require(
        mode.get("contract_sha256") == manifest.get("active_contract_sha256")
        and mode.get("selected_source_contract_epoch")
        == archive.get("selected_source_contract_epoch")
        and mode.get("selected_source_contract_sha256")
        == archive.get("selected_source_contract_sha256")
        and mode.get("refit_contract_epoch") == ACTIVE_EPOCH,
        "mode-support result does not bind active-v2 refit provenance",
    )
    _require(
        selection.get("selected_source_contract_epoch")
        == archive.get("selected_source_contract_epoch")
        and selection.get("selected_source_contract_sha256")
        == archive.get("selected_source_contract_sha256")
        and selection.get("refit_contract_epoch") == ACTIVE_EPOCH
        and selection.get("refit_contract_sha256")
        == manifest.get("active_contract_sha256"),
        "fitted-null selection does not bind active-v2 refit provenance",
    )
    _require(
        manifest.get("null_controller_program_sha256")
        == controller_program.get("program_sha256")
        == source_binding.get("null_controller_program_sha256")
        and manifest.get("null_controller_program_file_sha256")
        == _sha256_file(directory / "null_controller_program.json"),
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
    observed_attestation_dir: Path,
    tail_root: Path,
    snapshot_dir: Path,
    materialization_dir: Path,
    preparation_dir: Path,
    policy_path: Path,
    contract_path: Path,
    prefix_contract_path: Path,
    contract_transition_receipt_path: Path,
    contract_transition_spec_path: Path,
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
        transition_receipt_path=contract_transition_receipt_path,
        transition_spec_path=contract_transition_spec_path,
    )
    active_fields = active_record_fields(contract_provenance)
    contract_snapshot = capture_yaml_contract(contract_path)
    contract = contract_snapshot.document
    contract_sha256 = contract_snapshot.sha256
    _require(contract_sha256 == active_fields["scientific_contract_sha256"],
             "active contract path differs from transition provenance")
    _require(contract.get("real_connectivity_access_authorized") == "development_only"
             and contract.get("final_connectivity_access_authorized", False) is False,
             "scientific contract does not preserve the role firewall")
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
    archive, controls, selected, result_table, schedule_evidence = (
        build_archive_and_controls(
            run_root=run_root,
            observed_finalization_dir=observed_finalization_dir,
            observed_attestation_dir=observed_attestation_dir,
            tail_root=tail_root,
            policy=policy,
            contract_provenance=contract_provenance,
        )
    )
    controller_program, post36_plan_file_sha256 = build_null_controller_program(
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
        "source_post36_plan_file_sha256": post36_plan_file_sha256,
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
        contract_sha256=contract_sha256,
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
            "schema_version": RUNTIME_SOURCE_BINDING_SCHEMA_VERSION,
            "policy_sha256": policy.policy_hash,
            "prefix_contract_sha256": contract_provenance["prefix_contract"]["sha256"],
            "active_contract_sha256": contract_sha256,
            "contract_transition_semantic_sha256": active_fields[
                "contract_transition_semantic_sha256"
            ],
            "contract_transition_file_sha256": active_fields[
                "contract_transition_file_sha256"
            ],
            "selected_source_contract_epoch": selected["contract_epoch"],
            "selected_source_contract_sha256": selected[
                "scientific_contract_sha256"
            ],
            "refit_contract_epoch": ACTIVE_EPOCH,
            "comparison_archive_sha256": digest_object(archive),
            "null_generator_selection_sha256": digest_object(
                generator_selection
            ),
            "null_controller_program_sha256": controller_program[
                "program_sha256"
            ],
            "post36_plan_file_sha256": post36_plan_file_sha256,
            "callback_data_sha256": callback_metadata["callback_data_sha256"],
            "post36_schedule_file_sha256": schedule_evidence[
                "schedule_file_sha256"
            ],
            "post36_schedule_manifest_file_sha256": schedule_evidence[
                "schedule_manifest_file_sha256"
            ],
            "post36_ledger_file_sha256": schedule_evidence[
                "ledger_file_sha256"
            ],
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
            "prefix_contract_sha256": contract_provenance["prefix_contract"]["sha256"],
            "active_contract_sha256": contract_sha256,
            "contract_transition_semantic_sha256": active_fields[
                "contract_transition_semantic_sha256"
            ],
            "contract_transition_file_sha256": active_fields[
                "contract_transition_file_sha256"
            ],
            "selected_source_contract_epoch": selected["contract_epoch"],
            "selected_source_contract_sha256": selected[
                "scientific_contract_sha256"
            ],
            "refit_contract_epoch": ACTIVE_EPOCH,
            "prevalence_parameter_rank_definition": rank_rule,
            "selected_trial_id": selected["trial_id"],
            "selected_configuration_sha256": selected["configuration_sha256"],
            "realized_schedule_file_sha256": schedule_evidence[
                "schedule_file_sha256"
            ],
            "realized_ledger_file_sha256": schedule_evidence["ledger_file_sha256"],
        }
        snapshot_record_core.pop("snapshot_binding_sha256", None)
        snapshot_record = dict(snapshot_record_core)
        snapshot_record["snapshot_binding_sha256"] = digest_object(
            snapshot_record_core
        )
        atomic_json(stage / "snapshot_binding.json", snapshot_record)
        core = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "policy_sha256": policy.policy_hash,
            "prefix_contract_sha256": contract_provenance["prefix_contract"]["sha256"],
            "active_contract_sha256": contract_sha256,
            "contract_transition_semantic_sha256": active_fields[
                "contract_transition_semantic_sha256"
            ],
            "contract_transition_file_sha256": active_fields[
                "contract_transition_file_sha256"
            ],
            "selected_source_contract_epoch": selected["contract_epoch"],
            "selected_source_contract_sha256": selected[
                "scientific_contract_sha256"
            ],
            "refit_contract_epoch": ACTIVE_EPOCH,
            "selected_trial_id": selected["trial_id"],
            "selected_configuration_sha256": selected["configuration_sha256"],
            "comparison_archive_sha256": digest_object(archive),
            "mode_support_inference_sha256": mode["inference_sha256"],
            "control_inventory_sha256": digest_object(controls),
            "generator_selection_sha256": digest_object(generator_selection),
            "null_controller_program_sha256": controller_program[
                "program_sha256"
            ],
            "null_controller_program_file_sha256": _sha256_file(
                stage / "null_controller_program.json"
            ),
            "post36_plan_file_sha256": post36_plan_file_sha256,
            "callback_data_sha256": callback_metadata["callback_data_sha256"],
            "model_state_sha256": model_manifest["state_sha256"],
            "snapshot_binding_sha256": snapshot_record[
                "snapshot_binding_sha256"
            ],
            "runtime_source_binding_sha256": source_binding[
                "source_binding_sha256"
            ],
            "files": _inventory(stage),
            "development_only": True,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        }
        contract_snapshot.assert_unchanged()
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
    parser.add_argument("--observed-attestation-dir", type=Path, required=True)
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
    parser.add_argument("--contract-transition-receipt", type=Path, required=True)
    parser.add_argument("--contract-transition-spec", type=Path, required=True)
    parser.add_argument("--post36-plan", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = materialize_pre_null_inputs(
        output_dir=args.output_dir,
        run_root=args.run_root,
        observed_finalization_dir=args.observed_finalization_dir,
        observed_attestation_dir=args.observed_attestation_dir,
        tail_root=args.tail_root,
        snapshot_dir=args.snapshot_dir,
        materialization_dir=args.materialization_dir,
        preparation_dir=args.preparation_dir,
        policy_path=args.policy,
        contract_path=args.contract,
        prefix_contract_path=args.prefix_contract,
        contract_transition_receipt_path=args.contract_transition_receipt,
        contract_transition_spec_path=args.contract_transition_spec,
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
    "GENERATOR_SELECTION_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "PreNullMaterializerError",
    "build_archive_and_controls",
    "fit_selected_mode_and_generators",
    "load_authenticated_development_snapshot",
    "materialize_pre_null_inputs",
    "verify_pre_null_inputs",
]
