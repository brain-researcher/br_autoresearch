"""Outcome-blind verifier and receipt scaffolding for EP12's post-36 contract.

This module does not discover observed runs and does not read connectivity
outcomes.  A caller must provide the separately authenticated, status-only
36-trial attestation plus a narrow declaration for each authenticated trial
report.  The only permitted scientific-contract delta resolves which fitted
parameter dimension is used by the already-frozen prevalence statement.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contract_snapshot import (
    V1_CONTRACT_SHA256,
    V1_QUALIFICATION_SPEC_SHA256,
    YamlContractSnapshot,
    assert_immutable_v1_contract,
    assert_immutable_v1_qualification_spec,
    capture_yaml_contract,
)


TRANSITION_SPEC_SCHEMA = "ep12.scientific_contract_transition_spec.v2"
TRANSITION_RECEIPT_SCHEMA = "ep12.scientific_contract_transition_receipt.v2"
TRANSITION_ID = "ep12_post36_representation_rank_v1"
TRANSITION_SPEC_SHA256 = (
    "22428039fecf59628d68b7d7a3d52307e7f42dce02808c8011697bcd859380ec"
)
ATTESTATION_SCHEMA = "ep12.observed_trial_attestation_set.v1"
INVARIANCE_SCHEMA = "ep12.contract_transition_invariance_comparison.v1"
V1_SCIENTIFIC_MODELS_SHA256 = (
    "55218de7b35e78e7ad5b39ab38295eef84200d753d54a0c5b5e051aec15625f8"
)
V1_PRE_NULL_RANK_EVALUATOR_SHA256 = (
    "480e359e2f8e52e86ded8979fa55825a077b96ad3faf9befc1f6be9c1c35569d"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DECLARATION_FIELDS = {
    "trial_id",
    "source_report_schema",
    "source_report_status",
    "source_trial_report_sha256",
    "declared_scientific_contract_sha256",
    "data_role",
    "final_connectivity_accessed",
}
_TRANSITION_RECEIPT_FIELDS = {
    "schema_version",
    "transition_id",
    "transition_spec_sha256",
    "old_contract_sha256",
    "active_contract_sha256",
    "old_qualification_spec_sha256",
    "active_qualification_spec_sha256",
    "allowed_contract_changes_sha256",
    "allowed_qualification_spec_changes_sha256",
    "old_scientific_models_sha256",
    "active_scientific_models_sha256",
    "old_pre_null_rank_evaluator_sha256",
    "active_pre_null_rank_evaluator_sha256",
    "observed_attestation_manifest_sha256",
    "observed_attestation_sha256",
    "observed_attestation_authority",
    "observed_trial_count",
    "observed_report_contract_declarations_sha256",
    "every_observed_report_declares_old_contract",
    "synthetic_invariance_evidence_sha256",
    "rank_definition",
    "operator_exposure_incident_relative_path",
    "operator_exposure_incident_file_sha256",
    "operator_exposure_incident_semantic_sha256",
    "operator_development_outcome_values_displayed",
    "programmatic_transition_extraction_outcome_fields_accessed",
    "values_used_for_transition_decisions",
    "final_connectivity_accessed",
    "complete",
    "transition_receipt_sha256",
}
_ATTESTATION_AUTHORITY_FIELDS = {
    "schema_version",
    "authoritative_job_id",
    "authoritative_job_name",
    "replacement_terminal_identity_path",
    "replacement_terminal_identity_file_sha256",
    "replacement_terminal_identity_sha256",
    "replacement_launch_receipt_file_sha256",
    "replacement_launch_receipt_sha256",
    "replacement_release_receipt_file_sha256",
    "replacement_release_receipt_sha256",
    "post36_orchestration_intent_path",
    "post36_orchestration_intent_file_sha256",
    "post36_orchestration_intent_sha256",
    "post36_start_authorization_path",
    "post36_start_authorization_file_sha256",
    "post36_start_authorization_sha256",
    "obsolete_cancellation_receipt_path",
    "obsolete_cancellation_receipt_file_sha256",
    "obsolete_cancellation_receipt_sha256",
    "obsolete_job_id",
    "obsolete_job_name",
    "obsolete_terminal_scheduler_state",
    "obsolete_identity_file_sha256",
    "obsolete_identity_sha256",
    "obsolete_result_authoritative",
    "authoritative_result",
    "source_archive",
    "controller_final_receipt",
    "real_connectivity_accessed",
    "final_connectivity_accessed",
    "final_connectivity_access_authorized",
    "authority_sha256",
}
_SOURCE_ARCHIVE_FIELDS = {
    "archived_root",
    "archive_parent",
    "archive_receipt_path",
    "archive_receipt_file_sha256",
    "archive_receipt_sha256",
    "manifest_path",
    "manifest_file_sha256",
    "manifest_sha256",
    "tree_sha256",
    "surface_allowlist",
    "inputs_included",
    "development_outputs_included",
    "prelaunch_outputs_included",
    "bootstrap_sha256",
    "parent_first_publication_bound",
    "complete",
}
_CONTROLLER_FINAL_RECEIPT_FIELDS = {
    "status",
    "path",
    "file_sha256",
    "receipt_sha256",
    "required_before_irreversible_transition",
}


class ScientificContractTransitionError(ValueError):
    """The proposed post-36 transition lacks an exact authenticated binding."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScientificContractTransitionError(
            "transition evidence is not canonical-JSON serializable"
        ) from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except FileNotFoundError as exc:
        raise ScientificContractTransitionError(
            f"missing transition implementation file: {path}"
        ) from exc
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScientificContractTransitionError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScientificContractTransitionError(f"{label} must be a mapping")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ScientificContractTransitionError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _safe_relative_file(root: Path, relative: Any, label: str) -> Path:
    _require(isinstance(relative, str) and relative, f"{label} path must be text")
    candidate = Path(relative)
    _require(
        not candidate.is_absolute()
        and candidate.parts
        and all(part not in {"", ".", ".."} for part in candidate.parts),
        f"unsafe relative {label} path: {relative!r}",
    )
    return root / candidate


def _load_transition_spec(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ScientificContractTransitionError(
            f"missing transition specification: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ScientificContractTransitionError(
            f"malformed transition specification: {path}"
        ) from exc
    spec = dict(_mapping(payload, "transition specification"))
    claimed = _sha256(
        spec.get("transition_spec_sha256"), "transition_spec_sha256"
    )
    core = {key: value for key, value in spec.items() if key != "transition_spec_sha256"}
    calculated = _digest(core)
    _require(claimed == calculated, "transition specification self-hash mismatch")
    _require(
        claimed == TRANSITION_SPEC_SHA256,
        "transition specification is not the executor-pinned definition",
    )
    _require(
        spec.get("schema_version") == TRANSITION_SPEC_SCHEMA
        and spec.get("transition_id") == TRANSITION_ID
        and spec.get("episode_id") == "ep12",
        "transition specification identity mismatch",
    )
    return spec


def _document_changes(
    old: Mapping[str, Any], new: Mapping[str, Any], prefix: str = ""
) -> list[dict[str, Any]]:
    """Return deterministic JSON-pointer changes, treating added maps atomically."""

    changes: list[dict[str, Any]] = []
    for key in sorted(set(old) | set(new)):
        path = f"{prefix}/{key}"
        if key not in old:
            changes.append({"operation": "add", "path": path, "new": new[key]})
        elif key not in new:
            changes.append({"operation": "remove", "path": path, "old": old[key]})
        elif isinstance(old[key], Mapping) and isinstance(new[key], Mapping):
            changes.extend(_document_changes(old[key], new[key], path))
        elif old[key] != new[key]:
            changes.append(
                {
                    "operation": "replace",
                    "path": path,
                    "old": old[key],
                    "new": new[key],
                }
            )
    return changes


def _declared_changes(spec: Mapping[str, Any], field: str) -> list[dict[str, Any]]:
    raw = spec.get(field)
    _require(isinstance(raw, list) and raw, f"{field} must be a nonempty list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        record = dict(_mapping(item, f"{field}[{index}]"))
        classification = record.pop("classification", None)
        _require(
            classification in {"lineage_metadata", "scientific_definition_resolution"},
            f"{field}[{index}] has an invalid classification",
        )
        path = record.get("path")
        _require(isinstance(path, str) and path.startswith("/"), f"{field}[{index}] has an invalid path")
        _require(path not in seen, f"{field} repeats path {path}")
        seen.add(path)
        normalized.append(record)
    return sorted(normalized, key=lambda value: str(value["path"]))


def _verify_exact_changes(
    *,
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    declared: list[dict[str, Any]],
    label: str,
) -> None:
    observed = sorted(_document_changes(old, new), key=lambda value: str(value["path"]))
    _require(observed == declared, f"{label} contains a change outside the transition allowlist")


def _definition_snapshots(
    executor_root: Path,
    transition_spec_path: Path | None,
) -> tuple[
    dict[str, Any],
    YamlContractSnapshot,
    YamlContractSnapshot,
    YamlContractSnapshot,
    YamlContractSnapshot,
]:
    root = executor_root.resolve()
    spec_path = transition_spec_path or (root / "CONTRACT_TRANSITION_SPEC.json")
    spec = _load_transition_spec(spec_path)
    old_contract_spec = _mapping(spec["old_contract"], "old contract specification")
    new_contract_spec = _mapping(spec["new_contract"], "new contract specification")
    old_qualification_spec = _mapping(
        spec["old_qualification_spec"], "old qualification specification"
    )
    new_qualification_spec = _mapping(
        spec["new_qualification_spec"], "new qualification specification"
    )
    _require(
        old_contract_spec.get("sha256") == V1_CONTRACT_SHA256
        and old_qualification_spec.get("sha256") == V1_QUALIFICATION_SPEC_SHA256,
        "transition specification does not bind the immutable v1 files",
    )
    old_contract = assert_immutable_v1_contract(
        _safe_relative_file(root, old_contract_spec.get("path"), "old contract")
    )
    old_qualification = assert_immutable_v1_qualification_spec(
        _safe_relative_file(
            root, old_qualification_spec.get("path"), "old qualification spec"
        )
    )
    new_contract = capture_yaml_contract(
        _safe_relative_file(root, new_contract_spec.get("path"), "new contract"),
        expected_sha256=_sha256(new_contract_spec.get("sha256"), "new contract sha256"),
    )
    new_qualification = capture_yaml_contract(
        _safe_relative_file(
            root, new_qualification_spec.get("path"), "new qualification spec"
        ),
        expected_sha256=_sha256(
            new_qualification_spec.get("sha256"), "new qualification spec sha256"
        ),
    )
    _require(
        new_contract.document.get("schema_version")
        == new_contract_spec.get("schema_version")
        and new_contract.document.get("contract_id")
        == new_contract_spec.get("contract_id"),
        "post-36 contract identity differs from transition specification",
    )
    _require(
        new_qualification.document.get("schema_version")
        == new_qualification_spec.get("schema_version"),
        "post-36 qualification-spec identity differs from transition specification",
    )
    _verify_exact_changes(
        old=old_contract.document,
        new=new_contract.document,
        declared=_declared_changes(spec, "allowed_contract_changes"),
        label="post-36 scientific contract",
    )
    _verify_exact_changes(
        old=old_qualification.document,
        new=new_qualification.document,
        declared=_declared_changes(spec, "allowed_qualification_spec_changes"),
        label="post-36 qualification specification",
    )
    scientific_changes = [
        item
        for item in spec["allowed_contract_changes"]
        if item.get("classification") == "scientific_definition_resolution"
    ]
    _require(
        scientific_changes
        == [
            {
                "classification": "scientific_definition_resolution",
                "operation": "add",
                "path": "/mode_support/prevalence_parameter_rank_definition",
                "new": "representation_latent_effective_rank",
            }
        ],
        "transition must contain exactly one scientific definition resolution",
    )
    implementation = _mapping(spec["implementation_lineage"], "implementation lineage")
    model_lineage = _mapping(implementation["scientific_models"], "model lineage")
    rank_lineage = _mapping(
        implementation["pre_null_rank_evaluator"], "rank-evaluator lineage"
    )
    _require(
        model_lineage.get("old_sha256") == V1_SCIENTIFIC_MODELS_SHA256
        and rank_lineage.get("old_sha256") == V1_PRE_NULL_RANK_EVALUATOR_SHA256,
        "implementation lineage does not bind the observed-phase source",
    )
    for lineage, label in (
        (model_lineage, "scientific models"),
        (rank_lineage, "pre-null rank evaluator"),
    ):
        path = _safe_relative_file(root, lineage.get("path"), label)
        _require(
            _file_digest(path) == _sha256(lineage.get("new_sha256"), f"{label} sha256"),
            f"post-36 {label} source digest mismatch",
        )
    return spec, old_contract, new_contract, old_qualification, new_qualification


def verify_contract_transition_definition(
    *, executor_root: Path, transition_spec_path: Path | None = None
) -> dict[str, Any]:
    """Verify immutable v1, exact v2 deltas, and implementation lineage."""

    spec, old_contract, new_contract, old_qualification, new_qualification = (
        _definition_snapshots(executor_root, transition_spec_path)
    )
    for snapshot in (
        old_contract,
        new_contract,
        old_qualification,
        new_qualification,
    ):
        snapshot.assert_unchanged()
    return {
        "transition_id": TRANSITION_ID,
        "transition_spec_sha256": spec["transition_spec_sha256"],
        "old_contract_sha256": old_contract.sha256,
        "new_contract_sha256": new_contract.sha256,
        "old_qualification_spec_sha256": old_qualification.sha256,
        "new_qualification_spec_sha256": new_qualification.sha256,
        "allowed_contract_changes_sha256": _digest(spec["allowed_contract_changes"]),
        "allowed_qualification_spec_changes_sha256": _digest(
            spec["allowed_qualification_spec_changes"]
        ),
        "final_connectivity_accessed": False,
    }


def _verify_attestation(attestation: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    payload = dict(_mapping(attestation, "observed attestation"))
    claimed = _sha256(payload.get("attestation_sha256"), "attestation_sha256")
    core = {key: value for key, value in payload.items() if key != "attestation_sha256"}
    _require(_digest(core) == claimed, "observed attestation self-hash mismatch")
    _require(
        payload.get("schema_version") == ATTESTATION_SCHEMA
        and payload.get("episode_id") == "ep12"
        and payload.get("trial_count") == 36,
        "observed attestation identity or trial count mismatch",
    )
    _require(
        payload.get("stage_trial_counts")
        == {"coverage": 18, "adaptive_cycle_1": 9, "adaptive_cycle_2": 9},
        "observed attestation stage counts mismatch",
    )
    for field in (
        "objective_values_read",
        "objective_values_compared",
        "objective_values_emitted",
        "final_connectivity_accessed",
        "final_connectivity_access_authorized",
    ):
        _require(payload.get(field) is False, f"observed attestation violates {field}")
    _require(payload.get("development_only") is True, "observed attestation is not development-only")
    rows = payload.get("trials")
    _require(isinstance(rows, list) and len(rows) == 36, "observed attestation must have 36 trial rows")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"observed attestation trial {index}")
        trial_id = row.get("trial_id")
        _require(isinstance(trial_id, str) and trial_id and trial_id not in by_id, "observed attestation repeats or omits a trial ID")
        files = _mapping(row.get("source_files_sha256"), f"source files for {trial_id}")
        _sha256(files.get("trial_report.json"), f"trial report sha256 for {trial_id}")
        _require(
            row.get("source_report_schema") == "ep12.development_trial.v1"
            and row.get("source_report_status") == "completed"
            and row.get("data_role") == "development_only"
            and row.get("final_connectivity_accessed") is False,
            f"trial attestation status mismatch for {trial_id}",
        )
        by_id[trial_id] = row
    return by_id


def _verify_report_declarations(
    declarations: Sequence[Mapping[str, Any]],
    attested_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _require(
        not isinstance(declarations, (str, bytes)) and len(declarations) == 36,
        "contract declarations must contain exactly 36 rows",
    )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(declarations):
        row = dict(_mapping(raw, f"contract declaration {index}"))
        _require(set(row) == _DECLARATION_FIELDS, "contract declaration exposes fields outside the outcome-blind allowlist")
        trial_id = row.get("trial_id")
        _require(isinstance(trial_id, str) and trial_id in attested_by_id and trial_id not in seen, "contract declaration trial identity mismatch")
        seen.add(trial_id)
        attested = attested_by_id[trial_id]
        source_files = _mapping(attested["source_files_sha256"], "attested source files")
        _require(
            row.get("source_trial_report_sha256") == source_files["trial_report.json"],
            f"contract declaration is not bound to the authenticated report for {trial_id}",
        )
        _require(
            row.get("declared_scientific_contract_sha256") == V1_CONTRACT_SHA256,
            f"observed report does not declare the immutable v1 contract for {trial_id}",
        )
        _require(
            row.get("source_report_schema") == attested.get("source_report_schema")
            and row.get("source_report_status") == attested.get("source_report_status")
            and row.get("data_role") == "development_only"
            and row.get("final_connectivity_accessed") is False,
            f"contract declaration status mismatch for {trial_id}",
        )
        normalized.append(row)
    _require(seen == set(attested_by_id), "contract declarations do not cover the attested trial set")
    return sorted(normalized, key=lambda value: str(value["trial_id"]))


def extract_authenticated_report_contract_declarations(
    *,
    attested_by_id: Mapping[str, Mapping[str, Any]],
    authenticated_trial_report_bytes: Mapping[str, bytes],
) -> list[dict[str, Any]]:
    """Extract only contract/status fields from the 36 hash-bound reports.

    The complete report bytes are authenticated before JSON decoding.  Score,
    objective, aggregate, interval, and selection fields are never accessed or
    copied into the declaration returned by this function.
    """

    reports = _mapping(
        authenticated_trial_report_bytes, "authenticated trial report bytes"
    )
    _require(
        set(reports) == set(attested_by_id),
        "authenticated trial report bytes do not exactly cover the attested trials",
    )
    declarations: list[dict[str, Any]] = []
    for trial_id in sorted(attested_by_id):
        raw = reports[trial_id]
        _require(
            isinstance(raw, bytes),
            f"authenticated trial report must be bytes for {trial_id}",
        )
        attested = attested_by_id[trial_id]
        source_files = _mapping(
            attested["source_files_sha256"], "attested source files"
        )
        report_sha256 = hashlib.sha256(raw).hexdigest()
        _require(
            report_sha256 == source_files["trial_report.json"],
            f"trial report bytes do not match the authenticated hash for {trial_id}",
        )
        try:
            report = _mapping(
                json.loads(raw.decode("utf-8")), f"trial report {trial_id}"
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ScientificContractTransitionError(
                f"authenticated trial report is malformed for {trial_id}"
            ) from exc
        _require(
            report.get("trial_id") == trial_id,
            f"authenticated trial report identity mismatch for {trial_id}",
        )
        declarations.append(
            {
                "trial_id": trial_id,
                "source_report_schema": report.get("schema_version"),
                "source_report_status": report.get("status"),
                "source_trial_report_sha256": report_sha256,
                "declared_scientific_contract_sha256": report.get(
                    "scientific_contract_sha256"
                ),
                "data_role": report.get("data_role"),
                "final_connectivity_accessed": report.get(
                    "final_connectivity_accessed"
                ),
            }
        )
    return _verify_report_declarations(declarations, attested_by_id)


def verify_synthetic_invariance_evidence(
    evidence: Mapping[str, Any], spec: Mapping[str, Any]
) -> str:
    payload = dict(_mapping(evidence, "synthetic invariance evidence"))
    claimed = _sha256(
        payload.get("artifact_sha256"),
        "synthetic invariance comparison artifact sha256",
    )
    core = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    _require(_digest(core) == claimed, "synthetic invariance evidence self-hash mismatch")
    required = _mapping(
        spec["required_synthetic_invariance_evidence"],
        "synthetic invariance requirements",
    )
    _require(
        payload.get("schema_version") == required.get("schema_version")
        == INVARIANCE_SCHEMA,
        "synthetic invariance schema mismatch",
    )
    _require(
        payload.get("status") == "pass",
        "synthetic invariance comparison did not pass",
    )
    _require(
        payload.get("outcome_firewall")
        == {
            "generated_examples_only": True,
            "real_connectivity_accessed": False,
            "connectivity_paths_accepted": False,
            "annotation_or_outcome_files_accessed": False,
        },
        "synthetic invariance comparison violates the outcome firewall",
    )
    _require(
        payload.get("spec_sha256") == required.get("probe_spec_sha256"),
        "synthetic invariance comparison uses another probe specification",
    )
    legacy = _mapping(payload.get("legacy"), "legacy probe reference")
    post36 = _mapping(payload.get("post36"), "post36 probe reference")
    old_contract = _mapping(spec.get("old_contract"), "old contract specification")
    new_contract = _mapping(spec.get("new_contract"), "new contract specification")
    _require(
        legacy.get("contract_sha256") == old_contract.get("sha256")
        and post36.get("contract_sha256") == new_contract.get("sha256"),
        "synthetic comparison does not bind the old and active contracts",
    )
    checks = _mapping(payload.get("checks"), "synthetic comparison checks")
    for name in (
        "input_payload_byte_identical",
        "environment_byte_identical",
        "scientific_payload_byte_identical",
        "diagnostic_changes_limited_to_added_representation_rank",
    ):
        _require(checks.get(name) is True, f"synthetic comparison check failed: {name}")
    additions = checks.get("diagnostic_additions")
    _require(
        isinstance(additions, list)
        and additions == required.get("allowed_diagnostic_additions"),
        "synthetic comparison diagnostic additions differ from the frozen set",
    )
    return claimed


def verify_observed_attestation_authority(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    authority = dict(_mapping(value, "observed attestation authority"))
    _require(
        set(authority) == _ATTESTATION_AUTHORITY_FIELDS
        and authority.get("schema_version")
        == "ep12.dynamic_observed_attestation_authority.v1",
        "observed attestation authority schema changed",
    )
    claimed = _sha256(
        authority.get("authority_sha256"),
        "observed attestation authority sha256",
    )
    core = {
        key: item for key, item in authority.items() if key != "authority_sha256"
    }
    _require(
        _digest(core) == claimed,
        "observed attestation authority self-hash mismatch",
    )
    for field in (
        "replacement_terminal_identity_file_sha256",
        "replacement_terminal_identity_sha256",
        "replacement_launch_receipt_file_sha256",
        "replacement_launch_receipt_sha256",
        "replacement_release_receipt_file_sha256",
        "replacement_release_receipt_sha256",
        "post36_orchestration_intent_file_sha256",
        "post36_orchestration_intent_sha256",
        "post36_start_authorization_file_sha256",
        "post36_start_authorization_sha256",
        "obsolete_cancellation_receipt_file_sha256",
        "obsolete_cancellation_receipt_sha256",
        "obsolete_identity_file_sha256",
        "obsolete_identity_sha256",
    ):
        _sha256(authority.get(field), field)
    archive = _mapping(authority.get("source_archive"), "source archive identity")
    _require(
        set(archive) == _SOURCE_ARCHIVE_FIELDS
        and archive.get("complete") is True
        and archive.get("inputs_included") is False
        and archive.get("development_outputs_included") is False
        and archive.get("prelaunch_outputs_included") is False
        and archive.get("parent_first_publication_bound") is True
        and isinstance(archive.get("surface_allowlist"), list)
        and bool(archive["surface_allowlist"]),
        "source archive identity or safety fields changed",
    )
    for field in (
        "archive_receipt_file_sha256",
        "archive_receipt_sha256",
        "manifest_file_sha256",
        "manifest_sha256",
        "tree_sha256",
        "bootstrap_sha256",
    ):
        _sha256(archive.get(field), f"source archive {field}")
    final = _mapping(
        authority.get("controller_final_receipt"),
        "controller final-receipt binding",
    )
    _require(
        set(final) == _CONTROLLER_FINAL_RECEIPT_FIELDS
        and final.get("status")
        in {"absent_at_publication", "present_verified"}
        and isinstance(final.get("path"), str)
        and Path(str(final["path"])).is_absolute()
        and final.get("required_before_irreversible_transition") is True,
        "controller final-receipt binding changed",
    )
    if final["status"] == "present_verified":
        _sha256(final.get("file_sha256"), "controller receipt file sha256")
        _sha256(final.get("receipt_sha256"), "controller receipt semantic sha256")
    else:
        _require(
            final.get("file_sha256") is None and final.get("receipt_sha256") is None,
            "absent controller receipt unexpectedly has hashes",
        )
    _require(
        str(authority.get("authoritative_job_id", "")).isdigit()
        and int(str(authority["authoritative_job_id"])) > 0
        and isinstance(authority.get("authoritative_job_name"), str)
        and bool(authority["authoritative_job_name"])
        and authority.get("obsolete_job_id") == "45332084"
        and isinstance(authority.get("obsolete_job_name"), str)
        and bool(authority["obsolete_job_name"])
        and str(authority.get("obsolete_terminal_scheduler_state", "")).startswith(
            "CANCELLED"
        )
        and authority.get("obsolete_result_authoritative") is False
        and authority.get("authoritative_result") is True
        and authority.get("real_connectivity_accessed") is False
        and authority.get("final_connectivity_accessed") is False
        and authority.get("final_connectivity_access_authorized") is False,
        "observed attestation authority safety fields changed",
    )
    return authority


def build_transition_receipt(
    *,
    executor_root: Path,
    observed_attestation: Mapping[str, Any],
    observed_attestation_manifest_sha256: str,
    observed_attestation_authority: Mapping[str, Any],
    authenticated_trial_report_bytes: Mapping[str, bytes],
    synthetic_invariance_evidence: Mapping[str, Any],
    transition_spec_path: Path | None = None,
) -> dict[str, Any]:
    """Build a receipt from allowlisted extraction plus explicit exposure evidence."""

    spec, old_contract, new_contract, old_qualification, new_qualification = (
        _definition_snapshots(executor_root, transition_spec_path)
    )
    attested = _verify_attestation(observed_attestation)
    declarations = extract_authenticated_report_contract_declarations(
        attested_by_id=attested,
        authenticated_trial_report_bytes=authenticated_trial_report_bytes,
    )
    invariance_sha256 = verify_synthetic_invariance_evidence(
        synthetic_invariance_evidence, spec
    )
    manifest_sha256 = _sha256(
        observed_attestation_manifest_sha256,
        "observed attestation manifest sha256",
    )
    model_lineage = _mapping(
        _mapping(spec["implementation_lineage"], "implementation lineage")[
            "scientific_models"
        ],
        "scientific-model lineage",
    )
    rank_lineage = _mapping(
        _mapping(spec["implementation_lineage"], "implementation lineage")[
            "pre_null_rank_evaluator"
        ],
        "rank-evaluator lineage",
    )
    exposure = _mapping(_mapping(spec["outcome_firewall"], "transition exposure policy")["operator_exposure_incident"], "operator exposure incident")
    authority = verify_observed_attestation_authority(
        observed_attestation_authority
    )
    authority_requirement = _mapping(
        spec["required_observed_attestation_authority"],
        "observed attestation authority requirement",
    )
    _require(
        authority.get("obsolete_job_id")
        == authority_requirement.get("obsolete_job_id")
        and authority_requirement.get("obsolete_result_authoritative") is False,
        "observed attestation authority differs from the transition requirement",
    )
    core = {
        "schema_version": TRANSITION_RECEIPT_SCHEMA,
        "transition_id": TRANSITION_ID,
        "transition_spec_sha256": spec["transition_spec_sha256"],
        "old_contract_sha256": old_contract.sha256,
        "active_contract_sha256": new_contract.sha256,
        "old_qualification_spec_sha256": old_qualification.sha256,
        "active_qualification_spec_sha256": new_qualification.sha256,
        "allowed_contract_changes_sha256": _digest(spec["allowed_contract_changes"]),
        "allowed_qualification_spec_changes_sha256": _digest(
            spec["allowed_qualification_spec_changes"]
        ),
        "old_scientific_models_sha256": model_lineage["old_sha256"],
        "active_scientific_models_sha256": model_lineage["new_sha256"],
        "old_pre_null_rank_evaluator_sha256": rank_lineage["old_sha256"],
        "active_pre_null_rank_evaluator_sha256": rank_lineage["new_sha256"],
        "observed_attestation_manifest_sha256": manifest_sha256,
        "observed_attestation_sha256": observed_attestation["attestation_sha256"],
        "observed_attestation_authority": authority,
        "observed_trial_count": 36,
        "observed_report_contract_declarations_sha256": _digest(declarations),
        "every_observed_report_declares_old_contract": True,
        "synthetic_invariance_evidence_sha256": invariance_sha256,
        "rank_definition": "representation_latent_effective_rank",
        "operator_exposure_incident_relative_path": exposure["relative_path"],
        "operator_exposure_incident_file_sha256": exposure["file_sha256"],
        "operator_exposure_incident_semantic_sha256": exposure["semantic_sha256"],
        "operator_development_outcome_values_displayed": True,
        "programmatic_transition_extraction_outcome_fields_accessed": False,
        "values_used_for_transition_decisions": False,
        "final_connectivity_accessed": False,
        "complete": True,
    }
    receipt = {**core, "transition_receipt_sha256": _digest(core)}
    for snapshot in (
        old_contract,
        new_contract,
        old_qualification,
        new_qualification,
    ):
        snapshot.assert_unchanged()
    verify_transition_receipt(receipt)
    return receipt


def verify_transition_receipt(receipt: Mapping[str, Any]) -> None:
    """Authenticate the closed transition receipt and explicit exposure record."""

    payload = dict(_mapping(receipt, "transition receipt"))
    _require(
        set(payload) == _TRANSITION_RECEIPT_FIELDS,
        "transition receipt schema changed",
    )
    claimed = _sha256(
        payload.get("transition_receipt_sha256"), "transition receipt sha256"
    )
    core = {
        key: value for key, value in payload.items() if key != "transition_receipt_sha256"
    }
    _require(_digest(core) == claimed, "transition receipt self-hash mismatch")
    _require(
        payload.get("schema_version") == TRANSITION_RECEIPT_SCHEMA
        and payload.get("transition_id") == TRANSITION_ID
        and payload.get("transition_spec_sha256") == TRANSITION_SPEC_SHA256,
        "transition receipt identity mismatch",
    )
    for field in (
        "old_contract_sha256",
        "active_contract_sha256",
        "old_qualification_spec_sha256",
        "active_qualification_spec_sha256",
        "old_scientific_models_sha256",
        "active_scientific_models_sha256",
        "old_pre_null_rank_evaluator_sha256",
        "active_pre_null_rank_evaluator_sha256",
        "observed_attestation_manifest_sha256",
        "observed_attestation_sha256",
        "observed_report_contract_declarations_sha256",
        "synthetic_invariance_evidence_sha256",
        "operator_exposure_incident_file_sha256",
        "operator_exposure_incident_semantic_sha256",
    ):
        _sha256(payload.get(field), field)
    _require(
        payload.get("old_contract_sha256") == V1_CONTRACT_SHA256
        and payload.get("old_qualification_spec_sha256")
        == V1_QUALIFICATION_SPEC_SHA256
        and payload.get("observed_trial_count") == 36
        and payload.get("every_observed_report_declares_old_contract") is True
        and verify_observed_attestation_authority(
            _mapping(
                payload.get("observed_attestation_authority"),
                "observed attestation authority",
            )
        )
        and payload.get("rank_definition")
        == "representation_latent_effective_rank"
        and payload.get("operator_development_outcome_values_displayed") is True
        and payload.get(
            "programmatic_transition_extraction_outcome_fields_accessed"
        ) is False
        and payload.get("values_used_for_transition_decisions") is False
        and payload.get("final_connectivity_accessed") is False
        and payload.get("complete") is True,
        "transition receipt safety or completion fields mismatch",
    )
    for value in payload.values():
        if isinstance(value, float):
            _require(math.isfinite(value), "transition receipt contains a nonfinite value")
