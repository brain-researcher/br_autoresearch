"""Authenticate EP12's legacy-prefix to formal post-36 contract transition.

The generated-data probe comparison is evidence inside the transition bundle;
it is not itself the scientific-contract transition receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any, Mapping

from .contract_snapshot import capture_yaml_contract
from .scientific_contract_transition import (
    TRANSITION_RECEIPT_SCHEMA,
    verify_contract_transition_definition,
    verify_observed_attestation_authority,
    verify_transition_receipt,
)

PROVENANCE_SCHEMA = "ep12.contract_epoch_provenance.v2"
PREFIX_EPOCH = "observed_prefix_v1"
ACTIVE_EPOCH = "post36_active_v2"
RANK_RULE = "representation_latent_effective_rank"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractProvenanceError(ValueError):
    """The contract epoch boundary is missing or unauthenticated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractProvenanceError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractProvenanceError(f"{label} must be an object")
    return value


def _sha(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
             f"{label} must be a lowercase SHA-256 digest")
    return str(value)


def _validated_attestation_authority(value: Any) -> dict[str, Any]:
    try:
        return verify_observed_attestation_authority(
            _mapping(value, "dynamic observed-attestation authority")
        )
    except ValueError as exc:
        raise ContractProvenanceError(str(exc)) from exc


def _read_regular_bytes(path: Path) -> bytes:
    candidate = Path(os.path.abspath(os.fspath(path)))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ContractProvenanceError(
            f"cannot open provenance artifact without following links: {candidate}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), "provenance artifact is not regular")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        content = b"".join(chunks)
        _require(
            len(content) == metadata.st_size,
            "provenance artifact changed while it was read",
        )
        return content
    finally:
        os.close(descriptor)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(_read_regular_bytes(path)).hexdigest()


def validate_contract_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    provenance = dict(_mapping(value, "contract provenance"))
    _require(provenance.get("schema_version") == PROVENANCE_SCHEMA,
             "contract provenance schema mismatch")
    prefix = _mapping(provenance.get("prefix_contract"), "prefix contract")
    active = _mapping(provenance.get("active_contract"), "active contract")
    transition = _mapping(provenance.get("transition"), "contract transition")
    authority = _validated_attestation_authority(
        transition.get("observed_attestation_authority")
    )
    _require(prefix.get("epoch") == PREFIX_EPOCH, "prefix contract epoch mismatch")
    _require(active.get("epoch") == ACTIVE_EPOCH, "active contract epoch mismatch")
    _require(_sha(prefix.get("sha256"), "prefix contract sha256") !=
             _sha(active.get("sha256"), "active contract sha256"),
             "legacy and active contracts must be distinct")
    _sha(transition.get("semantic_sha256"), "transition semantic sha256")
    _sha(transition.get("file_sha256"), "transition file sha256")
    _sha(transition.get("spec_sha256"), "transition spec semantic sha256")
    _sha(transition.get("spec_file_sha256"), "transition spec file sha256")
    _require(transition.get("schema_version") == TRANSITION_RECEIPT_SCHEMA,
             "transition receipt schema mismatch")
    _require(transition.get("complete") is True, "formal transition is incomplete")
    _require(
        transition.get(
            "controller_final_receipt_required_before_irreversible_transition"
        )
        is True
        and transition.get("authoritative_attestation_job_id")
        == authority["authoritative_job_id"]
        and transition.get("obsolete_attestation_job_id")
        == authority["obsolete_job_id"],
        "formal transition dynamic authority summary changed",
    )
    _require(provenance.get("prevalence_parameter_rank_definition") == RANK_RULE,
             "active contract does not bind the representation-rank rule")
    _require(
        provenance.get("operator_development_outcome_values_displayed") is True
        and provenance.get("values_used_for_transition_decisions") is False
        and provenance.get("final_connectivity_accessed") is False,
        "contract transition exposure or use declaration is invalid",
    )
    return provenance


def load_contract_provenance(*, prefix_contract_path: Path,
                             active_contract_path: Path,
                             transition_receipt_path: Path,
                             transition_spec_path: Path) -> dict[str, Any]:
    prefix_path = Path(os.path.abspath(os.fspath(prefix_contract_path)))
    active_path = Path(os.path.abspath(os.fspath(active_contract_path)))
    receipt_path = Path(os.path.abspath(os.fspath(transition_receipt_path)))
    spec_path = Path(os.path.abspath(os.fspath(transition_spec_path)))
    prefix_snapshot = capture_yaml_contract(prefix_path)
    active_snapshot = capture_yaml_contract(active_path)
    receipt_bytes = _read_regular_bytes(receipt_path)
    spec_bytes = _read_regular_bytes(spec_path)
    receipt_file_hash = hashlib.sha256(receipt_bytes).hexdigest()
    spec_file_hash = hashlib.sha256(spec_bytes).hexdigest()
    try:
        receipt = json.loads(receipt_bytes.decode("utf-8"))
        spec = json.loads(spec_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractProvenanceError(
            "transition receipt or specification is not valid JSON"
        ) from exc
    receipt = dict(_mapping(receipt, "transition receipt"))
    spec = dict(_mapping(spec, "transition specification"))
    try:
        verify_transition_receipt(receipt)
        definition = verify_contract_transition_definition(
            executor_root=active_path.parent,
            transition_spec_path=spec_path,
        )
    except ValueError as exc:
        raise ContractProvenanceError(str(exc)) from exc
    semantic = _sha(
        receipt.get("transition_receipt_sha256"),
        "formal transition receipt semantic sha256",
    )
    spec_semantic_hash = _sha(
        spec.get("transition_spec_sha256"),
        "formal transition specification semantic sha256",
    )
    _require(
        receipt.get("old_contract_sha256") == prefix_snapshot.sha256
        == definition.get("old_contract_sha256"),
        "formal transition does not bind prefix contract bytes",
    )
    _require(
        receipt.get("active_contract_sha256") == active_snapshot.sha256
        == definition.get("new_contract_sha256"),
        "formal transition does not bind active contract bytes",
    )
    _require(
        receipt.get("transition_spec_sha256") == spec_semantic_hash
        == definition.get("transition_spec_sha256"),
        "formal transition specification semantic identity mismatch",
    )
    firewall = _mapping(spec.get("outcome_firewall"), "transition outcome firewall")
    exposure = _mapping(
        firewall.get("operator_exposure_incident"), "operator exposure incident"
    )
    authority_requirement = _mapping(
        spec.get("required_observed_attestation_authority"),
        "dynamic observed-attestation authority requirement",
    )
    authority = _validated_attestation_authority(
        receipt.get("observed_attestation_authority")
    )
    _require(
        receipt.get("rank_definition") == RANK_RULE
        and receipt.get("operator_development_outcome_values_displayed") is True
        and receipt.get(
            "programmatic_transition_extraction_outcome_fields_accessed"
        ) is False
        and receipt.get("values_used_for_transition_decisions") is False
        and receipt.get("final_connectivity_accessed") is False
        and receipt.get("complete") is True,
        "formal transition violates the scoped outcome firewall",
    )
    _require(
        receipt.get("operator_exposure_incident_relative_path")
        == exposure.get("relative_path")
        and receipt.get("operator_exposure_incident_file_sha256")
        == exposure.get("file_sha256")
        and receipt.get("operator_exposure_incident_semantic_sha256")
        == exposure.get("semantic_sha256"),
        "formal transition does not bind the pinned operator exposure incident",
    )
    _require(
        authority_requirement.get("schema_version")
        == "ep12.dynamic_observed_attestation_authority_requirement.v1"
        and authority_requirement.get("controller_receipt_policy")
        == "bind_if_present_require_before_irreversible_transition"
        and authority_requirement.get("dynamic_replacement_required") is True
        and authority_requirement.get("source_archive_required") is True
        and authority_requirement.get("obsolete_job_id")
        == authority.get("obsolete_job_id")
        == "45332084"
        and authority_requirement.get("obsolete_result_authoritative") is False
        and authority.get("obsolete_result_authoritative") is False
        and authority.get("authoritative_result") is True,
        "formal transition does not bind dynamic replacement authority",
    )
    prefix_snapshot.assert_unchanged()
    active_snapshot.assert_unchanged()
    _require(
        _read_regular_bytes(receipt_path) == receipt_bytes
        and _read_regular_bytes(spec_path) == spec_bytes,
        "transition receipt or specification changed during verification",
    )
    return validate_contract_provenance({
        "schema_version": PROVENANCE_SCHEMA,
        "prefix_contract": {
            "epoch": PREFIX_EPOCH, "path": str(prefix_path),
            "sha256": prefix_snapshot.sha256,
        },
        "active_contract": {
            "epoch": ACTIVE_EPOCH, "path": str(active_path),
            "sha256": active_snapshot.sha256,
        },
        "transition": {
            "schema_version": TRANSITION_RECEIPT_SCHEMA,
            "complete": True,
            "path": str(receipt_path),
            "semantic_sha256": semantic,
            "file_sha256": receipt_file_hash,
            "spec_path": str(spec_path),
            "spec_sha256": spec_semantic_hash,
            "spec_file_sha256": spec_file_hash,
            "observed_attestation_authority": authority,
            "authoritative_attestation_job_id": authority[
                "authoritative_job_id"
            ],
            "obsolete_attestation_job_id": authority["obsolete_job_id"],
            "controller_final_receipt_required_before_irreversible_transition": True,
        },
        "prevalence_parameter_rank_definition": RANK_RULE,
        "operator_development_outcome_values_displayed": True,
        "values_used_for_transition_decisions": False,
        "final_connectivity_accessed": False,
    })


def prefix_record_fields(provenance: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_contract_provenance(provenance)
    return {"contract_epoch": PREFIX_EPOCH,
            "scientific_contract_sha256": value["prefix_contract"]["sha256"]}


def active_record_fields(provenance: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_contract_provenance(provenance)
    return {
        "contract_epoch": ACTIVE_EPOCH,
        "scientific_contract_sha256": value["active_contract"]["sha256"],
        "contract_transition_semantic_sha256": value["transition"]["semantic_sha256"],
        "contract_transition_file_sha256": value["transition"]["file_sha256"],
    }


def require_record_fields(record: Mapping[str, Any], provenance: Mapping[str, Any],
                          *, active: bool, label: str) -> None:
    expected = active_record_fields(provenance) if active else prefix_record_fields(provenance)
    for key, value in expected.items():
        _require(record.get(key) == value, f"{label} contract provenance mismatch: {key}")


__all__ = ["ACTIVE_EPOCH", "PREFIX_EPOCH", "PROVENANCE_SCHEMA", "RANK_RULE",
           "ContractProvenanceError", "active_record_fields", "file_sha256",
           "load_contract_provenance", "prefix_record_fields", "require_record_fields",
           "validate_contract_provenance"]
