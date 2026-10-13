"""Authenticate EP12's published post-36 scientific-contract transition.

The verifier authenticates the documented operator-display incident and makes
only a narrow programmatic claim: transition extraction accesses the separate
status attestation, allowlisted report declarations, and generated-data probe
receipts, not candidate-discriminating outcome fields.  The observed-prefix
archive is authenticated as opaque bytes and is never decoded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping


TRANSITION_MANIFEST_SCHEMA = "ep12.scientific_contract_transition_manifest.v2"
TRANSITION_ID = "ep12_post36_representation_rank_v1"
TRANSITION_ROOT_RELATIVE = Path("outputs/prelaunch/scientific_contract_transition")
TRANSITION_MANIFEST_NAME = "manifest.json"
TRANSITION_RECEIPT_NAME = "receipt.json"
REPORT_DECLARATIONS_NAME = "report_declarations.json"
SYNTHETIC_INVARIANCE_NAME = "synthetic_invariance.json"
OBSERVED_ATTESTATION_RELATIVE = Path(
    "outputs/development_run002/observed_attestation"
)
OBSERVED_FINALIZATION_RELATIVE = Path(
    "outputs/development_run002/observed_finalization"
)
OBSERVED_ATTESTATION_NAME = "observed_trial_attestations.json"
OBSERVED_PREFIX_ARCHIVE_NAME = "observed_search_archive.json"

REPORT_DECLARATIONS_SCHEMA = "ep12.observed_report_contract_declarations.v1"
OPERATOR_EXPOSURE_SCHEMA = "ep12.operator_exposure_incident.v1"
ATTESTATION_JOB_IDENTITY_SCHEMA = (
    "ep12.pending_observed_attestation_job_identity.v1"
)
OBSERVED_PREFIX_CONTRACT_SHA256 = (
    "f7de6c70fac3f6690b909a47162c2c9183cc8747ce6fa7c2fc281c7e2363b006"
)
OBSERVED_PREFIX_QUALIFICATION_SPEC_SHA256 = (
    "e4a3e81b08b3c46b793823112758fbb811603e8ba4ea7bc9febfe2066ba5eac0"
)

_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_TRANSITION_FILES = {
    TRANSITION_RECEIPT_NAME,
    REPORT_DECLARATIONS_NAME,
    SYNTHETIC_INVARIANCE_NAME,
}
_DECLARATION_FIELDS = {
    "trial_id",
    "source_report_schema",
    "source_report_status",
    "source_trial_report_sha256",
    "declared_scientific_contract_sha256",
    "data_role",
    "final_connectivity_accessed",
}
_MANIFEST_FIELDS = {
    "schema_version",
    "files",
    "transition_id",
    "transition_receipt_sha256",
    "transition_spec_sha256",
    "observed_prefix_contract_sha256",
    "active_contract_sha256",
    "observed_prefix_qualification_spec_sha256",
    "active_qualification_spec_sha256",
    "observed_prefix_scientific_models_sha256",
    "active_scientific_models_sha256",
    "observed_attestation_manifest_file_sha256",
    "observed_attestation_artifact_file_sha256",
    "observed_attestation_sha256",
    "observed_attestation_authority_sha256",
    "authoritative_attestation_job_id",
    "obsolete_attestation_job_id",
    "source_archive_receipt_file_sha256",
    "source_archive_receipt_sha256",
    "source_archive_manifest_file_sha256",
    "source_archive_manifest_sha256",
    "source_archive_tree_sha256",
    "controller_final_receipt_status",
    "observed_trial_count",
    "observed_report_contract_declarations_sha256",
    "observed_prefix_archive_file_sha256",
    "observed_prefix_limitation_manifest_file_sha256",
    "legacy_probe_file_sha256",
    "legacy_probe_artifact_sha256",
    "legacy_probe_scientific_payload_sha256",
    "post36_probe_file_sha256",
    "post36_probe_artifact_sha256",
    "post36_probe_scientific_payload_sha256",
    "allowed_diagnostics_delta_sha256",
    "operator_exposure_incident_file_sha256",
    "operator_exposure_incident_semantic_sha256",
    "source_trial_reports_reverified",
    "operator_development_outcome_values_displayed",
    "programmatic_transition_extraction_outcome_fields_accessed",
    "values_used_for_transition_decisions",
    "final_connectivity_accessed",
    "complete",
}

_EXPOSURE_INCIDENT_FIELDS = {
    "schema_version",
    "episode_id",
    "episode_root",
    "observed_at",
    "actor",
    "access_class",
    "intended_purpose",
    "cause",
    "displayed_content_classes",
    "displayed_path_patterns",
    "development_connectivity_outcome_values_displayed",
    "development_connectivity_outcome_values_intentionally_requested",
    "development_connectivity_outcome_values_analyzed",
    "development_connectivity_outcome_values_used_for_scientific_decisions",
    "scientific_contract_or_threshold_changed_after_display",
    "final_connectivity_accessed",
    "remediation",
    "incident_sha256",
}

_ATTESTATION_JOB_IDENTITY_FIELDS = {
    "schema_version",
    "slurm_job_id",
    "slurm_job_name",
    "submission_user",
    "submitted_at",
    "identity_captured_at",
    "dependency",
    "partition",
    "cpus_per_task",
    "memory_gib",
    "time_limit",
    "spooled_batch_script_sha256",
    "current_on_disk_wrapper_sha256",
    "spooled_script_matches_current_on_disk_wrapper",
    "spooled_script_uses_live_oak_python_imports",
    "pre_run_live_import_source_sha256",
    "identity_limitations",
    "real_connectivity_outcome_values_read_to_build_identity",
    "final_connectivity_accessed",
    "identity_sha256",
}

ATTESTATION_AUTHORITY_SCHEMA = "ep12.dynamic_observed_attestation_authority.v1"
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


class ScientificTransitionBindingError(ValueError):
    """The durable post-36 transition bundle is missing or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScientificTransitionBindingError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a JSON object")
    return value


def _sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _HEX_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _canonical_digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScientificTransitionBindingError(
            "transition evidence is not finite canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _secure_path(
    path: Path,
    *,
    boundary: Path,
    label: str,
    require_file: bool = True,
) -> Path:
    candidate = _lexical(path)
    root = _lexical(boundary)
    _require(path.is_absolute() and boundary.is_absolute(), f"{label} must be absolute")
    _require(
        candidate == root or root in candidate.parents,
        f"{label} escapes its fixed boundary: {candidate}",
    )
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise ScientificTransitionBindingError(f"missing {label}: {candidate}") from exc
        except OSError as exc:
            raise ScientificTransitionBindingError(
                f"cannot inspect {label} component {current}: {exc}"
            ) from exc
        _require(not stat.S_ISLNK(metadata.st_mode), f"{label} contains a symlink: {current}")
    if require_file:
        _require(stat.S_ISREG(os.lstat(candidate).st_mode), f"{label} is not a regular file")
    else:
        _require(stat.S_ISDIR(os.lstat(candidate).st_mode), f"{label} is not a directory")
    return candidate


def _read_bytes(path: Path, *, boundary: Path, label: str) -> bytes:
    candidate = _lexical(path)
    root = _lexical(boundary)
    _require(path.is_absolute() and boundary.is_absolute(), f"{label} must be absolute")
    _require(
        candidate != root and root in candidate.parents,
        f"{label} escapes or equals its directory boundary: {candidate}",
    )
    relative = candidate.relative_to(root)
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    # Open every absolute boundary component relative to an already-open
    # descriptor.  Unlike a preflight lstat followed by open(path), this
    # cannot be redirected through an intermediate symlink between checks.
    directory_fd = os.open(Path(candidate.anchor), directory_flags)
    try:
        for component in root.parts[1:]:
            child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        for component in relative.parts[:-1]:
            _require(component not in {"", ".", ".."}, f"unsafe {label} component")
            child_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        leaf = relative.parts[-1]
        _require(leaf not in {"", ".", ".."}, f"unsafe {label} leaf")
        descriptor = os.open(
            leaf,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
    except OSError as exc:
        os.close(directory_fd)
        raise ScientificTransitionBindingError(
            f"cannot safely open {label} {candidate}: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        try:
            named = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise ScientificTransitionBindingError(
                f"{label} path changed while it was read: {exc}"
            ) from exc
        _require(
            stat.S_ISREG(named.st_mode)
            and (metadata.st_dev, metadata.st_ino) == (named.st_dev, named.st_ino),
            f"{label} path changed while it was read",
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)
        os.close(directory_fd)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"transition JSON repeats key {key!r}")
        result[key] = value
    return result


def _json_from_bytes(encoded: bytes, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScientificTransitionBindingError(f"malformed {label}") from exc
    return _mapping(value, label)


def _file_sha(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _spec_relative_file(
    *, episode_root: Path, relative: Any, label: str
) -> Path:
    _require(isinstance(relative, str) and relative, f"{label} path must be text")
    parsed = Path(relative)
    _require(
        not parsed.is_absolute()
        and parsed.parts
        and all(component not in {"", ".", ".."} for component in parsed.parts)
        and parsed.as_posix() == relative,
        f"unsafe or noncanonical {label} path: {relative!r}",
    )
    return _secure_path(
        _lexical(episode_root / parsed),
        boundary=_lexical(episode_root),
        label=label,
    )


def _verify_operator_exposure_incident(
    *, episode_root: Path, transition_spec: Mapping[str, Any]
) -> dict[str, str | bool]:
    """Authenticate the pinned operator-display incident without generalizing it.

    The incident explicitly records that development outcome values were shown
    to an operator.  It does *not* license a global outcome-blind claim.  The
    narrower facts needed by the transition are that the values were not
    analyzed or used for the rank-definition decision and that final
    connectivity stayed closed.
    """

    firewall = _mapping(
        transition_spec.get("outcome_firewall"), "transition outcome firewall"
    )
    expected = _mapping(
        firewall.get("operator_exposure_incident"),
        "pinned operator exposure incident",
    )
    path = _spec_relative_file(
        episode_root=episode_root,
        relative=expected.get("relative_path"),
        label="operator exposure incident",
    )
    encoded = _read_bytes(
        path, boundary=_lexical(episode_root), label="operator exposure incident"
    )
    file_sha = _file_sha(encoded)
    _require(
        file_sha == _sha(expected.get("file_sha256"), "operator incident file sha256"),
        "operator exposure incident file hash mismatch",
    )
    incident = _json_from_bytes(encoded, "operator exposure incident")
    _require(
        set(incident) == _EXPOSURE_INCIDENT_FIELDS,
        "operator exposure incident schema changed",
    )
    semantic_sha = _sha(incident.get("incident_sha256"), "operator incident semantic sha256")
    _require(
        _canonical_digest(
            {key: value for key, value in incident.items() if key != "incident_sha256"}
        )
        == semantic_sha
        == _sha(expected.get("semantic_sha256"), "pinned operator incident semantic sha256"),
        "operator exposure incident semantic hash mismatch",
    )
    _require(
        incident.get("schema_version")
        == expected.get("schema_version")
        == OPERATOR_EXPOSURE_SCHEMA
        and incident.get("episode_id") == "ep12"
        and _lexical(Path(str(incident.get("episode_root", ""))))
        == _lexical(episode_root),
        "operator exposure incident identity mismatch",
    )
    _require(
        incident.get("development_connectivity_outcome_values_displayed") is True
        and incident.get("development_connectivity_outcome_values_intentionally_requested")
        is False
        and incident.get("development_connectivity_outcome_values_analyzed") is False
        and incident.get(
            "development_connectivity_outcome_values_used_for_scientific_decisions"
        )
        is False
        and incident.get("scientific_contract_or_threshold_changed_after_display")
        is False
        and incident.get("final_connectivity_accessed") is False,
        "operator exposure incident safety facts changed",
    )
    _require(
        firewall.get("operator_development_outcome_values_displayed") is True
        and firewall.get("programmatic_transition_extraction_outcome_fields_accessed")
        is False
        and firewall.get("values_used_for_transition_decisions") is False
        and firewall.get("final_connectivity_accessed") is False,
        "transition outcome-firewall declarations changed",
    )
    return {
        "operator_exposure_incident_file_sha256": file_sha,
        "operator_exposure_incident_semantic_sha256": semantic_sha,
        "operator_development_outcome_values_displayed": True,
        "programmatic_transition_extraction_outcome_fields_accessed": False,
        "values_used_for_transition_decisions": False,
        "final_connectivity_accessed": False,
    }


def _verify_superseded_attestation_job_identity(
    *, episode_root: Path, transition_spec: Mapping[str, Any]
) -> dict[str, str | bool]:
    """Authenticate the old job as provenance only, never as authority."""

    expected = _mapping(
        transition_spec.get("superseded_observed_attestation_job_identity"),
        "superseded observed-attestation job identity",
    )
    _require(
        set(expected)
        == {
            "file_sha256",
            "identity_sha256",
            "relative_path",
            "schema_version",
            "slurm_job_id",
            "authoritative",
            "disposition",
        }
        and expected.get("authoritative") is False
        and expected.get("disposition")
        == "terminally_cancelled_superseded_non_authoritative",
        "transition spec misclassifies the superseded attestation job",
    )
    path = _spec_relative_file(
        episode_root=episode_root,
        relative=expected.get("relative_path"),
        label="observed-attestation job identity",
    )
    encoded = _read_bytes(
        path,
        boundary=_lexical(episode_root),
        label="observed-attestation job identity",
    )
    file_sha = _file_sha(encoded)
    _require(
        file_sha
        == _sha(expected.get("file_sha256"), "attestation job identity file sha256"),
        "observed-attestation job identity file hash mismatch",
    )
    identity = _json_from_bytes(encoded, "observed-attestation job identity")
    _require(
        set(identity) == _ATTESTATION_JOB_IDENTITY_FIELDS,
        "observed-attestation job identity schema changed",
    )
    semantic_sha = _sha(
        identity.get("identity_sha256"), "attestation job identity semantic sha256"
    )
    _require(
        _file_sha(
            (
                json.dumps(
                    {
                        key: value
                        for key, value in identity.items()
                        if key != "identity_sha256"
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        )
        == semantic_sha
        == _sha(expected.get("identity_sha256"), "pinned attestation job identity sha256"),
        "observed-attestation job identity semantic hash mismatch",
    )
    _require(
        identity.get("schema_version")
        == expected.get("schema_version")
        == ATTESTATION_JOB_IDENTITY_SCHEMA
        and identity.get("slurm_job_id") == expected.get("slurm_job_id")
        and identity.get("spooled_script_matches_current_on_disk_wrapper") is False
        and identity.get("spooled_script_uses_live_oak_python_imports") is True
        and identity.get("real_connectivity_outcome_values_read_to_build_identity")
        is False
        and identity.get("final_connectivity_accessed") is False,
        "observed-attestation job identity or safety facts changed",
    )
    _sha(identity.get("spooled_batch_script_sha256"), "spooled attestation script sha256")
    _sha(identity.get("current_on_disk_wrapper_sha256"), "attestation wrapper sha256")
    _require(
        isinstance(identity.get("pre_run_live_import_source_sha256"), Mapping)
        and bool(identity["pre_run_live_import_source_sha256"]),
        "superseded attestation identity has no recorded source provenance",
    )
    return {
        "superseded_job_id": str(identity["slurm_job_id"]),
        "superseded_identity_file_sha256": file_sha,
        "superseded_identity_sha256": semantic_sha,
        "superseded_result_authoritative": False,
    }


def _safe_requirement_path(
    *, episode_root: Path, relative: Any, label: str
) -> Path:
    _require(isinstance(relative, str) and bool(relative), f"{label} path is missing")
    candidate = Path(relative)
    _require(
        not candidate.is_absolute()
        and candidate.as_posix() == relative
        and all(part not in {"", ".", ".."} for part in candidate.parts),
        f"{label} path is unsafe",
    )
    return episode_root / candidate


def _verify_dynamic_attestation_authority(
    *,
    episode_root: Path,
    transition_spec: Mapping[str, Any],
    terminal_identity_path: Path,
    start_authorization_path: Path,
    cancellation_receipt_path: Path,
    attestation_manifest_file_sha256: str,
    attestation_artifact_file_sha256: str,
    attestation_sha256: str,
) -> dict[str, Any]:
    """Authenticate dynamic R, terminal cancellation, and the pre-R barrier."""

    requirement = _mapping(
        transition_spec.get("required_observed_attestation_authority"),
        "required observed-attestation authority",
    )
    _require(
        set(requirement)
        == {
            "schema_version",
            "terminal_identity_relative_path",
            "cancellation_receipt_relative_path",
            "start_authorization_relative_path",
            "controller_receipt_relative_path",
            "controller_receipt_policy",
            "dynamic_replacement_required",
            "source_archive_required",
            "obsolete_job_id",
            "obsolete_result_authoritative",
        }
        and requirement.get("schema_version")
        == "ep12.dynamic_observed_attestation_authority_requirement.v1"
        and requirement.get("controller_receipt_policy")
        == "bind_if_present_require_before_irreversible_transition"
        and requirement.get("dynamic_replacement_required") is True
        and requirement.get("source_archive_required") is True
        and requirement.get("obsolete_job_id") == "45332084"
        and requirement.get("obsolete_result_authoritative") is False,
        "dynamic observed-attestation authority requirement changed",
    )
    terminal_path = _safe_requirement_path(
        episode_root=episode_root,
        relative=requirement["terminal_identity_relative_path"],
        label="replacement terminal identity",
    )
    _require(
        terminal_identity_path.is_absolute()
        and _lexical(terminal_identity_path) == _lexical(terminal_path),
        "replacement terminal identity path is not canonical",
    )
    cancellation_path = _safe_requirement_path(
        episode_root=episode_root,
        relative=requirement["cancellation_receipt_relative_path"],
        label="obsolete cancellation receipt",
    )
    start_path = _safe_requirement_path(
        episode_root=episode_root,
        relative=requirement["start_authorization_relative_path"],
        label="post36 start authorization",
    )
    controller_path = _safe_requirement_path(
        episode_root=episode_root,
        relative=requirement["controller_receipt_relative_path"],
        label="post36 controller receipt",
    )
    _require(
        start_authorization_path.is_absolute()
        and _lexical(start_authorization_path) == _lexical(start_path),
        "post36 start authorization path is not canonical",
    )
    _require(
        cancellation_receipt_path.is_absolute()
        and _lexical(cancellation_receipt_path) == _lexical(cancellation_path),
        "obsolete cancellation receipt path is not canonical",
    )
    try:
        from .post36_unattended_controller import (
            verify_post36_orchestration_receipt,
            verify_post36_start_authorization,
        )
        from .replacement_attestation_terminal import (
            verify_replacement_attestation_terminal_identity,
        )

        terminal = verify_replacement_attestation_terminal_identity(
            episode_root=episode_root, identity_path=terminal_path
        )
        start = verify_post36_start_authorization(
            episode_root=episode_root,
            authorization_path=start_authorization_path,
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise ScientificTransitionBindingError(
            f"dynamic replacement authority verification failed: {exc}"
        ) from exc
    replacement = _mapping(start.get("replacement"), "replacement launch evidence")
    cancellation = _mapping(
        start.get("obsolete_cancellation"), "obsolete cancellation evidence"
    )
    _require(
        start.get("authorization_path") == str(start_path)
        and cancellation.get("cancellation_receipt_path")
        == str(cancellation_receipt_path)
        and terminal.get("slurm_job_id") == replacement.get("slurm_job_id")
        and terminal.get("terminal_identity_path") == str(terminal_path)
        and terminal.get("attestation_manifest_file_sha256")
        == _sha(
            attestation_manifest_file_sha256,
            "observed attestation manifest file sha256",
        )
        and terminal.get("attestation_artifact_file_sha256")
        == _sha(
            attestation_artifact_file_sha256,
            "observed attestation artifact file sha256",
        )
        and terminal.get("attestation_sha256")
        == _sha(attestation_sha256, "observed attestation semantic sha256")
        and terminal.get("attestation_result_authoritative") is True
        and terminal.get("superseded_result_authoritative") is False,
        "dynamic replacement terminal does not bind the observed attestation",
    )
    _require(
        cancellation.get("cancellation_receipt_path") == str(cancellation_path)
        and cancellation.get("obsolete_job_id") == requirement["obsolete_job_id"]
        and cancellation.get("replacement_job_id") == terminal["slurm_job_id"]
        and str(cancellation.get("terminal_scheduler_state", "")).startswith(
            "CANCELLED"
        )
        and cancellation.get("superseded_result_authoritative") is False,
        "obsolete cancellation is not bound to dynamic R",
    )
    superseded = _verify_superseded_attestation_job_identity(
        episode_root=episode_root, transition_spec=transition_spec
    )
    _require(
        superseded["superseded_job_id"] == cancellation["obsolete_job_id"],
        "superseded provenance and cancellation identify different jobs",
    )
    archive = dict(_mapping(terminal.get("archive"), "replacement source archive"))
    _require(
        set(archive) == _SOURCE_ARCHIVE_FIELDS
        and archive == replacement.get("archive")
        and archive.get("complete") is True
        and archive.get("inputs_included") is False
        and archive.get("development_outputs_included") is False
        and archive.get("prelaunch_outputs_included") is False
        and archive.get("parent_first_publication_bound") is True,
        "dynamic replacement source archive identity changed",
    )
    for field in (
        "archive_receipt_file_sha256",
        "archive_receipt_sha256",
        "manifest_file_sha256",
        "manifest_sha256",
        "tree_sha256",
        "bootstrap_sha256",
    ):
        _sha(archive.get(field), f"source archive {field}")

    if os.path.lexists(controller_path):
        try:
            final = verify_post36_orchestration_receipt(episode_root=episode_root)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            raise ScientificTransitionBindingError(
                f"present post36 controller receipt is invalid: {exc}"
            ) from exc
        _require(
            final.get("receipt_path") == str(controller_path)
            and final.get("intent_path") == start.get("intent_path")
            and final.get("intent_file_sha256") == start.get("intent_file_sha256")
            and final.get("intent_sha256") == start.get("intent_sha256")
            and _mapping(final.get("replacement"), "final replacement").get(
                "slurm_job_id"
            )
            == terminal["slurm_job_id"]
            and _mapping(
                final.get("obsolete_cancellation"), "final cancellation"
            ).get("cancellation_receipt_sha256")
            == cancellation["cancellation_receipt_sha256"]
            and _mapping(
                final.get("start_authorization"), "final start authorization"
            ).get("start_authorization_sha256")
            == start["start_authorization_sha256"],
            "post36 controller receipt does not extend the authenticated start barrier",
        )
        final_binding = {
            "status": "present_verified",
            "path": str(controller_path),
            "file_sha256": _sha(
                final.get("receipt_file_sha256"), "post36 controller receipt file sha256"
            ),
            "receipt_sha256": _sha(
                final.get("receipt_sha256"), "post36 controller receipt semantic sha256"
            ),
            "required_before_irreversible_transition": True,
        }
    else:
        final_binding = {
            "status": "absent_at_publication",
            "path": str(controller_path),
            "file_sha256": None,
            "receipt_sha256": None,
            "required_before_irreversible_transition": True,
        }
    core = {
        "schema_version": ATTESTATION_AUTHORITY_SCHEMA,
        "authoritative_job_id": terminal["slurm_job_id"],
        "authoritative_job_name": terminal["slurm_job_name"],
        "replacement_terminal_identity_path": terminal["terminal_identity_path"],
        "replacement_terminal_identity_file_sha256": terminal[
            "terminal_identity_file_sha256"
        ],
        "replacement_terminal_identity_sha256": terminal[
            "terminal_identity_sha256"
        ],
        "replacement_launch_receipt_file_sha256": terminal[
            "launch_receipt_file_sha256"
        ],
        "replacement_launch_receipt_sha256": terminal["launch_receipt_sha256"],
        "replacement_release_receipt_file_sha256": terminal[
            "release_receipt_file_sha256"
        ],
        "replacement_release_receipt_sha256": terminal[
            "release_receipt_sha256"
        ],
        "post36_orchestration_intent_path": start["intent_path"],
        "post36_orchestration_intent_file_sha256": start["intent_file_sha256"],
        "post36_orchestration_intent_sha256": start["intent_sha256"],
        "post36_start_authorization_path": start["authorization_path"],
        "post36_start_authorization_file_sha256": start[
            "authorization_file_sha256"
        ],
        "post36_start_authorization_sha256": start[
            "start_authorization_sha256"
        ],
        "obsolete_cancellation_receipt_path": cancellation[
            "cancellation_receipt_path"
        ],
        "obsolete_cancellation_receipt_file_sha256": cancellation[
            "cancellation_receipt_file_sha256"
        ],
        "obsolete_cancellation_receipt_sha256": cancellation[
            "cancellation_receipt_sha256"
        ],
        "obsolete_job_id": cancellation["obsolete_job_id"],
        "obsolete_job_name": cancellation["obsolete_job_name"],
        "obsolete_terminal_scheduler_state": cancellation[
            "terminal_scheduler_state"
        ],
        "obsolete_identity_file_sha256": superseded[
            "superseded_identity_file_sha256"
        ],
        "obsolete_identity_sha256": superseded["superseded_identity_sha256"],
        "obsolete_result_authoritative": False,
        "authoritative_result": True,
        "source_archive": archive,
        "controller_final_receipt": final_binding,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "authority_sha256": _canonical_digest(core)}


def _verify_published_attestation_authority(
    published: Mapping[str, Any], *, current: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the immutable publication snapshot against current authority."""

    value = dict(published)
    _require(
        set(value) == _ATTESTATION_AUTHORITY_FIELDS
        and value.get("schema_version") == ATTESTATION_AUTHORITY_SCHEMA,
        "published observed-attestation authority schema changed",
    )
    core = {key: item for key, item in value.items() if key != "authority_sha256"}
    _require(
        value.get("authority_sha256") == _canonical_digest(core),
        "published observed-attestation authority self-hash mismatch",
    )
    current_core = {
        key: item
        for key, item in current.items()
        if key not in {"authority_sha256", "controller_final_receipt"}
    }
    published_core = {
        key: item
        for key, item in value.items()
        if key not in {"authority_sha256", "controller_final_receipt"}
    }
    _require(
        published_core == current_core,
        "published dynamic replacement authority differs from current evidence",
    )
    published_final = _mapping(
        value.get("controller_final_receipt"),
        "published controller final-receipt binding",
    )
    current_final = _mapping(
        current.get("controller_final_receipt"),
        "current controller final-receipt binding",
    )
    _require(
        set(published_final) == _CONTROLLER_FINAL_RECEIPT_FIELDS
        and published_final.get("required_before_irreversible_transition") is True,
        "published controller final-receipt policy changed",
    )
    if published_final.get("status") == "present_verified":
        _require(
            dict(published_final) == dict(current_final),
            "published controller final receipt changed",
        )
    else:
        _require(
            published_final
            == {
                "status": "absent_at_publication",
                "path": current_final.get("path"),
                "file_sha256": None,
                "receipt_sha256": None,
                "required_before_irreversible_transition": True,
            }
            and current_final.get("status")
            in {"absent_at_publication", "present_verified"},
            "published controller final-receipt absence binding changed",
        )
    return value


def _transition_module() -> Any:
    try:
        from . import scientific_contract_transition as transition
    except ImportError as exc:  # integration supplies the contract-core patch
        raise ScientificTransitionBindingError(
            "scientific-contract transition core is not installed"
        ) from exc
    return transition


def _probe_module() -> Any:
    from . import contract_invariance_probe

    return contract_invariance_probe


def _verify_attestation_manifest(
    *, episode_root: Path, attestation_dir: Path
) -> tuple[Mapping[str, Any], str, Mapping[str, Any], str]:
    expected_dir = episode_root / OBSERVED_ATTESTATION_RELATIVE
    _require(
        _lexical(attestation_dir) == _lexical(expected_dir),
        "transition attestation directory is not the canonical observed root",
    )
    manifest_bytes = _read_bytes(
        attestation_dir / "manifest.json",
        boundary=episode_root,
        label="observed attestation manifest",
    )
    manifest = _json_from_bytes(manifest_bytes, "observed attestation manifest")
    expected_fields = {
        "schema_version",
        "files",
        "attestation_sha256",
        "trial_count",
        "source_trial_files_modified",
        "objective_values_emitted",
        "final_connectivity_accessed",
        "final_connectivity_access_authorized",
    }
    _require(set(manifest) == expected_fields, "observed attestation manifest schema changed")
    _require(
        manifest.get("schema_version")
        == "ep12.observed_trial_attestation_set_manifest.v1",
        "observed attestation manifest identity changed",
    )
    files = _mapping(manifest.get("files"), "observed attestation files")
    _require(set(files) == {OBSERVED_ATTESTATION_NAME}, "observed attestation file set changed")
    artifact_bytes = _read_bytes(
        attestation_dir / OBSERVED_ATTESTATION_NAME,
        boundary=episode_root,
        label="observed attestation artifact",
    )
    artifact_file_sha = _file_sha(artifact_bytes)
    _require(
        artifact_file_sha == files[OBSERVED_ATTESTATION_NAME],
        "observed attestation artifact hash mismatch",
    )
    artifact = _json_from_bytes(artifact_bytes, "observed attestation artifact")
    _require(
        manifest.get("attestation_sha256") == artifact.get("attestation_sha256")
        and manifest.get("trial_count") == 36
        and manifest.get("source_trial_files_modified") is False
        and manifest.get("objective_values_emitted") is False
        and manifest.get("final_connectivity_accessed") is False
        and manifest.get("final_connectivity_access_authorized") is False,
        "observed attestation manifest safety fields changed",
    )
    return manifest, _file_sha(manifest_bytes), artifact, artifact_file_sha


def _verify_observed_prefix_archive(
    *, episode_root: Path, finalization_dir: Path
) -> tuple[str, str]:
    expected_dir = episode_root / OBSERVED_FINALIZATION_RELATIVE
    _require(
        _lexical(finalization_dir) == _lexical(expected_dir),
        "transition finalization directory is not the canonical observed-prefix root",
    )
    manifest_bytes = _read_bytes(
        finalization_dir / "manifest.json",
        boundary=episode_root,
        label="observed-prefix limitation manifest",
    )
    manifest = _json_from_bytes(manifest_bytes, "observed-prefix limitation manifest")
    _require(
        set(manifest) == {"schema_version", "files", "final_connectivity_accessed"}
        and manifest.get("schema_version")
        == "ep12.observed_minimum_finalization_manifest.v1"
        and manifest.get("final_connectivity_accessed") is False,
        "observed-prefix limitation manifest changed",
    )
    files = _mapping(manifest.get("files"), "observed-prefix archive files")
    _require(set(files) == {OBSERVED_PREFIX_ARCHIVE_NAME}, "observed-prefix archive file set changed")
    archive_bytes = _read_bytes(
        finalization_dir / OBSERVED_PREFIX_ARCHIVE_NAME,
        boundary=episode_root,
        label="opaque observed-prefix archive",
    )
    archive_sha = _file_sha(archive_bytes)
    _require(
        archive_sha == files[OBSERVED_PREFIX_ARCHIVE_NAME],
        "observed-prefix archive hash mismatch",
    )
    # Deliberately do not decode archive_bytes: it contains observed outcomes.
    return archive_sha, _file_sha(manifest_bytes)


def _verify_declarations(
    *,
    payload: Mapping[str, Any],
    attested_by_id: Mapping[str, Mapping[str, Any]],
    attestation_manifest_sha: str,
    attestation_semantic_sha: str,
    transition: Any,
) -> str:
    expected_fields = {
        "schema_version",
        "transition_id",
        "observed_attestation_manifest_sha256",
        "observed_attestation_sha256",
        "trial_count",
        "rows",
        "objective_values_read",
        "objective_values_compared",
        "objective_values_emitted",
        "final_connectivity_accessed",
        "declarations_sha256",
    }
    _require(set(payload) == expected_fields, "transition declaration payload schema changed")
    rows = payload.get("rows")
    _require(isinstance(rows, list) and len(rows) == 36, "transition declarations are not exact-36")
    for row in rows:
        _require(isinstance(row, Mapping) and set(row) == _DECLARATION_FIELDS, "transition declaration row exceeds its allowlist")
    normalized = transition._verify_report_declarations(rows, attested_by_id)
    _require(rows == normalized, "transition declarations are not sorted canonically")
    digest = _canonical_digest(rows)
    _require(
        payload.get("schema_version") == REPORT_DECLARATIONS_SCHEMA
        and payload.get("transition_id") == TRANSITION_ID
        and payload.get("observed_attestation_manifest_sha256") == attestation_manifest_sha
        and payload.get("observed_attestation_sha256") == attestation_semantic_sha
        and payload.get("trial_count") == 36
        and payload.get("declarations_sha256") == digest
        and payload.get("objective_values_read") is False
        and payload.get("objective_values_compared") is False
        and payload.get("objective_values_emitted") is False
        and payload.get("final_connectivity_accessed") is False,
        "transition declaration identity or outcome firewall changed",
    )
    return digest


def _verify_probe_pair(
    *, episode_root: Path, executor_root: Path, comparison: Mapping[str, Any]
) -> dict[str, str]:
    probe = _probe_module()
    probe_spec = executor_root / "CONTRACT_TRANSITION_PROBE_SPEC.json"
    evidence: dict[str, str] = {}
    for phase in ("legacy", "post36"):
        reference = _mapping(comparison.get(phase), f"{phase} probe reference")
        path = Path(str(reference.get("path", "")))
        path = _secure_path(
            path,
            boundary=episode_root / "outputs",
            label=f"{phase} probe receipt",
        )
        encoded = _read_bytes(
            path,
            boundary=episode_root / "outputs",
            label=f"{phase} probe receipt",
        )
        record = probe.verify_artifact(artifact_path=path, spec_path=probe_spec)
        _require(record.get("phase") == phase, f"{phase} probe phase mismatch")
        _require(
            reference.get("file_sha256") == _file_sha(encoded)
            and reference.get("artifact_sha256") == record.get("artifact_sha256"),
            f"{phase} probe reference does not bind its receipt",
        )
        evidence[f"{phase}_probe_file_sha256"] = _file_sha(encoded)
        evidence[f"{phase}_probe_artifact_sha256"] = _sha(
            record.get("artifact_sha256"), f"{phase} probe artifact sha256"
        )
        evidence[f"{phase}_probe_scientific_payload_sha256"] = _sha(
            record.get("scientific_payload_sha256"),
            f"{phase} probe scientific payload sha256",
        )
    _require(
        evidence["legacy_probe_scientific_payload_sha256"]
        == evidence["post36_probe_scientific_payload_sha256"],
        "legacy/post36 probe scientific payloads differ",
    )
    checks = _mapping(comparison.get("checks"), "transition probe checks")
    additions = checks.get("diagnostic_additions")
    _require(
        checks.get("input_payload_byte_identical") is True
        and checks.get("environment_byte_identical") is True
        and checks.get("scientific_payload_byte_identical") is True
        and checks.get("diagnostic_changes_limited_to_added_representation_rank") is True
        and isinstance(additions, list),
        "transition probe comparison did not establish the allowed-only delta",
    )
    evidence["allowed_diagnostics_delta_sha256"] = _canonical_digest(additions)
    return evidence


def verify_scientific_contract_transition(
    *,
    episode_root: Path,
    executor_root: Path,
    manifest_path: Path | None = None,
    expected_manifest_file_sha256: str | None = None,
    expected_transition_receipt_sha256: str | None = None,
    observed_attestation_dir: Path | None = None,
    observed_finalization_dir: Path | None = None,
) -> dict[str, Any]:
    """Recompute every narrowly scoped binding in the canonical bundle."""

    root = _lexical(episode_root)
    executor = _secure_path(
        _lexical(executor_root),
        boundary=_lexical(executor_root),
        label="frozen executor root",
        require_file=False,
    )
    transition_root = root / TRANSITION_ROOT_RELATIVE
    expected_manifest = transition_root / TRANSITION_MANIFEST_NAME
    supplied_manifest = expected_manifest if manifest_path is None else _lexical(manifest_path)
    _require(supplied_manifest == expected_manifest, "transition manifest path is not canonical")
    manifest_bytes = _read_bytes(
        supplied_manifest, boundary=root, label="scientific transition manifest"
    )
    manifest_file_sha = _file_sha(manifest_bytes)
    if expected_manifest_file_sha256 is not None:
        _require(
            manifest_file_sha
            == _sha(expected_manifest_file_sha256, "expected transition manifest file sha256"),
            "scientific transition manifest file hash mismatch",
        )
    manifest = _json_from_bytes(manifest_bytes, "scientific transition manifest")
    _require(set(manifest) == _MANIFEST_FIELDS, "scientific transition manifest schema changed")
    _require(
        manifest.get("schema_version") == TRANSITION_MANIFEST_SCHEMA
        and manifest.get("transition_id") == TRANSITION_ID
        and manifest.get("observed_trial_count") == 36
        and manifest.get("source_trial_reports_reverified") is True
        and manifest.get("operator_development_outcome_values_displayed") is True
        and manifest.get(
            "programmatic_transition_extraction_outcome_fields_accessed"
        )
        is False
        and manifest.get("values_used_for_transition_decisions") is False
        and manifest.get("final_connectivity_accessed") is False
        and manifest.get("complete") is True,
        "scientific transition manifest safety or completion fields changed",
    )
    files = _mapping(manifest.get("files"), "scientific transition files")
    _require(set(files) == _TRANSITION_FILES, "scientific transition file set changed")
    encoded_files: dict[str, bytes] = {}
    for name in sorted(_TRANSITION_FILES):
        encoded = _read_bytes(
            transition_root / name,
            boundary=root,
            label=f"scientific transition {name}",
        )
        _require(_file_sha(encoded) == _sha(files.get(name), f"{name} file sha256"), f"scientific transition file hash mismatch: {name}")
        encoded_files[name] = encoded

    transition = _transition_module()
    receipt = _json_from_bytes(encoded_files[TRANSITION_RECEIPT_NAME], "scientific transition receipt")
    transition.verify_transition_receipt(receipt)
    receipt_semantic_sha = _sha(
        receipt.get("transition_receipt_sha256"), "transition receipt semantic sha256"
    )
    if expected_transition_receipt_sha256 is not None:
        _require(
            receipt_semantic_sha
            == _sha(expected_transition_receipt_sha256, "expected transition receipt sha256"),
            "scientific transition receipt semantic hash mismatch",
        )
    definition = transition.verify_contract_transition_definition(
        executor_root=executor,
        transition_spec_path=executor / "CONTRACT_TRANSITION_SPEC.json",
    )
    spec = _json_from_bytes(
        _read_bytes(
            executor / "CONTRACT_TRANSITION_SPEC.json",
            boundary=executor,
            label="transition specification",
        ),
        "transition specification",
    )
    implementation = _mapping(spec.get("implementation_lineage"), "transition implementation lineage")
    model_lineage = _mapping(implementation.get("scientific_models"), "scientific-model lineage")
    observed_model_sha = _sha(model_lineage.get("old_sha256"), "observed scientific-model sha256")
    active_model_sha = _sha(model_lineage.get("new_sha256"), "active scientific-model sha256")
    active_model_bytes = _read_bytes(
        executor / str(model_lineage.get("path")),
        boundary=executor,
        label="active scientific-model implementation",
    )
    _require(_file_sha(active_model_bytes) == active_model_sha, "active scientific-model implementation hash mismatch")
    exposure_evidence = _verify_operator_exposure_incident(
        episode_root=root, transition_spec=spec
    )

    attestation_dir = observed_attestation_dir or (root / OBSERVED_ATTESTATION_RELATIVE)
    attestation_manifest, attestation_manifest_file_sha, attestation, attestation_artifact_file_sha = _verify_attestation_manifest(
        episode_root=root, attestation_dir=attestation_dir
    )
    attested_by_id = transition._verify_attestation(attestation)
    attestation_semantic_sha = _sha(
        attestation.get("attestation_sha256"), "observed attestation semantic sha256"
    )
    published_authority = _mapping(
        receipt.get("observed_attestation_authority"),
        "published observed-attestation authority",
    )
    current_authority = _verify_dynamic_attestation_authority(
        episode_root=root,
        transition_spec=spec,
        terminal_identity_path=Path(
            str(published_authority.get("replacement_terminal_identity_path"))
        ),
        start_authorization_path=Path(
            str(published_authority.get("post36_start_authorization_path"))
        ),
        cancellation_receipt_path=Path(
            str(published_authority.get("obsolete_cancellation_receipt_path"))
        ),
        attestation_manifest_file_sha256=attestation_manifest_file_sha,
        attestation_artifact_file_sha256=attestation_artifact_file_sha,
        attestation_sha256=attestation_semantic_sha,
    )
    attestation_authority = _verify_published_attestation_authority(
        published_authority, current=current_authority
    )
    declarations = _json_from_bytes(
        encoded_files[REPORT_DECLARATIONS_NAME], "observed report contract declarations"
    )
    declaration_sha = _verify_declarations(
        payload=declarations,
        attested_by_id=attested_by_id,
        attestation_manifest_sha=attestation_manifest_file_sha,
        attestation_semantic_sha=attestation_semantic_sha,
        transition=transition,
    )
    authenticated_report_bytes: dict[str, bytes] = {}
    observed_run_root = root / "outputs" / "development_run002"
    for trial_id, attested in sorted(attested_by_id.items()):
        relative_trial_dir = attested.get("source_trial_directory")
        _require(
            isinstance(relative_trial_dir, str)
            and relative_trial_dir
            and not Path(relative_trial_dir).is_absolute()
            and ".." not in Path(relative_trial_dir).parts,
            f"attestation has an unsafe source trial directory for {trial_id}",
        )
        authenticated_report_bytes[trial_id] = _read_bytes(
            observed_run_root / relative_trial_dir / "trial_report.json",
            boundary=observed_run_root,
            label=f"authenticated observed report {trial_id}",
        )
    recomputed_declarations = (
        transition.extract_authenticated_report_contract_declarations(
            attested_by_id=attested_by_id,
            authenticated_trial_report_bytes=authenticated_report_bytes,
        )
    )
    _require(
        recomputed_declarations == declarations["rows"],
        "published transition declarations differ from the 36 authenticated report bytes",
    )
    archive_sha, limitation_manifest_sha = _verify_observed_prefix_archive(
        episode_root=root,
        finalization_dir=observed_finalization_dir
        or (root / OBSERVED_FINALIZATION_RELATIVE),
    )

    invariance_path = transition_root / SYNTHETIC_INVARIANCE_NAME
    comparison = _json_from_bytes(
        encoded_files[SYNTHETIC_INVARIANCE_NAME], "synthetic transition invariance"
    )
    probe = _probe_module()
    verified_comparison = probe.verify_artifact(
        artifact_path=invariance_path,
        spec_path=executor / "CONTRACT_TRANSITION_PROBE_SPEC.json",
    )
    _require(dict(comparison) == dict(verified_comparison), "synthetic transition comparison changed during verification")
    probe_evidence = _verify_probe_pair(
        episode_root=root, executor_root=executor, comparison=comparison
    )

    expected_bindings = {
        "transition_receipt_sha256": receipt_semantic_sha,
        "transition_spec_sha256": definition["transition_spec_sha256"],
        "observed_prefix_contract_sha256": definition["old_contract_sha256"],
        "active_contract_sha256": definition["new_contract_sha256"],
        "observed_prefix_qualification_spec_sha256": definition["old_qualification_spec_sha256"],
        "active_qualification_spec_sha256": definition["new_qualification_spec_sha256"],
        "observed_prefix_scientific_models_sha256": observed_model_sha,
        "active_scientific_models_sha256": active_model_sha,
        "observed_attestation_manifest_file_sha256": attestation_manifest_file_sha,
        "observed_attestation_artifact_file_sha256": attestation_artifact_file_sha,
        "observed_attestation_sha256": attestation_semantic_sha,
        "observed_attestation_authority_sha256": attestation_authority[
            "authority_sha256"
        ],
        "authoritative_attestation_job_id": attestation_authority[
            "authoritative_job_id"
        ],
        "obsolete_attestation_job_id": attestation_authority["obsolete_job_id"],
        "source_archive_receipt_file_sha256": attestation_authority[
            "source_archive"
        ]["archive_receipt_file_sha256"],
        "source_archive_receipt_sha256": attestation_authority["source_archive"][
            "archive_receipt_sha256"
        ],
        "source_archive_manifest_file_sha256": attestation_authority[
            "source_archive"
        ]["manifest_file_sha256"],
        "source_archive_manifest_sha256": attestation_authority[
            "source_archive"
        ]["manifest_sha256"],
        "source_archive_tree_sha256": attestation_authority["source_archive"][
            "tree_sha256"
        ],
        "controller_final_receipt_status": attestation_authority[
            "controller_final_receipt"
        ]["status"],
        "observed_report_contract_declarations_sha256": declaration_sha,
        "observed_prefix_archive_file_sha256": archive_sha,
        "observed_prefix_limitation_manifest_file_sha256": limitation_manifest_sha,
        "operator_exposure_incident_file_sha256": exposure_evidence[
            "operator_exposure_incident_file_sha256"
        ],
        "operator_exposure_incident_semantic_sha256": exposure_evidence[
            "operator_exposure_incident_semantic_sha256"
        ],
        **probe_evidence,
    }
    for field, expected in expected_bindings.items():
        _require(manifest.get(field) == expected, f"transition manifest binding mismatch: {field}")

    receipt_bindings = {
        "old_contract_sha256": expected_bindings["observed_prefix_contract_sha256"],
        "active_contract_sha256": expected_bindings["active_contract_sha256"],
        "old_qualification_spec_sha256": expected_bindings[
            "observed_prefix_qualification_spec_sha256"
        ],
        "active_qualification_spec_sha256": expected_bindings[
            "active_qualification_spec_sha256"
        ],
        "old_scientific_models_sha256": observed_model_sha,
        "active_scientific_models_sha256": active_model_sha,
        "observed_attestation_manifest_sha256": attestation_manifest_file_sha,
        "observed_attestation_sha256": attestation_semantic_sha,
        "observed_attestation_authority": attestation_authority,
        "observed_report_contract_declarations_sha256": declaration_sha,
        "operator_exposure_incident_file_sha256": expected_bindings[
            "operator_exposure_incident_file_sha256"
        ],
        "operator_exposure_incident_semantic_sha256": expected_bindings[
            "operator_exposure_incident_semantic_sha256"
        ],
    }
    for field, expected in receipt_bindings.items():
        _require(receipt.get(field) == expected, f"transition receipt binding mismatch: {field}")
    _require(
        receipt.get("synthetic_invariance_evidence_sha256")
        == comparison.get("artifact_sha256"),
        "transition receipt does not bind the synthetic comparison semantic hash",
    )
    _require(
        manifest.get("observed_prefix_contract_sha256")
        == OBSERVED_PREFIX_CONTRACT_SHA256
        and manifest.get("observed_prefix_qualification_spec_sha256")
        == OBSERVED_PREFIX_QUALIFICATION_SPEC_SHA256,
        "transition observed-prefix identity is not immutable v1",
    )
    for value in manifest.values():
        if isinstance(value, float):
            _require(math.isfinite(value), "transition manifest contains a nonfinite value")

    exact_set_sha = _canonical_digest(
        [
            {
                "trial_id": row["trial_id"],
                "source_trial_report_sha256": row["source_trial_report_sha256"],
            }
            for row in declarations["rows"]
        ]
    )
    return {
        "transition_id": TRANSITION_ID,
        "transition_manifest_path": str(supplied_manifest),
        "transition_manifest_file_sha256": manifest_file_sha,
        "transition_receipt_path": str(transition_root / TRANSITION_RECEIPT_NAME),
        "transition_receipt_file_sha256": _file_sha(
            encoded_files[TRANSITION_RECEIPT_NAME]
        ),
        "transition_receipt_sha256": receipt_semantic_sha,
        "transition_spec_sha256": definition["transition_spec_sha256"],
        **expected_bindings,
        "observed_attestation_authority": attestation_authority,
        "source_archive": dict(attestation_authority["source_archive"]),
        "controller_final_receipt_required_before_irreversible_transition": True,
        "observed_trial_count": 36,
        "exact_attested_trial_report_set_sha256": exact_set_sha,
        "exact_36_attestation_set_matches_transition_evidence": True,
        "source_trial_reports_reverified": True,
        "operator_development_outcome_values_displayed": True,
        "programmatic_transition_extraction_outcome_fields_accessed": False,
        "values_used_for_transition_decisions": False,
        "final_connectivity_accessed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ep12-scientific-transition-binding",
        description="Verify the canonical outcome-blind EP12 post-36 transition bundle",
    )
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--executor-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-file-sha256", required=True)
    parser.add_argument("--expected-transition-receipt-file-sha256", required=True)
    parser.add_argument("--expected-transition-receipt-sha256", required=True)
    parser.add_argument("--expected-observed-prefix-contract-sha256", required=True)
    parser.add_argument("--expected-active-contract-sha256", required=True)
    parser.add_argument("--expected-active-qualification-spec-sha256", required=True)
    parser.add_argument("--expected-active-scientific-models-sha256", required=True)
    arguments = parser.parse_args(argv)
    evidence = verify_scientific_contract_transition(
        episode_root=arguments.episode_root,
        executor_root=arguments.executor_root,
        manifest_path=arguments.manifest,
        expected_manifest_file_sha256=arguments.expected_manifest_file_sha256,
        expected_transition_receipt_sha256=arguments.expected_transition_receipt_sha256,
    )
    expected = {
        "transition_receipt_file_sha256": arguments.expected_transition_receipt_file_sha256,
        "observed_prefix_contract_sha256": arguments.expected_observed_prefix_contract_sha256,
        "active_contract_sha256": arguments.expected_active_contract_sha256,
        "active_qualification_spec_sha256": arguments.expected_active_qualification_spec_sha256,
        "active_scientific_models_sha256": arguments.expected_active_scientific_models_sha256,
    }
    for field, value in expected.items():
        _require(
            evidence.get(field) == _sha(value, f"expected {field}"),
            f"wrapper-bound transition identity mismatch: {field}",
        )
    print(json.dumps(evidence, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
