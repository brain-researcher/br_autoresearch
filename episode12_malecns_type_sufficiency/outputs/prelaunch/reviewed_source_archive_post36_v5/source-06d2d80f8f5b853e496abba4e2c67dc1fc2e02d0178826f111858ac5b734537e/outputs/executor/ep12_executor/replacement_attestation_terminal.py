"""Publish and verify the archive-staged replacement attestation terminal identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

from .attest_observed_trials import (
    build_observed_attestation,
    verify_attestation_root,
    write_attestation_artifacts,
)
from .policy import EpisodePolicy, canonical_json, digest_object
from .post36_orchestration import _effective_user, _read_json, _sha256_file, _write_or_verify
from .replacement_attestation_launch import (
    BOOTSTRAP_STAGE_RESULT_RELATIVE,
    INTENT_RELATIVE,
    STAGE_RECEIPT_RELATIVE,
    TERMINAL_IDENTITY_RELATIVE,
    verify_replacement_attestation_receipt,
)
from .source_archive import verify_source_archive
from .source_archive_bootstrap import (
    parse_manifest as parse_bootstrap_manifest,
    read_regular as read_bootstrap_regular,
    verify_staged as verify_bootstrap_staged,
)
from .synthetic import EPISODE_ROOT


TERMINAL_SCHEMA = "ep12.replacement_attestation_terminal_identity.v4"
_STAGE_RESULT_SCHEMA = "ep12.source_archive_bootstrap_stage.v1"
_ALLOWED_STAGE_STATUS = {"created", "recovered_identical", "reused_identical"}
_REQUIRED_IMPORT_MODULES = frozenset(
    {
    "ep12_executor.attest_observed_trials",
    "ep12_executor.policy",
    "ep12_executor.post36_orchestration",
    "ep12_executor.replacement_attestation_launch",
    "ep12_executor.replacement_attestation_terminal",
    "ep12_executor.source_archive",
    "ep12_executor.source_archive_bootstrap",
    "ep12_executor.synthetic",
    }
)

_TERMINAL_IDENTITY_FIELDS = frozenset(
    {
        "schema_version", "submission_user", "slurm_job_id", "slurm_job_name",
        "dependency", "scheduler_dependency", "logical_dependency_satisfied",
        "finalizer_terminal_completion", "launch_intent_path",
        "launch_intent_file_sha256",
        "launch_intent_sha256", "launch_receipt_path",
        "launch_receipt_file_sha256", "launch_receipt_sha256",
        "stage_intent_path", "stage_intent_file_sha256",
        "stage_intent_sha256", "stage_receipt_path",
        "stage_receipt_file_sha256", "stage_receipt_sha256",
        "release_intent_path", "release_intent_file_sha256",
        "release_intent_sha256", "release_authorization_path",
        "release_authorization_file_sha256", "release_authorization_sha256",
        "bootstrap_stage_result_path",
        "bootstrap_stage_result_file_sha256", "staged_root",
        "staged_executor_root", "executed_wrapper_path", "stage_arguments",
        "spooled_batch_script_sha256",
        "spooled_script_matches_archived_wrapper", "source_archive",
        "imported_module_provenance", "all_ep12_imports_under_staged_executor",
        "attestation_dir", "attestation_manifest_file_sha256",
        "attestation_artifact_file_sha256", "attestation_sha256",
        "verified_trial_count", "source_trial_files_modified",
        "objective_values_read", "objective_values_compared",
        "objective_values_emitted", "superseded_job_id",
        "superseded_result_authoritative",
        "authoritative_attestation_job_identity",
        "attestation_result_authoritative", "real_connectivity_accessed",
        "final_connectivity_accessed", "final_connectivity_access_authorized",
        "identity_sha256",
    }
)


class ReplacementAttestationTerminalError(RuntimeError):
    """Replacement terminal evidence is unsafe or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplacementAttestationTerminalError(message)


def _read_regular(path: Path, *, label: str) -> bytes:
    candidate = Path(os.path.abspath(os.fspath(path)))
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        metadata = os.lstat(current)
        _require(not stat.S_ISLNK(metadata.st_mode), f"{label} traverses a symlink")
    descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        _require(stat.S_ISREG(os.fstat(descriptor).st_mode), f"{label} is not regular")
        parts: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                return b"".join(parts)
            parts.append(block)
    finally:
        os.close(descriptor)


def _sha(path: Path, *, label: str) -> str:
    return hashlib.sha256(_read_regular(path, label=label)).hexdigest()


def _bootstrap_stage_result(
    *,
    episode_root: Path,
    path: Path,
    expected_file_sha256: str,
    expected_staged_root: Path,
    archive: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_path = episode_root / BOOTSTRAP_STAGE_RESULT_RELATIVE
    _require(
        Path(os.path.abspath(os.fspath(path))) == canonical_path,
        "bootstrap stage-result path is not canonical",
    )
    encoded = _read_regular(path, label="bootstrap stage result")
    _require(
        hashlib.sha256(encoded).hexdigest() == expected_file_sha256,
        "bootstrap stage-result file hash changed",
    )
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplacementAttestationTerminalError("bootstrap stage result is malformed") from exc
    _require(isinstance(value, Mapping), "bootstrap stage result is not an object")
    result = dict(value)
    _require(
        encoded == (canonical_json(result) + "\n").encode("ascii"),
        "bootstrap stage result is not canonical JSON",
    )
    _require(
        set(result)
        == {
            "schema_version", "stage_status", "staged_root",
            "upstream_archived_root", "archive_receipt_file_sha256",
            "archive_receipt_sha256", "manifest_file_sha256",
            "manifest_sha256", "tree_sha256", "complete",
        }
        and result["schema_version"] == _STAGE_RESULT_SCHEMA
        and result["stage_status"] in _ALLOWED_STAGE_STATUS
        and result["staged_root"] == str(expected_staged_root)
        and result["upstream_archived_root"] == archive["archived_root"]
        and result["archive_receipt_file_sha256"]
        == archive["archive_receipt_file_sha256"]
        and result["archive_receipt_sha256"] == archive["archive_receipt_sha256"]
        and result["manifest_file_sha256"] == archive["manifest_file_sha256"]
        and result["manifest_sha256"] == archive["manifest_sha256"]
        and result["tree_sha256"] == archive["tree_sha256"]
        and result["complete"] is True,
        "bootstrap stage result identity changed",
    )
    return result


def _verify_staged_archive(
    *, staged_root: Path, archive: Mapping[str, Any]
) -> dict[str, Any]:
    original_root = Path(str(archive["archived_root"]))
    try:
        verified = verify_source_archive(
            original_root,
            expected_receipt_file_sha256=str(archive["archive_receipt_file_sha256"]),
            expected_receipt_sha256=str(archive["archive_receipt_sha256"]),
            expected_manifest_file_sha256=str(archive["manifest_file_sha256"]),
            expected_manifest_sha256=str(archive["manifest_sha256"]),
            expected_tree_sha256=str(archive["tree_sha256"]),
        )
    except Exception as exc:
        raise ReplacementAttestationTerminalError(str(exc)) from exc
    _require(
        Path(str(verified["archived_root"])) == original_root,
        "source archive verifier returned another publication root",
    )
    try:
        manifest_bytes, _metadata = read_bootstrap_regular(
            Path(str(archive["manifest_path"])), label="source archive manifest"
        )
        manifest, directories, files = parse_bootstrap_manifest(manifest_bytes)
        _require(
            manifest["manifest_sha256"] == archive["manifest_sha256"]
            and manifest["tree_sha256"] == archive["tree_sha256"],
            "source archive manifest identity changed",
        )
        verify_bootstrap_staged(
            staged_root, manifest_bytes, directories, files
        )
    except Exception as exc:
        raise ReplacementAttestationTerminalError(
            f"staged archive differs from the authenticated manifest: {exc}"
        ) from exc
    return dict(verified)


def _import_provenance(staged_executor: Path) -> list[dict[str, str]]:
    """Record every loaded EP12 package module, not a handpicked subset."""

    records: list[dict[str, str]] = []
    boundary = staged_executor.resolve(strict=True)
    loaded = sorted(
        (name, module)
        for name, module in sys.modules.items()
        if name == "ep12_executor" or name.startswith("ep12_executor.")
    )
    _require(bool(loaded), "no EP12 package modules are loaded")
    for module_name, module in loaded:
        raw = getattr(module, "__file__", None)
        _require(isinstance(raw, str), f"loaded EP12 module has no file: {module_name}")
        path = Path(raw).resolve(strict=True)
        try:
            relative = path.relative_to(boundary)
        except ValueError as exc:
            raise ReplacementAttestationTerminalError(
                f"module imported outside staged executor: {module_name}"
            ) from exc
        records.append(
            {
                "module": module_name,
                "relative_path": relative.as_posix(),
                "file_sha256": _sha(path, label=f"imported module {module_name}"),
            }
        )
    return records


def _expected_wrapper_arguments(
    *,
    episode_root: Path,
    launch: Mapping[str, Any],
    bootstrap_stage_result: Path,
    staged_root: Path,
    stage_receipt_path: Path,
) -> list[str]:
    archive = launch["archive"]
    return [
        str(episode_root),
        str(archive["archived_root"]),
        str(archive["archive_receipt_file_sha256"]),
        str(archive["archive_receipt_sha256"]),
        str(archive["manifest_file_sha256"]),
        str(archive["manifest_sha256"]),
        str(archive["tree_sha256"]),
        str(episode_root / INTENT_RELATIVE),
        str(launch["launch_intent_file_sha256"]),
        str(launch["launch_intent_sha256"]),
        str(launch["attestation_output"]),
        str(bootstrap_stage_result),
        str(staged_root),
        str(stage_receipt_path),
        str(launch["terminal_identity_path"]),
    ]


def publish_terminal_identity(
    *,
    episode_root: Path,
    launch_intent_path: Path,
    launch_intent_file_sha256: str,
    launch_intent_sha256: str,
    attestation_output: Path,
    terminal_identity_output: Path,
    bootstrap_stage_result: Path,
    bootstrap_stage_result_file_sha256: str,
    staged_root: Path,
    stage_receipt_path: Path,
    executed_wrapper: Path,
    slurm_job_id: str,
    slurm_job_name: str,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(episode_root)))
    _require(root == EPISODE_ROOT, "terminal episode root is not canonical")
    launch = verify_replacement_attestation_receipt(
        episode_root=root, require_released=None
    )
    _require(
        launch["release_authorized"] is True,
        "terminal execution lacks durable pre-release authorization",
    )
    _require(
        Path(os.path.abspath(os.fspath(launch_intent_path))) == root / INTENT_RELATIVE
        and launch_intent_file_sha256 == launch["launch_intent_file_sha256"]
        and launch_intent_sha256 == launch["launch_intent_sha256"],
        "terminal launch-intent arguments changed",
    )
    _require(
        Path(os.path.abspath(os.fspath(attestation_output)))
        == Path(str(launch["attestation_output"]))
        and Path(os.path.abspath(os.fspath(terminal_identity_output)))
        == root / TERMINAL_IDENTITY_RELATIVE
        == Path(str(launch["terminal_identity_path"])),
        "terminal durable output arguments changed",
    )
    _require(
        Path(os.path.abspath(os.fspath(stage_receipt_path)))
        == root / STAGE_RECEIPT_RELATIVE
        == Path(str(launch["stage_receipt_path"])),
        "terminal stage-receipt path changed",
    )
    _require(
        slurm_job_id == launch["slurm_job_id"]
        and slurm_job_name == launch["slurm_job_name"],
        "running Slurm identity differs from the durable launch receipt",
    )
    _require(
        list(launch["arguments"])
        == _expected_wrapper_arguments(
            episode_root=root,
            launch=launch,
            bootstrap_stage_result=bootstrap_stage_result,
            staged_root=staged_root,
            stage_receipt_path=stage_receipt_path,
        ),
        "running wrapper arguments differ from the stage intent",
    )
    stage_receipt_file_sha256 = _sha(
        stage_receipt_path, label="replacement stage receipt"
    )
    _require(
        stage_receipt_file_sha256 == launch["stage_receipt_file_sha256"],
        "terminal stage-receipt file hash changed",
    )
    wrapper_sha256 = _sha(executed_wrapper, label="executed Slurm wrapper")
    _require(
        wrapper_sha256 == launch["wrapper_sha256"],
        "spooled replacement wrapper differs from reviewed archive wrapper",
    )
    archive = launch["archive"]
    _bootstrap_stage_result(
        episode_root=root,
        path=bootstrap_stage_result,
        expected_file_sha256=bootstrap_stage_result_file_sha256,
        expected_staged_root=staged_root,
        archive=archive,
    )
    _verify_staged_archive(staged_root=staged_root, archive=archive)
    staged_executor = staged_root / "outputs/executor"
    policy = EpisodePolicy.load(staged_root / "SEARCH_POLICY.yaml")
    verified_attestation = verify_attestation_root(
        run_root=root / "outputs/development_run002",
        attestation_dir=attestation_output,
        policy=policy,
        outputs_root=root / "outputs",
    )
    _require(
        verified_attestation.get("verified_trial_count") == 36
        and verified_attestation.get("source_trial_files_modified") is False
        and verified_attestation.get("objective_values_read") is False
        and verified_attestation.get("objective_values_compared") is False
        and verified_attestation.get("objective_values_emitted") is False
        and verified_attestation.get("final_connectivity_accessed") is False,
        "replacement attestation safety verification failed",
    )
    imports = _import_provenance(staged_executor)
    _require(
        _REQUIRED_IMPORT_MODULES.issubset(
            {record["module"] for record in imports}
        ),
        "replacement terminal process lacks required canonical module imports",
    )
    _verify_staged_archive(staged_root=staged_root, archive=archive)
    core = {
        "schema_version": TERMINAL_SCHEMA,
        "submission_user": _effective_user(),
        "slurm_job_id": slurm_job_id,
        "slurm_job_name": slurm_job_name,
        "dependency": launch["dependency"],
        "scheduler_dependency": launch["scheduler_dependency"],
        "logical_dependency_satisfied": launch[
            "logical_dependency_satisfied"
        ],
        "finalizer_terminal_completion": dict(
            launch["finalizer_terminal_completion"]
        ),
        "launch_intent_path": str(launch_intent_path),
        "launch_intent_file_sha256": launch_intent_file_sha256,
        "launch_intent_sha256": launch_intent_sha256,
        "launch_receipt_path": launch["launch_receipt_path"],
        "launch_receipt_file_sha256": launch["launch_receipt_file_sha256"],
        "launch_receipt_sha256": launch["launch_receipt_sha256"],
        "stage_intent_path": launch["stage_intent_path"],
        "stage_intent_file_sha256": launch["stage_intent_file_sha256"],
        "stage_intent_sha256": launch["stage_intent_sha256"],
        "stage_receipt_path": str(stage_receipt_path),
        "stage_receipt_file_sha256": stage_receipt_file_sha256,
        "stage_receipt_sha256": launch["stage_receipt_sha256"],
        "release_intent_path": launch["release_intent_path"],
        "release_intent_file_sha256": launch["release_intent_file_sha256"],
        "release_intent_sha256": launch["release_intent_sha256"],
        "release_authorization_path": launch["release_authorization_path"],
        "release_authorization_file_sha256": launch[
            "release_authorization_file_sha256"
        ],
        "release_authorization_sha256": launch[
            "release_authorization_sha256"
        ],
        "bootstrap_stage_result_path": str(bootstrap_stage_result),
        "bootstrap_stage_result_file_sha256": bootstrap_stage_result_file_sha256,
        "staged_root": str(staged_root),
        "staged_executor_root": str(staged_executor),
        "executed_wrapper_path": str(
            Path(os.path.abspath(os.fspath(executed_wrapper)))
        ),
        "stage_arguments": list(launch["arguments"]),
        "spooled_batch_script_sha256": wrapper_sha256,
        "spooled_script_matches_archived_wrapper": True,
        "source_archive": dict(archive),
        "imported_module_provenance": imports,
        "all_ep12_imports_under_staged_executor": True,
        "attestation_dir": str(attestation_output),
        "attestation_manifest_file_sha256": verified_attestation[
            "attestation_manifest_sha256"
        ],
        "attestation_artifact_file_sha256": verified_attestation[
            "attestation_artifact_file_sha256"
        ],
        "attestation_sha256": verified_attestation["attestation_sha256"],
        "verified_trial_count": 36,
        "source_trial_files_modified": verified_attestation[
            "source_trial_files_modified"
        ],
        "objective_values_read": False,
        "objective_values_compared": False,
        "objective_values_emitted": False,
        "superseded_job_id": launch["superseded_job_id"],
        "superseded_result_authoritative": False,
        "authoritative_attestation_job_identity": True,
        "attestation_result_authoritative": True,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    value = {**core, "identity_sha256": digest_object(core)}
    return _write_or_verify(
        terminal_identity_output,
        value,
        root=root,
        label="replacement attestation terminal identity",
    )


def attest_and_publish_terminal_identity(**kwargs: Any) -> dict[str, Any]:
    """Build, write, verify, and bind the status-only attestation in one process."""

    root = Path(os.path.abspath(os.fspath(kwargs["episode_root"])))
    staged_root = Path(os.path.abspath(os.fspath(kwargs["staged_root"])))
    attestation_output = Path(
        os.path.abspath(os.fspath(kwargs["attestation_output"]))
    )
    # Authenticate durable R release before reading any trial status metadata.
    launch = verify_replacement_attestation_receipt(
        episode_root=root, require_released=None
    )
    _require(
        launch["release_authorized"] is True,
        "attestation execution lacks durable pre-release authorization",
    )
    _bootstrap_stage_result(
        episode_root=root,
        path=Path(os.path.abspath(os.fspath(kwargs["bootstrap_stage_result"]))),
        expected_file_sha256=str(kwargs["bootstrap_stage_result_file_sha256"]),
        expected_staged_root=staged_root,
        archive=launch["archive"],
    )
    _verify_staged_archive(staged_root=staged_root, archive=launch["archive"])
    policy = EpisodePolicy.load(staged_root / "SEARCH_POLICY.yaml")
    run_root = root / "outputs/development_run002"
    attestation = build_observed_attestation(
        run_root=run_root,
        policy=policy,
        outputs_root=root / "outputs",
    )
    write_attestation_artifacts(
        run_root=run_root,
        output_dir=attestation_output,
        attestation=attestation,
        outputs_root=root / "outputs",
    )
    return publish_terminal_identity(**kwargs)


def verify_replacement_attestation_terminal_identity(
    *, episode_root: Path, identity_path: Path | None = None
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(episode_root)))
    launch = verify_replacement_attestation_receipt(
        episode_root=root, require_released=True
    )
    path = identity_path or (root / TERMINAL_IDENTITY_RELATIVE)
    _require(
        Path(os.path.abspath(os.fspath(path))) == root / TERMINAL_IDENTITY_RELATIVE,
        "replacement terminal identity path is not canonical",
    )
    identity = _read_json(
        path, root=root,
        label="replacement attestation terminal identity", canonical=True,
    )
    core = {key: value for key, value in identity.items() if key != "identity_sha256"}
    _require(
        set(identity) == _TERMINAL_IDENTITY_FIELDS
        and identity.get("schema_version") == TERMINAL_SCHEMA
        and identity.get("identity_sha256") == digest_object(core)
        and identity.get("submission_user") == _effective_user()
        and identity.get("slurm_job_id") == launch["slurm_job_id"]
        and identity.get("slurm_job_name") == launch["slurm_job_name"]
        and identity.get("dependency") == launch["dependency"]
        and identity.get("scheduler_dependency") is None
        and identity.get("scheduler_dependency")
        == launch["scheduler_dependency"]
        and identity.get("logical_dependency_satisfied") is True
        and identity.get("logical_dependency_satisfied")
        == launch["logical_dependency_satisfied"]
        and identity.get("finalizer_terminal_completion")
        == launch["finalizer_terminal_completion"]
        and identity.get("launch_intent_path") == launch["launch_intent_path"]
        and identity.get("launch_intent_file_sha256")
        == launch["launch_intent_file_sha256"]
        and identity.get("launch_intent_sha256") == launch["launch_intent_sha256"]
        and identity.get("launch_receipt_path") == launch["launch_receipt_path"]
        and identity.get("launch_receipt_file_sha256")
        == launch["launch_receipt_file_sha256"]
        and identity.get("launch_receipt_sha256") == launch["launch_receipt_sha256"]
        and identity.get("stage_intent_path") == launch["stage_intent_path"]
        and identity.get("stage_intent_file_sha256")
        == launch["stage_intent_file_sha256"]
        and identity.get("stage_intent_sha256") == launch["stage_intent_sha256"]
        and identity.get("stage_receipt_path") == launch["stage_receipt_path"]
        and identity.get("stage_receipt_file_sha256")
        == launch["stage_receipt_file_sha256"]
        and identity.get("stage_receipt_sha256") == launch["stage_receipt_sha256"]
        and identity.get("release_intent_path") == launch["release_intent_path"]
        and identity.get("release_intent_file_sha256")
        == launch["release_intent_file_sha256"]
        and identity.get("release_intent_sha256") == launch["release_intent_sha256"]
        and identity.get("release_authorization_path")
        == launch["release_authorization_path"]
        and identity.get("release_authorization_file_sha256")
        == launch["release_authorization_file_sha256"]
        and identity.get("release_authorization_sha256")
        == launch["release_authorization_sha256"]
        and identity.get("bootstrap_stage_result_path")
        == str(root / BOOTSTRAP_STAGE_RESULT_RELATIVE)
        and identity.get("stage_arguments") == launch["arguments"]
        and identity.get("spooled_batch_script_sha256") == launch["wrapper_sha256"]
        and identity.get("spooled_script_matches_archived_wrapper") is True
        and identity.get("source_archive") == launch["archive"]
        and identity.get("all_ep12_imports_under_staged_executor") is True
        and identity.get("attestation_dir") == launch["attestation_output"]
        and identity.get("verified_trial_count") == 36
        and identity.get("source_trial_files_modified") is False
        and identity.get("superseded_job_id") == launch["superseded_job_id"]
        and identity.get("superseded_result_authoritative") is False
        and identity.get("authoritative_attestation_job_identity") is True
        and identity.get("attestation_result_authoritative") is True
        and identity.get("objective_values_read") is False
        and identity.get("objective_values_compared") is False
        and identity.get("objective_values_emitted") is False
        and identity.get("real_connectivity_accessed") is False
        and identity.get("final_connectivity_accessed") is False
        and identity.get("final_connectivity_access_authorized") is False,
        "replacement terminal identity fields changed",
    )
    archive = launch["archive"]
    executed_wrapper = Path(str(identity["executed_wrapper_path"]))
    archived_wrapper = Path(str(launch["wrapper"]))
    expected_spool_parent = f"job{launch['slurm_job_id']}"
    _require(
        executed_wrapper == archived_wrapper
        or (
            executed_wrapper.is_absolute()
            and executed_wrapper.name == "slurm_script"
            and executed_wrapper.parent.name == expected_spool_parent
            and str(executed_wrapper).startswith("/var/spool/slurmd/")
        ),
        "replacement terminal executed-wrapper path is not an accepted Slurm spool path",
    )
    staged_root = Path(str(identity["staged_root"]))
    _require(
        identity["staged_executor_root"]
        == str(staged_root / "outputs/executor"),
        "replacement terminal staged executor root changed",
    )
    _bootstrap_stage_result(
        episode_root=root,
        path=Path(str(identity["bootstrap_stage_result_path"])),
        expected_file_sha256=str(identity["bootstrap_stage_result_file_sha256"]),
        expected_staged_root=staged_root,
        archive=archive,
    )
    _verify_staged_archive(staged_root=staged_root, archive=archive)
    records = identity.get("imported_module_provenance")
    _require(isinstance(records, list) and bool(records),
             "replacement terminal import provenance changed")
    staged_executor = staged_root / "outputs/executor"
    observed_names: list[str] = []
    for record in records:
        _require(
            isinstance(record, Mapping)
            and set(record) == {"module", "relative_path", "file_sha256"},
            "replacement terminal import record changed",
        )
        module_name = str(record["module"])
        observed_names.append(module_name)
        relative = Path(str(record["relative_path"]))
        _require(
            not relative.is_absolute() and ".." not in relative.parts,
            "replacement terminal import path is unsafe",
        )
        _require(
            _sha(staged_executor / relative, label=f"recorded import {module_name}")
            == record["file_sha256"],
            "replacement terminal imported module bytes changed",
        )
    _require(
        observed_names == sorted(observed_names)
        and len(observed_names) == len(set(observed_names))
        and _REQUIRED_IMPORT_MODULES.issubset(observed_names),
        "replacement terminal imported module set/order changed",
    )
    current_records = _import_provenance(staged_executor)
    current_by_name = {record["module"]: record for record in current_records}
    _require(
        all(current_by_name.get(record["module"]) == record for record in records),
        "replacement terminal recorded imports differ from current staged bytes",
    )
    policy = EpisodePolicy.load(staged_root / "SEARCH_POLICY.yaml")
    verified_attestation = verify_attestation_root(
        run_root=root / "outputs/development_run002",
        attestation_dir=Path(str(identity["attestation_dir"])),
        policy=policy,
        outputs_root=root / "outputs",
    )
    _require(
        verified_attestation["attestation_manifest_sha256"]
        == identity["attestation_manifest_file_sha256"]
        and verified_attestation["attestation_artifact_file_sha256"]
        == identity["attestation_artifact_file_sha256"]
        and verified_attestation["attestation_sha256"] == identity["attestation_sha256"]
        and verified_attestation["verified_trial_count"] == 36
        and verified_attestation["source_trial_files_modified"] is False
        and verified_attestation["objective_values_read"] is False
        and verified_attestation["objective_values_compared"] is False
        and verified_attestation["objective_values_emitted"] is False
        and verified_attestation["final_connectivity_accessed"] is False,
        "replacement terminal attestation evidence changed",
    )
    return {
        "schema_version": "ep12.replacement_attestation_terminal_evidence.v2",
        "terminal_identity_path": str(path),
        "terminal_identity_file_sha256": _sha256_file(
            path, root=root, label="replacement attestation terminal identity"
        ),
        "terminal_identity_sha256": identity["identity_sha256"],
        "slurm_job_id": identity["slurm_job_id"],
        "slurm_job_name": identity["slurm_job_name"],
        "dependency": identity["dependency"],
        "scheduler_dependency": identity["scheduler_dependency"],
        "logical_dependency_satisfied": identity[
            "logical_dependency_satisfied"
        ],
        "finalizer_terminal_completion": dict(
            identity["finalizer_terminal_completion"]
        ),
        "launch_receipt_file_sha256": identity["launch_receipt_file_sha256"],
        "launch_receipt_sha256": identity["launch_receipt_sha256"],
        "stage_receipt_file_sha256": identity["stage_receipt_file_sha256"],
        "stage_receipt_sha256": identity["stage_receipt_sha256"],
        "release_intent_path": identity["release_intent_path"],
        "release_intent_file_sha256": identity["release_intent_file_sha256"],
        "release_intent_sha256": identity["release_intent_sha256"],
        "release_authorization_path": identity["release_authorization_path"],
        "release_authorization_file_sha256": identity[
            "release_authorization_file_sha256"
        ],
        "release_authorization_sha256": identity[
            "release_authorization_sha256"
        ],
        "release_receipt_path": launch["release_receipt_path"],
        "release_receipt_file_sha256": launch["release_receipt_file_sha256"],
        "release_receipt_sha256": launch["release_receipt_sha256"],
        "archive": dict(archive),
        "bootstrap_stage_result_file_sha256": identity[
            "bootstrap_stage_result_file_sha256"
        ],
        "staged_root": identity["staged_root"],
        "attestation_manifest_file_sha256": identity[
            "attestation_manifest_file_sha256"
        ],
        "attestation_artifact_file_sha256": identity[
            "attestation_artifact_file_sha256"
        ],
        "attestation_sha256": identity["attestation_sha256"],
        "verified_trial_count": 36,
        "source_trial_files_modified": False,
        "objective_values_read": False,
        "objective_values_compared": False,
        "objective_values_emitted": False,
        "superseded_job_id": identity["superseded_job_id"],
        "superseded_result_authoritative": False,
        "attestation_result_authoritative": True,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    publish = subparsers.add_parser("publish")
    combined = subparsers.add_parser("attest-and-publish")
    for selected in (publish, combined):
        selected.add_argument("--episode-root", type=Path, required=True)
        selected.add_argument("--launch-intent", type=Path, required=True)
        selected.add_argument("--launch-intent-file-sha256", required=True)
        selected.add_argument("--launch-intent-sha256", required=True)
        selected.add_argument("--attestation-output", type=Path, required=True)
        selected.add_argument("--terminal-identity-output", type=Path, required=True)
        selected.add_argument("--bootstrap-stage-result", type=Path, required=True)
        selected.add_argument("--bootstrap-stage-result-file-sha256", required=True)
        selected.add_argument("--staged-root", type=Path, required=True)
        selected.add_argument("--stage-receipt", type=Path, required=True)
        selected.add_argument("--executed-wrapper", type=Path, required=True)
        selected.add_argument("--slurm-job-id", required=True)
        selected.add_argument("--slurm-job-name", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--episode-root", type=Path, required=True)
    verify.add_argument("--identity", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command in {"publish", "attest-and-publish"}:
        publisher = (
            attest_and_publish_terminal_identity
            if arguments.command == "attest-and-publish"
            else publish_terminal_identity
        )
        result = publisher(
            episode_root=arguments.episode_root,
            launch_intent_path=arguments.launch_intent,
            launch_intent_file_sha256=arguments.launch_intent_file_sha256,
            launch_intent_sha256=arguments.launch_intent_sha256,
            attestation_output=arguments.attestation_output,
            terminal_identity_output=arguments.terminal_identity_output,
            bootstrap_stage_result=arguments.bootstrap_stage_result,
            bootstrap_stage_result_file_sha256=arguments.bootstrap_stage_result_file_sha256,
            staged_root=arguments.staged_root,
            stage_receipt_path=arguments.stage_receipt,
            executed_wrapper=arguments.executed_wrapper,
            slurm_job_id=arguments.slurm_job_id,
            slurm_job_name=arguments.slurm_job_name,
        )
    else:
        result = verify_replacement_attestation_terminal_identity(
            episode_root=arguments.episode_root,
            identity_path=arguments.identity,
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
