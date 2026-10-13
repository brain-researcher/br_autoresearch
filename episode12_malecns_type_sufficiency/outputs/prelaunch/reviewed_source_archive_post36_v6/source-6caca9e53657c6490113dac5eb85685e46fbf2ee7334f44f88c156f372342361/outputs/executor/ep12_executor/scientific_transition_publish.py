"""Build and exact-once publish the EP12 post-36 transition bundle.

This publisher intentionally makes a narrow claim.  A pinned incident records
that an operator was shown development outcome values.  Publication verifies
that incident, then programmatically extracts only status/contract fields from
the 36 hash-authenticated trial reports.  It never decodes the finalization
archive and it records that the displayed values were not used for the
scientific-contract transition.

The four durable files are assembled in a private sibling directory and made
visible with one directory rename.  A retry may only reuse byte-identical
artifacts; an existing partial or different tree fails closed.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Mapping, Sequence

from . import contract_invariance_probe
from . import scientific_contract_transition
from . import scientific_transition_binding
from .secure_transaction import open_regular_lock_at, open_secure_directory


PUBLISH_LOCK_NAME = ".scientific_contract_transition.publish.lock"
_BUNDLE_NAMES = {
    scientific_transition_binding.TRANSITION_MANIFEST_NAME,
    scientific_transition_binding.TRANSITION_RECEIPT_NAME,
    scientific_transition_binding.REPORT_DECLARATIONS_NAME,
    scientific_transition_binding.SYNTHETIC_INVARIANCE_NAME,
}


class ScientificTransitionPublishError(RuntimeError):
    """Transition evidence is unsafe, unstable, or already differs on disk."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScientificTransitionPublishError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a JSON object")
    return value


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise ScientificTransitionPublishError(
            "transition publication evidence is not finite canonical JSON"
        ) from exc
    return _sha_bytes(encoded)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ScientificTransitionPublishError(
            "transition publication artifact is not finite JSON"
        ) from exc


def _read_json(
    path: Path, *, boundary: Path, label: str
) -> tuple[bytes, Mapping[str, Any]]:
    try:
        encoded = scientific_transition_binding._read_bytes(
            path, boundary=boundary, label=label
        )
        payload = scientific_transition_binding._json_from_bytes(encoded, label)
    except (OSError, ValueError) as exc:
        raise ScientificTransitionPublishError(str(exc)) from exc
    return encoded, payload


def _secure_directory(path: Path, *, label: str) -> Path:
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        return scientific_transition_binding._secure_path(
            candidate,
            boundary=candidate,
            label=label,
            require_file=False,
        )
    except (OSError, ValueError) as exc:
        raise ScientificTransitionPublishError(str(exc)) from exc


def _load_authenticated_reports(
    *,
    episode_root: Path,
    attested_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, bytes]:
    run_root = episode_root / "outputs/development_run002"
    reports: dict[str, bytes] = {}
    for trial_id, raw_row in sorted(attested_by_id.items()):
        row = _mapping(raw_row, f"attested trial {trial_id}")
        relative = row.get("source_trial_directory")
        _require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and Path(relative).as_posix() == relative
            and all(part not in {"", ".", ".."} for part in Path(relative).parts),
            f"unsafe authenticated report directory for {trial_id}",
        )
        path = run_root / relative / "trial_report.json"
        try:
            encoded = scientific_transition_binding._read_bytes(
                path,
                boundary=run_root,
                label=f"authenticated observed report {trial_id}",
            )
        except (OSError, ValueError) as exc:
            raise ScientificTransitionPublishError(str(exc)) from exc
        source_files = _mapping(
            row.get("source_files_sha256"), f"attested source hashes {trial_id}"
        )
        _require(
            _sha_bytes(encoded) == source_files.get("trial_report.json"),
            f"authenticated report bytes changed for {trial_id}",
        )
        reports[str(trial_id)] = encoded
    _require(len(reports) == 36, "authenticated report set is not exact-36")
    return reports


def _declaration_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    attestation_manifest_sha256: str,
    attestation_semantic_sha256: str,
) -> dict[str, Any]:
    normalized = [dict(row) for row in rows]
    _require(len(normalized) == 36, "report declarations are not exact-36")
    return {
        "schema_version": scientific_transition_binding.REPORT_DECLARATIONS_SCHEMA,
        "transition_id": scientific_transition_binding.TRANSITION_ID,
        "observed_attestation_manifest_sha256": attestation_manifest_sha256,
        "observed_attestation_sha256": attestation_semantic_sha256,
        "trial_count": 36,
        "rows": normalized,
        "objective_values_read": False,
        "objective_values_compared": False,
        "objective_values_emitted": False,
        "final_connectivity_accessed": False,
        "declarations_sha256": _canonical_digest(normalized),
    }


def _collect_bundle(
    *,
    episode_root: Path,
    executor_root: Path,
    synthetic_invariance_path: Path,
    replacement_terminal_identity_path: Path,
    post36_start_authorization_path: Path,
    obsolete_cancellation_receipt_path: Path,
    attestation_authority_snapshot: Mapping[str, Any] | None = None,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Verify all evidence and build the exact four output byte strings."""

    binding = scientific_transition_binding
    transition = scientific_contract_transition
    probe = contract_invariance_probe
    root = _secure_directory(episode_root, label="episode root")
    executor = _secure_directory(executor_root, label="executor root")

    spec_path = executor / "CONTRACT_TRANSITION_SPEC.json"
    _, spec = _read_json(
        spec_path, boundary=executor, label="scientific transition specification"
    )
    try:
        definition = transition.verify_contract_transition_definition(
            executor_root=executor, transition_spec_path=spec_path
        )
    except (OSError, ValueError) as exc:
        raise ScientificTransitionPublishError(str(exc)) from exc

    # The status-only attestation and its dynamic scheduler authority must both
    # authenticate before any transition artifact is assembled.
    attestation_dir = root / binding.OBSERVED_ATTESTATION_RELATIVE
    try:
        (
            _attestation_manifest,
            attestation_manifest_file_sha,
            attestation,
            attestation_artifact_file_sha,
        ) = binding._verify_attestation_manifest(
            episode_root=root, attestation_dir=attestation_dir
        )
        attested_by_id = transition._verify_attestation(attestation)
    except (OSError, ValueError) as exc:
        raise ScientificTransitionPublishError(str(exc)) from exc
    attestation_semantic_sha = str(attestation.get("attestation_sha256"))

    try:
        attestation_authority = binding._verify_dynamic_attestation_authority(
            episode_root=root,
            transition_spec=spec,
            terminal_identity_path=replacement_terminal_identity_path,
            start_authorization_path=post36_start_authorization_path,
            cancellation_receipt_path=obsolete_cancellation_receipt_path,
            attestation_manifest_file_sha256=attestation_manifest_file_sha,
            attestation_artifact_file_sha256=attestation_artifact_file_sha,
            attestation_sha256=attestation_semantic_sha,
        )
        if attestation_authority_snapshot is not None:
            attestation_authority = binding._verify_published_attestation_authority(
                attestation_authority_snapshot, current=attestation_authority
            )
        exposure = binding._verify_operator_exposure_incident(
            episode_root=root, transition_spec=spec
        )
    except (OSError, ValueError) as exc:
        raise ScientificTransitionPublishError(str(exc)) from exc

    reports = _load_authenticated_reports(
        episode_root=root, attested_by_id=attested_by_id
    )
    try:
        declarations = transition.extract_authenticated_report_contract_declarations(
            attested_by_id=attested_by_id,
            authenticated_trial_report_bytes=reports,
        )
    except (OSError, ValueError) as exc:
        raise ScientificTransitionPublishError(str(exc)) from exc
    declaration = _declaration_artifact(
        rows=declarations,
        attestation_manifest_sha256=attestation_manifest_file_sha,
        attestation_semantic_sha256=attestation_semantic_sha,
    )

    comparison_path = Path(
        os.path.abspath(os.fspath(synthetic_invariance_path))
    )
    comparison_bytes, comparison = _read_json(
        comparison_path,
        boundary=root / "outputs",
        label="synthetic transition invariance comparison",
    )
    try:
        verified_comparison = probe.verify_artifact(
            artifact_path=comparison_path,
            spec_path=executor / "CONTRACT_TRANSITION_PROBE_SPEC.json",
        )
        _require(
            dict(verified_comparison) == dict(comparison),
            "synthetic invariance comparison changed during verification",
        )
        transition.verify_synthetic_invariance_evidence(comparison, spec)
        probe_evidence = binding._verify_probe_pair(
            episode_root=root,
            executor_root=executor,
            comparison=comparison,
        )
    except (OSError, ValueError) as exc:
        raise ScientificTransitionPublishError(str(exc)) from exc
    _require(
        binding._read_bytes(
            comparison_path,
            boundary=root / "outputs",
            label="synthetic transition invariance comparison",
        )
        == comparison_bytes,
        "synthetic invariance comparison changed while being authenticated",
    )

    try:
        archive_sha, limitation_manifest_sha = binding._verify_observed_prefix_archive(
            episode_root=root,
            finalization_dir=root / binding.OBSERVED_FINALIZATION_RELATIVE,
        )
    except (OSError, ValueError) as exc:
        raise ScientificTransitionPublishError(str(exc)) from exc

    try:
        receipt = transition.build_transition_receipt(
            executor_root=executor,
            observed_attestation=attestation,
            observed_attestation_manifest_sha256=attestation_manifest_file_sha,
            observed_attestation_authority=attestation_authority,
            authenticated_trial_report_bytes=reports,
            synthetic_invariance_evidence=comparison,
            transition_spec_path=spec_path,
        )
        transition.verify_transition_receipt(receipt)
    except (OSError, ValueError) as exc:
        raise ScientificTransitionPublishError(str(exc)) from exc

    expected_receipt_bindings = {
        "observed_attestation_authority": attestation_authority,
        "operator_exposure_incident_file_sha256": exposure[
            "operator_exposure_incident_file_sha256"
        ],
        "operator_exposure_incident_semantic_sha256": exposure[
            "operator_exposure_incident_semantic_sha256"
        ],
        "observed_report_contract_declarations_sha256": declaration[
            "declarations_sha256"
        ],
    }
    for field, expected in expected_receipt_bindings.items():
        _require(
            receipt.get(field) == expected,
            f"transition receipt does not bind authenticated evidence: {field}",
        )
    _require(
        receipt.get("operator_development_outcome_values_displayed") is True
        and receipt.get(
            "programmatic_transition_extraction_outcome_fields_accessed"
        )
        is False
        and receipt.get("values_used_for_transition_decisions") is False
        and receipt.get("final_connectivity_accessed") is False,
        "transition receipt states an unscoped or false exposure claim",
    )

    receipt_bytes = _json_bytes(receipt)
    declaration_bytes = _json_bytes(declaration)
    content_files = {
        binding.TRANSITION_RECEIPT_NAME: receipt_bytes,
        binding.REPORT_DECLARATIONS_NAME: declaration_bytes,
        binding.SYNTHETIC_INVARIANCE_NAME: comparison_bytes,
    }
    manifest = {
        "schema_version": binding.TRANSITION_MANIFEST_SCHEMA,
        "files": {name: _sha_bytes(value) for name, value in content_files.items()},
        "transition_id": binding.TRANSITION_ID,
        "transition_receipt_sha256": receipt["transition_receipt_sha256"],
        "transition_spec_sha256": definition["transition_spec_sha256"],
        "observed_prefix_contract_sha256": definition["old_contract_sha256"],
        "active_contract_sha256": definition["new_contract_sha256"],
        "observed_prefix_qualification_spec_sha256": definition[
            "old_qualification_spec_sha256"
        ],
        "active_qualification_spec_sha256": definition[
            "new_qualification_spec_sha256"
        ],
        "observed_prefix_scientific_models_sha256": receipt[
            "old_scientific_models_sha256"
        ],
        "active_scientific_models_sha256": receipt[
            "active_scientific_models_sha256"
        ],
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
        "source_archive_manifest_sha256": attestation_authority["source_archive"][
            "manifest_sha256"
        ],
        "source_archive_tree_sha256": attestation_authority["source_archive"][
            "tree_sha256"
        ],
        "controller_final_receipt_status": attestation_authority[
            "controller_final_receipt"
        ]["status"],
        "observed_trial_count": 36,
        "observed_report_contract_declarations_sha256": declaration[
            "declarations_sha256"
        ],
        "observed_prefix_archive_file_sha256": archive_sha,
        "observed_prefix_limitation_manifest_file_sha256": limitation_manifest_sha,
        **probe_evidence,
        "operator_exposure_incident_file_sha256": exposure[
            "operator_exposure_incident_file_sha256"
        ],
        "operator_exposure_incident_semantic_sha256": exposure[
            "operator_exposure_incident_semantic_sha256"
        ],
        "source_trial_reports_reverified": True,
        "operator_development_outcome_values_displayed": True,
        "programmatic_transition_extraction_outcome_fields_accessed": False,
        "values_used_for_transition_decisions": False,
        "final_connectivity_accessed": False,
        "complete": True,
    }
    _require(
        set(manifest) == binding._MANIFEST_FIELDS,
        "publisher and transition manifest schemas disagree",
    )
    payloads = {
        **content_files,
        binding.TRANSITION_MANIFEST_NAME: _json_bytes(manifest),
    }
    return payloads, {
        "transition_receipt_sha256": receipt["transition_receipt_sha256"],
        "observed_attestation_authority": attestation_authority,
        "operator_development_outcome_values_displayed": True,
        "programmatic_transition_extraction_outcome_fields_accessed": False,
        "values_used_for_transition_decisions": False,
        "final_connectivity_accessed": False,
    }


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        _require(written > 0, "short write while publishing transition bundle")
        view = view[written:]


def _write_private_bundle(directory_fd: int, payloads: Mapping[str, bytes]) -> None:
    _require(set(payloads) == _BUNDLE_NAMES, "transition bundle is not exact-four")
    for name in sorted(payloads):
        _require(Path(name).name == name, f"unsafe transition artifact name: {name}")
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o640,
            dir_fd=directory_fd,
        )
        try:
            _write_all(descriptor, payloads[name])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    os.fsync(directory_fd)


def _read_existing_bundle(parent_fd: int, target_name: str) -> dict[str, bytes]:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        target_fd = os.open(target_name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise ScientificTransitionPublishError(
            f"cannot safely open existing transition bundle: {exc}"
        ) from exc
    try:
        metadata = os.fstat(target_fd)
        _require(stat.S_ISDIR(metadata.st_mode), "existing transition root is not a directory")
        names = set(os.listdir(target_fd))
        _require(names == _BUNDLE_NAMES, "existing transition bundle is partial or has extra files")
        result: dict[str, bytes] = {}
        for name in sorted(names):
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=target_fd,
            )
            try:
                child = os.fstat(descriptor)
                _require(stat.S_ISREG(child.st_mode), f"transition artifact is not regular: {name}")
                chunks: list[bytes] = []
                while True:
                    block = os.read(descriptor, 1024 * 1024)
                    if not block:
                        break
                    chunks.append(block)
                result[name] = b"".join(chunks)
            finally:
                os.close(descriptor)
        return result
    finally:
        os.close(target_fd)


def _remove_private_bundle(parent_fd: int, temporary_name: str) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    try:
        temporary_fd = os.open(temporary_name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return
    try:
        for name in os.listdir(temporary_fd):
            os.unlink(name, dir_fd=temporary_fd)
    finally:
        os.close(temporary_fd)
    os.rmdir(temporary_name, dir_fd=parent_fd)


def _publish_exact_once(
    *, episode_root: Path, payloads: Mapping[str, bytes]
) -> str:
    outputs_root = episode_root / "outputs"
    parent = episode_root / "outputs/prelaunch"
    target_name = scientific_transition_binding.TRANSITION_ROOT_RELATIVE.name
    _parent, parent_fd = open_secure_directory(
        parent, root=outputs_root, create=True, mode=0o750
    )
    lock_fd = -1
    temporary_name: str | None = None
    try:
        lock_fd = open_regular_lock_at(parent_fd, PUBLISH_LOCK_NAME)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            existing = _read_existing_bundle(parent_fd, target_name)
            _require(
                existing == dict(payloads),
                "existing transition bundle differs from this exact retry",
            )
            return "reused_identical"

        temporary_name = (
            f".{target_name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
        )
        os.mkdir(temporary_name, mode=0o750, dir_fd=parent_fd)
        flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
        temporary_fd = os.open(temporary_name, flags, dir_fd=parent_fd)
        try:
            _write_private_bundle(temporary_fd, payloads)
        finally:
            os.close(temporary_fd)
        try:
            os.rename(
                temporary_name,
                target_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            temporary_name = None
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            existing = _read_existing_bundle(parent_fd, target_name)
            _require(
                existing == dict(payloads),
                "concurrent transition publication differs from this transaction",
            )
        os.fsync(parent_fd)
        return "created"
    finally:
        if temporary_name is not None:
            try:
                _remove_private_bundle(parent_fd, temporary_name)
            except FileNotFoundError:
                pass
        if lock_fd >= 0:
            os.close(lock_fd)
        os.close(parent_fd)


def publish_scientific_transition(
    *,
    episode_root: Path,
    executor_root: Path,
    synthetic_invariance_path: Path,
    replacement_terminal_identity_path: Path,
    post36_start_authorization_path: Path,
    obsolete_cancellation_receipt_path: Path,
) -> dict[str, Any]:
    """Reverify stable evidence, publish exact-four, and verify the durable root."""

    root = _secure_directory(episode_root, label="episode root")
    first, first_summary = _collect_bundle(
        episode_root=root,
        executor_root=executor_root,
        synthetic_invariance_path=synthetic_invariance_path,
        replacement_terminal_identity_path=replacement_terminal_identity_path,
        post36_start_authorization_path=post36_start_authorization_path,
        obsolete_cancellation_receipt_path=obsolete_cancellation_receipt_path,
    )
    second, second_summary = _collect_bundle(
        episode_root=root,
        executor_root=executor_root,
        synthetic_invariance_path=synthetic_invariance_path,
        replacement_terminal_identity_path=replacement_terminal_identity_path,
        post36_start_authorization_path=post36_start_authorization_path,
        obsolete_cancellation_receipt_path=obsolete_cancellation_receipt_path,
        attestation_authority_snapshot=first_summary[
            "observed_attestation_authority"
        ],
    )
    _require(
        first == second and first_summary == second_summary,
        "transition inputs changed between the two prepublication verifications",
    )
    status = _publish_exact_once(episode_root=root, payloads=second)
    manifest_path = (
        root
        / scientific_transition_binding.TRANSITION_ROOT_RELATIVE
        / scientific_transition_binding.TRANSITION_MANIFEST_NAME
    )
    manifest_sha = _sha_bytes(second[scientific_transition_binding.TRANSITION_MANIFEST_NAME])
    try:
        verified = scientific_transition_binding.verify_scientific_contract_transition(
            episode_root=root,
            executor_root=executor_root,
            manifest_path=manifest_path,
            expected_manifest_file_sha256=manifest_sha,
            expected_transition_receipt_sha256=str(
                second_summary["transition_receipt_sha256"]
            ),
        )
    except (OSError, ValueError) as exc:
        raise ScientificTransitionPublishError(
            f"published transition bundle did not reverify: {exc}"
        ) from exc
    return {
        "publication_status": status,
        "transition_root": str(manifest_path.parent),
        "transition_manifest_path": str(manifest_path),
        "transition_manifest_file_sha256": manifest_sha,
        **second_summary,
        "durable_verification": verified,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ep12-scientific-transition-publish",
        description="Exact-once publish the authenticated EP12 post-36 transition",
    )
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--executor-root", type=Path, required=True)
    parser.add_argument("--synthetic-invariance", type=Path, required=True)
    parser.add_argument(
        "--replacement-terminal-identity", type=Path, required=True
    )
    parser.add_argument(
        "--post36-start-authorization", type=Path, required=True
    )
    parser.add_argument(
        "--obsolete-cancellation-receipt", type=Path, required=True
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    result = publish_scientific_transition(
        episode_root=arguments.episode_root,
        executor_root=arguments.executor_root,
        synthetic_invariance_path=arguments.synthetic_invariance,
        replacement_terminal_identity_path=arguments.replacement_terminal_identity,
        post36_start_authorization_path=arguments.post36_start_authorization,
        obsolete_cancellation_receipt_path=arguments.obsolete_cancellation_receipt,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ScientificTransitionPublishError",
    "publish_scientific_transition",
]
