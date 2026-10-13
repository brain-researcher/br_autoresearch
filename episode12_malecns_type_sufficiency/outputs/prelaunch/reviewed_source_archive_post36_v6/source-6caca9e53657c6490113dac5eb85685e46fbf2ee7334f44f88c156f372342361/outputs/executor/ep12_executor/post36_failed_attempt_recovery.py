"""Crash-safe preservation of the failed reviewed-v5 post-36 launch.

This transaction is deliberately scheduler-inert.  It first authenticates the
five terminal Slurm rows, the reviewed-v5 control-plane receipts, the exact
failure log, and the absence of every output that a successful first stage
would have published.  Only then does it atomically move the two canonical
transaction directories to immutable failed-attempt names.  A replay either
finishes an interrupted move or verifies the completed receipt; it never
deletes, overwrites, submits, releases, or cancels a job.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Sequence

from .policy import canonical_json, digest_object
from .post36_orchestration import (
    OBSERVED_FINALIZER_JOB_ID,
    _atomic_json,
    _effective_user,
    _run_scheduler,
)
from .secure_transaction import open_regular_lock_at, open_secure_directory
from .synthetic import EPISODE_ROOT


INTENT_SCHEMA = "ep12.post36_failed_attempt_recovery_intent.v1"
PROGRESS_SCHEMA = "ep12.post36_failed_attempt_recovery_progress.v1"
RECEIPT_SCHEMA = "ep12.post36_failed_attempt_recovery_receipt.v1"
EVIDENCE_SCHEMA = "ep12.post36_failed_attempt_recovery_evidence.v1"

RECOVERY_ROOT_RELATIVE = Path(
    "outputs/prelaunch/post36_failed_attempt_v5_adaptive_stage_contract_45396786"
)
INTENT_RELATIVE = RECOVERY_ROOT_RELATIVE / "intent.json"
PROGRESS_RELATIVE = RECOVERY_ROOT_RELATIVE / "progress.json"
RECEIPT_RELATIVE = RECOVERY_ROOT_RELATIVE / "receipt.json"
PRESERVED_LOG_RELATIVE = RECOVERY_ROOT_RELATIVE / (
    "slurm-attest36-archive-v2-45396786.out"
)

REPLACEMENT_SOURCE_RELATIVE = Path(
    "outputs/prelaunch/replacement_attestation_launch"
)
CONTROLLER_SOURCE_RELATIVE = Path("outputs/prelaunch/post36_orchestration")
REPLACEMENT_ARCHIVE_RELATIVE = Path(
    "outputs/prelaunch/"
    "replacement_attestation_launch_failed_v5_adaptive_stage_contract_45396786"
)
CONTROLLER_ARCHIVE_RELATIVE = Path(
    "outputs/prelaunch/"
    "post36_orchestration_failed_v5_adaptive_stage_contract_45396786"
)

FAILED_LOG_PATH = Path(
    "/scratch/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency/"
    "slurm-attest36-archive-v2-45396786.out"
)
EXPECTED_FAILURE_TEXT = (
    "adaptive_cycle_1 trial identity mismatch: adaptive_c1_001"
)
V5_TREE_SHA256 = (
    "06d2d80f8f5b853e496abba4e2c67dc1fc2e02d0178826f111858ac5b734537e"
)

FAILED_JOBS: tuple[dict[str, str], ...] = (
    {
        "job_id": "45396786",
        "job_name": "ep12-replacement-attest-a44252df28eeb4e44c334308",
        "state": "FAILED",
        "exit_code": "1:0",
    },
    {
        "job_id": "45397274",
        "job_name": "ep12-post36-probe-ce15e822e0774ba047d1d99e",
        "state": "CANCELLED",
        "exit_code": "0:0",
    },
    {
        "job_id": "45397275",
        "job_name": "ep12-comparison-a9aa15acf8e8342b9f27660f",
        "state": "CANCELLED",
        "exit_code": "0:0",
    },
    {
        "job_id": "45397276",
        "job_name": "ep12-publisher-03ad685fc34d539039879640",
        "state": "CANCELLED",
        "exit_code": "0:0",
    },
    {
        "job_id": "45397277",
        "job_name": "ep12-downstream-1254625058cdd47e402f720a",
        "state": "CANCELLED",
        "exit_code": "0:0",
    },
)

REQUIRED_ABSENT_OUTPUTS: tuple[Path, ...] = (
    Path("outputs/development_run002/observed_attestation"),
    REPLACEMENT_SOURCE_RELATIVE / "terminal_identity.json",
    Path("outputs/prelaunch/contract_transition_probe/post36_v1/probe.json"),
    Path("outputs/prelaunch/contract_transition_probe/comparison_v1/comparison.json"),
    Path("outputs/prelaunch/post36_downstream_launch"),
)


class FailedAttemptRecoveryError(RuntimeError):
    """The failed v5 launch cannot be preserved or authenticated safely."""


TerminalResolver = Callable[[str], Sequence[Mapping[str, str]]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FailedAttemptRecoveryError(message)


def _lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular(path: Path, *, label: str) -> bytes:
    absolute = _lexical(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        metadata = os.lstat(current)
        _require(not stat.S_ISLNK(metadata.st_mode), f"{label} traverses a symlink")
    descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    finally:
        os.close(descriptor)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    encoded = _read_regular(path, label=label)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FailedAttemptRecoveryError(f"{label} is malformed") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    _require(
        encoded == (canonical_json(value) + "\n").encode("ascii"),
        f"{label} is not canonical JSON",
    )
    return value


def _self_hashed_json(
    path: Path, *, label: str, hash_field: str
) -> tuple[dict[str, Any], str]:
    encoded = _read_regular(path, label=label)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FailedAttemptRecoveryError(f"{label} is malformed") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    _require(
        encoded == (canonical_json(value) + "\n").encode("ascii"),
        f"{label} is not canonical JSON",
    )
    core = {key: item for key, item in value.items() if key != hash_field}
    _require(
        value.get(hash_field) == digest_object(core),
        f"{label} semantic hash mismatch",
    )
    return value, _sha256_bytes(encoded)


def _tree_manifest(path: Path, *, label: str) -> dict[str, Any]:
    root = _lexical(path)
    metadata = os.lstat(root)
    _require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{label} is not a real directory",
    )
    directories: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    for current, names, filenames in os.walk(root, topdown=True, followlinks=False):
        names.sort()
        filenames.sort()
        current_path = Path(current)
        current_metadata = os.lstat(current_path)
        _require(
            stat.S_ISDIR(current_metadata.st_mode)
            and not stat.S_ISLNK(current_metadata.st_mode),
            f"{label} contains an unsafe directory",
        )
        relative_directory = current_path.relative_to(root).as_posix() or "."
        directories.append(
            {
                "path": relative_directory,
                "mode": f"{stat.S_IMODE(current_metadata.st_mode):04o}",
            }
        )
        for name in names:
            child = current_path / name
            child_metadata = os.lstat(child)
            _require(
                stat.S_ISDIR(child_metadata.st_mode)
                and not stat.S_ISLNK(child_metadata.st_mode),
                f"{label} contains a symlink or special directory",
            )
        for name in filenames:
            child = current_path / name
            child_metadata = os.lstat(child)
            _require(
                stat.S_ISREG(child_metadata.st_mode)
                and not stat.S_ISLNK(child_metadata.st_mode),
                f"{label} contains a symlink or special file",
            )
            encoded = _read_regular(child, label=f"{label} file")
            files.append(
                {
                    "path": child.relative_to(root).as_posix(),
                    "mode": f"{stat.S_IMODE(child_metadata.st_mode):04o}",
                    "size_bytes": len(encoded),
                    "sha256": _sha256_bytes(encoded),
                }
            )
    core = {
        "directories": directories,
        "files": files,
        "directory_count": len(directories),
        "file_count": len(files),
    }
    return {**core, "tree_sha256": digest_object(core)}


def _assert_outputs_absent(root: Path) -> None:
    for relative in REQUIRED_ABSENT_OUTPUTS:
        _require(
            not os.path.lexists(root / relative),
            f"failed v5 attempt published unexpected output: {relative}",
        )


def default_terminal_resolver(submission_user: str) -> Sequence[Mapping[str, str]]:
    """Read one exact terminal accounting row per failed-v5 job."""

    _require(submission_user == _effective_user(), "failed-job query user mismatch")
    identifiers = ",".join(record["job_id"] for record in FAILED_JOBS)
    accounting = _run_scheduler(
        [
            "/usr/bin/sacct", "-X", "--noheader", "--parsable2",
            "--user", submission_user, "--jobs", identifiers,
            "--starttime", "2026-09-25",
            "--format=JobIDRaw,JobName%128,User%64,State,ExitCode",
        ]
    )
    rows: list[dict[str, str]] = []
    expected_ids = {record["job_id"] for record in FAILED_JOBS}
    for line in accounting.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        _require(len(fields) == 5, "malformed failed-v5 sacct row")
        job_id, job_name, user, state, exit_code = fields
        if job_id not in expected_ids:
            continue
        rows.append(
            {
                "job_id": job_id,
                "job_name": job_name,
                "user": user,
                "state": state,
                "exit_code": exit_code,
            }
        )
    return rows


def _validated_terminal_rows(
    values: Sequence[Mapping[str, str]], *, submission_user: str
) -> list[dict[str, str]]:
    rows = [{str(key): str(item) for key, item in value.items()} for value in values]
    expected = [
        {**record, "user": submission_user}
        for record in FAILED_JOBS
    ]
    rows.sort(key=lambda value: value.get("job_id", ""))
    expected.sort(key=lambda value: value["job_id"])
    _require(rows == expected, "failed-v5 terminal Slurm identity changed")
    return rows


def _lineage(root: Path) -> dict[str, Any]:
    replacement_path = root / REPLACEMENT_SOURCE_RELATIVE / "receipt.json"
    cancellation_path = (
        root / REPLACEMENT_SOURCE_RELATIVE / "obsolete_cancellation/receipt.json"
    )
    controller_path = root / CONTROLLER_SOURCE_RELATIVE / "receipt.json"
    replacement, replacement_file_sha = _self_hashed_json(
        replacement_path,
        label="failed-v5 replacement receipt",
        hash_field="receipt_sha256",
    )
    cancellation, cancellation_file_sha = _self_hashed_json(
        cancellation_path,
        label="failed-v5 cancellation receipt",
        hash_field="receipt_sha256",
    )
    controller, controller_file_sha = _self_hashed_json(
        controller_path,
        label="failed-v5 controller receipt",
        hash_field="receipt_sha256",
    )
    _require(
        replacement.get("schema_version")
        == "ep12.replacement_attestation_launch_receipt.v2"
        and replacement.get("slurm_job_id") == FAILED_JOBS[0]["job_id"]
        and replacement.get("slurm_job_name") == FAILED_JOBS[0]["job_name"]
        and replacement.get("attestation_result_authoritative") is False
        and replacement.get("final_connectivity_accessed") is False,
        "failed-v5 replacement receipt identity changed",
    )
    finalizer = replacement.get("finalizer_terminal_completion")
    finalizer_core = (
        {
            key: item
            for key, item in finalizer.items()
            if key != "evidence_sha256"
        }
        if isinstance(finalizer, Mapping)
        else {}
    )
    _require(
        isinstance(finalizer, Mapping)
        and set(finalizer)
        == {
            "schema_version",
            "job_id",
            "job_name",
            "user",
            "state",
            "exit_code",
            "logical_dependency",
            "logical_dependency_satisfied",
            "scheduler_dependency",
            "scheduler_dependency_omitted_because_terminal_success",
            "evidence_sha256",
        }
        and finalizer.get("schema_version")
        == "ep12.observed_finalizer_terminal_success.v1"
        and finalizer.get("job_id") == OBSERVED_FINALIZER_JOB_ID
        and finalizer.get("job_name") == "ep12-finalize-observed"
        and finalizer.get("state") == "COMPLETED"
        and finalizer.get("exit_code") == "0:0"
        and finalizer.get("logical_dependency")
        == f"afterok:{OBSERVED_FINALIZER_JOB_ID}"
        and finalizer.get("logical_dependency_satisfied") is True
        and finalizer.get("scheduler_dependency") is None
        and finalizer.get("scheduler_dependency_omitted_because_terminal_success")
        is True
        and finalizer.get("evidence_sha256") == digest_object(finalizer_core),
        "failed-v5 finalizer completion evidence changed",
    )
    archive = replacement.get("archive")
    _require(
        isinstance(archive, Mapping)
        and archive.get("tree_sha256") == V5_TREE_SHA256
        and archive.get("complete") is True
        and archive.get("inputs_included") is False
        and archive.get("development_outputs_included") is False
        and archive.get("prelaunch_outputs_included") is False,
        "failed-v5 source archive identity changed",
    )
    archive_receipt_path = Path(str(archive.get("archive_receipt_path", "")))
    _require(
        archive_receipt_path.is_absolute()
        and _sha256_bytes(
            _read_regular(archive_receipt_path, label="failed-v5 archive receipt")
        )
        == archive.get("archive_receipt_file_sha256"),
        "failed-v5 archive receipt bytes changed",
    )
    _require(
        cancellation.get("schema_version")
        == "ep12.obsolete_attestation_cancellation_receipt.v2"
        and cancellation.get("obsolete_job_id") == "45332084"
        and cancellation.get("replacement_job_id") == FAILED_JOBS[0]["job_id"]
        and str(cancellation.get("terminal_scheduler_state", "")).startswith(
            "CANCELLED"
        )
        and cancellation.get("replacement_receipt_sha256")
        == replacement.get("receipt_sha256")
        and cancellation.get("descendants_immediately_before_cancel") == []
        and cancellation.get("scancel_was_exactly_scoped") is True
        and cancellation.get("obsolete_result_authoritative") is False
        and cancellation.get("replacement_attestation_result_authoritative")
        is False
        and cancellation.get("final_connectivity_accessed") is False,
        "failed-v5 cancellation lineage changed",
    )
    _require(
        finalizer.get("user") == cancellation.get("submission_user"),
        "failed-v5 finalizer user differs from cancellation user",
    )
    stages = controller.get("stages")
    _require(
        controller.get("schema_version")
        == "ep12.post36_unattended_orchestration_receipt.v1"
        and controller.get("replacement_job_id") == FAILED_JOBS[0]["job_id"]
        and controller.get("replacement_launch_receipt_sha256")
        == replacement.get("receipt_sha256")
        and controller.get("obsolete_cancellation_receipt_sha256")
        == cancellation.get("receipt_sha256")
        and isinstance(stages, Mapping)
        and controller.get("final_connectivity_accessed") is False
        and controller.get("final_connectivity_access_authorized") is False,
        "failed-v5 controller lineage changed",
    )
    expected_stage_jobs = {
        "post36_probe": FAILED_JOBS[1],
        "comparison": FAILED_JOBS[2],
        "publisher": FAILED_JOBS[3],
        "downstream": FAILED_JOBS[4],
    }
    for stage, expected in expected_stage_jobs.items():
        value = stages.get(stage) if isinstance(stages, Mapping) else None
        _require(
            isinstance(value, Mapping)
            and value.get("slurm_job_id") == expected["job_id"]
            and value.get("slurm_job_name") == expected["job_name"]
            and value.get("final_connectivity_accessed") is False,
            f"failed-v5 {stage} receipt identity changed",
        )
    return {
        "source_archive_tree_sha256": V5_TREE_SHA256,
        "replacement_job_id": FAILED_JOBS[0]["job_id"],
        "replacement_receipt_file_sha256": replacement_file_sha,
        "replacement_receipt_sha256": replacement["receipt_sha256"],
        "finalizer_terminal_completion": dict(finalizer),
        "obsolete_job_id": cancellation["obsolete_job_id"],
        "obsolete_cancellation_receipt_file_sha256": cancellation_file_sha,
        "obsolete_cancellation_receipt_sha256": cancellation["receipt_sha256"],
        "prior_obsolete_terminal_scheduler": {
            "job_id": cancellation["obsolete_job_id"],
            "job_name": cancellation["obsolete_job_name"],
            "user": cancellation["submission_user"],
            "state": cancellation["terminal_scheduler_state"],
            "reason": cancellation["terminal_scheduler_reason"],
            "priority": cancellation["terminal_scheduler_priority"],
        },
        "prior_scancel_was_exactly_scoped": True,
        "controller_receipt_file_sha256": controller_file_sha,
        "controller_receipt_sha256": controller["receipt_sha256"],
        "final_connectivity_accessed": False,
    }


def _intent(
    *,
    root: Path,
    submission_user: str,
    terminal_rows: Sequence[Mapping[str, str]],
    lineage: Mapping[str, Any],
    replacement_tree: Mapping[str, Any],
    controller_tree: Mapping[str, Any],
    failure_log: bytes,
) -> dict[str, Any]:
    core = {
        "schema_version": INTENT_SCHEMA,
        "episode_root": str(root),
        "submission_user": submission_user,
        "failure_class": "adaptive_stage_contract_mismatch",
        "failure_message": EXPECTED_FAILURE_TEXT,
        "terminal_jobs": [dict(value) for value in terminal_rows],
        "lineage": dict(lineage),
        "replacement_transaction": {
            "source_relative": str(REPLACEMENT_SOURCE_RELATIVE),
            "archive_relative": str(REPLACEMENT_ARCHIVE_RELATIVE),
            "tree": dict(replacement_tree),
        },
        "controller_transaction": {
            "source_relative": str(CONTROLLER_SOURCE_RELATIVE),
            "archive_relative": str(CONTROLLER_ARCHIVE_RELATIVE),
            "tree": dict(controller_tree),
        },
        "failure_log": {
            "source_path": str(FAILED_LOG_PATH),
            "preserved_relative": str(PRESERVED_LOG_RELATIVE),
            "size_bytes": len(failure_log),
            "sha256": _sha256_bytes(failure_log),
        },
        "required_absent_outputs": [str(value) for value in REQUIRED_ABSENT_OUTPUTS],
        "scheduler_mutations_authorized": False,
        "old_transactions_deleted_or_overwritten": False,
        "connectivity_outcome_values_read": False,
        "final_connectivity_accessed": False,
    }
    return {**core, "intent_sha256": digest_object(core)}


def _validated_intent(root: Path) -> dict[str, Any]:
    value = _read_json(root / INTENT_RELATIVE, label="failed-attempt recovery intent")
    core = {key: item for key, item in value.items() if key != "intent_sha256"}
    _require(
        value.get("schema_version") == INTENT_SCHEMA
        and value.get("episode_root") == str(root)
        and value.get("intent_sha256") == digest_object(core),
        "failed-attempt recovery intent changed",
    )
    return value


def _progress(*, intent: Mapping[str, Any], state: str) -> dict[str, Any]:
    _require(
        state
        in {
            "intent_persisted",
            "log_preserved",
            "replacement_archived",
            "transactions_archived",
        },
        "invalid failed-attempt recovery state",
    )
    core = {
        "schema_version": PROGRESS_SCHEMA,
        "intent_sha256": intent["intent_sha256"],
        "state": state,
        "scheduler_mutations_performed": False,
        "old_transactions_deleted_or_overwritten": False,
        "final_connectivity_accessed": False,
    }
    return {**core, "progress_sha256": digest_object(core)}


def _write_progress(root: Path, intent: Mapping[str, Any], state: str) -> None:
    _atomic_json(root / PROGRESS_RELATIVE, _progress(intent=intent, state=state), root=root)


def _preserve_log(root: Path, intent: Mapping[str, Any]) -> None:
    target = root / PRESERVED_LOG_RELATIVE
    expected = intent["failure_log"]
    if os.path.lexists(target):
        encoded = _read_regular(target, label="preserved failed-v5 log")
        _require(
            len(encoded) == expected["size_bytes"]
            and _sha256_bytes(encoded) == expected["sha256"],
            "preserved failed-v5 log changed",
        )
        return
    encoded = _read_regular(FAILED_LOG_PATH, label="failed-v5 Slurm log")
    _require(
        len(encoded) == expected["size_bytes"]
        and _sha256_bytes(encoded) == expected["sha256"],
        "failed-v5 Slurm log changed after intent",
    )
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "short write while preserving failed-v5 log")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, target)
    directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _tree_from_intent(intent: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = intent.get(key)
    _require(isinstance(value, Mapping), f"{key} recovery identity is absent")
    tree = value.get("tree")
    _require(isinstance(tree, Mapping), f"{key} tree identity is absent")
    return dict(tree)


def _rename_transaction(
    *, root: Path, source: Path, destination: Path, expected_tree: Mapping[str, Any]
) -> None:
    prelaunch = root / "outputs/prelaunch"
    _, descriptor = open_secure_directory(prelaunch, root=root, create=False)
    try:
        source_name = source.name
        destination_name = destination.name
        try:
            source_metadata = os.stat(
                source_name, dir_fd=descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            source_metadata = None
        try:
            destination_metadata = os.stat(
                destination_name, dir_fd=descriptor, follow_symlinks=False
            )
        except FileNotFoundError:
            destination_metadata = None
        _require(
            not (source_metadata is not None and destination_metadata is not None),
            f"both live and archived transaction paths exist for {source_name}",
        )
        _require(
            source_metadata is not None or destination_metadata is not None,
            f"both live and archived transaction paths are absent for {source_name}",
        )
        if source_metadata is not None:
            _require(
                stat.S_ISDIR(source_metadata.st_mode)
                and not stat.S_ISLNK(source_metadata.st_mode),
                f"live transaction is unsafe: {source_name}",
            )
            _require(
                _tree_manifest(root / source, label=f"live {source_name}")
                == dict(expected_tree),
                f"live transaction changed before archival: {source_name}",
            )
            os.rename(
                source_name,
                destination_name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
            )
            os.fsync(descriptor)
        _require(
            _tree_manifest(root / destination, label=f"archived {source_name}")
            == dict(expected_tree),
            f"archived transaction changed: {destination_name}",
        )
    finally:
        os.close(descriptor)


def _receipt(root: Path, intent: Mapping[str, Any], progress: Mapping[str, Any]) -> dict[str, Any]:
    lineage = intent["lineage"]
    core = {
        "schema_version": RECEIPT_SCHEMA,
        "intent_path": str(root / INTENT_RELATIVE),
        "intent_file_sha256": _sha256_bytes(
            _read_regular(root / INTENT_RELATIVE, label="failed-attempt recovery intent")
        ),
        "intent_sha256": intent["intent_sha256"],
        "progress_sha256": progress["progress_sha256"],
        "replacement_archive_path": str(root / REPLACEMENT_ARCHIVE_RELATIVE),
        "replacement_archive_tree_sha256": intent["replacement_transaction"]["tree"][
            "tree_sha256"
        ],
        "controller_archive_path": str(root / CONTROLLER_ARCHIVE_RELATIVE),
        "controller_archive_tree_sha256": intent["controller_transaction"]["tree"][
            "tree_sha256"
        ],
        "preserved_log_path": str(root / PRESERVED_LOG_RELATIVE),
        "preserved_log_sha256": intent["failure_log"]["sha256"],
        "failed_replacement_job_id": lineage["replacement_job_id"],
        "finalizer_terminal_completion": dict(
            lineage["finalizer_terminal_completion"]
        ),
        "prior_obsolete_job_id": lineage["obsolete_job_id"],
        "prior_obsolete_cancellation_receipt_file_sha256": lineage[
            "obsolete_cancellation_receipt_file_sha256"
        ],
        "prior_obsolete_cancellation_receipt_sha256": lineage[
            "obsolete_cancellation_receipt_sha256"
        ],
        "prior_obsolete_terminal_scheduler": dict(
            lineage["prior_obsolete_terminal_scheduler"]
        ),
        "prior_scancel_was_exactly_scoped": lineage[
            "prior_scancel_was_exactly_scoped"
        ],
        "terminal_jobs": list(intent["terminal_jobs"]),
        "required_outputs_absent_at_rotation": True,
        "old_transactions_preserved_byte_for_byte": True,
        "scheduler_mutations_performed": False,
        "connectivity_outcome_values_read": False,
        "final_connectivity_accessed": False,
    }
    return {**core, "receipt_sha256": digest_object(core)}


def rotate_failed_post36_attempt(
    *,
    episode_root: Path,
    terminal_resolver: TerminalResolver = default_terminal_resolver,
) -> dict[str, Any]:
    root = _lexical(episode_root)
    _require(root == EPISODE_ROOT, "failed-attempt recovery root is not canonical")
    transaction, transaction_fd = open_secure_directory(
        root / RECOVERY_ROOT_RELATIVE, root=root, create=True
    )
    lock_fd = open_regular_lock_at(transaction_fd, ".lock")
    os.close(transaction_fd)
    with os.fdopen(lock_fd, "a+b") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        try:
            try:
                existing = _read_json(
                    root / RECEIPT_RELATIVE,
                    label="failed-attempt recovery receipt",
                )
            except FileNotFoundError:
                existing = None
            if existing is not None:
                return verify_failed_post36_attempt_recovery(episode_root=root)

            try:
                intent = _validated_intent(root)
            except FileNotFoundError:
                _assert_outputs_absent(root)
                submission_user = _effective_user()
                terminal_rows = _validated_terminal_rows(
                    terminal_resolver(submission_user),
                    submission_user=submission_user,
                )
                failure_log = _read_regular(FAILED_LOG_PATH, label="failed-v5 Slurm log")
                _require(
                    EXPECTED_FAILURE_TEXT.encode("utf-8") in failure_log,
                    "failed-v5 Slurm log has a different failure",
                )
                lineage = _lineage(root)
                intent = _intent(
                    root=root,
                    submission_user=submission_user,
                    terminal_rows=terminal_rows,
                    lineage=lineage,
                    replacement_tree=_tree_manifest(
                        root / REPLACEMENT_SOURCE_RELATIVE,
                        label="failed-v5 replacement transaction",
                    ),
                    controller_tree=_tree_manifest(
                        root / CONTROLLER_SOURCE_RELATIVE,
                        label="failed-v5 controller transaction",
                    ),
                    failure_log=failure_log,
                )
                _atomic_json(root / INTENT_RELATIVE, intent, root=root)
            _write_progress(root, intent, "intent_persisted")
            _assert_outputs_absent(root)
            _preserve_log(root, intent)
            _write_progress(root, intent, "log_preserved")
            _rename_transaction(
                root=root,
                source=REPLACEMENT_SOURCE_RELATIVE,
                destination=REPLACEMENT_ARCHIVE_RELATIVE,
                expected_tree=_tree_from_intent(intent, "replacement_transaction"),
            )
            _write_progress(root, intent, "replacement_archived")
            _rename_transaction(
                root=root,
                source=CONTROLLER_SOURCE_RELATIVE,
                destination=CONTROLLER_ARCHIVE_RELATIVE,
                expected_tree=_tree_from_intent(intent, "controller_transaction"),
            )
            progress = _progress(intent=intent, state="transactions_archived")
            _atomic_json(root / PROGRESS_RELATIVE, progress, root=root)
            receipt = _receipt(root, intent, progress)
            _atomic_json(root / RECEIPT_RELATIVE, receipt, root=root)
            return verify_failed_post36_attempt_recovery(
                episode_root=root, require_outputs_absent=True
            )
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


def verify_failed_post36_attempt_recovery(
    *, episode_root: Path, require_outputs_absent: bool = False
) -> dict[str, Any]:
    root = _lexical(episode_root)
    _require(root == EPISODE_ROOT, "failed-attempt recovery root is not canonical")
    intent = _validated_intent(root)
    progress = _read_json(
        root / PROGRESS_RELATIVE, label="failed-attempt recovery progress"
    )
    _require(
        progress == _progress(intent=intent, state="transactions_archived"),
        "failed-attempt recovery progress changed",
    )
    _require(
        _tree_manifest(
            root / REPLACEMENT_ARCHIVE_RELATIVE,
            label="archived failed-v5 replacement transaction",
        )
        == _tree_from_intent(intent, "replacement_transaction"),
        "archived failed-v5 replacement transaction changed",
    )
    _require(
        _tree_manifest(
            root / CONTROLLER_ARCHIVE_RELATIVE,
            label="archived failed-v5 controller transaction",
        )
        == _tree_from_intent(intent, "controller_transaction"),
        "archived failed-v5 controller transaction changed",
    )
    preserved_log = _read_regular(
        root / PRESERVED_LOG_RELATIVE, label="preserved failed-v5 log"
    )
    _require(
        len(preserved_log) == intent["failure_log"]["size_bytes"]
        and _sha256_bytes(preserved_log) == intent["failure_log"]["sha256"],
        "preserved failed-v5 log changed",
    )
    if require_outputs_absent:
        _assert_outputs_absent(root)
    receipt = _read_json(root / RECEIPT_RELATIVE, label="failed-attempt recovery receipt")
    expected = _receipt(root, intent, progress)
    _require(receipt == expected, "failed-attempt recovery receipt changed")
    prior_cancellation_path = (
        root
        / REPLACEMENT_ARCHIVE_RELATIVE
        / "obsolete_cancellation/receipt.json"
    )
    prior_cancellation, prior_file_sha = _self_hashed_json(
        prior_cancellation_path,
        label="archived failed-v5 cancellation receipt",
        hash_field="receipt_sha256",
    )
    _require(
        prior_file_sha
        == receipt["prior_obsolete_cancellation_receipt_file_sha256"]
        and prior_cancellation["receipt_sha256"]
        == receipt["prior_obsolete_cancellation_receipt_sha256"],
        "archived failed-v5 cancellation receipt changed",
    )
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "recovery_receipt_path": str(root / RECEIPT_RELATIVE),
        "recovery_receipt_file_sha256": _sha256_bytes(
            _read_regular(root / RECEIPT_RELATIVE, label="failed-attempt recovery receipt")
        ),
        "recovery_receipt_sha256": receipt["receipt_sha256"],
        "failed_replacement_job_id": receipt["failed_replacement_job_id"],
        "finalizer_terminal_completion": dict(
            receipt["finalizer_terminal_completion"]
        ),
        "prior_obsolete_job_id": receipt["prior_obsolete_job_id"],
        "prior_obsolete_cancellation_receipt_path": str(prior_cancellation_path),
        "prior_obsolete_cancellation_receipt_file_sha256": prior_file_sha,
        "prior_obsolete_cancellation_receipt_sha256": prior_cancellation[
            "receipt_sha256"
        ],
        "prior_terminal_scheduler_state": prior_cancellation[
            "terminal_scheduler_state"
        ],
        "prior_obsolete_terminal_scheduler": dict(
            receipt["prior_obsolete_terminal_scheduler"]
        ),
        "prior_scancel_was_exactly_scoped": receipt[
            "prior_scancel_was_exactly_scoped"
        ],
        "replacement_archive_path": receipt["replacement_archive_path"],
        "controller_archive_path": receipt["controller_archive_path"],
        "terminal_jobs": list(receipt["terminal_jobs"]),
        "required_outputs_absent_at_rotation": True,
        "old_transactions_preserved_byte_for_byte": True,
        "scheduler_mutations_performed": False,
        "connectivity_outcome_values_read": False,
        "final_connectivity_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("rotate", "verify"):
        child = subparsers.add_parser(name)
        child.add_argument("--episode-root", type=Path, default=EPISODE_ROOT)
    arguments = parser.parse_args(argv)
    if arguments.command == "rotate":
        result = rotate_failed_post36_attempt(episode_root=arguments.episode_root)
    else:
        result = verify_failed_post36_attempt_recovery(
            episode_root=arguments.episode_root
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROLLER_ARCHIVE_RELATIVE",
    "CONTROLLER_SOURCE_RELATIVE",
    "FAILED_JOBS",
    "FailedAttemptRecoveryError",
    "RECEIPT_RELATIVE",
    "RECOVERY_ROOT_RELATIVE",
    "REPLACEMENT_ARCHIVE_RELATIVE",
    "REPLACEMENT_SOURCE_RELATIVE",
    "rotate_failed_post36_attempt",
    "verify_failed_post36_attempt_recovery",
]
