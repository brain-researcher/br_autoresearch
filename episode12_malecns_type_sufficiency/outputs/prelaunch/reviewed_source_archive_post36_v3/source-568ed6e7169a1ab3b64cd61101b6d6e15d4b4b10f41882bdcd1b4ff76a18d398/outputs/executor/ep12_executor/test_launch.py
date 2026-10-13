"""Crash-safe, exactly-once launcher for the frozen EP12 executor test job."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import stat
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .policy import canonical_json, digest_object
from .source_freeze import (
    SOURCE_FREEZE_RELATIVE,
    create_source_freeze_manifest,
    secure_source_path,
    source_freeze_file_sha256,
    verify_source_freeze_manifest,
)
from .slurm_client import sanitized_slurm_client_environment
from .synthetic import EPISODE_ROOT


INTENT_SCHEMA = "ep12.executor_test_launch_intent.v1"
PROGRESS_SCHEMA = "ep12.executor_test_launch_progress.v1"
RECEIPT_SCHEMA = "ep12.executor_test_launch_receipt.v1"
TRANSACTION_RELATIVE = Path("outputs/development_run002/test_launch")
INTENT_RELATIVE = TRANSACTION_RELATIVE / "intent.json"
PROGRESS_RELATIVE = TRANSACTION_RELATIVE / "progress.json"
RECEIPT_RELATIVE = TRANSACTION_RELATIVE / "receipt.json"
LOCK_RELATIVE = TRANSACTION_RELATIVE / ".lock"
TEST_WRAPPER_RELATIVE = Path("outputs/executor/slurm/test_executor.sbatch")

_SBATCH_RESULT_RE = re.compile(r"^(?P<job_id>[0-9]+)(?:;[A-Za-z0-9_.-]+)?$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class TestLaunchError(RuntimeError):
    """The frozen executor-test submission is unsafe or ambiguous."""


Submitter = Callable[[Sequence[str]], str]
JobResolver = Callable[[str, str], Mapping[str, str] | None]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TestLaunchError(message)


def _effective_user() -> str:
    try:
        value = pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError) as exc:
        raise TestLaunchError(f"cannot resolve effective OS user: {exc}") from exc
    _require(isinstance(value, str) and value, "effective OS user is empty")
    return value


def _read_bytes(path: Path, *, episode_root: Path, label: str) -> bytes:
    secured = secure_source_path(path, episode_root=episode_root, label=label)
    try:
        descriptor = os.open(secured, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise TestLaunchError(f"cannot open {label} {secured}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{label} is not regular: {secured}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    finally:
        os.close(descriptor)


def _exists(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise TestLaunchError(f"cannot inspect test-launch path {path}: {exc}") from exc
    return True


def _atomic_json(path: Path, value: Mapping[str, Any], *, episode_root: Path) -> None:
    path = secure_source_path(path, episode_root=episode_root, label="test-launch artifact")
    if _exists(path):
        _require(stat.S_ISREG(os.lstat(path).st_mode), f"test-launch artifact is not regular: {path}")
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}-{hashlib.sha256(encoded).hexdigest()[:12]}.tmp"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise TestLaunchError(f"short write while creating {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _read_json(path: Path, *, episode_root: Path) -> Mapping[str, Any]:
    encoded = _read_bytes(path, episode_root=episode_root, label="test-launch artifact")
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TestLaunchError(f"malformed test-launch artifact {path}: {exc}") from exc
    _require(isinstance(value, Mapping), f"test-launch artifact is not an object: {path}")
    _require(
        encoded == (canonical_json(value) + "\n").encode("utf-8"),
        f"test-launch artifact is not canonical JSON: {path}",
    )
    return value


def _paths(episode_root: Path) -> dict[str, Path]:
    _require(episode_root.is_absolute(), "EP12 root must be absolute")
    root = Path(os.path.abspath(os.fspath(episode_root)))
    root = secure_source_path(root, episode_root=root, label="EP12 root")
    _require(root.is_dir(), f"EP12 root is not a directory: {root}")
    paths = {
        "root": root,
        "transaction": root / TRANSACTION_RELATIVE,
        "intent": root / INTENT_RELATIVE,
        "progress": root / PROGRESS_RELATIVE,
        "receipt": root / RECEIPT_RELATIVE,
        "lock": root / LOCK_RELATIVE,
        "manifest": root / SOURCE_FREEZE_RELATIVE,
        "wrapper": root / TEST_WRAPPER_RELATIVE,
    }
    for label, path in paths.items():
        secure_source_path(path, episode_root=root, label=f"test-launch {label}")
    return paths


def _source_identity(episode_root: Path) -> dict[str, str]:
    paths = _paths(episode_root)
    manifest = create_source_freeze_manifest(
        episode_root=paths["root"], manifest_path=paths["manifest"]
    )
    manifest_file_sha256 = source_freeze_file_sha256(
        paths["manifest"], episode_root=paths["root"]
    )
    wrapper_sha256 = hashlib.sha256(
        _read_bytes(paths["wrapper"], episode_root=paths["root"], label="test wrapper")
    ).hexdigest()
    _require(_HEX_RE.fullmatch(manifest_file_sha256) is not None, "source-freeze file digest is invalid")
    return {
        "source_freeze_manifest": str(paths["manifest"]),
        "source_freeze_manifest_sha256": manifest_file_sha256,
        "source_freeze_self_sha256": str(manifest["manifest_sha256"]),
        "test_wrapper_sha256": wrapper_sha256,
    }


def _build_intent(*, episode_root: Path, submission_user: str) -> dict[str, Any]:
    paths = _paths(episode_root)
    _require(submission_user == _effective_user(), "test-launch submission user does not match effective OS user")
    core = {
        "schema_version": INTENT_SCHEMA,
        "submission_user": submission_user,
        "test_wrapper": str(paths["wrapper"]),
        "source_freeze": _source_identity(paths["root"]),
        "real_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    token = digest_object(core)[:32]
    with_name = {**core, "job_name": f"ep12-tests-{token}"}
    return {**with_name, "intent_sha256": digest_object(with_name)}


def _verify_intent(value: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    _require(set(value) == set(expected), "test-launch intent schema changed")
    core = {key: item for key, item in value.items() if key != "intent_sha256"}
    _require(
        value.get("schema_version") == INTENT_SCHEMA
        and value.get("intent_sha256") == digest_object(core),
        "test-launch intent digest mismatch",
    )
    _require(dict(value) == dict(expected), "test-launch intent mutated")
    return dict(value)


def _progress_value(*, intent: Mapping[str, Any], state: str, job_id: str | None) -> dict[str, Any]:
    _require(state in {"planned", "submission_uncertain", "submitted"}, "invalid test-launch state")
    if state == "submitted":
        _require(isinstance(job_id, str) and job_id.isdigit() and int(job_id) > 0, "submitted test launch lacks job ID")
    else:
        _require(job_id is None, f"{state} test launch unexpectedly has job ID")
    core = {
        "schema_version": PROGRESS_SCHEMA,
        "intent_sha256": intent["intent_sha256"],
        "job_name": intent["job_name"],
        "submission_user": intent["submission_user"],
        "state": state,
        "slurm_job_id": job_id,
        "real_connectivity_accessed": False,
    }
    return {**core, "progress_sha256": digest_object(core)}


def _verify_progress(value: Mapping[str, Any], *, intent: Mapping[str, Any]) -> tuple[str, str | None, dict[str, Any]]:
    expected_keys = set(_progress_value(intent=intent, state="planned", job_id=None))
    _require(set(value) == expected_keys, "test-launch progress schema changed")
    core = {key: item for key, item in value.items() if key != "progress_sha256"}
    _require(value.get("progress_sha256") == digest_object(core), "test-launch progress digest mismatch")
    state = str(value.get("state"))
    raw_job = value.get("slurm_job_id")
    rebuilt = _progress_value(
        intent=intent,
        state=state,
        job_id=str(raw_job) if raw_job is not None else None,
    )
    _require(dict(value) == rebuilt, "test-launch progress mutated")
    return state, rebuilt["slurm_job_id"], rebuilt


def _export_argument(intent: Mapping[str, Any]) -> str:
    source = intent["source_freeze"]
    _require(isinstance(source, Mapping), "test-launch source identity is missing")
    return ",".join(
        (
            f"EP12_SOURCE_FREEZE_MANIFEST={source['source_freeze_manifest']}",
            f"EP12_SOURCE_FREEZE_MANIFEST_SHA256={source['source_freeze_manifest_sha256']}",
        )
    )


def _submission_command(intent: Mapping[str, Any]) -> list[str]:
    return [
        "sbatch",
        "--parsable",
        f"--job-name={intent['job_name']}",
        f"--export={_export_argument(intent)}",
        str(intent["test_wrapper"]),
    ]


def _receipt_value(*, intent: Mapping[str, Any], progress: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    _require(progress.get("state") == "submitted" and progress.get("slurm_job_id") == job_id, "test-launch receipt/progress mismatch")
    core = {
        "schema_version": RECEIPT_SCHEMA,
        "submission_user": intent["submission_user"],
        "slurm_job_id": job_id,
        "slurm_job_name": intent["job_name"],
        "test_wrapper": intent["test_wrapper"],
        "source_freeze": dict(intent["source_freeze"]),
        "sbatch_command": _submission_command(intent),
        "intent_sha256": intent["intent_sha256"],
        "progress_sha256": progress["progress_sha256"],
        "real_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "receipt_sha256": digest_object(core)}


def verify_test_launch_receipt(value: Mapping[str, Any], *, episode_root: Path) -> dict[str, Any]:
    paths = _paths(episode_root)
    job_id = value.get("slurm_job_id")
    _require(isinstance(job_id, str) and job_id.isdigit() and int(job_id) > 0, "test-launch receipt job ID is invalid")
    user = value.get("submission_user")
    _require(user == _effective_user(), "test-launch receipt user does not match effective OS user")
    source = value.get("source_freeze")
    _require(isinstance(source, Mapping), "test-launch receipt lacks source freeze")
    expected_source_keys = {
        "source_freeze_manifest",
        "source_freeze_manifest_sha256",
        "source_freeze_self_sha256",
        "test_wrapper_sha256",
    }
    _require(set(source) == expected_source_keys, "test-launch source-freeze schema changed")
    _require(source.get("source_freeze_manifest") == str(paths["manifest"]), "test-launch source-freeze path changed")
    manifest = verify_source_freeze_manifest(
        paths["manifest"],
        episode_root=paths["root"],
        expected_file_sha256=str(source.get("source_freeze_manifest_sha256")),
        require_current_surface=True,
    )
    _require(manifest.get("manifest_sha256") == source.get("source_freeze_self_sha256"), "test-launch source-freeze self hash changed")
    wrapper_hash = hashlib.sha256(
        _read_bytes(paths["wrapper"], episode_root=paths["root"], label="test wrapper")
    ).hexdigest()
    _require(source.get("test_wrapper_sha256") == wrapper_hash, "test wrapper hash changed")
    intent_core = {
        "schema_version": INTENT_SCHEMA,
        "submission_user": user,
        "test_wrapper": str(paths["wrapper"]),
        "source_freeze": dict(source),
        "real_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    expected_name = f"ep12-tests-{digest_object(intent_core)[:32]}"
    intent_with_name = {**intent_core, "job_name": expected_name}
    _require(value.get("slurm_job_name") == expected_name, "test-launch deterministic job name changed")
    _require(value.get("intent_sha256") == digest_object(intent_with_name), "test-launch intent digest mismatch")
    expected_keys = {
        "schema_version",
        "submission_user",
        "slurm_job_id",
        "slurm_job_name",
        "test_wrapper",
        "source_freeze",
        "sbatch_command",
        "intent_sha256",
        "progress_sha256",
        "real_connectivity_accessed",
        "final_connectivity_access_authorized",
        "receipt_sha256",
    }
    _require(set(value) == expected_keys, "test-launch receipt schema changed")
    _require(value.get("schema_version") == RECEIPT_SCHEMA, "test-launch receipt schema mismatch")
    _require(value.get("test_wrapper") == str(paths["wrapper"]), "test-launch wrapper path changed")
    _require(value.get("sbatch_command") == _submission_command(intent_with_name), "test-launch command changed")
    _require(value.get("real_connectivity_accessed") is False and value.get("final_connectivity_access_authorized") is False, "test launch crosses connectivity boundary")
    core = {key: item for key, item in value.items() if key != "receipt_sha256"}
    _require(value.get("receipt_sha256") == digest_object(core), "test-launch receipt digest mismatch")
    return dict(value)


def read_test_launch_receipt(path: Path, *, episode_root: Path) -> dict[str, Any]:
    paths = _paths(episode_root)
    secured = secure_source_path(path, episode_root=paths["root"], label="test-launch receipt")
    _require(secured == paths["receipt"], "test-launch receipt path is not canonical")
    receipt = verify_test_launch_receipt(
        _read_json(secured, episode_root=paths["root"]), episode_root=paths["root"]
    )
    expected_intent = _build_intent(
        episode_root=paths["root"], submission_user=_effective_user()
    )
    intent = _verify_intent(
        _read_json(paths["intent"], episode_root=paths["root"]), expected_intent
    )
    state, job_id, progress = _verify_progress(
        _read_json(paths["progress"], episode_root=paths["root"]), intent=intent
    )
    _require(
        state == "submitted" and isinstance(job_id, str),
        "test-launch receipt lacks canonical submitted progress",
    )
    expected = _receipt_value(intent=intent, progress=progress, job_id=job_id)
    _require(receipt == expected, "test-launch receipt differs from its transaction")
    return receipt


def _default_submitter(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=sanitized_slurm_client_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TestLaunchError(f"executor-test sbatch failed: {getattr(exc, 'stderr', '')}") from exc
    return completed.stdout.strip()


def _parse_job_id(output: str) -> str:
    match = _SBATCH_RESULT_RE.fullmatch(output.strip())
    if match is None:
        raise TestLaunchError(f"unexpected executor-test sbatch --parsable output: {output!r}")
    return match.group("job_id")


def _unique_recovery(output: str, expected_name: str, expected_user: str) -> Mapping[str, str] | None:
    job_ids: set[str] = set()
    for raw in output.splitlines():
        if not raw.strip():
            continue
        fields = raw.strip().split("|")
        _require(len(fields) == 3, "malformed executor-test recovery row")
        job_id, name, user = fields
        _require(name == expected_name, "executor-test recovery returned mismatched job name")
        _require(user == expected_user, "executor-test recovery returned mismatched user")
        if job_id.isdigit():
            job_ids.add(job_id)
        else:
            base, separator, _step = job_id.partition(".")
            _require(separator == "." and base.isdigit(), "invalid executor-test recovery job ID")
    _require(len(job_ids) <= 1, "ambiguous executor-test recovery")
    if not job_ids:
        return None
    return {"job_id": next(iter(job_ids)), "job_name": expected_name, "user": expected_user}


def _default_job_resolver(job_name: str, submission_user: str) -> Mapping[str, str] | None:
    """Use one accounting query for exact-name lost-ack recovery."""

    _require(submission_user == _effective_user(), "executor-test recovery user mismatch")
    command = [
        "sacct",
        "-X",
        "--noheader",
        "--parsable2",
        "--user",
        submission_user,
        "--name",
        job_name,
        "--starttime",
        "2000-01-01",
        "--format=JobIDRaw,JobName%128,User%64",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=sanitized_slurm_client_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TestLaunchError(
            f"executor-test recovery query failed: {getattr(exc, 'stderr', '')}"
        ) from exc
    return _unique_recovery(completed.stdout, job_name, submission_user)


def _validated_recovery(value: Mapping[str, str] | None, *, job_name: str, submission_user: str) -> str | None:
    if value is None:
        return None
    _require(isinstance(value, Mapping) and set(value) == {"job_id", "job_name", "user"}, "executor-test recovery schema changed")
    _require(value.get("job_name") == job_name, "executor-test recovery returned mismatched job name")
    _require(value.get("user") == submission_user, "executor-test recovery returned mismatched user")
    job_id = value.get("job_id")
    _require(isinstance(job_id, str) and job_id.isdigit() and int(job_id) > 0, "executor-test recovery job ID is invalid")
    return job_id


def launch_executor_tests(
    *,
    episode_root: Path = EPISODE_ROOT,
    submitter: Submitter = _default_submitter,
    job_resolver: JobResolver = _default_job_resolver,
) -> dict[str, Any]:
    paths = _paths(episode_root)
    paths["transaction"].mkdir(parents=True, exist_ok=True)
    secure_source_path(paths["transaction"], episode_root=paths["root"], label="test-launch transaction")
    submission_user = _effective_user()
    expected_intent = _build_intent(episode_root=paths["root"], submission_user=submission_user)
    try:
        lock_descriptor = os.open(paths["lock"], os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o640)
    except OSError as exc:
        raise TestLaunchError(f"cannot open test-launch lock: {exc}") from exc
    if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
        os.close(lock_descriptor)
        raise TestLaunchError("test-launch lock is not regular")
    with os.fdopen(lock_descriptor, "a+b") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        try:
            if _exists(paths["intent"]):
                intent = _verify_intent(_read_json(paths["intent"], episode_root=paths["root"]), expected_intent)
            else:
                _atomic_json(paths["intent"], expected_intent, episode_root=paths["root"])
                intent = expected_intent
            if _exists(paths["progress"]):
                state, job_id, progress = _verify_progress(_read_json(paths["progress"], episode_root=paths["root"]), intent=intent)
            else:
                progress = _progress_value(intent=intent, state="planned", job_id=None)
                _atomic_json(paths["progress"], progress, episode_root=paths["root"])
                state, job_id = "planned", None
            if _exists(paths["receipt"]):
                _require(state == "submitted" and job_id is not None, "test-launch receipt lacks submitted progress")
                return read_test_launch_receipt(paths["receipt"], episode_root=paths["root"])
            if state in {"planned", "submission_uncertain"}:
                job_id = _validated_recovery(job_resolver(str(intent["job_name"]), submission_user), job_name=str(intent["job_name"]), submission_user=submission_user)
                if job_id is None and state == "submission_uncertain":
                    raise TestLaunchError("executor-test submission remains uncertain; refusing duplicate sbatch")
                if job_id is None:
                    progress = _progress_value(intent=intent, state="submission_uncertain", job_id=None)
                    _atomic_json(paths["progress"], progress, episode_root=paths["root"])
                    job_id = _parse_job_id(submitter(_submission_command(intent)))
                progress = _progress_value(intent=intent, state="submitted", job_id=job_id)
                _atomic_json(paths["progress"], progress, episode_root=paths["root"])
                state = "submitted"
            _require(state == "submitted" and isinstance(job_id, str), "executor-test launch did not reach submission")
            receipt = _receipt_value(intent=intent, progress=progress, job_id=job_id)
            _atomic_json(paths["receipt"], receipt, episode_root=paths["root"])
            return read_test_launch_receipt(paths["receipt"], episode_root=paths["root"])
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(canonical_json(launch_executor_tests()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
