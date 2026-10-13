"""Exactly-once Slurm launch after one exact EP12 resource authorization.

This is the only post-hold launcher.  It binds the historical pending receipt,
the explicit authorization, the executable source freeze, and the exact
procedure-lock wrapper before contacting Slurm.  A deterministic same-user job
name permits lost-acknowledgement recovery without a duplicate submission.
"""

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
from .slurm_client import sanitized_slurm_client_environment
from .source_freeze import (
    source_freeze_file_sha256,
    verify_source_freeze_manifest,
)


EPISODE_ROOT = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency"
)
RUN_ROOT = EPISODE_ROOT / "outputs" / "development_run002"
TRANSACTION_ROOT = RUN_ROOT / "resource_release_launch"
INTENT_PATH = TRANSACTION_ROOT / "intent.json"
PROGRESS_PATH = TRANSACTION_ROOT / "progress.json"
RECEIPT_PATH = TRANSACTION_ROOT / "receipt.json"
LOCK_PATH = TRANSACTION_ROOT / ".lock"
PENDING_PATH = RUN_ROOT / "pre_null_dispatch_pending" / "resource_authorization_pending.json"
PROCEDURE_WRAPPER = EPISODE_ROOT / "outputs" / "executor" / "slurm" / "run_procedure_lock.sbatch"
PROCEDURE_LOCK_PATH = RUN_ROOT / "procedure_lock" / "procedure_lock.json"
RUNTIME_BUNDLE = Path(
    "/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/"
    "development_run002/full_search_runtime_bundle"
)
POLICY_PATH = EPISODE_ROOT / "SEARCH_POLICY.yaml"

INTENT_SCHEMA = "ep12.resource_release_launch_intent.v1"
PROGRESS_SCHEMA = "ep12.resource_release_launch_progress.v1"
RECEIPT_SCHEMA = "ep12.resource_release_launch_receipt.v1"
AUTHORIZATION_SCHEMA_VERSION = "ep12.full_search_null_resource_authorization.v1"
PENDING_SCHEMA_VERSION = "ep12.resource_authorization_pending.v1"
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_SBATCH_RE = re.compile(r"^(?P<job_id>[0-9]+)(?:;[A-Za-z0-9_.-]+)?$")


class ResourceReleaseLaunchError(RuntimeError):
    """The authorized release cannot be launched without ambiguity."""


class SubmissionNotInvokedError(ResourceReleaseLaunchError):
    """The local client could not create an ``sbatch`` process."""


Submitter = Callable[[Sequence[str]], str]
JobResolver = Callable[[str], str | None]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResourceReleaseLaunchError(message)


def _effective_user() -> str:
    try:
        value = pwd.getpwuid(os.geteuid()).pw_name
    except (KeyError, OSError) as exc:
        raise ResourceReleaseLaunchError("cannot resolve effective OS user") from exc
    _require(bool(value) and "|" not in value, "effective OS user is invalid")
    return value


def _safe_path(path: Path, *, root: Path, label: str, must_exist: bool = True) -> Path:
    raw = Path(os.path.abspath(os.fspath(path)))
    boundary = root.resolve(strict=True)
    _require(path.is_absolute() and ".." not in path.parts, f"{label} path is unsafe")
    _require(raw == boundary or boundary in raw.parents, f"{label} escapes EP12")
    current = Path(raw.anchor)
    for component in raw.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            _require(not must_exist, f"missing {label}: {raw}")
            break
        _require(not stat.S_ISLNK(metadata.st_mode), f"{label} traverses a symlink")
    if must_exist:
        metadata = os.lstat(raw)
        _require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file")
    return raw


def _read_bytes(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ResourceReleaseLaunchError(f"cannot open {label}: {exc}") from exc
    try:
        _require(stat.S_ISREG(os.fstat(descriptor).st_mode), f"{label} is not regular")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                return b"".join(blocks)
            blocks.append(block)
    finally:
        os.close(descriptor)


def _json_file(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    encoded = _read_bytes(path, label=label)
    try:
        value = json.loads(encoded)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ResourceReleaseLaunchError(f"invalid {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    _require(
        encoded == (canonical_json(value) + "\n").encode(),
        f"{label} is not canonical JSON",
    )
    return value, encoded


def _self_hash(value: Mapping[str, Any], field: str, *, label: str) -> None:
    core = {key: item for key, item in value.items() if key != field}
    _require(value.get(field) == digest_object(core), f"{label} self-hash mismatch")


def _verify_full_resource_authorization(
    *, authorization_path: Path, authorization_file_sha256: str,
    pending_path: Path,
) -> dict[str, Any]:
    """Run the scientific resource-plan verifier before any ``sbatch`` call."""

    from .resource_authorization import (
        ResourceAuthorizationError,
        verify_resource_authorization,
    )
    from .policy import EpisodePolicy

    lock_path = _safe_path(
        PROCEDURE_LOCK_PATH, root=RUN_ROOT, label="procedure lock"
    )
    lock, lock_bytes = _json_file(lock_path, label="procedure lock")
    _self_hash(lock, "procedure_lock_sha256", label="procedure lock")
    pending, _ = _json_file(pending_path, label="pending receipt")
    runtime_bundle_sha256 = pending.get("runtime_bundle_sha256")
    _require(
        isinstance(runtime_bundle_sha256, str)
        and _HEX_RE.fullmatch(runtime_bundle_sha256) is not None,
        "pending receipt lacks an exact runtime-bundle identity",
    )
    try:
        return verify_resource_authorization(
            authorization_path=authorization_path,
            authorization_file_sha256=authorization_file_sha256,
            pending_path=pending_path,
            runtime_bundle=RUNTIME_BUNDLE,
            runtime_bundle_sha256=runtime_bundle_sha256,
            procedure_lock_sha256=str(lock["procedure_lock_sha256"]),
            procedure_lock_file_sha256=hashlib.sha256(lock_bytes).hexdigest(),
            policy=EpisodePolicy.load(POLICY_PATH),
            durable_root=RUN_ROOT,
        )
    except (ResourceAuthorizationError, KeyError, OSError, ValueError) as exc:
        raise ResourceReleaseLaunchError(
            f"full resource authorization verification failed: {exc}"
        ) from exc


def _atomic_json(
    path: Path, value: Mapping[str, Any], *, replace_existing: bool = False
) -> None:
    encoded = (canonical_json(value) + "\n").encode()
    if path.exists():
        _require(not path.is_symlink() and path.is_file(), f"unsafe artifact: {path}")
        if not replace_existing:
            _require(path.read_bytes() == encoded, f"existing artifact differs: {path}")
            return
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
            _require(written > 0, f"short write for {path}")
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


def _resource_identity(
    *, authorization_path: Path, authorization_file_sha256: str, pending_path: Path
) -> dict[str, Any]:
    authorization_path = _safe_path(
        authorization_path, root=RUN_ROOT, label="resource authorization"
    )
    pending_path = _safe_path(pending_path, root=RUN_ROOT, label="pending receipt")
    _require(pending_path == PENDING_PATH, "pending receipt path is not canonical")
    _require(_HEX_RE.fullmatch(authorization_file_sha256) is not None, "bad authorization SHA")
    authorization, authorization_bytes = _json_file(
        authorization_path, label="resource authorization"
    )
    pending, pending_bytes = _json_file(pending_path, label="pending receipt")
    _self_hash(authorization, "authorization_sha256", label="resource authorization")
    _self_hash(pending, "receipt_sha256", label="pending receipt")
    authorization_file = hashlib.sha256(authorization_bytes).hexdigest()
    pending_file = hashlib.sha256(pending_bytes).hexdigest()
    full_evidence = _verify_full_resource_authorization(
        authorization_path=authorization_path,
        authorization_file_sha256=authorization_file_sha256,
        pending_path=pending_path,
    )
    _require(authorization_file == authorization_file_sha256, "authorization bytes changed")
    _require(
        authorization.get("schema_version") == AUTHORIZATION_SCHEMA_VERSION
        and authorization.get("allow_full_search_null_dispatch") is True
        and pending.get("schema_version") == PENDING_SCHEMA_VERSION
        and pending.get("action") == "resource_authorization_pending"
        and pending.get("full_search_null_submitted") is False,
        "resource authorization or pending scope is invalid",
    )
    _require(
        authorization.get("pending_receipt_sha256") == pending["receipt_sha256"]
        and authorization.get("pending_receipt_file_sha256") == pending_file
        and authorization.get("procedure_lock_sha256") == pending.get("procedure_lock_sha256")
        and authorization.get("procedure_lock_file_sha256")
        == pending.get("procedure_lock_file_sha256"),
        "authorization is not bound to the exact pending receipt",
    )
    _require(
        full_evidence.get("authorization_path") == str(authorization_path)
        and full_evidence.get("authorization_sha256")
        == authorization["authorization_sha256"]
        and full_evidence.get("authorization_file_sha256") == authorization_file
        and full_evidence.get("pending_path") == str(pending_path)
        and full_evidence.get("pending_receipt_sha256") == pending["receipt_sha256"]
        and full_evidence.get("pending_receipt_file_sha256") == pending_file,
        "full authorization evidence differs from the exact release inputs",
    )
    return {
        "authorization_path": str(authorization_path),
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_file_sha256": authorization_file,
        "pending_path": str(pending_path),
        "pending_receipt_sha256": pending["receipt_sha256"],
        "pending_receipt_file_sha256": pending_file,
        "resource_plan_sha256": full_evidence["resource_plan_sha256"],
        "runtime_bundle_sha256": full_evidence["runtime_bundle_sha256"],
    }


def _source_identity(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    _require(_HEX_RE.fullmatch(expected_file_sha256) is not None, "bad source-freeze SHA")
    try:
        manifest = verify_source_freeze_manifest(
            path,
            episode_root=EPISODE_ROOT,
            expected_file_sha256=expected_file_sha256,
            require_current_surface=True,
        )
        observed = source_freeze_file_sha256(path, episode_root=EPISODE_ROOT)
    except Exception as exc:
        raise ResourceReleaseLaunchError(f"source-freeze verification failed: {exc}") from exc
    _require(observed == expected_file_sha256, "source-freeze bytes changed")
    return {
        "manifest_path": str(path.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": observed,
        "file_count": manifest["file_count"],
    }


def _intent(
    *, authorization_path: Path, authorization_file_sha256: str,
    source_freeze_manifest: Path, source_freeze_manifest_sha256: str,
) -> dict[str, Any]:
    resource = _resource_identity(
        authorization_path=authorization_path,
        authorization_file_sha256=authorization_file_sha256,
        pending_path=PENDING_PATH,
    )
    source = _source_identity(source_freeze_manifest, source_freeze_manifest_sha256)
    wrapper = _safe_path(PROCEDURE_WRAPPER, root=EPISODE_ROOT, label="procedure wrapper")
    wrapper_sha256 = hashlib.sha256(_read_bytes(wrapper, label="procedure wrapper")).hexdigest()
    exports = {
        "EP12_AUTO_LAUNCH_FULL_NULL": "1",
        "EP12_FULL_NULL_RESOURCE_AUTHORIZATION": resource["authorization_path"],
        "EP12_FULL_NULL_RESOURCE_AUTHORIZATION_SHA256": resource[
            "authorization_file_sha256"
        ],
        "EP12_PROCEDURE_LOCK_LAUNCH_MODE": "release",
        "EP12_RESOURCE_AUTHORIZATION_PENDING": resource["pending_path"],
        "EP12_RESOURCE_AUTHORIZATION_PENDING_FILE_SHA256": resource[
            "pending_receipt_file_sha256"
        ],
        "EP12_RESOURCE_RELEASE_INTENT": str(INTENT_PATH),
        "EP12_SOURCE_FREEZE_MANIFEST": source["manifest_path"],
        "EP12_SOURCE_FREEZE_MANIFEST_SHA256": source["manifest_file_sha256"],
    }
    core = {
        "schema_version": INTENT_SCHEMA,
        "effective_user": _effective_user(),
        "resource": resource,
        "source_freeze": source,
        "procedure_wrapper": str(wrapper),
        "procedure_wrapper_sha256": wrapper_sha256,
        "automatic_exports": exports,
        "ambient_environment_inherited": False,
        "duplicate_submission_permitted": False,
        "development_only": True,
        "final_connectivity_accessed": False,
    }
    token = digest_object(core)[:24]
    with_name = {**core, "job_name": f"ep12-resource-release-{token}"}
    return {**with_name, "intent_sha256": digest_object(with_name)}


def _export_argument(intent: Mapping[str, Any]) -> str:
    exports = intent["automatic_exports"]
    _require(isinstance(exports, Mapping), "release intent exports are missing")
    return ",".join(f"{key}={exports[key]}" for key in sorted(exports))


def _progress(intent: Mapping[str, Any], state: str, job_id: str | None) -> dict[str, Any]:
    _require(state in {"planned", "submission_uncertain", "submitted"}, "bad progress state")
    _require((state == "submitted") == bool(job_id), "progress job identity is inconsistent")
    core = {
        "schema_version": PROGRESS_SCHEMA,
        "intent_sha256": intent["intent_sha256"],
        "job_name": intent["job_name"],
        "state": state,
        "slurm_job_id": job_id,
        "duplicate_submission_permitted": False,
        "final_connectivity_accessed": False,
    }
    return {**core, "progress_sha256": digest_object(core)}


def _default_submitter(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command), check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            env=sanitized_slurm_client_environment(),
        )
    except OSError as exc:
        raise SubmissionNotInvokedError(f"sbatch was not invoked: {exc}") from exc
    except subprocess.CalledProcessError as exc:
        raise ResourceReleaseLaunchError(
            f"sbatch result is ambiguous after invocation: {exc.stderr}"
        ) from exc
    return completed.stdout.strip()


def _parse_job_id(value: str) -> str:
    match = _SBATCH_RE.fullmatch(value.strip())
    _require(match is not None, f"unexpected sbatch output: {value!r}")
    return match.group("job_id")


def _default_job_resolver(job_name: str) -> str | None:
    user = _effective_user()
    commands = (
        ["squeue", "--noheader", "--user", user, "--name", job_name,
         "--format=%A|%j|%u"],
        ["sacct", "-X", "--noheader", "--parsable2", "--user", user,
         "--name", job_name, "--starttime", "2000-01-01",
         "--format=JobIDRaw,JobName%64,User"],
    )
    matches: set[str] = set()
    for command in commands:
        try:
            completed = subprocess.run(
                command, check=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
                env=sanitized_slurm_client_environment(),
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ResourceReleaseLaunchError(f"scheduler recovery failed: {exc}") from exc
        for raw in completed.stdout.splitlines():
            fields = raw.strip().split("|")
            if len(fields) != 3 or fields[1] != job_name:
                continue
            _require(fields[2] == user, "release recovery found another user's job")
            if fields[0].isdigit():
                matches.add(fields[0])
    _require(len(matches) <= 1, "multiple same-user jobs carry the release name")
    return next(iter(matches)) if matches else None


def _open_lock(path: Path) -> int:
    descriptor = os.open(
        path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o640
    )
    try:
        _require(
            stat.S_ISREG(os.fstat(descriptor).st_mode),
            "release lock is not regular",
        )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def launch_resource_release(
    *, authorization_path: Path, authorization_file_sha256: str,
    source_freeze_manifest: Path, source_freeze_manifest_sha256: str,
    submitter: Submitter = _default_submitter,
    job_resolver: JobResolver = _default_job_resolver,
) -> dict[str, Any]:
    """Create or recover one exact authorized procedure-lock release job."""

    intent = _intent(
        authorization_path=authorization_path,
        authorization_file_sha256=authorization_file_sha256,
        source_freeze_manifest=source_freeze_manifest,
        source_freeze_manifest_sha256=source_freeze_manifest_sha256,
    )
    TRANSACTION_ROOT.mkdir(parents=True, exist_ok=True)
    _require(not TRANSACTION_ROOT.is_symlink(), "release transaction root is unsafe")
    descriptor = _open_lock(LOCK_PATH)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        _atomic_json(INTENT_PATH, intent)
        if PROGRESS_PATH.exists():
            progress, _ = _json_file(PROGRESS_PATH, label="release progress")
            _self_hash(progress, "progress_sha256", label="release progress")
            _require(progress.get("intent_sha256") == intent["intent_sha256"], "intent changed")
            state = str(progress.get("state"))
            job_id = progress.get("slurm_job_id")
        else:
            state, job_id = "planned", None
            progress = _progress(intent, state, None)
            _atomic_json(PROGRESS_PATH, progress)
        if RECEIPT_PATH.exists():
            receipt, _ = _json_file(RECEIPT_PATH, label="release receipt")
            _self_hash(receipt, "receipt_sha256", label="release receipt")
            _require(
                state == "submitted"
                and receipt.get("intent_sha256") == intent["intent_sha256"]
                and receipt.get("slurm_job_id") == job_id,
                "existing release receipt differs",
            )
            return receipt
        if state in {"planned", "submission_uncertain"}:
            recovered = job_resolver(str(intent["job_name"]))
            if recovered is None and state == "submission_uncertain":
                raise ResourceReleaseLaunchError(
                    "release submission remains uncertain; refusing duplicate sbatch"
                )
            if recovered is None:
                progress = _progress(intent, "submission_uncertain", None)
                _atomic_json(PROGRESS_PATH, progress, replace_existing=True)
                command = [
                    "sbatch", "--parsable", f"--job-name={intent['job_name']}",
                    "--partition=normal", "--cpus-per-task=32", "--mem=192G",
                    "--time=12:00:00", "--requeue", "--open-mode=append",
                    f"--export={_export_argument(intent)}", str(PROCEDURE_WRAPPER),
                ]
                try:
                    recovered = _parse_job_id(submitter(command))
                except SubmissionNotInvokedError:
                    _atomic_json(
                        PROGRESS_PATH,
                        _progress(intent, "planned", None),
                        replace_existing=True,
                    )
                    raise
                except Exception as exc:
                    recovered = job_resolver(str(intent["job_name"]))
                    if recovered is None:
                        raise ResourceReleaseLaunchError(
                            "release acceptance is uncertain; retry may only recover"
                        ) from exc
            job_id = recovered
            _require(isinstance(job_id, str) and job_id.isdigit(), "bad release job ID")
            progress = _progress(intent, "submitted", job_id)
            _atomic_json(PROGRESS_PATH, progress, replace_existing=True)
            state = "submitted"
        _require(state == "submitted" and isinstance(job_id, str), "release was not submitted")
        core = {
            "schema_version": RECEIPT_SCHEMA,
            "intent_sha256": intent["intent_sha256"],
            "progress_sha256": progress["progress_sha256"],
            "job_name": intent["job_name"],
            "slurm_job_id": job_id,
            "resource": intent["resource"],
            "source_freeze": intent["source_freeze"],
            "procedure_wrapper_sha256": intent["procedure_wrapper_sha256"],
            "automatic_exports": intent["automatic_exports"],
            "ambient_environment_inherited": False,
            "submission_accepted_or_same_user_recovered": True,
            "duplicate_submission_permitted": False,
            "final_connectivity_accessed": False,
        }
        receipt = {**core, "receipt_sha256": digest_object(core)}
        _atomic_json(RECEIPT_PATH, receipt)
        return receipt
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def verify_runtime_intent(
    *, intent_path: Path, authorization_path: Path,
    authorization_file_sha256: str, pending_path: Path,
    pending_file_sha256: str, source_freeze_manifest: Path,
    source_freeze_manifest_sha256: str,
) -> dict[str, Any]:
    """Authenticate the release transaction from inside its Slurm job."""

    _require(intent_path == INTENT_PATH, "runtime release intent path changed")
    intent, _ = _json_file(intent_path, label="release intent")
    _self_hash(intent, "intent_sha256", label="release intent")
    expected = _intent(
        authorization_path=authorization_path,
        authorization_file_sha256=authorization_file_sha256,
        source_freeze_manifest=source_freeze_manifest,
        source_freeze_manifest_sha256=source_freeze_manifest_sha256,
    )
    _require(intent == expected, "runtime release intent differs from launch")
    _require(
        pending_path == Path(intent["resource"]["pending_path"])
        and pending_file_sha256 == intent["resource"]["pending_receipt_file_sha256"],
        "runtime pending receipt differs from release intent",
    )
    observed_job_name = os.environ.get("SLURM_JOB_NAME")
    if observed_job_name is not None:
        _require(observed_job_name == intent["job_name"], "release Slurm name changed")
    for name, value in intent["automatic_exports"].items():
        _require(os.environ.get(name) == value, f"runtime export changed: {name}")
    return intent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ep12-resource-release-launch")
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--resource-authorization", type=Path, required=True)
    launch.add_argument("--resource-authorization-file-sha256", required=True)
    launch.add_argument("--source-freeze-manifest", type=Path, required=True)
    launch.add_argument("--source-freeze-manifest-sha256", required=True)
    verify = subparsers.add_parser("verify-runtime")
    verify.add_argument("--intent", type=Path, required=True)
    verify.add_argument("--resource-authorization", type=Path, required=True)
    verify.add_argument("--resource-authorization-file-sha256", required=True)
    verify.add_argument("--resource-authorization-pending", type=Path, required=True)
    verify.add_argument("--resource-authorization-pending-file-sha256", required=True)
    verify.add_argument("--source-freeze-manifest", type=Path, required=True)
    verify.add_argument("--source-freeze-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "launch":
        value = launch_resource_release(
            authorization_path=arguments.resource_authorization,
            authorization_file_sha256=arguments.resource_authorization_file_sha256,
            source_freeze_manifest=arguments.source_freeze_manifest,
            source_freeze_manifest_sha256=arguments.source_freeze_manifest_sha256,
        )
    else:
        value = verify_runtime_intent(
            intent_path=arguments.intent,
            authorization_path=arguments.resource_authorization,
            authorization_file_sha256=arguments.resource_authorization_file_sha256,
            pending_path=arguments.resource_authorization_pending,
            pending_file_sha256=arguments.resource_authorization_pending_file_sha256,
            source_freeze_manifest=arguments.source_freeze_manifest,
            source_freeze_manifest_sha256=arguments.source_freeze_manifest_sha256,
        )
    print(canonical_json(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "INTENT_PATH", "ResourceReleaseLaunchError", "launch_resource_release",
    "verify_runtime_intent",
]
