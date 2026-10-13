"""Unattended, fail-closed dispatch from an authenticated EP12 pre-null lock."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .policy import EpisodePolicy, canonical_json, digest_object
from .full_null_launch_permit import (
    create_full_null_launch_permit,
    derive_full_null_launch_permit,
)
from .procedure_lock import ProcedureLockError, verify_procedure_lock
from .resource_authorization import (
    ResourceAuthorizationError,
    verify_resource_authorization,
    write_resource_authorization_release,
)
from .slurm_client import sanitized_slurm_client_environment
from .secure_transaction import (
    SecureTransactionError,
    open_regular_lock_at,
    open_secure_directory,
)
from .source_freeze import (
    source_freeze_file_sha256,
    verify_source_freeze_manifest,
)


SCHEMA_VERSION = "ep12.pre_null_dispatch.v3"
MANIFEST_SCHEMA_VERSION = "ep12.pre_null_dispatch_manifest.v1"
PROGRESS_SCHEMA_VERSION = "ep12.pre_null_dispatch_progress.v4"
LAUNCH_SCHEMA_VERSION = "ep12.full_search_null_launch.v3"
SCHEDULER_STATUS_SCHEMA_VERSION = "ep12.slurm_exact_job_status.v1"
RECOVERY_SCHEMA_VERSION = "ep12.pre_null_dispatch_recovery.v2"
RECOVERY_PROGRESS_SCHEMA_VERSION = "ep12.pre_null_dispatch_recovery_progress.v2"
CANONICAL_RUNTIME_PROVIDER = (
    "ep12_executor.full_search_runtime:load_runtime_from_environment"
)
EPISODE_ROOT = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency"
)
AGGREGATE_CPUS_PER_TASK = 1
AGGREGATE_MEMORY = "32G"
AGGREGATE_TIME_LIMIT = "12:00:00"
_PROVIDER_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$"
)
_SBATCH_RESULT_RE = re.compile(r"^(?P<job_id>[0-9]+)(?:;[A-Za-z0-9_.-]+)?$")


class ProcedureDispatchError(RuntimeError):
    """The lock cannot be safely dispatched without human reinterpretation."""


Submitter = Callable[[Sequence[str]], str]
JobResolver = Callable[[str], str | None]
JobStatusResolver = Callable[[str, str], Mapping[str, Any]]
CheckpointVerifier = Callable[[int], Mapping[str, Any] | None]

_TERMINAL_STATES = frozenset({
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
    "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProcedureDispatchError(message)


def _effective_user() -> str:
    try:
        value = pwd.getpwuid(os.geteuid()).pw_name
    except (KeyError, OSError) as exc:
        raise ProcedureDispatchError("cannot resolve the effective OS user") from exc
    _require(bool(value) and "|" not in value, "effective OS user is invalid")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}-{hashlib.sha256(encoded).hexdigest()[:12]}.tmp"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ProcedureDispatchError(f"short write while creating {path}")
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


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcedureDispatchError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ProcedureDispatchError(f"JSON artifact is not an object: {path}")
    return value


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
        stderr = getattr(exc, "stderr", "")
        raise ProcedureDispatchError(f"sbatch submission failed: {stderr}") from exc
    return completed.stdout.strip()


def _job_id(output: str) -> str:
    match = _SBATCH_RESULT_RE.fullmatch(output.strip())
    if match is None:
        raise ProcedureDispatchError(f"unexpected sbatch --parsable output: {output!r}")
    return match.group("job_id")


def _resolved_job_id(
    output: str, job_name: str, *, expected_user: str | None = None
) -> str | None:
    matches: set[str] = set()
    for raw in output.splitlines():
        fields = raw.strip().split("|")
        if len(fields) != (3 if expected_user is not None else 2):
            continue
        job_id, observed_name = fields[:2]
        if observed_name == job_name and expected_user is not None:
            _require(
                fields[2] == expected_user,
                f"scheduler recovery name belongs to another user: {fields[2]!r}",
            )
        if observed_name == job_name and job_id.isdigit():
            matches.add(job_id)
    if len(matches) > 1:
        raise ProcedureDispatchError(
            f"multiple scheduler jobs carry recovery name {job_name!r}"
        )
    return next(iter(matches)) if matches else None


def _default_job_resolver(job_name: str) -> str | None:
    """Recover one exact-name submission across live and accounting views."""

    effective_user = _effective_user()
    commands = (
        [
            "squeue", "--noheader", "--user", effective_user,
            "--name", job_name, "--format=%A|%j|%u",
        ],
        [
            "sacct", "-X", "--noheader", "--parsable2", "--user",
            effective_user, "--name", job_name, "--starttime", "2000-01-01",
            "--format=JobIDRaw,JobName%64,User",
        ],
    )
    matches: set[str] = set()
    for command in commands:
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
            stderr = getattr(exc, "stderr", "")
            raise ProcedureDispatchError(
                f"scheduler recovery query failed for {job_name!r}: {stderr}"
            ) from exc
        recovered = _resolved_job_id(
            completed.stdout, job_name, expected_user=effective_user
        )
        if recovered is not None:
            matches.add(recovered)
    _require(
        len(matches) <= 1,
        f"multiple scheduler jobs carry recovery name {job_name!r}",
    )
    return next(iter(matches)) if matches else None


def _scheduler_state(value: Any) -> str:
    state = str(value).strip().split(None, 1)[0].split("+", 1)[0]
    _require(bool(state), "scheduler returned an empty job state")
    return state


def freeze_scheduler_job_status(
    *,
    job_id: str,
    job_name: str,
    parent_state: str,
    task_states: Mapping[int | str, str] | None = None,
) -> dict[str, Any]:
    """Freeze an exact-ID Slurm status used to authorize recovery."""

    _require(str(job_id).isdigit(), "scheduler status job ID is invalid")
    _require(bool(str(job_name)), "scheduler status job name is missing")
    normalized_tasks: dict[str, str] = {}
    for raw_index, raw_state in (task_states or {}).items():
        index = str(raw_index)
        _require(index.isdigit(), "scheduler array task index is invalid")
        _require(index not in normalized_tasks, "scheduler array task is duplicated")
        normalized_tasks[index] = _scheduler_state(raw_state)
    normalized_tasks = dict(
        sorted(normalized_tasks.items(), key=lambda item: int(item[0]))
    )
    normalized_parent = _scheduler_state(parent_state)
    states = [normalized_parent, *normalized_tasks.values()]
    terminal = all(state in _TERMINAL_STATES for state in states)
    successful = terminal and all(state == "COMPLETED" for state in states)
    core = {
        "schema_version": SCHEDULER_STATUS_SCHEMA_VERSION,
        "job_id": str(job_id),
        "job_name": str(job_name),
        "parent_state": normalized_parent,
        "task_states": normalized_tasks,
        "terminal": terminal,
        "successful": successful,
    }
    return {**core, "status_sha256": digest_object(core)}


def _verify_scheduler_job_status(
    value: Mapping[str, Any], *, job_id: str, job_name: str
) -> dict[str, Any]:
    required = {
        "schema_version", "job_id", "job_name", "parent_state",
        "task_states", "terminal", "successful", "status_sha256",
    }
    _require(set(value) == required, "scheduler status schema mismatch")
    core = {key: item for key, item in value.items() if key != "status_sha256"}
    _require(
        value.get("schema_version") == SCHEDULER_STATUS_SCHEMA_VERSION
        and value.get("status_sha256") == digest_object(core),
        "scheduler status digest mismatch",
    )
    rebuilt = freeze_scheduler_job_status(
        job_id=str(value.get("job_id")),
        job_name=str(value.get("job_name")),
        parent_state=str(value.get("parent_state")),
        task_states=value.get("task_states") if isinstance(
            value.get("task_states"), Mapping
        ) else None,
    )
    _require(rebuilt == dict(value), "scheduler status fields are inconsistent")
    _require(
        value.get("job_id") == job_id and value.get("job_name") == job_name,
        "scheduler status belongs to another exact job",
    )
    return rebuilt


def _default_job_status_resolver(job_id: str, job_name: str) -> Mapping[str, Any]:
    """Query one recorded Slurm ID; never infer recovery state by name."""

    effective_user = _effective_user()
    try:
        completed = subprocess.run(
            [
                "sacct", "-X", "--noheader", "--parsable2", "--user",
                effective_user, "-j", job_id,
                "--format=JobIDRaw,JobName%64,State,User",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=sanitized_slurm_client_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "")
        raise ProcedureDispatchError(
            f"exact scheduler query failed for job {job_id}: {stderr}"
        ) from exc
    parents: list[tuple[str, str]] = []
    tasks: dict[int, str] = {}
    task_pattern = re.compile(rf"^{re.escape(job_id)}_(?P<index>[0-9]+)$")
    for raw in completed.stdout.splitlines():
        fields = raw.strip().split("|", 3)
        if len(fields) != 4:
            continue
        observed_id, observed_name, observed_state, observed_user = fields
        _require(
            observed_user == effective_user,
            "exact scheduler query returned another user's job",
        )
        base_id = observed_id.split(".", 1)[0]
        if observed_id == job_id:
            parents.append((observed_name, observed_state))
            continue
        match = task_pattern.fullmatch(base_id)
        if match is None or observed_id != base_id:
            continue
        index = int(match.group("index"))
        _require(index not in tasks, "exact scheduler query duplicated an array task")
        tasks[index] = observed_state
    _require(len(parents) == 1, "exact scheduler query lacks one parent job row")
    observed_name, parent_state = parents[0]
    _require(observed_name == job_name, "exact scheduler job name changed")
    return freeze_scheduler_job_status(
        job_id=job_id,
        job_name=job_name,
        parent_state=parent_state,
        task_states=tasks,
    )


def _write_final_bundle(output_dir: Path, receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    receipt_path = output_dir / "dispatch.json"
    manifest_path = output_dir / "manifest.json"
    encoded = (canonical_json(receipt) + "\n").encode("utf-8")
    manifest_core = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dispatch_sha256": receipt["dispatch_sha256"],
        "files": {"dispatch.json": hashlib.sha256(encoded).hexdigest()},
        "action": receipt["action"],
        "episode_terminal_emitted": False,
        "final_connectivity_accessed": False,
    }
    manifest = dict(manifest_core)
    manifest["manifest_sha256"] = digest_object(manifest_core)
    if receipt_path.exists():
        _require(
            receipt_path.is_file() and _read_json(receipt_path) == receipt,
            "refusing to overwrite a different pre-null dispatch",
        )
    else:
        _atomic_json(receipt_path, receipt)
    if manifest_path.exists():
        _require(
            manifest_path.is_file() and _read_json(manifest_path) == manifest,
            "refusing to overwrite a different pre-null dispatch manifest",
        )
    else:
        _atomic_json(manifest_path, manifest)
    return manifest


def _source_freeze_identity(
    *, manifest_path: Path, manifest_file_sha256: str
) -> dict[str, Any]:
    try:
        manifest = verify_source_freeze_manifest(
            manifest_path,
            episode_root=EPISODE_ROOT,
            expected_file_sha256=manifest_file_sha256,
            require_current_surface=True,
        )
        observed = source_freeze_file_sha256(
            manifest_path, episode_root=EPISODE_ROOT
        )
    except Exception as exc:
        raise ProcedureDispatchError(f"source-freeze verification failed: {exc}") from exc
    _require(observed == manifest_file_sha256, "source-freeze bytes changed")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": observed,
        "manifest_sha256": manifest["manifest_sha256"],
        "file_count": manifest["file_count"],
    }


def _explicit_export(values: Mapping[str, str]) -> str:
    _require(bool(values), "explicit Slurm export set may not be empty")
    entries: list[str] = []
    for name in sorted(values):
        value = values[name]
        _require(
            re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) is not None
            and name not in {"ALL", "NONE"},
            f"invalid explicit Slurm export name: {name!r}",
        )
        _require(
            isinstance(value, str)
            and "," not in value
            and "\n" not in value
            and "\r" not in value,
            f"invalid explicit Slurm export value: {name}",
        )
        entries.append(f"{name}={value}")
    result = ",".join(entries)
    _require("ALL" not in result.split(","), "ambient Slurm export is forbidden")
    return result


def _replace_launch_permit_export(
    exported: str, permit: Mapping[str, Any]
) -> str:
    values: dict[str, str] = {}
    for entry in exported.split(","):
        _require("=" in entry, "existing explicit export is malformed")
        name, value = entry.split("=", 1)
        _require(name not in values, "existing explicit export is duplicated")
        values[name] = value
    values["EP12_FULL_NULL_LAUNCH_PERMIT"] = str(permit["permit_path"])
    values["EP12_FULL_NULL_LAUNCH_PERMIT_SHA256"] = str(
        permit["permit_file_sha256"]
    )
    return _explicit_export(values)


def _launch_tuple(
    *,
    runtime_provider: str,
    full_search_null_script: Path,
    full_search_null_aggregate_script: Path,
    runtime_bundle: Path,
    runtime_bundle_sha256: str,
    procedure_lock_path: Path,
    procedure_lock_sha256: str,
    procedure_lock_file_sha256: str,
    source_freeze: Mapping[str, Any],
    full_null_work_root: Path,
    full_null_durable_root: Path,
    array_spec: str,
    replicate_count: int,
) -> dict[str, Any]:
    main_wrapper_sha256 = hashlib.sha256(
        full_search_null_script.read_bytes()
    ).hexdigest()
    return {
        "schema_version": LAUNCH_SCHEMA_VERSION,
        "runtime_provider": runtime_provider,
        "full_search_null_script": str(full_search_null_script.resolve()),
        "full_search_null_script_sha256": main_wrapper_sha256,
        "full_search_null_aggregate_script": str(
            full_search_null_aggregate_script.resolve()
        ),
        "full_search_null_aggregate_script_sha256": hashlib.sha256(
            full_search_null_aggregate_script.read_bytes()
        ).hexdigest(),
        "runtime_bundle": str(runtime_bundle.resolve()),
        "runtime_bundle_sha256": runtime_bundle_sha256,
        "procedure_lock_path": str(procedure_lock_path.resolve()),
        "procedure_lock_sha256": procedure_lock_sha256,
        "procedure_lock_file_sha256": procedure_lock_file_sha256,
        "source_freeze": dict(source_freeze),
        "full_null_work_root": str(full_null_work_root.resolve()),
        "full_null_durable_root": str(full_null_durable_root.resolve()),
        "array_spec": array_spec,
        "replicate_count": int(replicate_count),
        "aggregate_dependency": "afterok",
        "aggregate_main_wrapper_sha256_export": (
            "EP12_FULL_NULL_MAIN_WRAPPER_SHA256="
            f"{main_wrapper_sha256}"
        ),
        "aggregate_resources": {
            "cpus_per_task": AGGREGATE_CPUS_PER_TASK,
            "memory": AGGREGATE_MEMORY,
            "time_limit": AGGREGATE_TIME_LIMIT,
            "array": False,
            "requeue": False,
        },
        "kill_on_invalid_dependency": True,
    }


def _verify_launch_tuple(value: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    _require(
        dict(value) == dict(expected)
        and value.get("schema_version") == LAUNCH_SCHEMA_VERSION,
        "dispatch launch tuple changed",
    )
    expected_export = (
        "EP12_FULL_NULL_MAIN_WRAPPER_SHA256="
        f"{expected['full_search_null_script_sha256']}"
    )
    _require(
        value.get("aggregate_main_wrapper_sha256_export") == expected_export,
        "dispatch aggregate main-wrapper hash export changed",
    )


def _progress(
    path: Path,
    *,
    launch: Mapping[str, Any],
    state: str,
    array_job_name: str,
    aggregate_job_name: str,
    array_job_id: str | None,
    aggregate_job_id: str | None,
) -> None:
    core = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "launch": dict(launch),
        "launch_sha256": digest_object(launch),
        "state": state,
        "array_job_name": array_job_name,
        "aggregate_job_name": aggregate_job_name,
        "array_job_id": array_job_id,
        "aggregate_job_id": aggregate_job_id,
        "episode_terminal_emitted": False,
        "final_connectivity_accessed": False,
    }
    value = dict(core)
    value["progress_sha256"] = digest_object(core)
    _atomic_json(path, value)


def _verify_progress(
    value: Mapping[str, Any], *, launch: Mapping[str, Any], array_job_name: str,
    aggregate_job_name: str,
) -> tuple[str, str | None, str | None]:
    _require(value.get("schema_version") == PROGRESS_SCHEMA_VERSION,
             "dispatch progress schema mismatch")
    claimed = value.get("progress_sha256")
    core = {key: item for key, item in value.items() if key != "progress_sha256"}
    _require(isinstance(claimed, str) and digest_object(core) == claimed,
             "dispatch progress digest mismatch")
    launch_value = value.get("launch")
    _require(isinstance(launch_value, Mapping), "dispatch progress lacks launch tuple")
    _verify_launch_tuple(launch_value, launch)
    _require(
        value.get("launch_sha256") == digest_object(launch),
        "dispatch progress launch digest mismatch",
    )
    _require(
        value.get("array_job_name") == array_job_name
        and value.get("aggregate_job_name") == aggregate_job_name,
        "dispatch progress scheduler recovery names changed",
    )
    _require(value.get("final_connectivity_accessed") is False,
             "dispatch progress records final-connectivity access")
    state = value.get("state")
    _require(
        state in {
            "planned",
            "array_submission_uncertain",
            "array_submitted",
            "aggregate_submission_uncertain",
            "aggregate_submitted",
        },
        "invalid dispatch progress state",
    )
    array_job_id = value.get("array_job_id")
    aggregate_job_id = value.get("aggregate_job_id")
    if state in {
        "array_submitted",
        "aggregate_submission_uncertain",
        "aggregate_submitted",
    }:
        _require(isinstance(array_job_id, str) and array_job_id.isdigit(),
                 "array-submitted progress lacks a valid job ID")
    else:
        _require(
            array_job_id is None,
            "pre-array-submission progress unexpectedly has a job ID",
        )
    if state == "aggregate_submitted":
        _require(
            isinstance(aggregate_job_id, str) and aggregate_job_id.isdigit(),
            "aggregate-submitted progress lacks a valid job ID",
        )
    else:
        _require(
            aggregate_job_id is None,
            "pre-aggregate progress unexpectedly has an aggregate job ID",
        )
    return str(state), array_job_id, aggregate_job_id


def _recover_named_submission(
    *,
    job_name: str,
    job_resolver: JobResolver,
    job_status_resolver: JobStatusResolver,
) -> str | None:
    job_id = job_resolver(job_name)
    if job_id is None:
        return None
    _require(job_id.isdigit(), "scheduler name recovery returned an invalid job ID")
    status = _verify_scheduler_job_status(
        job_status_resolver(job_id, job_name),
        job_id=job_id,
        job_name=job_name,
    )
    _require(
        not (status["terminal"] is True and status["successful"] is False),
        "scheduler name recovery found a terminal failed job; explicit recovery required",
    )
    return job_id


def _checkpoint_inventory(
    *,
    work_root: Path,
    replicate_count: int,
    runtime_procedure_lock_sha256: str,
    checkpoint_verifier: CheckpointVerifier,
) -> dict[str, Any]:
    completed: dict[str, dict[str, str]] = {}
    for index in range(replicate_count):
        path = work_root / "replicates" / f"null_{index + 1:03d}" / "checkpoint.json"
        verified = checkpoint_verifier(index)
        if verified is None:
            _require(not path.exists(),
                     "checkpoint verifier omitted an existing checkpoint")
            continue
        _require(path.is_file() and not path.is_symlink(),
                 "replicate checkpoint path is not a regular file")
        checkpoint = dict(verified)
        claimed = checkpoint.get("checkpoint_sha256")
        core = {
            key: item for key, item in checkpoint.items()
            if key != "checkpoint_sha256"
        }
        _require(
            isinstance(claimed, str)
            and digest_object(core) == claimed
            and checkpoint.get("complete") is True
            and int(checkpoint.get("replicate_index", -1)) == index
            and checkpoint.get("replicate_id") == f"null_{index + 1:03d}"
            and checkpoint.get("procedure_lock_sha256")
            == runtime_procedure_lock_sha256,
            f"replicate {index} checkpoint is not an authenticated completion",
        )
        completed[str(index)] = {
            "checkpoint_sha256": claimed,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    core = {
        "replicate_count": replicate_count,
        "runtime_procedure_lock_sha256": runtime_procedure_lock_sha256,
        "completed": completed,
        "completed_indices": [int(index) for index in completed],
    }
    return {**core, "inventory_sha256": digest_object(core)}


def _production_checkpoint_verifier(
    *, launch: Mapping[str, Any], policy: EpisodePolicy
) -> CheckpointVerifier:
    """Load the bound runtime once and fully authenticate each completion."""

    from .full_search_null import authenticate_null_replicate_checkpoint
    from .full_search_runtime import load_runtime_bundle

    runtime = load_runtime_bundle(
        Path(str(launch["runtime_bundle"])),
        policy=policy,
        artifact_root=Path(str(launch["full_null_work_root"])),
        expected_bundle_sha256=str(launch["runtime_bundle_sha256"]),
        upstream_procedure_lock_sha256=str(launch["procedure_lock_sha256"]),
        upstream_procedure_lock_file_sha256=str(
            launch["procedure_lock_file_sha256"]
        ),
    )

    def verify(index: int) -> Mapping[str, Any] | None:
        return authenticate_null_replicate_checkpoint(
            runtime,
            policy=policy,
            replicate_index=index,
            output_dir=Path(str(launch["full_null_work_root"])),
        )

    return verify


def _compressed_array_spec(indices: Sequence[int]) -> str:
    normalized = sorted(set(map(int, indices)))
    _require(bool(normalized), "retry array requires at least one task")
    ranges: list[str] = []
    start = previous = normalized[0]
    for index in normalized[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _aggregate_submit_command(
    *,
    job_name: str,
    exported: str,
    aggregate_script: Path,
    dependency_job_id: str | None,
) -> list[str]:
    """Build the one explicit, non-array aggregate allocation."""

    command = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        "--partition=normal",
        f"--cpus-per-task={AGGREGATE_CPUS_PER_TASK}",
        f"--mem={AGGREGATE_MEMORY}",
        f"--time={AGGREGATE_TIME_LIMIT}",
        "--no-requeue",
        "--open-mode=append",
    ]
    if dependency_job_id is not None:
        _require(dependency_job_id.isdigit(), "aggregate dependency job ID is invalid")
        command.extend([
            f"--dependency=afterok:{dependency_job_id}",
            "--kill-on-invalid-dep=yes",
        ])
    command.extend([
        f"--export={exported}",
        str(aggregate_script.resolve()),
        "aggregate",
    ])
    return command


def _aggregate_exported(
    *, exported: str, launch: Mapping[str, Any]
) -> str:
    """Append the launch-bound main-wrapper hash required by the shim."""

    expected = launch.get("aggregate_main_wrapper_sha256_export")
    _require(
        isinstance(expected, str)
        and expected
        == (
            "EP12_FULL_NULL_MAIN_WRAPPER_SHA256="
            f"{launch.get('full_search_null_script_sha256')}"
        ),
        "aggregate main-wrapper hash export is not bound to the launch",
    )
    return f"{exported},{expected}"


def _recovery_progress(
    path: Path,
    *,
    launch: Mapping[str, Any],
    source_array_status: Mapping[str, Any] | None,
    source_aggregate_status: Mapping[str, Any],
    checkpoint_inventory: Mapping[str, Any],
    generation: int,
    retry_indices: Sequence[int],
    retry_array_spec: str | None,
    retry_array_job_name: str | None,
    retry_aggregate_job_name: str,
    state: str,
    retry_array_job_id: str | None,
    retry_aggregate_job_id: str | None,
) -> None:
    core = {
        "schema_version": RECOVERY_PROGRESS_SCHEMA_VERSION,
        "launch": dict(launch),
        "launch_sha256": digest_object(launch),
        "source_array_status": (
            dict(source_array_status) if source_array_status is not None else None
        ),
        "source_aggregate_status": dict(source_aggregate_status),
        "checkpoint_inventory": dict(checkpoint_inventory),
        "generation": int(generation),
        "retry_indices": list(map(int, retry_indices)),
        "retry_array_spec": retry_array_spec,
        "retry_array_job_name": retry_array_job_name,
        "retry_aggregate_job_name": retry_aggregate_job_name,
        "state": state,
        "retry_array_job_id": retry_array_job_id,
        "retry_aggregate_job_id": retry_aggregate_job_id,
        "episode_terminal_emitted": False,
        "final_connectivity_accessed": False,
    }
    _atomic_json(path, {**core, "progress_sha256": digest_object(core)})


def _verify_recovery_progress(
    value: Mapping[str, Any], *, launch: Mapping[str, Any]
) -> dict[str, Any]:
    claimed = value.get("progress_sha256")
    core = {key: item for key, item in value.items() if key != "progress_sha256"}
    _require(
        value.get("schema_version") == RECOVERY_PROGRESS_SCHEMA_VERSION
        and isinstance(claimed, str)
        and claimed == digest_object(core),
        "dispatch recovery progress digest mismatch",
    )
    launch_value = value.get("launch")
    _require(isinstance(launch_value, Mapping), "recovery progress lacks launch tuple")
    _verify_launch_tuple(launch_value, launch)
    _require(
        value.get("launch_sha256") == digest_object(launch)
        and isinstance(value.get("generation"), int)
        and int(value["generation"]) >= 1
        and value.get("state") in {
            "planned",
            "retry_array_submission_uncertain",
            "retry_array_submitted",
            "retry_aggregate_submission_uncertain",
            "retry_aggregate_submitted",
        }
        and value.get("episode_terminal_emitted") is False
        and value.get("final_connectivity_accessed") is False,
        "dispatch recovery progress fields are invalid",
    )
    state = str(value["state"])
    retry_array_spec = value.get("retry_array_spec")
    retry_array_name = value.get("retry_array_job_name")
    retry_array_id = value.get("retry_array_job_id")
    retry_aggregate_name = value.get("retry_aggregate_job_name")
    retry_aggregate_id = value.get("retry_aggregate_job_id")
    _require(
        isinstance(retry_aggregate_name, str) and bool(retry_aggregate_name),
        "recovery progress aggregate name is missing",
    )
    if retry_array_spec is None:
        _require(
            retry_array_name is None and retry_array_id is None,
            "aggregate-only recovery unexpectedly records a retry array",
        )
        _require(
            state
            in {
                "planned",
                "retry_aggregate_submission_uncertain",
                "retry_aggregate_submitted",
            },
            "aggregate-only recovery has an array progress state",
        )
    else:
        _require(
            isinstance(retry_array_spec, str)
            and bool(retry_array_spec)
            and isinstance(retry_array_name, str)
            and bool(retry_array_name),
            "retry-array recovery identity is incomplete",
        )
        if state in {"planned", "retry_array_submission_uncertain"}:
            _require(
                retry_array_id is None,
                "pre-retry-array progress unexpectedly has an array job ID",
            )
        else:
            _require(
                isinstance(retry_array_id, str) and retry_array_id.isdigit(),
                "post-retry-array progress lacks a valid array job ID",
            )
    if state == "retry_aggregate_submitted":
        _require(
            isinstance(retry_aggregate_id, str)
            and retry_aggregate_id.isdigit(),
            "submitted retry aggregate lacks a valid job ID",
        )
    else:
        _require(
            retry_aggregate_id is None,
            "pre-retry-aggregate progress unexpectedly has an aggregate job ID",
        )
    return dict(value)


def _write_recovery_receipt(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        _require(
            path.is_file() and _read_json(path) == value,
            "refusing to overwrite a different dispatch recovery receipt",
        )
    else:
        _atomic_json(path, value)


def _verified_recovery_receipt(
    path: Path, *, launch: Mapping[str, Any]
) -> dict[str, Any]:
    existing = dict(_read_json(path))
    claimed = existing.get("recovery_sha256")
    core = {key: item for key, item in existing.items() if key != "recovery_sha256"}
    _require(
        existing.get("schema_version") == RECOVERY_SCHEMA_VERSION
        and isinstance(claimed, str)
        and claimed == digest_object(core),
        "existing dispatch recovery receipt is invalid",
    )
    launch_value = existing.get("launch")
    _require(isinstance(launch_value, Mapping), "recovery receipt lacks launch tuple")
    _verify_launch_tuple(launch_value, launch)
    _require(
        isinstance(existing.get("generation"), int)
        and int(existing["generation"]) >= 1,
        "recovery receipt generation is invalid",
    )
    return existing


def _recover_terminal_dispatch(
    *,
    output_dir: Path,
    receipt: Mapping[str, Any],
    launch: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
    exported: str,
    full_search_null_script: Path,
    full_search_null_aggregate_script: Path,
    submitter: Submitter,
    job_resolver: JobResolver,
    job_status_resolver: JobStatusResolver,
    policy: EpisodePolicy,
    checkpoint_verifier: CheckpointVerifier | None,
) -> dict[str, Any]:
    recovery_receipt_path = output_dir / "dispatch_recovery.json"
    jobs = receipt.get("jobs_submitted")
    _require(isinstance(jobs, list) and len(jobs) == 2,
             "dispatch receipt lacks the recorded Slurm jobs")
    array_job = jobs[0]
    aggregate_job = jobs[1]
    _require(
        isinstance(array_job, Mapping)
        and isinstance(aggregate_job, Mapping)
        and array_job.get("role") == "full_search_null_replicates"
        and aggregate_job.get("role") == "full_search_null_aggregate",
        "dispatch receipt job roles changed",
    )
    array_job_id = str(array_job.get("job_id"))
    array_job_name = str(array_job.get("job_name"))
    aggregate_job_id = str(aggregate_job.get("job_id"))
    aggregate_job_name = str(aggregate_job.get("job_name"))
    replicate_count = int(launch["replicate_count"])
    generation = 1
    previous_recovery_sha256: str | None = None
    if recovery_receipt_path.exists():
        existing = _verified_recovery_receipt(
            recovery_receipt_path, launch=launch
        )
        previous_recovery_sha256 = str(existing["recovery_sha256"])
        generation = int(existing["generation"]) + 1
        prior_array_id = existing.get("retry_array_job_id")
        prior_array_name = existing.get("retry_array_job_name")
        if prior_array_id is None:
            _require(prior_array_name is None,
                     "aggregate-only recovery has an array name")
            source_array_status = None
        else:
            _require(
                isinstance(prior_array_id, str) and prior_array_id.isdigit()
                and isinstance(prior_array_name, str),
                "prior recovery array identity is invalid",
            )
            source_array_status = _verify_scheduler_job_status(
                job_status_resolver(prior_array_id, prior_array_name),
                job_id=prior_array_id,
                job_name=prior_array_name,
            )
        prior_aggregate_id = str(existing.get("retry_aggregate_job_id"))
        prior_aggregate_name = str(existing.get("retry_aggregate_job_name"))
        source_aggregate_status = _verify_scheduler_job_status(
            job_status_resolver(prior_aggregate_id, prior_aggregate_name),
            job_id=prior_aggregate_id,
            job_name=prior_aggregate_name,
        )
        if (
            source_aggregate_status["terminal"] is not True
            or (
                source_array_status is not None
                and source_array_status["terminal"] is not True
            )
        ):
            return existing
        if source_aggregate_status["successful"] is True:
            _require(
                source_array_status is None
                or source_array_status["successful"] is True,
                "aggregate succeeded after a failed retry array",
            )
            return existing
        _require(
            source_aggregate_status["successful"] is False
            and not source_aggregate_status["task_states"],
            "prior recovery aggregate status is inconsistent",
        )
        if source_array_status is not None:
            expected_prior_tasks = {
                str(index) for index in existing["retry_indices"]
            }
            _require(
                set(source_array_status["task_states"]) == expected_prior_tasks,
                "prior retry array exact task identity changed",
            )
    else:
        source_array_status = _verify_scheduler_job_status(
            job_status_resolver(array_job_id, array_job_name),
            job_id=array_job_id,
            job_name=array_job_name,
        )
        source_aggregate_status = _verify_scheduler_job_status(
            job_status_resolver(aggregate_job_id, aggregate_job_name),
            job_id=aggregate_job_id,
            job_name=aggregate_job_name,
        )
        expected_tasks = {str(index) for index in range(replicate_count)}
        _require(
            source_array_status["terminal"] is True
            and source_array_status["successful"] is False
            and set(source_array_status["task_states"]) == expected_tasks,
            "recorded array is not an exact terminal-failed 99-task job",
        )
        _require(
            source_aggregate_status["terminal"] is True
            and source_aggregate_status["successful"] is False
            and not source_aggregate_status["task_states"],
            "recorded aggregate is not terminal failed/cancelled",
        )

    recovery_progress_path = output_dir / (
        f"dispatch_recovery_progress_{generation:03d}.json"
    )
    if recovery_progress_path.exists():
        progress = _verify_recovery_progress(
            _read_json(recovery_progress_path), launch=launch
        )
        _require(int(progress["generation"]) == generation,
                 "recovery progress generation changed")
    else:
        verifier = checkpoint_verifier or _production_checkpoint_verifier(
            launch=launch, policy=policy
        )
        checkpoint_inventory = _checkpoint_inventory(
            work_root=Path(str(launch["full_null_work_root"])),
            replicate_count=replicate_count,
            runtime_procedure_lock_sha256=str(
                runtime_manifest["procedure_lock_sha256"]
            ),
            checkpoint_verifier=verifier,
        )
        complete = set(map(int, checkpoint_inventory["completed_indices"]))
        retry_indices = [
            index for index in range(replicate_count) if index not in complete
        ]
        retry_array_spec = (
            _compressed_array_spec(retry_indices) if retry_indices else None
        )
        token = digest_object(launch)[:16]
        retry_array_job_name = (
            f"ep12-null-r-{token}-retry{generation}" if retry_indices else None
        )
        retry_aggregate_job_name = (
            f"ep12-null-a-{token}-retry{generation}"
        )
        _recovery_progress(
            recovery_progress_path,
            launch=launch,
            source_array_status=source_array_status,
            source_aggregate_status=source_aggregate_status,
            checkpoint_inventory=checkpoint_inventory,
            generation=generation,
            retry_indices=retry_indices,
            retry_array_spec=retry_array_spec,
            retry_array_job_name=retry_array_job_name,
            retry_aggregate_job_name=retry_aggregate_job_name,
            state="planned",
            retry_array_job_id=None,
            retry_aggregate_job_id=None,
        )
        progress = dict(_read_json(recovery_progress_path))

    state = str(progress["state"])
    retry_array_job_id = progress.get("retry_array_job_id")
    retry_aggregate_job_id = progress.get("retry_aggregate_job_id")
    retry_array_spec = progress.get("retry_array_spec")
    retry_array_job_name = progress.get("retry_array_job_name")
    retry_aggregate_job_name = str(progress["retry_aggregate_job_name"])
    parent_permit = launch.get("launch_permit")
    _require(isinstance(parent_permit, Mapping), "recovery lacks primary launch permit")
    retry_permit = derive_full_null_launch_permit(
        parent_identity=parent_permit,
        permit_path=output_dir / f"full_null_launch_permit_recovery_{generation:03d}.json",
        array_job_name=(
            str(retry_array_job_name) if retry_array_job_name is not None else None
        ),
        aggregate_job_name=retry_aggregate_job_name,
        array_indices=list(map(int, progress["retry_indices"])),
        generation=generation,
    )
    retry_exported = _replace_launch_permit_export(exported, retry_permit)
    if (
        state in {"planned", "retry_array_submission_uncertain"}
        and retry_array_spec is not None
    ):
        _require(isinstance(retry_array_job_name, str), "retry array name is missing")
        retry_array_job_id = _recover_named_submission(
            job_name=retry_array_job_name,
            job_resolver=job_resolver,
            job_status_resolver=job_status_resolver,
        )
        if retry_array_job_id is None:
            if state == "retry_array_submission_uncertain":
                raise ProcedureDispatchError(
                    "retry-array submission remains uncertain: the unique "
                    "deterministic job name is not yet visible; refusing "
                    "duplicate sbatch"
                )
            _recovery_progress(
                recovery_progress_path,
                launch=launch,
                source_array_status=progress["source_array_status"],
                source_aggregate_status=progress["source_aggregate_status"],
                checkpoint_inventory=progress["checkpoint_inventory"],
                generation=generation,
                retry_indices=progress["retry_indices"],
                retry_array_spec=str(retry_array_spec),
                retry_array_job_name=retry_array_job_name,
                retry_aggregate_job_name=retry_aggregate_job_name,
                state="retry_array_submission_uncertain",
                retry_array_job_id=None,
                retry_aggregate_job_id=None,
            )
            retry_array_job_id = _job_id(submitter([
                "sbatch", "--parsable", f"--job-name={retry_array_job_name}",
                "--partition=normal", "--cpus-per-task=32", "--mem=192G",
                "--time=2-00:00:00", "--requeue", "--open-mode=append",
                "--signal=B:SIGUSR1@46800",
                f"--array={retry_array_spec}", f"--export={retry_exported}",
                str(full_search_null_script.resolve()), "replicate",
            ]))
        _recovery_progress(
            recovery_progress_path,
            launch=launch,
            source_array_status=progress["source_array_status"],
            source_aggregate_status=progress["source_aggregate_status"],
            checkpoint_inventory=progress["checkpoint_inventory"],
            generation=generation,
            retry_indices=progress["retry_indices"],
            retry_array_spec=str(retry_array_spec),
            retry_array_job_name=retry_array_job_name,
            retry_aggregate_job_name=retry_aggregate_job_name,
            state="retry_array_submitted",
            retry_array_job_id=str(retry_array_job_id),
            retry_aggregate_job_id=None,
        )
        progress = dict(_read_json(recovery_progress_path))
        state = "retry_array_submitted"
    if state in {
        "planned",
        "retry_array_submitted",
        "retry_aggregate_submission_uncertain",
    }:
        retry_aggregate_job_id = _recover_named_submission(
            job_name=retry_aggregate_job_name,
            job_resolver=job_resolver,
            job_status_resolver=job_status_resolver,
        )
        if retry_aggregate_job_id is None:
            if state == "retry_aggregate_submission_uncertain":
                raise ProcedureDispatchError(
                    "retry-aggregate submission remains uncertain: the unique "
                    "deterministic job name is not yet visible; refusing "
                    "duplicate sbatch"
                )
            dependency_job_id: str | None = None
            if retry_array_spec is not None:
                _require(
                    isinstance(retry_array_job_id, str)
                    and retry_array_job_id.isdigit(),
                    "retry aggregate lacks its retry-array dependency",
                )
                dependency_job_id = retry_array_job_id
            _recovery_progress(
                recovery_progress_path,
                launch=launch,
                source_array_status=progress["source_array_status"],
                source_aggregate_status=progress["source_aggregate_status"],
                checkpoint_inventory=progress["checkpoint_inventory"],
                generation=generation,
                retry_indices=progress["retry_indices"],
                retry_array_spec=(
                    str(retry_array_spec)
                    if retry_array_spec is not None else None
                ),
                retry_array_job_name=(
                    str(retry_array_job_name)
                    if retry_array_job_name is not None else None
                ),
                retry_aggregate_job_name=retry_aggregate_job_name,
                state="retry_aggregate_submission_uncertain",
                retry_array_job_id=(
                    str(retry_array_job_id)
                    if retry_array_job_id is not None else None
                ),
                retry_aggregate_job_id=None,
            )
            command = _aggregate_submit_command(
                job_name=retry_aggregate_job_name,
                exported=_aggregate_exported(exported=retry_exported, launch=launch),
                aggregate_script=full_search_null_aggregate_script,
                dependency_job_id=dependency_job_id,
            )
            retry_aggregate_job_id = _job_id(submitter(command))
        _recovery_progress(
            recovery_progress_path,
            launch=launch,
            source_array_status=progress["source_array_status"],
            source_aggregate_status=progress["source_aggregate_status"],
            checkpoint_inventory=progress["checkpoint_inventory"],
            generation=generation,
            retry_indices=progress["retry_indices"],
            retry_array_spec=(
                str(retry_array_spec) if retry_array_spec is not None else None
            ),
            retry_array_job_name=(
                str(retry_array_job_name) if retry_array_job_name is not None else None
            ),
            retry_aggregate_job_name=retry_aggregate_job_name,
            state="retry_aggregate_submitted",
            retry_array_job_id=(
                str(retry_array_job_id) if retry_array_job_id is not None else None
            ),
            retry_aggregate_job_id=str(retry_aggregate_job_id),
        )
        progress = dict(_read_json(recovery_progress_path))
    _require(
        progress.get("state") == "retry_aggregate_submitted",
        "dispatch recovery did not reach aggregate submission",
    )
    core = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "action": "full_search_null_terminal_failure_resubmitted",
        "launch": dict(launch),
        "launch_sha256": digest_object(launch),
        "source_dispatch_sha256": receipt["dispatch_sha256"],
        "generation": generation,
        "previous_recovery_sha256": previous_recovery_sha256,
        "source_array_status": progress["source_array_status"],
        "source_aggregate_status": progress["source_aggregate_status"],
        "checkpoint_inventory": progress["checkpoint_inventory"],
        "retry_indices": progress["retry_indices"],
        "retry_array_spec": progress["retry_array_spec"],
        "retry_array_job_id": progress["retry_array_job_id"],
        "retry_array_job_name": progress["retry_array_job_name"],
        "retry_aggregate_job_id": progress["retry_aggregate_job_id"],
        "retry_aggregate_job_name": progress["retry_aggregate_job_name"],
        "aggregate_main_wrapper_sha256_export": launch[
            "aggregate_main_wrapper_sha256_export"
        ],
        "launch_permit": retry_permit,
        "checkpoint_resume_required": True,
        "blind_duplicate_submission": False,
        "episode_terminal_emitted": False,
        "final_connectivity_accessed": False,
    }
    recovery = {**core, "recovery_sha256": digest_object(core)}
    generation_path = output_dir / f"dispatch_recovery_{generation:03d}.json"
    _write_recovery_receipt(generation_path, recovery)
    _atomic_json(recovery_receipt_path, recovery)
    return recovery


def dispatch_procedure_lock(
    *,
    lock: Mapping[str, Any],
    policy: EpisodePolicy,
    output_dir: Path,
    procedure_lock_path: Path,
    full_search_null_script: Path,
    full_search_null_aggregate_script: Path | None = None,
    runtime_provider: str | None = None,
    runtime_bundle: Path | None = None,
    runtime_bundle_sha256: str | None = None,
    full_null_work_root: Path | None = None,
    full_null_durable_root: Path | None = None,
    resource_authorization_path: Path | None = None,
    resource_authorization_file_sha256: str | None = None,
    resource_authorization_pending_path: Path | None = None,
    source_freeze_manifest: Path | None = None,
    source_freeze_manifest_sha256: str | None = None,
    submitter: Submitter = _default_submitter,
    job_resolver: JobResolver = _default_job_resolver,
    job_status_resolver: JobStatusResolver = _default_job_status_resolver,
    recover_terminal_failure: bool = False,
    checkpoint_verifier: CheckpointVerifier | None = None,
) -> dict[str, Any]:
    """Dispatch the full null when eligible, or freeze a non-episode status.

    The noneligible branch intentionally does *not* invent an EP12 policy
    terminal: all scientific terminal classes available here require a valid
    final evaluation.  It records ``episode_terminal_emitted=false`` instead.
    """

    try:
        verify_procedure_lock(lock, policy=policy)
    except ProcedureLockError as exc:
        raise ProcedureDispatchError(str(exc)) from exc
    _require(lock.get("procedure_locked") is True,
             "an unresolved/unlocked procedure cannot be dispatched")
    _require(lock.get("full_search_null_started") is False,
             "procedure lock claims the full-search null already started")
    try:
        output_dir, output_fd = open_secure_directory(
            output_dir, root=output_dir.parent, create=True
        )
        lease_fd = open_regular_lock_at(output_fd, ".dispatch.lock")
    except SecureTransactionError as exc:
        raise ProcedureDispatchError(str(exc)) from exc
    try:
        fcntl.flock(lease_fd, fcntl.LOCK_EX)
        try:
            receipt_path = output_dir / "dispatch.json"
            existing_receipt: dict[str, Any] | None = None
            if receipt_path.exists():
                existing = dict(_read_json(receipt_path))
                claimed = existing.get("dispatch_sha256")
                core = {
                    key: value
                    for key, value in existing.items()
                    if key != "dispatch_sha256"
                }
                _require(isinstance(claimed, str) and digest_object(core) == claimed,
                         "existing dispatch digest mismatch")
                _require(existing.get("procedure_lock_sha256")
                         == lock["procedure_lock_sha256"],
                         "existing dispatch belongs to a different procedure lock")
                if existing.get("action") != "full_search_null_submitted":
                    _require(
                        recover_terminal_failure is False,
                        "noneligible dispatch has no Slurm array to recover",
                    )
                    _write_final_bundle(output_dir, existing)
                    return existing
                existing_receipt = existing

            if lock.get("null_eligible") is not True:
                _require(existing_receipt is None,
                         "eligible dispatch receipt conflicts with current lock")
                core = {
                    "schema_version": SCHEMA_VERSION,
                    "procedure_lock_sha256": lock["procedure_lock_sha256"],
                    "action": "null_not_eligible",
                    "reason": "pre_null_mechanical_qualification_incomplete",
                    "scientific_support_state": lock["scientific_support_state"],
                    "unresolved_reasons": lock["unresolved_reasons"],
                    "jobs_submitted": [],
                    "pre_null_dispatch_complete": True,
                    "episode_terminal_emitted": False,
                    "episode_terminal_reason": (
                        "frozen policy scientific terminals require valid final evaluation"
                    ),
                    "full_search_null_started": False,
                    "full_search_null_completed": False,
                    "final_connectivity_accessed": False,
                    "final_connectivity_access_authorized": False,
                }
                receipt = dict(core)
                receipt["dispatch_sha256"] = digest_object(core)
                _write_final_bundle(output_dir, receipt)
                return receipt

            _require(
                runtime_provider == CANONICAL_RUNTIME_PROVIDER,
                "eligible lock requires the canonical full-null runtime provider",
            )
            _require(runtime_bundle is not None and runtime_bundle.is_dir(),
                     "eligible lock requires an existing runtime bundle")
            runtime_manifest_path = runtime_bundle / "runtime_bundle.json"
            _require(runtime_manifest_path.is_file(),
                     "eligible lock runtime bundle lacks its manifest")
            runtime_manifest = _read_json(runtime_manifest_path)
            _require(
                isinstance(runtime_bundle_sha256, str)
                and len(runtime_bundle_sha256) == 64
                and all(c in "0123456789abcdef" for c in runtime_bundle_sha256)
                and runtime_manifest.get("bundle_sha256")
                == runtime_bundle_sha256,
                "eligible lock runtime-bundle identity is missing or changed",
            )
            _require(
                runtime_manifest.get("upstream_procedure_lock_sha256")
                == lock["procedure_lock_sha256"],
                "runtime bundle is not bound to this procedure lock",
            )
            _require(full_null_work_root is not None and full_null_work_root.is_absolute(),
                     "eligible lock requires an absolute full-null work root")
            _require(
                full_null_durable_root is not None
                and full_null_durable_root.is_absolute(),
                "eligible lock requires an absolute full-null durable root",
            )
            _require(full_search_null_script.is_file(),
                     "full-search-null sbatch script is missing")
            _require(
                full_search_null_aggregate_script is not None
                and full_search_null_aggregate_script.is_file(),
                "full-search-null aggregate sbatch script is missing",
            )
            _require(procedure_lock_path.is_file(), "procedure-lock JSON path is missing")
            on_disk_lock = _read_json(procedure_lock_path)
            _require(on_disk_lock == lock, "procedure-lock path bytes do not match dispatch lock")
            procedure_lock_file_sha256 = hashlib.sha256(
                procedure_lock_path.read_bytes()
            ).hexdigest()
            _require(
                runtime_manifest.get("upstream_procedure_lock_file_sha256")
                == procedure_lock_file_sha256,
                "runtime bundle is not bound to the exact procedure-lock file",
            )
            _require(
                resource_authorization_path is not None
                and resource_authorization_pending_path is not None
                and isinstance(resource_authorization_file_sha256, str),
                "eligible lock requires an explicit exact resource authorization",
            )
            try:
                authorization_evidence = verify_resource_authorization(
                    authorization_path=resource_authorization_path,
                    authorization_file_sha256=resource_authorization_file_sha256,
                    pending_path=resource_authorization_pending_path,
                    runtime_bundle=runtime_bundle,
                    runtime_bundle_sha256=runtime_bundle_sha256,
                    procedure_lock_sha256=str(lock["procedure_lock_sha256"]),
                    procedure_lock_file_sha256=procedure_lock_file_sha256,
                    policy=policy,
                    durable_root=output_dir.parent,
                )
            except ResourceAuthorizationError as error:
                raise ProcedureDispatchError(str(error)) from error
            resource_release_path = (
                resource_authorization_pending_path.parent
                / "resource_authorization_release.json"
            )
            _require(
                source_freeze_manifest is not None
                and isinstance(source_freeze_manifest_sha256, str),
                "eligible lock requires an exact executable source freeze",
            )
            source_freeze = _source_freeze_identity(
                manifest_path=source_freeze_manifest,
                manifest_file_sha256=source_freeze_manifest_sha256,
            )
            array_spec = f"0-{policy.null_replicates - 1}"
            launch = _launch_tuple(
                runtime_provider=runtime_provider,
                full_search_null_script=full_search_null_script,
                full_search_null_aggregate_script=(
                    full_search_null_aggregate_script
                ),
                runtime_bundle=runtime_bundle,
                runtime_bundle_sha256=runtime_bundle_sha256,
                procedure_lock_path=procedure_lock_path,
                procedure_lock_sha256=str(lock["procedure_lock_sha256"]),
                procedure_lock_file_sha256=procedure_lock_file_sha256,
                source_freeze=source_freeze,
                full_null_work_root=full_null_work_root,
                full_null_durable_root=full_null_durable_root,
                array_spec=array_spec,
                replicate_count=policy.null_replicates,
            )
            dispatch_token = digest_object(launch)[:20]
            array_job_name = f"ep12-null-r-{dispatch_token}"
            aggregate_job_name = f"ep12-null-a-{dispatch_token}"
            launch_permit = create_full_null_launch_permit(
                permit_path=output_dir / "full_null_launch_permit.json",
                authorization_evidence=authorization_evidence,
                launch=launch,
                array_job_name=array_job_name,
                aggregate_job_name=aggregate_job_name,
                array_indices=list(range(policy.null_replicates)),
            )
            explicit_exports = {
                "EP12_FULL_NULL_RUNTIME_PROVIDER": runtime_provider,
                "EP12_FULL_NULL_BUNDLE": str(runtime_bundle.resolve()),
                "EP12_FULL_NULL_BUNDLE_SHA256": runtime_bundle_sha256,
                "EP12_FULL_NULL_WORK_ROOT": str(full_null_work_root.resolve()),
                "EP12_FULL_NULL_DURABLE_ROOT": str(full_null_durable_root.resolve()),
                "EP12_PRE_NULL_PROCEDURE_LOCK_PATH": str(
                    procedure_lock_path.resolve()
                ),
                "EP12_PRE_NULL_PROCEDURE_LOCK_SHA256": str(
                    lock["procedure_lock_sha256"]
                ),
                "EP12_PRE_NULL_PROCEDURE_LOCK_FILE_SHA256": (
                    procedure_lock_file_sha256
                ),
                "EP12_SOURCE_FREEZE_MANIFEST": source_freeze["manifest_path"],
                "EP12_SOURCE_FREEZE_MANIFEST_SHA256": source_freeze[
                    "manifest_file_sha256"
                ],
                "EP12_FULL_NULL_LAUNCH_PERMIT": launch_permit["permit_path"],
                "EP12_FULL_NULL_LAUNCH_PERMIT_SHA256": launch_permit[
                    "permit_file_sha256"
                ],
            }
            exported = _explicit_export(explicit_exports)
            launch = {
                **launch,
                "launch_permit": launch_permit,
                "explicit_exports": explicit_exports,
                "ambient_environment_inherited": False,
            }
            if existing_receipt is not None:
                existing_launch = existing_receipt.get("launch")
                _require(isinstance(existing_launch, Mapping),
                         "existing dispatch lacks its launch tuple")
                _verify_launch_tuple(existing_launch, launch)
                _require(
                    existing_receipt.get("launch_sha256") == digest_object(launch),
                    "existing dispatch launch digest changed",
                )
                _require(
                    existing_receipt.get("resource_authorization_sha256")
                    == authorization_evidence["authorization_sha256"]
                    and existing_receipt.get(
                        "resource_authorization_file_sha256"
                    ) == authorization_evidence["authorization_file_sha256"]
                    and existing_receipt.get("resource_plan_sha256")
                    == authorization_evidence["resource_plan_sha256"],
                    "existing dispatch belongs to another resource authorization",
                )
                _write_final_bundle(output_dir, existing_receipt)
                try:
                    write_resource_authorization_release(
                        release_path=resource_release_path,
                        authorization_evidence=authorization_evidence,
                        dispatch_receipt_path=output_dir / "dispatch.json",
                        dispatch_receipt=existing_receipt,
                        durable_root=output_dir.parent,
                    )
                except ResourceAuthorizationError as error:
                    raise ProcedureDispatchError(str(error)) from error
                if not recover_terminal_failure:
                    return existing_receipt
                return _recover_terminal_dispatch(
                    output_dir=output_dir,
                    receipt=existing_receipt,
                    launch=launch,
                    runtime_manifest=runtime_manifest,
                    exported=exported,
                    full_search_null_script=full_search_null_script,
                    full_search_null_aggregate_script=(
                        full_search_null_aggregate_script
                    ),
                    submitter=submitter,
                    job_resolver=job_resolver,
                    job_status_resolver=job_status_resolver,
                    policy=policy,
                    checkpoint_verifier=checkpoint_verifier,
                )
            progress_path = output_dir / "dispatch_progress.json"
            if progress_path.exists():
                state, array_job_id, aggregate_job_id = _verify_progress(
                    _read_json(progress_path),
                    launch=launch,
                    array_job_name=array_job_name,
                    aggregate_job_name=aggregate_job_name,
                )
            else:
                state, array_job_id, aggregate_job_id = "planned", None, None
                _progress(
                    progress_path,
                    launch=launch,
                    state=state,
                    array_job_name=array_job_name,
                    aggregate_job_name=aggregate_job_name,
                    array_job_id=None,
                    aggregate_job_id=None,
                )
            if state in {"planned", "array_submission_uncertain"}:
                array_job_id = _recover_named_submission(
                    job_name=array_job_name,
                    job_resolver=job_resolver,
                    job_status_resolver=job_status_resolver,
                )
                if array_job_id is None:
                    if state == "array_submission_uncertain":
                        raise ProcedureDispatchError(
                            "array submission remains uncertain: the unique "
                            "deterministic job name is not yet visible; refusing "
                            "duplicate sbatch"
                        )
                    _progress(
                        progress_path,
                        launch=launch,
                        state="array_submission_uncertain",
                        array_job_name=array_job_name,
                        aggregate_job_name=aggregate_job_name,
                        array_job_id=None,
                        aggregate_job_id=None,
                    )
                    array_output = submitter(
                        [
                            "sbatch",
                            "--parsable",
                            f"--job-name={array_job_name}",
                            "--partition=normal",
                            "--cpus-per-task=32",
                            "--mem=192G",
                            "--time=2-00:00:00",
                            "--requeue",
                            "--open-mode=append",
                            "--signal=B:SIGUSR1@46800",
                            f"--array={array_spec}",
                            f"--export={exported}",
                            str(full_search_null_script.resolve()),
                            "replicate",
                        ]
                    )
                    array_job_id = _job_id(array_output)
                _progress(
                    progress_path,
                    launch=launch,
                    state="array_submitted",
                    array_job_name=array_job_name,
                    aggregate_job_name=aggregate_job_name,
                    array_job_id=array_job_id,
                    aggregate_job_id=None,
                )
                state = "array_submitted"
            _require(array_job_id is not None, "array job ID recovery failed")
            if state in {
                "array_submitted",
                "aggregate_submission_uncertain",
            }:
                aggregate_job_id = _recover_named_submission(
                    job_name=aggregate_job_name,
                    job_resolver=job_resolver,
                    job_status_resolver=job_status_resolver,
                )
                if aggregate_job_id is None:
                    if state == "aggregate_submission_uncertain":
                        raise ProcedureDispatchError(
                            "aggregate submission remains uncertain: the unique "
                            "deterministic job name is not yet visible; refusing "
                            "duplicate sbatch"
                        )
                    _progress(
                        progress_path,
                        launch=launch,
                        state="aggregate_submission_uncertain",
                        array_job_name=array_job_name,
                        aggregate_job_name=aggregate_job_name,
                        array_job_id=array_job_id,
                        aggregate_job_id=None,
                    )
                    aggregate_output = submitter(
                        _aggregate_submit_command(
                            job_name=aggregate_job_name,
                            exported=_aggregate_exported(
                                exported=exported, launch=launch
                            ),
                            aggregate_script=full_search_null_aggregate_script,
                            dependency_job_id=array_job_id,
                        )
                    )
                    aggregate_job_id = _job_id(aggregate_output)
                _progress(
                    progress_path,
                    launch=launch,
                    state="aggregate_submitted",
                    array_job_name=array_job_name,
                    aggregate_job_name=aggregate_job_name,
                    array_job_id=array_job_id,
                    aggregate_job_id=aggregate_job_id,
                )
            _require(aggregate_job_id is not None, "aggregate job ID recovery failed")
            core = {
                "schema_version": SCHEMA_VERSION,
                "procedure_lock_sha256": lock["procedure_lock_sha256"],
                "procedure_lock_file_sha256": procedure_lock_file_sha256,
                "launch": launch,
                "launch_sha256": digest_object(launch),
                "action": "full_search_null_submitted",
                "reason": "authenticated_pre_null_support_and_controls_passed",
                "jobs_submitted": [
                    {
                        "role": "full_search_null_replicates",
                        "job_id": array_job_id,
                        "job_name": array_job_name,
                        "array": array_spec,
                        "replicate_count": policy.null_replicates,
                    },
                    {
                        "role": "full_search_null_aggregate",
                        "job_id": aggregate_job_id,
                        "job_name": aggregate_job_name,
                        "dependency": f"afterok:{array_job_id}",
                        "cpus_per_task": AGGREGATE_CPUS_PER_TASK,
                        "memory": AGGREGATE_MEMORY,
                        "time_limit": AGGREGATE_TIME_LIMIT,
                        "array": False,
                        "requeue": False,
                        "aggregate_main_wrapper_sha256_export": launch[
                            "aggregate_main_wrapper_sha256_export"
                        ],
                    },
                ],
                "runtime_provider": runtime_provider,
                "runtime_bundle": str(runtime_bundle.resolve()),
                "runtime_bundle_sha256": runtime_bundle_sha256,
                "source_freeze": source_freeze,
                "explicit_exports": explicit_exports,
                "ambient_environment_inherited": False,
                "resource_plan_sha256": authorization_evidence[
                    "resource_plan_sha256"
                ],
                "resource_authorization_path": authorization_evidence[
                    "authorization_path"
                ],
                "resource_authorization_sha256": authorization_evidence[
                    "authorization_sha256"
                ],
                "resource_authorization_file_sha256": authorization_evidence[
                    "authorization_file_sha256"
                ],
                "resource_authorization_pending_path": authorization_evidence[
                    "pending_path"
                ],
                "resource_authorization_pending_sha256": authorization_evidence[
                    "pending_receipt_sha256"
                ],
                "full_null_work_root": str(full_null_work_root.resolve()),
                "full_null_durable_root": str(full_null_durable_root.resolve()),
                "full_search_null_script": str(full_search_null_script.resolve()),
                "full_search_null_script_sha256": hashlib.sha256(
                    full_search_null_script.read_bytes()
                ).hexdigest(),
                "full_search_null_aggregate_script": str(
                    full_search_null_aggregate_script.resolve()
                ),
                "full_search_null_aggregate_script_sha256": hashlib.sha256(
                    full_search_null_aggregate_script.read_bytes()
                ).hexdigest(),
                "pre_null_dispatch_complete": True,
                "episode_terminal_emitted": False,
                "full_search_null_started": True,
                "full_search_null_completed": False,
                "final_connectivity_accessed": False,
                "final_connectivity_access_authorized": False,
            }
            receipt = dict(core)
            receipt["dispatch_sha256"] = digest_object(core)
            _write_final_bundle(output_dir, receipt)
            try:
                write_resource_authorization_release(
                    release_path=resource_release_path,
                    authorization_evidence=authorization_evidence,
                    dispatch_receipt_path=output_dir / "dispatch.json",
                    dispatch_receipt=receipt,
                    durable_root=output_dir.parent,
                )
            except ResourceAuthorizationError as error:
                raise ProcedureDispatchError(str(error)) from error
            return receipt
        finally:
            fcntl.flock(lease_fd, fcntl.LOCK_UN)
    finally:
        os.close(lease_fd)
        os.close(output_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--procedure-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-search-null-script", type=Path, required=True)
    parser.add_argument(
        "--full-search-null-aggregate-script", type=Path, required=True
    )
    parser.add_argument("--runtime-provider")
    parser.add_argument("--runtime-bundle", type=Path)
    parser.add_argument("--runtime-bundle-sha256")
    parser.add_argument("--full-null-work-root", type=Path)
    parser.add_argument("--full-null-durable-root", type=Path)
    parser.add_argument("--resource-authorization", type=Path)
    parser.add_argument("--resource-authorization-file-sha256")
    parser.add_argument("--resource-authorization-pending", type=Path)
    parser.add_argument("--source-freeze-manifest", type=Path, required=True)
    parser.add_argument("--source-freeze-manifest-sha256", required=True)
    parser.add_argument(
        "--recover-terminal-failure",
        action="store_true",
        help="authenticate a recorded terminal-failed array and resume missing indices",
    )
    args = parser.parse_args(argv)
    policy = EpisodePolicy.load(args.policy)
    receipt = dispatch_procedure_lock(
        lock=_read_json(args.procedure_lock),
        policy=policy,
        output_dir=args.output_dir,
        procedure_lock_path=args.procedure_lock,
        full_search_null_script=args.full_search_null_script,
        full_search_null_aggregate_script=(
            args.full_search_null_aggregate_script
        ),
        runtime_provider=args.runtime_provider,
        runtime_bundle=args.runtime_bundle,
        runtime_bundle_sha256=args.runtime_bundle_sha256,
        full_null_work_root=args.full_null_work_root,
        full_null_durable_root=args.full_null_durable_root,
        resource_authorization_path=args.resource_authorization,
        resource_authorization_file_sha256=(
            args.resource_authorization_file_sha256
        ),
        resource_authorization_pending_path=(
            args.resource_authorization_pending
        ),
        source_freeze_manifest=args.source_freeze_manifest,
        source_freeze_manifest_sha256=args.source_freeze_manifest_sha256,
        recover_terminal_failure=args.recover_terminal_failure,
    )
    print(canonical_json(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_CPUS_PER_TASK",
    "AGGREGATE_MEMORY",
    "AGGREGATE_TIME_LIMIT",
    "LAUNCH_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "PROGRESS_SCHEMA_VERSION",
    "ProcedureDispatchError",
    "RECOVERY_SCHEMA_VERSION",
    "SCHEDULER_STATUS_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "dispatch_procedure_lock",
    "freeze_scheduler_job_status",
]
