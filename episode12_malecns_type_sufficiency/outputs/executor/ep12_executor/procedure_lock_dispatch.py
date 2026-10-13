"""Dispatch the frozen EP12 null program from one semantic procedure lock."""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .policy import EpisodePolicy, canonical_json, digest_object
from .procedure_lock import ProcedureLockError, verify_procedure_lock
from .resource_authorization import (
    ResourceAuthorizationError,
    verify_resource_authorization,
)
from .runtime import EPISODE_ROOT, atomic_json
from .slurm_client import sanitized_slurm_client_environment


OUTPUTS_ROOT = EPISODE_ROOT / "outputs"
AGGREGATE_CPUS_PER_TASK = 1
AGGREGATE_MEMORY = "32G"
AGGREGATE_TIME_LIMIT = "12:00:00"
DISPATCH_STATE_SCHEMA = "ep12.procedure_lock_dispatch_state.v1"
_JOB_ID = re.compile(r"^(?P<job_id>[0-9]+)(?:;[A-Za-z0-9_.-]+)?$")


class ProcedureDispatchError(RuntimeError):
    """The procedure lock is not ready for one bounded dispatch."""


class SubmissionRejectedError(ProcedureDispatchError):
    """Slurm explicitly rejected a submission before returning a job ID."""


Submitter = Callable[[Sequence[str]], str]
SchedulerLookup = Callable[[str, str, str], str | None]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProcedureDispatchError(message)


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    boundary = root.resolve()
    if resolved != boundary and boundary not in resolved.parents:
        raise ProcedureDispatchError(f"{label} must stay under {boundary}: {resolved}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProcedureDispatchError(f"cannot read JSON artifact {path}: {error}") from error
    _require(isinstance(value, Mapping), f"JSON artifact is not an object: {path}")
    return dict(value)


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
    except OSError as error:
        stderr = getattr(error, "stderr", "")
        raise SubmissionRejectedError(f"sbatch submission failed: {stderr}") from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr or ""
        if error.returncode < 0:
            raise ProcedureDispatchError(
                "sbatch was interrupted after submission began; scheduler outcome is unresolved"
            ) from error
        raise SubmissionRejectedError(f"sbatch submission failed: {stderr}") from error
    return completed.stdout.strip()


def _run_scheduler_query(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=sanitized_slurm_client_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "")
        raise ProcedureDispatchError(
            f"cannot reconcile an uncertain Slurm submission: {stderr}"
        ) from error
    return completed.stdout


def _base_job_id(value: str) -> str | None:
    match = re.match(r"^(?P<job_id>[0-9]+)(?:[_.+].*)?$", value.strip())
    return None if match is None else match.group("job_id")


def _default_scheduler_lookup(job_name: str, comment: str, start_date: str) -> str | None:
    """Find one previously accepted submission by its lock-bound identity.

    This path is used only after the durable ledger says submission began but
    lacks the returned job ID.  Any ambiguous scheduler result fails closed.
    """

    identifiers: set[str] = set()
    active = _run_scheduler_query(
        [
            "squeue",
            "--noheader",
            f"--name={job_name}",
            # %F is the array master ID (and the ordinary job ID for a
            # non-array job).  %A would identify individual active elements.
            "--format=%F|%100j|%120k",
        ]
    )
    for line in active.splitlines():
        fields = [field.strip() for field in line.split("|", 2)]
        if len(fields) != 3 or fields[1] != job_name or fields[2] != comment:
            continue
        job_id = _base_job_id(fields[0])
        _require(job_id is not None, "scheduler returned an invalid active job ID")
        identifiers.add(job_id)

    accounted = _run_scheduler_query(
        [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            f"--name={job_name}",
            f"--starttime={start_date}",
            "--format=JobIDRaw,JobName,Comment",
        ]
    )
    for line in accounted.splitlines():
        fields = [field.strip() for field in line.split("|", 2)]
        if len(fields) != 3 or fields[1] != job_name or fields[2] != comment:
            continue
        job_id = _base_job_id(fields[0])
        _require(job_id is not None, "scheduler returned an invalid accounted job ID")
        identifiers.add(job_id)

    _require(
        len(identifiers) <= 1,
        "multiple Slurm jobs match one dispatch submission identity",
    )
    return next(iter(identifiers), None)


def _job_id(output: str) -> str:
    match = _JOB_ID.fullmatch(output.strip())
    if match is None:
        raise ProcedureDispatchError(f"unexpected sbatch --parsable output: {output!r}")
    return match.group("job_id")


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        _require(_read_json(path) == dict(value), f"refusing to overwrite {path}")
        return
    atomic_json(path, dict(value))


def _write_state(path: Path, value: Mapping[str, Any]) -> None:
    """Advance the lock-bound dispatch ledger atomically."""

    atomic_json(path, dict(value))


def _validated_job_id(value: Any, *, label: str) -> str:
    _require(
        isinstance(value, str) and _JOB_ID.fullmatch(value) is not None,
        f"stored {label} job ID is invalid",
    )
    return value


def _submission_identity(
    dispatch_identity: Mapping[str, Any], *, role: str
) -> tuple[str, str]:
    token = digest_object(
        {"dispatch_identity": dict(dispatch_identity), "submission_role": role}
    )
    role_code = "a" if role == "array" else "g"
    return f"ep12-null-{role_code}-{token}", f"ep12-dispatch:{role}:{token}"


def _validate_state(
    state: Mapping[str, Any], *, dispatch_identity: Mapping[str, Any]
) -> str:
    _require(
        state.get("schema_version") == DISPATCH_STATE_SCHEMA
        and state.get("dispatch_identity") == dict(dispatch_identity),
        "existing dispatch state belongs to another lock, runtime, or policy",
    )
    phase = state.get("phase")
    _require(
        phase
        in {
            "authorized",
            "array_submitting",
            "array_submitted",
            "aggregate_submitting",
            "aggregate_submitted",
            "launch_published",
        },
        "dispatch state has an invalid phase",
    )
    created_on = state.get("created_on")
    _require(
        isinstance(created_on, str)
        and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", created_on) is not None,
        "dispatch state lacks a valid creation date",
    )
    array_job_id = state.get("array_job_id")
    aggregate_job_id = state.get("aggregate_job_id")
    if phase in {"authorized", "array_submitting"}:
        _require(
            array_job_id is None and aggregate_job_id is None,
            "pre-array dispatch state contains a job ID",
        )
    elif phase in {"array_submitted", "aggregate_submitting"}:
        _validated_job_id(array_job_id, label="array")
        _require(
            aggregate_job_id is None,
            "pre-aggregate dispatch state contains an aggregate job ID",
        )
    else:
        _validated_job_id(array_job_id, label="array")
        _validated_job_id(aggregate_job_id, label="aggregate")
    _require(
        state.get("final_connectivity_accessed") is False
        and state.get("final_connectivity_access_authorized") is False,
        "dispatch state violates the final-connectivity boundary",
    )
    return str(phase)


def _launch_job_ids(launch: Mapping[str, Any]) -> tuple[str, str]:
    jobs = launch.get("jobs_submitted")
    _require(
        isinstance(jobs, list) and len(jobs) == 2,
        "existing launch has an invalid submitted-job inventory",
    )
    first = jobs[0] if isinstance(jobs[0], Mapping) else {}
    second = jobs[1] if isinstance(jobs[1], Mapping) else {}
    _require(
        first.get("role") == "full_search_null_replicates"
        and second.get("role") == "full_search_null_aggregate",
        "existing launch has an invalid submitted-job order",
    )
    return (
        _validated_job_id(first.get("job_id"), label="launch array"),
        _validated_job_id(second.get("job_id"), label="launch aggregate"),
    )


def dispatch_procedure_lock(
    *,
    lock: Mapping[str, Any],
    policy: EpisodePolicy,
    output_dir: Path,
    procedure_lock_path: Path,
    full_search_null_script: Path,
    full_search_null_aggregate_script: Path,
    full_null_work_root: Path | None = None,
    full_null_durable_root: Path | None = None,
    resource_authorization_path: Path | None = None,
    runtime_bundle: Path | None = None,
    submitter: Submitter = _default_submitter,
    scheduler_lookup: SchedulerLookup = _default_scheduler_lookup,
    **_: Any,
) -> dict[str, Any]:
    """Submit the 99 null searches once, or record an ineligible decision."""

    try:
        verify_procedure_lock(lock, policy=policy)
    except ProcedureLockError as error:
        raise ProcedureDispatchError(str(error)) from error
    output = _inside(output_dir, OUTPUTS_ROOT, label="dispatch output directory")
    output.mkdir(parents=True, exist_ok=True)
    launch_path = output / "launch.json"
    lease_path = output / ".dispatch.lock"
    with lease_path.open("a+b") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        try:
            if lock.get("procedure_locked") is not True or lock.get("null_eligible") is not True:
                ineligible_identity = {
                    "policy_id": policy.policy_id,
                    "procedure_lock_sha256": lock.get("procedure_lock_sha256"),
                    "required_replicates": policy.null_replicates,
                    "null_eligible": False,
                }
                if launch_path.exists():
                    existing = _read_json(launch_path)
                    _require(
                        existing.get("action") == "null_not_eligible"
                        and existing.get("dispatch_identity") == ineligible_identity,
                        "existing dispatch decision belongs to another lock or policy",
                    )
                    return existing
                decision = {
                    "action": "null_not_eligible",
                    "reason": "scientific_procedure_lock_not_eligible",
                    "dispatch_identity": ineligible_identity,
                    "jobs_submitted": [],
                    "procedure_locked": lock.get("procedure_locked") is True,
                    "full_search_null_started": False,
                    "full_search_null_completed": False,
                    "final_connectivity_accessed": False,
                    "final_connectivity_access_authorized": False,
                }
                _write_once(launch_path, decision)
                return decision
            _require(
                resource_authorization_path is not None,
                "eligible full-search null requires explicit resource authorization",
            )
            _require(
                runtime_bundle is not None,
                "eligible full-search null requires its scientific runtime bundle",
            )
            try:
                authorization = verify_resource_authorization(
                    authorization_path=resource_authorization_path,
                    policy=policy,
                    procedure_lock=lock,
                    runtime_bundle=runtime_bundle,
                )
            except ResourceAuthorizationError as error:
                raise ProcedureDispatchError(str(error)) from error

            procedure_lock_path = procedure_lock_path.resolve(strict=True)
            _require(
                _read_json(procedure_lock_path) == dict(lock),
                "procedure-lock path differs from the verified lock",
            )
            null_script = full_search_null_script.resolve(strict=True)
            aggregate_script = full_search_null_aggregate_script.resolve(strict=True)
            runtime_root = runtime_bundle.resolve(strict=True)
            runtime_manifest = _read_json(runtime_root / "runtime_bundle.json")
            runtime_bundle_sha256 = runtime_manifest.get("bundle_sha256")
            _require(
                isinstance(runtime_bundle_sha256, str) and runtime_bundle_sha256,
                "runtime bundle identity is missing",
            )
            work_root = (full_null_work_root or Path("/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/full_search_null")).resolve()
            durable_root = _inside(
                full_null_durable_root
                or OUTPUTS_ROOT / "development_run002" / "full_search_null",
                OUTPUTS_ROOT,
                label="full-null durable root",
            )
            dispatch_identity = {
                "policy_id": policy.policy_id,
                "policy_sha256": policy.policy_hash,
                "procedure_lock_sha256": lock["procedure_lock_sha256"],
                "runtime_bundle_path": str(runtime_root),
                "runtime_bundle_sha256": runtime_bundle_sha256,
                "full_search_null_script": str(null_script),
                "full_search_null_aggregate_script": str(aggregate_script),
                "full_null_work_root": str(work_root),
                "full_null_durable_root": str(durable_root),
                "resource_authorization_path": str(
                    resource_authorization_path.resolve(strict=True)
                ),
                "required_replicates": policy.null_replicates,
                "null_eligible": True,
            }
            state_path = output / "dispatch_state.json"
            if state_path.exists():
                state = _read_json(state_path)
            else:
                state = {
                    "schema_version": DISPATCH_STATE_SCHEMA,
                    "dispatch_identity": dispatch_identity,
                    "created_on": datetime.now().astimezone().date().isoformat(),
                    "phase": "authorized",
                    "array_job_id": None,
                    "aggregate_job_id": None,
                    "final_connectivity_accessed": False,
                    "final_connectivity_access_authorized": False,
                }
                _write_state(state_path, state)
            phase = _validate_state(state, dispatch_identity=dispatch_identity)

            if launch_path.exists():
                existing = _read_json(launch_path)
                _require(
                    existing.get("action") == "full_search_null_submitted"
                    and existing.get("dispatch_identity") == dispatch_identity,
                    "existing launch belongs to another lock, runtime, or policy",
                )
                launch_array_id, launch_aggregate_id = _launch_job_ids(existing)
                _require(
                    phase in {"aggregate_submitted", "launch_published"}
                    and state.get("array_job_id") == launch_array_id
                    and state.get("aggregate_job_id") == launch_aggregate_id,
                    "existing launch disagrees with the dispatch ledger",
                )
                if phase == "aggregate_submitted":
                    state = {**state, "phase": "launch_published"}
                    _write_state(state_path, state)
                return existing

            array_job_name, array_comment = _submission_identity(
                dispatch_identity, role="array"
            )
            aggregate_job_name, aggregate_comment = _submission_identity(
                dispatch_identity, role="aggregate"
            )
            array_spec = f"0-{policy.null_replicates - 1}"
            array_command = [
                "sbatch",
                "--parsable",
                f"--job-name={array_job_name}",
                f"--comment={array_comment}",
                f"--array={array_spec}",
                str(null_script),
                str(policy.path),
                str(procedure_lock_path),
                str(runtime_root),
                str(work_root),
                str(durable_root),
            ]
            if phase == "array_submitting":
                recovered = scheduler_lookup(
                    array_job_name, array_comment, str(state["created_on"])
                )
                _require(
                    recovered is not None,
                    "array submission outcome is unresolved; refusing a duplicate submission",
                )
                array_job_id = _validated_job_id(recovered, label="recovered array")
                state = {
                    **state,
                    "phase": "array_submitted",
                    "array_job_id": array_job_id,
                }
                _write_state(state_path, state)
                phase = "array_submitted"
            elif phase == "authorized":
                state = {**state, "phase": "array_submitting"}
                _write_state(state_path, state)
                try:
                    array_job_id = _job_id(submitter(array_command))
                except SubmissionRejectedError:
                    state = {**state, "phase": "authorized"}
                    _write_state(state_path, state)
                    raise
                state = {
                    **state,
                    "phase": "array_submitted",
                    "array_job_id": array_job_id,
                }
                _write_state(state_path, state)
                phase = "array_submitted"
            else:
                array_job_id = _validated_job_id(
                    state.get("array_job_id"), label="array"
                )
            aggregate_command = [
                "sbatch",
                "--parsable",
                f"--job-name={aggregate_job_name}",
                f"--comment={aggregate_comment}",
                "--kill-on-invalid-dep=yes",
                f"--dependency=afterok:{array_job_id}",
                str(aggregate_script),
                str(policy.path),
                str(procedure_lock_path),
                str(runtime_root),
                str(work_root),
                str(durable_root),
            ]
            if phase == "aggregate_submitting":
                recovered = scheduler_lookup(
                    aggregate_job_name, aggregate_comment, str(state["created_on"])
                )
                _require(
                    recovered is not None,
                    "aggregate submission outcome is unresolved; refusing a duplicate submission",
                )
                aggregate_job_id = _validated_job_id(
                    recovered, label="recovered aggregate"
                )
                state = {
                    **state,
                    "phase": "aggregate_submitted",
                    "aggregate_job_id": aggregate_job_id,
                }
                _write_state(state_path, state)
                phase = "aggregate_submitted"
            elif phase == "array_submitted":
                state = {**state, "phase": "aggregate_submitting"}
                _write_state(state_path, state)
                try:
                    aggregate_job_id = _job_id(submitter(aggregate_command))
                except SubmissionRejectedError:
                    state = {**state, "phase": "array_submitted"}
                    _write_state(state_path, state)
                    raise
                state = {
                    **state,
                    "phase": "aggregate_submitted",
                    "aggregate_job_id": aggregate_job_id,
                }
                _write_state(state_path, state)
                phase = "aggregate_submitted"
            else:
                aggregate_job_id = _validated_job_id(
                    state.get("aggregate_job_id"), label="aggregate"
                )
            launch = {
                "action": "full_search_null_submitted",
                "dispatch_identity": dispatch_identity,
                "policy_id": policy.policy_id,
                "required_replicates": policy.null_replicates,
                "jobs_submitted": [
                    {
                        "role": "full_search_null_replicates",
                        "job_id": array_job_id,
                        "array": array_spec,
                    },
                    {
                        "role": "full_search_null_aggregate",
                        "job_id": aggregate_job_id,
                        "dependency": f"afterok:{array_job_id}",
                        "cpus_per_task": AGGREGATE_CPUS_PER_TASK,
                        "memory": AGGREGATE_MEMORY,
                        "time_limit": AGGREGATE_TIME_LIMIT,
                    },
                ],
                "resource_authorization": authorization,
                "procedure_lock_path": str(procedure_lock_path),
                "full_null_work_root": str(work_root),
                "full_null_durable_root": str(durable_root),
                "full_search_null_started": True,
                "full_search_null_completed": False,
                "final_connectivity_accessed": False,
                "final_connectivity_access_authorized": False,
            }
            _write_once(launch_path, launch)
            _write_state(
                state_path,
                {**state, "phase": "launch_published"},
            )
            return launch
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--procedure-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-search-null-script", type=Path, required=True)
    parser.add_argument("--full-search-null-aggregate-script", type=Path, required=True)
    parser.add_argument("--full-null-work-root", type=Path, required=True)
    parser.add_argument("--full-null-durable-root", type=Path, required=True)
    parser.add_argument("--resource-authorization", type=Path)
    parser.add_argument("--runtime-bundle", type=Path)
    arguments = parser.parse_args(argv)
    policy = EpisodePolicy.load(arguments.policy)
    result = dispatch_procedure_lock(
        lock=_read_json(arguments.procedure_lock),
        policy=policy,
        output_dir=arguments.output_dir,
        procedure_lock_path=arguments.procedure_lock,
        full_search_null_script=arguments.full_search_null_script,
        full_search_null_aggregate_script=arguments.full_search_null_aggregate_script,
        full_null_work_root=arguments.full_null_work_root,
        full_null_durable_root=arguments.full_null_durable_root,
        resource_authorization_path=arguments.resource_authorization,
        runtime_bundle=arguments.runtime_bundle,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_CPUS_PER_TASK",
    "AGGREGATE_MEMORY",
    "AGGREGATE_TIME_LIMIT",
    "ProcedureDispatchError",
    "SubmissionRejectedError",
    "dispatch_procedure_lock",
]
