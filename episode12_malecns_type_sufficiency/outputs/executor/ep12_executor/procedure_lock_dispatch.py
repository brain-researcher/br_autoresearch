"""Dispatch the frozen EP12 null program from one semantic procedure lock."""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .policy import EpisodePolicy, canonical_json
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
_JOB_ID = re.compile(r"^(?P<job_id>[0-9]+)(?:;[A-Za-z0-9_.-]+)?$")


class ProcedureDispatchError(RuntimeError):
    """The procedure lock is not ready for one bounded dispatch."""


Submitter = Callable[[Sequence[str]], str]


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
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "")
        raise ProcedureDispatchError(f"sbatch submission failed: {stderr}") from error
    return completed.stdout.strip()


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
            if launch_path.exists():
                return _read_json(launch_path)
            if lock.get("procedure_locked") is not True or lock.get("null_eligible") is not True:
                decision = {
                    "action": "null_not_eligible",
                    "reason": "scientific_procedure_lock_not_eligible",
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
            null_script = full_search_null_script.resolve(strict=True)
            aggregate_script = full_search_null_aggregate_script.resolve(strict=True)
            work_root = (full_null_work_root or Path("/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/full_search_null")).resolve()
            durable_root = _inside(
                full_null_durable_root
                or OUTPUTS_ROOT / "development_run002" / "full_search_null",
                OUTPUTS_ROOT,
                label="full-null durable root",
            )
            array_spec = f"0-{policy.null_replicates - 1}"
            array_command = [
                "sbatch",
                "--parsable",
                f"--array={array_spec}",
                str(null_script),
                str(policy.path),
                str(procedure_lock_path),
                str(runtime_bundle.resolve(strict=True)),
                str(work_root),
                str(durable_root),
            ]
            array_job_id = _job_id(submitter(array_command))
            aggregate_command = [
                "sbatch",
                "--parsable",
                f"--dependency=afterok:{array_job_id}",
                str(aggregate_script),
                str(policy.path),
                str(procedure_lock_path),
                str(runtime_bundle.resolve(strict=True)),
                str(work_root),
                str(durable_root),
            ]
            aggregate_job_id = _job_id(submitter(aggregate_command))
            launch = {
                "action": "full_search_null_submitted",
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
    "dispatch_procedure_lock",
]
