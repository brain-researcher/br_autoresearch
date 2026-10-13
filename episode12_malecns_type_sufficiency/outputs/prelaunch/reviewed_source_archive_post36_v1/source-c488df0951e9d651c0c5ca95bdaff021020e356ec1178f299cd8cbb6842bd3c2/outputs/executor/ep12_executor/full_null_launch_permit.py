"""Durable authorization-bound permit for every EP12 full-null Slurm job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

from .policy import canonical_json, digest_object
from .source_freeze import verify_source_freeze_manifest


SCHEMA_VERSION = "ep12.full_null_launch_permit.v1"
EPISODE_ROOT = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency"
)
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class FullNullLaunchPermitError(RuntimeError):
    """A full-null worker lacks its exact authorized dispatch permit."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullNullLaunchPermitError(message)


def _read_bytes(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FullNullLaunchPermitError(f"cannot open {label}: {exc}") from exc
    try:
        _require(stat.S_ISREG(os.fstat(descriptor).st_mode), f"{label} is not regular")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _read(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    encoded = _read_bytes(path, label=label)
    try:
        value = json.loads(encoded)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FullNullLaunchPermitError(f"invalid {label}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    _require(encoded == (canonical_json(value) + "\n").encode(), f"{label} is not canonical")
    return value, encoded


def _file_sha256(path: Path, *, expected: str, label: str) -> None:
    _require(_HEX_RE.fullmatch(expected) is not None, f"{label} digest is malformed")
    encoded = _read_bytes(path, label=label)
    _require(hashlib.sha256(encoded).hexdigest() == expected, f"{label} bytes changed")


def _atomic_create(path: Path, value: Mapping[str, Any]) -> bytes:
    encoded = (canonical_json(value) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.parent.is_symlink(), "launch-permit directory is a symlink")
    if path.exists():
        _require(not path.is_symlink() and path.is_file(), "launch permit is unsafe")
        _require(path.read_bytes() == encoded, "existing launch permit differs")
        return encoded
    temporary = path.with_name(f".{path.name}.{os.getpid()}-{hashlib.sha256(encoded).hexdigest()[:12]}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o640,
    )
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "short launch-permit write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    return encoded


def create_full_null_launch_permit(
    *, permit_path: Path, authorization_evidence: Mapping[str, Any],
    launch: Mapping[str, Any], array_job_name: str | None,
    aggregate_job_name: str, array_indices: Sequence[int], generation: int = 0,
    parent_permit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one immutable permit for exact deterministic Slurm job names."""

    indices = sorted(set(map(int, array_indices)))
    _require(indices == list(map(int, array_indices)), "permit indices are not exact/sorted")
    _require(all(index >= 0 for index in indices), "permit has a negative array index")
    _require(bool(aggregate_job_name), "permit aggregate name is missing")
    _require((array_job_name is None) == (not indices), "permit array identity is inconsistent")
    core = {
        "schema_version": SCHEMA_VERSION,
        "authorization_evidence": dict(authorization_evidence),
        "launch": dict(launch),
        "launch_sha256": digest_object(launch),
        "jobs": {
            "generation": int(generation),
            "array_job_name": array_job_name,
            "aggregate_job_name": aggregate_job_name,
            "array_indices": indices,
        },
        "parent_permit": dict(parent_permit) if parent_permit is not None else None,
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    permit = {**core, "permit_sha256": digest_object(core)}
    encoded = _atomic_create(permit_path, permit)
    return {
        "permit_path": str(permit_path.resolve()),
        "permit_sha256": permit["permit_sha256"],
        "permit_file_sha256": hashlib.sha256(encoded).hexdigest(),
        "array_job_name": array_job_name,
        "aggregate_job_name": aggregate_job_name,
        "array_indices": indices,
        "generation": int(generation),
    }


def derive_full_null_launch_permit(
    *, parent_identity: Mapping[str, Any], permit_path: Path,
    array_job_name: str | None, aggregate_job_name: str,
    array_indices: Sequence[int], generation: int,
) -> dict[str, Any]:
    """Derive an exact recovery permit from an authenticated primary permit."""

    parent_path = Path(str(parent_identity.get("permit_path")))
    parent, encoded = _read(parent_path, label="parent full-null launch permit")
    parent_core = {
        key: value for key, value in parent.items() if key != "permit_sha256"
    }
    _require(
        parent.get("permit_sha256") == parent_identity.get("permit_sha256")
        and parent.get("permit_sha256") == digest_object(parent_core)
        and hashlib.sha256(encoded).hexdigest()
        == parent_identity.get("permit_file_sha256"),
        "parent launch-permit identity changed",
    )
    return create_full_null_launch_permit(
        permit_path=permit_path,
        authorization_evidence=parent["authorization_evidence"],
        launch=parent["launch"],
        array_job_name=array_job_name,
        aggregate_job_name=aggregate_job_name,
        array_indices=array_indices,
        generation=generation,
        parent_permit={
            "permit_path": str(parent_path.resolve()),
            "permit_sha256": parent["permit_sha256"],
            "permit_file_sha256": hashlib.sha256(encoded).hexdigest(),
        },
    )


def verify_full_null_launch_permit(
    *, permit_path: Path, expected_file_sha256: str, mode: str,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Independently authenticate a worker or aggregate before any work."""

    env = os.environ if environment is None else environment
    permit, encoded = _read(permit_path, label="full-null launch permit")
    core = {key: value for key, value in permit.items() if key != "permit_sha256"}
    _require(
        permit.get("schema_version") == SCHEMA_VERSION
        and permit.get("permit_sha256") == digest_object(core),
        "launch-permit self-hash mismatch",
    )
    _require(
        _HEX_RE.fullmatch(expected_file_sha256) is not None
        and hashlib.sha256(encoded).hexdigest() == expected_file_sha256,
        "launch-permit file digest mismatch",
    )
    launch = permit.get("launch")
    authorization = permit.get("authorization_evidence")
    jobs = permit.get("jobs")
    _require(isinstance(launch, Mapping), "permit launch is missing")
    _require(isinstance(authorization, Mapping), "permit authorization is missing")
    _require(isinstance(jobs, Mapping), "permit jobs are missing")
    _require(permit.get("launch_sha256") == digest_object(launch), "permit launch changed")

    for path_key, digest_key, label in (
        ("authorization_path", "authorization_file_sha256", "resource authorization"),
        ("pending_path", "pending_receipt_file_sha256", "pending receipt"),
    ):
        _file_sha256(Path(str(authorization[path_key])), expected=str(authorization[digest_key]), label=label)
    _file_sha256(
        Path(str(launch["procedure_lock_path"])),
        expected=str(launch["procedure_lock_file_sha256"]),
        label="procedure lock",
    )
    _file_sha256(
        Path(str(launch["full_search_null_script"])),
        expected=str(launch["full_search_null_script_sha256"]),
        label="full-null wrapper",
    )
    _file_sha256(
        Path(str(launch["full_search_null_aggregate_script"])),
        expected=str(launch["full_search_null_aggregate_script_sha256"]),
        label="aggregate wrapper",
    )
    source = launch.get("source_freeze")
    _require(isinstance(source, Mapping), "permit source freeze is missing")
    verify_source_freeze_manifest(
        Path(str(source["manifest_path"])),
        episode_root=EPISODE_ROOT,
        expected_file_sha256=str(source["manifest_file_sha256"]),
        require_current_surface=True,
    )
    expected_environment = {
        "EP12_FULL_NULL_RUNTIME_PROVIDER": launch["runtime_provider"],
        "EP12_FULL_NULL_BUNDLE": launch["runtime_bundle"],
        "EP12_FULL_NULL_BUNDLE_SHA256": launch["runtime_bundle_sha256"],
        "EP12_FULL_NULL_WORK_ROOT": launch["full_null_work_root"],
        "EP12_FULL_NULL_DURABLE_ROOT": launch["full_null_durable_root"],
        "EP12_PRE_NULL_PROCEDURE_LOCK_PATH": launch["procedure_lock_path"],
        "EP12_PRE_NULL_PROCEDURE_LOCK_SHA256": launch["procedure_lock_sha256"],
        "EP12_PRE_NULL_PROCEDURE_LOCK_FILE_SHA256": launch["procedure_lock_file_sha256"],
        "EP12_SOURCE_FREEZE_MANIFEST": source["manifest_path"],
        "EP12_SOURCE_FREEZE_MANIFEST_SHA256": source["manifest_file_sha256"],
    }
    for name, expected in expected_environment.items():
        _require(env.get(name) == expected, f"permit runtime export changed: {name}")
    observed_name = env.get("SLURM_JOB_NAME")
    if mode == "replicate":
        _require(observed_name == jobs.get("array_job_name"), "permit array job name changed")
        raw_index = env.get("SLURM_ARRAY_TASK_ID", "")
        _require(raw_index.isdigit(), "permit requires a numeric array task")
        _require(int(raw_index) in jobs.get("array_indices", []), "array task is outside permit")
        _require(env.get("SLURM_ARRAY_JOB_ID", "").isdigit(), "permit lacks array job ID")
    elif mode == "aggregate":
        _require(observed_name == jobs.get("aggregate_job_name"), "permit aggregate job name changed")
        _require(not env.get("SLURM_ARRAY_TASK_ID"), "aggregate permit used by an array task")
        _require(env.get("SLURM_JOB_ID", "").isdigit(), "permit lacks aggregate job ID")
    else:
        raise FullNullLaunchPermitError("permit mode must be replicate or aggregate")
    return permit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ep12-full-null-launch-permit")
    parser.add_argument("verify", choices=("verify",))
    parser.add_argument("--permit", type=Path, required=True)
    parser.add_argument("--expected-file-sha256", required=True)
    parser.add_argument("--mode", choices=("replicate", "aggregate"), required=True)
    args = parser.parse_args(argv)
    value = verify_full_null_launch_permit(
        permit_path=args.permit,
        expected_file_sha256=args.expected_file_sha256,
        mode=args.mode,
    )
    print(canonical_json({"permit_sha256": value["permit_sha256"], "mode": args.mode}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FullNullLaunchPermitError",
    "create_full_null_launch_permit",
    "derive_full_null_launch_permit",
    "verify_full_null_launch_permit",
]
