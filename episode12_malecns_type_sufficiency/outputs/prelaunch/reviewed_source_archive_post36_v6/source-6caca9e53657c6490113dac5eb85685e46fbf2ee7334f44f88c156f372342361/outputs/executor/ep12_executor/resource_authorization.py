"""Fail-closed authorization for the measured EP12 full-null resource plan.

The measured procedure-lock controller creates only the pending receipt; it
never authorizes itself.  A dispatch is accepted only when a separately
provided expected file digest authenticates an exact, self-hashed authorization
whose runtime bundle, resource plan, projections, and pending receipt all match.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .full_search_runtime import verify_resource_plan
from .policy import EpisodePolicy, canonical_json, digest_object


PENDING_SCHEMA_VERSION = "ep12.resource_authorization_pending.v1"
AUTHORIZATION_SCHEMA_VERSION = "ep12.full_search_null_resource_authorization.v1"
RELEASE_SCHEMA_VERSION = "ep12.resource_authorization_release.v1"
AUTHORIZATION_SOURCE = "explicit_user_instruction_in_active_EP12_Codex_task"
AUTHORIZATION_SCOPE = "one_exact_99_replicate_full_search_null_program"
PROJECTION_FIELDS = (
    "projected_minimum_program_busy_core_hours",
    "projected_maximum_program_busy_core_hours",
    "projected_minimum_program_allocated_core_hours",
    "projected_maximum_program_allocated_core_hours",
)


class ResourceAuthorizationError(RuntimeError):
    """The expensive full-null dispatch lacks one exact authorization."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResourceAuthorizationError(message)


def _sha(value: Any, label: str) -> str:
    result = str(value)
    _require(
        len(result) == 64
        and all(character in "0123456789abcdef" for character in result),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return result


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a safe file")
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ResourceAuthorizationError(f"cannot read {label}: {error}") from error
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value, encoded


def _durable_file(path: Path, *, durable_root: Path, label: str) -> Path:
    raw = Path(path)
    _require(
        raw.is_absolute() and ".." not in raw.parts,
        f"{label} must be an absolute traversal-free path",
    )
    lexical = Path(os.path.abspath(os.fspath(raw)))
    resolved = raw.resolve(strict=True)
    root = durable_root.resolve(strict=True)
    _require(lexical == resolved, f"{label} may not traverse a symbolic link")
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ResourceAuthorizationError(
            f"{label} must remain under the durable dispatch root"
        ) from error
    _require(bool(relative.parts), f"{label} may not equal the durable dispatch root")
    _require(resolved.is_file() and not resolved.is_symlink(), f"{label} is unsafe")
    return resolved


def _exact_number(left: Any, right: Any, label: str) -> float:
    _require(
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and type(left) is type(right)
        and left == right,
        f"{label} differs from the exact generated resource plan",
    )
    result = float(left)
    _require(math.isfinite(result) and result > 0.0, f"{label} is not positive finite")
    return result


def _verify_self_hash(value: Mapping[str, Any], field: str, label: str) -> None:
    core = {key: item for key, item in value.items() if key != field}
    _require(value.get(field) == digest_object(core), f"{label} self-hash mismatch")


def _runtime_resource_evidence(
    *,
    runtime_bundle: Path,
    runtime_bundle_sha256: str,
    policy: EpisodePolicy,
) -> dict[str, Any]:
    manifest_path = runtime_bundle / "runtime_bundle.json"
    resource_plan_path = runtime_bundle / "resource_plan.json"
    manifest, manifest_bytes = _read_json(manifest_path, "runtime-bundle manifest")
    resource_plan_value, resource_plan_bytes = _read_json(
        resource_plan_path, "resource plan"
    )
    _verify_self_hash(manifest, "bundle_sha256", "runtime-bundle manifest")
    _require(
        manifest.get("bundle_sha256") == _sha(runtime_bundle_sha256, "runtime bundle"),
        "runtime-bundle identity differs from the requested launch",
    )
    try:
        resource_plan = verify_resource_plan(resource_plan_value, policy=policy)
    except Exception as error:
        raise ResourceAuthorizationError(str(error)) from error
    resource_plan_file_sha256 = hashlib.sha256(resource_plan_bytes).hexdigest()
    _require(
        manifest.get("resource_plan_sha256") == resource_plan["plan_sha256"]
        and isinstance(manifest.get("files"), Mapping)
        and manifest["files"].get("resource_plan.json")
        == resource_plan_file_sha256,
        "runtime bundle does not authenticate the exact resource plan",
    )
    projections = {
        field: resource_plan[field]
        for field in PROJECTION_FIELDS
    }
    values = [float(projections[field]) for field in PROJECTION_FIELDS]
    _require(
        all(math.isfinite(value) and value > 0.0 for value in values)
        and values[0] <= values[1]
        and values[2] <= values[3],
        "resource-plan program projections are invalid",
    )
    return {
        "runtime_bundle_sha256": manifest["bundle_sha256"],
        "runtime_bundle_manifest_file_sha256": hashlib.sha256(
            manifest_bytes
        ).hexdigest(),
        "resource_plan_sha256": resource_plan["plan_sha256"],
        "resource_plan_file_sha256": resource_plan_file_sha256,
        **projections,
        "program_search_count": int(resource_plan["program_search_count"]),
        "synthetic_null_search_count": int(
            resource_plan["synthetic_null_search_count"]
        ),
    }


def _verify_pending(
    value: Mapping[str, Any],
    *,
    encoded: bytes,
    evidence: Mapping[str, Any],
    procedure_lock_sha256: str,
    procedure_lock_file_sha256: str,
) -> dict[str, Any]:
    required = {
        "schema_version", "action", "reason", "auto_launch_full_null",
        "runtime_bundle_sha256", "runtime_bundle_manifest_file_sha256",
        "resource_plan_sha256", "resource_plan_file_sha256",
        "scratch_stage_receipt_sha256", "procedure_lock_sha256",
        "procedure_lock_file_sha256", *PROJECTION_FIELDS,
        "program_search_count", "synthetic_null_search_count",
        "full_search_null_submitted", "jobs_submitted", "development_only",
        "final_connectivity_accessed", "final_connectivity_access_authorized",
        "receipt_sha256",
    }
    pending = dict(value)
    _require(set(pending) == required, "resource pending-receipt schema differs")
    _verify_self_hash(pending, "receipt_sha256", "resource pending receipt")
    _require(
        pending.get("schema_version") == PENDING_SCHEMA_VERSION
        and pending.get("action") == "resource_authorization_pending"
        and pending.get("reason")
        == "explicit_exact_resource_authorization_required_before_full_null_dispatch"
        and pending.get("auto_launch_full_null") is False
        and pending.get("full_search_null_submitted") is False
        and pending.get("jobs_submitted") == []
        and pending.get("development_only") is True
        and pending.get("final_connectivity_accessed") is False
        and pending.get("final_connectivity_access_authorized") is False,
        "resource pending receipt does not describe a safe historical hold",
    )
    for field in (
        "runtime_bundle_sha256", "runtime_bundle_manifest_file_sha256",
        "resource_plan_sha256", "resource_plan_file_sha256",
        "program_search_count", "synthetic_null_search_count",
    ):
        _require(pending.get(field) == evidence[field], f"pending {field} changed")
    for field in PROJECTION_FIELDS:
        _exact_number(pending.get(field), evidence[field], f"pending {field}")
    _require(
        pending.get("procedure_lock_sha256") == procedure_lock_sha256
        and pending.get("procedure_lock_file_sha256")
        == procedure_lock_file_sha256,
        "resource pending receipt belongs to another procedure lock",
    )
    _sha(pending.get("scratch_stage_receipt_sha256"), "scratch-stage receipt")
    return {
        **pending,
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def verify_resource_authorization(
    *,
    authorization_path: Path,
    authorization_file_sha256: str,
    pending_path: Path,
    runtime_bundle: Path,
    runtime_bundle_sha256: str,
    procedure_lock_sha256: str,
    procedure_lock_file_sha256: str,
    policy: EpisodePolicy,
    durable_root: Path,
) -> dict[str, Any]:
    """Authenticate one exact, finite full-null resource authorization."""

    authorization_path = _durable_file(
        authorization_path, durable_root=durable_root, label="resource authorization"
    )
    pending_path = _durable_file(
        pending_path, durable_root=durable_root, label="resource pending receipt"
    )
    evidence = _runtime_resource_evidence(
        runtime_bundle=runtime_bundle,
        runtime_bundle_sha256=runtime_bundle_sha256,
        policy=policy,
    )
    pending_value, pending_bytes = _read_json(pending_path, "resource pending receipt")
    pending = _verify_pending(
        pending_value,
        encoded=pending_bytes,
        evidence=evidence,
        procedure_lock_sha256=procedure_lock_sha256,
        procedure_lock_file_sha256=procedure_lock_file_sha256,
    )
    authorization, authorization_bytes = _read_json(
        authorization_path, "resource authorization"
    )
    actual_file_sha256 = hashlib.sha256(authorization_bytes).hexdigest()
    _require(
        actual_file_sha256
        == _sha(authorization_file_sha256, "expected resource-authorization file"),
        "resource authorization bytes differ from the independently supplied digest",
    )
    required = {
        "schema_version", "episode_id", "action", "authorization_source",
        "scope", "allow_full_search_null_dispatch",
        "runtime_bundle_sha256", "runtime_bundle_manifest_file_sha256",
        "resource_plan_sha256", "resource_plan_file_sha256",
        "pending_receipt_sha256", "pending_receipt_file_sha256",
        "procedure_lock_sha256", "procedure_lock_file_sha256",
        *PROJECTION_FIELDS, "program_search_count",
        "synthetic_null_search_count", "development_only",
        "final_connectivity_accessed", "final_connectivity_access_authorized",
        "authorization_sha256",
    }
    _require(set(authorization) == required, "resource authorization schema differs")
    _verify_self_hash(
        authorization, "authorization_sha256", "resource authorization"
    )
    _require(
        authorization.get("schema_version") == AUTHORIZATION_SCHEMA_VERSION
        and authorization.get("episode_id") == "ep12"
        and authorization.get("action") == "authorize_full_search_null_dispatch"
        and authorization.get("authorization_source") == AUTHORIZATION_SOURCE
        and authorization.get("scope") == AUTHORIZATION_SCOPE
        and authorization.get("allow_full_search_null_dispatch") is True
        and authorization.get("development_only") is True
        and authorization.get("final_connectivity_accessed") is False
        and authorization.get("final_connectivity_access_authorized") is False,
        "resource authorization scope or safety fields are invalid",
    )
    for field in (
        "runtime_bundle_sha256", "runtime_bundle_manifest_file_sha256",
        "resource_plan_sha256", "resource_plan_file_sha256",
        "program_search_count", "synthetic_null_search_count",
    ):
        _require(
            authorization.get(field) == evidence[field],
            f"resource authorization {field} changed",
        )
    for field in PROJECTION_FIELDS:
        _exact_number(
            authorization.get(field), evidence[field],
            f"resource authorization {field}",
        )
    _require(
        authorization.get("pending_receipt_sha256") == pending["receipt_sha256"]
        and authorization.get("pending_receipt_file_sha256") == pending["file_sha256"]
        and authorization.get("procedure_lock_sha256") == procedure_lock_sha256
        and authorization.get("procedure_lock_file_sha256")
        == procedure_lock_file_sha256,
        "resource authorization is not bound to the pending hold and procedure lock",
    )
    return {
        "authorization_path": str(authorization_path),
        "authorization_sha256": authorization["authorization_sha256"],
        "authorization_file_sha256": actual_file_sha256,
        "pending_path": str(pending_path),
        "pending_receipt_sha256": pending["receipt_sha256"],
        "pending_receipt_file_sha256": pending["file_sha256"],
        **evidence,
    }


def write_resource_authorization_release(
    *,
    release_path: Path,
    authorization_evidence: Mapping[str, Any],
    dispatch_receipt_path: Path,
    dispatch_receipt: Mapping[str, Any],
    durable_root: Path,
) -> dict[str, Any]:
    """Write the immutable receipt that supersedes, but retains, the hold."""

    raw_release = Path(release_path)
    _require(
        raw_release.is_absolute() and ".." not in raw_release.parts,
        "resource release receipt path must be absolute and traversal-free",
    )
    release_parent = raw_release.parent.resolve(strict=True)
    root = durable_root.resolve(strict=True)
    try:
        relative = release_parent.relative_to(root)
    except ValueError as error:
        raise ResourceAuthorizationError(
            "resource release receipt must remain under the durable dispatch root"
        ) from error
    _require(bool(relative.parts), "resource release receipt may not use the root")
    release_path = Path(os.path.abspath(os.fspath(raw_release)))
    _require(
        release_path.parent == release_parent,
        "resource release receipt may not traverse a symbolic link",
    )
    dispatch_receipt_path = _durable_file(
        dispatch_receipt_path,
        durable_root=durable_root,
        label="dispatch receipt",
    )
    dispatch_bytes = dispatch_receipt_path.read_bytes()
    _require(
        dispatch_receipt.get("action") == "full_search_null_submitted"
        and dispatch_receipt.get("full_search_null_started") is True,
        "resource release requires a completed full-null dispatch receipt",
    )
    jobs = dispatch_receipt.get("jobs_submitted")
    _require(isinstance(jobs, list) and len(jobs) == 2, "dispatch job inventory differs")
    job_ids = [str(row.get("job_id")) for row in jobs if isinstance(row, Mapping)]
    _require(
        len(job_ids) == 2 and all(job_id.isdigit() for job_id in job_ids),
        "dispatch receipt lacks exact Slurm job IDs",
    )
    core = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "action": "resource_authorization_released_and_full_null_dispatched",
        "pending_receipt_status": "historical_superseded_by_authorized_dispatch",
        "pending_receipt_retained": True,
        "pending_receipt_path": authorization_evidence["pending_path"],
        "pending_receipt_sha256": authorization_evidence["pending_receipt_sha256"],
        "pending_receipt_file_sha256": authorization_evidence[
            "pending_receipt_file_sha256"
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
        "runtime_bundle_sha256": authorization_evidence["runtime_bundle_sha256"],
        "resource_plan_sha256": authorization_evidence["resource_plan_sha256"],
        **{
            field: authorization_evidence[field]
            for field in PROJECTION_FIELDS
        },
        "dispatch_receipt_path": str(dispatch_receipt_path),
        "dispatch_sha256": dispatch_receipt["dispatch_sha256"],
        "dispatch_receipt_file_sha256": hashlib.sha256(dispatch_bytes).hexdigest(),
        "dispatch_job_ids": job_ids,
        "jobs_submitted": [dict(row) for row in jobs],
        "full_search_null_submitted": True,
        "development_only": True,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    release = {**core, "release_sha256": digest_object(core)}
    encoded = (canonical_json(release) + "\n").encode("utf-8")
    if release_path.exists():
        existing, existing_bytes = _read_json(
            release_path, "resource authorization release"
        )
        _require(
            existing == release and existing_bytes == encoded,
            "existing resource authorization release differs",
        )
        return release
    temporary = release_path.with_name(
        f".{release_path.name}.{os.getpid()}-{hashlib.sha256(encoded).hexdigest()[:12]}.tmp"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "short write while creating resource release")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, release_path)
    return release


__all__: Sequence[str] = (
    "AUTHORIZATION_SCHEMA_VERSION",
    "AUTHORIZATION_SCOPE",
    "AUTHORIZATION_SOURCE",
    "PENDING_SCHEMA_VERSION",
    "PROJECTION_FIELDS",
    "RELEASE_SCHEMA_VERSION",
    "ResourceAuthorizationError",
    "verify_resource_authorization",
    "write_resource_authorization_release",
)
