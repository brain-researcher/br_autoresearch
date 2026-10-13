"""Crash-safe, exact-once Slurm handoffs for the EP12 observed chain.

Each handoff is one durable transaction keyed by its exact parent job and
transition kind.  The transaction binds an authenticated readiness or tail
ledger state, the child script bytes, exports, and the exact ``afterany``
dependency before contacting Slurm.  The context is durably committed before
submission and independently authenticated by the child, so parent success is
not a second scientific gate.  A deterministic child name permits
recovery through ``squeue`` and then ``sacct`` if ``sbatch`` accepted a job
but its acknowledgement did not reach the client.
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
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .policy import canonical_json, digest_object
from .slurm_client import sanitized_slurm_client_environment
from .secure_transaction import (
    SecureTransactionError,
    open_regular_lock_at,
    open_secure_directory,
)
from .source_freeze import (
    source_freeze_file_sha256,
    verify_staged_source,
    verify_source_freeze_manifest,
)


EPISODE_ROOT = Path(
    "/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/"
    "episode12_malecns_type_sufficiency"
)
OUTPUTS_ROOT = EPISODE_ROOT / "outputs"
SCRATCH_ROOT = Path(
    "/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency"
)
SOURCE_STAGE_PARENT = SCRATCH_ROOT / "source_stages"
DEFAULT_RUN_ROOT = OUTPUTS_ROOT / "development_run002"
DEFAULT_TRANSACTION_ROOT = DEFAULT_RUN_ROOT / "slurm_handoffs"

INTENT_SCHEMA = "ep12.slurm_child_handoff_intent.v3"
PROGRESS_SCHEMA = "ep12.slurm_child_handoff_progress.v3"
RECEIPT_SCHEMA = "ep12.slurm_child_handoff_receipt.v3"
CONTEXT_SCHEMA = "ep12.slurm_child_handoff_context.v1"

READINESS_SCHEMA = "ep12.production_readiness_gate.v2"
READINESS_MANIFEST_SCHEMA = "ep12.production_readiness_gate_manifest.v2"
_READINESS_V2_IDENTITY_FIELDS = (
    "observed_prefix_contract_sha256",
    "active_contract_sha256",
    "transition_manifest_file_sha256",
    "transition_receipt_file_sha256",
    "transition_receipt_sha256",
)
TAIL_STATUS_SCHEMA = "ep12.post36_tail_status.v2"
TAIL_SCHEDULE_SCHEMA = "ep12.post36_realized_schedule.v2"
TAIL_SCHEDULE_MANIFEST_SCHEMA = "ep12.post36_realized_schedule_manifest.v2"

HANDOFF_KINDS = {
    "readiness_to_post36_tail": "r2t",
    "post36_tail_to_next_tail": "t2t",
    "post36_tail_to_procedure_lock": "t2p",
}
CANONICAL_CHILD_SCRIPTS = {
    "readiness_to_post36_tail": (
        EPISODE_ROOT / "outputs" / "executor" / "slurm" / "run_post36_tail.sbatch"
    ),
    "post36_tail_to_next_tail": (
        EPISODE_ROOT / "outputs" / "executor" / "slurm" / "run_post36_tail.sbatch"
    ),
    "post36_tail_to_procedure_lock": (
        EPISODE_ROOT / "outputs" / "executor" / "slurm" / "run_procedure_lock.sbatch"
    ),
}
CANONICAL_CHILD_RELATIVES = {
    "readiness_to_post36_tail": Path(
        "outputs/executor/slurm/run_post36_tail.sbatch"
    ),
    "post36_tail_to_next_tail": Path(
        "outputs/executor/slurm/run_post36_tail.sbatch"
    ),
    "post36_tail_to_procedure_lock": Path(
        "outputs/executor/slurm/run_procedure_lock.sbatch"
    ),
}
_SBATCH_RESULT_RE = re.compile(r"^(?P<job_id>[0-9]+)(?:;[A-Za-z0-9_.-]+)?$")
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

_SOURCE_FREEZE_EXPORTS = {
    "EP12_SOURCE_FREEZE_MANIFEST",
    "EP12_SOURCE_FREEZE_MANIFEST_SHA256",
    "EP12_SOURCE_STAGE_ROOT",
}
_HANDOFF_EXPORTS = {
    "readiness_to_post36_tail": _SOURCE_FREEZE_EXPORTS,
    "post36_tail_to_next_tail": _SOURCE_FREEZE_EXPORTS,
    "post36_tail_to_procedure_lock": _SOURCE_FREEZE_EXPORTS
    | {"EP12_PROCEDURE_LOCK_LAUNCH_MODE", "EP12_AUTO_LAUNCH_FULL_NULL"},
}


class SlurmHandoffError(RuntimeError):
    """A child cannot be submitted without risking a duplicate or mutation."""


class SubmissionNotInvokedError(SlurmHandoffError):
    """The local client could not invoke ``sbatch``; no request was sent."""


Submitter = Callable[[Sequence[str]], str]
JobResolver = Callable[[str], str | None]
SourceFreezeAuthenticator = Callable[[Mapping[str, str], Path], dict[str, Any]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SlurmHandoffError(message)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a JSON object")
    return value


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    boundary = root.resolve()
    _require(
        resolved == boundary or boundary in resolved.parents,
        f"{label} must stay under {boundary}: {resolved}",
    )
    return resolved


def _regular(path: Path, *, label: str) -> Path:
    _require(path.exists(), f"missing {label}: {path}")
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    return path.resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _effective_user() -> str:
    """Return the scheduler account name from the effective OS identity."""

    try:
        value = pwd.getpwuid(os.geteuid()).pw_name
    except (KeyError, OSError) as exc:
        raise SlurmHandoffError("cannot resolve the effective OS user") from exc
    _require(bool(value) and "|" not in value, "effective OS user is invalid")
    return value


def _explicit_exports(*, kind: str, exported: str) -> tuple[dict[str, str], str]:
    """Parse one exact Slurm export list without ambient inheritance."""

    _require(
        exported == exported.strip() and bool(exported),
        "Slurm export string is invalid",
    )
    entries = exported.split(",")
    _require(
        all(entry and entry not in {"ALL", "NONE"} for entry in entries),
        "Slurm export may not inherit or use an ambient environment",
    )
    values: dict[str, str] = {}
    for entry in entries:
        _require("=" in entry, "every Slurm export must freeze an explicit value")
        name, value = entry.split("=", 1)
        _require(
            _ENVIRONMENT_NAME_RE.fullmatch(name) is not None,
            f"invalid Slurm export name: {name!r}",
        )
        _require(name not in values, f"duplicate Slurm export: {name}")
        _require(
            "\n" not in value and "\r" not in value and "," not in value,
            f"unsafe Slurm export value: {name}",
        )
        values[name] = value
    _require(set(values) == _HANDOFF_EXPORTS[kind], "handoff export set changed")
    if kind == "post36_tail_to_procedure_lock":
        _require(
            values["EP12_PROCEDURE_LOCK_LAUNCH_MODE"] == "hold"
            and values["EP12_AUTO_LAUNCH_FULL_NULL"] == "0",
            "tail may launch only the frozen resource-hold procedure lock",
        )
    _require(
        _HEX_RE.fullmatch(values["EP12_SOURCE_FREEZE_MANIFEST_SHA256"])
        is not None,
        "source-freeze file SHA-256 is invalid",
    )
    _require(
        values["EP12_SOURCE_STAGE_ROOT"]
        == str(
            SOURCE_STAGE_PARENT
            / f"source-{values['EP12_SOURCE_FREEZE_MANIFEST_SHA256']}"
        ),
        "source-stage root is not the freeze content address",
    )
    canonical = ",".join(f"{name}={values[name]}" for name in sorted(values))
    return values, canonical


def _authenticate_source_freeze(
    exports: Mapping[str, str], required_child_script: Path
) -> dict[str, Any]:
    manifest_path = Path(exports["EP12_SOURCE_FREEZE_MANIFEST"])
    expected_file_sha256 = exports["EP12_SOURCE_FREEZE_MANIFEST_SHA256"]
    try:
        try:
            relative = required_child_script.relative_to(EPISODE_ROOT)
        except ValueError:
            stage_root = SOURCE_STAGE_PARENT / f"source-{expected_file_sha256}"
            relative = required_child_script.relative_to(stage_root)
            manifest = verify_staged_source(
                stage_root,
                manifest_path,
                episode_root=EPISODE_ROOT,
                scratch_root=SCRATCH_ROOT,
                expected_file_sha256=expected_file_sha256,
            )
            source_origin = "immutable_source_stage"
        else:
            manifest = verify_source_freeze_manifest(
                manifest_path,
                episode_root=EPISODE_ROOT,
                expected_file_sha256=expected_file_sha256,
                require_current_surface=True,
            )
            source_origin = "canonical_live_surface"
        observed_file_sha256 = source_freeze_file_sha256(
            manifest_path, episode_root=EPISODE_ROOT
        )
    except Exception as exc:
        raise SlurmHandoffError(f"source-freeze verification failed: {exc}") from exc
    _require(
        observed_file_sha256 == expected_file_sha256,
        "source-freeze bytes changed during handoff authentication",
    )
    relative_text = relative.as_posix()
    records = [
        record for record in manifest.get("files", [])
        if isinstance(record, Mapping) and record.get("path") == relative_text
    ]
    _require(len(records) == 1, "canonical child script is absent from source freeze")
    record = dict(records[0])
    _require(
        record.get("sha256") == _sha256_file(required_child_script)
        and record.get("size_bytes") == required_child_script.stat().st_size,
        "canonical child script differs from source freeze",
    )
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": observed_file_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "file_count": manifest["file_count"],
        "required_child_script": record,
        "child_source_origin": source_origin,
    }


def _canonical_child_script(
    kind: str, child_script: Path, *, source_freeze_file_sha256: str
) -> Path:
    expected = CANONICAL_CHILD_SCRIPTS[kind]
    supplied = Path(os.path.abspath(os.fspath(child_script)))
    if supplied == expected:
        return _regular(expected, label="canonical child Slurm script")
    expected_stage_root = SOURCE_STAGE_PARENT / f"source-{source_freeze_file_sha256}"
    expected_staged = expected_stage_root / CANONICAL_CHILD_RELATIVES[kind]
    _require(
        supplied == expected_staged,
        "handoff child is neither the canonical wrapper nor its immutable staged copy",
    )
    return _regular(expected_staged, label="immutable staged child Slurm script")


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    path = _regular(path, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SlurmHandoffError(f"cannot read {label}: {exc}") from exc
    return _mapping(value, label=label)


def _encoded(value: Mapping[str, Any]) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _atomic_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}-{hashlib.sha256(content).hexdigest()[:12]}.tmp"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, f"short write while creating {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_or_verify(path: Path, value: Mapping[str, Any], *, label: str) -> None:
    content = _encoded(value)
    if path.exists():
        _require(path.is_file() and not path.is_symlink(), f"existing {label} is unsafe")
        _require(path.read_bytes() == content, f"existing {label} differs from transaction")
        return
    _atomic_bytes(path, content)


def _read_canonical(path: Path, *, label: str) -> dict[str, Any]:
    value = dict(_read_json(path, label=label))
    _require(path.read_bytes() == _encoded(value), f"existing {label} bytes were mutated")
    return value


def _verify_digest(value: Mapping[str, Any], field: str, *, label: str) -> dict[str, Any]:
    claimed = value.get(field)
    core = {key: item for key, item in value.items() if key != field}
    _require(
        isinstance(claimed, str) and claimed == digest_object(core),
        f"{label} digest mismatch",
    )
    return core


def _verify_ledger(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _regular(path, label="tail ledger")
    raw = path.read_bytes()
    _require(bool(raw) and raw.endswith(b"\n"), "tail ledger lacks a final record boundary")
    records: list[dict[str, Any]] = []
    previous: str | None = None
    event_ids: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SlurmHandoffError(f"invalid tail ledger line {line_number}") from exc
        _require(isinstance(record, dict), f"tail ledger line {line_number} is not an object")
        _require(
            record.get("previous_record_hash") == previous,
            f"broken tail ledger chain at line {line_number}",
        )
        claimed = record.get("record_hash")
        unhashed = dict(record)
        unhashed.pop("record_hash", None)
        _require(
            isinstance(claimed, str) and claimed == digest_object(unhashed),
            f"tail ledger record hash mismatch at line {line_number}",
        )
        event_id = record.get("event_id")
        _require(
            isinstance(event_id, str) and event_id and event_id not in event_ids,
            f"invalid or duplicate tail ledger event at line {line_number}",
        )
        event_ids.add(event_id)
        records.append(record)
        previous = claimed
    identity = {
        "record_count": len(records),
        "head_hash": records[-1]["record_hash"],
        "journal_hash": digest_object(records),
    }
    return records, identity


def _authenticate_readiness(context_root: Path) -> dict[str, Any]:
    report_path = context_root / "production_readiness.json"
    manifest_path = context_root / "manifest.json"
    report = _read_json(report_path, label="production readiness report")
    manifest = _read_json(manifest_path, label="production readiness manifest")
    _require(report.get("schema_version") == READINESS_SCHEMA, "readiness schema mismatch")
    _require(
        manifest.get("schema_version") == READINESS_MANIFEST_SCHEMA,
        "readiness manifest schema mismatch",
    )
    expected_manifest_fields = {
        "schema_version",
        "files",
        "production_readiness_passed",
        *_READINESS_V2_IDENTITY_FIELDS,
        "procedure_locked",
        "full_search_null_completed",
        "final_connectivity_accessed",
        "final_connectivity_access_authorized",
    }
    _require(
        set(manifest) == expected_manifest_fields,
        "readiness v2 manifest field set changed",
    )
    files = _mapping(manifest.get("files"), label="readiness manifest files")
    _require(set(files) == {"production_readiness.json"}, "readiness file set changed")
    report_sha256 = _sha256_file(report_path)
    _require(
        files.get("production_readiness.json") == report_sha256,
        "readiness report hash mismatch",
    )
    transition_identity: dict[str, str] = {}
    for field in _READINESS_V2_IDENTITY_FIELDS:
        value = report.get(field)
        _require(
            isinstance(value, str) and _HEX_RE.fullmatch(value) is not None,
            f"readiness v2 report has invalid {field}",
        )
        _require(
            manifest.get(field) == value,
            f"readiness v2 manifest disagrees on {field}",
        )
        transition_identity[field] = value
    policy_sha256 = report.get("policy_sha256")
    _require(
        isinstance(policy_sha256, str)
        and _HEX_RE.fullmatch(policy_sha256) is not None,
        "readiness v2 policy SHA-256 is invalid",
    )
    guards = _mapping(report.get("claim_guards"), label="readiness claim guards")
    observed_guard = _mapping(
        guards.get("observed_post36_tail_launch"), label="observed-tail readiness guard"
    )
    checks = report.get("checks")
    _require(isinstance(checks, list) and bool(checks), "readiness checks are missing")
    _require(
        report.get("production_readiness_passed") is True
        and manifest.get("production_readiness_passed") is True
        and observed_guard.get("permitted") is True
        and report.get("failed_foundational_checks") == []
        and all(isinstance(check, Mapping) and check.get("passed") is True for check in checks),
        "readiness does not authenticate a permitted observed-tail launch",
    )
    for value, label in (
        (report.get("observed_outcome_values_read"), "observed-outcome decoding"),
        (report.get("procedure_locked"), "procedure lock"),
        (report.get("full_search_null_completed"), "full-search null completion"),
        (report.get("final_connectivity_accessed"), "final connectivity access"),
        (report.get("final_connectivity_access_authorized"), "final authorization"),
        (manifest.get("procedure_locked"), "manifest procedure lock"),
        (manifest.get("full_search_null_completed"), "manifest null completion"),
        (manifest.get("final_connectivity_accessed"), "manifest final access"),
        (manifest.get("final_connectivity_access_authorized"), "manifest final authorization"),
    ):
        _require(value is False, f"readiness unexpectedly records {label}")
    return {
        "context_kind": "readiness_passed",
        "context_root": str(context_root),
        "readiness_report_sha256": report_sha256,
        "readiness_manifest_sha256": _sha256_file(manifest_path),
        "policy_sha256": report.get("policy_sha256"),
        "readiness_schema_version": READINESS_SCHEMA,
        "readiness_manifest_schema_version": READINESS_MANIFEST_SCHEMA,
        "scientific_transition": transition_identity,
        "observed_outcome_values_read": False,
        "final_connectivity_accessed": False,
    }


def _authenticate_tail(context_root: Path, *, stopped: bool) -> dict[str, Any]:
    status_path = context_root / "controller_status.json"
    ledger_path = context_root / "ledger.jsonl"
    status = _read_json(status_path, label="tail controller status")
    records, ledger = _verify_ledger(ledger_path)
    _require(status.get("schema_version") == TAIL_STATUS_SCHEMA, "tail status schema mismatch")
    expected_action = "stopped" if stopped else "continue"
    _require(status.get("action") == expected_action, f"tail action is not {expected_action}")
    _require(status.get("ledger") == ledger, "tail status does not bind the exact ledger")
    ordinal = status.get("current_valid_ordinal")
    _require(isinstance(ordinal, int) and ordinal >= 36, "tail status ordinal is invalid")
    last = records[-1]
    if stopped:
        _require(last.get("record_kind") == "controller_stop", "stopped tail lacks a stop record")
        _require(
            last.get("valid_trial_ordinal") == ordinal
            and last.get("reason") == status.get("reason"),
            "tail stop record and status disagree",
        )
        _require(status.get("next_slot_id") is None, "stopped tail names a next slot")
    else:
        _require(
            last.get("record_kind") == "slot_disposition"
            and last.get("valid_ordinal_advanced") is True
            and last.get("valid_trial_ordinal") == ordinal,
            "continuing tail is not bound to a committed slot disposition",
        )
        _require(
            isinstance(status.get("next_slot_id"), str) and status.get("next_slot_id"),
            "continuing tail does not identify its next slot",
        )
    for value, label in (
        (status.get("procedure_locked"), "procedure lock"),
        (status.get("full_search_null_completed"), "full-search null completion"),
        (status.get("final_connectivity_accessed"), "final connectivity access"),
        (status.get("final_connectivity_access_authorized"), "final authorization"),
    ):
        _require(value is False, f"tail status unexpectedly records {label}")
    context: dict[str, Any] = {
        "context_kind": "tail_stopped" if stopped else "tail_continue",
        "context_root": str(context_root),
        "controller_status_sha256": _sha256_file(status_path),
        "ledger_file_sha256": _sha256_file(ledger_path),
        "ledger_record_count": ledger["record_count"],
        "ledger_head_hash": ledger["head_hash"],
        "ledger_hash": ledger["journal_hash"],
        "current_valid_ordinal": ordinal,
        "next_slot_id": status.get("next_slot_id"),
        "final_connectivity_accessed": False,
    }
    if stopped:
        schedule_path = context_root / "realized_schedule.json"
        manifest_path = context_root / "realized_schedule_manifest.json"
        schedule = _read_json(schedule_path, label="tail realized schedule")
        manifest = _read_json(manifest_path, label="tail realized schedule manifest")
        _require(
            schedule.get("schema_version") == TAIL_SCHEDULE_SCHEMA,
            "tail realized schedule schema mismatch",
        )
        _require(
            manifest.get("schema_version") == TAIL_SCHEDULE_MANIFEST_SCHEMA,
            "tail schedule manifest schema mismatch",
        )
        files = _mapping(manifest.get("files"), label="tail schedule manifest files")
        _require(
            set(files) == {"realized_schedule.json", "ledger.jsonl"},
            "tail schedule manifest file set changed",
        )
        _require(
            files.get("realized_schedule.json") == _sha256_file(schedule_path)
            and files.get("ledger.jsonl") == _sha256_file(ledger_path),
            "tail schedule manifest hash mismatch",
        )
        _require(
            schedule.get("stop_valid_trial_ordinal") == ordinal
            and schedule.get("stop_reason") == status.get("reason")
            and schedule.get("ledger_record_count") == ledger["record_count"]
            and schedule.get("ledger_head_hash") == ledger["head_hash"]
            and schedule.get("ledger_hash") == ledger["journal_hash"]
            and manifest.get("ledger_head_hash") == ledger["head_hash"],
            "tail realized schedule is not bound to the stop ledger",
        )
        for value in (
            schedule.get("procedure_locked"),
            schedule.get("full_search_null_completed"),
            schedule.get("final_connectivity_accessed"),
            schedule.get("final_connectivity_access_authorized"),
            manifest.get("procedure_locked"),
            manifest.get("full_search_null_completed"),
            manifest.get("final_connectivity_accessed"),
        ):
            _require(value is False, "tail schedule crosses a downstream gate")
        context.update(
            {
                "realized_schedule_sha256": _sha256_file(schedule_path),
                "realized_schedule_manifest_sha256": _sha256_file(manifest_path),
                "completed_promotable_search": schedule.get("completed_promotable_search"),
            }
        )
    return context


def authenticate_handoff_context(
    *, kind: str, context_root: Path, run_root: Path = DEFAULT_RUN_ROOT
) -> dict[str, Any]:
    """Authenticate only operational state needed for one child transition."""

    _require(kind in HANDOFF_KINDS, f"unsupported handoff kind: {kind}")
    root = _inside(context_root, run_root, label="handoff context")
    _require(root.is_dir() and not root.is_symlink(), "handoff context is not a directory")
    if kind == "readiness_to_post36_tail":
        core = _authenticate_readiness(root)
    elif kind == "post36_tail_to_next_tail":
        core = _authenticate_tail(root, stopped=False)
    else:
        core = _authenticate_tail(root, stopped=True)
    value = {"schema_version": CONTEXT_SCHEMA, "handoff_kind": kind, **core}
    return {**value, "context_sha256": digest_object(value)}


def _parse_job_id(output: str) -> str:
    match = _SBATCH_RESULT_RE.fullmatch(output.strip())
    if match is None:
        raise SlurmHandoffError(f"unexpected sbatch --parsable output: {output!r}")
    return match.group("job_id")


def _resolved_job_id(
    output: str, job_name: str, *, expected_user: str | None = None
) -> str | None:
    matches: set[str] = set()
    for raw in output.splitlines():
        fields = raw.strip().split("|")
        expected_fields = 3 if expected_user is not None else 2
        if len(fields) != expected_fields:
            continue
        job_id, observed_name = fields[:2]
        if observed_name == job_name and expected_user is not None:
            _require(
                fields[2] == expected_user,
                f"scheduler recovery name belongs to another user: {fields[2]!r}",
            )
        if job_id.isdigit() and observed_name == job_name:
            matches.add(job_id)
    _require(
        len(matches) <= 1,
        f"multiple scheduler jobs carry exact recovery name {job_name!r}",
    )
    return next(iter(matches)) if matches else None


def _run_scheduler(command: Sequence[str]) -> str:
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
        raise SlurmHandoffError(f"scheduler command failed: {stderr}") from exc
    return completed.stdout


def default_job_resolver(job_name: str) -> str | None:
    """Resolve exactly one accepted child across live and accounting state."""

    effective_user = _effective_user()
    commands = (
        [
            "squeue", "--noheader", "--user", effective_user,
            "--name", job_name, "--format=%A|%j|%u",
        ],
        [
            "sacct",
            "-X",
            "--noheader",
            "--parsable2",
            "--user",
            effective_user,
            "--name",
            job_name,
            "--starttime",
            "2000-01-01",
            "--format=JobIDRaw,JobName%64,User",
        ],
    )
    matches: set[str] = set()
    for command in commands:
        recovered = _resolved_job_id(
            _run_scheduler(command), job_name, expected_user=effective_user
        )
        if recovered is not None:
            matches.add(recovered)
    _require(
        len(matches) <= 1,
        f"multiple scheduler jobs carry exact recovery name {job_name!r}",
    )
    return next(iter(matches)) if matches else None


def default_submitter(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=sanitized_slurm_client_environment(),
        )
    except OSError as exc:
        # Failure to create the local sbatch process is the one case in which
        # retrying the invocation is known to be safe.
        raise SubmissionNotInvokedError(
            f"sbatch was not invoked: {exc}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        # Treat every post-invocation failure as ambiguous.  The caller has
        # already durably entered submission_uncertain before reaching here.
        raise SlurmHandoffError(
            f"sbatch invocation failed after launch: {exc.stderr}"
        ) from exc
    return completed.stdout.strip()


def _progress_value(
    *, intent_sha256: str, child_job_name: str, state: str, child_job_id: str | None
) -> dict[str, Any]:
    core = {
        "schema_version": PROGRESS_SCHEMA,
        "intent_sha256": intent_sha256,
        "state": state,
        "child_job_name": child_job_name,
        "child_job_id": child_job_id,
        "duplicate_submission_permitted": False,
        "final_connectivity_accessed": False,
    }
    return {**core, "progress_sha256": digest_object(core)}


def _verify_progress(
    value: Mapping[str, Any], *, intent_sha256: str, child_job_name: str
) -> tuple[str, str | None]:
    _verify_digest(value, "progress_sha256", label="handoff progress")
    _require(value.get("schema_version") == PROGRESS_SCHEMA, "handoff progress schema mismatch")
    _require(value.get("intent_sha256") == intent_sha256, "handoff progress intent changed")
    _require(value.get("child_job_name") == child_job_name, "child recovery name changed")
    _require(value.get("duplicate_submission_permitted") is False, "duplicate submission enabled")
    _require(value.get("final_connectivity_accessed") is False, "progress records final access")
    state = value.get("state")
    job_id = value.get("child_job_id")
    _require(
        state in {"planned", "submission_uncertain", "submitted"},
        "invalid handoff progress state",
    )
    if state in {"planned", "submission_uncertain"}:
        _require(job_id is None, f"{state} handoff unexpectedly has a job ID")
    else:
        _require(isinstance(job_id, str) and job_id.isdigit(), "submitted handoff lacks job ID")
    return str(state), job_id if isinstance(job_id, str) else None


def submit_handoff(
    *,
    kind: str,
    parent_job_id: str,
    context_root: Path,
    transaction_root: Path,
    child_script: Path,
    exported: str,
    child_arguments: Sequence[str] = (),
    run_root: Path = DEFAULT_RUN_ROOT,
    submitter: Submitter = default_submitter,
    job_resolver: JobResolver = default_job_resolver,
    source_freeze_authenticator: SourceFreezeAuthenticator = _authenticate_source_freeze,
) -> dict[str, Any]:
    """Create, recover, and finalize one exact parent-to-child transaction."""

    _require(kind in HANDOFF_KINDS, f"unsupported handoff kind: {kind}")
    _require(parent_job_id.isdigit(), "parent Slurm job ID must be numeric")
    exports, canonical_exported = _explicit_exports(kind=kind, exported=exported)
    script = _canonical_child_script(
        kind,
        child_script,
        source_freeze_file_sha256=exports["EP12_SOURCE_FREEZE_MANIFEST_SHA256"],
    )
    source_freeze = source_freeze_authenticator(exports, script)
    run_root = run_root.resolve()
    lexical_transactions = Path(os.path.abspath(os.fspath(transaction_root)))
    _require(
        lexical_transactions == run_root or run_root in lexical_transactions.parents,
        "handoff transaction root escapes run root",
    )
    context = authenticate_handoff_context(
        kind=kind, context_root=context_root, run_root=run_root
    )
    launch = {
        "handoff_kind": kind,
        "parent_job_id": parent_job_id,
        "dependency": f"afterany:{parent_job_id}",
        "parent_success_required": False,
        "authenticated_context_committed_before_submission": True,
        "child_independently_authenticates_context": True,
        "kill_on_invalid_dependency": True,
        "context": context,
        "context_sha256": context["context_sha256"],
        "child_script": str(script),
        "child_script_sha256": _sha256_file(script),
        "exports": exports,
        "exported": canonical_exported,
        "ambient_environment_inherited": False,
        "source_freeze": source_freeze,
        "child_arguments": [str(value) for value in child_arguments],
    }
    launch_sha256 = digest_object(launch)
    child_job_name = (
        f"ep12-{HANDOFF_KINDS[kind]}-{parent_job_id}-{launch_sha256[:16]}"
    )
    intent_core = {
        "schema_version": INTENT_SCHEMA,
        "launch": launch,
        "launch_sha256": launch_sha256,
        "child_job_name": child_job_name,
        "exact_parent_afterany_dependency_required": True,
        "parent_success_required": False,
        "duplicate_submission_permitted": False,
        "final_connectivity_accessed": False,
    }
    intent = {**intent_core, "intent_sha256": digest_object(intent_core)}

    try:
        transaction_dir, transaction_fd = open_secure_directory(
            lexical_transactions / kind / f"parent-{parent_job_id}",
            root=run_root,
            create=True,
        )
        lease_fd = open_regular_lock_at(transaction_fd, ".handoff.lock")
    except SecureTransactionError as exc:
        raise SlurmHandoffError(str(exc)) from exc
    try:
        fcntl.flock(lease_fd, fcntl.LOCK_EX)
        try:
            intent_path = transaction_dir / "intent.json"
            progress_path = transaction_dir / "progress.json"
            receipt_path = transaction_dir / "handoff_receipt.json"
            _write_or_verify(intent_path, intent, label="handoff intent")

            if progress_path.exists():
                progress = _read_canonical(progress_path, label="handoff progress")
                state, child_job_id = _verify_progress(
                    progress,
                    intent_sha256=intent["intent_sha256"],
                    child_job_name=child_job_name,
                )
            else:
                state, child_job_id = "planned", None
                progress = _progress_value(
                    intent_sha256=intent["intent_sha256"],
                    child_job_name=child_job_name,
                    state=state,
                    child_job_id=None,
                )
                _write_or_verify(progress_path, progress, label="handoff progress")

            if receipt_path.exists():
                receipt = _read_canonical(receipt_path, label="handoff receipt")
                _verify_digest(receipt, "receipt_sha256", label="handoff receipt")
                _require(state == "submitted", "final handoff receipt lacks submitted progress")
                _require(
                    receipt.get("schema_version") == RECEIPT_SCHEMA
                    and receipt.get("handoff_kind") == kind
                    and receipt.get("parent_job_id") == parent_job_id
                    and receipt.get("dependency") == f"afterany:{parent_job_id}"
                    and receipt.get("parent_success_required") is False
                    and receipt.get("authenticated_context_committed_before_submission")
                    is True
                    and receipt.get("child_independently_authenticates_context")
                    is True
                    and receipt.get("kill_on_invalid_dependency") is True
                    and receipt.get("intent_sha256") == intent["intent_sha256"]
                    and receipt.get("progress_sha256") == progress["progress_sha256"]
                    and receipt.get("context_sha256") == context["context_sha256"]
                    and receipt.get("child_job_id") == child_job_id
                    and receipt.get("child_job_name") == child_job_name
                    and receipt.get("exports") == exports
                    and receipt.get("ambient_environment_inherited") is False
                    and receipt.get("source_freeze") == source_freeze,
                    "final handoff receipt does not match its transaction",
                )
                return receipt

            if state == "planned":
                child_job_id = job_resolver(child_job_name)
                if child_job_id is None:
                    command = [
                        "sbatch",
                        "--parsable",
                        f"--job-name={child_job_name}",
                        "--partition=normal",
                        "--cpus-per-task=32",
                        "--mem=192G",
                        "--time=12:00:00",
                        "--requeue",
                        "--open-mode=append",
                        f"--dependency=afterany:{parent_job_id}",
                        "--kill-on-invalid-dep=yes",
                        f"--export={canonical_exported}",
                        str(script),
                        *[str(value) for value in child_arguments],
                    ]
                    # Enter the no-resubmit state before invoking sbatch.  This
                    # also closes the hard-interruption window between scheduler
                    # acceptance and a Python exception handler.
                    progress = _progress_value(
                        intent_sha256=intent["intent_sha256"],
                        child_job_name=child_job_name,
                        state="submission_uncertain",
                        child_job_id=None,
                    )
                    _atomic_bytes(progress_path, _encoded(progress))
                    state = "submission_uncertain"
                    try:
                        child_job_id = _parse_job_id(submitter(command))
                    except SubmissionNotInvokedError:
                        # Only an error known to precede process invocation may
                        # return the transaction to a retryable planned state.
                        progress = _progress_value(
                            intent_sha256=intent["intent_sha256"],
                            child_job_name=child_job_name,
                            state="planned",
                            child_job_id=None,
                        )
                        _atomic_bytes(progress_path, _encoded(progress))
                        raise
                    except Exception as submission_error:
                        recovered = job_resolver(child_job_name)
                        if recovered is None:
                            raise SlurmHandoffError(
                                "child submission acceptance is uncertain; "
                                "later retries may resolve the exact child name "
                                "but must not invoke sbatch again"
                            ) from submission_error
                        child_job_id = recovered
            if state == "submission_uncertain" and child_job_id is None:
                child_job_id = job_resolver(child_job_name)
                if child_job_id is None:
                    raise SlurmHandoffError(
                        "child submission remains uncertain and is not yet "
                        "visible by its exact scheduler name; refusing duplicate sbatch"
                    )
            if state in {"planned", "submission_uncertain"}:
                _require(
                    isinstance(child_job_id, str) and child_job_id.isdigit(),
                    "scheduler recovery returned an invalid child job ID",
                )
                progress = _progress_value(
                    intent_sha256=intent["intent_sha256"],
                    child_job_name=child_job_name,
                    state="submitted",
                    child_job_id=child_job_id,
                )
                _atomic_bytes(progress_path, _encoded(progress))
                state = "submitted"

            _require(state == "submitted" and child_job_id is not None, "handoff did not submit")
            receipt_core = {
                "schema_version": RECEIPT_SCHEMA,
                "handoff_kind": kind,
                "parent_job_id": parent_job_id,
                "dependency": f"afterany:{parent_job_id}",
                "parent_success_required": False,
                "authenticated_context_committed_before_submission": True,
                "child_independently_authenticates_context": True,
                "kill_on_invalid_dependency": True,
                "intent_sha256": intent["intent_sha256"],
                "progress_sha256": progress["progress_sha256"],
                "context_sha256": context["context_sha256"],
                "child_job_id": child_job_id,
                "child_job_name": child_job_name,
                "child_script_sha256": launch["child_script_sha256"],
                "exports": exports,
                "ambient_environment_inherited": False,
                "source_freeze": source_freeze,
                "submission_accepted": True,
                "duplicate_submission_permitted": False,
                "final_connectivity_accessed": False,
            }
            receipt = {**receipt_core, "receipt_sha256": digest_object(receipt_core)}
            _write_or_verify(receipt_path, receipt, label="handoff receipt")
            return receipt
        finally:
            fcntl.flock(lease_fd, fcntl.LOCK_UN)
    finally:
        os.close(lease_fd)
        os.close(transaction_fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ep12-slurm-handoff")
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--kind", choices=tuple(HANDOFF_KINDS), required=True)
    submit.add_argument("--parent-job-id", required=True)
    submit.add_argument("--context-root", type=Path, required=True)
    submit.add_argument("--transaction-root", type=Path, default=DEFAULT_TRANSACTION_ROOT)
    submit.add_argument("--child-script", type=Path, required=True)
    submit.add_argument("--export-env", dest="exported", required=True)
    submit.add_argument("--child-argument", action="append", default=[])
    submit.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    current_job_id = os.environ.get("SLURM_JOB_ID")
    _require(current_job_id is not None, "handoff submission requires a Slurm parent job")
    _require(
        current_job_id == arguments.parent_job_id,
        "requested handoff parent differs from the current Slurm job",
    )
    receipt = submit_handoff(
        kind=arguments.kind,
        parent_job_id=arguments.parent_job_id,
        context_root=arguments.context_root,
        transaction_root=arguments.transaction_root,
        child_script=arguments.child_script,
        exported=arguments.exported,
        child_arguments=arguments.child_argument,
        run_root=arguments.run_root,
    )
    print(
        json.dumps(
            {
                "handoff_kind": receipt["handoff_kind"],
                "parent_job_id": receipt["parent_job_id"],
                "child_job_id": receipt["child_job_id"],
                "child_job_name": receipt["child_job_name"],
                "submission_accepted": True,
                "duplicate_submission_permitted": False,
                "final_connectivity_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
