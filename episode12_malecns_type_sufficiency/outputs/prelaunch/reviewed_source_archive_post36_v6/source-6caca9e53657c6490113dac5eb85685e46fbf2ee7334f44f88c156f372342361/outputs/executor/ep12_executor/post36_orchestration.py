"""Crash-safe held-submit and exact-release primitives for EP12 post-36 jobs.

Every scheduler call receives a strictly sanitized environment. Lost
``sbatch`` replies are recovered with one same-user, exact-name accounting
query, never a polling loop. A child is submitted under a user hold, its job-ID
receipt is made durable, and only a separate explicit transaction may release
that exact job.
"""

from __future__ import annotations

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
from .secure_transaction import open_secure_directory
from .slurm_client import sanitized_slurm_client_environment


STAGE_INTENT_SCHEMA = "ep12.post36_orchestration_stage_intent.v1"
STAGE_PROGRESS_SCHEMA = "ep12.post36_orchestration_stage_progress.v1"
STAGE_RECEIPT_SCHEMA = "ep12.post36_orchestration_stage_receipt.v1"
STAGE_RELEASE_INTENT_SCHEMA = "ep12.post36_stage_release_intent.v1"
STAGE_RELEASE_AUTHORIZATION_SCHEMA = "ep12.post36_stage_release_authorization.v1"
STAGE_RELEASE_PROGRESS_SCHEMA = "ep12.post36_stage_release_progress.v1"
STAGE_RELEASE_RECEIPT_SCHEMA = "ep12.post36_stage_release_receipt.v1"

OBSERVED_ATTESTATION_JOB_ID = "45332084"
OBSERVED_ATTESTATION_JOB_NAME = "ep12-attest36"
OBSERVED_FINALIZER_JOB_ID = "45323481"

ATTESTATION_IDENTITY_RELATIVE = Path(
    "outputs/prelaunch/observed_attestation_job_identity_45332084.json"
)
ATTESTATION_IDENTITY_FILE_SHA256 = (
    "ebc9269fde1d27fe9dff1430a33353c779c60a00317a6fc3caeeddca223c48eb"
)
ATTESTATION_IDENTITY_SHA256 = (
    "2a9426a35ec1c250e8768d82a0730591703f57bb8174a7e6dd3b0c84bbf010b2"
)
ATTESTATION_POST_RUN_REHASH_SHA256 = (
    "808092126520de07afaae55bcb55ef9c708b9a38a9f8161110b7df0e4db1aa68"
)

_SBATCH_RESULT_RE = re.compile(r"^(?P<job_id>[0-9]+)(?:;[A-Za-z0-9_.-]+)?$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class Post36OrchestrationError(RuntimeError):
    """The post-36 chain cannot be authenticated or submitted uniquely."""


Submitter = Callable[[Sequence[str]], str]
JobResolver = Callable[[str, str], Mapping[str, str] | None]
JobStateResolver = Callable[[str, str, str], Mapping[str, str]]
Releaser = Callable[[Sequence[str]], str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Post36OrchestrationError(message)


def _effective_user() -> str:
    try:
        value = pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError) as exc:
        raise Post36OrchestrationError(
            f"cannot resolve effective OS user: {exc}"
        ) from exc
    _require(bool(value), "effective OS user is empty")
    return value


def _lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _secure_path(
    path: Path, *, root: Path, label: str, allow_missing_leaf: bool = False
) -> Path:
    boundary = _lexical(root)
    candidate = _lexical(path)
    _require(boundary.is_absolute() and candidate.is_absolute(), f"{label} is not absolute")
    try:
        candidate.relative_to(boundary)
    except ValueError as exc:
        raise Post36OrchestrationError(f"{label} escapes EP12 root") from exc
    current = Path(candidate.anchor)
    for index, component in enumerate(candidate.parts[1:]):
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_leaf or index < len(candidate.parts[1:]):
                break
            raise
        except OSError as exc:
            raise Post36OrchestrationError(
                f"cannot inspect {label} path component {current}: {exc}"
            ) from exc
        _require(not stat.S_ISLNK(metadata.st_mode), f"{label} traverses a symlink: {current}")
    return candidate


def _read_bytes(path: Path, *, root: Path, label: str) -> bytes:
    secured = _secure_path(path, root=root, label=label)
    try:
        descriptor = os.open(secured, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise Post36OrchestrationError(f"cannot open {label} {secured}: {exc}") from exc
    try:
        _require(stat.S_ISREG(os.fstat(descriptor).st_mode), f"{label} is not regular")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                return b"".join(chunks)
            chunks.append(block)
    finally:
        os.close(descriptor)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, root: Path, label: str) -> str:
    return _sha256_bytes(_read_bytes(path, root=root, label=label))


def _decode_json(encoded: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Post36OrchestrationError(f"malformed {label}: {exc}") from exc
    _require(isinstance(value, Mapping), f"{label} is not a JSON object")
    return value


def _read_json(path: Path, *, root: Path, label: str, canonical: bool = False) -> dict[str, Any]:
    encoded = _read_bytes(path, root=root, label=label)
    value = dict(_decode_json(encoded, label=label))
    if canonical:
        _require(
            encoded == (canonical_json(value) + "\n").encode("utf-8"),
            f"{label} is not canonical JSON",
        )
    return value


def _encoded(value: Mapping[str, Any]) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, Any], *, root: Path) -> None:
    secured = _secure_path(
        path, root=root, label="post36 orchestration artifact", allow_missing_leaf=True
    )
    encoded = _encoded(value)
    temporary = secured.with_name(
        f".{secured.name}.{os.getpid()}-{_sha256_bytes(encoded)[:12]}.tmp"
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
            _require(written > 0, f"short write while creating {secured}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, secured)
    directory = os.open(secured.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_or_verify(
    path: Path, value: Mapping[str, Any], *, root: Path, label: str
) -> dict[str, Any]:
    try:
        existing = _read_json(path, root=root, label=label, canonical=True)
    except FileNotFoundError:
        _atomic_json(path, value, root=root)
        existing = _read_json(path, root=root, label=label, canonical=True)
    _require(existing == dict(value), f"{label} mutated")
    return existing


def _parse_job_id(output: str) -> str:
    match = _SBATCH_RESULT_RE.fullmatch(output.strip())
    _require(match is not None, f"unexpected sbatch --parsable output: {output!r}")
    job_id = str(match.group("job_id"))
    _require(int(job_id) > 0, "sbatch returned a nonpositive job ID")
    return job_id


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
        raise Post36OrchestrationError(
            f"Slurm client command failed: {getattr(exc, 'stderr', '')}"
        ) from exc
    return completed.stdout


def default_submitter(command: Sequence[str]) -> str:
    """Submit once with no ambient SBATCH options."""

    return _run_scheduler(command).strip()


def default_releaser(command: Sequence[str]) -> str:
    """Release one already-receipted held job with a sanitized environment."""

    return _run_scheduler(command).strip()


def default_job_state_resolver(
    job_id: str, job_name: str, submission_user: str
) -> Mapping[str, str]:
    """Resolve one exact same-user job once from squeue, then sacct if absent."""

    _require(submission_user == _effective_user(), "job-state user mismatch")
    live = _run_scheduler(
        [
            "/usr/bin/squeue", "--noheader", "--user", submission_user, "--jobs", job_id,
            "--format=%i|%j|%u|%T|%R",
        ]
    )
    rows: list[dict[str, str]] = []
    for line in live.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        _require(len(fields) == 5, "malformed held-job squeue row")
        raw_id, observed_name, observed_user, state, reason = fields
        if raw_id != job_id:
            continue
        rows.append(
            {
                "job_id": raw_id,
                "job_name": observed_name,
                "user": observed_user,
                "state": state,
                "reason": reason,
            }
        )
    _require(len(rows) <= 1, "held job is ambiguous in squeue")
    if not rows:
        accounting = _run_scheduler(
            [
                "/usr/bin/sacct", "-X", "--noheader", "--parsable2", "--user",
                submission_user, "--jobs", job_id, "--starttime", "2026-09-25",
                "--format=JobIDRaw,JobName%128,User%64,State,Reason",
            ]
        )
        for line in accounting.splitlines():
            if not line.strip():
                continue
            fields = line.strip().split("|")
            _require(len(fields) == 5, "malformed held-job sacct row")
            raw_id, observed_name, observed_user, state, reason = fields
            if raw_id != job_id:
                continue
            rows.append(
                {
                    "job_id": raw_id,
                    "job_name": observed_name,
                    "user": observed_user,
                    "state": state.rstrip("+").split()[0],
                    "reason": reason,
                }
            )
        _require(len(rows) == 1, "accepted job is absent or ambiguous in accounting")
    row = rows[0]
    _require(
        row["job_name"] == job_name and row["user"] == submission_user,
        "accepted job scheduler identity changed",
    )
    return row


def _base_job_id(raw: str) -> str | None:
    if raw.isdigit():
        return raw
    base, separator, _step = raw.partition(".")
    if separator == "." and base.isdigit():
        return base
    return None


def default_exact_name_resolver(
    job_name: str, submission_user: str
) -> Mapping[str, str] | None:
    """Use one exact-name accounting query; never poll or query twice."""

    _require(submission_user == _effective_user(), "Slurm recovery user mismatch")
    output = _run_scheduler(
        [
            "/usr/bin/sacct",
            "-X",
            "--noheader",
            "--parsable2",
            "--user",
            submission_user,
            "--name",
            job_name,
            "--starttime",
            "2026-09-25",
            "--format=JobIDRaw,JobName%128,User%64",
        ]
    )
    matches: set[str] = set()
    for raw in output.splitlines():
        if not raw.strip():
            continue
        fields = raw.strip().split("|")
        _require(len(fields) == 3, "malformed exact-name sacct recovery row")
        job_id, observed_name, observed_user = fields
        if observed_name != job_name:
            continue
        _require(observed_user == submission_user, "sacct recovery returned another user")
        base = _base_job_id(job_id)
        _require(base is not None, "sacct recovery returned an invalid job ID")
        matches.add(base)
    _require(len(matches) <= 1, f"ambiguous Slurm recovery for {job_name!r}")
    if not matches:
        return None
    return {
        "job_id": next(iter(matches)),
        "job_name": job_name,
        "user": submission_user,
    }


def _validated_recovery(
    value: Mapping[str, str] | None, *, job_name: str, submission_user: str
) -> str | None:
    if value is None:
        return None
    _require(
        set(value) == {"job_id", "job_name", "user"},
        "Slurm recovery schema changed",
    )
    _require(value.get("job_name") == job_name, "Slurm recovery name mismatch")
    _require(value.get("user") == submission_user, "Slurm recovery user mismatch")
    job_id = value.get("job_id")
    _require(
        isinstance(job_id, str) and job_id.isdigit() and int(job_id) > 0,
        "Slurm recovery job ID is invalid",
    )
    return job_id


def _superseded_attestation_evidence(
    *,
    episode_root: Path,
    submission_user: str,
) -> dict[str, Any]:
    """Authenticate the obsolete identity without trusting mutable live imports.

    Job 45332084 never runs and is cancelled under its durable hold.  Its
    recorded pre-run map remains useful provenance, but final reviewed source
    installation necessarily changes live bytes, so no current-live equality
    or transitive-freeze claim is made for the superseded job.
    """
    path = episode_root / ATTESTATION_IDENTITY_RELATIVE
    encoded = _read_bytes(path, root=episode_root, label="attestation job identity")
    _require(
        _sha256_bytes(encoded) == ATTESTATION_IDENTITY_FILE_SHA256,
        "attestation job identity file hash mismatch",
    )
    identity = dict(_decode_json(encoded, label="attestation job identity"))
    declared = identity.get("identity_sha256")
    core = {key: value for key, value in identity.items() if key != "identity_sha256"}
    _require(
        _sha256_bytes(_encoded(core))
        == declared
        == ATTESTATION_IDENTITY_SHA256,
        "attestation job identity semantic hash mismatch",
    )
    limitations = identity.get("identity_limitations")
    _require(
        identity.get("schema_version")
        == "ep12.pending_observed_attestation_job_identity.v1"
        and identity.get("slurm_job_id") == OBSERVED_ATTESTATION_JOB_ID
        and identity.get("slurm_job_name") == OBSERVED_ATTESTATION_JOB_NAME
        and identity.get("submission_user") == submission_user
        and identity.get("spooled_script_matches_current_on_disk_wrapper") is False
        and identity.get("spooled_script_uses_live_oak_python_imports") is True
        and isinstance(limitations, list)
        and "the_spooled_script_does_not_cryptographically_freeze_transitive_imports"
        in limitations
        and identity.get("real_connectivity_outcome_values_read_to_build_identity") is False
        and identity.get("final_connectivity_accessed") is False,
        "attestation identity facts or explicit limitation changed",
    )
    pre_run = identity.get("pre_run_live_import_source_sha256")
    _require(isinstance(pre_run, Mapping) and bool(pre_run), "attestation source map is empty")
    pinned_source_map: dict[str, str] = {}
    for relative, expected_sha in sorted(pre_run.items()):
        _require(
            isinstance(relative, str)
            and isinstance(expected_sha, str)
            and _HEX_RE.fullmatch(expected_sha) is not None,
            "attestation source map is malformed",
        )
        pinned_source_map[relative] = expected_sha
    pinned_source_map_digest = digest_object(pinned_source_map)
    _require(
        pinned_source_map_digest == ATTESTATION_POST_RUN_REHASH_SHA256,
        "attestation pinned pre-run source-map digest changed",
    )
    return {
        "slurm_job_id": OBSERVED_ATTESTATION_JOB_ID,
        "slurm_job_name": OBSERVED_ATTESTATION_JOB_NAME,
        "submission_user": submission_user,
        "authoritative": False,
        "disposition": "superseded_pending_mutable_source_job",
        "identity_path": str(path),
        "identity_file_sha256": ATTESTATION_IDENTITY_FILE_SHA256,
        "identity_sha256": ATTESTATION_IDENTITY_SHA256,
        "spooled_batch_script_sha256": identity["spooled_batch_script_sha256"],
        "spooled_script_matches_current_on_disk_wrapper": False,
        "spooled_script_uses_live_oak_python_imports": True,
        "pinned_pre_run_source_map_sha256": pinned_source_map_digest,
        "current_live_source_equality_claimed": False,
        "live_source_map_verified_at_replacement_preparation": False,
        "identity_limitations": list(limitations),
        "real_connectivity_outcome_values_read_to_build_identity": False,
        "final_connectivity_accessed": False,
    }


def _stage_command(stage_intent: Mapping[str, Any]) -> list[str]:
    _require(
        stage_intent.get("submission_barrier") == "user_hold_receipt_before_release_v1",
        "stage intent lacks the receipt-before-release barrier",
    )
    prefix = [
        "/usr/bin/sbatch",
        "--parsable",
        "--hold",
        f"--job-name={stage_intent['job_name']}",
    ]
    logical_dependency = stage_intent.get("dependency")
    scheduler_dependency = stage_intent.get(
        "scheduler_dependency", logical_dependency
    )
    if scheduler_dependency is None:
        predecessor = str(stage_intent.get("predecessor_job_id", ""))
        evidence_sha = stage_intent.get(
            "logical_predecessor_evidence_sha256"
        )
        _require(
            stage_intent.get("stage") == "replacement_attestation"
            and predecessor.isdigit()
            and logical_dependency == f"afterok:{predecessor}"
            and stage_intent.get("logical_dependency_satisfied") is True
            and stage_intent.get("kill_on_invalid_dependency") is False
            and isinstance(evidence_sha, str)
            and _HEX_RE.fullmatch(evidence_sha) is not None,
            "dependency-free held submission lacks authenticated logical predecessor",
        )
    else:
        _require(
            isinstance(scheduler_dependency, str)
            and scheduler_dependency == logical_dependency
            and stage_intent.get("kill_on_invalid_dependency") is True,
            "scheduler dependency declaration is inconsistent",
        )
        prefix.extend(
            [
                f"--dependency={scheduler_dependency}",
                "--kill-on-invalid-dep=yes",
            ]
        )
    arguments = stage_intent.get("arguments")
    _require(
        isinstance(arguments, list) and all(
            isinstance(value, str)
            and value
            and "\x00" not in value
            and "\n" not in value
            for value in arguments
        ),
        "stage positional arguments are invalid",
    )
    return [*prefix, "--export=NONE", str(stage_intent["wrapper"]), *arguments]


def _stage_progress(
    *, stage_intent: Mapping[str, Any], state: str, job_id: str | None
) -> dict[str, Any]:
    _require(state in {"planned", "submission_uncertain", "submitted"}, "invalid stage progress state")
    if state == "submitted":
        _require(
            isinstance(job_id, str) and job_id.isdigit() and int(job_id) > 0,
            "submitted stage lacks a positive job ID",
        )
    else:
        _require(job_id is None, f"{state} stage unexpectedly has a job ID")
    core = {
        "schema_version": STAGE_PROGRESS_SCHEMA,
        "stage_intent_sha256": stage_intent["stage_intent_sha256"],
        "stage": stage_intent["stage"],
        "job_name": stage_intent["job_name"],
        "submission_user": stage_intent["submission_user"],
        "state": state,
        "slurm_job_id": job_id,
        "final_connectivity_accessed": False,
    }
    return {**core, "stage_progress_sha256": digest_object(core)}


def _verify_stage_progress(
    value: Mapping[str, Any], *, stage_intent: Mapping[str, Any]
) -> tuple[str, str | None, dict[str, Any]]:
    state = value.get("state")
    raw_job = value.get("slurm_job_id")
    expected = _stage_progress(
        stage_intent=stage_intent,
        state=str(state),
        job_id=str(raw_job) if raw_job is not None else None,
    )
    _require(dict(value) == expected, "stage progress mutated")
    return str(state), expected["slurm_job_id"], expected


def _stage_receipt(
    *, stage_intent: Mapping[str, Any], progress: Mapping[str, Any], job_id: str
) -> dict[str, Any]:
    _require(progress.get("state") == "submitted", "stage receipt requires submitted progress")
    _require(progress.get("slurm_job_id") == job_id, "stage receipt/progress mismatch")
    dependency_evidence: dict[str, Any] = {}
    if "scheduler_dependency" in stage_intent:
        dependency_evidence = {
            "scheduler_dependency": stage_intent["scheduler_dependency"],
            "logical_dependency_satisfied": stage_intent[
                "logical_dependency_satisfied"
            ],
            "logical_predecessor_evidence_sha256": stage_intent[
                "logical_predecessor_evidence_sha256"
            ],
        }
    core = {
        "schema_version": STAGE_RECEIPT_SCHEMA,
        "stage": stage_intent["stage"],
        "submission_user": stage_intent["submission_user"],
        "slurm_job_id": job_id,
        "slurm_job_name": stage_intent["job_name"],
        "predecessor_job_id": stage_intent["predecessor_job_id"],
        "dependency": stage_intent["dependency"],
        **dependency_evidence,
        "kill_on_invalid_dependency": stage_intent[
            "kill_on_invalid_dependency"
        ],
        "submission_barrier": "user_hold_receipt_before_release_v1",
        "wrapper": stage_intent["wrapper"],
        "wrapper_sha256": stage_intent["wrapper_sha256"],
        **(
            {"arguments": list(stage_intent["arguments"])}
        ),
        "sbatch_command": _stage_command(stage_intent),
        "stage_intent_sha256": stage_intent["stage_intent_sha256"],
        "stage_progress_sha256": progress["stage_progress_sha256"],
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "stage_receipt_sha256": digest_object(core)}


def _validated_job_state(
    value: Mapping[str, str],
    *,
    job_id: str,
    job_name: str,
    submission_user: str,
) -> dict[str, str]:
    row = dict(value)
    _require(
        set(row) == {"job_id", "job_name", "user", "state", "reason"}
        and row["job_id"] == job_id
        and row["job_name"] == job_name
        and row["user"] == submission_user,
        "accepted job state identity changed",
    )
    if row["reason"] == "(JobHeldUser)":
        row["reason"] = "JobHeldUser"
    return row


def _release_intent(
    *,
    stage_receipt: Mapping[str, Any],
    stage_receipt_file_sha256: str,
    authorization_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    core = {
        "schema_version": STAGE_RELEASE_INTENT_SCHEMA,
        "stage": stage_receipt["stage"],
        "submission_user": stage_receipt["submission_user"],
        "slurm_job_id": stage_receipt["slurm_job_id"],
        "slurm_job_name": stage_receipt["slurm_job_name"],
        "stage_receipt_file_sha256": stage_receipt_file_sha256,
        "stage_receipt_sha256": stage_receipt["stage_receipt_sha256"],
        "release_command": [
            "/usr/bin/scontrol", "release", stage_receipt["slurm_job_id"]
        ],
        "authorization_context": (
            dict(authorization_context)
            if authorization_context is not None else None
        ),
        "receipt_was_durable_before_release": True,
        "final_connectivity_accessed": False,
    }
    return {**core, "release_intent_sha256": digest_object(core)}


def _release_progress(
    *, release_intent: Mapping[str, Any], state: str
) -> dict[str, Any]:
    _require(
        state in {"planned", "release_uncertain", "released"},
        "invalid stage release progress state",
    )
    core = {
        "schema_version": STAGE_RELEASE_PROGRESS_SCHEMA,
        "release_intent_sha256": release_intent["release_intent_sha256"],
        "stage": release_intent["stage"],
        "slurm_job_id": release_intent["slurm_job_id"],
        "state": state,
        "duplicate_sbatch_allowed": False,
        "final_connectivity_accessed": False,
    }
    return {**core, "release_progress_sha256": digest_object(core)}


def _release_authorization(
    *, release_intent: Mapping[str, Any]
) -> dict[str, Any]:
    """Immutable pre-call proof that the exact held job may start."""

    core = {
        "schema_version": STAGE_RELEASE_AUTHORIZATION_SCHEMA,
        "stage": release_intent["stage"],
        "submission_user": release_intent["submission_user"],
        "slurm_job_id": release_intent["slurm_job_id"],
        "slurm_job_name": release_intent["slurm_job_name"],
        "stage_receipt_file_sha256": release_intent[
            "stage_receipt_file_sha256"
        ],
        "stage_receipt_sha256": release_intent["stage_receipt_sha256"],
        "release_intent_sha256": release_intent["release_intent_sha256"],
        "release_command": list(release_intent["release_command"]),
        "authorization_persisted_before_first_scontrol_byte": True,
        "duplicate_release_after_uncertain_ack_allowed": False,
        "final_connectivity_accessed": False,
    }
    return {**core, "release_authorization_sha256": digest_object(core)}


def _release_receipt(
    *,
    release_intent: Mapping[str, Any],
    release_progress: Mapping[str, Any],
) -> dict[str, Any]:
    _require(release_progress["state"] == "released", "release receipt is premature")
    core = {
        "schema_version": STAGE_RELEASE_RECEIPT_SCHEMA,
        "stage": release_intent["stage"],
        "submission_user": release_intent["submission_user"],
        "slurm_job_id": release_intent["slurm_job_id"],
        "slurm_job_name": release_intent["slurm_job_name"],
        "stage_receipt_file_sha256": release_intent["stage_receipt_file_sha256"],
        "stage_receipt_sha256": release_intent["stage_receipt_sha256"],
        "release_intent_sha256": release_intent["release_intent_sha256"],
        "release_progress_sha256": release_progress["release_progress_sha256"],
        "receipt_was_durable_before_release": True,
        "release_was_exactly_scoped": True,
        "final_connectivity_accessed": False,
    }
    return {**core, "release_receipt_sha256": digest_object(core)}


def _release_stage(
    *,
    stage_dir: Path,
    root: Path,
    stage_receipt: Mapping[str, Any],
    job_state_resolver: JobStateResolver,
    releaser: Releaser,
    authorization_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stage = str(stage_receipt["stage"])
    intent_path = stage_dir / "release_intent.json"
    authorization_path = stage_dir / "release_authorization.json"
    progress_path = stage_dir / "release_progress.json"
    receipt_path = stage_dir / "release_receipt.json"
    release_intent = _write_or_verify(
        intent_path,
        _release_intent(
            stage_receipt=stage_receipt,
            stage_receipt_file_sha256=_sha256_file(
                stage_dir / "receipt.json", root=root,
                label=f"{stage} stage receipt",
            ),
            authorization_context=authorization_context,
        ),
        root=root,
        label=f"{stage} release intent",
    )
    release_authorization = _write_or_verify(
        authorization_path,
        _release_authorization(release_intent=release_intent),
        root=root,
        label=f"{stage} release authorization",
    )
    try:
        progress = _read_json(
            progress_path, root=root,
            label=f"{stage} release progress", canonical=True,
        )
        _require(
            progress
            == _release_progress(
                release_intent=release_intent, state=str(progress.get("state"))
            ),
            f"{stage} release progress mutated",
        )
    except FileNotFoundError:
        progress = _release_progress(release_intent=release_intent, state="planned")
        _atomic_json(progress_path, progress, root=root)
    try:
        receipt = _read_json(
            receipt_path, root=root,
            label=f"{stage} release receipt", canonical=True,
        )
    except FileNotFoundError:
        receipt = None
    if receipt is not None:
        _require(
            receipt == _release_receipt(
                release_intent=release_intent,
                release_progress=progress,
            ),
            f"{stage} release receipt mutated",
        )
        return receipt

    state = str(progress["state"])
    if state in {"planned", "release_uncertain"}:
        observed = _validated_job_state(
            job_state_resolver(
                str(stage_receipt["slurm_job_id"]),
                str(stage_receipt["slurm_job_name"]),
                str(stage_receipt["submission_user"]),
            ),
            job_id=str(stage_receipt["slurm_job_id"]),
            job_name=str(stage_receipt["slurm_job_name"]),
            submission_user=str(stage_receipt["submission_user"]),
        )
        held = observed["state"] == "PENDING" and observed["reason"] == "JobHeldUser"
        if state == "planned":
            _require(held, f"{stage} was not held before first release attempt")
            _require(
                release_authorization
                == _release_authorization(release_intent=release_intent),
                f"{stage} release authorization mutated",
            )
            progress = _release_progress(
                release_intent=release_intent, state="release_uncertain"
            )
            _atomic_json(progress_path, progress, root=root)
            releaser(list(release_intent["release_command"]))
        else:
            _require(
                observed["state"]
                in {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "COMPLETED"},
                f"{stage} release recovery found unsafe state {observed['state']!r}",
            )
            _require(
                not held,
                f"{stage} release acknowledgement remains uncertain; refusing duplicate scontrol",
            )
        progress = _release_progress(release_intent=release_intent, state="released")
        _atomic_json(progress_path, progress, root=root)
    receipt = _release_receipt(
        release_intent=release_intent,
        release_progress=progress,
    )
    return _write_or_verify(
        receipt_path, receipt, root=root, label=f"{stage} release receipt"
    )


def _stage_submission_evidence(
    *,
    stage_dir: Path,
    root: Path,
    stage_receipt: Mapping[str, Any],
    release_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the accepted-job receipt with its durable release proof bound."""

    release_path = stage_dir / "release_receipt.json"
    authorization_path = stage_dir / "release_authorization.json"
    release_intent = _read_json(
        stage_dir / "release_intent.json", root=root,
        label="stage release intent", canonical=True,
    )
    release_authorization = _read_json(
        authorization_path, root=root,
        label="stage release authorization", canonical=True,
    )
    _require(
        release_receipt.get("stage_receipt_sha256")
        == stage_receipt.get("stage_receipt_sha256")
        and release_receipt.get("slurm_job_id") == stage_receipt.get("slurm_job_id"),
        "stage release receipt does not bind the accepted job",
    )
    _require(
        release_authorization
        == _release_authorization(release_intent=release_intent),
        "stage release authorization mutated",
    )
    return {
        **dict(stage_receipt),
        "release_authorization_path": str(authorization_path),
        "release_authorization_file_sha256": _sha256_file(
            authorization_path, root=root, label="stage release authorization"
        ),
        "release_authorization_sha256": release_authorization[
            "release_authorization_sha256"
        ],
        "release_receipt_path": str(release_path),
        "release_receipt_file_sha256": _sha256_file(
            release_path, root=root, label="stage release receipt"
        ),
        "release_receipt_sha256": release_receipt["release_receipt_sha256"],
    }


def _submit_stage_held(
    *,
    transaction_root: Path,
    root: Path,
    stage_intent: Mapping[str, Any],
    submitter: Submitter,
    job_resolver: JobResolver,
    job_state_resolver: JobStateResolver = default_job_state_resolver,
) -> dict[str, Any]:
    stage = str(stage_intent["stage"])
    stage_dir, stage_fd = open_secure_directory(
        transaction_root / "stages" / stage, root=root, create=True
    )
    os.close(stage_fd)
    intent_path = stage_dir / "intent.json"
    progress_path = stage_dir / "progress.json"
    receipt_path = stage_dir / "receipt.json"
    canonical_intent = _write_or_verify(
        intent_path, stage_intent, root=root, label=f"{stage} stage intent"
    )
    try:
        progress_value = _read_json(
            progress_path, root=root, label=f"{stage} stage progress", canonical=True
        )
        state, job_id, progress = _verify_stage_progress(
            progress_value, stage_intent=canonical_intent
        )
    except FileNotFoundError:
        progress = _stage_progress(
            stage_intent=canonical_intent, state="planned", job_id=None
        )
        _atomic_json(progress_path, progress, root=root)
        state, job_id = "planned", None
    try:
        receipt = _read_json(
            receipt_path, root=root, label=f"{stage} stage receipt", canonical=True
        )
    except FileNotFoundError:
        receipt = None
    if receipt is not None:
        _require(
            state == "submitted" and job_id is not None,
            "stage receipt lacks submitted progress",
        )
        _require(
            receipt
            == _stage_receipt(
                stage_intent=canonical_intent, progress=progress, job_id=job_id
            ),
            f"{stage} stage receipt mutated",
        )
    else:
        if state in {"planned", "submission_uncertain"}:
            recovered = _validated_recovery(
                job_resolver(
                    str(canonical_intent["job_name"]),
                    str(canonical_intent["submission_user"]),
                ),
                job_name=str(canonical_intent["job_name"]),
                submission_user=str(canonical_intent["submission_user"]),
            )
            if recovered is None and state == "submission_uncertain":
                raise Post36OrchestrationError(
                    f"{stage} submission remains uncertain; refusing duplicate sbatch"
                )
            if recovered is None:
                progress = _stage_progress(
                    stage_intent=canonical_intent,
                    state="submission_uncertain",
                    job_id=None,
                )
                _atomic_json(progress_path, progress, root=root)
                job_id = _parse_job_id(submitter(_stage_command(canonical_intent)))
            else:
                job_id = recovered
            progress = _stage_progress(
                stage_intent=canonical_intent, state="submitted", job_id=job_id
            )
            _atomic_json(progress_path, progress, root=root)
        _require(
            isinstance(job_id, str) and job_id.isdigit(),
            f"{stage} lost accepted job ID",
        )
        accepted = _validated_job_state(
            job_state_resolver(
                job_id,
                str(canonical_intent["job_name"]),
                str(canonical_intent["submission_user"]),
            ),
            job_id=job_id,
            job_name=str(canonical_intent["job_name"]),
            submission_user=str(canonical_intent["submission_user"]),
        )
        _require(
            accepted["state"] == "PENDING"
            and accepted["reason"] == "JobHeldUser",
            f"{stage} was not held through durable receipt publication",
        )
        receipt = _write_or_verify(
            receipt_path,
            _stage_receipt(
                stage_intent=canonical_intent,
                progress=progress,
                job_id=job_id,
            ),
            root=root,
            label=f"{stage} stage receipt",
        )
    return receipt


def _release_submitted_stage(
    *,
    transaction_root: Path,
    root: Path,
    stage_intent: Mapping[str, Any],
    stage_receipt: Mapping[str, Any],
    job_state_resolver: JobStateResolver = default_job_state_resolver,
    releaser: Releaser = default_releaser,
    authorization_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    stage = str(stage_intent["stage"])
    stage_dir, stage_fd = open_secure_directory(
        transaction_root / "stages" / stage, root=root, create=False
    )
    os.close(stage_fd)
    canonical_intent = _read_json(
        stage_dir / "intent.json", root=root,
        label=f"{stage} stage intent", canonical=True,
    )
    _require(canonical_intent == dict(stage_intent), f"{stage} stage intent mutated")
    progress_value = _read_json(
        stage_dir / "progress.json", root=root,
        label=f"{stage} stage progress", canonical=True,
    )
    state, job_id, progress = _verify_stage_progress(
        progress_value, stage_intent=canonical_intent
    )
    _require(state == "submitted" and job_id is not None, f"{stage} was not accepted")
    canonical_receipt = _read_json(
        stage_dir / "receipt.json", root=root,
        label=f"{stage} stage receipt", canonical=True,
    )
    _require(
        canonical_receipt
        == _stage_receipt(
            stage_intent=canonical_intent, progress=progress, job_id=job_id
        )
        == dict(stage_receipt),
        f"{stage} stage receipt mutated",
    )
    release_receipt = _release_stage(
        stage_dir=stage_dir,
        root=root,
        stage_receipt=canonical_receipt,
        job_state_resolver=job_state_resolver,
        releaser=releaser,
        authorization_context=authorization_context,
    )
    return _stage_submission_evidence(
        stage_dir=stage_dir,
        root=root,
        stage_receipt=canonical_receipt,
        release_receipt=release_receipt,
    )


def _submit_stage(
    *,
    transaction_root: Path,
    root: Path,
    stage_intent: Mapping[str, Any],
    submitter: Submitter,
    job_resolver: JobResolver,
    job_state_resolver: JobStateResolver = default_job_state_resolver,
    releaser: Releaser = default_releaser,
) -> dict[str, Any]:
    held_receipt = _submit_stage_held(
        transaction_root=transaction_root,
        root=root,
        stage_intent=stage_intent,
        submitter=submitter,
        job_resolver=job_resolver,
        job_state_resolver=job_state_resolver,
    )
    return _release_submitted_stage(
        transaction_root=transaction_root,
        root=root,
        stage_intent=stage_intent,
        stage_receipt=held_receipt,
        job_state_resolver=job_state_resolver,
        releaser=releaser,
    )
