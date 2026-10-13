"""Exactly-once Slurm launcher for EP12 full synthetic qualification003.

The launcher freezes the complete generated-data code identity before asking
Slurm to accept work.  Its deterministic job name is also a recovery key: if
the client loses the ``sbatch`` response, a retry searches ``squeue`` and then
``sacct`` for that exact name instead of blindly submitting a duplicate.

This module never reads real connectivity and never authorizes final-data
access.  The only submitted payload is the synthetic qualification wrapper.
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
from .launch_chain_identity import (
    LaunchChainIdentityError,
    authenticate_test_launch,
)
from .scientific_contract_transition import verify_observed_attestation_authority
from .synthetic import EPISODE_ROOT
from .scientific_transition_binding import (
    ScientificTransitionBindingError,
    TRANSITION_MANIFEST_NAME,
    TRANSITION_ROOT_RELATIVE,
    verify_scientific_contract_transition,
)
from .source_freeze import SourceFreezeError, verify_source_freeze_manifest
from .yaml_subset import load_yaml_subset


INTENT_SCHEMA = "ep12.full_synthetic_qualification_launch_intent.v3"
PROGRESS_SCHEMA = "ep12.full_synthetic_qualification_launch_progress.v3"
LAUNCH_SCHEMA = "ep12.full_synthetic_qualification_launch.v3"
QUALIFICATION_VERSION = "qualification003"
OUTPUT_RELATIVE = Path("outputs/prelaunch/scientific_qualification_qualification003")
RECEIPT_RELATIVE = Path(
    "outputs/prelaunch/full_synthetic_qualification_launch_qualification003.json"
)
INTENT_RELATIVE = Path(
    "outputs/prelaunch/full_synthetic_qualification_launch_qualification003.intent.json"
)
PROGRESS_RELATIVE = Path(
    "outputs/prelaunch/full_synthetic_qualification_launch_qualification003.progress.json"
)
LOCK_RELATIVE = Path(
    "outputs/prelaunch/.full_synthetic_qualification_launch_qualification003.lock"
)

_SBATCH_RESULT_RE = re.compile(r"^(?P<job_id>[0-9]+)(?:;[A-Za-z0-9_.-]+)?$")
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

QUALIFICATION_HASH_TARGETS = {
    "qualification_spec_sha256": Path(
        "outputs/executor/SYNTHETIC_QUALIFICATION_SPEC_POST36.yaml"
    ),
    "scientific_contract_sha256": Path(
        "outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml"
    ),
    "scientific_qualification_code_sha256": Path(
        "outputs/executor/ep12_executor/scientific_qualification.py"
    ),
    "scientific_models_code_sha256": Path(
        "outputs/executor/ep12_executor/scientific_models.py"
    ),
    "scientific_synthetic_code_sha256": Path(
        "outputs/executor/ep12_executor/scientific_synthetic.py"
    ),
    "synthetic_code_sha256": Path(
        "outputs/executor/ep12_executor/synthetic.py"
    ),
    "policy_code_sha256": Path(
        "outputs/executor/ep12_executor/policy.py"
    ),
    "yaml_subset_code_sha256": Path(
        "outputs/executor/ep12_executor/yaml_subset.py"
    ),
    "cli_code_sha256": Path("outputs/executor/ep12_executor/cli.py"),
    "main_code_sha256": Path("outputs/executor/ep12_executor/__main__.py"),
    "scientific_transition_binding_code_sha256": Path(
        "outputs/executor/ep12_executor/scientific_transition_binding.py"
    ),
    "slurm_script_sha256": Path("outputs/executor/slurm/qualify_methods.sbatch"),
}
_HASH_TARGETS = QUALIFICATION_HASH_TARGETS

_TRANSITION_IDENTITY_FIELDS = {
    "transition_id",
    "transition_manifest_path",
    "transition_manifest_file_sha256",
    "transition_receipt_path",
    "transition_receipt_file_sha256",
    "transition_receipt_sha256",
    "transition_spec_sha256",
    "observed_prefix_contract_sha256",
    "active_contract_sha256",
    "observed_prefix_qualification_spec_sha256",
    "active_qualification_spec_sha256",
    "observed_prefix_scientific_models_sha256",
    "active_scientific_models_sha256",
    "observed_attestation_manifest_file_sha256",
    "observed_attestation_artifact_file_sha256",
    "observed_attestation_sha256",
    "observed_attestation_authority_sha256",
    "authoritative_attestation_job_id",
    "obsolete_attestation_job_id",
    "source_archive_receipt_file_sha256",
    "source_archive_receipt_sha256",
    "source_archive_manifest_file_sha256",
    "source_archive_manifest_sha256",
    "source_archive_tree_sha256",
    "controller_final_receipt_status",
    "observed_attestation_authority",
    "source_archive",
    "controller_final_receipt_required_before_irreversible_transition",
    "observed_report_contract_declarations_sha256",
    "observed_prefix_archive_file_sha256",
    "observed_prefix_limitation_manifest_file_sha256",
    "legacy_probe_file_sha256",
    "legacy_probe_artifact_sha256",
    "legacy_probe_scientific_payload_sha256",
    "post36_probe_file_sha256",
    "post36_probe_artifact_sha256",
    "post36_probe_scientific_payload_sha256",
    "allowed_diagnostics_delta_sha256",
    "operator_exposure_incident_file_sha256",
    "operator_exposure_incident_semantic_sha256",
    "observed_trial_count",
    "exact_attested_trial_report_set_sha256",
    "exact_36_attestation_set_matches_transition_evidence",
    "source_trial_reports_reverified",
    "operator_development_outcome_values_displayed",
    "programmatic_transition_extraction_outcome_fields_accessed",
    "values_used_for_transition_decisions",
    "final_connectivity_accessed",
}


class QualificationLaunchError(RuntimeError):
    """The frozen qualification launch cannot proceed without ambiguity."""


Submitter = Callable[[Sequence[str]], str]
JobResolver = Callable[[str, str], Mapping[str, str] | None]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationLaunchError(message)


def _effective_user() -> str:
    try:
        value = pwd.getpwuid(os.getuid()).pw_name
    except (KeyError, OSError) as exc:
        raise QualificationLaunchError(
            f"cannot resolve effective OS user: {exc}"
        ) from exc
    _require(isinstance(value, str) and value, "effective OS user is empty")
    return value


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def secure_qualification_path(
    path: Path,
    *,
    episode_root: Path,
    label: str,
) -> Path:
    """Return a lexical in-episode path after rejecting every symlink component."""

    root = _lexical_absolute(episode_root)
    candidate = _lexical_absolute(path)
    _require(episode_root.is_absolute(), f"{label} EP12 root must be absolute")
    _require(path.is_absolute(), f"{label} must be absolute")
    _require(
        candidate == root or root in candidate.parents,
        f"{label} escapes EP12 root: {candidate}",
    )
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise QualificationLaunchError(
                f"cannot inspect {label} path component {current}: {exc}"
            ) from exc
        _require(
            not stat.S_ISLNK(metadata.st_mode),
            f"{label} contains a symlink component: {current}",
        )
    return candidate


def _path_exists_nofollow(path: Path) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise QualificationLaunchError(f"cannot inspect launch path {path}: {exc}") from exc
    return True


def _read_bytes_nofollow(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationLaunchError(f"cannot open {label} {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_read_bytes_nofollow(path, label="launch source"))
    return digest.hexdigest()


def _atomic_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    episode_root: Path,
) -> None:
    path = secure_qualification_path(
        path, episode_root=episode_root, label="qualification launch artifact"
    )
    if _path_exists_nofollow(path):
        metadata = os.lstat(path)
        _require(
            stat.S_ISREG(metadata.st_mode),
            f"qualification launch artifact is not regular: {path}",
        )
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
                raise QualificationLaunchError(f"short write while creating {path}")
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


def _read_json(
    path: Path,
    *,
    episode_root: Path,
    require_canonical: bool = True,
) -> Mapping[str, Any]:
    path = secure_qualification_path(
        path, episode_root=episode_root, label="qualification launch artifact"
    )
    try:
        encoded = _read_bytes_nofollow(path, label="qualification launch artifact")
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationLaunchError(f"cannot read launch artifact {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise QualificationLaunchError(f"launch artifact is not a JSON object: {path}")
    if require_canonical:
        expected = (canonical_json(value) + "\n").encode("utf-8")
        _require(encoded == expected, f"launch artifact is not canonical JSON: {path}")
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
        raise QualificationLaunchError(f"qualification sbatch failed: {stderr}") from exc
    return completed.stdout.strip()


def _parse_job_id(output: str) -> str:
    match = _SBATCH_RESULT_RE.fullmatch(output.strip())
    if match is None:
        raise QualificationLaunchError(
            f"unexpected qualification sbatch --parsable output: {output!r}"
        )
    return match.group("job_id")


def _job_ids_for_exact_name(
    output: str, job_name: str, submission_user: str | None = None
) -> set[str]:
    matches: set[str] = set()
    for raw in output.splitlines():
        if not raw.strip():
            continue
        fields = raw.strip().split("|")
        expected_count = 3 if submission_user is not None else 2
        _require(len(fields) == expected_count, "malformed qualification recovery row")
        job_id, observed_name = fields[:2]
        if observed_name != job_name:
            continue
        if submission_user is not None:
            _require(
                fields[2] == submission_user,
                "qualification recovery returned a mismatched user",
            )
        if job_id.isdigit():
            matches.add(job_id)
        else:
            base, separator, _step = job_id.partition(".")
            _require(
                separator == "." and base.isdigit(),
                "qualification recovery returned an invalid job ID",
            )
    return matches


def _unique_resolved_job_id(
    output: str, job_name: str, submission_user: str | None = None
) -> str | None:
    matches = _job_ids_for_exact_name(output, job_name, submission_user)
    if len(matches) > 1:
        raise QualificationLaunchError(
            f"multiple Slurm jobs carry qualification recovery name {job_name!r}"
        )
    return next(iter(matches)) if matches else None


def _default_job_resolver(
    job_name: str, submission_user: str
) -> Mapping[str, str] | None:
    """Resolve one exact-name job with one accounting query."""

    _require(
        submission_user == _effective_user(),
        "qualification recovery user does not match effective OS user",
    )
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
        "--format=JobIDRaw,JobName%64,User%64",
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
        stderr = getattr(exc, "stderr", "")
        raise QualificationLaunchError(
            f"qualification recovery query failed for {job_name!r}: {stderr}"
        ) from exc
    recovered = _unique_resolved_job_id(
        completed.stdout, job_name, submission_user
    )
    if recovered is None:
        return None
    return {
        "job_id": recovered,
        "job_name": job_name,
        "user": submission_user,
    }


def _validated_recovery(
    value: Mapping[str, str] | None,
    *,
    job_name: str,
    submission_user: str,
) -> str | None:
    if value is None:
        return None
    _require(
        isinstance(value, Mapping)
        and set(value) == {"job_id", "job_name", "user"},
        "qualification recovery schema changed",
    )
    _require(
        value.get("job_name") == job_name,
        "qualification recovery returned a mismatched job name",
    )
    _require(
        value.get("user") == submission_user,
        "qualification recovery returned a mismatched user",
    )
    job_id = value.get("job_id")
    _require(
        isinstance(job_id, str) and job_id.isdigit() and int(job_id) > 0,
        "qualification recovery returned an invalid job ID",
    )
    return job_id


def _paths(episode_root: Path) -> dict[str, Path]:
    _require(episode_root.is_absolute(), "EP12 root must be absolute")
    root = _lexical_absolute(episode_root)
    root = secure_qualification_path(root, episode_root=root, label="EP12 root")
    _require(root.is_dir(), f"EP12 root is not a directory: {root}")
    paths = {
        "root": root,
        "output": root / OUTPUT_RELATIVE,
        "receipt": root / RECEIPT_RELATIVE,
        "intent": root / INTENT_RELATIVE,
        "progress": root / PROGRESS_RELATIVE,
        "lock": root / LOCK_RELATIVE,
    }
    for name, path in paths.items():
        if name != "root":
            secure_qualification_path(
                path, episode_root=root, label=f"qualification {name}"
            )
    return paths


def _source_identity(episode_root: Path) -> dict[str, str]:
    identity: dict[str, str] = {}
    for field, relative in _HASH_TARGETS.items():
        path = secure_qualification_path(
            episode_root / relative,
            episode_root=episode_root,
            label=f"qualification source {field}",
        )
        try:
            metadata = os.lstat(path)
        except OSError as exc:
            raise QualificationLaunchError(f"missing regular source {path}: {exc}") from exc
        _require(stat.S_ISREG(metadata.st_mode), f"missing regular source: {path}")
        identity[field] = _sha256_file(path)
    _require(
        all(_HEX_RE.fullmatch(value) for value in identity.values()),
        "qualification source identity contains an invalid SHA-256",
    )
    return identity


def _bind_source_freeze_records(
    *,
    episode_root: Path,
    test_launch_identity: Mapping[str, Any],
    source_identity: Mapping[str, str],
) -> None:
    """Require every explicitly qualified source to occur in the same freeze."""

    manifest_path = Path(str(test_launch_identity.get("source_freeze_manifest", "")))
    expected_file_sha = str(
        test_launch_identity.get("source_freeze_manifest_sha256", "")
    )
    try:
        manifest = verify_source_freeze_manifest(
            manifest_path,
            episode_root=episode_root,
            expected_file_sha256=expected_file_sha,
            require_current_surface=True,
        )
    except (OSError, SourceFreezeError) as exc:
        raise QualificationLaunchError(str(exc)) from exc
    records = manifest.get("files")
    _require(isinstance(records, list), "source-freeze manifest lacks file records")
    by_path: dict[str, Mapping[str, Any]] = {}
    for raw in records:
        _require(isinstance(raw, Mapping), "source-freeze file record is malformed")
        relative = raw.get("path")
        _require(
            isinstance(relative, str) and relative not in by_path,
            "source-freeze file path is missing or duplicated",
        )
        by_path[relative] = raw
    for field, relative in _HASH_TARGETS.items():
        record = by_path.get(relative.as_posix())
        _require(
            isinstance(record, Mapping)
            and record.get("sha256") == source_identity[field],
            f"source-freeze manifest does not bind qualification source {relative}",
        )


def _validated_transition_authority(
    evidence: Mapping[str, Any], *, episode_root: Path
) -> dict[str, Any]:
    """Require the qualification identity to preserve dynamic-R authority."""

    raw_authority = evidence.get("observed_attestation_authority")
    if not isinstance(raw_authority, Mapping):
        raise QualificationLaunchError(
            "scientific transition lacks dynamic observed-attestation authority"
        )
    try:
        authority = verify_observed_attestation_authority(raw_authority)
    except ValueError as exc:
        raise QualificationLaunchError(str(exc)) from exc
    archive = authority.get("source_archive")
    final = authority.get("controller_final_receipt")
    _require(isinstance(archive, Mapping), "transition authority lacks source archive")
    _require(
        isinstance(final, Mapping),
        "transition authority lacks controller final-receipt binding",
    )
    expected_terminal = (
        episode_root
        / "outputs/prelaunch/replacement_attestation_launch/terminal_identity.json"
    )
    expected_start = (
        episode_root
        / "outputs/prelaunch/post36_orchestration/start_authorization.json"
    )
    expected_cancellation = (
        episode_root
        / "outputs/prelaunch/replacement_attestation_launch/"
        "obsolete_cancellation/receipt.json"
    )
    expected_final = episode_root / "outputs/prelaunch/post36_orchestration/receipt.json"
    _require(
        authority.get("authoritative_job_id")
        == evidence.get("authoritative_attestation_job_id")
        and authority.get("authoritative_job_id") != "45332084"
        and authority.get("obsolete_job_id")
        == evidence.get("obsolete_attestation_job_id")
        == "45332084"
        and authority.get("obsolete_result_authoritative") is False
        and authority.get("authoritative_result") is True
        and authority.get("authority_sha256")
        == evidence.get("observed_attestation_authority_sha256")
        and authority.get("replacement_terminal_identity_path")
        == str(expected_terminal)
        and authority.get("post36_start_authorization_path") == str(expected_start)
        and authority.get("obsolete_cancellation_receipt_path")
        == str(expected_cancellation)
        and final.get("path") == str(expected_final)
        and final.get("status") == evidence.get("controller_final_receipt_status")
        and final.get("required_before_irreversible_transition") is True
        and evidence.get(
            "controller_final_receipt_required_before_irreversible_transition"
        )
        is True
        and dict(archive) == evidence.get("source_archive")
        and archive.get("archive_receipt_file_sha256")
        == evidence.get("source_archive_receipt_file_sha256")
        and archive.get("archive_receipt_sha256")
        == evidence.get("source_archive_receipt_sha256")
        and archive.get("manifest_file_sha256")
        == evidence.get("source_archive_manifest_file_sha256")
        and archive.get("manifest_sha256")
        == evidence.get("source_archive_manifest_sha256")
        and archive.get("tree_sha256")
        == evidence.get("source_archive_tree_sha256"),
        "qualification transition identity misstates dynamic-R authority",
    )
    return authority


def _transition_identity(
    *, episode_root: Path, executor_root: Path
) -> dict[str, Any]:
    manifest = episode_root / TRANSITION_ROOT_RELATIVE / TRANSITION_MANIFEST_NAME
    try:
        evidence = verify_scientific_contract_transition(
            episode_root=episode_root,
            executor_root=executor_root,
            manifest_path=manifest,
        )
    except (OSError, ScientificTransitionBindingError) as exc:
        raise QualificationLaunchError(str(exc)) from exc
    _require(
        set(evidence) == _TRANSITION_IDENTITY_FIELDS,
        "scientific-transition verifier evidence schema changed",
    )
    _require(
        evidence.get("exact_36_attestation_set_matches_transition_evidence")
        is True
        and evidence.get("source_trial_reports_reverified") is True
        and evidence.get("operator_development_outcome_values_displayed") is True
        and evidence.get(
            "programmatic_transition_extraction_outcome_fields_accessed"
        )
        is False
        and evidence.get("values_used_for_transition_decisions") is False
        and evidence.get("final_connectivity_accessed") is False,
        "scientific transition crosses the outcome firewall",
    )
    _validated_transition_authority(evidence, episode_root=episode_root)
    return dict(evidence)


def _qualification_shape(episode_root: Path) -> tuple[int, int]:
    spec_path = secure_qualification_path(
        episode_root / _HASH_TARGETS["qualification_spec_sha256"],
        episode_root=episode_root,
        label="qualification specification",
    )
    try:
        spec = load_yaml_subset(spec_path)
        run_tiers = spec["run_tiers"]
        repetitions = int(run_tiers["qualification_replicates_per_core_error_cell"])
        whole_types = int(run_tiers["whole_types_per_replicate"])
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise QualificationLaunchError(
            f"cannot derive frozen qualification shape from {spec_path}: {exc}"
        ) from exc
    _require(repetitions == 72, "qualification003 requires the frozen 72 replicates")
    _require(whole_types == 8, "qualification003 requires the frozen 8 whole types")
    return repetitions, whole_types


def _build_intent(*, episode_root: Path, test_job_id: str) -> dict[str, Any]:
    _require(test_job_id.isdigit(), "test Slurm job ID must contain only digits")
    _require(int(test_job_id) > 0, "test Slurm job ID must be positive")
    paths = _paths(episode_root)
    try:
        test_launch_identity = authenticate_test_launch(
            episode_root=paths["root"], expected_job_id=test_job_id
        )
    except LaunchChainIdentityError as exc:
        raise QualificationLaunchError(str(exc)) from exc
    submission_user = _effective_user()
    _require(
        test_launch_identity.get("submission_user") == submission_user,
        "test launch and qualification have different effective users",
    )
    source_identity = _source_identity(paths["root"])
    _bind_source_freeze_records(
        episode_root=paths["root"],
        test_launch_identity=test_launch_identity,
        source_identity=source_identity,
    )
    transition_identity = _transition_identity(
        episode_root=paths["root"],
        executor_root=paths["root"] / "outputs" / "executor",
    )
    _require(
        transition_identity["active_contract_sha256"]
        == source_identity["scientific_contract_sha256"]
        and transition_identity["active_qualification_spec_sha256"]
        == source_identity["qualification_spec_sha256"]
        and transition_identity["active_scientific_models_sha256"]
        == source_identity["scientific_models_code_sha256"],
        "qualification source identity differs from the active transition",
    )
    repetitions, whole_types = _qualification_shape(paths["root"])
    script = paths["root"] / _HASH_TARGETS["slurm_script_sha256"]
    core = {
        "schema_version": INTENT_SCHEMA,
        "qualification_version": QUALIFICATION_VERSION,
        "submission_user": submission_user,
        "test_job_id": test_job_id,
        "test_launch_identity": test_launch_identity,
        "dependency": f"afterok:{test_job_id}",
        "kill_on_invalid_dependency": True,
        "tier": "qualification",
        "replicates_per_core_error_cell": repetitions,
        "whole_types_per_replicate": whole_types,
        "output_dir": str(paths["output"]),
        "slurm_script": str(script),
        "source_identity": source_identity,
        "scientific_contract_transition": transition_identity,
        "real_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    token = digest_object(core)[:32]
    with_name = {**core, "job_name": f"ep12-qual003-{token}"}
    return {**with_name, "intent_sha256": digest_object(with_name)}


def _verify_intent(value: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    _require(set(value) == set(expected), "qualification launch intent schema changed")
    claimed = value.get("intent_sha256")
    core = {key: item for key, item in value.items() if key != "intent_sha256"}
    _require(
        value.get("schema_version") == INTENT_SCHEMA
        and isinstance(claimed, str)
        and claimed == digest_object(core),
        "qualification launch intent digest mismatch",
    )
    _require(dict(value) == dict(expected), "qualification launch intent mutated")
    return dict(value)


def _progress_value(
    *, intent: Mapping[str, Any], state: str, slurm_job_id: str | None
) -> dict[str, Any]:
    _require(
        state in {"planned", "submission_uncertain", "submitted"},
        "invalid qualification progress state",
    )
    if state in {"planned", "submission_uncertain"}:
        _require(
            slurm_job_id is None,
            f"{state} qualification unexpectedly has a job ID",
        )
    else:
        _require(
            isinstance(slurm_job_id, str) and slurm_job_id.isdigit(),
            "submitted qualification lacks a numeric job ID",
        )
    core = {
        "schema_version": PROGRESS_SCHEMA,
        "qualification_version": QUALIFICATION_VERSION,
        "intent_sha256": intent["intent_sha256"],
        "job_name": intent["job_name"],
        "submission_user": intent["submission_user"],
        "state": state,
        "slurm_job_id": slurm_job_id,
        "real_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "progress_sha256": digest_object(core)}


def _verify_progress(
    value: Mapping[str, Any], *, intent: Mapping[str, Any]
) -> tuple[str, str | None]:
    expected_keys = set(_progress_value(intent=intent, state="planned", slurm_job_id=None))
    _require(set(value) == expected_keys, "qualification launch progress schema changed")
    claimed = value.get("progress_sha256")
    core = {key: item for key, item in value.items() if key != "progress_sha256"}
    _require(
        value.get("schema_version") == PROGRESS_SCHEMA
        and isinstance(claimed, str)
        and claimed == digest_object(core),
        "qualification launch progress digest mismatch",
    )
    state = str(value.get("state"))
    job_id = value.get("slurm_job_id")
    rebuilt = _progress_value(
        intent=intent,
        state=state,
        slurm_job_id=str(job_id) if job_id is not None else None,
    )
    _require(dict(value) == rebuilt, "qualification launch progress mutated")
    return state, rebuilt["slurm_job_id"]


def _launch_receipt(*, intent: Mapping[str, Any], job_id: str) -> dict[str, Any]:
    source_identity = intent["source_identity"]
    _require(isinstance(source_identity, Mapping), "qualification intent lacks source hashes")
    core = {
        "schema_version": LAUNCH_SCHEMA,
        "qualification_version": QUALIFICATION_VERSION,
        "tier": "qualification",
        "replicates_per_core_error_cell": intent["replicates_per_core_error_cell"],
        "whole_types_per_replicate": intent["whole_types_per_replicate"],
        "output_dir": intent["output_dir"],
        "slurm_script": intent["slurm_script"],
        "slurm_job_id": job_id,
        "slurm_job_name": intent["job_name"],
        "submission_user": intent["submission_user"],
        "test_job_id": intent["test_job_id"],
        "test_launch_identity": dict(intent["test_launch_identity"]),
        "scientific_contract_transition": dict(
            intent["scientific_contract_transition"]
        ),
        "dependency": intent["dependency"],
        "kill_on_invalid_dependency": True,
        "intent_sha256": intent["intent_sha256"],
        "real_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
        **dict(source_identity),
    }
    return {**core, "launch_sha256": digest_object(core)}


def verify_qualification_launch_receipt(
    value: Mapping[str, Any],
    *,
    episode_root: Path,
) -> dict[str, Any]:
    """Verify the exact qualification003 launch schema and its intent identity."""

    paths = _paths(episode_root)
    expected_keys = {
        "schema_version",
        "qualification_version",
        "tier",
        "replicates_per_core_error_cell",
        "whole_types_per_replicate",
        "output_dir",
        "slurm_script",
        "slurm_job_id",
        "slurm_job_name",
        "submission_user",
        "test_job_id",
        "test_launch_identity",
        "scientific_contract_transition",
        "dependency",
        "kill_on_invalid_dependency",
        "intent_sha256",
        "real_connectivity_accessed",
        "final_connectivity_access_authorized",
        *_HASH_TARGETS.keys(),
        "launch_sha256",
    }
    _require(set(value) == expected_keys, "qualification launch receipt schema changed")
    _require(value.get("schema_version") == LAUNCH_SCHEMA, "invalid launch schema")
    _require(
        value.get("qualification_version") == QUALIFICATION_VERSION,
        "invalid qualification launch version",
    )
    _require(value.get("tier") == "qualification", "invalid qualification tier")
    _require(
        type(value.get("replicates_per_core_error_cell")) is int
        and value.get("replicates_per_core_error_cell") == 72,
        "invalid qualification replicate count",
    )
    _require(
        type(value.get("whole_types_per_replicate")) is int
        and value.get("whole_types_per_replicate") == 8,
        "invalid qualification whole-type count",
    )
    _require(
        value.get("output_dir") == str(paths["output"]),
        "qualification launch output path changed",
    )
    expected_script = paths["root"] / _HASH_TARGETS["slurm_script_sha256"]
    _require(
        value.get("slurm_script") == str(expected_script),
        "qualification launch Slurm script path changed",
    )
    test_job_id = value.get("test_job_id")
    slurm_job_id = value.get("slurm_job_id")
    _require(
        isinstance(test_job_id, str)
        and test_job_id.isdigit()
        and int(test_job_id) > 0,
        "qualification launch test job ID is invalid",
    )
    _require(
        isinstance(slurm_job_id, str)
        and slurm_job_id.isdigit()
        and int(slurm_job_id) > 0,
        "qualification launch job ID is invalid",
    )
    submission_user = value.get("submission_user")
    _require(
        submission_user == _effective_user(),
        "qualification launch user does not match effective OS user",
    )
    try:
        test_launch_identity = authenticate_test_launch(
            episode_root=paths["root"], expected_job_id=test_job_id
        )
    except LaunchChainIdentityError as exc:
        raise QualificationLaunchError(str(exc)) from exc
    _require(
        value.get("test_launch_identity") == test_launch_identity,
        "qualification launch does not bind the canonical test transaction",
    )
    transition_identity = _transition_identity(
        episode_root=paths["root"],
        executor_root=paths["root"] / "outputs" / "executor",
    )
    _require(
        value.get("scientific_contract_transition") == transition_identity,
        "qualification launch does not bind the canonical scientific transition",
    )
    _require(
        value.get("dependency") == f"afterok:{test_job_id}",
        "qualification launch dependency changed",
    )
    _require(
        value.get("kill_on_invalid_dependency") is True,
        "qualification launch no longer kills invalid dependencies",
    )
    _require(
        value.get("real_connectivity_accessed") is False
        and value.get("final_connectivity_access_authorized") is False,
        "qualification launch crosses a real-connectivity boundary",
    )
    source_identity = {field: value.get(field) for field in _HASH_TARGETS}
    _require(
        all(isinstance(item, str) and _HEX_RE.fullmatch(item) for item in source_identity.values()),
        "qualification launch source identity is invalid",
    )
    _require(
        source_identity == _source_identity(paths["root"]),
        "qualification launch source bytes changed",
    )
    intent_core = {
        "schema_version": INTENT_SCHEMA,
        "qualification_version": QUALIFICATION_VERSION,
        "submission_user": submission_user,
        "test_job_id": test_job_id,
        "test_launch_identity": test_launch_identity,
        "dependency": f"afterok:{test_job_id}",
        "kill_on_invalid_dependency": True,
        "tier": "qualification",
        "replicates_per_core_error_cell": 72,
        "whole_types_per_replicate": 8,
        "output_dir": str(paths["output"]),
        "slurm_script": str(expected_script),
        "source_identity": source_identity,
        "scientific_contract_transition": transition_identity,
        "real_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    expected_name = f"ep12-qual003-{digest_object(intent_core)[:32]}"
    _require(
        value.get("slurm_job_name") == expected_name,
        "qualification launch deterministic job name changed",
    )
    intent_with_name = {**intent_core, "job_name": expected_name}
    _require(
        value.get("intent_sha256") == digest_object(intent_with_name),
        "qualification launch intent digest mismatch",
    )
    launch_core = {
        key: item for key, item in value.items() if key != "launch_sha256"
    }
    _require(
        isinstance(value.get("launch_sha256"), str)
        and value.get("launch_sha256") == digest_object(launch_core),
        "qualification launch receipt digest mismatch",
    )
    return dict(value)


def read_qualification_launch_receipt(
    path: Path,
    *,
    episode_root: Path,
) -> dict[str, Any]:
    """Read canonical bytes from the one versioned receipt path and verify them."""

    paths = _paths(episode_root)
    secured = secure_qualification_path(
        path, episode_root=paths["root"], label="qualification launch receipt"
    )
    _require(
        secured == paths["receipt"],
        "qualification launch receipt path is not the versioned qualification003 path",
    )
    receipt = verify_qualification_launch_receipt(
        _read_json(secured, episode_root=paths["root"], require_canonical=True),
        episode_root=paths["root"],
    )
    expected_intent = _build_intent(
        episode_root=paths["root"], test_job_id=str(receipt["test_job_id"])
    )
    intent = _verify_intent(
        _read_json(paths["intent"], episode_root=paths["root"]),
        expected_intent,
    )
    state, job_id = _verify_progress(
        _read_json(paths["progress"], episode_root=paths["root"]),
        intent=intent,
    )
    _require(
        state == "submitted" and job_id == receipt["slurm_job_id"],
        "qualification receipt lacks matching canonical submitted progress",
    )
    _require(
        receipt == _launch_receipt(intent=intent, job_id=job_id),
        "qualification receipt differs from its canonical transaction",
    )
    return receipt


def _verify_receipt(
    value: Mapping[str, Any], *, intent: Mapping[str, Any], job_id: str | None = None
) -> dict[str, Any]:
    observed_job_id = str(value.get("slurm_job_id", ""))
    _require(observed_job_id.isdigit(), "qualification launch receipt job ID is invalid")
    if job_id is not None:
        _require(observed_job_id == job_id, "qualification receipt job ID changed")
    expected = _launch_receipt(intent=intent, job_id=observed_job_id)
    verified = verify_qualification_launch_receipt(
        value, episode_root=Path(str(intent["output_dir"])).parents[2]
    )
    _require(verified == expected, "qualification launch receipt mutated")
    return expected


def _submission_command(intent: Mapping[str, Any]) -> list[str]:
    test_identity = intent.get("test_launch_identity")
    _require(isinstance(test_identity, Mapping), "qualification intent lacks test identity")
    transition = intent.get("scientific_contract_transition")
    _require(
        isinstance(transition, Mapping),
        "qualification intent lacks scientific-transition identity",
    )
    export = (
        "EP12_TIER=qualification,"
        f"EP12_OUTPUT_DIR={intent['output_dir']},"
        f"EP12_REPETITIONS={intent['replicates_per_core_error_cell']},"
        f"EP12_WHOLE_TYPES={intent['whole_types_per_replicate']},"
        f"EP12_SOURCE_FREEZE_MANIFEST={test_identity['source_freeze_manifest']},"
        "EP12_SOURCE_FREEZE_MANIFEST_SHA256="
        f"{test_identity['source_freeze_manifest_sha256']},"
        "EP12_TRANSITION_MANIFEST="
        f"{transition['transition_manifest_path']},"
        "EP12_TRANSITION_MANIFEST_SHA256="
        f"{transition['transition_manifest_file_sha256']},"
        "EP12_TRANSITION_RECEIPT_SHA256="
        f"{transition['transition_receipt_sha256']},"
        "EP12_TRANSITION_RECEIPT_FILE_SHA256="
        f"{transition['transition_receipt_file_sha256']},"
        "EP12_OBSERVED_PREFIX_CONTRACT_SHA256="
        f"{transition['observed_prefix_contract_sha256']},"
        "EP12_ACTIVE_CONTRACT_SHA256="
        f"{transition['active_contract_sha256']},"
        "EP12_ACTIVE_QUALIFICATION_SPEC_SHA256="
        f"{transition['active_qualification_spec_sha256']},"
        "EP12_ACTIVE_SCIENTIFIC_MODELS_SHA256="
        f"{transition['active_scientific_models_sha256']}"
    )
    return [
        "sbatch",
        "--parsable",
        f"--job-name={intent['job_name']}",
        f"--dependency={intent['dependency']}",
        "--kill-on-invalid-dep=yes",
        f"--export={export}",
        str(intent["slurm_script"]),
    ]


def launch_qualification003(
    *,
    test_job_id: str | int,
    episode_root: Path = EPISODE_ROOT,
    submitter: Submitter = _default_submitter,
    job_resolver: JobResolver = _default_job_resolver,
) -> dict[str, Any]:
    """Freeze and submit qualification003 once, recovering a lost reply by name."""

    paths = _paths(episode_root)
    prelaunch = paths["receipt"].parent
    prelaunch.mkdir(parents=True, exist_ok=True)
    secure_qualification_path(
        prelaunch, episode_root=paths["root"], label="qualification prelaunch directory"
    )
    expected_intent = _build_intent(
        episode_root=paths["root"], test_job_id=str(test_job_id)
    )
    lock_path = secure_qualification_path(
        paths["lock"], episode_root=paths["root"], label="qualification launch lock"
    )
    try:
        lock_descriptor = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o640,
        )
    except OSError as exc:
        raise QualificationLaunchError(f"cannot open qualification launch lock: {exc}") from exc
    lock_metadata = os.fstat(lock_descriptor)
    if not stat.S_ISREG(lock_metadata.st_mode):
        os.close(lock_descriptor)
        raise QualificationLaunchError("qualification launch lock is not a regular file")
    with os.fdopen(lock_descriptor, "a+b") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        try:
            if _path_exists_nofollow(paths["intent"]):
                intent = _verify_intent(
                    _read_json(paths["intent"], episode_root=paths["root"]),
                    expected_intent,
                )
            else:
                _require(
                    not _path_exists_nofollow(paths["output"]),
                    "qualification003 output directory is not fresh",
                )
                _atomic_json(
                    paths["intent"], expected_intent, episode_root=paths["root"]
                )
                intent = _verify_intent(
                    _read_json(paths["intent"], episode_root=paths["root"]),
                    expected_intent,
                )

            if _path_exists_nofollow(paths["receipt"]):
                _require(
                    _path_exists_nofollow(paths["progress"]),
                    "qualification launch receipt lacks durable progress",
                )
                state, job_id = _verify_progress(
                    _read_json(paths["progress"], episode_root=paths["root"]),
                    intent=intent,
                )
                _require(
                    state == "submitted" and job_id is not None,
                    "qualification launch receipt lacks a submitted job ID",
                )
                receipt = read_qualification_launch_receipt(
                    paths["receipt"], episode_root=paths["root"]
                )
                return _verify_receipt(receipt, intent=intent, job_id=job_id)

            if _path_exists_nofollow(paths["progress"]):
                state, job_id = _verify_progress(
                    _read_json(paths["progress"], episode_root=paths["root"]),
                    intent=intent,
                )
            else:
                state, job_id = "planned", None
                _atomic_json(
                    paths["progress"],
                    _progress_value(intent=intent, state=state, slurm_job_id=None),
                    episode_root=paths["root"],
                )

            if state in {"planned", "submission_uncertain"}:
                job_id = _validated_recovery(
                    job_resolver(
                        str(intent["job_name"]), str(intent["submission_user"])
                    ),
                    job_name=str(intent["job_name"]),
                    submission_user=str(intent["submission_user"]),
                )
                if job_id is None and state == "submission_uncertain":
                    raise QualificationLaunchError(
                        "qualification submission remains uncertain: exact Slurm "
                        "job name is not yet visible; refusing duplicate sbatch"
                    )
                if job_id is None:
                    command = _submission_command(intent)
                    _atomic_json(
                        paths["progress"],
                        _progress_value(
                            intent=intent,
                            state="submission_uncertain",
                            slurm_job_id=None,
                        ),
                        episode_root=paths["root"],
                    )
                    job_id = _parse_job_id(submitter(command))
                _atomic_json(
                    paths["progress"],
                    _progress_value(
                        intent=intent, state="submitted", slurm_job_id=job_id
                    ),
                    episode_root=paths["root"],
                )
                state = "submitted"

            _require(state == "submitted", "qualification launch did not reach submission")
            _require(
                isinstance(job_id, str) and job_id.isdigit(),
                "qualification launch lost its accepted job ID",
            )
            receipt = _launch_receipt(intent=intent, job_id=job_id)
            _atomic_json(
                paths["receipt"], receipt, episode_root=paths["root"]
            )
            verified_receipt = read_qualification_launch_receipt(
                paths["receipt"], episode_root=paths["root"]
            )
            return _verify_receipt(
                verified_receipt, intent=intent, job_id=job_id
            )
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-job-id", required=True)
    args = parser.parse_args(argv)
    receipt = launch_qualification003(test_job_id=args.test_job_id)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
