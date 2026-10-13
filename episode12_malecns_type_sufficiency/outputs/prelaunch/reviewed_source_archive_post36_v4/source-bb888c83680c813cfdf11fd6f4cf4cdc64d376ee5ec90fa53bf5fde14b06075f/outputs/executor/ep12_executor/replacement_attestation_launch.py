"""Exact-once archive-bound replacement for obsolete EP12 attestation job 45332084.

The launch step is safe while the obsolete job remains held.  It submits one
replacement afterok:45323481 from a content-addressed reviewed episode archive,
then returns a durable receipt containing the acknowledged dynamic job ID.
The enclosing controller cancels only that held job and continues without a manual ID.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Callable, Mapping, Sequence

from .policy import canonical_json, digest_object
from .post36_orchestration import (
    OBSERVED_ATTESTATION_JOB_ID,
    OBSERVED_ATTESTATION_JOB_NAME,
    OBSERVED_FINALIZER_JOB_ID,
    STAGE_INTENT_SCHEMA,
    Post36OrchestrationError,
    _atomic_json,
    _effective_user,
    _release_authorization,
    _release_intent,
    _release_progress,
    _release_receipt,
    _read_json,
    _run_scheduler,
    _sha256_file,
    _stage_receipt,
    _stage_submission_evidence,
    _submit_stage_held,
    _superseded_attestation_evidence,
    _verify_stage_progress,
    _write_or_verify,
    default_exact_name_resolver,
    default_job_state_resolver,
    default_submitter,
)
from .secure_transaction import open_regular_lock_at, open_secure_directory
from .source_archive import verify_source_archive
from .synthetic import EPISODE_ROOT


INTENT_SCHEMA = "ep12.replacement_attestation_launch_intent.v1"
PROGRESS_SCHEMA = "ep12.replacement_attestation_launch_progress.v1"
RECEIPT_SCHEMA = "ep12.replacement_attestation_launch_receipt.v1"
TRANSACTION_RELATIVE = Path("outputs/prelaunch/replacement_attestation_launch")
INTENT_RELATIVE = TRANSACTION_RELATIVE / "intent.json"
PROGRESS_RELATIVE = TRANSACTION_RELATIVE / "progress.json"
RECEIPT_RELATIVE = TRANSACTION_RELATIVE / "receipt.json"
STAGE_ROOT_RELATIVE = TRANSACTION_RELATIVE / "stages/replacement_attestation"
STAGE_INTENT_RELATIVE = STAGE_ROOT_RELATIVE / "intent.json"
STAGE_PROGRESS_RELATIVE = STAGE_ROOT_RELATIVE / "progress.json"
STAGE_RECEIPT_RELATIVE = STAGE_ROOT_RELATIVE / "receipt.json"
STAGE_RELEASE_INTENT_RELATIVE = STAGE_ROOT_RELATIVE / "release_intent.json"
STAGE_RELEASE_AUTHORIZATION_RELATIVE = (
    STAGE_ROOT_RELATIVE / "release_authorization.json"
)
STAGE_RELEASE_PROGRESS_RELATIVE = STAGE_ROOT_RELATIVE / "release_progress.json"
STAGE_RELEASE_RECEIPT_RELATIVE = STAGE_ROOT_RELATIVE / "release_receipt.json"
HOLD_RELATIVE = Path("outputs/prelaunch/obsolete_attestation_hold_45332084.json")
HOLD_FILE_SHA256 = (
    "91ed223a366fed5c600e9fce83ae3ca5a2c1c25c3a73db889e023727956f415e"
)
WRAPPER_RELATIVE = Path("slurm/attest_observed_trials_from_archive_v2.sbatch")
BOOTSTRAP_RELATIVE = Path("ep12_executor/source_archive_bootstrap.py")
BOOTSTRAP_SHA256 = (
    "65f84b2879325e2549cdff7f138aebd3df40db64811327d7f4bd7d50c1bc3aaa"
)
ATTESTATION_OUTPUT_RELATIVE = Path(
    "outputs/development_run002/observed_attestation"
)
TERMINAL_IDENTITY_RELATIVE = (
    TRANSACTION_RELATIVE / "terminal_identity.json"
)
BOOTSTRAP_STAGE_RESULT_RELATIVE = (
    TRANSACTION_RELATIVE / "bootstrap_stage_result.json"
)
ARCHIVE_STAGE_PARENT = Path(
    "/scratch/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/source_archive_stage"
)
_HEX_RE = re.compile(r"^[0-9a-f]{64}$")

DYNAMIC_TRANSITION_ARCHIVE_FILES = frozenset(
    {
        "outputs/executor/SCIENTIFIC_CONTRACT.yaml",
        "outputs/executor/SCIENTIFIC_CONTRACT_POST36.yaml",
        "outputs/executor/SYNTHETIC_QUALIFICATION_SPEC.yaml",
        "outputs/executor/SYNTHETIC_QUALIFICATION_SPEC_POST36.yaml",
        "outputs/executor/ep12_executor/__init__.py",
        "outputs/executor/ep12_executor/__main__.py",
        "outputs/executor/ep12_executor/cli.py",
        "outputs/executor/ep12_executor/contract_provenance.py",
        "outputs/executor/ep12_executor/contract_snapshot.py",
        "outputs/executor/ep12_executor/controller.py",
        "outputs/executor/ep12_executor/journal.py",
        "outputs/executor/ep12_executor/launch_chain_identity.py",
        "outputs/executor/ep12_executor/policy.py",
        "outputs/executor/ep12_executor/scientific_models.py",
        "outputs/executor/ep12_executor/scientific_qualification.py",
        "outputs/executor/ep12_executor/scientific_synthetic.py",
        "outputs/executor/ep12_executor/secure_transaction.py",
        "outputs/executor/ep12_executor/synthetic.py",
        "outputs/executor/ep12_executor/yaml_subset.py",
        "outputs/executor/tests/test_archive_wrappers.py",
        "outputs/executor/tests/test_contract_transition.py",
        "outputs/executor/tests/test_post36_unattended_controller.py",
        "outputs/executor/tests/test_production_readiness_transition.py",
        "outputs/executor/tests/test_qualification_launch.py",
        "outputs/executor/tests/test_scientific_transition_binding.py",
        "outputs/executor/tests/test_scientific_transition_publish.py",
    }
)

TAIL_HANDOFF_ARCHIVE_FILES = frozenset(
    {
        "outputs/executor/DEVELOPMENT_BUDGET_OVERRIDE.yaml",
        "outputs/executor/FALSIFIER_CONTROL_IMPLEMENTATION_SPEC.json",
        "outputs/executor/OBSERVED_POST36_TAIL_PLAN.json",
        "outputs/executor/ep12_executor/archive_nested_qualification.py",
        "outputs/executor/ep12_executor/archive_slurm_handoff.py",
        "outputs/executor/ep12_executor/post36_archive_nested_stages.py",
        "outputs/executor/ep12_executor/post36_downstream_receipt_bound.py",
        "outputs/executor/ep12_executor/post36_tail.py",
        "outputs/executor/ep12_executor/slurm_handoff.py",
        "outputs/executor/slurm/run_post36_tail_from_archive.sbatch",
        "outputs/executor/slurm/run_procedure_lock_from_archive.sbatch",
        "outputs/executor/slurm/run_procedure_lock.sbatch",
        "outputs/executor/tests/test_archive_downstream_wrappers.py",
        "outputs/executor/tests/test_post36_downstream_receipt_bound.py",
        "outputs/executor/tests/test_post36_downstream_launch.py",
        "outputs/executor/tests/test_post36_exact_once_barrier.py",
        "outputs/executor/tests/test_post36_tail.py",
        "outputs/executor/tests/test_slurm_handoff.py",
        "outputs/executor/tests/test_slurm_handoffs.py",
    }
)

REQUIRED_ARCHIVE_FILES = frozenset(
    {
        "SEARCH_POLICY.yaml",
        "outputs/executor/CONTRACT_TRANSITION_PROBE_SPEC.json",
        "outputs/executor/CONTRACT_TRANSITION_SPEC.json",
        "outputs/executor/ep12_executor/attest_observed_trials.py",
        "outputs/executor/ep12_executor/contract_invariance_probe.py",
        "outputs/executor/ep12_executor/obsolete_attestation_cancel.py",
        "outputs/executor/ep12_executor/post36_downstream_launch.py",
        "outputs/executor/ep12_executor/post36_orchestration.py",
        "outputs/executor/ep12_executor/post36_receipt_bound.py",
        "outputs/executor/ep12_executor/post36_unattended_controller.py",
        "outputs/executor/ep12_executor/production_readiness.py",
        "outputs/executor/ep12_executor/qualification_launch.py",
        "outputs/executor/ep12_executor/replacement_attestation_terminal.py",
        "outputs/executor/ep12_executor/replacement_attestation_launch.py",
        "outputs/executor/ep12_executor/scientific_contract_transition.py",
        "outputs/executor/ep12_executor/scientific_transition_binding.py",
        "outputs/executor/ep12_executor/scientific_transition_publish.py",
        "outputs/executor/ep12_executor/slurm_client.py",
        "outputs/executor/ep12_executor/source_archive.py",
        "outputs/executor/ep12_executor/source_archive_bootstrap.py",
        "outputs/executor/ep12_executor/source_archive_freeze_bridge.py",
        "outputs/executor/ep12_executor/source_freeze.py",
        "outputs/executor/ep12_executor/test_launch.py",
        "outputs/executor/slurm/attest_observed_trials_from_archive_v2.sbatch",
        "outputs/executor/slurm/check_production_readiness.sbatch",
        "outputs/executor/slurm/compare_contract_invariance_from_archive.sbatch",
        "outputs/executor/slurm/launch_post36_downstream_from_archive.sbatch",
        "outputs/executor/slurm/publish_scientific_transition_from_archive.sbatch",
        "outputs/executor/slurm/qualify_methods.sbatch",
        "outputs/executor/slurm/run_contract_invariance_probe_from_archive.sbatch",
        "outputs/executor/slurm/run_post36_tail.sbatch",
        "outputs/executor/slurm/test_executor.sbatch",
    }
) | DYNAMIC_TRANSITION_ARCHIVE_FILES | TAIL_HANDOFF_ARCHIVE_FILES


class ReplacementAttestationLaunchError(RuntimeError):
    """The held-job replacement cannot be authenticated or submitted uniquely."""


JobStateResolver = Callable[[str, str, str], Mapping[str, str]]
Submitter = Callable[[Sequence[str]], str]
JobResolver = Callable[[str, str], Mapping[str, str] | None]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplacementAttestationLaunchError(message)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular(path: Path, *, label: str) -> bytes:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        metadata = os.lstat(current)
        _require(not stat.S_ISLNK(metadata.st_mode), f"{label} traverses a symlink")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
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


def _hold_evidence(episode_root: Path) -> dict[str, Any]:
    path = episode_root / HOLD_RELATIVE
    encoded = _read_regular(path, label="obsolete-attestation hold record")
    _require(_sha_bytes(encoded) == HOLD_FILE_SHA256, "obsolete hold record bytes changed")
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplacementAttestationLaunchError("obsolete hold record is malformed") from exc
    _require(
        isinstance(value, Mapping)
        and value.get("schema_version") == "ep12.obsolete_attestation_safety_hold.v1"
        and value.get("job_id") == OBSERVED_ATTESTATION_JOB_ID
        and value.get("job_name") == OBSERVED_ATTESTATION_JOB_NAME
        and value.get("dependency") == f"afterok:{OBSERVED_FINALIZER_JOB_ID}(unfulfilled)"
        and value.get("post_action_state") == "PENDING"
        and value.get("post_action_reason") == "JobHeldUser"
        and value.get("post_action_priority") == 0
        and value.get("descendants_in_same_user_queue_before_hold") == []
        and value.get("reversible") is True,
        "obsolete hold record facts changed",
    )
    return {
        "path": str(path),
        "file_sha256": HOLD_FILE_SHA256,
        "job_id": OBSERVED_ATTESTATION_JOB_ID,
        "state": "PENDING",
        "reason": "JobHeldUser",
        "priority": 0,
        "queued_descendants_before_hold": [],
        "reversible": True,
    }


def default_obsolete_state_resolver(
    job_id: str, job_name: str, submission_user: str
) -> Mapping[str, str]:
    """Resolve a live hold from squeue, falling back once to terminal sacct."""

    _require(submission_user == _effective_user(), "obsolete-state user mismatch")
    live_output = _run_scheduler(
        [
            "/usr/bin/squeue", "--noheader", "--user", submission_user, "--jobs", job_id,
            "--format=%i|%j|%u|%T|%R|%Q",
        ]
    )
    live_rows: list[dict[str, str]] = []
    for line in live_output.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        _require(len(fields) == 6, "malformed obsolete-job squeue row")
        raw_id, observed_name, observed_user, state, reason, priority = fields
        if raw_id != job_id:
            continue
        live_rows.append(
            {
                "job_id": raw_id,
                "job_name": observed_name,
                "user": observed_user,
                "state": state,
                "reason": reason,
                "priority": priority,
            }
        )
    _require(len(live_rows) <= 1, "obsolete job is ambiguous in squeue")
    if live_rows:
        row = live_rows[0]
        _require(
            row["job_name"] == job_name and row["user"] == submission_user,
            "live obsolete scheduler identity changed",
        )
        return row

    accounting_output = _run_scheduler(
        [
            "/usr/bin/sacct", "-X", "--noheader", "--parsable2", "--user",
            submission_user, "--jobs", job_id, "--starttime", "2026-09-25",
            "--format=JobIDRaw,JobName%128,User%64,State,Reason,Priority",
        ]
    )
    rows: list[dict[str, str]] = []
    for line in accounting_output.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split("|")
        _require(len(fields) == 6, "malformed obsolete-job sacct row")
        raw_id, observed_name, observed_user, state, reason, priority = fields
        if raw_id != job_id:
            continue
        rows.append(
            {
                "job_id": raw_id,
                "job_name": observed_name,
                "user": observed_user,
                "state": state.rstrip("+").split()[0],
                "reason": reason,
                "priority": priority,
            }
        )
    _require(len(rows) == 1, "obsolete job is absent or ambiguous in sacct")
    row = rows[0]
    _require(
        row["job_name"] == job_name and row["user"] == submission_user,
        "terminal obsolete scheduler identity changed",
    )
    _require(
        row["state"].startswith("CANCELLED"),
        "sacct fallback is not terminal cancellation; refusing stale pending state",
    )
    return row


def _validate_obsolete_state(
    value: Mapping[str, str], *, submission_user: str, allow_cancelled: bool = False
) -> dict[str, str]:
    row = dict(value)
    _require(
        set(row) == {"job_id", "job_name", "user", "state", "reason", "priority"}
        and row["job_id"] == OBSERVED_ATTESTATION_JOB_ID
        and row["job_name"] == OBSERVED_ATTESTATION_JOB_NAME
        and row["user"] == submission_user,
        "obsolete scheduler row changed",
    )
    if row["state"] == "PENDING":
        _require(
            row["reason"] in {"JobHeldUser", "(JobHeldUser)"}
            and row["priority"] == "0",
            "obsolete job is pending without the exact safety hold",
        )
        # Sherlock's squeue %R renders the same user hold both with and
        # without parentheses across client/output contexts.  Persist one
        # canonical value so the launch intent is stable across replay.
        row["reason"] = "JobHeldUser"
    else:
        _require(
            allow_cancelled and row["state"].startswith("CANCELLED"),
            "obsolete job was not held through replacement acknowledgement",
        )
    return row


def _archive_identity(
    *,
    archive_root: Path,
    receipt_file_sha256: str,
    receipt_sha256: str,
    manifest_file_sha256: str,
    manifest_sha256: str,
    tree_sha256: str,
) -> dict[str, Any]:
    try:
        verified = verify_source_archive(
            archive_root,
            expected_receipt_file_sha256=receipt_file_sha256,
            expected_receipt_sha256=receipt_sha256,
            expected_manifest_file_sha256=manifest_file_sha256,
            expected_manifest_sha256=manifest_sha256,
            expected_tree_sha256=tree_sha256,
        )
    except Exception as exc:
        raise ReplacementAttestationLaunchError(str(exc)) from exc
    manifest_path = Path(str(verified["manifest_path"]))
    manifest = json.loads(_read_regular(manifest_path, label="source archive manifest"))
    file_paths = {str(record["path"]) for record in manifest["files"]}
    directory_paths = set(str(value) for value in manifest["directories"])
    _require(
        file_paths == {
            value for value in file_paths
            if value == "SEARCH_POLICY.yaml" or value.startswith("outputs/executor/")
        },
        "archive contains files outside SEARCH_POLICY.yaml + outputs/executor/**",
    )
    _require(
        "SEARCH_POLICY.yaml" in file_paths
        and REQUIRED_ARCHIVE_FILES <= file_paths,
        "archive omits required replacement or post36 source",
    )
    _require(
        directory_paths == {
            value for value in directory_paths
            if value in {"outputs", "outputs/executor"}
            or value.startswith("outputs/executor/")
        },
        "archive contains directories outside the exact episode-source allowlist",
    )
    bootstrap = Path(str(verified["archived_root"])) / "outputs/executor" / BOOTSTRAP_RELATIVE
    _require(
        _sha256_file(
            bootstrap,
            root=Path(str(verified["archived_root"])),
            label="archive bootstrap",
        )
        == BOOTSTRAP_SHA256,
        "archive bootstrap bytes are not the independently reviewed trust anchor",
    )
    return {
        "archived_root": verified["archived_root"],
        "archive_parent": verified["archive_parent"],
        "archive_receipt_path": verified["archive_receipt_path"],
        "archive_receipt_file_sha256": verified["archive_receipt_file_sha256"],
        "archive_receipt_sha256": verified["archive_receipt_sha256"],
        "manifest_path": verified["manifest_path"],
        "manifest_file_sha256": verified["manifest_file_sha256"],
        "manifest_sha256": verified["manifest_sha256"],
        "tree_sha256": verified["tree_sha256"],
        "surface_allowlist": ["SEARCH_POLICY.yaml", "outputs/executor/**"],
        "inputs_included": False,
        "development_outputs_included": False,
        "prelaunch_outputs_included": False,
        "bootstrap_sha256": BOOTSTRAP_SHA256,
        "parent_first_publication_bound": True,
        "complete": True,
    }


def _build_intent(
    *,
    episode_root: Path,
    archive: Mapping[str, Any],
    submission_user: str,
    held_observation: Mapping[str, str],
    superseded: Mapping[str, Any],
) -> dict[str, Any]:
    hold = _hold_evidence(episode_root)
    _require(superseded["authoritative"] is False, "obsolete identity became authoritative")
    archive_executor = Path(str(archive["archived_root"])) / "outputs/executor"
    archived_launcher = archive_executor / "ep12_executor/replacement_attestation_launch.py"
    _require(
        _sha_bytes(_read_regular(Path(__file__), label="active replacement launcher"))
        == _sha256_file(
            archived_launcher,
            root=Path(str(archive["archived_root"])),
            label="archived replacement launcher",
        ),
        "active replacement launcher differs from reviewed archive bytes",
    )
    core = {
        "schema_version": INTENT_SCHEMA,
        "submission_user": submission_user,
        "episode_root": str(episode_root),
        "archive": dict(archive),
        "dependency": f"afterok:{OBSERVED_FINALIZER_JOB_ID}",
        "observed_finalizer_job_id": OBSERVED_FINALIZER_JOB_ID,
        "superseded_attestation": dict(superseded),
        "obsolete_hold": hold,
        "held_observation_immediately_before_submit": dict(held_observation),
        "replacement_attestation_output": str(
            episode_root / ATTESTATION_OUTPUT_RELATIVE
        ),
        "terminal_identity_output": str(
            episode_root / TERMINAL_IDENTITY_RELATIVE
        ),
        "replacement_job_identity_is_only_candidate_authority": True,
        "attestation_result_authoritative": False,
        "obsolete_job_must_remain_held_until_replacement_ack": True,
        "obsolete_job_cancellation_is_automatic_and_exact": True,
        "real_connectivity_accessed": False,
        "connectivity_outcome_values_read": False,
        "final_connectivity_accessed": False,
    }
    return {**core, "intent_sha256": digest_object(core)}


def _stage_intent(
    *,
    episode_root: Path,
    intent: Mapping[str, Any],
    intent_file_sha256: str,
) -> dict[str, Any]:
    archive = intent["archive"]
    _require(isinstance(archive, Mapping), "replacement intent lacks archive identity")
    stage_parent = ARCHIVE_STAGE_PARENT / str(archive["tree_sha256"])
    staged_root = stage_parent / f"source-{archive['tree_sha256']}"
    bootstrap_stage_result = episode_root / BOOTSTRAP_STAGE_RESULT_RELATIVE
    stage_receipt_path = episode_root / STAGE_RECEIPT_RELATIVE
    archived_executor = Path(str(archive["archived_root"])) / "outputs/executor"
    wrapper = archived_executor / WRAPPER_RELATIVE
    wrapper_sha256 = _sha256_file(
        wrapper,
        root=Path(str(archive["archived_root"])),
        label="replacement attestation wrapper",
    )
    arguments = [
        str(episode_root),
        str(archive["archived_root"]),
        str(archive["archive_receipt_file_sha256"]),
        str(archive["archive_receipt_sha256"]),
        str(archive["manifest_file_sha256"]),
        str(archive["manifest_sha256"]),
        str(archive["tree_sha256"]),
        str(episode_root / INTENT_RELATIVE),
        intent_file_sha256,
        str(intent["intent_sha256"]),
        str(intent["replacement_attestation_output"]),
        str(bootstrap_stage_result),
        str(staged_root),
        str(stage_receipt_path),
        str(intent["terminal_identity_output"]),
    ]
    core = {
        "schema_version": STAGE_INTENT_SCHEMA,
        "orchestration_intent_sha256": intent["intent_sha256"],
        "stage": "replacement_attestation",
        "submission_user": intent["submission_user"],
        "predecessor_job_id": OBSERVED_FINALIZER_JOB_ID,
        "dependency": f"afterok:{OBSERVED_FINALIZER_JOB_ID}",
        "kill_on_invalid_dependency": True,
        "submission_barrier": "user_hold_receipt_before_release_v1",
        "wrapper": str(wrapper),
        "wrapper_sha256": wrapper_sha256,
        "arguments": arguments,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    token = digest_object(core)[:24]
    named = {**core, "job_name": f"ep12-replacement-attest-{token}"}
    return {**named, "stage_intent_sha256": digest_object(named)}


def _progress(
    *, intent: Mapping[str, Any], stage_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    core = {
        "schema_version": PROGRESS_SCHEMA,
        "intent_sha256": intent["intent_sha256"],
        "state": "replacement_accepted_held",
        "slurm_job_id": stage_receipt["slurm_job_id"],
        "stage_receipt_sha256": stage_receipt["stage_receipt_sha256"],
        "final_connectivity_accessed": False,
    }
    return {**core, "progress_sha256": digest_object(core)}


def _receipt(
    *,
    intent: Mapping[str, Any],
    intent_file_sha256: str,
    progress: Mapping[str, Any],
    stage_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": RECEIPT_SCHEMA,
        "intent_path": str(Path(str(intent["episode_root"])) / INTENT_RELATIVE),
        "intent_file_sha256": intent_file_sha256,
        "intent_sha256": intent["intent_sha256"],
        "progress_sha256": progress["progress_sha256"],
        "slurm_job_id": stage_receipt["slurm_job_id"],
        "slurm_job_name": stage_receipt["slurm_job_name"],
        "dependency": stage_receipt["dependency"],
        "stage_receipt_sha256": stage_receipt["stage_receipt_sha256"],
        "replacement_remains_held": True,
        "release_owner": "ep12.post36_unattended_controller.v1",
        "archive": dict(intent["archive"]),
        "terminal_identity_path": intent["terminal_identity_output"],
        "attestation_output": intent["replacement_attestation_output"],
        "supersedes_job_id": OBSERVED_ATTESTATION_JOB_ID,
        "obsolete_hold_file_sha256": HOLD_FILE_SHA256,
        "authoritative_attestation_job_identity": True,
        "attestation_result_authoritative": False,
        "held_observation_immediately_before_submit": dict(
            intent["held_observation_immediately_before_submit"]
        ),
        "job_id_is_receipt_bound": True,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
    }
    return {**core, "receipt_sha256": digest_object(core)}


def launch_replacement_attestation(
    *,
    episode_root: Path,
    archive_root: Path,
    archive_receipt_file_sha256: str,
    archive_receipt_sha256: str,
    archive_manifest_file_sha256: str,
    archive_manifest_sha256: str,
    archive_tree_sha256: str,
    submitter: Submitter = default_submitter,
    job_resolver: JobResolver = default_exact_name_resolver,
    accepted_job_state_resolver: JobStateResolver = default_job_state_resolver,
    obsolete_state_resolver: JobStateResolver = default_obsolete_state_resolver,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(episode_root)))
    _require(root == EPISODE_ROOT, "replacement launch episode root is not canonical")
    for value in (
        archive_receipt_file_sha256, archive_receipt_sha256,
        archive_manifest_file_sha256, archive_manifest_sha256,
        archive_tree_sha256,
    ):
        _require(_HEX_RE.fullmatch(value) is not None, "archive digest is invalid")
    try:
        existing_receipt = _read_json(
            root / RECEIPT_RELATIVE,
            root=root,
            label="replacement attestation launch receipt",
            canonical=True,
        )
    except FileNotFoundError:
        existing_receipt = None
    if existing_receipt is not None:
        evidence = verify_replacement_attestation_receipt(
            episode_root=root, require_released=None
        )
        requested_archive = {
            "archived_root": str(Path(os.path.abspath(os.fspath(archive_root)))),
            "archive_receipt_file_sha256": archive_receipt_file_sha256,
            "archive_receipt_sha256": archive_receipt_sha256,
            "manifest_file_sha256": archive_manifest_file_sha256,
            "manifest_sha256": archive_manifest_sha256,
            "tree_sha256": archive_tree_sha256,
        }
        _require(
            all(evidence["archive"].get(key) == value
                for key, value in requested_archive.items()),
            "replacement replay requested a different source archive",
        )
        return existing_receipt
    submission_user = _effective_user()
    archive = _archive_identity(
        archive_root=archive_root,
        receipt_file_sha256=archive_receipt_file_sha256,
        receipt_sha256=archive_receipt_sha256,
        manifest_file_sha256=archive_manifest_file_sha256,
        manifest_sha256=archive_manifest_sha256,
        tree_sha256=archive_tree_sha256,
    )
    transaction, fd = open_secure_directory(
        root / TRANSACTION_RELATIVE, root=root, create=True
    )
    lock_fd = open_regular_lock_at(fd, ".lock")
    os.close(fd)
    with os.fdopen(lock_fd, "a+b") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        try:
            held_observation = _validate_obsolete_state(
                obsolete_state_resolver(
                    OBSERVED_ATTESTATION_JOB_ID,
                    OBSERVED_ATTESTATION_JOB_NAME,
                    submission_user,
                ),
                submission_user=submission_user,
                allow_cancelled=False,
            )
            expected_intent = _build_intent(
                episode_root=root,
                archive=archive,
                submission_user=submission_user,
                held_observation=held_observation,
                superseded=_superseded_attestation_evidence(
                    episode_root=root,
                    submission_user=submission_user,
                ),
            )
            intent = _write_or_verify(
                root / INTENT_RELATIVE,
                expected_intent,
                root=root,
                label="replacement attestation launch intent",
            )
            intent_file_sha256 = _sha256_file(
                root / INTENT_RELATIVE,
                root=root,
                label="replacement attestation launch intent",
            )
            stage_receipt = _submit_stage_held(
                transaction_root=transaction,
                root=root,
                stage_intent=_stage_intent(
                    episode_root=root,
                    intent=intent,
                    intent_file_sha256=intent_file_sha256,
                ),
                submitter=submitter,
                job_resolver=job_resolver,
                job_state_resolver=accepted_job_state_resolver,
            )
            progress = _write_or_verify(
                root / PROGRESS_RELATIVE,
                _progress(intent=intent, stage_receipt=stage_receipt),
                root=root,
                label="replacement attestation launch progress",
            )
            receipt = _receipt(
                intent=intent,
                intent_file_sha256=intent_file_sha256,
                progress=progress,
                stage_receipt=stage_receipt,
            )
            return _write_or_verify(
                root / RECEIPT_RELATIVE,
                receipt,
                root=root,
                label="replacement attestation launch receipt",
            )
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)

def _verified_release_evidence(
    *,
    root: Path,
    stage_receipt: Mapping[str, Any],
    require_released: bool | None,
) -> dict[str, Any]:
    intent_path = root / STAGE_RELEASE_INTENT_RELATIVE
    authorization_path = root / STAGE_RELEASE_AUTHORIZATION_RELATIVE
    progress_path = root / STAGE_RELEASE_PROGRESS_RELATIVE
    receipt_path = root / STAGE_RELEASE_RECEIPT_RELATIVE
    try:
        release_intent = _read_json(
            intent_path, root=root,
            label="replacement attestation release intent", canonical=True,
        )
    except FileNotFoundError:
        for candidate, label in (
            (authorization_path, "replacement attestation release authorization"),
            (progress_path, "replacement attestation release progress"),
            (receipt_path, "replacement attestation release receipt"),
        ):
            try:
                _read_json(candidate, root=root, label=label, canonical=True)
            except FileNotFoundError:
                continue
            raise Post36OrchestrationError(
                "replacement release transaction is only partially present"
            )
        _require(
            require_released is not True,
            "replacement attestation has not been durably released",
        )
        return {
            "replacement_released": False,
            "release_intent_path": None,
            "release_intent_file_sha256": None,
            "release_intent_sha256": None,
            "release_authorization_context": None,
            "release_authorized": False,
            "release_authorization_path": None,
            "release_authorization_file_sha256": None,
            "release_authorization_sha256": None,
            "release_receipt_path": None,
            "release_receipt_file_sha256": None,
            "release_receipt_sha256": None,
            "receipt_was_durable_before_release": False,
        }
    context = release_intent.get("authorization_context")
    _require(isinstance(context, Mapping),
             "replacement release lacks the post36 start-authorization context")
    _require(
        set(context)
        == {
            "schema_version", "start_authorization_path",
            "start_authorization_file_sha256", "start_authorization_sha256",
        }
        and context.get("schema_version")
        == "ep12.post36_replacement_release_context.v1",
        "replacement release authorization context schema changed",
    )
    start_path = root / "outputs/prelaunch/post36_orchestration/start_authorization.json"
    _require(
        Path(os.path.abspath(os.fspath(context["start_authorization_path"])))
        == start_path,
        "replacement release start-authorization path is not canonical",
    )
    start = _read_json(
        start_path, root=root, label="post36 start authorization", canonical=True
    )
    start_core = {
        key: value for key, value in start.items()
        if key != "start_authorization_sha256"
    }
    _require(
        start.get("schema_version") == "ep12.post36_start_authorization.v1"
        and start.get("start_authorization_sha256") == digest_object(start_core)
        == context["start_authorization_sha256"]
        and _sha256_file(
            start_path, root=root, label="post36 start authorization"
        ) == context["start_authorization_file_sha256"]
        and start.get("replacement_job_id") == stage_receipt["slurm_job_id"]
        and start.get("replacement_stage_receipt_path")
        == str(root / STAGE_RECEIPT_RELATIVE)
        and start.get("replacement_stage_receipt_file_sha256")
        == _sha256_file(
            root / STAGE_RECEIPT_RELATIVE, root=root,
            label="replacement attestation stage receipt",
        )
        and start.get("replacement_stage_receipt_sha256")
        == stage_receipt["stage_receipt_sha256"]
        and start.get("all_child_stage_and_release_receipts_durable") is True
        and start.get("obsolete_job_terminally_cancelled") is True
        and start.get("replacement_may_be_released") is True
        and start.get("obsolete_job_is_execution_dependency") is False
        and start.get("values_used_for_transition_decisions") is False
        and start.get("final_connectivity_accessed") is False
        and start.get("final_connectivity_access_authorized") is False,
        "replacement release start authorization changed",
    )
    stage_map = start.get("stages")
    _require(
        isinstance(stage_map, Mapping)
        and set(stage_map) == {"post36_probe", "comparison", "publisher", "downstream"}
        and all(
            isinstance(value, Mapping)
            and str(value.get("slurm_job_id", "")).isdigit()
            and int(str(value["slurm_job_id"])) > 0
            and OBSERVED_ATTESTATION_JOB_ID not in str(value.get("dependency", ""))
            and isinstance(value.get("stage_receipt_file_sha256"), str)
            and isinstance(value.get("release_receipt_file_sha256"), str)
            for value in stage_map.values()
        ),
        "replacement release child-stage authorization changed",
    )
    cancellation_path = (
        root
        / "outputs/prelaunch/replacement_attestation_launch/obsolete_cancellation/receipt.json"
    )
    _require(
        start.get("obsolete_cancellation_receipt_path") == str(cancellation_path)
        and _sha256_file(
            cancellation_path, root=root,
            label="obsolete cancellation receipt",
        ) == start.get("obsolete_cancellation_receipt_file_sha256"),
        "replacement release cancellation binding changed",
    )
    cancellation_receipt = _read_json(
        cancellation_path, root=root,
        label="obsolete cancellation receipt", canonical=True,
    )
    cancellation_core = {
        key: value for key, value in cancellation_receipt.items()
        if key != "receipt_sha256"
    }
    _require(
        cancellation_receipt.get("receipt_sha256")
        == digest_object(cancellation_core)
        == start.get("obsolete_cancellation_receipt_sha256")
        and str(cancellation_receipt.get("terminal_scheduler_state", ""))
        .startswith("CANCELLED")
        and cancellation_receipt.get("replacement_job_id")
        == stage_receipt["slurm_job_id"],
        "replacement release cancellation receipt changed",
    )
    dependency_predecessor = stage_receipt["slurm_job_id"]
    for stage_name in ("post36_probe", "comparison", "publisher", "downstream"):
        value = stage_map[stage_name]
        expected_dir = (
            root / "outputs/prelaunch/post36_orchestration/stages" / stage_name
        )
        stage_path = expected_dir / "receipt.json"
        _require(
            set(value)
            == {
                "slurm_job_id", "slurm_job_name", "dependency",
                "stage_receipt_path", "stage_receipt_file_sha256",
                "stage_receipt_sha256", "release_authorization_file_sha256",
                "release_authorization_sha256", "release_receipt_file_sha256",
                "release_receipt_sha256",
            }
            and value["dependency"] == f"afterok:{dependency_predecessor}"
            and value["stage_receipt_path"] == str(stage_path)
            and _sha256_file(
                stage_path, root=root, label=f"{stage_name} stage receipt"
            ) == value["stage_receipt_file_sha256"],
            f"replacement release {stage_name} stage binding changed",
        )
        persisted_stage = _read_json(
            stage_path, root=root, label=f"{stage_name} stage receipt",
            canonical=True,
        )
        _require(
            persisted_stage.get("stage") == stage_name
            and persisted_stage.get("slurm_job_id") == value["slurm_job_id"]
            and persisted_stage.get("slurm_job_name") == value["slurm_job_name"]
            and persisted_stage.get("dependency") == value["dependency"]
            and persisted_stage.get("stage_receipt_sha256")
            == value["stage_receipt_sha256"],
            f"replacement release {stage_name} stage receipt changed",
        )
        for filename, file_key, semantic_key, artifact_field in (
            (
                "release_authorization.json",
                "release_authorization_file_sha256",
                "release_authorization_sha256",
                "release_authorization_sha256",
            ),
            (
                "release_receipt.json", "release_receipt_file_sha256",
                "release_receipt_sha256", "release_receipt_sha256",
            ),
        ):
            artifact_path = expected_dir / filename
            artifact = _read_json(
                artifact_path, root=root,
                label=f"{stage_name} {filename}", canonical=True,
            )
            _require(
                _sha256_file(
                    artifact_path, root=root,
                    label=f"{stage_name} {filename}",
                ) == value[file_key]
                and artifact.get(artifact_field) == value[semantic_key]
                and artifact.get("slurm_job_id") == value["slurm_job_id"],
                f"replacement release {stage_name} {filename} changed",
            )
        dependency_predecessor = str(value["slurm_job_id"])
    expected_intent = _release_intent(
        stage_receipt=stage_receipt,
        stage_receipt_file_sha256=_sha256_file(
            root / STAGE_RECEIPT_RELATIVE, root=root,
            label="replacement attestation stage receipt",
        ),
        authorization_context=dict(context),
    )
    _require(release_intent == expected_intent, "replacement release intent mutated")
    release_authorization = _read_json(
        authorization_path, root=root,
        label="replacement attestation release authorization", canonical=True,
    )
    _require(
        release_authorization
        == _release_authorization(release_intent=release_intent),
        "replacement release authorization mutated",
    )
    _require(
        require_released is not False,
        "replacement attestation was release-authorized before cancellation completed",
    )
    release_progress = _read_json(
        progress_path, root=root,
        label="replacement attestation release progress", canonical=True,
    )
    uncertain = _release_progress(
        release_intent=release_intent, state="release_uncertain"
    )
    released = _release_progress(
        release_intent=release_intent, state="released"
    )
    _require(
        release_progress == uncertain or release_progress == released,
        "replacement release was not durably authorized before scontrol",
    )
    try:
        release_receipt = _read_json(
            receipt_path, root=root,
            label="replacement attestation release receipt", canonical=True,
        )
    except FileNotFoundError:
        release_receipt = None
    if release_receipt is None:
        _require(
            release_progress == uncertain and require_released is not True,
            "replacement attestation lacks its final release receipt",
        )
        return {
            "replacement_released": False,
            "release_intent_path": str(intent_path),
            "release_intent_file_sha256": _sha256_file(
                intent_path, root=root,
                label="replacement attestation release intent",
            ),
            "release_intent_sha256": release_intent["release_intent_sha256"],
            "release_authorization_context": dict(context),
            "release_authorized": True,
            "release_authorization_path": str(authorization_path),
            "release_authorization_file_sha256": _sha256_file(
                authorization_path, root=root,
                label="replacement attestation release authorization",
            ),
            "release_authorization_sha256": release_authorization[
                "release_authorization_sha256"
            ],
            "release_receipt_path": None,
            "release_receipt_file_sha256": None,
            "release_receipt_sha256": None,
            "receipt_was_durable_before_release": True,
        }
    # Avoid a cross-file read race: a receipt can become visible after the
    # first progress read, so authenticate the final progress again.
    release_progress = _read_json(
        progress_path, root=root,
        label="replacement attestation release progress", canonical=True,
    )
    _require(release_progress == released, "replacement final release progress mutated")
    _require(
        release_receipt
        == _release_receipt(
            release_intent=release_intent, release_progress=release_progress
        ),
        "replacement release receipt mutated",
    )
    submission = _stage_submission_evidence(
        stage_dir=root / STAGE_ROOT_RELATIVE,
        root=root,
        stage_receipt=stage_receipt,
        release_receipt=release_receipt,
    )
    return {
        "replacement_released": True,
        "release_intent_path": str(intent_path),
        "release_intent_file_sha256": _sha256_file(
            intent_path, root=root, label="replacement attestation release intent"
        ),
        "release_intent_sha256": release_intent["release_intent_sha256"],
        "release_authorization_context": dict(context),
        "release_authorized": True,
        "release_authorization_path": submission[
            "release_authorization_path"
        ],
        "release_authorization_file_sha256": submission[
            "release_authorization_file_sha256"
        ],
        "release_authorization_sha256": submission[
            "release_authorization_sha256"
        ],
        "release_receipt_path": submission["release_receipt_path"],
        "release_receipt_file_sha256": submission["release_receipt_file_sha256"],
        "release_receipt_sha256": submission["release_receipt_sha256"],
        "receipt_was_durable_before_release": True,
    }


def verify_replacement_attestation_receipt(
    *,
    episode_root: Path,
    receipt_path: Path | None = None,
    require_released: bool | None = None,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(episode_root)))
    _require(root == EPISODE_ROOT, "replacement receipt episode root is not canonical")
    path = receipt_path or (root / RECEIPT_RELATIVE)
    _require(
        Path(os.path.abspath(os.fspath(path))) == root / RECEIPT_RELATIVE,
        "replacement launch receipt path is not canonical",
    )
    receipt = _read_json(
        path, root=root, label="replacement attestation launch receipt", canonical=True
    )
    intent = _read_json(
        root / INTENT_RELATIVE,
        root=root,
        label="replacement attestation launch intent",
        canonical=True,
    )
    held_observation = intent.get("held_observation_immediately_before_submit")
    _require(
        isinstance(held_observation, Mapping),
        "replacement intent lacks its immediate held-job observation",
    )
    validated_held_observation = _validate_obsolete_state(
        {str(key): str(value) for key, value in held_observation.items()},
        submission_user=_effective_user(),
        allow_cancelled=False,
    )
    archive = intent.get("archive")
    _require(isinstance(archive, Mapping), "replacement intent lacks archive identity")
    _require(
        set(archive)
        == {
            "archived_root", "archive_parent", "archive_receipt_path",
            "archive_receipt_file_sha256", "archive_receipt_sha256",
            "manifest_path", "manifest_file_sha256", "manifest_sha256",
            "tree_sha256", "surface_allowlist", "inputs_included",
            "development_outputs_included", "prelaunch_outputs_included",
            "bootstrap_sha256", "parent_first_publication_bound", "complete",
        },
        "replacement intent archive field set changed",
    )
    verified_archive = _archive_identity(
        archive_root=Path(str(archive["archived_root"])),
        receipt_file_sha256=str(archive["archive_receipt_file_sha256"]),
        receipt_sha256=str(archive["archive_receipt_sha256"]),
        manifest_file_sha256=str(archive["manifest_file_sha256"]),
        manifest_sha256=str(archive["manifest_sha256"]),
        tree_sha256=str(archive["tree_sha256"]),
    )
    _require(dict(archive) == verified_archive, "replacement archive identity changed")
    expected_intent = _build_intent(
        episode_root=root,
        archive=verified_archive,
        submission_user=_effective_user(),
        held_observation=validated_held_observation,
        superseded=_superseded_attestation_evidence(
            episode_root=root,
            submission_user=_effective_user(),
        ),
    )
    _require(intent == expected_intent, "replacement launch intent mutated")
    intent_file_sha256 = _sha256_file(
        root / INTENT_RELATIVE, root=root,
        label="replacement attestation launch intent",
    )

    stage_intent = _read_json(
        root / STAGE_INTENT_RELATIVE, root=root,
        label="replacement attestation stage intent", canonical=True,
    )
    _require(
        stage_intent
        == _stage_intent(
            episode_root=root,
            intent=intent,
            intent_file_sha256=intent_file_sha256,
        ),
        "replacement stage intent mutated",
    )
    stage_progress_value = _read_json(
        root / STAGE_PROGRESS_RELATIVE, root=root,
        label="replacement attestation stage progress", canonical=True,
    )
    state, job_id, stage_progress = _verify_stage_progress(
        stage_progress_value, stage_intent=stage_intent
    )
    _require(state == "submitted" and job_id is not None, "replacement was not submitted")
    stage_receipt = _read_json(
        root / STAGE_RECEIPT_RELATIVE, root=root,
        label="replacement attestation stage receipt", canonical=True,
    )
    _require(
        stage_receipt
        == _stage_receipt(
            stage_intent=stage_intent,
            progress=stage_progress,
            job_id=job_id,
        ),
        "replacement stage receipt mutated",
    )
    release_evidence = _verified_release_evidence(
        root=root,
        stage_receipt=stage_receipt,
        require_released=require_released,
    )
    progress = _read_json(
        root / PROGRESS_RELATIVE, root=root,
        label="replacement attestation launch progress", canonical=True,
    )
    _require(
        progress == _progress(intent=intent, stage_receipt=stage_receipt),
        "replacement launch progress mutated",
    )
    _require(
        receipt
        == _receipt(
            intent=intent,
            intent_file_sha256=intent_file_sha256,
            progress=progress,
            stage_receipt=stage_receipt,
        ),
        "replacement launch receipt mutated",
    )
    _require(
        job_id not in {OBSERVED_FINALIZER_JOB_ID, OBSERVED_ATTESTATION_JOB_ID},
        "replacement launch receipt reused a prerequisite job ID",
    )
    return {
        "schema_version": "ep12.replacement_attestation_launch_evidence.v1",
        "launch_intent_path": str(root / INTENT_RELATIVE),
        "launch_intent_file_sha256": intent_file_sha256,
        "launch_intent_sha256": intent["intent_sha256"],
        "launch_receipt_path": str(path),
        "launch_receipt_file_sha256": _sha256_file(
            path, root=root, label="replacement attestation launch receipt"
        ),
        "launch_receipt_sha256": receipt["receipt_sha256"],
        "stage_intent_path": str(root / STAGE_INTENT_RELATIVE),
        "stage_intent_file_sha256": _sha256_file(
            root / STAGE_INTENT_RELATIVE, root=root,
            label="replacement attestation stage intent",
        ),
        "stage_intent_sha256": stage_intent["stage_intent_sha256"],
        "stage_receipt_path": str(root / STAGE_RECEIPT_RELATIVE),
        "stage_receipt_file_sha256": _sha256_file(
            root / STAGE_RECEIPT_RELATIVE, root=root,
            label="replacement attestation stage receipt",
        ),
        "stage_receipt_sha256": stage_receipt["stage_receipt_sha256"],
        **release_evidence,
        "slurm_job_id": job_id,
        "slurm_job_name": stage_receipt["slurm_job_name"],
        "dependency": stage_receipt["dependency"],
        "wrapper": stage_receipt["wrapper"],
        "wrapper_sha256": stage_receipt["wrapper_sha256"],
        "arguments": list(stage_receipt["arguments"]),
        "archive": {
            key: verified_archive[key]
            for key in (
                "archived_root", "archive_receipt_path",
                "archive_receipt_file_sha256", "archive_receipt_sha256",
                "manifest_path", "manifest_file_sha256", "manifest_sha256",
                "tree_sha256", "bootstrap_sha256",
            )
        },
        "terminal_identity_path": receipt["terminal_identity_path"],
        "attestation_output": receipt["attestation_output"],
        "superseded_job_id": OBSERVED_ATTESTATION_JOB_ID,
        "obsolete_hold_file_sha256": HOLD_FILE_SHA256,
        "superseded_result_authoritative": False,
        "authoritative_attestation_job_identity": True,
        "attestation_result_authoritative": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, default=EPISODE_ROOT)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--archive-receipt-file-sha256", required=True)
    parser.add_argument("--archive-receipt-sha256", required=True)
    parser.add_argument("--archive-manifest-file-sha256", required=True)
    parser.add_argument("--archive-manifest-sha256", required=True)
    parser.add_argument("--archive-tree-sha256", required=True)
    arguments = parser.parse_args(argv)
    result = launch_replacement_attestation(
        episode_root=arguments.episode_root,
        archive_root=arguments.archive_root,
        archive_receipt_file_sha256=arguments.archive_receipt_file_sha256,
        archive_receipt_sha256=arguments.archive_receipt_sha256,
        archive_manifest_file_sha256=arguments.archive_manifest_file_sha256,
        archive_manifest_sha256=arguments.archive_manifest_sha256,
        archive_tree_sha256=arguments.archive_tree_sha256,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
