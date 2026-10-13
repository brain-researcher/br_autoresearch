"""Crash-safe unattended EP12 post-36 dependency-DAG controller.

The controller submits every descendant under a user hold, persists all job-ID
receipts, releases B--E while they remain dependency-blocked, and releases the
dynamic replacement attestation R last.  The obsolete job 45332084 is never an
execution dependency and is retained only as authenticated superseded evidence.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .obsolete_attestation_cancel import (
    RECEIPT_RELATIVE as CANCELLATION_RECEIPT_RELATIVE,
    cancel_obsolete_attestation,
    verify_obsolete_attestation_cancellation_receipt,
)
from .policy import canonical_json, digest_object
from .post36_orchestration import (
    STAGE_INTENT_SCHEMA,
    _atomic_json,
    _effective_user,
    _read_bytes,
    _read_json,
    _release_authorization,
    _release_intent,
    _release_progress,
    _release_receipt,
    _release_submitted_stage,
    _sha256_bytes,
    _sha256_file,
    _stage_receipt,
    _stage_submission_evidence,
    _submit_stage_held,
    _verify_stage_progress,
    _write_or_verify,
    default_exact_name_resolver,
    default_job_state_resolver,
    default_releaser,
    default_submitter,
)
from .replacement_attestation_launch import (
    RECEIPT_RELATIVE as REPLACEMENT_RECEIPT_RELATIVE,
    INTENT_RELATIVE as REPLACEMENT_INTENT_RELATIVE,
    STAGE_INTENT_RELATIVE as REPLACEMENT_STAGE_INTENT_RELATIVE,
    STAGE_RECEIPT_RELATIVE as REPLACEMENT_STAGE_RECEIPT_RELATIVE,
    TRANSACTION_RELATIVE as REPLACEMENT_TRANSACTION_RELATIVE,
    _archive_identity,
    launch_replacement_attestation,
    verify_replacement_attestation_receipt,
)
from .secure_transaction import open_regular_lock_at, open_secure_directory
from .synthetic import EPISODE_ROOT


INTENT_SCHEMA = "ep12.post36_unattended_orchestration_intent.v1"
PROGRESS_SCHEMA = "ep12.post36_unattended_orchestration_progress.v1"
RECEIPT_SCHEMA = "ep12.post36_unattended_orchestration_receipt.v1"
TRANSACTION_RELATIVE = Path("outputs/prelaunch/post36_orchestration")
INTENT_RELATIVE = TRANSACTION_RELATIVE / "intent.json"
PROGRESS_RELATIVE = TRANSACTION_RELATIVE / "progress.json"
RECEIPT_RELATIVE = TRANSACTION_RELATIVE / "receipt.json"
START_AUTHORIZATION_RELATIVE = TRANSACTION_RELATIVE / "start_authorization.json"
START_AUTHORIZATION_SCHEMA = "ep12.post36_start_authorization.v1"

STAGE_ORDER = ("post36_probe", "comparison", "publisher", "downstream")
WRAPPERS = {
    "post36_probe": "run_contract_invariance_probe_from_archive.sbatch",
    "comparison": "compare_contract_invariance_from_archive.sbatch",
    "publisher": "publish_scientific_transition_from_archive.sbatch",
    "downstream": "launch_post36_downstream_from_archive.sbatch",
}

LEGACY_JOB_ID = "45363147"
LEGACY_EXIT_CODE = "0:0"
LEGACY_ROOT_RELATIVE = Path(
    "outputs/prelaunch/contract_transition_probe/legacy_v1"
)
LEGACY_FIXED = {
    "launch_receipt": (
        "launch_receipt_45363147.json",
        "e2780d6e82896314e7f021c0e9e8fb04580527482ccfd274a12dfcafc7f1e661",
        "receipt_sha256",
        "796494d2a9f6c2ca17ab813515d41e408ad0f918ac1ee123a48e7908a9747167",
    ),
    "completion": (
        "completion_45363147.json",
        "6fdfe9ec8f74ddfb25a7fd77f6c87d64588fb011de1342a59fa1f6328409113f",
        "completion_sha256",
        "ef051515813fc10c14eaa6edb655aade3c7b52231ba435926a45425c32e60ea4",
    ),
    "probe": (
        "probe.json",
        "216b6ff5ecb5e71a0c437b6ab603ffd655913fffc8bd5707a783576e0349526d",
        "artifact_sha256",
        "b78c43b9ebb5cc53c38c8bc2cde2b7347df02f6bd50a34fdbf4c244b2ab438d3",
    ),
}
LEGACY_SCIENTIFIC_PAYLOAD_SHA256 = (
    "19e3100cad6f20bad7dfafa348d035d9df16ff59f3727bb3fb9ff62e11f5f678"
)

INCIDENT_RELATIVE = Path(
    "outputs/prelaunch/exposure_incidents/"
    "development_outcome_terminal_exposure_20260926T061201-0700.json"
)
INCIDENT_FILE_SHA256 = (
    "0ab9a2070643ac37f10f05a5899417a96d9cdf7b54923c7ef1442be6d248a6c1"
)
INCIDENT_SHA256 = (
    "37f5bdfcbecd8c522e994473522502ca708e9afd057f9e7629b6408e5c1c1618"
)

POST36_PROBE_RELATIVE = Path(
    "outputs/prelaunch/contract_transition_probe/post36_v1/probe.json"
)
COMPARISON_RELATIVE = Path(
    "outputs/prelaunch/contract_transition_probe/comparison_v1/comparison.json"
)


class Post36UnattendedControllerError(RuntimeError):
    """The exact-once post-36 DAG is incomplete or unauthenticated."""


Submitter = Callable[[Sequence[str]], str]
JobResolver = Callable[[str, str], Mapping[str, str] | None]
JobStateResolver = Callable[[str, str, str], Mapping[str, str]]
Releaser = Callable[[Sequence[str]], str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Post36UnattendedControllerError(message)


def _json_bytes(path: Path, *, root: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    encoded = _read_bytes(path, root=root, label=label)
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Post36UnattendedControllerError(f"malformed {label}") from exc
    _require(isinstance(value, Mapping), f"{label} is not an object")
    return encoded, dict(value)


def _legacy_evidence(root: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for label, (name, file_sha, semantic_field, semantic_sha) in LEGACY_FIXED.items():
        path = root / LEGACY_ROOT_RELATIVE / name
        encoded, value = _json_bytes(path, root=root, label=f"legacy {label}")
        _require(hashlib.sha256(encoded).hexdigest() == file_sha,
                 f"legacy {label} file hash changed")
        _require(value.get(semantic_field) == semantic_sha,
                 f"legacy {label} semantic identity changed")
        records[label] = {
            "path": str(path),
            "file_sha256": file_sha,
            "semantic_field": semantic_field,
            "semantic_sha256": semantic_sha,
        }
    probe = _json_bytes(
        root / LEGACY_ROOT_RELATIVE / "probe.json",
        root=root,
        label="legacy probe",
    )[1]
    completion = _json_bytes(
        root / LEGACY_ROOT_RELATIVE / "completion_45363147.json",
        root=root,
        label="legacy completion",
    )[1]
    _require(
        probe.get("schema_version")
        == "ep12.contract_transition_invariance_probe.v1"
        and probe.get("phase") == "legacy"
        and probe.get("scientific_payload_sha256")
        == LEGACY_SCIENTIFIC_PAYLOAD_SHA256
        and completion.get("slurm_job_id") == LEGACY_JOB_ID
        and completion.get("state") == "COMPLETED"
        and completion.get("exit_code") == LEGACY_EXIT_CODE
        and completion.get("artifact_verifier_passed") is True
        and completion.get("annotation_or_outcome_files_accessed") is False
        and completion.get("real_connectivity_accessed") is False,
        "legacy probe authority facts changed",
    )
    return {
        "schema_version": "ep12.legacy_contract_probe_evidence.v1",
        "slurm_job_id": LEGACY_JOB_ID,
        "artifacts": records,
        "scientific_payload_sha256": LEGACY_SCIENTIFIC_PAYLOAD_SHA256,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
    }


def _incident_evidence(root: Path) -> dict[str, Any]:
    path = root / INCIDENT_RELATIVE
    encoded, value = _json_bytes(path, root=root, label="operator exposure incident")
    _require(hashlib.sha256(encoded).hexdigest() == INCIDENT_FILE_SHA256,
             "operator exposure incident file hash changed")
    core = {key: item for key, item in value.items() if key != "incident_sha256"}
    _require(
        value.get("schema_version") == "ep12.operator_exposure_incident.v1"
        and value.get("incident_sha256") == INCIDENT_SHA256
        and digest_object(core) == INCIDENT_SHA256
        and value.get("development_connectivity_outcome_values_displayed") is True
        and value.get("development_connectivity_outcome_values_analyzed") is False
        and value.get(
            "development_connectivity_outcome_values_used_for_scientific_decisions"
        ) is False
        and value.get("scientific_contract_or_threshold_changed_after_display") is False
        and value.get("final_connectivity_accessed") is False,
        "operator exposure incident facts changed",
    )
    return {
        "path": str(path),
        "file_sha256": INCIDENT_FILE_SHA256,
        "incident_sha256": INCIDENT_SHA256,
        "operator_development_outcome_values_displayed": True,
        "values_used_for_transition_decisions": False,
        "scientific_contract_or_threshold_changed": False,
        "final_connectivity_accessed": False,
    }


def _stable_replacement(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "launch_intent_path", "launch_intent_file_sha256", "launch_intent_sha256",
        "launch_receipt_path", "launch_receipt_file_sha256", "launch_receipt_sha256",
        "stage_intent_path", "stage_intent_file_sha256", "stage_intent_sha256",
        "stage_receipt_path", "stage_receipt_file_sha256", "stage_receipt_sha256",
        "slurm_job_id", "slurm_job_name", "dependency", "wrapper", "wrapper_sha256",
        "arguments", "archive", "terminal_identity_path", "attestation_output",
        "superseded_job_id", "obsolete_hold_file_sha256",
    )
    return {key: value[key] for key in keys}


def _controller_source(root: Path, archive: Mapping[str, Any]) -> dict[str, Any]:
    archived = (
        Path(str(archive["archived_root"]))
        / "outputs/executor/ep12_executor/post36_unattended_controller.py"
    )
    active = Path(os.path.abspath(__file__))
    active_executor = active.parent.parent
    _require(
        active.relative_to(active_executor).as_posix()
        == "ep12_executor/post36_unattended_controller.py",
        "active post36 controller path is outside an executor root",
    )
    live_hash = _sha256_file(
        active, root=active_executor, label="active post36 controller"
    )
    archive_hash = _sha256_file(
        archived, root=Path(str(archive["archived_root"])),
        label="archived post36 controller",
    )
    _require(live_hash == archive_hash,
             "active post36 controller differs from reviewed archive")
    return {
        "archived_path": str(archived),
        "file_sha256": archive_hash,
        "active_bytes_match_archive": True,
    }


def _intent(
    *, root: Path, replacement: Mapping[str, Any],
    cancellation: Mapping[str, Any], submission_user: str,
) -> dict[str, Any]:
    archive = replacement["archive"]
    core = {
        "schema_version": INTENT_SCHEMA,
        "episode_root": str(root),
        "submission_user": submission_user,
        "archive": dict(archive),
        "controller_source": _controller_source(root, archive),
        "replacement_launch": _stable_replacement(replacement),
        "obsolete_cancellation": dict(cancellation),
        "legacy_probe": _legacy_evidence(root),
        "operator_exposure_incident": _incident_evidence(root),
        "stage_order": list(STAGE_ORDER),
        "stage_wrappers": dict(WRAPPERS),
        "post36_probe_output": str(root / POST36_PROBE_RELATIVE),
        "comparison_output": str(root / COMPARISON_RELATIVE),
        "replacement_is_dynamic_authoritative_job": True,
        "obsolete_job_is_execution_dependency": False,
        "all_descendants_submitted_held_before_replacement_release": True,
        "connectivity_outcome_values_read_by_controller": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "intent_sha256": digest_object(core)}


def _pre_cancel_preflight(
    root: Path, archive: Mapping[str, Any], *, require_outputs_absent: bool
) -> None:
    """Exhaust immutable/path checks before the obsolete cancellation."""

    _controller_source(root, archive)
    _legacy_evidence(root)
    _incident_evidence(root)
    archive_root = Path(str(archive["archived_root"]))
    for stage in STAGE_ORDER:
        wrapper = archive_root / "outputs/executor/slurm" / WRAPPERS[stage]
        _sha256_file(wrapper, root=archive_root,
                     label=f"archived {stage} wrapper preflight")
    if require_outputs_absent:
        for relative in (
            POST36_PROBE_RELATIVE,
            COMPARISON_RELATIVE,
            Path("outputs/development_run002/observed_attestation"),
            Path("outputs/prelaunch/replacement_attestation_launch/terminal_identity.json"),
        ):
            _require(
                not os.path.lexists(root / relative),
                f"pre-cancel output collision: {relative}",
            )


def _common_arguments(
    *, root: Path, intent: Mapping[str, Any], intent_file_sha256: str,
    stage: str,
) -> list[str]:
    archive = intent["archive"]
    return [
        str(root), str(archive["archived_root"]),
        str(archive["archive_receipt_file_sha256"]),
        str(archive["archive_receipt_sha256"]),
        str(archive["manifest_file_sha256"]), str(archive["manifest_sha256"]),
        str(archive["tree_sha256"]), str(root / INTENT_RELATIVE),
        intent_file_sha256, str(intent["intent_sha256"]),
        str(root / TRANSACTION_RELATIVE / "stages" / stage / "receipt.json"),
    ]


def build_stage_intent(
    *, root: Path, intent: Mapping[str, Any], intent_file_sha256: str,
    stage: str, predecessor_job_id: str,
) -> dict[str, Any]:
    _require(stage in STAGE_ORDER, "unknown post36 stage")
    _require(predecessor_job_id.isdigit() and int(predecessor_job_id) > 0,
             "post36 predecessor job ID is invalid")
    archive_root = Path(str(intent["archive"]["archived_root"]))
    wrapper = archive_root / "outputs/executor/slurm" / WRAPPERS[stage]
    wrapper_sha = _sha256_file(
        wrapper, root=archive_root, label=f"archived {stage} wrapper"
    )
    arguments = _common_arguments(
        root=root, intent=intent, intent_file_sha256=intent_file_sha256,
        stage=stage,
    )
    if stage == "post36_probe":
        arguments.append(str(root / POST36_PROBE_RELATIVE))
    elif stage == "comparison":
        arguments.extend([
            str(root / LEGACY_ROOT_RELATIVE / "probe.json"),
            str(root / POST36_PROBE_RELATIVE), str(root / COMPARISON_RELATIVE),
        ])
    elif stage == "publisher":
        arguments.extend([
            str(root / COMPARISON_RELATIVE),
            str(intent["replacement_launch"]["terminal_identity_path"]),
            str(root / START_AUTHORIZATION_RELATIVE),
            str(root / CANCELLATION_RECEIPT_RELATIVE),
        ])
    else:
        arguments.append(predecessor_job_id)
    core = {
        "schema_version": STAGE_INTENT_SCHEMA,
        "orchestration_intent_sha256": intent["intent_sha256"],
        "stage": stage,
        "submission_user": intent["submission_user"],
        "predecessor_job_id": predecessor_job_id,
        "dependency": f"afterok:{predecessor_job_id}",
        "kill_on_invalid_dependency": True,
        "submission_barrier": "user_hold_receipt_before_release_v1",
        "wrapper": str(wrapper),
        "wrapper_sha256": wrapper_sha,
        "arguments": arguments,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    token = digest_object(core)[:24]
    named = {**core, "job_name": f"ep12-{stage.replace('_', '-')}-{token}"}
    return {**named, "stage_intent_sha256": digest_object(named)}


def _progress(
    *, intent: Mapping[str, Any], state: str,
    stage_job_ids: Mapping[str, str], released_stages: Sequence[str],
    replacement_released: bool,
) -> dict[str, Any]:
    _require(state in {"children_held", "children_released", "replacement_released"},
             "invalid post36 controller progress")
    core = {
        "schema_version": PROGRESS_SCHEMA,
        "intent_sha256": intent["intent_sha256"],
        "state": state,
        "stage_job_ids": dict(stage_job_ids),
        "released_stages": list(released_stages),
        "replacement_released": replacement_released,
        "final_connectivity_accessed": False,
    }
    return {**core, "progress_sha256": digest_object(core)}


def _write_progress(root: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = root / PROGRESS_RELATIVE
    try:
        prior = _read_json(path, root=root, label="post36 controller progress", canonical=True)
    except FileNotFoundError:
        prior = None
    if prior is not None:
        core = {key: item for key, item in prior.items() if key != "progress_sha256"}
        _require(prior.get("progress_sha256") == digest_object(core),
                 "post36 controller progress mutated")
        rank = {"children_held": 0, "children_released": 1, "replacement_released": 2}
        _require(rank[str(value["state"])] >= rank[str(prior["state"])],
                 "post36 controller progress regressed")
    _atomic_json(path, value, root=root)
    return dict(value)


def _verify_stage(
    *, root: Path, transaction: Path, stage_intent: Mapping[str, Any]
) -> dict[str, Any]:
    stage = str(stage_intent["stage"])
    stage_dir = transaction / "stages" / stage
    persisted_intent = _read_json(
        stage_dir / "intent.json", root=root, label=f"{stage} intent", canonical=True
    )
    _require(persisted_intent == dict(stage_intent), f"{stage} intent mutated")
    state, job_id, progress = _verify_stage_progress(
        _read_json(stage_dir / "progress.json", root=root,
                   label=f"{stage} progress", canonical=True),
        stage_intent=stage_intent,
    )
    _require(state == "submitted" and job_id is not None, f"{stage} not submitted")
    receipt = _read_json(
        stage_dir / "receipt.json", root=root, label=f"{stage} receipt", canonical=True
    )
    _require(receipt == _stage_receipt(
        stage_intent=stage_intent, progress=progress, job_id=job_id
    ), f"{stage} receipt mutated")
    release_intent = _read_json(
        stage_dir / "release_intent.json", root=root,
        label=f"{stage} release intent", canonical=True,
    )
    expected_release_intent = _release_intent(
        stage_receipt=receipt,
        stage_receipt_file_sha256=_sha256_file(
            stage_dir / "receipt.json", root=root, label=f"{stage} receipt"
        ),
    )
    _require(release_intent == expected_release_intent,
             f"{stage} release intent mutated")
    authorization = _read_json(
        stage_dir / "release_authorization.json", root=root,
        label=f"{stage} release authorization", canonical=True,
    )
    _require(authorization == _release_authorization(release_intent=release_intent),
             f"{stage} release authorization mutated")
    release_progress = _read_json(
        stage_dir / "release_progress.json", root=root,
        label=f"{stage} release progress", canonical=True,
    )
    _require(release_progress == _release_progress(
        release_intent=release_intent, state="released"
    ), f"{stage} is not durably released")
    release_receipt = _read_json(
        stage_dir / "release_receipt.json", root=root,
        label=f"{stage} release receipt", canonical=True,
    )
    _require(release_receipt == _release_receipt(
        release_intent=release_intent, release_progress=release_progress
    ), f"{stage} release receipt mutated")
    return _stage_submission_evidence(
        stage_dir=stage_dir, root=root, stage_receipt=receipt,
        release_receipt=release_receipt,
    )


def _start_authorization(
    *, root: Path, intent: Mapping[str, Any], intent_file_sha256: str,
    replacement: Mapping[str, Any], cancellation: Mapping[str, Any],
    stages: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    stage_bindings: dict[str, Any] = {}
    for stage in STAGE_ORDER:
        value = stages[stage]
        stage_dir = root / TRANSACTION_RELATIVE / "stages" / stage
        stage_bindings[stage] = {
            "slurm_job_id": value["slurm_job_id"],
            "slurm_job_name": value["slurm_job_name"],
            "dependency": value["dependency"],
            "stage_receipt_path": str(stage_dir / "receipt.json"),
            "stage_receipt_file_sha256": _sha256_file(
                stage_dir / "receipt.json", root=root,
                label=f"{stage} stage receipt",
            ),
            "stage_receipt_sha256": value["stage_receipt_sha256"],
            "release_authorization_file_sha256": value[
                "release_authorization_file_sha256"
            ],
            "release_authorization_sha256": value[
                "release_authorization_sha256"
            ],
            "release_receipt_file_sha256": value[
                "release_receipt_file_sha256"
            ],
            "release_receipt_sha256": value["release_receipt_sha256"],
        }
    core = {
        "schema_version": START_AUTHORIZATION_SCHEMA,
        "intent_path": str(root / INTENT_RELATIVE),
        "intent_file_sha256": intent_file_sha256,
        "intent_sha256": intent["intent_sha256"],
        "replacement_job_id": replacement["slurm_job_id"],
        "replacement_stage_receipt_path": replacement["stage_receipt_path"],
        "replacement_stage_receipt_file_sha256": replacement[
            "stage_receipt_file_sha256"
        ],
        "replacement_stage_receipt_sha256": replacement[
            "stage_receipt_sha256"
        ],
        "obsolete_cancellation_receipt_path": cancellation[
            "cancellation_receipt_path"
        ],
        "obsolete_cancellation_receipt_file_sha256": cancellation[
            "cancellation_receipt_file_sha256"
        ],
        "obsolete_cancellation_receipt_sha256": cancellation[
            "cancellation_receipt_sha256"
        ],
        "stages": stage_bindings,
        "all_child_stage_and_release_receipts_durable": True,
        "obsolete_job_terminally_cancelled": True,
        "replacement_may_be_released": True,
        "obsolete_job_is_execution_dependency": False,
        "values_used_for_transition_decisions": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "start_authorization_sha256": digest_object(core)}


def _replacement_release_context(
    *, root: Path, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    path = root / START_AUTHORIZATION_RELATIVE
    return {
        "schema_version": "ep12.post36_replacement_release_context.v1",
        "start_authorization_path": str(path),
        "start_authorization_file_sha256": _sha256_file(
            path, root=root, label="post36 start authorization"
        ),
        "start_authorization_sha256": authorization[
            "start_authorization_sha256"
        ],
    }


def _receipt(
    *, root: Path, intent: Mapping[str, Any], intent_file_sha256: str,
    progress: Mapping[str, Any], replacement: Mapping[str, Any],
    cancellation: Mapping[str, Any], stages: Mapping[str, Mapping[str, Any]],
    start_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": RECEIPT_SCHEMA,
        "intent_path": str(root / INTENT_RELATIVE),
        "intent_file_sha256": intent_file_sha256,
        "intent_sha256": intent["intent_sha256"],
        "progress_sha256": progress["progress_sha256"],
        "replacement_job_id": replacement["slurm_job_id"],
        "replacement_launch_receipt_file_sha256": replacement[
            "launch_receipt_file_sha256"
        ],
        "replacement_launch_receipt_sha256": replacement[
            "launch_receipt_sha256"
        ],
        "replacement_release_authorization_file_sha256": replacement[
            "release_authorization_file_sha256"
        ],
        "replacement_release_authorization_sha256": replacement[
            "release_authorization_sha256"
        ],
        "replacement_release_receipt_file_sha256": replacement[
            "release_receipt_file_sha256"
        ],
        "replacement_release_receipt_sha256": replacement[
            "release_receipt_sha256"
        ],
        "obsolete_cancellation_receipt_file_sha256": cancellation[
            "cancellation_receipt_file_sha256"
        ],
        "obsolete_cancellation_receipt_sha256": cancellation[
            "cancellation_receipt_sha256"
        ],
        "start_authorization_path": str(root / START_AUTHORIZATION_RELATIVE),
        "start_authorization_file_sha256": _sha256_file(
            root / START_AUTHORIZATION_RELATIVE, root=root,
            label="post36 start authorization",
        ),
        "start_authorization_sha256": start_authorization[
            "start_authorization_sha256"
        ],
        "stages": {key: dict(stages[key]) for key in STAGE_ORDER},
        "all_child_receipts_durable_before_replacement_release": True,
        "all_child_release_receipts_durable_before_replacement_release": True,
        "replacement_released_last": True,
        "obsolete_job_is_execution_dependency": False,
        "operator_development_outcome_values_displayed": True,
        "values_used_for_transition_decisions": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "receipt_sha256": digest_object(core)}


def orchestrate_post36_unattended(
    *, episode_root: Path, archive_root: Path,
    archive_receipt_file_sha256: str, archive_receipt_sha256: str,
    archive_manifest_file_sha256: str, archive_manifest_sha256: str,
    archive_tree_sha256: str,
    submitter: Submitter = default_submitter,
    job_resolver: JobResolver = default_exact_name_resolver,
    job_state_resolver: JobStateResolver = default_job_state_resolver,
    releaser: Releaser = default_releaser,
    replacement_kwargs: Mapping[str, Any] | None = None,
    cancellation_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(episode_root)))
    _require(root == EPISODE_ROOT, "post36 controller episode root is not canonical")
    try:
        _read_json(root / RECEIPT_RELATIVE, root=root,
                   label="post36 controller receipt", canonical=True)
    except FileNotFoundError:
        pass
    else:
        return verify_post36_orchestration_receipt(episode_root=root)

    archive = _archive_identity(
        archive_root=archive_root,
        receipt_file_sha256=archive_receipt_file_sha256,
        receipt_sha256=archive_receipt_sha256,
        manifest_file_sha256=archive_manifest_file_sha256,
        manifest_sha256=archive_manifest_sha256,
        tree_sha256=archive_tree_sha256,
    )
    try:
        _read_json(
            root / CANCELLATION_RECEIPT_RELATIVE, root=root,
            label="obsolete cancellation receipt", canonical=True,
        )
    except FileNotFoundError:
        cancellation_exists = False
    else:
        cancellation_exists = True
    _pre_cancel_preflight(
        root, archive, require_outputs_absent=not cancellation_exists
    )

    launch_replacement_attestation(
        episode_root=root, archive_root=archive_root,
        archive_receipt_file_sha256=archive_receipt_file_sha256,
        archive_receipt_sha256=archive_receipt_sha256,
        archive_manifest_file_sha256=archive_manifest_file_sha256,
        archive_manifest_sha256=archive_manifest_sha256,
        archive_tree_sha256=archive_tree_sha256,
        **dict(replacement_kwargs or {}),
    )
    replacement = verify_replacement_attestation_receipt(
        episode_root=root, require_released=None
    )
    cancel_obsolete_attestation(
        episode_root=root, **dict(cancellation_kwargs or {})
    )
    cancellation = verify_obsolete_attestation_cancellation_receipt(
        episode_root=root
    )

    transaction, transaction_fd = open_secure_directory(
        root / TRANSACTION_RELATIVE, root=root, create=True
    )
    lock_fd = open_regular_lock_at(transaction_fd, ".lock")
    os.close(transaction_fd)
    with os.fdopen(lock_fd, "a+b") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        try:
            intent = _write_or_verify(
                root / INTENT_RELATIVE,
                _intent(
                    root=root, replacement=replacement,
                    cancellation=cancellation, submission_user=_effective_user(),
                ),
                root=root, label="post36 controller intent",
            )
            intent_file_sha = _sha256_file(
                root / INTENT_RELATIVE, root=root, label="post36 controller intent"
            )
            held: dict[str, dict[str, Any]] = {}
            predecessor = str(replacement["slurm_job_id"])
            stage_intents: dict[str, dict[str, Any]] = {}
            for stage in STAGE_ORDER:
                stage_intent = build_stage_intent(
                    root=root, intent=intent, intent_file_sha256=intent_file_sha,
                    stage=stage, predecessor_job_id=predecessor,
                )
                stage_intents[stage] = stage_intent
                held[stage] = _submit_stage_held(
                    transaction_root=transaction, root=root,
                    stage_intent=stage_intent, submitter=submitter,
                    job_resolver=job_resolver,
                    job_state_resolver=job_state_resolver,
                )
                predecessor = str(held[stage]["slurm_job_id"])
            job_ids = {stage: str(held[stage]["slurm_job_id"])
                       for stage in STAGE_ORDER}
            _write_progress(
                root,
                _progress(
                    intent=intent, state="children_held", stage_job_ids=job_ids,
                    released_stages=[], replacement_released=False,
                ),
            )
            released: dict[str, dict[str, Any]] = {}
            for stage in STAGE_ORDER:
                released[stage] = _release_submitted_stage(
                    transaction_root=transaction, root=root,
                    stage_intent=stage_intents[stage], stage_receipt=held[stage],
                    job_state_resolver=job_state_resolver, releaser=releaser,
                )
            _write_progress(
                root,
                _progress(
                    intent=intent, state="children_released", stage_job_ids=job_ids,
                    released_stages=STAGE_ORDER, replacement_released=False,
                ),
            )
            start_authorization = _write_or_verify(
                root / START_AUTHORIZATION_RELATIVE,
                _start_authorization(
                    root=root, intent=intent,
                    intent_file_sha256=intent_file_sha,
                    replacement=replacement, cancellation=cancellation,
                    stages=released,
                ),
                root=root, label="post36 start authorization",
            )
            replacement_stage_intent = _read_json(
                root / REPLACEMENT_STAGE_INTENT_RELATIVE, root=root,
                label="replacement stage intent", canonical=True,
            )
            replacement_stage_receipt = _read_json(
                root / REPLACEMENT_STAGE_RECEIPT_RELATIVE, root=root,
                label="replacement stage receipt", canonical=True,
            )
            _release_submitted_stage(
                transaction_root=root / REPLACEMENT_TRANSACTION_RELATIVE,
                root=root, stage_intent=replacement_stage_intent,
                stage_receipt=replacement_stage_receipt,
                job_state_resolver=job_state_resolver, releaser=releaser,
                authorization_context=_replacement_release_context(
                    root=root, authorization=start_authorization
                ),
            )
            replacement = verify_replacement_attestation_receipt(
                episode_root=root, require_released=True
            )
            progress = _write_progress(
                root,
                _progress(
                    intent=intent, state="replacement_released",
                    stage_job_ids=job_ids, released_stages=STAGE_ORDER,
                    replacement_released=True,
                ),
            )
            receipt = _receipt(
                root=root, intent=intent, intent_file_sha256=intent_file_sha,
                progress=progress, replacement=replacement,
                cancellation=cancellation, stages=released,
                start_authorization=start_authorization,
            )
            _write_or_verify(
                root / RECEIPT_RELATIVE, receipt, root=root,
                label="post36 controller receipt",
            )
            return verify_post36_orchestration_receipt(episode_root=root)
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


def verify_post36_orchestration_receipt(
    *, episode_root: Path, receipt_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(episode_root)))
    _require(root == EPISODE_ROOT, "post36 verifier episode root is not canonical")
    path = receipt_path or (root / RECEIPT_RELATIVE)
    _require(Path(os.path.abspath(os.fspath(path))) == root / RECEIPT_RELATIVE,
             "post36 receipt path is not canonical")
    replacement = verify_replacement_attestation_receipt(
        episode_root=root, require_released=True
    )
    cancellation = verify_obsolete_attestation_cancellation_receipt(
        episode_root=root
    )
    intent = _read_json(root / INTENT_RELATIVE, root=root,
                        label="post36 controller intent", canonical=True)
    _require(intent == _intent(
        root=root, replacement=replacement, cancellation=cancellation,
        submission_user=_effective_user(),
    ), "post36 controller intent mutated")
    intent_file_sha = _sha256_file(
        root / INTENT_RELATIVE, root=root, label="post36 controller intent"
    )
    transaction = root / TRANSACTION_RELATIVE
    stages: dict[str, dict[str, Any]] = {}
    predecessor = str(replacement["slurm_job_id"])
    for stage in STAGE_ORDER:
        stage_intent = build_stage_intent(
            root=root, intent=intent, intent_file_sha256=intent_file_sha,
            stage=stage, predecessor_job_id=predecessor,
        )
        stages[stage] = _verify_stage(
            root=root, transaction=transaction, stage_intent=stage_intent
        )
        predecessor = str(stages[stage]["slurm_job_id"])
    start_authorization = _read_json(
        root / START_AUTHORIZATION_RELATIVE, root=root,
        label="post36 start authorization", canonical=True,
    )
    _require(
        start_authorization == _start_authorization(
            root=root, intent=intent, intent_file_sha256=intent_file_sha,
            replacement=replacement, cancellation=cancellation, stages=stages,
        ),
        "post36 start authorization mutated",
    )
    job_ids = {stage: str(stages[stage]["slurm_job_id"])
               for stage in STAGE_ORDER}
    progress = _read_json(
        root / PROGRESS_RELATIVE, root=root,
        label="post36 controller progress", canonical=True,
    )
    _require(progress == _progress(
        intent=intent, state="replacement_released", stage_job_ids=job_ids,
        released_stages=STAGE_ORDER, replacement_released=True,
    ), "post36 controller final progress mutated")
    receipt = _read_json(path, root=root,
                         label="post36 controller receipt", canonical=True)
    expected = _receipt(
        root=root, intent=intent, intent_file_sha256=intent_file_sha,
        progress=progress, replacement=replacement,
        cancellation=cancellation, stages=stages,
        start_authorization=start_authorization,
    )
    _require(receipt == expected, "post36 controller receipt mutated")
    all_dependencies = [str(stages[stage]["dependency"]) for stage in STAGE_ORDER]
    _require(all("45332084" not in dependency for dependency in all_dependencies),
             "obsolete attestation appears in an execution dependency")
    return {
        "schema_version": "ep12.post36_unattended_orchestration_evidence.v1",
        "intent_path": str(root / INTENT_RELATIVE),
        "intent_file_sha256": intent_file_sha,
        "intent_sha256": intent["intent_sha256"],
        "receipt_path": str(path),
        "receipt_file_sha256": _sha256_file(
            path, root=root, label="post36 controller receipt"
        ),
        "receipt_sha256": receipt["receipt_sha256"],
        "replacement": dict(replacement),
        "obsolete_cancellation": dict(cancellation),
        "stages": stages,
        "legacy_probe": dict(intent["legacy_probe"]),
        "operator_exposure_incident": dict(intent["operator_exposure_incident"]),
        "start_authorization": {
            "path": str(root / START_AUTHORIZATION_RELATIVE),
            "file_sha256": _sha256_file(
                root / START_AUTHORIZATION_RELATIVE, root=root,
                label="post36 start authorization",
            ),
            "start_authorization_sha256": start_authorization[
                "start_authorization_sha256"
            ],
        },
        "obsolete_job_is_execution_dependency": False,
        "operator_development_outcome_values_displayed": True,
        "values_used_for_transition_decisions": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }


def verify_post36_start_authorization(
    *, episode_root: Path, authorization_path: Path | None = None,
) -> dict[str, Any]:
    """Authenticate the immutable B--E/R start barrier before top receipt exists."""

    root = Path(os.path.abspath(os.fspath(episode_root)))
    _require(root == EPISODE_ROOT, "post36 start verifier root is not canonical")
    path = authorization_path or (root / START_AUTHORIZATION_RELATIVE)
    _require(
        Path(os.path.abspath(os.fspath(path))) == root / START_AUTHORIZATION_RELATIVE,
        "post36 start authorization path is not canonical",
    )
    replacement = verify_replacement_attestation_receipt(
        episode_root=root, require_released=None
    )
    cancellation = verify_obsolete_attestation_cancellation_receipt(
        episode_root=root
    )
    intent = _read_json(
        root / INTENT_RELATIVE, root=root,
        label="post36 controller intent", canonical=True,
    )
    _require(
        intent == _intent(
            root=root, replacement=replacement, cancellation=cancellation,
            submission_user=_effective_user(),
        ),
        "post36 controller intent mutated",
    )
    intent_file_sha = _sha256_file(
        root / INTENT_RELATIVE, root=root, label="post36 controller intent"
    )
    stages: dict[str, dict[str, Any]] = {}
    predecessor = str(replacement["slurm_job_id"])
    for stage in STAGE_ORDER:
        stage_intent = build_stage_intent(
            root=root, intent=intent, intent_file_sha256=intent_file_sha,
            stage=stage, predecessor_job_id=predecessor,
        )
        stages[stage] = _verify_stage(
            root=root, transaction=root / TRANSACTION_RELATIVE,
            stage_intent=stage_intent,
        )
        predecessor = str(stages[stage]["slurm_job_id"])
    authorization = _read_json(
        path, root=root, label="post36 start authorization", canonical=True
    )
    _require(
        authorization == _start_authorization(
            root=root, intent=intent, intent_file_sha256=intent_file_sha,
            replacement=replacement, cancellation=cancellation, stages=stages,
        ),
        "post36 start authorization mutated",
    )
    return {
        "schema_version": "ep12.post36_start_authorization_evidence.v1",
        "intent_path": str(root / INTENT_RELATIVE),
        "intent_file_sha256": intent_file_sha,
        "intent_sha256": intent["intent_sha256"],
        "authorization_path": str(path),
        "authorization_file_sha256": _sha256_file(
            path, root=root, label="post36 start authorization"
        ),
        "start_authorization_sha256": authorization[
            "start_authorization_sha256"
        ],
        "replacement": dict(replacement),
        "obsolete_cancellation": dict(cancellation),
        "stages": stages,
        "legacy_probe": dict(intent["legacy_probe"]),
        "operator_exposure_incident": dict(intent["operator_exposure_incident"]),
        "obsolete_job_is_execution_dependency": False,
        "values_used_for_transition_decisions": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
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
    result = orchestrate_post36_unattended(
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
