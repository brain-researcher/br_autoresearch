"""Exact-once archive-bound test, qualification, and readiness launch chain.

The downstream coordinator submits every child under a user hold, persists
accepted-job and release receipts, writes an immutable start authorization,
and writes the final nested receipt before it exits.  The first child uses an
afterok dependency on that coordinator, so it cannot start before the runtime
authority is durable.
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .policy import canonical_json, digest_object
from .post36_orchestration import (
    JobResolver,
    JobStateResolver,
    OBSERVED_FINALIZER_JOB_ID,
    Releaser,
    Submitter,
    _effective_user,
    _read_json,
    _release_authorization,
    _release_intent,
    _release_progress,
    _release_receipt,
    _release_submitted_stage,
    _sha256_file,
    _stage_receipt,
    _submit_stage_held,
    _verify_stage_progress,
    _write_or_verify,
    default_exact_name_resolver,
    default_job_state_resolver,
    default_releaser,
    default_submitter,
)
from .post36_unattended_controller import verify_post36_start_authorization
from .secure_transaction import open_regular_lock_at, open_secure_directory
from .synthetic import EPISODE_ROOT


INTENT_SCHEMA = "ep12.archive_nested_stage_intent.v1"
START_AUTHORIZATION_SCHEMA = "ep12.archive_nested_start_authorization.v1"
RECEIPT_SCHEMA = "ep12.archive_nested_stage_receipt.v1"
EVIDENCE_SCHEMA = "ep12.archive_nested_stage_evidence.v1"
TRANSACTION_RELATIVE = Path("outputs/prelaunch/archive_nested_stages")
INTENT_RELATIVE = TRANSACTION_RELATIVE / "intent.json"
START_AUTHORIZATION_RELATIVE = TRANSACTION_RELATIVE / "start_authorization.json"
RECEIPT_RELATIVE = TRANSACTION_RELATIVE / "receipt.json"
STAGE_ORDER = (
    "synthetic_tests",
    "scientific_qualification",
    "production_readiness",
)
WRAPPERS = {
    "synthetic_tests": "test_executor.sbatch",
    "scientific_qualification": "qualify_methods.sbatch",
    "production_readiness": "check_production_readiness.sbatch",
}
_TRANSITION_FIELDS = (
    "transition_manifest_path",
    "transition_manifest_file_sha256",
    "transition_receipt_file_sha256",
    "transition_receipt_sha256",
    "observed_prefix_contract_sha256",
    "active_contract_sha256",
    "active_qualification_spec_sha256",
    "active_scientific_models_sha256",
)


class ArchiveNestedStageError(RuntimeError):
    """The archive-bound nested chain is incomplete, ambiguous, or mutated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchiveNestedStageError(message)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_paths(root: Path) -> dict[str, Path]:
    return {
        "intent": root / INTENT_RELATIVE,
        "start_authorization": root / START_AUTHORIZATION_RELATIVE,
        "receipt": root / RECEIPT_RELATIVE,
        "qualification_launch": (
            root
            / "outputs/prelaunch/"
            "full_synthetic_qualification_launch_qualification003.json"
        ),
        "qualification_dir": (
            root
            / "outputs/prelaunch/scientific_qualification_qualification003"
        ),
        "transition_manifest": (
            root
            / "outputs/prelaunch/scientific_contract_transition/manifest.json"
        ),
        "controller_start_authorization": (
            root
            / "outputs/prelaunch/post36_orchestration/start_authorization.json"
        ),
    }


def _archive_arguments(root: Path, archive: Mapping[str, Any]) -> list[str]:
    return [
        str(root),
        str(archive["archived_root"]),
        str(archive["archive_receipt_file_sha256"]),
        str(archive["archive_receipt_sha256"]),
        str(archive["manifest_file_sha256"]),
        str(archive["manifest_sha256"]),
        str(archive["tree_sha256"]),
    ]


def _controller_binding(
    evidence: Mapping[str, Any],
    *,
    coordinator_job_id: str,
    coordinator_job_name: str,
) -> dict[str, Any]:
    stages = evidence.get("stages")
    downstream = stages.get("downstream") if isinstance(stages, Mapping) else None
    _require(isinstance(downstream, Mapping), "top receipt lacks downstream stage")
    _require(
        downstream.get("slurm_job_id") == coordinator_job_id
        and downstream.get("slurm_job_name") == coordinator_job_name,
        "nested coordinator differs from the receipted downstream stage",
    )
    return {
        "controller_start_authorization_path": evidence["authorization_path"],
        "controller_start_authorization_file_sha256": evidence[
            "authorization_file_sha256"
        ],
        "controller_start_authorization_sha256": evidence[
            "start_authorization_sha256"
        ],
        "coordinator_job_id": coordinator_job_id,
        "coordinator_job_name": coordinator_job_name,
        "coordinator_stage_receipt_file_sha256": downstream[
            "stage_receipt_file_sha256"
        ],
        "coordinator_stage_receipt_sha256": downstream[
            "stage_receipt_sha256"
        ],
        "coordinator_release_receipt_file_sha256": downstream[
            "release_receipt_file_sha256"
        ],
        "coordinator_release_receipt_sha256": downstream[
            "release_receipt_sha256"
        ],
        "replacement_job_id": evidence["replacement"]["slurm_job_id"],
    }


def _transition_parameters(
    root: Path, transition: Mapping[str, Any]
) -> dict[str, str]:
    paths = _canonical_paths(root)
    _require(
        set(transition) == set(_TRANSITION_FIELDS),
        "nested transition field set changed",
    )
    _require(
        _absolute(Path(str(transition["transition_manifest_path"])))
        == paths["transition_manifest"],
        "nested transition manifest path is not canonical",
    )
    for field in _TRANSITION_FIELDS[1:]:
        _require(_hex(transition[field]), f"nested transition {field} is invalid")
    return {key: str(transition[key]) for key in _TRANSITION_FIELDS}


def build_nested_intent(
    *,
    episode_root: Path,
    archive: Mapping[str, Any],
    controller_evidence: Mapping[str, Any],
    coordinator_job_id: str,
    coordinator_job_name: str,
    transition: Mapping[str, Any],
    submission_user: str,
) -> dict[str, Any]:
    root = _absolute(episode_root)
    _require(root == EPISODE_ROOT, "nested intent root is not canonical")
    _require(
        submission_user == _effective_user(),
        "nested intent submission user differs from effective user",
    )
    _require(
        coordinator_job_id.isdigit() and int(coordinator_job_id) > 0,
        "nested coordinator job ID is invalid",
    )
    _require(
        isinstance(coordinator_job_name, str)
        and coordinator_job_name
        and "\n" not in coordinator_job_name,
        "nested coordinator job name is invalid",
    )
    replacement = controller_evidence.get("replacement")
    _require(isinstance(replacement, Mapping), "top evidence lacks replacement")
    _require(
        dict(archive) == dict(replacement.get("archive", {})),
        "nested archive differs from top replacement archive",
    )
    paths = _canonical_paths(root)
    wrapper_identity: dict[str, dict[str, str]] = {}
    for stage in STAGE_ORDER:
        wrapper = (
            Path(str(archive["archived_root"]))
            / "outputs/executor/slurm"
            / WRAPPERS[stage]
        )
        wrapper_identity[stage] = {
            "path": str(wrapper),
            "file_sha256": _sha256_file(
                wrapper,
                root=Path(str(archive["archived_root"])),
                label=f"archived nested {stage} wrapper",
            ),
        }
    transition_parameters = _transition_parameters(root, transition)
    parameters = {
        "synthetic_tests": {},
        "scientific_qualification": {
            "output_dir": str(paths["qualification_dir"]),
            "repetitions": 72,
            "whole_types": 8,
            "transition_manifest_path": transition_parameters[
                "transition_manifest_path"
            ],
            "transition_manifest_file_sha256": transition_parameters[
                "transition_manifest_file_sha256"
            ],
            "transition_receipt_sha256": transition_parameters[
                "transition_receipt_sha256"
            ],
        },
        "production_readiness": {
            "qualification_launch": str(paths["qualification_launch"]),
            "qualification_dir": str(paths["qualification_dir"]),
            **transition_parameters,
            "controller_start_authorization": str(
                paths["controller_start_authorization"]
            ),
            "controller_start_authorization_file_sha256": controller_evidence[
                "authorization_file_sha256"
            ],
            "controller_start_authorization_sha256": controller_evidence[
                "start_authorization_sha256"
            ],
            "replacement_job_id": replacement["slurm_job_id"],
        },
    }
    core = {
        "schema_version": INTENT_SCHEMA,
        "episode_root": str(root),
        "submission_user": submission_user,
        "archive": dict(archive),
        "controller": _controller_binding(
            controller_evidence,
            coordinator_job_id=coordinator_job_id,
            coordinator_job_name=coordinator_job_name,
        ),
        "stage_order": list(STAGE_ORDER),
        "stage_wrappers": wrapper_identity,
        "stage_parameters": parameters,
        "first_child_waits_for_coordinator_afterok": True,
        "all_children_submit_held": True,
        "full_null_submitted": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "intent_sha256": digest_object(core)}


def nested_stage_arguments(
    *,
    root: Path,
    intent: Mapping[str, Any],
    intent_file_sha256: str,
    stage: str,
) -> list[str]:
    _require(stage in STAGE_ORDER, "unknown nested stage")
    common = [
        *_archive_arguments(root, intent["archive"]),
        str(root / INTENT_RELATIVE),
        intent_file_sha256,
        str(intent["intent_sha256"]),
        str(root / TRANSACTION_RELATIVE / "stages" / stage / "receipt.json"),
    ]
    parameters = intent["stage_parameters"][stage]
    if stage == "synthetic_tests":
        return common
    if stage == "scientific_qualification":
        return [
            *common,
            str(parameters["output_dir"]),
            str(parameters["repetitions"]),
            str(parameters["whole_types"]),
            str(parameters["transition_manifest_path"]),
            str(parameters["transition_manifest_file_sha256"]),
            str(parameters["transition_receipt_sha256"]),
        ]
    return [
        *common,
        str(parameters["qualification_launch"]),
        str(parameters["qualification_dir"]),
        str(parameters["transition_manifest_path"]),
        str(parameters["transition_manifest_file_sha256"]),
        str(parameters["transition_receipt_file_sha256"]),
        str(parameters["transition_receipt_sha256"]),
        str(parameters["observed_prefix_contract_sha256"]),
        str(parameters["active_contract_sha256"]),
        str(parameters["active_qualification_spec_sha256"]),
        str(parameters["active_scientific_models_sha256"]),
        str(parameters["controller_start_authorization"]),
        str(parameters["controller_start_authorization_file_sha256"]),
        str(parameters["controller_start_authorization_sha256"]),
        str(parameters["replacement_job_id"]),
    ]


def build_nested_stage_intent(
    *,
    root: Path,
    intent: Mapping[str, Any],
    intent_file_sha256: str,
    stage: str,
    predecessor_job_id: str,
    dependency: str,
) -> dict[str, Any]:
    _require(stage in STAGE_ORDER, "unknown nested stage")
    _require(
        predecessor_job_id.isdigit() and int(predecessor_job_id) > 0,
        "nested predecessor is invalid",
    )
    wrapper = intent["stage_wrappers"][stage]
    core = {
        "schema_version": "ep12.post36_orchestration_stage_intent.v1",
        "stage": stage,
        "submission_user": intent["submission_user"],
        "predecessor_job_id": predecessor_job_id,
        "dependency": dependency,
        "kill_on_invalid_dependency": True,
        "submission_barrier": "user_hold_receipt_before_release_v1",
        "wrapper": wrapper["path"],
        "wrapper_sha256": wrapper["file_sha256"],
        "arguments": nested_stage_arguments(
            root=root,
            intent=intent,
            intent_file_sha256=intent_file_sha256,
            stage=stage,
        ),
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    token = digest_object(core)[:24]
    named = {**core, "job_name": f"ep12-{stage.replace('_', '-')}-{token}"}
    return {**named, "stage_intent_sha256": digest_object(named)}


def _stage_dependency(
    *, stage: str, predecessor_job_id: str, replacement_job_id: str
) -> str:
    if stage == "production_readiness":
        return (
            f"afterok:{OBSERVED_FINALIZER_JOB_ID}:"
            f"{replacement_job_id}:{predecessor_job_id}"
        )
    return f"afterok:{predecessor_job_id}"


def _verify_generic_stage(
    *,
    root: Path,
    intent: Mapping[str, Any],
    intent_file_sha256: str,
    stage: str,
    predecessor_job_id: str,
) -> dict[str, Any]:
    dependency = _stage_dependency(
        stage=stage,
        predecessor_job_id=predecessor_job_id,
        replacement_job_id=str(intent["controller"]["replacement_job_id"]),
    )
    expected_intent = build_nested_stage_intent(
        root=root,
        intent=intent,
        intent_file_sha256=intent_file_sha256,
        stage=stage,
        predecessor_job_id=predecessor_job_id,
        dependency=dependency,
    )
    stage_dir = root / TRANSACTION_RELATIVE / "stages" / stage
    observed_intent = _read_json(
        stage_dir / "intent.json",
        root=root,
        label=f"nested {stage} intent",
        canonical=True,
    )
    _require(observed_intent == expected_intent, f"nested {stage} intent mutated")
    progress_value = _read_json(
        stage_dir / "progress.json",
        root=root,
        label=f"nested {stage} progress",
        canonical=True,
    )
    state, job_id, progress = _verify_stage_progress(
        progress_value, stage_intent=expected_intent
    )
    _require(
        state == "submitted" and isinstance(job_id, str),
        f"nested {stage} was not submitted",
    )
    stage_receipt = _read_json(
        stage_dir / "receipt.json",
        root=root,
        label=f"nested {stage} receipt",
        canonical=True,
    )
    _require(
        stage_receipt
        == _stage_receipt(
            stage_intent=expected_intent, progress=progress, job_id=job_id
        ),
        f"nested {stage} receipt mutated",
    )
    release_intent = _read_json(
        stage_dir / "release_intent.json",
        root=root,
        label=f"nested {stage} release intent",
        canonical=True,
    )
    expected_release_intent = _release_intent(
        stage_receipt=stage_receipt,
        stage_receipt_file_sha256=_sha256_file(
            stage_dir / "receipt.json",
            root=root,
            label=f"nested {stage} receipt",
        ),
    )
    _require(
        release_intent == expected_release_intent,
        f"nested {stage} release intent mutated",
    )
    release_authorization = _read_json(
        stage_dir / "release_authorization.json",
        root=root,
        label=f"nested {stage} release authorization",
        canonical=True,
    )
    _require(
        release_authorization
        == _release_authorization(release_intent=release_intent),
        f"nested {stage} release authorization mutated",
    )
    release_progress = _read_json(
        stage_dir / "release_progress.json",
        root=root,
        label=f"nested {stage} release progress",
        canonical=True,
    )
    _require(
        release_progress
        == _release_progress(release_intent=release_intent, state="released"),
        f"nested {stage} was not durably released",
    )
    release_receipt = _read_json(
        stage_dir / "release_receipt.json",
        root=root,
        label=f"nested {stage} release receipt",
        canonical=True,
    )
    _require(
        release_receipt
        == _release_receipt(
            release_intent=release_intent,
            release_progress=release_progress,
        ),
        f"nested {stage} release receipt mutated",
    )
    return {
        **dict(stage_receipt),
        "stage_receipt_path": str(stage_dir / "receipt.json"),
        "stage_receipt_file_sha256": _sha256_file(
            stage_dir / "receipt.json",
            root=root,
            label=f"nested {stage} receipt",
        ),
        "release_authorization_path": str(
            stage_dir / "release_authorization.json"
        ),
        "release_authorization_file_sha256": _sha256_file(
            stage_dir / "release_authorization.json",
            root=root,
            label=f"nested {stage} release authorization",
        ),
        "release_authorization_sha256": release_authorization[
            "release_authorization_sha256"
        ],
        "release_receipt_path": str(stage_dir / "release_receipt.json"),
        "release_receipt_file_sha256": _sha256_file(
            stage_dir / "release_receipt.json",
            root=root,
            label=f"nested {stage} release receipt",
        ),
        "release_receipt_sha256": release_receipt["release_receipt_sha256"],
    }


def _start_authorization(
    *,
    root: Path,
    intent: Mapping[str, Any],
    intent_file_sha256: str,
    stages: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    keys = (
        "slurm_job_id",
        "slurm_job_name",
        "dependency",
        "stage_receipt_path",
        "stage_receipt_file_sha256",
        "stage_receipt_sha256",
        "release_authorization_file_sha256",
        "release_authorization_sha256",
        "release_receipt_file_sha256",
        "release_receipt_sha256",
    )
    bindings = {
        stage: {key: stages[stage][key] for key in keys}
        for stage in STAGE_ORDER
    }
    core = {
        "schema_version": START_AUTHORIZATION_SCHEMA,
        "intent_path": str(root / INTENT_RELATIVE),
        "intent_file_sha256": intent_file_sha256,
        "intent_sha256": intent["intent_sha256"],
        "controller": dict(intent["controller"]),
        "stages": bindings,
        "all_children_receipted_and_released_while_coordinator_running": True,
        "first_child_waits_for_coordinator_afterok": True,
        "full_null_submitted": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "start_authorization_sha256": digest_object(core)}


def _nested_receipt(
    *,
    root: Path,
    intent: Mapping[str, Any],
    intent_file_sha256: str,
    authorization: Mapping[str, Any],
    stages: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    core = {
        "schema_version": RECEIPT_SCHEMA,
        "intent_path": str(root / INTENT_RELATIVE),
        "intent_file_sha256": intent_file_sha256,
        "intent_sha256": intent["intent_sha256"],
        "start_authorization_path": str(root / START_AUTHORIZATION_RELATIVE),
        "start_authorization_file_sha256": _sha256_file(
            root / START_AUTHORIZATION_RELATIVE,
            root=root,
            label="nested start authorization",
        ),
        "start_authorization_sha256": authorization[
            "start_authorization_sha256"
        ],
        "controller": dict(intent["controller"]),
        "archive": dict(intent["archive"]),
        "stage_order": list(STAGE_ORDER),
        "stages": {stage: dict(stages[stage]) for stage in STAGE_ORDER},
        "all_job_ids_receipt_bound": True,
        "all_release_receipts_durable": True,
        "manual_continuation_required": False,
        "full_null_submitted": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "receipt_sha256": digest_object(core)}


def verify_archive_nested_stage_receipt(
    *, episode_root: Path, receipt_path: Path | None = None
) -> dict[str, Any]:
    root = _absolute(episode_root)
    _require(root == EPISODE_ROOT, "nested verifier root is not canonical")
    paths = _canonical_paths(root)
    path = paths["receipt"] if receipt_path is None else _absolute(receipt_path)
    _require(path == paths["receipt"], "nested receipt path is not canonical")
    controller = verify_post36_start_authorization(episode_root=root)
    intent = _read_json(
        paths["intent"], root=root, label="nested intent", canonical=True
    )
    readiness = intent.get("stage_parameters", {}).get(
        "production_readiness", {}
    )
    transition = {key: readiness[key] for key in _TRANSITION_FIELDS}
    expected_intent = build_nested_intent(
        episode_root=root,
        archive=controller["replacement"]["archive"],
        controller_evidence=controller,
        coordinator_job_id=str(
            controller["stages"]["downstream"]["slurm_job_id"]
        ),
        coordinator_job_name=str(
            controller["stages"]["downstream"]["slurm_job_name"]
        ),
        transition=transition,
        submission_user=_effective_user(),
    )
    _require(intent == expected_intent, "nested intent mutated")
    intent_file_sha = _sha256_file(
        paths["intent"], root=root, label="nested intent"
    )
    stages: dict[str, dict[str, Any]] = {}
    predecessor = str(intent["controller"]["coordinator_job_id"])
    for stage in STAGE_ORDER:
        stages[stage] = _verify_generic_stage(
            root=root,
            intent=intent,
            intent_file_sha256=intent_file_sha,
            stage=stage,
            predecessor_job_id=predecessor,
        )
        predecessor = str(stages[stage]["slurm_job_id"])
    authorization = _read_json(
        paths["start_authorization"],
        root=root,
        label="nested start authorization",
        canonical=True,
    )
    expected_authorization = _start_authorization(
        root=root,
        intent=intent,
        intent_file_sha256=intent_file_sha,
        stages=stages,
    )
    _require(
        authorization == expected_authorization,
        "nested start authorization mutated",
    )
    receipt = _read_json(
        path, root=root, label="nested receipt", canonical=True
    )
    expected_receipt = _nested_receipt(
        root=root,
        intent=intent,
        intent_file_sha256=intent_file_sha,
        authorization=authorization,
        stages=stages,
    )
    _require(receipt == expected_receipt, "nested receipt mutated")
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "intent_path": str(paths["intent"]),
        "intent_file_sha256": intent_file_sha,
        "intent_sha256": intent["intent_sha256"],
        "receipt_path": str(path),
        "receipt_file_sha256": _sha256_file(
            path, root=root, label="nested receipt"
        ),
        "receipt_sha256": receipt["receipt_sha256"],
        "start_authorization_path": str(paths["start_authorization"]),
        "start_authorization_file_sha256": receipt[
            "start_authorization_file_sha256"
        ],
        "start_authorization_sha256": receipt[
            "start_authorization_sha256"
        ],
        "controller": dict(intent["controller"]),
        "archive": dict(intent["archive"]),
        "stages": stages,
        "full_null_submitted": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }


def launch_archive_nested_stages(
    *,
    episode_root: Path,
    coordinator_job_id: str,
    coordinator_job_name: str,
    transition: Mapping[str, Any],
    submitter: Submitter = default_submitter,
    job_resolver: JobResolver = default_exact_name_resolver,
    job_state_resolver: JobStateResolver = default_job_state_resolver,
    releaser: Releaser = default_releaser,
) -> dict[str, Any]:
    root = _absolute(episode_root)
    _require(root == EPISODE_ROOT, "nested launch root is not canonical")
    paths = _canonical_paths(root)
    try:
        _read_json(
            paths["receipt"], root=root, label="nested receipt", canonical=True
        )
    except FileNotFoundError:
        pass
    else:
        return verify_archive_nested_stage_receipt(episode_root=root)
    controller = verify_post36_start_authorization(episode_root=root)
    archive = controller["replacement"]["archive"]
    transaction, transaction_fd = open_secure_directory(
        root / TRANSACTION_RELATIVE, root=root, create=True
    )
    lock_fd = open_regular_lock_at(transaction_fd, ".lock")
    os.close(transaction_fd)
    with os.fdopen(lock_fd, "a+b") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        intent = _write_or_verify(
            paths["intent"],
            build_nested_intent(
                episode_root=root,
                archive=archive,
                controller_evidence=controller,
                coordinator_job_id=coordinator_job_id,
                coordinator_job_name=coordinator_job_name,
                transition=transition,
                submission_user=_effective_user(),
            ),
            root=root,
            label="nested intent",
        )
        intent_file_sha = _sha256_file(
            paths["intent"], root=root, label="nested intent"
        )
        held: dict[str, dict[str, Any]] = {}
        stage_intents: dict[str, dict[str, Any]] = {}
        predecessor = coordinator_job_id
        for stage in STAGE_ORDER:
            dependency = _stage_dependency(
                stage=stage,
                predecessor_job_id=predecessor,
                replacement_job_id=str(
                    intent["controller"]["replacement_job_id"]
                ),
            )
            stage_intent = build_nested_stage_intent(
                root=root,
                intent=intent,
                intent_file_sha256=intent_file_sha,
                stage=stage,
                predecessor_job_id=predecessor,
                dependency=dependency,
            )
            stage_intents[stage] = stage_intent
            held[stage] = _submit_stage_held(
                transaction_root=transaction,
                root=root,
                stage_intent=stage_intent,
                submitter=submitter,
                job_resolver=job_resolver,
                job_state_resolver=job_state_resolver,
            )
            predecessor = str(held[stage]["slurm_job_id"])
        released: dict[str, dict[str, Any]] = {}
        for stage in STAGE_ORDER:
            released[stage] = _release_submitted_stage(
                transaction_root=transaction,
                root=root,
                stage_intent=stage_intents[stage],
                stage_receipt=held[stage],
                job_state_resolver=job_state_resolver,
                releaser=releaser,
            )
        authorization = _write_or_verify(
            paths["start_authorization"],
            _start_authorization(
                root=root,
                intent=intent,
                intent_file_sha256=intent_file_sha,
                stages=released,
            ),
            root=root,
            label="nested start authorization",
        )
        _write_or_verify(
            paths["receipt"],
            _nested_receipt(
                root=root,
                intent=intent,
                intent_file_sha256=intent_file_sha,
                authorization=authorization,
                stages=released,
            ),
            root=root,
            label="nested receipt",
        )
        return verify_archive_nested_stage_receipt(episode_root=root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    verify = parser.add_subparsers(dest="command", required=True).add_parser(
        "verify"
    )
    verify.add_argument("--episode-root", type=Path, required=True)
    verify.add_argument("--receipt-path", type=Path)
    arguments = parser.parse_args(argv)
    result = verify_archive_nested_stage_receipt(
        episode_root=arguments.episode_root,
        receipt_path=arguments.receipt_path,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
