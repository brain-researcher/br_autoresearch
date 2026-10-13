"""Exact-once archive-authenticated handoffs for the EP12 post-36 tail."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

from .policy import canonical_json, digest_object
from .post36_orchestration import (
    STAGE_INTENT_SCHEMA,
    JobResolver,
    JobStateResolver,
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
    _stage_submission_evidence,
    _submit_stage_held,
    _verify_stage_progress,
    _write_or_verify,
    default_exact_name_resolver,
    default_job_state_resolver,
    default_releaser,
    default_submitter,
)
from .post36_downstream_launch import verify_post36_downstream_receipt
from .post36_receipt_bound import (
    _accepted_executed_wrapper,
    _sha,
    _stage_result,
)
from .post36_unattended_controller import (
    INTENT_RELATIVE as UPSTREAM_INTENT_RELATIVE,
    RECEIPT_RELATIVE as UPSTREAM_RECEIPT_RELATIVE,
    START_AUTHORIZATION_RELATIVE as UPSTREAM_START_RELATIVE,
    verify_post36_orchestration_receipt,
    verify_post36_start_authorization,
)
from .secure_transaction import open_regular_lock_at, open_secure_directory
from .slurm_handoff import authenticate_handoff_context
from .synthetic import EPISODE_ROOT


INTENT_SCHEMA = "ep12.archive_slurm_handoff_intent.v1"
RECEIPT_SCHEMA = "ep12.archive_slurm_handoff_receipt.v1"
EVIDENCE_SCHEMA = "ep12.archive_slurm_handoff_evidence.v1"
RUNTIME_SCHEMA = "ep12.archive_slurm_child_runtime_evidence.v1"
TRANSACTION_RELATIVE = Path(
    "outputs/development_run002/archive_slurm_handoffs"
)
RUN_RELATIVE = Path("outputs/development_run002")
KINDS = {
    "readiness_to_archive_tail": {
        "context_kind": "readiness_to_post36_tail",
        "wrapper": "run_post36_tail_from_archive.sbatch",
    },
    "archive_tail_to_next_tail": {
        "context_kind": "post36_tail_to_next_tail",
        "wrapper": "run_post36_tail_from_archive.sbatch",
    },
    "archive_tail_to_procedure_lock": {
        "context_kind": "post36_tail_to_procedure_lock",
        "wrapper": "run_procedure_lock_from_archive.sbatch",
    },
}


class ArchiveSlurmHandoffError(RuntimeError):
    """An archive tail child cannot be submitted or authenticated safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchiveSlurmHandoffError(message)


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _transaction(root: Path, kind: str, parent_job_id: str) -> Path:
    return root / TRANSACTION_RELATIVE / kind / f"parent-{parent_job_id}"


def _active_executor(executor_root: Path) -> Path:
    executor = _absolute(executor_root)
    expected = (
        executor / "ep12_executor/archive_slurm_handoff.py"
    ).resolve(strict=True)
    _require(
        Path(__file__).resolve(strict=True) == expected,
        "archive handoff was imported outside the authenticated stage",
    )
    return executor


def _authority(root: Path, executor: Path, *, kind: str) -> dict[str, Any]:
    start = verify_post36_start_authorization(episode_root=root)
    publisher = start["stages"]["publisher"]
    coordinator = start["stages"]["downstream"]
    downstream = verify_post36_downstream_receipt(
        episode_root=root,
        executor_root=executor,
        orchestration_intent=root / UPSTREAM_INTENT_RELATIVE,
        orchestration_intent_sha256=str(start["intent_sha256"]),
        publisher_job_id=str(publisher["slurm_job_id"]),
        coordinator_job_id=str(coordinator["slurm_job_id"]),
    )
    expected_start = {
        "path": start["authorization_path"],
        "file_sha256": start["authorization_file_sha256"],
        "semantic_sha256": start["start_authorization_sha256"],
    }
    _require(
        start["replacement"]["archive"] == downstream["archive"]
        and start["replacement"]["slurm_job_id"]
        == downstream["replacement_job_id"]
        and downstream["upstream_start_authorization"] == expected_start,
        "tail authority disagrees on start authorization, archive, or replacement identity",
    )
    final = None
    final_path = root / UPSTREAM_RECEIPT_RELATIVE
    final_present = os.path.lexists(final_path)
    _require(
        final_present or kind != "archive_tail_to_procedure_lock",
        "procedure-lock boundary requires the final top receipt",
    )
    if final_present:
        _require(
            final_path.is_file() and not final_path.is_symlink(),
            "present final top receipt is not a regular file",
        )
        final = verify_post36_orchestration_receipt(
            episode_root=root,
            receipt_path=final_path,
        )
        _require(
            final["intent_sha256"] == start["intent_sha256"]
            and final["replacement"]["slurm_job_id"]
            == start["replacement"]["slurm_job_id"]
            and final["replacement"]["archive"]
            == start["replacement"]["archive"]
            and final["start_authorization"]["path"]
            == start["authorization_path"]
            and final["start_authorization"]["file_sha256"]
            == start["authorization_file_sha256"]
            and final["start_authorization"]["start_authorization_sha256"]
            == start["start_authorization_sha256"],
            "final top receipt differs from immutable start authority",
        )
    return {
        "start": start,
        "final": final,
        "downstream": downstream,
        "use_final_top_receipt": kind == "archive_tail_to_procedure_lock",
    }


def _stable_authority(authority: Mapping[str, Any]) -> dict[str, Any]:
    start = authority["start"]
    final = authority["final"]
    downstream = authority["downstream"]
    use_final = authority["use_final_top_receipt"] is True
    if not use_final:
        controller_kind = "start_authorization"
        controller_path = start["authorization_path"]
        controller_file_sha = start["authorization_file_sha256"]
        controller_semantic_sha = start["start_authorization_sha256"]
        final_top_receipt_verified = False
    else:
        _require(final is not None, "selected final top receipt is missing")
        controller_kind = "final_top_receipt"
        controller_path = final["receipt_path"]
        controller_file_sha = final["receipt_file_sha256"]
        controller_semantic_sha = final["receipt_sha256"]
        final_top_receipt_verified = True
    return {
        "controller_authority_kind": controller_kind,
        "controller_authority_path": controller_path,
        "controller_authority_file_sha256": controller_file_sha,
        "controller_authority_sha256": controller_semantic_sha,
        "final_top_receipt_verified": final_top_receipt_verified,
        "downstream_receipt_path": downstream["receipt_path"],
        "downstream_receipt_file_sha256": downstream[
            "receipt_file_sha256"
        ],
        "downstream_receipt_sha256": downstream["receipt_sha256"],
        "replacement_job_id": start["replacement"]["slurm_job_id"],
        "archive": dict(start["replacement"]["archive"]),
    }


def _child_arguments(
    *, root: Path, kind: str, receipt_path: Path,
    authority: Mapping[str, Any],
) -> list[str]:
    stable = _stable_authority(authority)
    archive = stable["archive"]
    return [
        str(root),
        str(archive["archived_root"]),
        str(archive["archive_receipt_file_sha256"]),
        str(archive["archive_receipt_sha256"]),
        str(archive["manifest_file_sha256"]),
        str(archive["manifest_sha256"]),
        str(archive["tree_sha256"]),
        str(stable["controller_authority_path"]),
        str(stable["controller_authority_file_sha256"]),
        str(stable["controller_authority_sha256"]),
        str(stable["downstream_receipt_path"]),
        str(stable["downstream_receipt_file_sha256"]),
        str(stable["downstream_receipt_sha256"]),
        str(stable["replacement_job_id"]),
        kind,
        str(receipt_path),
    ]


def _build_intent(
    *, root: Path, executor: Path, kind: str, parent_job_id: str,
    context_root: Path, authority: Mapping[str, Any], submission_user: str,
) -> dict[str, Any]:
    _require(kind in KINDS, "archive handoff kind is unknown")
    _require(
        parent_job_id.isdigit() and int(parent_job_id) > 0,
        "archive handoff parent job ID is invalid",
    )
    transaction = _transaction(root, kind, parent_job_id)
    archive = authority["start"]["replacement"]["archive"]
    wrapper = (
        Path(str(archive["archived_root"]))
        / "outputs/executor/slurm"
        / str(KINDS[kind]["wrapper"])
    )
    context = authenticate_handoff_context(
        kind=str(KINDS[kind]["context_kind"]),
        context_root=context_root,
        run_root=root / RUN_RELATIVE,
    )
    if kind == "readiness_to_archive_tail":
        readiness_job = authority["downstream"]["stages"][
            "production_readiness"
        ]["slurm_job_id"]
        _require(
            readiness_job == parent_job_id,
            "first archive tail parent is not the receipted readiness job",
        )
    core = {
        "schema_version": INTENT_SCHEMA,
        "episode_root": str(root),
        "submission_user": submission_user,
        "kind": kind,
        "parent_job_id": parent_job_id,
        "dependency": f"afterany:{parent_job_id}",
        "context_root": str(_absolute(context_root)),
        "context": context,
        "authority": _stable_authority(authority),
        "child_wrapper": str(wrapper),
        "child_wrapper_sha256": _sha256_file(
            wrapper,
            root=Path(str(archive["archived_root"])),
            label="archived tail handoff wrapper",
        ),
        "child_arguments": _child_arguments(
            root=root,
            kind=kind,
            receipt_path=transaction / "receipt.json",
            authority=authority,
        ),
        "submission_barrier": "user_hold_receipt_before_release_v1",
        "parent_dependency_is_start_barrier": True,
        "manual_continuation_required": False,
        "full_null_submitted": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "intent_sha256": digest_object(core)}


def _stage_intent(intent: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        "schema_version": STAGE_INTENT_SCHEMA,
        "stage": str(intent["kind"]),
        "submission_user": intent["submission_user"],
        "predecessor_job_id": intent["parent_job_id"],
        "dependency": intent["dependency"],
        "kill_on_invalid_dependency": True,
        "submission_barrier": "user_hold_receipt_before_release_v1",
        "wrapper": intent["child_wrapper"],
        "wrapper_sha256": intent["child_wrapper_sha256"],
        "arguments": list(intent["child_arguments"]),
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    token = digest_object(core)[:24]
    named = {**core, "job_name": f"ep12-{intent['kind']}-{token}"}
    return {**named, "stage_intent_sha256": digest_object(named)}


def _verify_stage(
    *, root: Path, transaction: Path, stage_intent: Mapping[str, Any]
) -> dict[str, Any]:
    stage = str(stage_intent["stage"])
    stage_dir = transaction / "stages" / stage
    observed_intent = _read_json(
        stage_dir / "intent.json", root=root,
        label="archive handoff stage intent", canonical=True,
    )
    _require(observed_intent == dict(stage_intent), "archive handoff stage intent mutated")
    state, job_id, progress = _verify_stage_progress(
        _read_json(
            stage_dir / "progress.json", root=root,
            label="archive handoff stage progress", canonical=True,
        ),
        stage_intent=stage_intent,
    )
    _require(state == "submitted" and job_id is not None, "archive child was not submitted")
    receipt = _read_json(
        stage_dir / "receipt.json", root=root,
        label="archive handoff stage receipt", canonical=True,
    )
    _require(
        receipt == _stage_receipt(
            stage_intent=stage_intent, progress=progress, job_id=job_id
        ),
        "archive handoff stage receipt mutated",
    )
    release_intent = _read_json(
        stage_dir / "release_intent.json", root=root,
        label="archive handoff release intent", canonical=True,
    )
    _require(
        release_intent == _release_intent(
            stage_receipt=receipt,
            stage_receipt_file_sha256=_sha256_file(
                stage_dir / "receipt.json", root=root,
                label="archive handoff stage receipt",
            ),
        ),
        "archive handoff release intent mutated",
    )
    authorization = _read_json(
        stage_dir / "release_authorization.json", root=root,
        label="archive handoff release authorization", canonical=True,
    )
    _require(
        authorization == _release_authorization(release_intent=release_intent),
        "archive handoff release authorization mutated",
    )
    release_progress = _read_json(
        stage_dir / "release_progress.json", root=root,
        label="archive handoff release progress", canonical=True,
    )
    _require(
        release_progress == _release_progress(
            release_intent=release_intent, state="released"
        ),
        "archive child was not durably released",
    )
    release_receipt = _read_json(
        stage_dir / "release_receipt.json", root=root,
        label="archive handoff release receipt", canonical=True,
    )
    _require(
        release_receipt == _release_receipt(
            release_intent=release_intent,
            release_progress=release_progress,
        ),
        "archive handoff release receipt mutated",
    )
    return _stage_submission_evidence(
        stage_dir=stage_dir,
        root=root,
        stage_receipt=receipt,
        release_receipt=release_receipt,
    )


def _receipt(
    *, root: Path, transaction: Path, intent: Mapping[str, Any],
    intent_file_sha256: str, stage: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": RECEIPT_SCHEMA,
        "intent_path": str(transaction / "intent.json"),
        "intent_file_sha256": intent_file_sha256,
        "intent_sha256": intent["intent_sha256"],
        "kind": intent["kind"],
        "parent_job_id": intent["parent_job_id"],
        "dependency": intent["dependency"],
        "context": dict(intent["context"]),
        "authority": dict(intent["authority"]),
        "child": dict(stage),
        "child_stage_and_release_receipts_durable_before_parent_exit": True,
        "parent_dependency_is_start_barrier": True,
        "manual_continuation_required": False,
        "full_null_submitted": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "receipt_sha256": digest_object(core)}


def submit_archive_handoff(
    *, episode_root: Path, executor_root: Path, kind: str,
    parent_job_id: str, context_root: Path,
    submitter: Submitter = default_submitter,
    job_resolver: JobResolver = default_exact_name_resolver,
    job_state_resolver: JobStateResolver = default_job_state_resolver,
    releaser: Releaser = default_releaser,
) -> dict[str, Any]:
    root = _absolute(episode_root)
    _require(root == EPISODE_ROOT, "archive handoff root is not canonical")
    executor = _active_executor(executor_root)
    transaction = _transaction(root, kind, parent_job_id)
    receipt_path = transaction / "receipt.json"
    try:
        _read_json(
            receipt_path, root=root, label="archive handoff receipt", canonical=True
        )
    except FileNotFoundError:
        pass
    else:
        return verify_archive_handoff_receipt(
            episode_root=root,
            executor_root=executor,
            kind=kind,
            parent_job_id=parent_job_id,
            context_root=context_root,
            receipt_path=receipt_path,
        )
    authority = _authority(root, executor, kind=kind)
    transaction, transaction_fd = open_secure_directory(
        transaction, root=root, create=True
    )
    lock_fd = open_regular_lock_at(transaction_fd, ".lock")
    os.close(transaction_fd)
    with os.fdopen(lock_fd, "a+b") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        intent = _write_or_verify(
            transaction / "intent.json",
            _build_intent(
                root=root, executor=executor, kind=kind,
                parent_job_id=parent_job_id, context_root=context_root,
                authority=authority, submission_user=_effective_user(),
            ),
            root=root,
            label="archive handoff intent",
        )
        intent_file_sha = _sha256_file(
            transaction / "intent.json", root=root,
            label="archive handoff intent",
        )
        stage_intent = _stage_intent(intent)
        held = _submit_stage_held(
            transaction_root=transaction,
            root=root,
            stage_intent=stage_intent,
            submitter=submitter,
            job_resolver=job_resolver,
            job_state_resolver=job_state_resolver,
        )
        released = _release_submitted_stage(
            transaction_root=transaction,
            root=root,
            stage_intent=stage_intent,
            stage_receipt=held,
            job_state_resolver=job_state_resolver,
            releaser=releaser,
        )
        _write_or_verify(
            receipt_path,
            _receipt(
                root=root, transaction=transaction, intent=intent,
                intent_file_sha256=intent_file_sha, stage=released,
            ),
            root=root,
            label="archive handoff receipt",
        )
        return verify_archive_handoff_receipt(
            episode_root=root,
            executor_root=executor,
            kind=kind,
            parent_job_id=parent_job_id,
            context_root=context_root,
            receipt_path=receipt_path,
        )


def verify_archive_handoff_receipt(
    *, episode_root: Path, executor_root: Path, kind: str,
    parent_job_id: str, context_root: Path, receipt_path: Path,
) -> dict[str, Any]:
    root = _absolute(episode_root)
    _require(root == EPISODE_ROOT, "archive handoff verifier root is not canonical")
    executor = _active_executor(executor_root)
    transaction = _transaction(root, kind, parent_job_id)
    _require(
        _absolute(receipt_path) == transaction / "receipt.json",
        "archive handoff receipt path is not canonical",
    )
    authority = _authority(root, executor, kind=kind)
    expected_intent = _build_intent(
        root=root, executor=executor, kind=kind,
        parent_job_id=parent_job_id, context_root=context_root,
        authority=authority, submission_user=_effective_user(),
    )
    intent = _read_json(
        transaction / "intent.json", root=root,
        label="archive handoff intent", canonical=True,
    )
    _require(intent == expected_intent, "archive handoff intent mutated")
    intent_file_sha = _sha256_file(
        transaction / "intent.json", root=root,
        label="archive handoff intent",
    )
    stage = _verify_stage(
        root=root, transaction=transaction, stage_intent=_stage_intent(intent)
    )
    receipt = _read_json(
        receipt_path, root=root, label="archive handoff receipt", canonical=True
    )
    expected_receipt = _receipt(
        root=root, transaction=transaction, intent=intent,
        intent_file_sha256=intent_file_sha, stage=stage,
    )
    _require(receipt == expected_receipt, "archive handoff receipt mutated")
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "intent_path": str(transaction / "intent.json"),
        "intent_file_sha256": intent_file_sha,
        "intent_sha256": intent["intent_sha256"],
        "receipt_path": str(receipt_path),
        "receipt_file_sha256": _sha256_file(
            receipt_path, root=root, label="archive handoff receipt"
        ),
        "receipt_sha256": receipt["receipt_sha256"],
        "kind": kind,
        "parent_job_id": parent_job_id,
        "authority": dict(intent["authority"]),
        "context": dict(intent["context"]),
        "child": stage,
        "full_null_submitted": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }


def verify_running_archive_child(
    *, episode_root: Path, executor_root: Path, kind: str,
    handoff_receipt_path: Path, executed_wrapper: Path,
    slurm_job_id: str, slurm_job_name: str,
    bootstrap_stage_result_path: Path, staged_root: Path,
    wrapper_arguments: Sequence[str],
) -> dict[str, Any]:
    root = _absolute(episode_root)
    executor = _active_executor(executor_root)
    path = _absolute(handoff_receipt_path)
    expected_parent_dir = path.parent.name
    _require(
        expected_parent_dir.startswith("parent-")
        and expected_parent_dir[7:].isdigit(),
        "archive child handoff parent path is invalid",
    )
    parent = expected_parent_dir[7:]
    evidence = verify_archive_handoff_receipt(
        episode_root=root, executor_root=executor, kind=kind,
        parent_job_id=parent, context_root=(
            root / RUN_RELATIVE / "production_readiness"
            if kind == "readiness_to_archive_tail"
            else root / RUN_RELATIVE / "post36_tail"
        ),
        receipt_path=path,
    )
    child = evidence["child"]
    _require(
        child["slurm_job_id"] == slurm_job_id
        and child["slurm_job_name"] == slurm_job_name,
        "running archive child differs from the handoff receipt",
    )
    arguments = list(wrapper_arguments)
    _require(
        arguments == child["arguments"],
        "running archive child arguments differ from the handoff receipt",
    )
    observed_wrapper = _absolute(executed_wrapper)
    _require(
        _accepted_executed_wrapper(
            observed=observed_wrapper,
            archived=Path(str(child["wrapper"])),
            job_id=slurm_job_id,
        )
        and _sha(observed_wrapper, label="executed archive child wrapper")
        == child["wrapper_sha256"],
        "running archive child wrapper identity changed",
    )
    archive = evidence["authority"]["archive"]
    stage_result = _stage_result(
        path=bootstrap_stage_result_path,
        staged_root=_absolute(staged_root),
        archive=archive,
    )
    boundary = (_absolute(staged_root) / "outputs/executor").resolve(strict=True)
    for name, imported in tuple(sys.modules.items()):
        if name != "ep12_executor" and not name.startswith("ep12_executor."):
            continue
        imported_path = getattr(imported, "__file__", None)
        if imported_path is None:
            continue
        resolved = Path(str(imported_path)).resolve(strict=True)
        _require(
            resolved == boundary or boundary in resolved.parents,
            f"EP12 module was imported outside the authenticated stage: {name}",
        )
    return {
        "schema_version": RUNTIME_SCHEMA,
        "kind": kind,
        "slurm_job_id": slurm_job_id,
        "slurm_job_name": slurm_job_name,
        "handoff_receipt_path": evidence["receipt_path"],
        "handoff_receipt_file_sha256": evidence["receipt_file_sha256"],
        "handoff_receipt_sha256": evidence["receipt_sha256"],
        "archive": dict(archive),
        "authority": dict(evidence["authority"]),
        "bootstrap_stage_result": dict(stage_result),
        "staged_root": str(_absolute(staged_root)),
        "all_stage_and_release_receipts_verified": True,
        "full_null_submitted": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit = subparsers.add_parser("submit")
    submit.add_argument("--episode-root", type=Path, required=True)
    submit.add_argument("--executor-root", type=Path, required=True)
    submit.add_argument("--kind", choices=tuple(KINDS), required=True)
    submit.add_argument("--parent-job-id", required=True)
    submit.add_argument("--context-root", type=Path, required=True)
    runtime = subparsers.add_parser("verify-running-child")
    runtime.add_argument("--episode-root", type=Path, required=True)
    runtime.add_argument("--executor-root", type=Path, required=True)
    runtime.add_argument("--kind", choices=tuple(KINDS), required=True)
    runtime.add_argument("--handoff-receipt-path", type=Path, required=True)
    runtime.add_argument("--executed-wrapper", type=Path, required=True)
    runtime.add_argument("--slurm-job-id", required=True)
    runtime.add_argument("--slurm-job-name", required=True)
    runtime.add_argument("--bootstrap-stage-result-path", type=Path, required=True)
    runtime.add_argument("--staged-root", type=Path, required=True)
    runtime.add_argument("--wrapper-arguments", nargs=argparse.REMAINDER, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "submit":
        current = os.environ.get("SLURM_JOB_ID")
        _require(current == arguments.parent_job_id, "archive handoff parent differs from current job")
        result = submit_archive_handoff(
            episode_root=arguments.episode_root,
            executor_root=arguments.executor_root,
            kind=arguments.kind,
            parent_job_id=arguments.parent_job_id,
            context_root=arguments.context_root,
        )
    else:
        result = verify_running_archive_child(
            episode_root=arguments.episode_root,
            executor_root=arguments.executor_root,
            kind=arguments.kind,
            handoff_receipt_path=arguments.handoff_receipt_path,
            executed_wrapper=arguments.executed_wrapper,
            slurm_job_id=arguments.slurm_job_id,
            slurm_job_name=arguments.slurm_job_name,
            bootstrap_stage_result_path=arguments.bootstrap_stage_result_path,
            staged_root=arguments.staged_root,
            wrapper_arguments=arguments.wrapper_arguments,
        )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
