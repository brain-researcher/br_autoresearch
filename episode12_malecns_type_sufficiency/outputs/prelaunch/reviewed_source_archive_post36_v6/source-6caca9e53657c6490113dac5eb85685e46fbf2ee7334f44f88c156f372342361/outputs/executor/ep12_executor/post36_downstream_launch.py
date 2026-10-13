"""Archive-bound unattended EP12 test, qualification, and readiness launcher.

This coordinator runs as the final job in the post-36 control-plane DAG.  It
submits all three descendants under user holds, persists every accepted job ID,
releases only those exact jobs, and publishes a replayable receipt before this
coordinator can satisfy the first child's dependency.
"""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Sequence

from .policy import canonical_json, digest_object
from .post36_orchestration import (
    OBSERVED_FINALIZER_JOB_ID,
    STAGE_INTENT_SCHEMA,
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
from .post36_unattended_controller import (
    INTENT_RELATIVE as UPSTREAM_INTENT_RELATIVE,
    RECEIPT_RELATIVE as UPSTREAM_RECEIPT_RELATIVE,
    START_AUTHORIZATION_RELATIVE as UPSTREAM_AUTHORIZATION_RELATIVE,
    verify_post36_orchestration_receipt,
    verify_post36_start_authorization,
)
from .scientific_transition_binding import (
    TRANSITION_MANIFEST_NAME,
    TRANSITION_ROOT_RELATIVE,
    verify_scientific_contract_transition,
)
from .secure_transaction import open_regular_lock_at, open_secure_directory
from .source_archive_freeze_bridge import bridge_source_archive_to_live_freeze
from .synthetic import EPISODE_ROOT


INTENT_SCHEMA = "ep12.post36_downstream_launch_intent.v1"
RECEIPT_SCHEMA = "ep12.post36_downstream_launch_receipt.v1"
TRANSACTION_RELATIVE = Path("outputs/prelaunch/post36_downstream_launch")
INTENT_RELATIVE = TRANSACTION_RELATIVE / "intent.json"
RECEIPT_RELATIVE = TRANSACTION_RELATIVE / "receipt.json"
STAGE_ORDER = ("synthetic_tests", "scientific_qualification", "production_readiness")
WRAPPERS = {
    "synthetic_tests": "test_executor.sbatch",
    "scientific_qualification": "qualify_methods.sbatch",
    "production_readiness": "check_production_readiness.sbatch",
}
QUALIFICATION_OUTPUT_RELATIVE = Path(
    "outputs/prelaunch/scientific_qualification_qualification003"
)
QUALIFICATION_REPETITIONS = "72"
QUALIFICATION_WHOLE_TYPES = "8"


class Post36DownstreamLaunchError(RuntimeError):
    """The archive-bound downstream launch is incomplete or unauthenticated."""


Submitter = Callable[[Sequence[str]], str]
JobResolver = Callable[[str, str], Mapping[str, str] | None]
JobStateResolver = Callable[[str, str, str], Mapping[str, str]]
Releaser = Callable[[Sequence[str]], str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Post36DownstreamLaunchError(message)


def _lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except FileNotFoundError:
        return False


def _active_executor(executor_root: Path) -> Path:
    executor = _lexical(executor_root)
    active = Path(__file__).resolve(strict=True)
    expected = (executor / "ep12_executor/post36_downstream_launch.py").resolve(
        strict=True
    )
    _require(active == expected, "downstream coordinator was not imported from the staged executor")
    return executor


def _stable_bridge(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = {
        "receipt_path", "receipt_file_sha256", "receipt_sha256",
        "archive_identity", "live_episode_root", "source_freeze_identity",
        "executable_record_map_sha256", "file_count",
        "archive_and_live_file_maps_identical",
        "archive_modes_bound_upstream_not_required_live_equal",
        "archive_reverified_after_freeze", "live_surface_reverified_after_freeze",
        "real_connectivity_accessed", "final_connectivity_access_authorized",
    }
    _require(keys.issubset(value), "source-archive bridge evidence is incomplete")
    return {key: value[key] for key in sorted(keys)}


def _controller_authority(
    *, root: Path, upstream_intent: Path, expected_intent_sha256: str,
    publisher_job_id: str, coordinator_job_id: str,
) -> dict[str, Any]:
    _require(
        _lexical(upstream_intent) == root / UPSTREAM_INTENT_RELATIVE,
        "upstream orchestration intent path is not canonical",
    )
    authority = verify_post36_start_authorization(episode_root=root)
    _require(
        authority["intent_path"] == str(root / UPSTREAM_INTENT_RELATIVE)
        and authority["intent_sha256"] == expected_intent_sha256,
        "upstream start authority does not bind the requested intent",
    )
    stages = authority["stages"]
    replacement = authority["replacement"]
    _require(
        stages["publisher"]["slurm_job_id"] == publisher_job_id
        and stages["downstream"]["slurm_job_id"] == coordinator_job_id,
        "running publisher/coordinator IDs differ from upstream authority",
    )
    final_path = root / UPSTREAM_RECEIPT_RELATIVE
    if os.path.lexists(final_path):
        _require(_is_regular(final_path), "upstream final receipt is not regular")
        final = verify_post36_orchestration_receipt(
            episode_root=root, receipt_path=final_path
        )
        _require(
            final["intent_sha256"] == authority["intent_sha256"]
            and final["replacement"]["slurm_job_id"]
            == replacement["slurm_job_id"]
            and final["start_authorization"]["start_authorization_sha256"]
            == authority["start_authorization_sha256"],
            "upstream final receipt differs from immutable start authority",
        )
    return authority


def _transition_evidence(*, root: Path, executor: Path) -> dict[str, Any]:
    manifest = root / TRANSITION_ROOT_RELATIVE / TRANSITION_MANIFEST_NAME
    evidence = verify_scientific_contract_transition(
        episode_root=root, executor_root=executor, manifest_path=manifest
    )
    _require(
        evidence.get("operator_development_outcome_values_displayed") is True
        and evidence.get("values_used_for_transition_decisions") is False
        and evidence.get("final_connectivity_accessed") is False,
        "scientific transition exposure/safety evidence changed",
    )
    return dict(evidence)


def _intent(
    *, root: Path, executor: Path, controller: Mapping[str, Any],
    transition: Mapping[str, Any], bridge: Mapping[str, Any],
    publisher_job_id: str, coordinator_job_id: str, submission_user: str,
) -> dict[str, Any]:
    archive = controller["replacement"]["archive"]
    core = {
        "schema_version": INTENT_SCHEMA,
        "episode_root": str(root),
        "submission_user": submission_user,
        "executor_source": {
            "archived_relative_path": "outputs/executor/ep12_executor/post36_downstream_launch.py",
            "file_sha256": _sha256_file(
                executor / "ep12_executor/post36_downstream_launch.py",
                root=executor, label="staged downstream coordinator",
            ),
            "active_module_under_staged_executor": True,
        },
        "archive": dict(archive),
        "upstream_start_authorization": {
            "path": controller["authorization_path"],
            "file_sha256": controller["authorization_file_sha256"],
            "semantic_sha256": controller["start_authorization_sha256"],
        },
        "replacement_job_id": controller["replacement"]["slurm_job_id"],
        "publisher_job_id": publisher_job_id,
        "coordinator_job_id": coordinator_job_id,
        "source_archive_freeze_bridge": dict(bridge),
        "scientific_transition": dict(transition),
        "stage_order": list(STAGE_ORDER),
        "stage_wrappers": dict(WRAPPERS),
        "qualification_repetitions": int(QUALIFICATION_REPETITIONS),
        "qualification_whole_types": int(QUALIFICATION_WHOLE_TYPES),
        "all_children_submitted_held_before_any_release": True,
        "obsolete_job_is_execution_dependency": False,
        "operator_development_outcome_values_displayed": True,
        "values_used_for_transition_decisions": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "intent_sha256": digest_object(core)}


def _common_arguments(
    *, root: Path, intent: Mapping[str, Any], intent_file_sha256: str, stage: str,
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
    stage: str, predecessor_job_id: str, dependency: str,
) -> dict[str, Any]:
    _require(stage in STAGE_ORDER, "unknown downstream stage")
    _require(predecessor_job_id.isdigit() and int(predecessor_job_id) > 0,
             "downstream predecessor job ID is invalid")
    _require(dependency.startswith("afterok:"), "downstream dependency is invalid")
    archive_root = Path(str(intent["archive"]["archived_root"]))
    wrapper = archive_root / "outputs/executor/slurm" / WRAPPERS[stage]
    wrapper_sha = _sha256_file(
        wrapper, root=archive_root, label=f"archived {stage} wrapper"
    )
    arguments = _common_arguments(
        root=root, intent=intent, intent_file_sha256=intent_file_sha256,
        stage=stage,
    )
    transition = intent["scientific_transition"]
    if stage == "scientific_qualification":
        arguments.extend([
            str(root / QUALIFICATION_OUTPUT_RELATIVE),
            QUALIFICATION_REPETITIONS, QUALIFICATION_WHOLE_TYPES,
            str(transition["transition_manifest_path"]),
            str(transition["transition_manifest_file_sha256"]),
            str(transition["transition_receipt_sha256"]),
        ])
    elif stage == "production_readiness":
        upstream = intent["upstream_start_authorization"]
        arguments.extend([
            str(root / TRANSACTION_RELATIVE / "stages/scientific_qualification/receipt.json"),
            str(root / QUALIFICATION_OUTPUT_RELATIVE),
            str(transition["transition_manifest_path"]),
            str(transition["transition_manifest_file_sha256"]),
            str(transition["transition_receipt_file_sha256"]),
            str(transition["transition_receipt_sha256"]),
            str(transition["observed_prefix_contract_sha256"]),
            str(transition["active_contract_sha256"]),
            str(transition["active_qualification_spec_sha256"]),
            str(transition["active_scientific_models_sha256"]),
            str(upstream["path"]), str(upstream["file_sha256"]),
            str(upstream["semantic_sha256"]), str(intent["replacement_job_id"]),
        ])
    core = {
        "schema_version": STAGE_INTENT_SCHEMA,
        "orchestration_intent_sha256": intent["intent_sha256"],
        "stage": stage,
        "submission_user": intent["submission_user"],
        "predecessor_job_id": predecessor_job_id,
        "dependency": dependency,
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


def _verify_stage(
    *, root: Path, intent: Mapping[str, Any], intent_file_sha256: str,
    stage: str, predecessor_job_id: str, dependency: str,
) -> dict[str, Any]:
    stage_dir = root / TRANSACTION_RELATIVE / "stages" / stage
    expected_intent = build_stage_intent(
        root=root, intent=intent, intent_file_sha256=intent_file_sha256,
        stage=stage, predecessor_job_id=predecessor_job_id,
        dependency=dependency,
    )
    persisted_intent = _read_json(
        stage_dir / "intent.json", root=root, label=f"{stage} intent",
        canonical=True,
    )
    _require(persisted_intent == expected_intent, f"{stage} intent mutated")
    state, job_id, progress = _verify_stage_progress(
        _read_json(
            stage_dir / "progress.json", root=root, label=f"{stage} progress",
            canonical=True,
        ),
        stage_intent=expected_intent,
    )
    _require(state == "submitted" and job_id is not None, f"{stage} is not submitted")
    receipt = _read_json(
        stage_dir / "receipt.json", root=root, label=f"{stage} receipt",
        canonical=True,
    )
    _require(
        receipt == _stage_receipt(
            stage_intent=expected_intent, progress=progress, job_id=job_id
        ),
        f"{stage} receipt mutated",
    )
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
    _require(
        release_progress == _release_progress(
            release_intent=release_intent, state="released"
        ),
        f"{stage} is not durably released",
    )
    release_receipt = _read_json(
        stage_dir / "release_receipt.json", root=root,
        label=f"{stage} release receipt", canonical=True,
    )
    _require(
        release_receipt == _release_receipt(
            release_intent=release_intent, release_progress=release_progress
        ),
        f"{stage} release receipt mutated",
    )
    return _stage_submission_evidence(
        stage_dir=stage_dir, root=root, stage_receipt=receipt,
        release_receipt=release_receipt,
    )


def _receipt(
    *, root: Path, intent: Mapping[str, Any], intent_file_sha256: str,
    stages: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    core = {
        "schema_version": RECEIPT_SCHEMA,
        "intent_path": str(root / INTENT_RELATIVE),
        "intent_file_sha256": intent_file_sha256,
        "intent_sha256": intent["intent_sha256"],
        "archive": dict(intent["archive"]),
        "upstream_start_authorization": dict(
            intent["upstream_start_authorization"]
        ),
        "replacement_job_id": intent["replacement_job_id"],
        "publisher_job_id": intent["publisher_job_id"],
        "coordinator_job_id": intent["coordinator_job_id"],
        "scientific_transition": dict(intent["scientific_transition"]),
        "stages": {stage: dict(stages[stage]) for stage in STAGE_ORDER},
        "all_child_receipts_durable_before_coordinator_success": True,
        "all_child_release_receipts_durable_before_coordinator_success": True,
        "completed_finalizer_is_scheduler_dependency": False,
        "obsolete_job_is_execution_dependency": False,
        "operator_development_outcome_values_displayed": True,
        "values_used_for_transition_decisions": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return {**core, "receipt_sha256": digest_object(core)}


def launch_post36_downstream(
    *, episode_root: Path, executor_root: Path, orchestration_intent: Path,
    orchestration_intent_sha256: str, publisher_job_id: str,
    coordinator_job_id: str, submitter: Submitter = default_submitter,
    job_resolver: JobResolver = default_exact_name_resolver,
    job_state_resolver: JobStateResolver = default_job_state_resolver,
    releaser: Releaser = default_releaser,
) -> dict[str, Any]:
    root = _lexical(episode_root)
    _require(root == EPISODE_ROOT, "downstream episode root is not canonical")
    _require(publisher_job_id.isdigit() and int(publisher_job_id) > 0,
             "publisher job ID is invalid")
    _require(coordinator_job_id.isdigit() and int(coordinator_job_id) > 0,
             "coordinator job ID is invalid")
    executor = _active_executor(executor_root)
    try:
        _read_json(root / RECEIPT_RELATIVE, root=root,
                   label="downstream launch receipt", canonical=True)
    except FileNotFoundError:
        pass
    else:
        return verify_post36_downstream_receipt(
            episode_root=root, executor_root=executor,
            orchestration_intent=orchestration_intent,
            orchestration_intent_sha256=orchestration_intent_sha256,
            publisher_job_id=publisher_job_id,
            coordinator_job_id=coordinator_job_id,
        )

    controller = _controller_authority(
        root=root, upstream_intent=orchestration_intent,
        expected_intent_sha256=orchestration_intent_sha256,
        publisher_job_id=publisher_job_id,
        coordinator_job_id=coordinator_job_id,
    )
    archive = controller["replacement"]["archive"]
    bridge = _stable_bridge(bridge_source_archive_to_live_freeze(
        archive_root=Path(str(archive["archived_root"])),
        archive_receipt_file_sha256=str(archive["archive_receipt_file_sha256"]),
        archive_receipt_sha256=str(archive["archive_receipt_sha256"]),
        archive_manifest_file_sha256=str(archive["manifest_file_sha256"]),
        archive_manifest_sha256=str(archive["manifest_sha256"]),
        archive_tree_sha256=str(archive["tree_sha256"]),
        live_episode_root=root, require_canonical_live_root=True,
    ))
    transition = _transition_evidence(root=root, executor=executor)
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
                    root=root, executor=executor, controller=controller,
                    transition=transition, bridge=bridge,
                    publisher_job_id=publisher_job_id,
                    coordinator_job_id=coordinator_job_id,
                    submission_user=_effective_user(),
                ),
                root=root, label="downstream launch intent",
            )
            intent_file_sha = _sha256_file(
                root / INTENT_RELATIVE, root=root, label="downstream launch intent"
            )
            held: dict[str, dict[str, Any]] = {}
            stage_intents: dict[str, dict[str, Any]] = {}
            predecessor = coordinator_job_id
            for stage in STAGE_ORDER:
                dependency = f"afterok:{predecessor}"
                stage_intent = build_stage_intent(
                    root=root, intent=intent, intent_file_sha256=intent_file_sha,
                    stage=stage, predecessor_job_id=predecessor,
                    dependency=dependency,
                )
                stage_intents[stage] = stage_intent
                held[stage] = _submit_stage_held(
                    transaction_root=transaction, root=root,
                    stage_intent=stage_intent, submitter=submitter,
                    job_resolver=job_resolver,
                    job_state_resolver=job_state_resolver,
                )
                predecessor = str(held[stage]["slurm_job_id"])
            released: dict[str, dict[str, Any]] = {}
            for stage in STAGE_ORDER:
                released[stage] = _release_submitted_stage(
                    transaction_root=transaction, root=root,
                    stage_intent=stage_intents[stage], stage_receipt=held[stage],
                    job_state_resolver=job_state_resolver, releaser=releaser,
                )
            receipt = _receipt(
                root=root, intent=intent, intent_file_sha256=intent_file_sha,
                stages=released,
            )
            _write_or_verify(
                root / RECEIPT_RELATIVE, receipt, root=root,
                label="downstream launch receipt",
            )
            return verify_post36_downstream_receipt(
                episode_root=root, executor_root=executor,
                orchestration_intent=orchestration_intent,
                orchestration_intent_sha256=orchestration_intent_sha256,
                publisher_job_id=publisher_job_id,
                coordinator_job_id=coordinator_job_id,
            )
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


def verify_post36_downstream_receipt(
    *, episode_root: Path, executor_root: Path, orchestration_intent: Path,
    orchestration_intent_sha256: str, publisher_job_id: str,
    coordinator_job_id: str,
) -> dict[str, Any]:
    root = _lexical(episode_root)
    _require(root == EPISODE_ROOT, "downstream verifier root is not canonical")
    executor = _active_executor(executor_root)
    controller = _controller_authority(
        root=root, upstream_intent=orchestration_intent,
        expected_intent_sha256=orchestration_intent_sha256,
        publisher_job_id=publisher_job_id,
        coordinator_job_id=coordinator_job_id,
    )
    archive = controller["replacement"]["archive"]
    bridge = _stable_bridge(bridge_source_archive_to_live_freeze(
        archive_root=Path(str(archive["archived_root"])),
        archive_receipt_file_sha256=str(archive["archive_receipt_file_sha256"]),
        archive_receipt_sha256=str(archive["archive_receipt_sha256"]),
        archive_manifest_file_sha256=str(archive["manifest_file_sha256"]),
        archive_manifest_sha256=str(archive["manifest_sha256"]),
        archive_tree_sha256=str(archive["tree_sha256"]),
        live_episode_root=root, require_canonical_live_root=True,
    ))
    transition = _transition_evidence(root=root, executor=executor)
    intent = _read_json(
        root / INTENT_RELATIVE, root=root, label="downstream launch intent",
        canonical=True,
    )
    expected_intent = _intent(
        root=root, executor=executor, controller=controller,
        transition=transition, bridge=bridge,
        publisher_job_id=publisher_job_id,
        coordinator_job_id=coordinator_job_id,
        submission_user=_effective_user(),
    )
    _require(intent == expected_intent, "downstream launch intent mutated")
    intent_file_sha = _sha256_file(
        root / INTENT_RELATIVE, root=root, label="downstream launch intent"
    )
    stages: dict[str, dict[str, Any]] = {}
    predecessor = coordinator_job_id
    for stage in STAGE_ORDER:
        dependency = f"afterok:{predecessor}"
        stages[stage] = _verify_stage(
            root=root, intent=intent, intent_file_sha256=intent_file_sha,
            stage=stage, predecessor_job_id=predecessor,
            dependency=dependency,
        )
        predecessor = str(stages[stage]["slurm_job_id"])
    receipt = _read_json(
        root / RECEIPT_RELATIVE, root=root, label="downstream launch receipt",
        canonical=True,
    )
    expected_receipt = _receipt(
        root=root, intent=intent, intent_file_sha256=intent_file_sha,
        stages=stages,
    )
    _require(receipt == expected_receipt, "downstream launch receipt mutated")
    dependencies = [str(stages[stage]["dependency"]) for stage in STAGE_ORDER]
    _require(all("45332084" not in value for value in dependencies),
             "obsolete attestation appears in a downstream dependency")
    _require(
        all(OBSERVED_FINALIZER_JOB_ID not in value for value in dependencies),
        "completed finalizer appears in a downstream scheduler dependency",
    )
    return {
        "schema_version": "ep12.post36_downstream_launch_evidence.v1",
        "intent_path": str(root / INTENT_RELATIVE),
        "intent_file_sha256": intent_file_sha,
        "intent_sha256": intent["intent_sha256"],
        "receipt_path": str(root / RECEIPT_RELATIVE),
        "receipt_file_sha256": _sha256_file(
            root / RECEIPT_RELATIVE, root=root,
            label="downstream launch receipt",
        ),
        "receipt_sha256": receipt["receipt_sha256"],
        "archive": dict(intent["archive"]),
        "upstream_start_authorization": dict(
            intent["upstream_start_authorization"]
        ),
        "replacement_job_id": intent["replacement_job_id"],
        "scientific_transition": dict(intent["scientific_transition"]),
        "stages": stages,
        "completed_finalizer_is_scheduler_dependency": False,
        "obsolete_job_is_execution_dependency": False,
        "operator_development_outcome_values_displayed": True,
        "values_used_for_transition_decisions": False,
        "real_connectivity_accessed": False,
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--executor-root", type=Path, required=True)
    parser.add_argument("--orchestration-intent", type=Path, required=True)
    parser.add_argument("--orchestration-intent-sha256", required=True)
    parser.add_argument("--publisher-job-id", required=True)
    parser.add_argument("--coordinator-job-id", required=True)
    arguments = parser.parse_args(argv)
    result = launch_post36_downstream(
        episode_root=arguments.episode_root,
        executor_root=arguments.executor_root,
        orchestration_intent=arguments.orchestration_intent,
        orchestration_intent_sha256=arguments.orchestration_intent_sha256,
        publisher_job_id=arguments.publisher_job_id,
        coordinator_job_id=arguments.coordinator_job_id,
    )
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
