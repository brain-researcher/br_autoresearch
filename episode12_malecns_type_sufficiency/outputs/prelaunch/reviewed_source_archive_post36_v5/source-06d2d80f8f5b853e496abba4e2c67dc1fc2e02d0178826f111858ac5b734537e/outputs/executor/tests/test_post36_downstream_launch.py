from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ep12_executor import post36_downstream_launch as downstream


def _patch(
    patches: ExitStack, target: object, name: str, value: object
) -> None:
    patches.enter_context(mock.patch.object(target, name, value))


def _canonical(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path):
    root = tmp_path / "episode"
    archive_root = tmp_path / "archive"
    staged_executor = tmp_path / "stage/outputs/executor"
    (archive_root / "outputs/executor/slurm").mkdir(parents=True)
    (staged_executor / "ep12_executor").mkdir(parents=True)
    (staged_executor / "ep12_executor/post36_downstream_launch.py").write_text(
        "# synthetic staged coordinator\n", encoding="utf-8"
    )
    for wrapper in downstream.WRAPPERS.values():
        (archive_root / "outputs/executor/slurm" / wrapper).write_text(
            "#!/bin/bash\n", encoding="utf-8"
        )
    archive = {
        "archived_root": str(archive_root),
        "archive_receipt_path": str(tmp_path / "source_archive_receipt.json"),
        "archive_receipt_file_sha256": "1" * 64,
        "archive_receipt_sha256": "2" * 64,
        "manifest_path": str(archive_root / "source_archive_manifest.json"),
        "manifest_file_sha256": "3" * 64,
        "manifest_sha256": "4" * 64,
        "tree_sha256": "5" * 64,
    }
    upstream_intent = root / downstream.UPSTREAM_INTENT_RELATIVE
    _canonical(upstream_intent, {"intent_sha256": "6" * 64})
    controller = {
        "intent_path": str(upstream_intent),
        "intent_file_sha256": "7" * 64,
        "intent_sha256": "6" * 64,
        "authorization_path": str(root / downstream.UPSTREAM_AUTHORIZATION_RELATIVE),
        "authorization_file_sha256": "8" * 64,
        "start_authorization_sha256": "9" * 64,
        "replacement": {"slurm_job_id": "400", "archive": archive},
        "stages": {
            "publisher": {"slurm_job_id": "450"},
            "downstream": {"slurm_job_id": "500"},
        },
    }
    transition = {
        "transition_manifest_path": str(
            root / "outputs/prelaunch/scientific_contract_transition/manifest.json"
        ),
        "transition_manifest_file_sha256": "a" * 64,
        "transition_receipt_file_sha256": "b" * 64,
        "transition_receipt_sha256": "c" * 64,
        "observed_prefix_contract_sha256": "d" * 64,
        "active_contract_sha256": "e" * 64,
        "active_qualification_spec_sha256": "f" * 64,
        "active_scientific_models_sha256": "0" * 64,
        "operator_development_outcome_values_displayed": True,
        "values_used_for_transition_decisions": False,
        "final_connectivity_accessed": False,
    }
    bridge = {
        "publication_status": "created",
        "receipt_path": str(
            root / "outputs/prelaunch/source_archive_freeze_bridge/receipt.json"
        ),
        "receipt_file_sha256": "1" * 64,
        "receipt_sha256": "2" * 64,
        "archive_identity": archive,
        "live_episode_root": str(root),
        "source_freeze_identity": {
            "schema_version": "ep12.executable_source_freeze_manifest.v1",
            "manifest_path": str(root / "outputs/prelaunch/executable_source_freeze_manifest.json"),
            "manifest_file_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "file_count": 1,
        },
        "executable_record_map_sha256": "5" * 64,
        "file_count": 1,
        "archive_and_live_file_maps_identical": True,
        "archive_modes_bound_upstream_not_required_live_equal": True,
        "archive_reverified_after_freeze": True,
        "live_surface_reverified_after_freeze": True,
        "real_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    return root, archive_root, staged_executor, upstream_intent, controller, transition, bridge


def _case_downstream_dag_receipts_precede_release_and_completed_replay_is_inert(
    case: unittest.TestCase, tmp_path: Path, patches: ExitStack
) -> None:
    (
        root, archive_root, executor, upstream_intent, controller,
        transition, bridge,
    ) = _fixture(tmp_path)
    _patch(patches, downstream, "EPISODE_ROOT", root)
    _patch(patches, downstream, "_active_executor", lambda _root: executor)
    _patch(patches, downstream, "_controller_authority", lambda **_kwargs: controller)
    _patch(patches, downstream, "_transition_evidence", lambda **_kwargs: transition)
    _patch(
        patches,
        downstream,
        "bridge_source_archive_to_live_freeze",
        lambda **_kwargs: bridge,
    )
    _patch(patches, downstream, "_effective_user", lambda: "zijiao")

    scheduler: dict[str, tuple[str, str]] = {}
    submissions: list[list[str]] = []
    releases: list[str] = []
    next_job = 600

    def submit(command: list[str]) -> str:
        nonlocal next_job
        job_id = str(next_job)
        next_job += 1
        name = next(
            token.split("=", 1)[1]
            for token in command if token.startswith("--job-name=")
        )
        scheduler[job_id] = ("PENDING", "JobHeldUser")
        submissions.append(command)
        return job_id

    def state(job_id: str, job_name: str, user: str):
        state_name, reason = scheduler[job_id]
        return {
            "job_id": job_id, "job_name": job_name, "user": user,
            "state": state_name, "reason": reason,
        }

    def release(command: list[str]) -> str:
        assert len(submissions) == 3
        for stage in downstream.STAGE_ORDER:
            assert (
                root / downstream.TRANSACTION_RELATIVE / "stages" / stage / "receipt.json"
            ).is_file()
        job_id = command[-1]
        scheduler[job_id] = ("PENDING", "Dependency")
        releases.append(job_id)
        return ""

    kwargs = {
        "episode_root": root,
        "executor_root": executor,
        "orchestration_intent": upstream_intent,
        "orchestration_intent_sha256": "6" * 64,
        "publisher_job_id": "450",
        "coordinator_job_id": "500",
        "submitter": submit,
        "job_resolver": lambda _name, _user: None,
        "job_state_resolver": state,
        "releaser": release,
    }
    first = downstream.launch_post36_downstream(**kwargs)
    assert releases == ["600", "601", "602"]
    assert first["stages"]["synthetic_tests"]["dependency"] == "afterok:500"
    assert first["stages"]["scientific_qualification"]["dependency"] == "afterok:600"
    assert first["stages"]["production_readiness"]["dependency"] == "afterok:601"
    assert first["completed_finalizer_is_scheduler_dependency"] is False
    assert all("45323481" not in " ".join(command) for command in submissions)
    assert all("45332084" not in " ".join(command) for command in submissions)
    assert (
        root / downstream.RECEIPT_RELATIVE
    ).is_file(), "coordinator must publish before afterok:500 can be satisfied"

    def poisoned(*_args, **_kwargs):
        raise AssertionError("completed replay performed scheduler I/O")

    replay = downstream.launch_post36_downstream(
        **{
            **kwargs,
            "submitter": poisoned,
            "job_resolver": poisoned,
            "job_state_resolver": poisoned,
            "releaser": poisoned,
        }
    )
    assert replay == first


def _case_controller_authority_allows_absent_final_but_rejects_present_tamper(
    case: unittest.TestCase, tmp_path: Path, patches: ExitStack
) -> None:
    root, _archive, _executor, upstream, controller, _transition, _bridge = _fixture(
        tmp_path
    )
    _patch(
        patches,
        downstream,
        "verify_post36_start_authorization",
        lambda **_kwargs: controller,
    )
    authority = downstream._controller_authority(
        root=root, upstream_intent=upstream,
        expected_intent_sha256="6" * 64,
        publisher_job_id="450", coordinator_job_id="500",
    )
    assert authority == controller

    _canonical(root / downstream.UPSTREAM_RECEIPT_RELATIVE, {"tampered": True})
    _patch(
        patches,
        downstream,
        "verify_post36_orchestration_receipt",
        lambda **_kwargs: (_ for _ in ()).throw(
            downstream.Post36DownstreamLaunchError("tampered final receipt")
        ),
    )
    with case.assertRaisesRegex(
        downstream.Post36DownstreamLaunchError, "tampered final receipt"
    ):
        downstream._controller_authority(
            root=root, upstream_intent=upstream,
            expected_intent_sha256="6" * 64,
            publisher_job_id="450", coordinator_job_id="500",
        )


class Post36DownstreamLaunchTests(unittest.TestCase):
    def _run_with_temporary_path(self, function) -> None:
        with (
            tempfile.TemporaryDirectory(prefix="ep12-downstream-") as directory,
            ExitStack() as patches,
        ):
            function(self, Path(directory), patches)

    def test_downstream_dag_receipts_precede_release_and_completed_replay_is_inert(
        self,
    ) -> None:
        self._run_with_temporary_path(
            _case_downstream_dag_receipts_precede_release_and_completed_replay_is_inert
        )

    def test_controller_authority_allows_absent_final_but_rejects_present_tamper(
        self,
    ) -> None:
        self._run_with_temporary_path(
            _case_controller_authority_allows_absent_final_but_rejects_present_tamper
        )


if __name__ == "__main__":
    unittest.main()
