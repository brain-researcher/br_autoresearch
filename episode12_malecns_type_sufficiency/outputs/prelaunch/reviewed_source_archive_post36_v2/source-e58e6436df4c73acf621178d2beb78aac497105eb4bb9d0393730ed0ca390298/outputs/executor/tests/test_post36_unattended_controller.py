from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ep12_executor.policy import digest_object
from ep12_executor import post36_orchestration as primitives
from ep12_executor import replacement_attestation_launch as replacement_launch
from ep12_executor import post36_unattended_controller as controller


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


def _stage_intent(root: Path, wrapper: Path) -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": primitives.STAGE_INTENT_SCHEMA,
        "orchestration_intent_sha256": "a" * 64,
        "stage": "replacement_attestation",
        "submission_user": "zijiao",
        "predecessor_job_id": "99",
        "dependency": "afterok:99",
        "kill_on_invalid_dependency": True,
        "submission_barrier": "user_hold_receipt_before_release_v1",
        "wrapper": str(wrapper),
        "wrapper_sha256": primitives._sha256_file(
            wrapper, root=root, label="synthetic replacement wrapper"
        ),
        "arguments": ["replacement"],
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    named = {**core, "job_name": "synthetic-replacement"}
    return {**named, "stage_intent_sha256": digest_object(named)}


def _case_dynamic_r_dag_is_fully_receipted_before_r_release_and_replay_is_inert(
    case: unittest.TestCase, tmp_path: Path, patches: ExitStack
) -> None:
    root = tmp_path / "episode"
    archive_root = tmp_path / "archive"
    executor = archive_root / "outputs/executor"
    (executor / "slurm").mkdir(parents=True)
    (executor / "ep12_executor").mkdir(parents=True)
    for name in controller.WRAPPERS.values():
        (executor / "slurm" / name).write_text("#!/bin/bash\n", encoding="utf-8")

    archive = {
        "archived_root": str(archive_root),
        "archive_receipt_path": str(tmp_path / "source_archive_receipt.json"),
        "archive_receipt_file_sha256": "1" * 64,
        "archive_receipt_sha256": "2" * 64,
        "manifest_path": str(archive_root / "source_archive_manifest.json"),
        "manifest_file_sha256": "3" * 64,
        "manifest_sha256": "4" * 64,
        "tree_sha256": "5" * 64,
        "bootstrap_sha256": "6" * 64,
    }
    replacement_dir = (
        root / "outputs/prelaunch/replacement_attestation_launch"
    )
    replacement_stage = replacement_dir / "stages/replacement_attestation"
    replacement_stage.mkdir(parents=True)
    replacement_wrapper = root / "replacement.sbatch"
    replacement_wrapper.parent.mkdir(parents=True, exist_ok=True)
    replacement_wrapper.write_text("#!/bin/bash\n", encoding="utf-8")
    r_intent = _stage_intent(root, replacement_wrapper)
    r_progress = primitives._stage_progress(
        stage_intent=r_intent, state="submitted", job_id="100"
    )
    r_receipt = primitives._stage_receipt(
        stage_intent=r_intent, progress=r_progress, job_id="100"
    )
    _canonical(replacement_stage / "intent.json", r_intent)
    _canonical(replacement_stage / "progress.json", r_progress)
    _canonical(replacement_stage / "receipt.json", r_receipt)

    cancellation_path = replacement_dir / "obsolete_cancellation/receipt.json"
    cancellation_core = {
        "schema_version": "synthetic-cancellation",
        "terminal_scheduler_state": "CANCELLED",
        "replacement_job_id": "100",
    }
    cancellation_receipt = {
        **cancellation_core, "receipt_sha256": digest_object(cancellation_core)
    }
    _canonical(cancellation_path, cancellation_receipt)
    cancellation = {
        "cancellation_receipt_path": str(cancellation_path),
        "cancellation_receipt_file_sha256": primitives._sha256_file(
            cancellation_path, root=root, label="synthetic cancellation"
        ),
        "cancellation_receipt_sha256": cancellation_receipt["receipt_sha256"],
        "obsolete_job_id": "45332084",
        "replacement_job_id": "100",
        "superseded_result_authoritative": False,
        "final_connectivity_accessed": False,
    }

    base_replacement = {
        "launch_intent_path": str(replacement_dir / "intent.json"),
        "launch_intent_file_sha256": "7" * 64,
        "launch_intent_sha256": "8" * 64,
        "launch_receipt_path": str(replacement_dir / "receipt.json"),
        "launch_receipt_file_sha256": "9" * 64,
        "launch_receipt_sha256": "a" * 64,
        "stage_intent_path": str(replacement_stage / "intent.json"),
        "stage_intent_file_sha256": primitives._sha256_file(
            replacement_stage / "intent.json", root=root, label="R intent"
        ),
        "stage_intent_sha256": r_intent["stage_intent_sha256"],
        "stage_receipt_path": str(replacement_stage / "receipt.json"),
        "stage_receipt_file_sha256": primitives._sha256_file(
            replacement_stage / "receipt.json", root=root, label="R receipt"
        ),
        "stage_receipt_sha256": r_receipt["stage_receipt_sha256"],
        "slurm_job_id": "100",
        "slurm_job_name": "synthetic-replacement",
        "dependency": "afterok:99",
        "wrapper": str(replacement_wrapper),
        "wrapper_sha256": r_intent["wrapper_sha256"],
        "arguments": ["replacement"],
        "archive": archive,
        "terminal_identity_path": str(replacement_dir / "terminal_identity.json"),
        "attestation_output": str(root / "outputs/development_run002/observed_attestation"),
        "superseded_job_id": "45332084",
        "obsolete_hold_file_sha256": "b" * 64,
    }

    scheduler: dict[str, tuple[str, str]] = {"100": ("PENDING", "JobHeldUser")}
    submitted: list[list[str]] = []
    released: list[str] = []
    next_job = 200

    def submit(command: list[str]) -> str:
        nonlocal next_job
        job = str(next_job)
        next_job += 1
        name = next(item.split("=", 1)[1] for item in command if item.startswith("--job-name="))
        scheduler[job] = ("PENDING", "JobHeldUser")
        submitted.append(command)
        return job

    def resolve_name(_name: str, _user: str):
        return None

    def resolve_state(job: str, name: str, user: str):
        state, reason = scheduler[job]
        return {"job_id": job, "job_name": name, "user": user,
                "state": state, "reason": reason}

    def release(command: list[str]) -> str:
        job = command[-1]
        if job == "100":
            start = root / controller.START_AUTHORIZATION_RELATIVE
            assert start.is_file()
            for stage in controller.STAGE_ORDER:
                assert (root / controller.TRANSACTION_RELATIVE / "stages" / stage / "release_receipt.json").is_file()
        scheduler[job] = ("PENDING", "Dependency")
        released.append(job)
        return ""

    def replacement_evidence(*, episode_root: Path, require_released=None, **_kwargs):
        assert episode_root == root
        final = replacement_stage / "release_receipt.json"
        evidence = dict(base_replacement)
        if final.exists():
            release_receipt = json.loads(final.read_text())
            authorization = json.loads(
                (replacement_stage / "release_authorization.json").read_text()
            )
            release_intent = json.loads(
                (replacement_stage / "release_intent.json").read_text()
            )
            evidence.update({
                "replacement_released": True,
                "release_intent_path": str(replacement_stage / "release_intent.json"),
                "release_intent_file_sha256": primitives._sha256_file(
                    replacement_stage / "release_intent.json", root=root,
                    label="R release intent",
                ),
                "release_intent_sha256": release_intent["release_intent_sha256"],
                "release_authorized": True,
                "release_authorization_path": str(replacement_stage / "release_authorization.json"),
                "release_authorization_file_sha256": primitives._sha256_file(
                    replacement_stage / "release_authorization.json", root=root,
                    label="R release authorization",
                ),
                "release_authorization_sha256": authorization["release_authorization_sha256"],
                "release_receipt_path": str(final),
                "release_receipt_file_sha256": primitives._sha256_file(
                    final, root=root, label="R release receipt"
                ),
                "release_receipt_sha256": release_receipt["release_receipt_sha256"],
            })
        else:
            assert require_released is not True
            evidence.update({
                "replacement_released": False,
                "release_intent_path": None,
                "release_intent_file_sha256": None,
                "release_intent_sha256": None,
                "release_authorized": False,
                "release_authorization_path": None,
                "release_authorization_file_sha256": None,
                "release_authorization_sha256": None,
                "release_receipt_path": None,
                "release_receipt_file_sha256": None,
                "release_receipt_sha256": None,
            })
        return evidence

    _patch(patches, controller, "EPISODE_ROOT", root)
    _patch(patches, controller, "_archive_identity", lambda **_kwargs: archive)
    _patch(patches, controller, "_pre_cancel_preflight", lambda *_args, **_kwargs: None)
    _patch(patches, controller, "launch_replacement_attestation", lambda **_kwargs: {})
    _patch(patches, controller, "verify_replacement_attestation_receipt", replacement_evidence)
    _patch(patches, controller, "cancel_obsolete_attestation", lambda **_kwargs: {})
    _patch(patches, controller, "verify_obsolete_attestation_cancellation_receipt", lambda **_kwargs: cancellation)
    _patch(patches, controller, "_controller_source", lambda *_args, **_kwargs: {"archived_path": "synthetic", "file_sha256": "c" * 64, "active_bytes_match_archive": True})
    _patch(patches, controller, "_legacy_evidence", lambda _root: {"schema_version": "synthetic-legacy", "final_connectivity_accessed": False})
    _patch(patches, controller, "_incident_evidence", lambda _root: {"path": "synthetic-incident", "operator_development_outcome_values_displayed": True, "values_used_for_transition_decisions": False})
    _patch(patches, controller, "_effective_user", lambda: "zijiao")

    kwargs = dict(
        episode_root=root, archive_root=archive_root,
        archive_receipt_file_sha256="1" * 64,
        archive_receipt_sha256="2" * 64,
        archive_manifest_file_sha256="3" * 64,
        archive_manifest_sha256="4" * 64,
        archive_tree_sha256="5" * 64,
        submitter=submit, job_resolver=resolve_name,
        job_state_resolver=resolve_state, releaser=release,
    )
    first = controller.orchestrate_post36_unattended(**kwargs)
    assert [item["stage"] for item in first["stages"].values()] == list(controller.STAGE_ORDER)
    assert released == ["200", "201", "202", "203", "100"]
    assert all("45332084" not in " ".join(command) for command in submitted)
    publisher_arguments = first["stages"]["publisher"]["arguments"]
    assert publisher_arguments[-3:] == [
        str(replacement_dir / "terminal_identity.json"),
        str(root / controller.START_AUTHORIZATION_RELATIVE),
        str(root / controller.CANCELLATION_RECEIPT_RELATIVE),
    ]

    def poisoned(*_args, **_kwargs):
        raise AssertionError("scheduler was touched during completed replay")

    replay = controller.orchestrate_post36_unattended(
        **{**kwargs, "submitter": poisoned, "job_resolver": poisoned,
           "job_state_resolver": poisoned, "releaser": poisoned}
    )
    assert replay == first

    # Exercise the real replacement-release verifier, not the controller's
    # synthetic replacement-evidence stub. Both the pre-ack uncertain state
    # and the final released state must bind the exact non-null start context.
    release_intent_path = replacement_stage / "release_intent.json"
    release_authorization_path = replacement_stage / "release_authorization.json"
    release_progress_path = replacement_stage / "release_progress.json"
    release_receipt_path = replacement_stage / "release_receipt.json"
    exact_intent = json.loads(release_intent_path.read_text(encoding="utf-8"))
    exact_authorization = json.loads(
        release_authorization_path.read_text(encoding="utf-8")
    )
    exact_progress = json.loads(release_progress_path.read_text(encoding="utf-8"))
    exact_receipt = json.loads(release_receipt_path.read_text(encoding="utf-8"))
    released_evidence = replacement_launch._verified_release_evidence(
        root=root, stage_receipt=r_receipt, require_released=True
    )
    assert released_evidence["replacement_released"] is True
    assert released_evidence["release_authorization_context"] == exact_intent[
        "authorization_context"
    ]

    release_receipt_path.unlink()
    _canonical(
        release_progress_path,
        primitives._release_progress(
            release_intent=exact_intent, state="release_uncertain"
        ),
    )
    uncertain_evidence = replacement_launch._verified_release_evidence(
        root=root, stage_receipt=r_receipt, require_released=None
    )
    assert uncertain_evidence["replacement_released"] is False
    assert uncertain_evidence["release_authorized"] is True

    for key, bad_value, match in (
        ("start_authorization_path", str(root / "wrong.json"), "path is not canonical"),
        ("start_authorization_file_sha256", "0" * 64, "start authorization changed"),
        ("schema_version", "ep12.bad.v1", "context schema changed"),
    ):
        bad_context = dict(exact_intent["authorization_context"])
        bad_context[key] = bad_value
        bad_intent = primitives._release_intent(
            stage_receipt=r_receipt,
            stage_receipt_file_sha256=primitives._sha256_file(
                replacement_stage / "receipt.json", root=root,
                label="R receipt for tamper test",
            ),
            authorization_context=bad_context,
        )
        _canonical(release_intent_path, bad_intent)
        _canonical(
            release_authorization_path,
            primitives._release_authorization(release_intent=bad_intent),
        )
        _canonical(
            release_progress_path,
            primitives._release_progress(
                release_intent=bad_intent, state="release_uncertain"
            ),
        )
        with case.assertRaisesRegex(
            replacement_launch.ReplacementAttestationLaunchError, match
        ):
            replacement_launch._verified_release_evidence(
                root=root, stage_receipt=r_receipt, require_released=None
            )

    _canonical(release_intent_path, exact_intent)
    _canonical(release_authorization_path, exact_authorization)
    _canonical(release_progress_path, exact_progress)
    _canonical(release_receipt_path, exact_receipt)


def _case_release_uncertain_held_refuses_duplicate_scontrol(
    case: unittest.TestCase, tmp_path: Path, patches: ExitStack,
) -> None:
    root = tmp_path / "episode"
    transaction = root / "outputs/prelaunch/txn"
    transaction.mkdir(parents=True)
    wrapper = root / "wrapper.sbatch"
    wrapper.write_text("#!/bin/bash\n", encoding="utf-8")
    core = {
        "schema_version": primitives.STAGE_INTENT_SCHEMA,
        "orchestration_intent_sha256": "d" * 64,
        "stage": "synthetic",
        "submission_user": "zijiao",
        "predecessor_job_id": "1",
        "dependency": "afterok:1",
        "kill_on_invalid_dependency": True,
        "submission_barrier": "user_hold_receipt_before_release_v1",
        "wrapper": str(wrapper),
        "wrapper_sha256": primitives._sha256_file(wrapper, root=root, label="wrapper"),
        "arguments": ["x"],
        "final_connectivity_accessed": False,
        "final_connectivity_access_authorized": False,
    }
    named = {**core, "job_name": "synthetic-release"}
    intent = {**named, "stage_intent_sha256": digest_object(named)}
    calls = 0

    def held(job: str, name: str, user: str):
        return {"job_id": job, "job_name": name, "user": user,
                "state": "PENDING", "reason": "JobHeldUser"}

    def lost(_command: list[str]) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("lost release acknowledgement")

    with case.assertRaisesRegex(RuntimeError, "lost release"):
        primitives._submit_stage(
            transaction_root=transaction, root=root, stage_intent=intent,
            submitter=lambda _command: "10",
            job_resolver=lambda _name, _user: None,
            job_state_resolver=held, releaser=lost,
        )
    with case.assertRaisesRegex(
        primitives.Post36OrchestrationError, "refusing duplicate"
    ):
        primitives._submit_stage(
            transaction_root=transaction, root=root, stage_intent=intent,
            submitter=lambda _command: (_ for _ in ()).throw(AssertionError()),
            job_resolver=lambda _name, _user: None,
            job_state_resolver=held,
            releaser=lambda _command: (_ for _ in ()).throw(AssertionError()),
        )
    assert calls == 1


def _case_pre_cancel_preflight_failure_has_no_scheduler_or_cancellation_side_effects(
    case: unittest.TestCase, tmp_path: Path, patches: ExitStack
) -> None:
    root = tmp_path / "episode"
    root.mkdir()
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    archive = {
        "archived_root": str(archive_root),
        "archive_receipt_file_sha256": "1" * 64,
        "archive_receipt_sha256": "2" * 64,
        "manifest_file_sha256": "3" * 64,
        "manifest_sha256": "4" * 64,
        "tree_sha256": "5" * 64,
    }
    calls: list[str] = []

    def reject_preflight(*_args, **_kwargs):
        calls.append("preflight")
        raise controller.Post36UnattendedControllerError("synthetic archive collision")

    def forbidden(label: str):
        def invoke(*_args, **_kwargs):
            calls.append(label)
            raise AssertionError(f"{label} ran after failed preflight")

        return invoke

    _patch(patches, controller, "EPISODE_ROOT", root)
    _patch(patches, controller, "_archive_identity", lambda **_kwargs: archive)
    _patch(patches, controller, "_pre_cancel_preflight", reject_preflight)
    _patch(
        patches, controller, "launch_replacement_attestation", forbidden("replacement")
    )
    _patch(
        patches, controller, "cancel_obsolete_attestation", forbidden("cancellation")
    )

    with case.assertRaisesRegex(
        controller.Post36UnattendedControllerError,
        "synthetic archive collision",
    ):
        controller.orchestrate_post36_unattended(
            episode_root=root,
            archive_root=archive_root,
            archive_receipt_file_sha256="1" * 64,
            archive_receipt_sha256="2" * 64,
            archive_manifest_file_sha256="3" * 64,
            archive_manifest_sha256="4" * 64,
            archive_tree_sha256="5" * 64,
            submitter=forbidden("child submission"),
            job_resolver=forbidden("job recovery"),
            job_state_resolver=forbidden("job state"),
            releaser=forbidden("release"),
        )
    assert calls == ["preflight"]


def _case_controller_source_uses_boolean_archive_match_marker(
    case: unittest.TestCase, tmp_path: Path, patches: ExitStack
) -> None:
    archive_root = tmp_path / "archive"
    archived = (
        archive_root
        / "outputs/executor/ep12_executor/post36_unattended_controller.py"
    )
    archived.parent.mkdir(parents=True)
    archived.write_bytes(Path(controller.__file__).read_bytes())
    evidence = controller._controller_source(
        tmp_path / "unused-episode", {"archived_root": str(archive_root)}
    )
    assert evidence["active_bytes_match_archive"] is True
    assert type(evidence["active_bytes_match_archive"]) is bool


def _case_legacy_evidence_binds_exact_durable_slurm_exit_code(
    case: unittest.TestCase, tmp_path: Path, patches: ExitStack
) -> None:
    root = tmp_path / "episode"
    legacy_root = root / controller.LEGACY_ROOT_RELATIVE
    records: dict[str, tuple[str, dict[str, object], str]] = {
        "launch_receipt": (
            "launch.json",
            {"receipt_sha256": "1" * 64},
            "receipt_sha256",
        ),
        "completion": (
            "completion_45363147.json",
            {
                "completion_sha256": "2" * 64,
                "slurm_job_id": controller.LEGACY_JOB_ID,
                "state": "COMPLETED",
                "exit_code": "0:0",
                "artifact_verifier_passed": True,
                "annotation_or_outcome_files_accessed": False,
                "real_connectivity_accessed": False,
            },
            "completion_sha256",
        ),
        "probe": (
            "probe.json",
            {
                "artifact_sha256": "3" * 64,
                "schema_version": "ep12.contract_transition_invariance_probe.v1",
                "phase": "legacy",
                "scientific_payload_sha256": (
                    controller.LEGACY_SCIENTIFIC_PAYLOAD_SHA256
                ),
            },
            "artifact_sha256",
        ),
    }
    for filename, value, _semantic_field in records.values():
        _canonical(legacy_root / filename, value)

    fixed: dict[str, tuple[str, str, str, object]] = {}

    def refresh_fixed() -> None:
        fixed.clear()
        for label, (filename, value, semantic_field) in records.items():
            path = legacy_root / filename
            fixed[label] = (
                filename,
                primitives._sha256_file(path, root=root, label=label),
                semantic_field,
                value[semantic_field],
            )

    refresh_fixed()
    _patch(patches, controller, "LEGACY_FIXED", fixed)

    evidence = controller._legacy_evidence(root)
    assert evidence["slurm_job_id"] == controller.LEGACY_JOB_ID

    # The durable completion serializes Slurm's status and signal as "0:0".
    # Numerically equivalent-looking alternatives are not authoritative.
    completion = records["completion"][1]
    completion["exit_code"] = 0
    _canonical(legacy_root / records["completion"][0], completion)
    refresh_fixed()
    with case.assertRaisesRegex(
        controller.Post36UnattendedControllerError,
        "legacy probe authority facts changed",
    ):
        controller._legacy_evidence(root)


class Post36UnattendedControllerTests(unittest.TestCase):
    def _run_with_temporary_path(self, function) -> None:
        with (
            tempfile.TemporaryDirectory(prefix="ep12-unattended-") as directory,
            ExitStack() as patches,
        ):
            function(self, Path(directory), patches)

    def test_dynamic_r_dag_is_fully_receipted_before_r_release_and_replay_is_inert(
        self,
    ) -> None:
        self._run_with_temporary_path(
            _case_dynamic_r_dag_is_fully_receipted_before_r_release_and_replay_is_inert
        )

    def test_release_uncertain_held_refuses_duplicate_scontrol(self) -> None:
        self._run_with_temporary_path(
            _case_release_uncertain_held_refuses_duplicate_scontrol
        )

    def test_pre_cancel_preflight_failure_has_no_scheduler_or_cancellation_side_effects(
        self,
    ) -> None:
        self._run_with_temporary_path(
            _case_pre_cancel_preflight_failure_has_no_scheduler_or_cancellation_side_effects
        )

    def test_controller_source_uses_boolean_archive_match_marker(self) -> None:
        self._run_with_temporary_path(
            _case_controller_source_uses_boolean_archive_match_marker
        )

    def test_legacy_evidence_binds_exact_durable_slurm_exit_code(self) -> None:
        self._run_with_temporary_path(
            _case_legacy_evidence_binds_exact_durable_slurm_exit_code
        )


if __name__ == "__main__":
    unittest.main()
