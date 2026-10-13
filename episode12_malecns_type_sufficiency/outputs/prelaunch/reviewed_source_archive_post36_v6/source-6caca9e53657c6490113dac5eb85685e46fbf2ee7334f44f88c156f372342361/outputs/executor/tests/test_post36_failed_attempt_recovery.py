from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ep12_executor.policy import digest_object
from ep12_executor import obsolete_attestation_cancel as cancellation
from ep12_executor import post36_failed_attempt_recovery as recovery
from ep12_executor import replacement_attestation_launch as replacement


def _canonical(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )


def _self_hashed(path: Path, core: dict[str, object], field: str) -> dict[str, object]:
    value = {**core, field: digest_object(core)}
    _canonical(path, value)
    return value


def _failed_tree(root: Path, log_path: Path) -> list[dict[str, str]]:
    prelaunch = root / "outputs/prelaunch"
    replacement_root = prelaunch / "replacement_attestation_launch"
    controller_root = prelaunch / "post36_orchestration"

    archive_receipt = prelaunch / "reviewed_source_archive_post36_v5/source_archive_receipt.json"
    _canonical(archive_receipt, {"synthetic": "v5"})
    archive_receipt_sha = hashlib.sha256(archive_receipt.read_bytes()).hexdigest()
    finalizer_core = {
        "schema_version": replacement.FINALIZER_COMPLETION_SCHEMA,
        "job_id": replacement.OBSERVED_FINALIZER_JOB_ID,
        "job_name": replacement.OBSERVED_FINALIZER_JOB_NAME,
        "user": "zijiao",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "logical_dependency": f"afterok:{replacement.OBSERVED_FINALIZER_JOB_ID}",
        "logical_dependency_satisfied": True,
        "scheduler_dependency": None,
        "scheduler_dependency_omitted_because_terminal_success": True,
    }
    finalizer = {
        **finalizer_core,
        "evidence_sha256": digest_object(finalizer_core),
    }
    replacement_receipt = _self_hashed(
        replacement_root / "receipt.json",
        {
            "schema_version": "ep12.replacement_attestation_launch_receipt.v2",
            "slurm_job_id": recovery.FAILED_JOBS[0]["job_id"],
            "slurm_job_name": recovery.FAILED_JOBS[0]["job_name"],
            "attestation_result_authoritative": False,
            "final_connectivity_accessed": False,
            "finalizer_terminal_completion": finalizer,
            "archive": {
                "tree_sha256": recovery.V5_TREE_SHA256,
                "complete": True,
                "inputs_included": False,
                "development_outputs_included": False,
                "prelaunch_outputs_included": False,
                "archive_receipt_path": str(archive_receipt),
                "archive_receipt_file_sha256": archive_receipt_sha,
            },
        },
        "receipt_sha256",
    )
    obsolete = _self_hashed(
        replacement_root / "obsolete_cancellation/receipt.json",
        {
            "schema_version": "ep12.obsolete_attestation_cancellation_receipt.v2",
            "obsolete_job_id": "45332084",
            "obsolete_job_name": "ep12-attest36",
            "submission_user": "zijiao",
            "replacement_job_id": recovery.FAILED_JOBS[0]["job_id"],
            "terminal_scheduler_state": "CANCELLED",
            "terminal_scheduler_reason": "Dependency",
            "terminal_scheduler_priority": "1",
            "replacement_receipt_sha256": replacement_receipt["receipt_sha256"],
            "descendants_immediately_before_cancel": [],
            "scancel_was_exactly_scoped": True,
            "obsolete_result_authoritative": False,
            "replacement_attestation_result_authoritative": False,
            "final_connectivity_accessed": False,
        },
        "receipt_sha256",
    )
    stages: dict[str, object] = {}
    for stage, record in zip(
        ("post36_probe", "comparison", "publisher", "downstream"),
        recovery.FAILED_JOBS[1:],
        strict=True,
    ):
        stages[stage] = {
            "slurm_job_id": record["job_id"],
            "slurm_job_name": record["job_name"],
            "final_connectivity_accessed": False,
        }
    _self_hashed(
        controller_root / "receipt.json",
        {
            "schema_version": "ep12.post36_unattended_orchestration_receipt.v1",
            "replacement_job_id": recovery.FAILED_JOBS[0]["job_id"],
            "replacement_launch_receipt_sha256": replacement_receipt["receipt_sha256"],
            "obsolete_cancellation_receipt_sha256": obsolete["receipt_sha256"],
            "stages": stages,
            "final_connectivity_accessed": False,
            "final_connectivity_access_authorized": False,
        },
        "receipt_sha256",
    )
    (replacement_root / ".lock").write_bytes(b"")
    (controller_root / ".lock").write_bytes(b"")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "ObservedAttestationError: " + recovery.EXPECTED_FAILURE_TEXT + "\n",
        encoding="utf-8",
    )
    return [
        {**record, "user": "zijiao"}
        for record in recovery.FAILED_JOBS
    ]


class FailedAttemptRecoveryTests(unittest.TestCase):
    def _fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix="ep12-v5-recovery-")
        base = Path(temporary.name)
        root = base / "episode"
        log = base / "failed-job.out"
        rows = _failed_tree(root, log)
        patches = ExitStack()
        patches.enter_context(mock.patch.object(recovery, "EPISODE_ROOT", root))
        patches.enter_context(mock.patch.object(recovery, "FAILED_LOG_PATH", log))
        patches.enter_context(mock.patch.object(recovery, "_effective_user", lambda: "zijiao"))
        return temporary, root, log, rows, patches

    def test_rotation_preserves_both_trees_and_replay_is_scheduler_inert(self) -> None:
        temporary, root, log, rows, patches = self._fixture()
        with temporary, patches:
            evidence = recovery.rotate_failed_post36_attempt(
                episode_root=root,
                terminal_resolver=lambda _user: rows,
            )
            self.assertTrue(evidence["old_transactions_preserved_byte_for_byte"])
            self.assertFalse(evidence["scheduler_mutations_performed"])
            self.assertFalse((root / recovery.REPLACEMENT_SOURCE_RELATIVE).exists())
            self.assertFalse((root / recovery.CONTROLLER_SOURCE_RELATIVE).exists())
            self.assertTrue((root / recovery.REPLACEMENT_ARCHIVE_RELATIVE).is_dir())
            self.assertTrue((root / recovery.CONTROLLER_ARCHIVE_RELATIVE).is_dir())
            log.unlink()
            replay = recovery.rotate_failed_post36_attempt(
                episode_root=root,
                terminal_resolver=mock.Mock(
                    side_effect=AssertionError("receipt replay queried Slurm")
                ),
            )
            self.assertEqual(replay, evidence)

    def test_default_terminal_resolver_uses_one_scoped_sacct_query(self) -> None:
        output = "\n".join(
            "|".join(
                (
                    record["job_id"],
                    record["job_name"],
                    "zijiao",
                    record["state"],
                    record["exit_code"],
                )
            )
            for record in recovery.FAILED_JOBS
        )
        scheduler = mock.Mock(return_value=output)
        with (
            mock.patch.object(recovery, "_effective_user", lambda: "zijiao"),
            mock.patch.object(recovery, "_run_scheduler", scheduler),
        ):
            rows = recovery.default_terminal_resolver("zijiao")

        self.assertEqual(
            rows,
            [{**record, "user": "zijiao"} for record in recovery.FAILED_JOBS],
        )
        scheduler.assert_called_once()
        command = scheduler.call_args.args[0]
        self.assertEqual(command[0], "/usr/bin/sacct")
        self.assertNotIn("/usr/bin/squeue", command)
        self.assertEqual(
            command[command.index("--jobs") + 1],
            ",".join(record["job_id"] for record in recovery.FAILED_JOBS),
        )

    def test_recovered_launch_reuses_terminal_prerequisites_without_queries(self) -> None:
        temporary, root, _log, rows, patches = self._fixture()
        with temporary, patches:
            evidence = recovery.rotate_failed_post36_attempt(
                episode_root=root,
                terminal_resolver=lambda _user: rows,
            )
            forbidden = mock.Mock(
                side_effect=AssertionError("recovered launch queried Slurm")
            )
            obsolete, finalizer = replacement._replacement_prerequisite_evidence(
                submission_user="zijiao",
                failed_attempt_recovery=evidence,
                obsolete_state_resolver=forbidden,
                finalizer_state_resolver=forbidden,
            )

        self.assertTrue(obsolete["state"].startswith("CANCELLED"))
        self.assertEqual(finalizer, evidence["finalizer_terminal_completion"])
        forbidden.assert_not_called()

    def test_interrupted_after_first_rename_resumes_without_overwrite(self) -> None:
        temporary, root, _log, rows, patches = self._fixture()
        with temporary, patches:
            original = recovery._rename_transaction
            calls = 0

            def interrupt_after_first(**kwargs):
                nonlocal calls
                calls += 1
                original(**kwargs)
                if calls == 1:
                    raise RuntimeError("synthetic interruption")

            with mock.patch.object(recovery, "_rename_transaction", interrupt_after_first):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    recovery.rotate_failed_post36_attempt(
                        episode_root=root,
                        terminal_resolver=lambda _user: rows,
                    )
            self.assertTrue((root / recovery.REPLACEMENT_ARCHIVE_RELATIVE).is_dir())
            self.assertTrue((root / recovery.CONTROLLER_SOURCE_RELATIVE).is_dir())
            evidence = recovery.rotate_failed_post36_attempt(
                episode_root=root,
                terminal_resolver=mock.Mock(
                    side_effect=AssertionError("resume requeried terminal jobs")
                ),
            )
            self.assertTrue(evidence["old_transactions_preserved_byte_for_byte"])

    def test_terminal_mismatch_aborts_before_any_move(self) -> None:
        temporary, root, _log, rows, patches = self._fixture()
        with temporary, patches:
            unsafe = [dict(value) for value in rows]
            unsafe[0]["state"] = "COMPLETED"
            with self.assertRaisesRegex(
                recovery.FailedAttemptRecoveryError,
                "terminal Slurm identity changed",
            ):
                recovery.rotate_failed_post36_attempt(
                    episode_root=root,
                    terminal_resolver=lambda _user: unsafe,
                )
            self.assertTrue((root / recovery.REPLACEMENT_SOURCE_RELATIVE).is_dir())
            self.assertTrue((root / recovery.CONTROLLER_SOURCE_RELATIVE).is_dir())

    def test_output_collision_aborts_before_terminal_query_or_move(self) -> None:
        temporary, root, _log, _rows, patches = self._fixture()
        with temporary, patches:
            collision = root / recovery.REQUIRED_ABSENT_OUTPUTS[0]
            collision.mkdir(parents=True)
            with self.assertRaisesRegex(
                recovery.FailedAttemptRecoveryError,
                "published unexpected output",
            ):
                recovery.rotate_failed_post36_attempt(
                    episode_root=root,
                    terminal_resolver=mock.Mock(
                        side_effect=AssertionError("collision queried Slurm")
                    ),
                )
            self.assertTrue((root / recovery.REPLACEMENT_SOURCE_RELATIVE).is_dir())
            self.assertTrue((root / recovery.CONTROLLER_SOURCE_RELATIVE).is_dir())

    def test_carry_forward_binds_new_replacement_without_scancel(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-cancel-carry-") as directory:
            root = Path(directory) / "episode"
            root.mkdir()
            prior = {
                "prior_obsolete_cancellation_receipt_path": str(
                    root / "archived/obsolete_cancellation/receipt.json"
                ),
                "prior_obsolete_cancellation_receipt_file_sha256": "a" * 64,
                "prior_obsolete_cancellation_receipt_sha256": "b" * 64,
                "prior_terminal_scheduler_state": "CANCELLED",
                "prior_obsolete_terminal_scheduler": {
                    "job_id": cancellation.OBSERVED_ATTESTATION_JOB_ID,
                    "job_name": cancellation.OBSERVED_ATTESTATION_JOB_NAME,
                    "user": "zijiao",
                    "state": "CANCELLED",
                    "reason": "Dependency",
                    "priority": "1",
                },
                "prior_scancel_was_exactly_scoped": True,
                "scheduler_mutations_performed": False,
                "final_connectivity_accessed": False,
            }
            replacement = {
                "slurm_job_id": "900",
                "launch_receipt_path": str(root / "replacement/receipt.json"),
                "launch_receipt_file_sha256": "c" * 64,
                "launch_receipt_sha256": "d" * 64,
                "failed_attempt_recovery": prior,
                "obsolete_job_already_terminally_cancelled": True,
                "obsolete_scheduler_observation_immediately_before_submit": {
                    "job_id": cancellation.OBSERVED_ATTESTATION_JOB_ID,
                    "job_name": cancellation.OBSERVED_ATTESTATION_JOB_NAME,
                    "user": "zijiao",
                    "state": "CANCELLED",
                    "reason": "Dependency",
                    "priority": "1",
                },
            }
            cancelled = {
                "job_id": cancellation.OBSERVED_ATTESTATION_JOB_ID,
                "job_name": cancellation.OBSERVED_ATTESTATION_JOB_NAME,
                "user": "zijiao",
                "state": "CANCELLED",
                "reason": "Dependency",
                "priority": "1",
            }
            forbidden = mock.Mock(
                side_effect=AssertionError("carry-forward issued scheduler mutation")
            )
            with (
                mock.patch.object(
                    cancellation,
                    "verify_replacement_attestation_receipt",
                    lambda **_kwargs: replacement,
                ),
                mock.patch.object(
                    cancellation,
                    "verify_failed_post36_attempt_recovery",
                    lambda **_kwargs: prior,
                ),
                mock.patch.object(cancellation, "_effective_user", lambda: "zijiao"),
            ):
                receipt = cancellation.cancel_obsolete_attestation(
                    episode_root=root,
                    state_resolver=forbidden,
                    descendant_resolver=forbidden,
                    canceller=forbidden,
                )
                replay = cancellation.cancel_obsolete_attestation(
                    episode_root=root,
                    state_resolver=forbidden,
                    descendant_resolver=forbidden,
                    canceller=forbidden,
                )
            self.assertEqual(replay, receipt)
            self.assertFalse(receipt["scancel_issued_for_this_replacement"])
            self.assertTrue(receipt["scancel_was_exactly_scoped"])
            self.assertTrue(receipt["prior_scancel_was_exactly_scoped"])
            self.assertEqual(receipt["replacement_job_id"], "900")
            self.assertEqual(
                receipt["prior_obsolete_cancellation_receipt_sha256"], "b" * 64
            )
            forbidden.assert_not_called()

    def test_fresh_cancellation_accepts_parenthesized_user_hold_once(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-cancel-held-") as directory:
            root = Path(directory) / "episode"
            root.mkdir()
            replacement_evidence = {
                "slurm_job_id": "900",
                "launch_receipt_path": str(root / "replacement/receipt.json"),
                "launch_receipt_file_sha256": "c" * 64,
                "launch_receipt_sha256": "d" * 64,
                "failed_attempt_recovery": None,
                "obsolete_job_already_terminally_cancelled": False,
            }
            cancelled = False
            state_calls = 0
            cancel_commands: list[list[str]] = []

            def resolve_state(_job_id: str, _job_name: str, user: str):
                nonlocal state_calls
                state_calls += 1
                return {
                    "job_id": cancellation.OBSERVED_ATTESTATION_JOB_ID,
                    "job_name": cancellation.OBSERVED_ATTESTATION_JOB_NAME,
                    "user": user,
                    "state": "CANCELLED" if cancelled else "PENDING",
                    "reason": "Dependency" if cancelled else "(JobHeldUser)",
                    "priority": "1" if cancelled else "0",
                }

            def cancel(command: list[str]) -> str:
                nonlocal cancelled
                cancel_commands.append(command)
                cancelled = True
                return ""

            with (
                mock.patch.object(
                    cancellation,
                    "verify_replacement_attestation_receipt",
                    lambda **_kwargs: replacement_evidence,
                ),
                mock.patch.object(cancellation, "_effective_user", lambda: "zijiao"),
            ):
                receipt = cancellation.cancel_obsolete_attestation(
                    episode_root=root,
                    state_resolver=resolve_state,
                    descendant_resolver=lambda _job_id, _user: [],
                    canceller=cancel,
                )
                replay = cancellation.cancel_obsolete_attestation(
                    episode_root=root,
                    state_resolver=mock.Mock(
                        side_effect=AssertionError("receipt replay queried Slurm")
                    ),
                    descendant_resolver=mock.Mock(
                        side_effect=AssertionError("receipt replay queried descendants")
                    ),
                    canceller=mock.Mock(
                        side_effect=AssertionError("receipt replay repeated scancel")
                    ),
                )

            self.assertEqual(replay, receipt)
            self.assertEqual(state_calls, 3)
            self.assertEqual(cancel_commands, [cancellation._command("zijiao")])
            self.assertTrue(receipt["scancel_issued_for_this_replacement"])
            self.assertEqual(
                receipt["cancellation_evidence_mode"], "issue_exact_scancel"
            )

    def test_incomplete_rotation_blocks_replacement_before_canonical_recreation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ep12-incomplete-rotation-") as directory:
            root = Path(directory) / "episode"
            (root / recovery.RECOVERY_ROOT_RELATIVE).mkdir(parents=True)
            with mock.patch.object(replacement, "EPISODE_ROOT", root):
                with self.assertRaisesRegex(
                    replacement.ReplacementAttestationLaunchError,
                    "recovery is incomplete",
                ):
                    replacement.launch_replacement_attestation(
                        episode_root=root,
                        archive_root=root / "unused-archive",
                        archive_receipt_file_sha256="1" * 64,
                        archive_receipt_sha256="2" * 64,
                        archive_manifest_file_sha256="3" * 64,
                        archive_manifest_sha256="4" * 64,
                        archive_tree_sha256="5" * 64,
                    )
            self.assertFalse((root / replacement.TRANSACTION_RELATIVE).exists())


if __name__ == "__main__":
    unittest.main()
