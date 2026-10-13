from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from ep12_executor.policy import canonical_json, digest_object
import ep12_executor.resource_release_launch as release
from ep12_executor.resource_release_launch import ResourceReleaseLaunchError
from ep12_executor.secure_publish_lock import (
    SecurePublishLockError,
    acquire_publish_lock,
)
from ep12_executor.full_null_launch_permit import (
    FullNullLaunchPermitError,
    create_full_null_launch_permit,
    verify_full_null_launch_permit,
)
from ep12_executor.secure_transaction import (
    SecureTransactionError,
    open_regular_lock_at,
    open_secure_directory,
)


def _write_canonical(path: Path, value: object) -> bytes:
    encoded = (canonical_json(value) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return encoded


def _write_production_atomic_json(path: Path, value: object) -> bytes:
    """Match the indented bytes emitted by model_artifacts.atomic_json."""

    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return encoded


class ResourceReleaseLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "episode"
        self.run_root = self.root / "outputs" / "development_run002"
        self.transaction = self.run_root / "resource_release_launch"
        self.pending = (
            self.run_root / "pre_null_dispatch_pending"
            / "resource_authorization_pending.json"
        )
        self.authorization = (
            self.run_root / "resource_authorization"
            / "full_search_null_authorization.json"
        )
        self.wrapper = self.root / "outputs" / "executor" / "slurm" / "run.sbatch"
        self.wrapper.parent.mkdir(parents=True)
        self.wrapper.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        self.source_stage_parent = (
            Path(self.temporary.name) / "scratch" / "source_stages"
        )
        self.source_stage_parent.mkdir(parents=True)
        self.source_stage_root = self.source_stage_parent / ("source-" + "4" * 64)
        (self.source_stage_root / "outputs" / "executor").mkdir(parents=True)
        pending_core = {
            "schema_version": "ep12.resource_authorization_pending.v1",
            "action": "resource_authorization_pending",
            "full_search_null_submitted": False,
            "procedure_lock_sha256": "1" * 64,
            "procedure_lock_file_sha256": "2" * 64,
        }
        self.pending_value = {
            **pending_core, "receipt_sha256": digest_object(pending_core)
        }
        pending_bytes = _write_production_atomic_json(
            self.pending, self.pending_value
        )
        authorization_core = {
            "schema_version": "ep12.full_search_null_resource_authorization.v1",
            "allow_full_search_null_dispatch": True,
            "pending_receipt_sha256": self.pending_value["receipt_sha256"],
            "pending_receipt_file_sha256": hashlib.sha256(pending_bytes).hexdigest(),
            "procedure_lock_sha256": "1" * 64,
            "procedure_lock_file_sha256": "2" * 64,
        }
        self.authorization_value = {
            **authorization_core,
            "authorization_sha256": digest_object(authorization_core),
        }
        authorization_bytes = _write_canonical(
            self.authorization, self.authorization_value
        )
        self.authorization_file_sha256 = hashlib.sha256(authorization_bytes).hexdigest()
        self.source = {
            "manifest_path": str(self.root / "freeze.json"),
            "manifest_sha256": "3" * 64,
            "manifest_file_sha256": "4" * 64,
            "stage_root": str(self.source_stage_root),
            "file_count": 50,
        }
        self.constants = mock.patch.multiple(
            release,
            EPISODE_ROOT=self.root,
            RUN_ROOT=self.run_root,
            TRANSACTION_ROOT=self.transaction,
            INTENT_PATH=self.transaction / "intent.json",
            PROGRESS_PATH=self.transaction / "progress.json",
            RECEIPT_PATH=self.transaction / "receipt.json",
            LOCK_PATH=self.transaction / ".lock",
            PENDING_PATH=self.pending,
            PROCEDURE_WRAPPER=self.wrapper,
            SOURCE_STAGE_PARENT=self.source_stage_parent,
        )
        self.constants.start()
        self.source_patch = mock.patch.object(
            release, "_source_identity", return_value=dict(self.source)
        )
        self.source_patch.start()
        self.authorization_patch = mock.patch.object(
            release,
            "_verify_full_resource_authorization",
            side_effect=self.full_authorization_evidence,
        )
        self.authorization_patch.start()

    def tearDown(self) -> None:
        self.authorization_patch.stop()
        self.source_patch.stop()
        self.constants.stop()
        self.temporary.cleanup()

    def arguments(self) -> dict[str, object]:
        return {
            "authorization_path": self.authorization,
            "authorization_file_sha256": self.authorization_file_sha256,
            "source_freeze_manifest": self.root / "freeze.json",
            "source_freeze_manifest_sha256": "4" * 64,
            "source_stage_root": self.source_stage_root,
        }

    def full_authorization_evidence(self, **_: object) -> dict[str, object]:
        return {
            "authorization_path": str(self.authorization.resolve()),
            "authorization_sha256": self.authorization_value["authorization_sha256"],
            "authorization_file_sha256": self.authorization_file_sha256,
            "pending_path": str(self.pending.resolve()),
            "pending_receipt_sha256": self.pending_value["receipt_sha256"],
            "pending_receipt_file_sha256": hashlib.sha256(
                self.pending.read_bytes()
            ).hexdigest(),
            "resource_plan_sha256": "5" * 64,
            "runtime_bundle_sha256": "6" * 64,
        }

    def test_exact_launch_is_idempotent_and_never_exports_all(self) -> None:
        commands: list[list[str]] = []

        def submit(command: object) -> str:
            commands.append(list(command))
            return "88001;sherlock"

        receipt = release.launch_resource_release(
            **self.arguments(), submitter=submit, job_resolver=lambda _: None
        )
        replay = release.launch_resource_release(
            **self.arguments(),
            submitter=lambda _: self.fail("receipt replay resubmitted"),
            job_resolver=lambda _: self.fail("receipt replay queried Slurm"),
        )
        self.assertEqual(receipt, replay)
        self.assertEqual(len(commands), 1)
        export = next(value for value in commands[0] if value.startswith("--export="))
        self.assertNotIn("ALL", export.split(","))
        self.assertIn("EP12_PROCEDURE_LOCK_LAUNCH_MODE=release", export)
        self.assertIn("EP12_SOURCE_FREEZE_MANIFEST=", export)
        self.assertIn(
            f"EP12_SOURCE_STAGE_ROOT={self.source_stage_root}", export
        )
        self.assertFalse(receipt["ambient_environment_inherited"])
        self.assertEqual(receipt["resource"]["authorization_sha256"],
                         self.authorization_value["authorization_sha256"])

    def test_release_stage_root_reaches_and_is_verified_by_runtime_consumer(self) -> None:
        receipt = release.launch_resource_release(
            **self.arguments(), submitter=lambda _: "88003", job_resolver=lambda _: None
        )
        exports = dict(receipt["automatic_exports"])
        self.assertEqual(
            exports["EP12_SOURCE_STAGE_ROOT"], str(self.source_stage_root)
        )
        runtime_environment = {
            **exports,
            "SLURM_JOB_NAME": str(receipt["job_name"]),
        }
        with mock.patch.dict(os.environ, runtime_environment, clear=True):
            intent = release.verify_runtime_intent(
                intent_path=release.INTENT_PATH,
                authorization_path=self.authorization,
                authorization_file_sha256=self.authorization_file_sha256,
                pending_path=self.pending,
                pending_file_sha256=hashlib.sha256(
                    self.pending.read_bytes()
                ).hexdigest(),
                source_freeze_manifest=self.root / "freeze.json",
                source_freeze_manifest_sha256="4" * 64,
                source_stage_root=self.source_stage_root,
            )
        self.assertEqual(
            intent["source_freeze"]["stage_root"], str(self.source_stage_root)
        )

    def test_source_identity_requires_the_content_addressed_stage_root(self) -> None:
        self.source_patch.stop()
        try:
            manifest = {"manifest_sha256": "3" * 64, "file_count": 50}
            with mock.patch.object(
                release, "verify_source_freeze_manifest", return_value=manifest
            ), mock.patch.object(
                release, "source_freeze_file_sha256", return_value="4" * 64
            ):
                identity = release._source_identity(
                    self.root / "freeze.json",
                    "4" * 64,
                    self.source_stage_root,
                )
                self.assertEqual(
                    identity["stage_root"], str(self.source_stage_root)
                )
                wrong = self.source_stage_parent / ("source-" + "5" * 64)
                (wrong / "outputs" / "executor").mkdir(parents=True)
                with self.assertRaisesRegex(
                    ResourceReleaseLaunchError, "freeze content address"
                ):
                    release._source_identity(
                        self.root / "freeze.json", "4" * 64, wrong
                    )
        finally:
            self.source_patch.start()

    def test_lost_ack_is_recovered_without_duplicate(self) -> None:
        accepted: dict[str, str] = {}
        visible = False
        submissions = 0

        def resolve(name: str) -> str | None:
            return accepted.get(name) if visible else None

        def submit(command: object) -> str:
            nonlocal submissions
            submissions += 1
            name = next(
                value.split("=", 1)[1]
                for value in list(command)
                if value.startswith("--job-name=")
            )
            accepted[name] = "88002"
            raise RuntimeError("synthetic lost acknowledgement")

        with self.assertRaisesRegex(ResourceReleaseLaunchError, "acceptance is uncertain"):
            release.launch_resource_release(
                **self.arguments(), submitter=submit, job_resolver=resolve
            )
        progress = json.loads((self.transaction / "progress.json").read_bytes())
        self.assertEqual(progress["state"], "submission_uncertain")
        with self.assertRaisesRegex(ResourceReleaseLaunchError, "refusing duplicate"):
            release.launch_resource_release(
                **self.arguments(),
                submitter=lambda _: self.fail("uncertain retry resubmitted"),
                job_resolver=resolve,
            )
        visible = True
        receipt = release.launch_resource_release(
            **self.arguments(),
            submitter=lambda _: self.fail("recovery resubmitted"),
            job_resolver=resolve,
        )
        self.assertEqual(receipt["slurm_job_id"], "88002")
        self.assertEqual(submissions, 1)

    def test_default_recovery_is_scoped_to_effective_user(self) -> None:
        responses = [
            subprocess.CompletedProcess([], 0, stdout="", stderr=""),
            subprocess.CompletedProcess([], 0, stdout="991|release-name|zijiao\n", stderr=""),
        ]
        with mock.patch.object(release, "_effective_user", return_value="zijiao"), \
             mock.patch.object(release.subprocess, "run", side_effect=responses) as run:
            self.assertEqual(release._default_job_resolver("release-name"), "991")
        self.assertIn("--user", run.call_args_list[0].args[0])
        self.assertIn("--user", run.call_args_list[1].args[0])

    def test_release_submitter_ignores_adversarial_sbatch_environment(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="992\n", stderr="")
        poison = {
            "SBATCH_EXPORT": "ALL",
            "SBATCH_JOB_NAME": "wrong-name",
            "SBATCH_PARTITION": "wrong-partition",
            "SLURM_EXPORT_ENV": "ALL",
        }
        with mock.patch.dict(os.environ, poison, clear=False), mock.patch.object(
            release.subprocess, "run", return_value=completed
        ) as run:
            self.assertEqual(release._default_submitter(["sbatch", "--parsable"]), "992")
        environment = run.call_args.kwargs["env"]
        self.assertIn("PATH", environment)
        self.assertTrue(set(environment) <= {"PATH", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE"})
        self.assertFalse(any(name.startswith(("SBATCH_", "SLURM_")) for name in environment))

    def test_release_requires_full_resource_authorization_before_submission(self) -> None:
        self.authorization_patch.stop()
        try:
            with mock.patch.object(
                release,
                "_verify_full_resource_authorization",
                side_effect=ResourceReleaseLaunchError("synthetic full-verifier rejection"),
            ):
                with self.assertRaisesRegex(
                    ResourceReleaseLaunchError, "full-verifier rejection"
                ):
                    release.launch_resource_release(
                        **self.arguments(),
                        submitter=lambda _: self.fail("rejected authorization reached sbatch"),
                        job_resolver=lambda _: None,
                    )
        finally:
            self.authorization_patch.start()


class SecurePublishLockTests(unittest.TestCase):
    def test_symlink_lock_is_rejected_without_truncating_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.txt"
            target.write_text("do-not-truncate", encoding="utf-8")
            lock = root / "publish.lock"
            lock.symlink_to(target)
            with self.assertRaises((SecurePublishLockError, OSError)):
                acquire_publish_lock(lock)
            self.assertEqual(target.read_text(encoding="utf-8"), "do-not-truncate")


class SecureTransactionTests(unittest.TestCase):
    def test_symlinked_directory_and_lock_are_rejected_without_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            outside = root / "outside"
            outside.mkdir()
            (root / "linked").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(SecureTransactionError):
                open_secure_directory(root / "linked" / "child", root=root)

            safe, directory_fd = open_secure_directory(root / "safe", root=root)
            self.assertEqual(safe, root / "safe")
            target = root / "target"
            target.write_text("do-not-truncate", encoding="utf-8")
            (safe / ".lock").symlink_to(target)
            try:
                with self.assertRaises(SecureTransactionError):
                    open_regular_lock_at(directory_fd, ".lock")
            finally:
                os.close(directory_fd)
            self.assertEqual(target.read_text(encoding="utf-8"), "do-not-truncate")


class FullNullLaunchPermitTests(unittest.TestCase):
    def test_permit_binds_authorization_wrappers_job_and_array_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authorization = root / "authorization.json"
            pending = root / "pending.json"
            procedure = root / "procedure.json"
            wrapper = root / "run.sbatch"
            aggregate = root / "aggregate.sbatch"
            for path, value in (
                (authorization, {"authorization": True}),
                (pending, {"pending": True}),
                (procedure, {"procedure": True}),
            ):
                _write_canonical(path, value)
            wrapper.write_text("#!/bin/bash\n", encoding="utf-8")
            aggregate.write_text("#!/bin/bash\n", encoding="utf-8")
            file_sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            source_path = root / "freeze.json"
            source_path.write_text("{}\n", encoding="utf-8")
            authorization_evidence = {
                "authorization_path": str(authorization),
                "authorization_sha256": "1" * 64,
                "authorization_file_sha256": file_sha(authorization),
                "pending_path": str(pending),
                "pending_receipt_sha256": "2" * 64,
                "pending_receipt_file_sha256": file_sha(pending),
                "resource_plan_sha256": "3" * 64,
                "runtime_bundle_sha256": "4" * 64,
            }
            launch = {
                "runtime_provider": "provider:load",
                "runtime_bundle": str(root / "bundle"),
                "runtime_bundle_sha256": "4" * 64,
                "full_null_work_root": str(root / "work"),
                "full_null_durable_root": str(root / "durable"),
                "procedure_lock_path": str(procedure),
                "procedure_lock_sha256": "5" * 64,
                "procedure_lock_file_sha256": file_sha(procedure),
                "full_search_null_script": str(wrapper),
                "full_search_null_script_sha256": file_sha(wrapper),
                "full_search_null_aggregate_script": str(aggregate),
                "full_search_null_aggregate_script_sha256": file_sha(aggregate),
                "source_freeze": {
                    "manifest_path": str(source_path),
                    "manifest_file_sha256": file_sha(source_path),
                },
            }
            identity = create_full_null_launch_permit(
                permit_path=root / "permit.json",
                authorization_evidence=authorization_evidence,
                launch=launch,
                array_job_name="array-name",
                aggregate_job_name="aggregate-name",
                array_indices=[0, 1],
            )
            environment = {
                "EP12_FULL_NULL_RUNTIME_PROVIDER": "provider:load",
                "EP12_FULL_NULL_BUNDLE": str(root / "bundle"),
                "EP12_FULL_NULL_BUNDLE_SHA256": "4" * 64,
                "EP12_FULL_NULL_WORK_ROOT": str(root / "work"),
                "EP12_FULL_NULL_DURABLE_ROOT": str(root / "durable"),
                "EP12_PRE_NULL_PROCEDURE_LOCK_PATH": str(procedure),
                "EP12_PRE_NULL_PROCEDURE_LOCK_SHA256": "5" * 64,
                "EP12_PRE_NULL_PROCEDURE_LOCK_FILE_SHA256": file_sha(procedure),
                "EP12_SOURCE_FREEZE_MANIFEST": str(source_path),
                "EP12_SOURCE_FREEZE_MANIFEST_SHA256": file_sha(source_path),
                "SLURM_JOB_NAME": "array-name",
                "SLURM_ARRAY_JOB_ID": "100",
                "SLURM_ARRAY_TASK_ID": "1",
            }
            with mock.patch(
                "ep12_executor.full_null_launch_permit.verify_source_freeze_manifest"
            ):
                verify_full_null_launch_permit(
                    permit_path=Path(identity["permit_path"]),
                    expected_file_sha256=str(identity["permit_file_sha256"]),
                    mode="replicate",
                    environment=environment,
                )
                environment["SLURM_ARRAY_TASK_ID"] = "2"
                with self.assertRaisesRegex(FullNullLaunchPermitError, "outside permit"):
                    verify_full_null_launch_permit(
                        permit_path=Path(identity["permit_path"]),
                        expected_file_sha256=str(identity["permit_file_sha256"]),
                        mode="replicate",
                        environment=environment,
                    )
            authorization.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(FullNullLaunchPermitError, "authorization bytes changed"):
                verify_full_null_launch_permit(
                    permit_path=Path(identity["permit_path"]),
                    expected_file_sha256=str(identity["permit_file_sha256"]),
                    mode="replicate",
                    environment=environment,
                )


if __name__ == "__main__":
    unittest.main()
