from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess
import tempfile
import unittest
from unittest import mock

from ep12_executor.policy import canonical_json
from ep12_executor import source_freeze
from ep12_executor.source_freeze import (
    SOURCE_FREEZE_RELATIVE,
    SourceFreezeError,
    create_source_freeze_manifest,
    source_freeze_file_sha256,
    stage_source_freeze,
    verify_source_freeze_manifest,
    verify_staged_source,
)
from ep12_executor.test_launch import (
    PROGRESS_RELATIVE,
    RECEIPT_RELATIVE,
    TEST_WRAPPER_RELATIVE,
    TestLaunchError,
    _default_job_resolver,
    _unique_recovery,
    launch_executor_tests,
    read_test_launch_receipt,
)


class SourceFreezeAndTestLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        (self.root / "SEARCH_POLICY.yaml").write_text("schema: test\n", encoding="utf-8")
        (self.root / "outputs/executor/ep12_executor").mkdir(parents=True)
        (self.root / "outputs/executor/ep12_executor/example.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        wrapper = self.root / TEST_WRAPPER_RELATIVE
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        self.scratch = self.root / "scratch"
        self.stage_parent = self.scratch / "source_stages"
        self.stage_parent.mkdir(parents=True)
        self.user = pwd.getpwuid(os.getuid()).pw_name

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_manifest_is_canonical_self_hashed_and_detects_code_mutation(self) -> None:
        manifest = create_source_freeze_manifest(episode_root=self.root)
        path = self.root / SOURCE_FREEZE_RELATIVE
        self.assertEqual(
            path.read_bytes(), (canonical_json(manifest) + "\n").encode("utf-8")
        )
        exact_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(
            source_freeze_file_sha256(path, episode_root=self.root), exact_hash
        )
        self.assertEqual(
            verify_source_freeze_manifest(
                path,
                episode_root=self.root,
                expected_file_sha256=exact_hash,
            ),
            manifest,
        )
        source = self.root / "outputs/executor/ep12_executor/example.py"
        source.write_text("VALUE = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(SourceFreezeError, "surface changed"):
            verify_source_freeze_manifest(
                path,
                episode_root=self.root,
                expected_file_sha256=exact_hash,
            )

    def test_manifest_tamper_and_executor_symlink_fail_closed(self) -> None:
        create_source_freeze_manifest(episode_root=self.root)
        path = self.root / SOURCE_FREEZE_RELATIVE
        value = json.loads(path.read_text(encoding="utf-8"))
        value["file_count"] += 1
        path.write_text(canonical_json(value) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(SourceFreezeError, "self hash mismatch"):
            verify_source_freeze_manifest(path, episode_root=self.root)

        path.unlink()
        outside = self.root / "outside.py"
        outside.write_text("OUTSIDE = True\n", encoding="utf-8")
        alias = self.root / "outputs/executor/ep12_executor/alias.py"
        alias.symlink_to(outside)
        with self.assertRaisesRegex(SourceFreezeError, "contains a symlink"):
            create_source_freeze_manifest(episode_root=self.root)

    def test_stage_is_exact_private_rehashed_surface_without_cache(self) -> None:
        manifest = create_source_freeze_manifest(episode_root=self.root)
        manifest_path = self.root / SOURCE_FREEZE_RELATIVE
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        cache = self.root / "outputs/executor/ep12_executor/__pycache__"
        cache.mkdir()
        (cache / "example.cpython-312.pyc").write_bytes(b"untrusted-cache")

        stage = stage_source_freeze(
            manifest_path,
            expected_file_sha256=manifest_hash,
            stage_parent=self.stage_parent,
            episode_root=self.root,
            scratch_root=self.scratch,
        )

        self.assertEqual(stage.parent, self.stage_parent)
        self.assertEqual(stage.stat().st_mode & 0o777, 0o500)
        observed = sorted(
            path.relative_to(stage).as_posix()
            for path in stage.rglob("*")
            if path.is_file()
        )
        self.assertEqual(observed, [record["path"] for record in manifest["files"]])
        self.assertFalse(any("__pycache__" in path.parts for path in stage.rglob("*")))
        self.assertEqual(
            verify_staged_source(
                stage,
                manifest_path,
                expected_file_sha256=manifest_hash,
                episode_root=self.root,
                scratch_root=self.scratch,
            ),
            manifest,
        )

    def test_stage_rejects_live_mutation_and_cleans_partial_tree(self) -> None:
        create_source_freeze_manifest(episode_root=self.root)
        manifest_path = self.root / SOURCE_FREEZE_RELATIVE
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        source = self.root / "outputs/executor/ep12_executor/example.py"
        source.write_text("VALUE = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(SourceFreezeError, "differs from manifest"):
            stage_source_freeze(
                manifest_path,
                expected_file_sha256=manifest_hash,
                stage_parent=self.stage_parent,
                episode_root=self.root,
                scratch_root=self.scratch,
            )
        self.assertEqual(list(self.stage_parent.iterdir()), [])

    def test_stage_fails_closed_when_live_file_changes_after_read(self) -> None:
        create_source_freeze_manifest(episode_root=self.root)
        manifest_path = self.root / SOURCE_FREEZE_RELATIVE
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        live_source = self.root / "outputs/executor/ep12_executor/example.py"
        original_reader = source_freeze._read_relative_bytes_nofollow
        changed = False

        def read_then_mutate(
            root_descriptor: int, relative: object, *, label: str
        ) -> bytes:
            nonlocal changed
            encoded = original_reader(root_descriptor, relative, label=label)
            if (
                not changed
                and label == "live source"
                and str(relative) == "outputs/executor/ep12_executor/example.py"
            ):
                live_source.write_text("VALUE = 999\n", encoding="utf-8")
                changed = True
            return encoded

        with mock.patch.object(
            source_freeze,
            "_read_relative_bytes_nofollow",
            side_effect=read_then_mutate,
        ):
            with self.assertRaisesRegex(SourceFreezeError, "surface changed"):
                stage_source_freeze(
                    manifest_path,
                    expected_file_sha256=manifest_hash,
                    stage_parent=self.stage_parent,
                    episode_root=self.root,
                    scratch_root=self.scratch,
                )
        self.assertTrue(changed)
        self.assertEqual(list(self.stage_parent.iterdir()), [])

    def test_content_addressed_stage_is_reused_without_live_recopy(self) -> None:
        create_source_freeze_manifest(episode_root=self.root)
        manifest_path = self.root / SOURCE_FREEZE_RELATIVE
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        first = stage_source_freeze(
            manifest_path,
            expected_file_sha256=manifest_hash,
            stage_parent=self.stage_parent,
            episode_root=self.root,
            scratch_root=self.scratch,
        )
        live_source = self.root / "outputs/executor/ep12_executor/example.py"
        os.chmod(live_source, 0o600)
        live_source.write_text("VALUE = 999\n", encoding="utf-8")
        with mock.patch.object(
            source_freeze,
            "_surface_paths",
            side_effect=AssertionError("existing stage must not read live source"),
        ):
            second = stage_source_freeze(
                manifest_path,
                expected_file_sha256=manifest_hash,
                stage_parent=self.stage_parent,
                episode_root=self.root,
                scratch_root=self.scratch,
            )
        self.assertEqual(first, second)
        self.assertEqual(
            (second / "outputs/executor/ep12_executor/example.py").read_text(),
            "VALUE = 1\n",
        )

    def test_stage_verifier_rejects_extra_and_symlink_entries(self) -> None:
        create_source_freeze_manifest(episode_root=self.root)
        manifest_path = self.root / SOURCE_FREEZE_RELATIVE
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        stage = stage_source_freeze(
            manifest_path,
            expected_file_sha256=manifest_hash,
            stage_parent=self.stage_parent,
            episode_root=self.root,
            scratch_root=self.scratch,
        )
        os.chmod(stage, 0o700)
        extra = stage / "extra.py"
        extra.write_text("EXTRA = True\n", encoding="utf-8")
        with self.assertRaisesRegex(SourceFreezeError, "file set differs"):
            verify_staged_source(
                stage,
                manifest_path,
                expected_file_sha256=manifest_hash,
                episode_root=self.root,
                scratch_root=self.scratch,
            )
        extra.unlink()
        (stage / "alias.py").symlink_to(stage / "SEARCH_POLICY.yaml")
        with self.assertRaisesRegex(SourceFreezeError, "contains a symlink"):
            verify_staged_source(
                stage,
                manifest_path,
                expected_file_sha256=manifest_hash,
                episode_root=self.root,
                scratch_root=self.scratch,
            )

    def test_stage_parent_must_be_real_directory_inside_scratch(self) -> None:
        create_source_freeze_manifest(episode_root=self.root)
        manifest_path = self.root / SOURCE_FREEZE_RELATIVE
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        real_parent = self.scratch / "real"
        real_parent.mkdir()
        alias_parent = self.scratch / "alias"
        alias_parent.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(SourceFreezeError, "symlink component"):
            stage_source_freeze(
                manifest_path,
                expected_file_sha256=manifest_hash,
                stage_parent=alias_parent,
                episode_root=self.root,
                scratch_root=self.scratch,
            )
        outside_parent = self.root / "outside-stage"
        outside_parent.mkdir()
        with self.assertRaisesRegex(SourceFreezeError, "escapes EP12 root"):
            stage_source_freeze(
                manifest_path,
                expected_file_sha256=manifest_hash,
                stage_parent=outside_parent,
                episode_root=self.root,
                scratch_root=self.scratch,
            )

    def test_exact_once_test_launch_lost_ack_and_recovery(self) -> None:
        accepted: list[list[str]] = []

        def disconnect(command: list[str]) -> str:
            progress = json.loads((self.root / PROGRESS_RELATIVE).read_text())
            self.assertEqual(progress["state"], "submission_uncertain")
            accepted.append(command)
            raise RuntimeError("lost accepted reply")

        with self.assertRaisesRegex(RuntimeError, "lost accepted"):
            launch_executor_tests(
                episode_root=self.root,
                submitter=disconnect,
                job_resolver=lambda name, user: None,
            )
        self.assertEqual(len(accepted), 1)
        with self.assertRaisesRegex(TestLaunchError, "refusing duplicate"):
            launch_executor_tests(
                episode_root=self.root,
                submitter=lambda command: self.fail("must not resubmit"),
                job_resolver=lambda name, user: None,
            )

        receipt = launch_executor_tests(
            episode_root=self.root,
            submitter=lambda command: self.fail("recovery must not resubmit"),
            job_resolver=lambda name, user: {
                "job_id": "80801",
                "job_name": name,
                "user": user,
            },
        )
        self.assertEqual(receipt["slurm_job_id"], "80801")
        self.assertEqual(receipt["submission_user"], self.user)
        self.assertNotIn("ALL", receipt["sbatch_command"][-2])
        export = next(
            item for item in receipt["sbatch_command"] if item.startswith("--export=")
        )
        self.assertIn("EP12_SOURCE_FREEZE_MANIFEST=", export)
        self.assertIn("EP12_SOURCE_FREEZE_MANIFEST_SHA256=", export)
        self.assertEqual(
            read_test_launch_receipt(
                self.root / RECEIPT_RELATIVE, episode_root=self.root
            ),
            receipt,
        )

    def test_test_receipt_tamper_and_recovery_user_mismatch_fail(self) -> None:
        launch_executor_tests(
            episode_root=self.root,
            submitter=lambda command: "80802;sherlock",
            job_resolver=lambda name, user: None,
        )
        path = self.root / RECEIPT_RELATIVE
        value = json.loads(path.read_text(encoding="utf-8"))
        value["slurm_job_id"] = "99999"
        path.write_text(canonical_json(value) + "\n", encoding="utf-8")
        with self.assertRaises(TestLaunchError):
            read_test_launch_receipt(path, episode_root=self.root)

        name = "ep12-tests-deadbeef"
        with self.assertRaisesRegex(TestLaunchError, "mismatched user"):
            _unique_recovery(f"80802|{name}|other-user\n", name, self.user)

    @mock.patch("ep12_executor.test_launch.subprocess.run")
    def test_recovery_queries_are_scoped_and_sanitized(self, run: mock.Mock) -> None:
        name = "ep12-tests-feedface"
        run.return_value = subprocess.CompletedProcess(
            [], 0, stdout=f"80803|{name}|{self.user}\n", stderr=""
        )
        self.assertEqual(
            _default_job_resolver(name, self.user),
            {"job_id": "80803", "job_name": name, "user": self.user},
        )
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertEqual(command[0], "sacct")
        self.assertIn("--user", command)
        self.assertEqual(command[command.index("--user") + 1], self.user)
        environment = run.call_args.kwargs["env"]
        self.assertFalse(
            any(name.startswith(("SBATCH_", "SLURM_", "EP12_")) for name in environment)
        )


if __name__ == "__main__":
    unittest.main()
