from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ep12_executor.launch_chain_identity import (
    LaunchChainIdentityError,
    authenticate_test_launch,
)
from ep12_executor.policy import canonical_json, digest_object
from ep12_executor.test_launch import (
    PROGRESS_RELATIVE,
    TEST_WRAPPER_RELATIVE,
    launch_executor_tests,
)


class LaunchChainIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        (self.root / "SEARCH_POLICY.yaml").write_text("schema: test\n", encoding="utf-8")
        wrapper = self.root / TEST_WRAPPER_RELATIVE
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        (self.root / "outputs/executor/example.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        self.receipt = launch_executor_tests(
            episode_root=self.root,
            submitter=lambda command: "81001;sherlock",
            job_resolver=lambda name, user: None,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_authenticates_exact_job_and_rejects_unrelated_id(self) -> None:
        identity = authenticate_test_launch(
            episode_root=self.root, expected_job_id="81001"
        )
        self.assertEqual(identity["test_job_id"], "81001")
        self.assertEqual(
            identity["test_launch_receipt_sha256"], self.receipt["receipt_sha256"]
        )
        with self.assertRaisesRegex(LaunchChainIdentityError, "unrelated"):
            authenticate_test_launch(
                episode_root=self.root, expected_job_id="81002"
            )

    def test_recomputed_progress_tamper_is_rejected_by_transaction_auth(self) -> None:
        path = self.root / PROGRESS_RELATIVE
        progress = json.loads(path.read_text(encoding="utf-8"))
        progress["slurm_job_id"] = "81003"
        core = {
            key: value
            for key, value in progress.items()
            if key != "progress_sha256"
        }
        progress["progress_sha256"] = digest_object(core)
        path.write_text(canonical_json(progress) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(LaunchChainIdentityError, "differs from its transaction"):
            authenticate_test_launch(episode_root=self.root)


if __name__ == "__main__":
    unittest.main()
