from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from ep12_executor import production_readiness as readiness


class ProductionReadinessTransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.episode_root = Path(self.temporary.name).resolve()
        self.outputs_root = self.episode_root / "outputs"
        self.outputs_root.mkdir()
        self.run_root = self.outputs_root / "development_run002"
        self.attestation_dir = self.run_root / "observed_attestation"
        self.finalization_dir = self.run_root / "observed_finalization"
        self.executor_root = self.outputs_root / "executor"
        self.manifest = (
            self.outputs_root
            / "prelaunch/scientific_contract_transition/manifest.json"
        )
        self.attestation = {
            "attestation_manifest_sha256": "1" * 64,
            "attestation_artifact_file_sha256": "2" * 64,
            "attestation_sha256": "3" * 64,
            "verified_trial_count": 36,
        }
        self.transition = {
            "observed_attestation_manifest_file_sha256": "1" * 64,
            "observed_attestation_artifact_file_sha256": "2" * 64,
            "observed_attestation_sha256": "3" * 64,
            "observed_trial_count": 36,
            "exact_36_attestation_set_matches_transition_evidence": True,
            "observed_prefix_contract_sha256": "4" * 64,
            "active_contract_sha256": "5" * 64,
            "transition_manifest_file_sha256": "6" * 64,
            "transition_receipt_file_sha256": "7" * 64,
            "transition_receipt_sha256": "8" * 64,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _call(self) -> dict:
        with mock.patch.object(
            readiness, "verify_attestation_root", return_value=self.attestation
        ), mock.patch.object(
            readiness,
            "verify_scientific_contract_transition",
            return_value=self.transition,
        ) as verify:
            result = readiness.verify_scientific_contract_transition_gate(
                transition_manifest_path=self.manifest,
                transition_manifest_file_sha256="6" * 64,
                transition_receipt_sha256="8" * 64,
                observed_run_root=self.run_root,
                observed_attestation_dir=self.attestation_dir,
                observed_finalization_dir=self.finalization_dir,
                policy=mock.Mock(),
                executor_root=self.executor_root,
                outputs_root=self.outputs_root,
            )
        verify.assert_called_once_with(
            episode_root=self.episode_root,
            executor_root=self.executor_root,
            manifest_path=self.manifest,
            expected_manifest_file_sha256="6" * 64,
            expected_transition_receipt_sha256="8" * 64,
            observed_attestation_dir=self.attestation_dir,
            observed_finalization_dir=self.finalization_dir,
        )
        return result

    def test_exact_attestation_file_and_semantic_hashes_pass(self) -> None:
        self.assertEqual(self._call(), self.transition)

    def test_same_count_but_different_attestation_semantic_hash_fails(self) -> None:
        self.transition["observed_attestation_sha256"] = "9" * 64
        with self.assertRaisesRegex(
            readiness.ReadinessError,
            "transition and rebuilt observed attestation differ",
        ):
            self._call()


if __name__ == "__main__":
    unittest.main()
