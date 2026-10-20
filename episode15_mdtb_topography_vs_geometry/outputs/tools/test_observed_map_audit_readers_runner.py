"""Boundary tests only; no release containers or scientific payloads are used."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import observed_map_audit_readers as readers
import run_observed_map_audit as runner
from observed_map_audit_context import ADOPTION_KEY, SPECIFICATION


POLICY = {ADOPTION_KEY: {"status": "adopted", "specification": SPECIFICATION,
                       "authority": "test fixture only", "adopted_at_utc": "fixture"}}


class NeverInspect:
    def __iter__(self):
        raise AssertionError("unadopted audit input inspected")


class AuditReaderRunnerTests(unittest.TestCase):
    def test_unadopted_readers_reject_before_input_or_support_access(self):
        with patch.object(readers, "load_selection", side_effect=AssertionError("support opened")):
            with self.assertRaises(PermissionError):
                readers.read_audit_calibration(policy={}, release_bindings=NeverInspect(),
                    member_rows=NeverInspect(), condition_multiplicity_by_session=NeverInspect(),
                    common_support=NeverInspect(), derived_support=NeverInspect(), check=lambda: None)
            with self.assertRaises(PermissionError):
                readers.read_audit_observations(policy={}, frozen_predictions=NeverInspect(),
                    release_bindings=NeverInspect(), member_rows=NeverInspect(),
                    condition_multiplicity_by_session=NeverInspect(), check=lambda: None)

    def test_B_requires_exact_frozen_axis_before_inspecting_bindings(self):
        with patch.object(readers, "validate_prediction_bundle", return_value={"voxel_ids": (5, 6)}), \
             patch.object(readers, "load_selection", return_value={"selected_voxel_ids": (5, 7)}):
            with self.assertRaisesRegex(ValueError, "existing_development_frozen188_axis"):
                readers.read_audit_observations(policy=POLICY, frozen_predictions={},
                    release_bindings=NeverInspect(), member_rows=NeverInspect(),
                    condition_multiplicity_by_session=NeverInspect(), check=lambda: None)

    def _paths(self, root):
        return patch.multiple(runner, RUN_ROOT=root, PREDICTION_DIR=root / "predictions",
                              EVALUATION_DIR=root / "evaluation", EXECUTION_PATH=root / "audit_execution.json")

    def test_published_complete_evaluation_does_not_reopen_B(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "sealed-role-metadata.json"
            (root / "evaluation").mkdir()
            state = {"status": "incomplete", "binding": {"prediction_dir": str(root / "predictions"),
                "evaluation_dir": str(root / "evaluation"), "evaluation_manifest": str(manifest)},
                "attempts": [{"started_utc": "fixture"}]}
            (root / "audit_execution.json").write_text(json.dumps(state))
            with self._paths(root), patch.object(runner, "load_frozen_predictions", return_value={}), \
                 patch.object(runner, "_completed_evaluation", return_value={"status": "complete"}), \
                 patch.object(runner, "load_audit_inputs", side_effect=AssertionError("B metadata reopened")):
                result = runner.evaluate(evaluation_manifest=manifest, policy=POLICY,
                                         check=lambda: None, durable_byte_cap=100000)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(json.loads((root / "audit_execution.json").read_text())["status"], "complete")

    def test_failed_access_can_only_resume_explicitly_with_same_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "sealed-role-metadata.json"
            with self._paths(root), patch.dict(runner.os.environ, {"SLURM_JOB_ID": "fixture"}), \
                 patch.object(runner, "load_frozen_predictions", return_value={}), \
                 patch.object(runner, "load_audit_inputs", return_value={}), \
                 patch.object(runner, "read_audit_observations", side_effect=RuntimeError("interruption")) as opening:
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    runner.evaluate(evaluation_manifest=manifest, policy=POLICY,
                                    check=lambda: None, durable_byte_cap=100000)
                self.assertEqual(opening.call_count, 1)
                with self.assertRaisesRegex(ValueError, "explicit_same_procedure"):
                    runner.evaluate(evaluation_manifest=manifest, policy=POLICY,
                                    check=lambda: None, durable_byte_cap=100000)
                with self.assertRaisesRegex(ValueError, "identical_freeze_and_paths"):
                    runner.evaluate(evaluation_manifest=root / "other.json", policy=POLICY,
                        check=lambda: None, durable_byte_cap=100000, engineering_resume_reason="repaired reader")
                self.assertEqual(opening.call_count, 1)
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    runner.evaluate(evaluation_manifest=manifest, policy=POLICY,
                        check=lambda: None, durable_byte_cap=100000, engineering_resume_reason="repaired reader")
                self.assertEqual(opening.call_count, 2)
            state = json.loads((root / "audit_execution.json").read_text())
            self.assertEqual(len(state["attempts"]), 2)
            self.assertFalse(state["attempts"][0]["prior_B_access_disclosed"])
            self.assertTrue(state["attempts"][1]["prior_B_access_disclosed"])
            self.assertEqual(state["status"], "failed")

    def test_started_B_execution_prohibits_refitting_before_development_IO(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "audit_execution.json").write_text("{}")
            with self._paths(root), patch.object(runner, "load_released_inputs",
                                                side_effect=AssertionError("development refit started")):
                with self.assertRaisesRegex(ValueError, "refitting_forbidden"):
                    runner.predict(development_manifest=root / "development.json",
                        calibration_manifest=root / "A.json", policy=POLICY,
                        check=lambda: None, durable_byte_cap=100000)


if __name__ == "__main__":
    unittest.main()
