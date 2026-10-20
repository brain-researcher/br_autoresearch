"""Center001 admission and temporary-hook checks; coordinated compute only."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import reference_s5_centering as centering
import reference_s5_family as family
import reference_s5_prediction as prediction
import reference_s5_refinement as refinement


class TestS5Centering(unittest.TestCase):
    def arguments(self, path, structure="R1"):
        return SimpleNamespace(manifest=path, structure=structure, point=0,
                               stop_after_frame=4000, run_dir="/unused")

    def test_exact_center_manifest_preserves_science_and_search_history(self):
        manifest = centering.centering_manifest()
        prior = refinement.refinement_manifest()
        for key in ("family", "operator", "calibration", "contact_alternatives"):
            self.assertEqual(manifest[key], prior[key])
        self.assertEqual(manifest["round_id"], "center001")
        self.assertEqual(manifest["search_history_required"], dict(initial001=36, interior001=27, center001=3))
        self.assertFalse(manifest["selection"]["predicted_interpolation_is_measured_result"])
        self.assertEqual(len(manifest["points"]), 1)
        point = manifest["points"][0]
        self.assertEqual(point["source_load_siemens"], 5.799642800220423e-6)
        self.assertGreater(point["adjacent_source_capacitance_farad"], 5.6e-12)
        self.assertLess(point["adjacent_source_capacitance_farad"], 5.7e-12)
        family.Parameters(**point)
        self.assertIsNone(centering.validate_manifest(json.loads(json.dumps(manifest))))
        changed = []
        value = copy.deepcopy(manifest); value["points"][0]["source_load_siemens"] = 6e-6; changed.append(value)
        value = copy.deepcopy(manifest); value["points"].append(copy.deepcopy(point)); changed.append(value)
        value = copy.deepcopy(manifest); value["calibration"]["absolute_error_max_db"] = 4.; changed.append(value)
        value = copy.deepcopy(manifest); value["operator"]["frames"] = 100; changed.append(value)
        value = copy.deepcopy(manifest); value["family"]["structures"] = ["R1"]; changed.append(value)
        value = copy.deepcopy(manifest); value["search_history_required"]["initial001"] = 0; changed.append(value)
        value = copy.deepcopy(manifest); value["held_aside"] = "open_now"; changed.append(value)
        for index, value in enumerate(changed):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                centering.validate_manifest(value)

    def test_invalid_point_or_manifest_and_short_record_do_not_run_physics(self):
        context = dict(allocated_cpus=1, numerical_workers=1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            centering.prepare(path)
            with patch.object(prediction, "run") as physics:
                args = self.arguments(path); args.point = 1
                with self.assertRaises(ValueError): centering.run(args, context)
                args = self.arguments(path); args.stop_after_frame = 100
                with self.assertRaises(ValueError): centering.run(args, context)
                path.write_text(json.dumps(refinement.refinement_manifest()))
                with self.assertRaises(ValueError): centering.run(self.arguments(path), context)
                physics.assert_not_called()

    def test_nested_hooks_restore_after_success_and_failure_for_all_structures(self):
        saved = (refinement.validate_manifest, refinement.ROUND_ID,
                 prediction.validate_manifest, family.AirFamily)
        context = dict(allocated_cpus=1, numerical_workers=1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            centering.prepare(path)
            for structure in ("R1", "R2", "R3"):
                for fail in (False, True):
                    def delegated(args, actual_context):
                        self.assertIs(refinement.validate_manifest, centering.validate_manifest)
                        self.assertEqual(refinement.ROUND_ID, "center001")
                        self.assertIs(prediction.validate_manifest, centering.validate_manifest)
                        self.assertIsNone(centering.validate_manifest(json.loads(path.read_text())))
                        if structure == "R1": self.assertIs(family.AirFamily, saved[3])
                        else: self.assertIsNot(family.AirFamily, saved[3])
                        if fail: raise RuntimeError("deliberate delegated failure")
                        return dict(status="complete_record")
                    with self.subTest(structure=structure, fail=fail), patch.object(prediction, "run", side_effect=delegated):
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                                centering.run(self.arguments(path, structure), context)
                        else:
                            self.assertEqual(centering.run(self.arguments(path, structure), context)["status"], "complete_record")
                    self.assertIs(refinement.validate_manifest, saved[0])
                    self.assertEqual(refinement.ROUND_ID, saved[1])
                    self.assertIs(prediction.validate_manifest, saved[2])
                    self.assertIs(family.AirFamily, saved[3])


if __name__ == "__main__":
    unittest.main()
