"""Focused follow-up admission checks; run only on coordinated compute."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import reference_s5_family as family
import reference_s5_operator as operator
import reference_s5_prediction as prediction
import reference_s5_refinement as refinement


class TestS5Refinement(unittest.TestCase):
    def arguments(self, manifest, structure="R1"):
        return SimpleNamespace(manifest=manifest, structure=structure, point=0,
                               stop_after_frame=4000, run_dir="/unused")

    def test_selected_wave_preserves_science_and_rejects_changed_admission(self):
        manifest = refinement.refinement_manifest()
        self.assertEqual(manifest["family"], family.specification())
        self.assertEqual(manifest["operator"], operator.operator_specification())
        self.assertEqual(manifest["calibration"], dict(source_target_db=-12.6,
            gate_target_db=-37.2, absolute_error_max_db=3., source_must_exceed_gate=True))
        self.assertEqual(len(manifest["points"]), 9)
        self.assertEqual(len({tuple(sorted(p.items())) for p in manifest["points"]}), 9)
        for point in manifest["points"]:
            family.Parameters(**point)
        self.assertIsNone(refinement.validate_manifest(json.loads(json.dumps(manifest))))
        mutations = []
        changed = copy.deepcopy(manifest); changed["points"].pop(); mutations.append(changed)
        changed = copy.deepcopy(manifest); changed["points"].reverse(); mutations.append(changed)
        changed = copy.deepcopy(manifest); changed["points"][0] = changed["points"][1]; mutations.append(changed)
        changed = copy.deepcopy(manifest); changed["points"][0]["source_load_siemens"] = 21e-6; mutations.append(changed)
        changed = copy.deepcopy(manifest); changed["calibration"]["absolute_error_max_db"] = 4.; mutations.append(changed)
        changed = copy.deepcopy(manifest); changed["operator"]["frames"] = 100; mutations.append(changed)
        changed = copy.deepcopy(manifest); changed["family"]["structures"] = ["R1"]; mutations.append(changed)
        changed = copy.deepcopy(manifest); changed["held_aside"] = "open_now"; mutations.append(changed)
        for value in mutations:
            with self.subTest(value=value), self.assertRaises(ValueError):
                refinement.validate_manifest(value)

    def test_short_record_bad_point_or_manifest_stops_before_delegation(self):
        context = dict(allocated_cpus=1, numerical_workers=1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(refinement.refinement_manifest()))
            with patch.object(prediction, "run") as delegate:
                args = self.arguments(path); args.stop_after_frame = 100
                with self.assertRaises(ValueError): refinement.run(args, context)
                args = self.arguments(path); args.point = 9
                with self.assertRaises(ValueError): refinement.run(args, context)
                args = self.arguments(path)
                with self.assertRaises(ValueError):
                    refinement.run(args, dict(allocated_cpus=2, numerical_workers=1))
                invalid = refinement.refinement_manifest()
                invalid["points"][0]["adjacent_source_capacitance_farad"] *= 2
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError): refinement.run(args, context)
                delegate.assert_not_called()

    def test_process_local_factory_and_validator_restore_on_success_and_failure(self):
        original_factory, original_validator = family.AirFamily, prediction.validate_manifest
        context = dict(allocated_cpus=1, numerical_workers=1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            refinement.prepare(path)
            for structure in ("R1", "R2", "R3"):
                for fail in (False, True):
                    def delegate(args, actual_context):
                        self.assertIs(prediction.validate_manifest, refinement.validate_manifest)
                        if structure == "R1": self.assertIs(family.AirFamily, original_factory)
                        else: self.assertIsNot(family.AirFamily, original_factory)
                        self.assertEqual(actual_context, context)
                        if fail: raise RuntimeError("deliberate delegate failure")
                        return {"status": "complete_record"}
                    with self.subTest(structure=structure, fail=fail), patch.object(prediction, "run", side_effect=delegate):
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                                refinement.run(self.arguments(path, structure), context)
                        else:
                            self.assertEqual(refinement.run(self.arguments(path, structure), context)["status"], "complete_record")
                    self.assertIs(family.AirFamily, original_factory)
                    self.assertIs(prediction.validate_manifest, original_validator)


if __name__ == "__main__":
    unittest.main()
