"""Bounded reporting fixtures; run only in the coordinator's Slurm allocation."""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import stdtrit

import report_repeated_image_control as reporter
import repeated_image_control as model


def participant(person, second_gain=.25):
    recipe = json.loads((Path(__file__).resolve().parents[1] / "config/core_recipe_20261013_001.json").read_text())
    binding = {"schema": model.SCHEMA, "participant_id": person, "data_role": "development",
        "recipe": recipe, "identity_recipe": deepcopy(model.IDENTITY_RECIPE),
        "image_ids": [f"image-{i:03d}" for i in range(200)],
        "channel_order": [f"c{i:02d}" for i in range(62)],
        "sequence_support": [{"sequence_id": f"sequence-{i}", "ordinal": i, "samples": 1000,
                              "retained_samples": 900} for i in range(12)]}
    directions = []
    for parity, name in enumerate(model.DIRECTIONS):
        train = [f"sequence-{i}" for i in range(parity, 12, 2)]
        heldout = [f"sequence-{i}" for i in range(1-parity, 12, 2)]
        # First direction: ratio of means yields .5; mean ratios would not.
        raw = [4., 1.] * 3 if parity == 0 else [1.-second_gain] * 6
        nuisance = [1., 9.] * 3 if parity == 0 else [1.] * 6
        gain = .5 if parity == 0 else float(1 - np.mean(raw))
        directions.append({"status": "complete_direction", "binding": binding,
            "direction": name, "training_sequence_ids": train, "heldout_sequence_ids": heldout,
            "training_whitening": {"sequence_ids": train}, "inner_validations_completed": 6,
            "inner_mean_normalized_identity_losses": [1.] * 13, "selected_penalty_index": 12,
            "selected_ridge_penalty": float(model.RIDGE_GRID[12]),
            "identity_sequence_losses": raw, "nuisance_sequence_losses": nuisance, "gain": gain})
    return {"schema": model.SCHEMA, "status": "complete_repeated_image_control", "participant_id": person,
        "data_role": "development", "binding": binding, "directions": directions,
        "gain": float(np.mean([item["gain"] for item in directions])), "image_identity_count": 200,
        "physical_sequence_count": 12, "inference_unit": "participant", "cohort_inference_performed": False,
        "audit_access": False, "concept_or_false_models_computed": False}


class RepeatedControlReportTests(unittest.TestCase):
    def test_paired_directions_not_sequences_are_averaged_within_person(self):
        row = participant("sub-07")
        self.assertEqual(reporter._participant(row, "sub-07"), .375)
        zero_sequence = deepcopy(row)
        zero_sequence["directions"][0]["nuisance_sequence_losses"] = [0., 10.] * 3
        self.assertEqual(reporter._participant(zero_sequence, "sub-07"), .375)
        invalid = deepcopy(row)
        invalid["gain"] = .4
        with self.assertRaisesRegex(ValueError, "participant_gain_must_average"):
            reporter._participant(invalid, "sub-07")

    def test_wrong_person_split_whitening_and_incomplete_direction_rejected(self):
        row = participant("sub-07")
        for change in ("person", "split", "whitening", "direction", "inner", "penalty"):
            invalid = deepcopy(row)
            if change == "person":
                invalid["participant_id"] = "sub-08"
            elif change == "split":
                invalid["directions"][0]["heldout_sequence_ids"][0] = "sequence-0"
            elif change == "whitening":
                invalid["directions"][0]["training_whitening"]["sequence_ids"] = ["sequence-1"] * 6
            elif change == "direction":
                invalid["directions"].pop()
            elif change == "inner":
                invalid["directions"][0]["inner_validations_completed"] = 5
            else:
                invalid["directions"][0]["selected_penalty_index"] = 0
            with self.subTest(change=change), self.assertRaises(ValueError):
                reporter._participant(invalid, "sub-07")

    def test_original_three_endpoint_bounds_and_strict_thresholds(self):
        # Constant exact margins must not pass strict inequalities.
        values = np.tile([.002, .001, 0.], (30, 1))
        edge = reporter._evidence(values)
        self.assertEqual(edge["lower_bound_exceeds_threshold"], {"G": False, "T": False, "R": False})
        values[:, 2] = np.arange(30) / 100.
        actual = reporter._evidence(values)
        family = actual["simultaneous_one_sided_positive_family"]
        # Arithmetic progression: mean=.145; sample variance=30*31/(12*100^2).
        se = math.sqrt(31 / 120000.)
        self.assertAlmostEqual(family["mean"][2], .145)
        self.assertAlmostEqual(family["lower"][2], .145 - stdtrit(29, 1-.05/3)*se)
        self.assertEqual(family["participant_count"], 30)
        self.assertEqual(family["degrees_of_freedom"], 29)
        self.assertEqual(family["tail_alpha"], .05/3)
        self.assertTrue(actual["endpoint_control_pass"]["R"])
        # Positive cohort mean is insufficient when one person's deletion reverses it.
        values[:, 2] = -.01
        values[0, 2] = 1.
        self.assertFalse(reporter._evidence(values)["every_leave_one_participant_mean_positive"]["R"])

    def test_complete_cohort_required_before_publication_and_three_line_render(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            people = reporter.DEVELOPMENT_PARTICIPANTS
            manifest = {"role": "development", "participants": people, "event_count": 739440,
                        "frozen_event_join": "development_event_image_join.tsv.gz"}
            (root / "manifest.json").write_text(json.dumps(manifest))
            core = {"status": "complete_30_person_core_calibration", "participants": people,
                "participant_count": 30, "outer_splits": 180, "inner_splits": 900,
                "participant_G_T": {person: [-.0001, -.0005] for person in people},
                "G": -.0001, "T": -.0005, "delta_Q": .002, "delta_T": .001,
                "reference_underidentified": True, "audit_access": False, "concept_or_false_scores_computed": False,
                "scientific_role": "registered_development_core_calibration"}
            (root / "core.json").write_text(json.dumps(core))
            args = dict(development_manifest=root/"manifest.json", repeated_root=root/"repeated",
                        core_report=root/"core.json", output_root=root/"report", figure_root=root/"figures")
            wrong_manifest = deepcopy(manifest)
            wrong_manifest["participants"] = tuple(f"sub-{i:02d}" for i in range(1, 31))
            (root / "manifest.json").write_text(json.dumps(wrong_manifest))
            with self.assertRaisesRegex(ValueError, "original_complete_development_manifest"):
                reporter.report(**args)
            (root / "manifest.json").write_text(json.dumps(manifest))
            for person in people[:-1]:
                directory = root / "repeated" / person
                directory.mkdir(parents=True)
                (directory / "complete.json").write_text(json.dumps(participant(person)))
            with self.assertRaisesRegex(ValueError, "all30_completed"):
                reporter.report(**args)
            self.assertFalse(args["output_root"].exists())
            directory = root / "repeated" / people[-1]
            directory.mkdir()
            (directory / "complete.json").write_text(json.dumps(participant(people[-1])))
            # A complete but mismatched core roster must not change the inferred population.
            changed = deepcopy(core)
            changed["participants"] = list(people[:-1]) + ["sub-99"]
            (root / "core.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "same30_released_core"):
                reporter.report(**args)
            (root / "core.json").write_text(json.dumps(core))
            result = reporter.report(**args)
            self.assertTrue(result["endpoint_control_pass"]["R"])
            self.assertFalse(result["all_three_positive_controls_pass"])
            self.assertFalse(result["primary_result_rescued"])
            self.assertEqual(result["participant_count"], 30)
            self.assertEqual(len((args["output_root"] / "summary.md").read_text().splitlines()), 3)
            for extension in ("png", "pdf"):
                self.assertGreater((args["figure_root"] / ("repeated_image_G_T_R." + extension)).stat().st_size, 0)
            with self.assertRaisesRegex(ValueError, "fresh_distinct"):
                reporter.report(**args)


if __name__ == "__main__":
    unittest.main()
