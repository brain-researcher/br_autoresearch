"""One exact-grid/stage-boundary test; run only in coordinated Slurm."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_followup_profiles as runner


def check():
    pass


def report(candidate_id, raised_assignments=()):
    # Relative to (.5,.5), moving the4-pair accuracy by-1/16 and
    # the3-pair accuracy by+1/12 improves the mean by exactly1/96.
    values = [[.5] * 128, [.5] * 128]
    for index in raised_assignments:
        values[0][index] = 7 / 16.
        values[1][index] = math.nextafter(7 / 12., 0.)
    directions = []
    animals = ("Mihili", "Chewie-L")
    for index in range(2):
        directions.append({"source_animal": animals[index], "target_animal": animals[1-index],
            "source_sessions": runner.initial.ROSTERS[index],
            "target_sessions": runner.initial.ROSTERS[1-index],
            "observed_balanced_accuracy": values[index][0],
            "assignment_balanced_accuracies": values[index]})
    return {"candidate": {"candidate_id": candidate_id}, "joint_label_reference": {
        "assignments": runner.ASSIGNMENTS, "complete": True, "directions": directions,
        "blocks": [asdict(block) for block in runner.initial.BLOCKS[:6]]}}


class FollowupProfileTests(unittest.TestCase):
    def test_exact_grid_noise_ties_incomplete_references_and_unpublished_predictions_cannot_advance(self):
        baseline = report("starting_10")
        candidate = report("adaptive00", range(128))
        ordinary_floating_delta = .5 * (7/16. + math.nextafter(7/12., 0.)) - .5
        self.assertLess(ordinary_floating_delta, 1/96.)
        result = runner.compare_to_starting10(candidate, baseline, check=check)
        self.assertEqual(result["observed_grid_improvement"], 1)
        self.assertEqual(result["observed_improvement"], 1/96.)
        self.assertEqual(result["median_paired_difference"], 1/96.)
        self.assertEqual(result["noise_tolerance"], 0.)
        self.assertTrue(result["at_least_one_grid_step"])
        self.assertTrue(result["strictly_above_noise"])
        self.assertTrue(result["numerical_improvement_pass"])
        self.assertIsNone(result["applicable_controls_passed"])
        self.assertIsNone(result["material_improvement"])
        self.assertFalse(result["patience_reset_authorized"])
        # Seven nonzero differences leave121 zeros: the122nd ordered
        # deviation is exactly1/96. Equality must fail the strict noise rule.
        tied = runner.compare_to_starting10(report("adaptive01", range(7)), baseline, check=check)
        self.assertEqual(tied["median_paired_difference"], 0.)
        self.assertEqual(tied["noise_tolerance"], 1/96.)
        self.assertTrue(tied["at_least_one_grid_step"])
        self.assertFalse(tied["strictly_above_noise"])
        self.assertFalse(tied["numerical_improvement_pass"])
        missing = deepcopy(candidate)
        missing["joint_label_reference"]["directions"][1]["assignment_balanced_accuracies"][127] = None
        partial = runner.compare_to_starting10(missing, baseline, check=check)
        self.assertFalse(partial["complete_paired_reference"])
        self.assertEqual(partial["observed_improvement"], 1/96.)
        self.assertEqual(len(partial["paired_reference_differences"]), 128)
        self.assertIsNone(partial["paired_reference_differences"][127])
        self.assertIsNone(partial["noise_tolerance"])
        self.assertIsNone(partial["numerical_improvement_pass"])
        self.assertFalse(partial["patience_reset_authorized"])
        wrong_support = deepcopy(candidate)
        wrong_support["joint_label_reference"]["directions"][0]["target_sessions"] = runner.initial.ROSTERS[1][:-1]
        with self.assertRaises(ValueError):
            runner.compare_to_starting10(wrong_support, baseline, check=check)
        off_grid = deepcopy(candidate)
        off_grid["joint_label_reference"]["directions"][0]["assignment_balanced_accuracies"][127] = 1/7.
        with self.assertRaises(ValueError):
            runner.compare_to_starting10(off_grid, baseline, check=check)

        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp) / "adaptive001"
            attempt.mkdir()
            marker = attempt / "fit_complete.json"
            with patch.object(runner, "ATTEMPT_PATH", attempt), \
                 patch.object(runner, "_compute_only"), \
                 patch.object(runner, "_load_baseline", side_effect=AssertionError("incomplete fit stage reached baseline")) as incumbent, \
                 patch.object(runner, "DevelopmentOutcomeViews", side_effect=AssertionError("closed targets opened")) as outcomes:
                with self.assertRaises(FileNotFoundError):
                    runner.score(attempt_dir=attempt, check=check)
                marker.write_text(json.dumps({"completed_cells": 27,
                    "families": [name for name, _ in runner.FAMILIES]}))
                with self.assertRaises(PermissionError):
                    runner.score(attempt_dir=attempt, check=check)
                # A claimed28-cell completion cannot substitute for actual
                # published streams. Refusal precedes even incumbent loading.
                marker.write_text(json.dumps({"completed_cells": 28,
                    "families": [name for name, _ in runner.FAMILIES]}))
                (attempt / "recipes.json").write_text(json.dumps({"program": runner._program(),
                    "views_path": str(Path(temp) / "development_views.h5")}))
                with self.assertRaises(FileNotFoundError):
                    runner.score(attempt_dir=attempt, check=check)
                self.assertFalse((attempt / "scores").exists())
                incumbent.assert_not_called()
                outcomes.assert_not_called()


if __name__ == "__main__":
    unittest.main()
