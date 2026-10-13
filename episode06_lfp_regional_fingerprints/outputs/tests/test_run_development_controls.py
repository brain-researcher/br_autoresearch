"""One control-runner access/publication integration; Slurm execution only.

Numerical fitting and the separately tested metadata/profile-control models
are mocked. Actual donor maps, native identity validation, exclusive streams,
profile scoring and complete-roster outcome admission remain exercised.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_development_controls as runner
from component_scoring import BoundActivity
from primary_profile import BlockScore, ProfileScore
from test_run_initial_profiles import (CandidateFixture, design, fabricated_fit,
                                       fabricated_prediction)


def check():
    pass


class DevelopmentControlRunnerTests(unittest.TestCase):
    def test_shared_maps_precede_values_and_complete_predictions_precede_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            original, attempt = base / "starting001", base / "controls001"
            original.mkdir()
            (original / "scores").mkdir()
            view_path, report_path = base / "development_views.h5", base / "metadata.json"
            initial = runner.initial
            initial._json_new(original / "recipes.json",
                {"program": initial._program(), "views_path": str(view_path)})
            initial._json_new(original / "scores/complete.json", {"scored_cells": 84,
                "candidates": [candidate.candidate_id for candidate in initial.CANDIDATES]})
            initial._json_new(original / "scores/starting_10.json", {
                "candidate": initial._plain(asdict(runner.BASELINE)),
                "directional_balanced_accuracy": [1., 1.]})
            initial._json_new(report_path, {"fixture": "metadata_only"})
            # Complete original regional profiles with independently supplied
            # opposite frequency patterns. These are already-scored artifacts,
            # not new outcome reads or re-fits in the control runner.
            with initial._new_file(original / "scores/regional_scores.pickle", binary=True) as handle:
                for spec in initial.SESSIONS:
                    for region in initial.REGIONS:
                        for family, recipe in initial.FAMILIES:
                            bound = replace(design(spec.filename, region, "evaluation"),
                                axis_ids=tuple(f"{region}:pc_{index}" for index in range(recipe.rank)))
                            scores = range(1, 10) if region == "M1" else range(9, 0, -1)
                            profile = ProfileScore(region, bound, 100., tuple(
                                BlockScore(block, 100. - value, value / 100., None)
                                for block, value in zip(initial.BLOCKS, scores)))
                            initial._dump_cell(handle, {"family": family, "session": spec.filename,
                                                        "region": region, "profile": profile})
            metadata_calls, fit_calls, prediction_calls, reads = [], [], [], []

            def donor_indices(session, role):
                records = json.loads((attempt / "donor_mappings.json").read_text())
                self.assertEqual(len(records), 14)
                matches = [record for record in records
                           if record["session_id"] == session and record["role"] == role]
                self.assertEqual(len(matches), 1)
                return np.asarray(matches[0]["source_indices"])

            def marked(lfp):
                offset = 0 if lfp.region == "M1" else 1000
                return replace(lfp, values=np.broadcast_to(
                    (np.arange(16) + offset)[:, None, None, None], (16, 10, 15, 9)).copy())

            class Candidates(CandidateFixture):
                def metadata(self, session, region, role):
                    metadata_calls.append((session, region, role))
                    return SimpleNamespace(design=design(session, region, role))
                def training(self, session, region):
                    self_outer.assertEqual(len(metadata_calls), 28)
                    donor_indices(session, "training")
                    lfp, counts = super().training(session, region)
                    return marked(lfp), counts
                def features(self, session, region, role):
                    self_outer.assertEqual(len(metadata_calls), 28)
                    donor_indices(session, role)
                    return marked(super().features(session, region, role))

            def fit(lfp, counts, *, recipe, check):
                fit_calls.append((lfp.design.session_id, lfp.region))
                expected = donor_indices(lfp.design.session_id, "training")
                offset = 0 if lfp.region == "M1" else 1000
                np.testing.assert_array_equal(lfp.values[:, 0, 0, 0], expected + offset)
                np.testing.assert_array_equal(counts.values, np.full(counts.design.shape, 100.))
                self.assertEqual(recipe, runner.RECIPE)
                return fabricated_fit(lfp, counts, recipe=recipe, check=check)

            def predict(fitted, features, *, check):
                prediction_calls.append((features.design.session_id, features.region))
                expected = donor_indices(features.design.session_id, "evaluation")
                offset = 0 if features.region == "M1" else 1000
                np.testing.assert_array_equal(features.values[:, 0, 0, 0], expected + offset)
                return fabricated_prediction(fitted, features, check=check)

            self_outer = self
            class Outcomes:
                def __init__(self, path, *, role, check):
                    self_outer.assertEqual((path, role), (view_path, "evaluation"))
                    self_outer.assertEqual(len(fit_calls), 14)
                    self_outer.assertEqual(len(prediction_calls), 14)
                    self_outer.assertEqual(json.loads((attempt / "fit_complete.json").read_text())["cells"], 14)
                    check()
                def targets(self, session, region):
                    reads.append((session, region))
                    bound = design(session, region, "evaluation")
                    return BoundActivity(bound, np.full(bound.shape, 110.))

            def metadata_control(report, *, check):
                self.assertEqual(report, {"fixture": "metadata_only"})
                return {"status": "complete", "directions": [
                    {"balanced_accuracy": .5}, {"balanced_accuracy": .5}]}

            def profile_controls(left, right, **kwargs):
                self.assertEqual((len(left), len(right)), (3, 4))
                self.assertTrue(all(len(value.profile.blocks) == 6 for value in (*left, *right)))
                self.assertEqual(kwargs["recipe"], runner.BASELINE.classifier)
                return {"fixture": "two_separately_tested_profile_controls",
                        "leave_one_out": {"complete": True},
                        "band_permutation": {"complete": True}}

            with patch.object(initial, "ATTEMPT_PATH", original), \
                 patch.object(initial, "_compute_only"), \
                 patch.object(runner, "ATTEMPT", attempt), \
                 patch.object(runner, "REPORT", report_path), \
                 patch.object(runner, "CandidateDevelopmentViews", side_effect=Candidates), \
                 patch.object(runner, "DevelopmentOutcomeViews", side_effect=Outcomes) as outcomes, \
                 patch.object(runner.prediction, "fit_bandwise", side_effect=fit), \
                 patch.object(runner.prediction, "predict_bandwise", side_effect=predict), \
                 patch.object(runner, "evaluate_metadata_classifier", side_effect=metadata_control), \
                 patch.object(runner, "evaluate_profile_controls", side_effect=profile_controls):
                runner.fit_predict(views_path=view_path, attempt_dir=attempt, check=check)
                outcomes.assert_not_called()
                expected_roster = [(spec.filename, region)
                    for spec in initial.SESSIONS for region in initial.REGIONS]
                self.assertEqual(fit_calls, expected_roster)
                self.assertEqual(prediction_calls, expected_roster)
                mappings = json.loads((attempt / "donor_mappings.json").read_text())
                self.assertEqual([(row["session_id"], row["role"]) for row in mappings],
                    [(spec.filename, role) for spec in initial.SESSIONS for role in ("training", "evaluation")])
                for row, spec in zip(mappings[::2], initial.SESSIONS):
                    self.assertEqual(row["seed"], [6006201, int(spec.animal != "Mihili"), spec.original_position, 0])
                with patch.object(runner.prediction, "fit_bandwise", side_effect=AssertionError("scoring refit")), \
                     patch.object(runner.prediction, "predict_bandwise", side_effect=AssertionError("scoring prediction")):
                    destination = runner.score(attempt_dir=attempt, check=check)
                self.assertEqual(reads, expected_roster)
                self.assertEqual(outcomes.call_count, 1)
                result = json.loads((destination / "control03_mismatch.json").read_text())
                self.assertEqual(result["directional_balanced_accuracy"], [1., 1.])
                self.assertEqual(result["directional_accuracy_change"], [0., 0.])
                self.assertEqual(result["claim_scope"], "single_fixed_pairing_sensitivity_not_significance_or_control_pass")
                complete = json.loads((destination / "complete.json").read_text())
                self.assertEqual(complete["valid_scientific_control_count"], 4)
                self.assertEqual(complete["profile_controls_report"], "profile_controls.json")
                self.assertTrue(all(complete["available_controls"].values()))
                self.assertIsNone(complete["required_controls_passed"])
                self.assertIsNone(complete["scientific_acceptance"])
                with self.assertRaises(FileExistsError):
                    runner.score(attempt_dir=attempt, check=check)
                self.assertEqual(outcomes.call_count, 1)
                # A partial fit keeps its first native prediction and cannot
                # open outcomes even if a false complete marker is supplied.
                attempt = base / "partial_controls"
                metadata_calls.clear()
                partial_calls = 0
                def stop_second_fit(*args, **kwargs):
                    nonlocal partial_calls
                    partial_calls += 1
                    if partial_calls == 2:
                        raise TimeoutError("fixture deadline")
                    return fit(*args, **kwargs)
                with patch.object(runner, "ATTEMPT", attempt), \
                     patch.object(runner.prediction, "fit_bandwise", side_effect=stop_second_fit):
                    with self.assertRaisesRegex(TimeoutError, "fixture deadline"):
                        runner.fit_predict(views_path=view_path, attempt_dir=attempt, check=check)
                    stream = attempt / "mismatch.pickle"
                    preserved = stream.read_bytes()
                    with stream.open("rb") as handle:
                        first = pickle.load(handle)
                        self.assertEqual((first["session"], first["region"]), expected_roster[0])
                        self.assertEqual(handle.read(), b"")
                    status = [json.loads(line) for line in (attempt / "fit_status.jsonl").read_text().splitlines()]
                    self.assertEqual((status[-1]["event"], status[-1]["cells"]), ("fit_predict_failed", 1))
                    with self.assertRaises(FileNotFoundError):
                        runner.score(attempt_dir=attempt, check=check)
                    initial._json_new(attempt / "fit_complete.json", {"cells": 14})
                    with self.assertRaises(EOFError):
                        runner.score(attempt_dir=attempt, check=check)
                    with self.assertRaises(FileExistsError):
                        runner.fit_predict(views_path=view_path, attempt_dir=attempt, check=check)
                    self.assertEqual(stream.read_bytes(), preserved)
                    self.assertFalse((attempt / "scores").exists())
                    self.assertEqual(outcomes.call_count, 1)


if __name__ == "__main__":
    unittest.main()
