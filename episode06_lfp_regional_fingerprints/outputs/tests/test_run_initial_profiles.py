"""Two bounded stage/reuse integrations; execute only in coordinated Slurm.

Tiny fabricated frozen fits deliberately bypass numerical fitting, which has
its own analytic tests. Real serialization, stage admission, profile scoring,
paired block projection, classification and reference diagnostics run here.
No provider array, HDF file, scheduler or reserved outcome is accessed.
"""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bandwise_prediction as prediction
from component_scoring import BoundActivity, FrozenDesign
import run_initial_profiles as runner


def check():
    pass


def design(session, region, role):
    return FrozenDesign(session, role, tuple(range(8)),
        tuple(f"{session}:{role}:{index:03d}" for index in range(16)),
        tuple(index // 2 for index in range(16)), tuple(range(5, 15)),
        tuple(f"{region}:native_unit_{index}" for index in range(4)))


class CandidateFixture:
    def __init__(self, path, *, check):
        check()

    def training(self, session, region):
        bound = design(session, region, "training")
        return (prediction.BandwiseLFP(region, bound, tuple(range(1, 16)),
                    prediction.BANDS, np.zeros((16, 10, 15, 9))),
                BoundActivity(bound, np.full(bound.shape, 100.)))

    def features(self, session, region, role):
        if role != "evaluation":
            raise AssertionError("the initial program must not request calibration values")
        return prediction.BandwiseLFP(region, design(session, region, role),
            tuple(range(1, 16)), prediction.BANDS, np.zeros((16, 10, 15, 9)))


def fabricated_fit(lfp, observed, *, recipe, check):
    """Known frozen coordinate/template state, not a numerical fitting test."""
    check()
    if observed.design != lfp.design or observed.design.role != "training":
        raise AssertionError("training inputs do not share their exact native design")
    common = dict(region=lfp.region, training_design=lfp.design,
        electrode_ids=lfp.electrode_ids, bands=lfp.bands, recipe=recipe,
        population_axis_ids=tuple(f"{lfp.region}:pc_{index}" for index in range(recipe.rank)),
        population_mean=np.full(4, 100.), loadings=np.eye(4, recipe.rank),
        singular_values=np.asarray([4., 3., 2., 1.]), numerical_rank=4, rank_tolerance=1e-12,
        training_template=np.full((8, 10, 4), 100.))
    if recipe.feature_family == "pca2_cca":
        blocks = tuple(prediction.FittedCCAFeature(block, index, np.zeros(15), np.ones(15),
            np.zeros(15), np.eye(15, 2), np.asarray([2., 1.]), 2, 1e-12, None)
            for index, block in enumerate(recipe.blocks))
        return prediction.FittedCCABandwise(**common, blocks=blocks)
    blocks = tuple(prediction.FittedBlock(block, (index,), np.zeros((1, recipe.rank)),
        np.zeros(recipe.rank)) for index, block in enumerate(recipe.blocks))
    return prediction.FittedBandwise(**common, predictor_mean=np.zeros(9),
                                    predictor_scale=np.ones(9), blocks=blocks)


def fabricated_prediction(fitted, features, *, check):
    check()
    bound = prediction._bind_fitted(fitted, features.design, features.region)
    # Outcomes have residual10 in every coordinate. Known pooled block scores
    # are therefore1-(10-gain)^2/100, opposite across the two physical regions.
    gains = range(1, 10) if features.region == "M1" else range(9, 0, -1)
    values = tuple(BoundActivity(bound, np.full(bound.shape, float(gain))) for gain in gains)
    return prediction.BandwisePredictions(features.region, bound, fitted.recipe.blocks, values)


class InitialProfileTests(unittest.TestCase):
    def test_all_predictions_precede_outcomes_and_twelve_candidates_reuse_six_full_band_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp) / "starting001"
            view_path = Path(temp) / "development_views.h5"
            reads = []
            class OutcomeFixture:
                def __init__(self, path, *, role, check):
                    if path != view_path or role != "evaluation":
                        raise AssertionError("outcome path/role changed after prediction publication")
                    if not (attempt / "fit_complete.json").is_file():
                        raise AssertionError("outcomes opened before all predictions were published")
                    check()
                def targets(self, session, region):
                    reads.append((session, region))
                    bound = design(session, region, "evaluation")
                    return BoundActivity(bound, np.full(bound.shape, 110.))
            with patch.object(runner, "ATTEMPT_PATH", attempt), \
                 patch.object(runner, "_compute_only"), \
                 patch.object(runner, "CandidateDevelopmentViews", side_effect=CandidateFixture) as candidates, \
                 patch.object(runner, "DevelopmentOutcomeViews", side_effect=OutcomeFixture) as outcomes, \
                 patch.object(prediction, "fit_bandwise", side_effect=fabricated_fit) as fits, \
                 patch.object(prediction, "predict_bandwise", side_effect=fabricated_prediction) as predictions:
                runner.fit_predict(views_path=view_path, attempt_dir=attempt, check=check)
                self.assertEqual(fits.call_count, 84)
                self.assertEqual(predictions.call_count, 84)
                self.assertEqual(candidates.call_count, 1)
                outcomes.assert_not_called()
                self.assertEqual(Counter(call.kwargs["recipe"] for call in fits.call_args_list),
                                 Counter({recipe: 14 for _, recipe in runner.FAMILIES}))
                # Each container has exactly14 complete ordered cells, not12
                # duplicated fits per recording or a score-dependent subset.
                for family, _ in runner.FAMILIES:
                    with (attempt / f"{family}.pickle").open("rb") as handle:
                        roster = [(pickle.load(handle)["session"]) for _ in range(14)]
                        self.assertEqual(roster, [spec.filename for spec in runner.SESSIONS for _ in range(2)])
                        self.assertEqual(handle.read(), b"")
                with patch.object(prediction, "fit_bandwise", side_effect=AssertionError("refit during score")), \
                     patch.object(prediction, "predict_bandwise", side_effect=AssertionError("prediction changed during score")), \
                     patch.object(runner, "score_band_profile", wraps=runner.score_band_profile) as scores:
                    destination = runner.score(attempt_dir=attempt, check=check)
                    self.assertEqual(scores.call_count, 84)
                self.assertEqual(outcomes.call_count, 1)
                self.assertEqual(reads, [(spec.filename, region)
                    for spec in runner.SESSIONS for region in ("M1", "PMd")])
                reports = [json.loads((destination / f"starting_{index:02d}.json").read_text())
                           for index in range(12)]
                expected = (("starting_00", 9, None), ("starting_00", 6, None),
                    ("starting_02", 9, None), ("starting_03", 9, None),
                    ("starting_04", 9, None), ("starting_05", 9, None),
                    ("starting_00", 9, 1.), ("starting_00", 6, 1.),
                    ("starting_00", 9, .1), ("starting_09", 9, None),
                    ("starting_09", 6, None), ("starting_09", 9, 1.))
                m1 = [1 - (10 - gain)**2 / 100 for gain in range(1, 10)]
                for report, (family, width, penalty) in zip(reports, expected):
                    self.assertEqual(report["candidate"]["family_id"], family)
                    self.assertEqual(report["candidate"]["block_indices"], list(range(width)))
                    self.assertEqual(report["candidate"]["classifier"]["penalty"], penalty)
                    self.assertEqual(report["candidate"]["classifier"]["method"],
                        "cosine_prototype" if penalty is None else "l2_logistic")
                    self.assertEqual(report["directional_balanced_accuracy"], [1., 1.])
                    self.assertEqual(report["mean_balanced_accuracy_minus_half"], .5)
                    self.assertEqual(len(report["profiles"]), 7)
                    for profile in report["profiles"]:
                        np.testing.assert_allclose(profile["m1_scores"], m1[:width], rtol=0, atol=2e-15)
                        np.testing.assert_allclose(profile["pmd_scores"], m1[::-1][:width], rtol=0, atol=2e-15)
                    self.assertEqual(len(report["joint_label_reference"]["assignments"]), 128)
                    self.assertIsNone(report["scientific_acceptance"])
                    self.assertIsNone(report["search_stop_decision"])
                for a, b in ((0, 6), (0, 8), (1, 7), (9, 11)):
                    self.assertEqual(reports[a]["profiles"], reports[b]["profiles"])
                # Independent fixed-menu oracle: assignments0,1,2 alone reach
                # score.5; tied winners follow the lowest starting ID. This is
                # a report calculation, with no classifier or neural fitting.
                analytic = json.loads(json.dumps(reports))
                for report in analytic:
                    reference = report["joint_label_reference"]
                    reference["complete"] = True
                    for direction in reference["directions"]:
                        direction["assignment_balanced_accuracies"] = [.5] * 128
                for candidate_index, assignment_index in ((0, 0), (3, 0), (1, 1), (2, 1), (2, 2)):
                    for direction in analytic[candidate_index]["joint_label_reference"]["directions"]:
                        direction["assignment_balanced_accuracies"][assignment_index] = 1.
                menu = runner.fixed_menu_reference(analytic, check=check)
                self.assertTrue(menu["complete"])
                self.assertEqual(menu["inclusive_upper_tail_count"], 3)
                self.assertEqual(menu["inclusive_upper_tail_fraction"], 3 / 128.)
                self.assertEqual([value["candidate_id"] for value in menu["selections"][:4]],
                                 ["starting_00", "starting_01", "starting_02", "starting_00"])
                analytic[11]["joint_label_reference"]["directions"][1]["assignment_balanced_accuracies"][127] = None
                incomplete = runner.fixed_menu_reference(analytic, check=check)
                self.assertFalse(incomplete["complete"])
                self.assertIsNone(incomplete["inclusive_upper_tail_fraction"])
                self.assertEqual(incomplete["observed_selection"]["candidate_id"], "starting_00")
                self.assertIsNone(incomplete["selections"][127])
                self.assertEqual(incomplete["unavailable_candidates_by_assignment"][127], ("starting_11",))
                with self.assertRaises(FileExistsError):
                    runner.score(attempt_dir=attempt, check=check)
                self.assertEqual(outcomes.call_count, 1)

    def test_partial_prediction_attempt_is_preserved_and_cannot_open_outcomes_even_with_false_complete_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            attempt = Path(temp) / "starting001"
            view_path = Path(temp) / "development_views.h5"
            calls = 0
            def stop_second_fit(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise TimeoutError("fixture shared resource deadline")
                return fabricated_fit(*args, **kwargs)
            with patch.object(runner, "ATTEMPT_PATH", attempt), \
                 patch.object(runner, "_compute_only"), \
                 patch.object(runner, "CandidateDevelopmentViews", side_effect=CandidateFixture), \
                 patch.object(runner, "DevelopmentOutcomeViews", side_effect=AssertionError("closed outcomes reached")) as outcomes, \
                 patch.object(prediction, "fit_bandwise", side_effect=stop_second_fit), \
                 patch.object(prediction, "predict_bandwise", side_effect=fabricated_prediction):
                with self.assertRaisesRegex(TimeoutError, "shared resource deadline"):
                    runner.fit_predict(views_path=view_path, attempt_dir=attempt, check=check)
                stream = attempt / "starting_00.pickle"
                preserved = stream.read_bytes()
                with stream.open("rb") as handle:
                    first = pickle.load(handle)
                    self.assertEqual((first["session"], first["region"]),
                                     ("Mihili_CO_20140217.mat", "M1"))
                    self.assertEqual(handle.read(), b"")
                status = [json.loads(line) for line in (attempt / "fit_status.jsonl").read_text().splitlines()]
                self.assertEqual(status[-1]["event"], "fit_predict_failed")
                self.assertEqual(status[-1]["completed_cells"], 1)
                self.assertFalse((attempt / "fit_complete.json").exists())
                with self.assertRaises(FileNotFoundError):
                    runner.score(attempt_dir=attempt, check=check)
                with self.assertRaises(FileExistsError):
                    runner.fit_predict(views_path=view_path, attempt_dir=attempt, check=check)
                self.assertEqual(stream.read_bytes(), preserved)
                # A completion declaration is insufficient: the whole exact
                # roster is validated before the outcome reader is constructed.
                (attempt / "fit_complete.json").write_text(json.dumps({"completed_cells": 84,
                    "families": [family for family, _ in runner.FAMILIES]}))
                with self.assertRaises(EOFError):
                    runner.score(attempt_dir=attempt, check=check)
                self.assertFalse((attempt / "scores").exists())
                outcomes.assert_not_called()


if __name__ == "__main__":
    unittest.main()
