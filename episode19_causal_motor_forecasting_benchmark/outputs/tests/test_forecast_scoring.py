"""Five focused scoring checks; execute in the coordinator's Slurm allocation."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
from classical_forecast_backend import GroupIdentity
from forecast_scoring import HORIZONS, ScoringScope, UnresolvedScore, horizon_log_scores, score_matched_forecasts

REGISTRY = json.loads((CODE.parent / "model_recipe_registry.json").read_text())


def probabilities(near=.2, strict=.2, longer=.2):
    return np.log(np.r_[np.repeat(near / 6, 6), np.repeat(strict / 6, 6),
                       np.repeat(longer / 6, 6), 1 - near - strict - longer])


def fixture(source="self_paced", stage="held_out", rows_per_person=2):
    scope = ScoringScope.from_registry(REGISTRY, source=source, stage=stage)
    rows = []
    for person in scope.participants:
        for series in scope.allowed_series or ("run-1",):
            identity = GroupIdentity(source, person, series, f"{source}.{person}.{series}")
            rows.extend((identity, 1. + i * .05) for i in range(rows_per_person))
    rows = tuple(rows)
    predictions = {name: np.tile(probabilities(), (len(rows), 1)) for name in scope.conditions}
    return dict(registry=REGISTRY, scope=scope, expected_row_keys=rows,
                condition_log_probabilities=predictions,
                condition_row_keys={name: rows for name in scope.conditions},
                observed_category=np.full(len(rows), -1, dtype=int),
                observed_through_bins=np.full(len(rows), 18, dtype=int))


class ForecastScoringTests(unittest.TestCase):
    def test_horizon_edges_and_extreme_probabilities_use_finite_binary_log_scores(self):
        p = np.arange(1., 20.) / 190.
        categories = np.array([0, 5, 6, 11, 12, 17, 18, -1])
        coverage = np.full(len(categories), 18)
        for horizon, (lower, upper) in HORIZONS.items():
            result = horizon_log_scores(np.tile(np.log(p), (len(categories), 1)), categories,
                                        coverage, horizon=horizon)
            outcome = (categories >= lower) & (categories < upper)
            np.testing.assert_array_equal(result["binary_outcomes"], outcome.astype(float))
            expected = np.where(outcome, np.log(p[lower:upper].sum()), np.log(1 - p[lower:upper].sum()))
            np.testing.assert_allclose(result["log_scores"], expected, atol=1e-14)
        logits = np.full((2, 19), -1200.)
        logits[:, 6:12] = 0.
        logp = logits - logsumexp(logits, axis=1, keepdims=True)
        result = horizon_log_scores(logp, np.array([6, -1]), np.array([18, 18]), horizon="primary_strict")
        self.assertTrue(np.isfinite(result["log_scores"]).all())
        self.assertLess(result["log_scores"][1], -1100.)
        self.assertAlmostEqual(result["log_scores"][1], -1200. + np.log(13. / 6.), places=10)

    def test_censored_and_early_event_rows_require_full_upper_edge_coverage(self):
        category = np.array([0, -1, -1, 6, 12, 18])
        coverage = np.array([0, 5, 6, 11, 12, 18])
        logp = np.tile(probabilities(), (len(category), 1))
        expected = {"near_event": [False, False, True, True, True, True],
                    "primary_strict": [False, False, False, False, True, True],
                    "longer_diagnostic": [False, False, False, False, False, True]}
        for horizon, mask in expected.items():
            result = horizon_log_scores(logp, category, coverage, horizon=horizon)
            np.testing.assert_array_equal(result["eligible"], mask)
            self.assertTrue(np.isnan(result["log_scores"][~result["eligible"]]).all())
            self.assertTrue(np.isnan(result["binary_outcomes"][~result["eligible"]]).all())
        with self.assertRaisesRegex(ValueError, "follow-up coverage"):
            horizon_log_scores(logp[:1], np.array([18]), np.array([17]), horizon="primary_strict")
        with self.assertRaisesRegex(ValueError, "integer categories"):
            horizon_log_scores(logp, category, coverage.astype(float), horizon="primary_strict")

    def test_unequal_WAY_series_pool_natural_rows_before_equal_participant_weighting(self):
        arguments = fixture("WAY", "development", rows_per_person=1)
        rows = list(arguments["expected_row_keys"])
        identity = rows[0][0]
        rows.extend((identity, 1. + i * .05) for i in range(1, 9))
        rows = tuple(rows)
        baseline = np.stack([probabilities(near=.05, strict=.8 if row[0].participant == 1 and row[0].series_or_run == 1 else .2,
                                            longer=.05) for row in rows])
        arguments.update(expected_row_keys=rows,
            condition_log_probabilities={name: baseline.copy() if name == "non_neural" else np.tile(probabilities(), (len(rows), 1))
                                         for name in arguments["scope"].conditions},
            condition_row_keys={name: rows for name in arguments["scope"].conditions},
            observed_category=np.full(len(rows), -1, dtype=int), observed_through_bins=np.full(len(rows), 18, dtype=int))
        result = score_matched_forecasts(**arguments, horizon="primary_strict")
        person_one = (9 * np.log(.2) + 5 * np.log(.8)) / 14
        self.assertAlmostEqual(result["participant_log_scores"]["non_neural"][0], person_one)
        self.assertAlmostEqual(result["macro_log_scores"]["non_neural"], (person_one + 11 * np.log(.8)) / 12)
        wrong_equal_series = (np.log(.2) + 5 * np.log(.8)) / 6
        self.assertGreater(abs(person_one - wrong_equal_series), .1)
        wrong_equal_rows = (9 * np.log(.2) + (len(rows) - 9) * np.log(.8)) / len(rows)
        self.assertGreater(abs(result["macro_log_scores"]["non_neural"] - wrong_equal_rows), .01)
        self.assertEqual(result["participant_row_counts"][1], 14)

    def test_full_registered_cohort_scope_support_and_matched_row_alignment_are_required(self):
        for source, stage, count in (("WAY", "development", 12), ("self_paced", "development", 15),
                                     ("self_paced", "held_out", 8)):
            arguments = fixture(source, stage, rows_per_person=1)
            self.assertEqual(len(arguments["scope"].participants), count)
            score_matched_forecasts(**arguments, horizon="primary_strict")
        arguments = fixture()
        missing = dict(arguments["condition_log_probabilities"])
        missing.pop("phase_4")
        with self.assertRaisesRegex(UnresolvedScore, "every matched condition"):
            score_matched_forecasts(**{**arguments, "condition_log_probabilities": missing}, horizon="primary_strict")
        order = dict(arguments["condition_row_keys"])
        order["zero_EEG"] = order["zero_EEG"][::-1]
        with self.assertRaisesRegex(ValueError, "exact admitted row order"):
            score_matched_forecasts(**{**arguments, "condition_row_keys": order}, horizon="primary_strict")
        with self.assertRaisesRegex(ValueError, "canonical"):
            score_matched_forecasts(**{**arguments, "scope": replace(arguments["scope"], participants=arguments["scope"].participants[:-1])},
                                   horizon="primary_strict")
        coverage = arguments["observed_through_bins"].copy()
        coverage[-2:] = 0
        with self.assertRaisesRegex(UnresolvedScore, "every registered participant"):
            score_matched_forecasts(**{**arguments, "observed_through_bins": coverage}, horizon="primary_strict")
        rows = arguments["expected_row_keys"][:-2]
        with self.assertRaisesRegex(UnresolvedScore, "complete registered participant cohort"):
            score_matched_forecasts(**{**arguments, "expected_row_keys": rows}, horizon="primary_strict")
        way = fixture("WAY", "development", 1)
        rows = list(way["expected_row_keys"])
        identity, timestamp = rows[0]
        rows[0] = (replace(identity, series_or_run=8), timestamp)
        with self.assertRaisesRegex(ValueError, "scoring scope"):
            score_matched_forecasts(**{**way, "expected_row_keys": tuple(rows)}, horizon="primary_strict")

    def test_strongest_surrogate_is_person_and_horizon_specific_not_a_rowwise_oracle(self):
        arguments = fixture()
        values = arguments["condition_log_probabilities"]
        for name in arguments["scope"].surrogates:
            values[name][:] = probabilities(near=.45, strict=.45, longer=.05)
        for person in range(8):
            first, second = (("shift_1", "phase_1") if person < 7 else ("phase_1", "shift_1"))
            values[first][2 * person] = probabilities(near=.01, strict=.8, longer=.05)
            values[first][2 * person + 1] = probabilities(near=.01, strict=.001, longer=.05)
            values[second][2 * person:2 * person + 2] = probabilities(near=.4, strict=.05, longer=.05)
        near = score_matched_forecasts(**arguments, horizon="near_event")
        strict = score_matched_forecasts(**arguments, horizon="primary_strict")
        self.assertEqual(near["strongest_surrogate_names"], (("shift_1",),) * 7 + (("phase_1",),))
        self.assertEqual(strict["strongest_surrogate_names"], (("phase_1",),) * 7 + (("shift_1",),))
        np.testing.assert_allclose(strict["strongest_surrogate_log_scores"], np.log(.95), atol=1e-14)
        wrong_rowwise_score = (np.log(.95) + np.log(.999)) / 2
        self.assertGreater(abs(wrong_rowwise_score - strict["strongest_surrogate_log_scores"][0]), .01)
        expected_increment = np.log(.8) - np.log(.95)
        self.assertAlmostEqual(strict["macro_increments"]["real_minus_strongest_surrogate"], expected_increment)
        self.assertEqual(strict["horizon_ms"], (300, 600))


if __name__ == "__main__":
    unittest.main()
