"""Small known-answer tests; no source files, model fitting, or experiment runner."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from development_score_core import (DEVELOPMENT_RECORDS, DevelopmentScore, MouseFold,
                                    ScoreError, score_loso_panel, selection_key,
                                    session_binomial_logpmf)


SEEDS = (13, 29, 47, 71, 101)


def fold(mouse, *, successes=(1,), trials=(2,), candidate=None, null=None):
    columns = len(trials)
    return MouseFold(mouse, tuple(other for other in DEVELOPMENT_RECORDS if other != mouse),
                     tuple(f"verified_session_{index}" for index in range(columns)),
                     successes, trials, SEEDS,
                     candidate if candidate is not None else ((0.0,) * columns,) * 5,
                     null if null is not None else ((0.0,) * columns,) * 5)


def panel(**kwargs):
    return tuple(fold(mouse, **kwargs) for mouse in DEVELOPMENT_RECORDS)


def score(folds):
    return score_loso_panel(folds, frozen_complexity=3.0, configuration_id="fixed_configuration")


class DevelopmentScoreTests(unittest.TestCase):
    def test_session_binomial_includes_known_combination_term(self):
        self.assertAlmostEqual(session_binomial_logpmf(1, 4, 0), math.log(4 / 16), places=13)
        expected = math.log(6 * (0.75 ** 2) * (0.25 ** 2))
        self.assertAlmostEqual(session_binomial_logpmf(2, 4, math.log(3)), expected, places=13)

    def test_exact_boundary_predictions_and_empty_elementary_distribution(self):
        self.assertEqual(session_binomial_logpmf(4, 4, math.inf), 0.0)
        self.assertEqual(session_binomial_logpmf(0, 4, -math.inf), 0.0)
        self.assertEqual(session_binomial_logpmf(3, 4, math.inf), -math.inf)
        self.assertEqual(session_binomial_logpmf(1, 4, -math.inf), -math.inf)
        for eta in (-math.inf, 0.0, math.inf):
            self.assertEqual(session_binomial_logpmf(0, 0, eta), 0.0)

    def test_large_finite_logits_do_not_clip_or_overflow_exp(self):
        self.assertEqual(session_binomial_logpmf(0, 10, 1000), -10000.0)
        self.assertEqual(session_binomial_logpmf(10, 10, -1000), -10000.0)
        self.assertEqual(session_binomial_logpmf(10, 10, 1000), 0.0)

    def test_unequal_session_trial_denominators_use_total_trial_weight(self):
        # Candidate differs only in the ten-trial session; likelihood constants
        # cancel in the paired gain. Averaging two per-session scores is wrong.
        candidate = ((0.0, math.log(3)),) * 5
        result = score(panel(successes=(1, 9), trials=(2, 10), candidate=candidate))
        second_gain = 9 * math.log(0.75 / 0.5) + math.log(0.25 / 0.5)
        expected = second_gain / 12
        self.assertAlmostEqual(result.mean_mouse_gain, expected, places=13)
        self.assertAlmostEqual(result.minimum_mouse_gain, expected, places=13)
        self.assertNotAlmostEqual(result.mean_mouse_gain, second_gain / 20, places=8)

    def test_five_seed_log_scores_are_averaged_not_probability_mixed(self):
        etas = (-2.0, -1.0, 0.0, 1.0, 2.0)
        candidate = tuple((eta,) for eta in etas)
        result = score(panel(successes=(1,), trials=(1,), candidate=candidate))
        expected = sum(math.log(1 / (1 + math.exp(-eta))) for eta in etas) / 5 + math.log(2)
        self.assertAlmostEqual(result.mean_mouse_gain, expected, places=13)
        self.assertLess(result.mean_mouse_gain, -0.1)
        # Symmetric logits instead mixed as probabilities would give zero gain.
        self.assertAlmostEqual(sum(1 / (1 + math.exp(-eta)) for eta in etas) / 5, 0.5)

    def test_mouse_aggregation_is_equal_weight_with_worst_and_median_coordinates(self):
        folds, expected = [], []
        for index, mouse in enumerate(DEVELOPMENT_RECORDS, 1):
            probability = 0.1 * index
            eta = math.log(probability / (1 - probability))
            folds.append(fold(mouse, successes=(index,), trials=(index,), candidate=((eta,),) * 5))
            expected.append(math.log(probability / 0.5))
        result = score(tuple(folds))
        self.assertAlmostEqual(result.minimum_mouse_gain, min(expected), places=13)
        self.assertAlmostEqual(result.median_mouse_gain, sorted(expected)[4], places=13)
        self.assertAlmostEqual(result.mean_mouse_gain, sum(expected) / 9, places=13)
        self.assertNotAlmostEqual(result.mean_mouse_gain, sum((i + 1) * x for i, x in enumerate(expected)) / 45)

    def test_fixed_fold_and_seed_order_preserve_results_without_mutating_inputs(self):
        original = panel(successes=(1,), trials=(1,), candidate=tuple((float(i - 2),) for i in range(5)))
        permutation = (4, 1, 3, 0, 2)
        reordered = tuple(replace(item,
                                  seed_ids=tuple(item.seed_ids[i] for i in permutation),
                                  candidate_logits=tuple(item.candidate_logits[i] for i in permutation),
                                  constant_null_logits=tuple(item.constant_null_logits[i] for i in permutation))
                          for item in reversed(original))
        self.assertEqual(score(original), score(reordered))
        self.assertEqual(original[0].seed_ids, SEEDS)
        self.assertEqual(original[0].candidate_logits, ((-2.0,), (-1.0,), (0.0,), (1.0,), (2.0,)))

    def test_complete_nine_mouse_identity_and_declared_training_lineage(self):
        good = panel()
        bad_panels = (good[:-1], good[:-1] + (good[0],),
                      (replace(good[0], held_mouse_id=3),) + good[1:],
                      (replace(good[0], training_mouse_ids=DEVELOPMENT_RECORDS[:8]),) + good[1:])
        for bad in bad_panels:
            with self.subTest(bad=bad[0].held_mouse_id, length=len(bad)), self.assertRaises(ScoreError):
                score(bad)

    def test_exact_five_seed_bank_and_complete_prediction_matrices(self):
        good = panel()
        bad_folds = (replace(good[0], seed_ids=SEEDS[:4]),
                     replace(good[0], seed_ids=(13, 13, 47, 71, 101)),
                     replace(good[0], seed_ids=(13, 29, 47, 71, 102)),
                     replace(good[0], candidate_logits=((0.0,),) * 4),
                     replace(good[0], constant_null_logits=((0.0, 0.0),) * 5))
        for bad in bad_folds:
            with self.subTest(bad=bad), self.assertRaises(ScoreError):
                score((bad,) + good[1:])

    def test_invalid_counts_sessions_and_nan_are_not_dropped(self):
        good = panel()
        bad_folds = (replace(good[0], successes=(3,)), replace(good[0], trials=(0,)),
                     replace(good[0], successes=(True,)), replace(good[0], successes=(0.5,)),
                     replace(good[0], session_ids=()),
                     replace(good[0], session_ids=("same", "same"), successes=(0, 0), trials=(1, 1)),
                     replace(good[0], candidate_logits=((math.nan,),) * 5))
        for bad in bad_folds:
            with self.subTest(bad=bad), self.assertRaises(ScoreError):
                score((bad,) + good[1:])
        with self.assertRaises(ScoreError):
            session_binomial_logpmf(0, 0, math.nan)

    def test_one_nonfinite_fold_refuses_whole_panel_but_valid_boundary_scores_pass(self):
        good = panel(successes=(1,), trials=(1,), candidate=((math.inf,),) * 5)
        self.assertAlmostEqual(score(good).mean_mouse_gain, math.log(2), places=13)
        bad = (replace(good[0], candidate_logits=((-math.inf,),) * 5),) + good[1:]
        with self.assertRaises(ScoreError):
            score(bad)
        bad_null = (replace(good[0], constant_null_logits=((-math.inf,),) * 5),) + good[1:]
        with self.assertRaises(ScoreError):
            score(bad_null)

    def test_exact_objective_order_precedes_complexity_and_configuration_id(self):
        reference = DevelopmentScore(0.1, 0.2, 0.3, 2, "z")
        better_worst = replace(reference, minimum_mouse_gain=math.nextafter(0.1, math.inf), frozen_complexity=99)
        better_median = replace(reference, median_mouse_gain=0.21, mean_mouse_gain=0.22, frozen_complexity=99)
        better_mean = replace(reference, mean_mouse_gain=0.31, frozen_complexity=99)
        simpler = replace(reference, frozen_complexity=1)
        lower_id = replace(reference, configuration_id="a")
        for better in (better_worst, better_median, better_mean, simpler, lower_id):
            self.assertLess(selection_key(better), selection_key(reference))
        self.assertLess(selection_key(better_worst), selection_key(better_median))
        self.assertLess(selection_key(better_median), selection_key(better_mean))
        self.assertLess(selection_key(better_mean), selection_key(simpler))
        self.assertLess(selection_key(simpler), selection_key(lower_id))
        with self.assertRaises(ScoreError):
            selection_key(replace(reference, mean_mouse_gain=math.inf))


if __name__ == "__main__":
    unittest.main()
