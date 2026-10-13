"""Small analytic optimizer checks, not simulation cohorts or empirical fits."""
from dataclasses import replace
from pathlib import Path
import math
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from grouped_binomial_map import (DEVELOPMENT_RECORDS, FitError, NewtonSettings,
                                  fit_grouped_binomial_map, held_mouse_plugin_logits)


TRAINING = DEVELOPMENT_RECORDS[1:]
SETTINGS = NewtonSettings(60, 1e-8, 40, 1e-4, 0.5)


def arguments(**changes):
    result = dict(design=np.ones((8, 1)), offsets=np.zeros(8),
                  successes=(1,) * 8, trials=(2,) * 8,
                  row_mouse_ids=TRAINING, training_mouse_ids=TRAINING, held_mouse_id=1,
                  penalty_precision=[[1.0]], prior_center=[0.0],
                  initial_coefficients=[2.0], settings=SETTINGS)
    result.update(changes)
    return result


class GroupedBinomialMapTests(unittest.TestCase):
    def test_known_balanced_optimum_and_extreme_offsets(self):
        fit = fit_grouped_binomial_map(**arguments())
        self.assertAlmostEqual(fit.coefficients[0], 0.0, places=8)
        self.assertEqual(fit.trial_count, 16)
        self.assertAlmostEqual(fit.negative_log_kernel_plus_penalty, 16 * math.log(2), places=11)
        # Zero design makes the exact optimum the supplied prior center, even
        # when offsets +/-1000 make direct exp(eta) unsafe.
        other = fit_grouped_binomial_map(**arguments(
            design=np.zeros((8, 2)), offsets=[1000, -1000] * 4,
            penalty_precision=[[2, 0.5], [0.5, 3]], prior_center=[2, -3],
            initial_coefficients=[-1, 8]))
        np.testing.assert_allclose(other.coefficients, [2, -3], atol=1e-8, rtol=0)
        self.assertAlmostEqual(other.negative_log_kernel_plus_penalty, 8000, places=8)

    def test_unequal_denominators_and_session_splitting_preserve_likelihood(self):
        # At beta=log(3), p=.75. Total n=44, k=28, so the likelihood
        # derivative is +5. Explicit center=log(3)+2.5 with P=2 cancels it.
        center = math.log(3) + 2.5
        supplied = arguments(successes=[7, 3, 3, 3, 3, 3, 3, 3],
                             trials=[16, 4, 4, 4, 4, 4, 4, 4],
                             prior_center=[center], penalty_precision=[[2]])
        fit = fit_grouped_binomial_map(**supplied)
        self.assertAlmostEqual(fit.coefficients[0], math.log(3), places=8)
        self.assertEqual(fit.trial_count, 44)
        split = dict(supplied, design=np.ones((9, 1)), offsets=np.zeros(9),
                     row_mouse_ids=(2, 2) + TRAINING[1:],
                     successes=[3, 4] + [3] * 7, trials=[8, 8] + [4] * 7)
        second = fit_grouped_binomial_map(**split)
        np.testing.assert_allclose(second.coefficients, fit.coefficients, atol=1e-8, rtol=0)
        self.assertAlmostEqual(second.negative_log_kernel_plus_penalty,
                               fit.negative_log_kernel_plus_penalty, places=10)

    def test_coupled_design_offset_and_penalty_have_known_stationary_solution(self):
        # At beta=(0,0), zero offsets and n=2 give residuals 1-k.
        # X' residual = (-2,0); P center = (-2,0) makes gradient zero.
        x = np.array([[1, -1], [1, 1]] * 4, dtype=float)
        p = np.array([[2, 1], [1, 2]], dtype=float)
        center = [-4 / 3, 2 / 3]
        supplied = arguments(design=x, successes=[2, 2, 1, 1, 1, 1, 1, 1],
                             penalty_precision=p, prior_center=center,
                             initial_coefficients=[2, -2])
        before = x.copy(), p.copy()
        fit = fit_grouped_binomial_map(**supplied)
        np.testing.assert_allclose(fit.coefficients, [0, 0], atol=1e-8, rtol=0)
        np.testing.assert_array_equal(x, before[0])
        np.testing.assert_array_equal(p, before[1])
        # Explicit offsets are used without fitting them: beta=0 remains
        # optimal for p=.75 and k=3,n=4 on every row, centered penalty at 0.
        offset_fit = fit_grouped_binomial_map(**arguments(
            offsets=[math.log(3)] * 8, successes=[3] * 8, trials=[4] * 8))
        self.assertAlmostEqual(offset_fit.coefficients[0], 0.0, places=8)

    def test_fold_integrity_counts_and_positive_definite_penalty_are_required(self):
        bad = [dict(row_mouse_ids=(1,) + TRAINING[1:]),
               dict(training_mouse_ids=TRAINING[:-1] + (TRAINING[0],)),
               dict(held_mouse_id=3), dict(successes=[True] * 8),
               dict(trials=[0] * 8), dict(trials=[2.5] * 8),
               dict(successes=[3] * 8), dict(trials=[2**53 + 1] * 8),
               dict(penalty_precision=[[0]]), dict(penalty_precision=[[-1]]),
               dict(offsets=[math.nan] * 8)]
        for change in bad:
            with self.subTest(change=change), self.assertRaises(FitError):
                fit_grouped_binomial_map(**arguments(**change))
        with self.assertRaises(FitError):
            fit_grouped_binomial_map(**arguments(
                design=np.ones((8, 2)), initial_coefficients=[0, 0], prior_center=[0, 0],
                penalty_precision=[[2, 1], [0, 2]]))

    def test_permutation_and_fixed_new_mouse_prediction(self):
        supplied = arguments(successes=[2, 1, 1, 1, 0, 1, 1, 1])
        fit = fit_grouped_binomial_map(**supplied)
        permuted = dict(supplied, row_mouse_ids=TRAINING[::-1],
                        training_mouse_ids=TRAINING[::-1], successes=supplied['successes'][::-1])
        again = fit_grouped_binomial_map(**permuted)
        np.testing.assert_allclose(again.coefficients, fit.coefficients, atol=1e-8, rtol=0)
        self.assertEqual(held_mouse_plugin_logits(fit, held_mouse_id=1,
                                                 design=[[0], [2]], offsets=[3, -1]),
                         (3.0, -1 + 2 * fit.coefficients[0]))
        with self.assertRaises(FitError):
            held_mouse_plugin_logits(fit, held_mouse_id=2, design=[[1]], offsets=[0])
        self.assertEqual(fit.training_mouse_ids, TRAINING)

    def test_iteration_exhaustion_is_not_a_successful_fit(self):
        with self.assertRaisesRegex(FitError, "maximum iterations"):
            fit_grouped_binomial_map(**arguments(
                initial_coefficients=[5], settings=replace(SETTINGS, max_iterations=1)))


if __name__ == "__main__":
    unittest.main()
