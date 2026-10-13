"""Focused fitting arithmetic checks, not a synthetic recovery study.

These tests are written for coordinator-owned compute only. Role admission,
capacity matching, candidate search and empirical outcomes are outside them.
"""
import math
import unittest

import numpy as np

from regression import (
    RegressionNotConverged, RegressionRecipe, fit_regression, predict_regression,
)


def check():
    pass


class RegressionTests(unittest.TestCase):
    def test_weighted_ridge_matches_scalar_normal_equations_and_intercept(self):
        x = np.asarray([[-1.], [0.], [2.], [10.]])
        y = np.column_stack((3 * x[:, 0] + 5, -2 * x[:, 0] + 7))
        y[-1] = (900, -900)  # finite, explicitly zero-weighted row
        weights = np.asarray([1., 2., 1., 0.])
        recipe = RegressionRecipe("ridge", penalty=.5)
        fitted = fit_regression(x, y, row_weights=weights, recipe=recipe, check=check)
        normalized = [value / 4 for value in weights]
        mean_x = math.fsum(w * float(row[0]) for w, row in zip(normalized, x))
        variance = math.fsum(w * (float(row[0]) - mean_x)**2 for w, row in zip(normalized, x))
        means, coefficients = [], []
        for output in range(2):
            mean_y = math.fsum(w * float(row[output]) for w, row in zip(normalized, y))
            cross = math.fsum(w * (float(xi[0]) - mean_x) * (float(yi[output]) - mean_y)
                             for w, xi, yi in zip(normalized, x, y))
            means.append(mean_y)
            coefficients.append(cross / (variance + .5))
        np.testing.assert_allclose(fitted.coefficients, [coefficients], rtol=0, atol=1e-12)
        expected = np.asarray([[(row[0] - mean_x) * coefficient + mean
                                for coefficient, mean in zip(coefficients, means)] for row in x])
        np.testing.assert_allclose(predict_regression(fitted, x, check=check), expected, rtol=0, atol=1e-12)
        again = fit_regression(x, y, row_weights=1e200 * weights, recipe=recipe, check=check)
        np.testing.assert_array_equal(again.coefficients, fitted.coefficients)
        self.assertEqual(fitted.positive_weight_rows, 3)
        self.assertEqual(fitted.stopping_reason, "closed_form")
        with self.assertRaises(ValueError):
            fitted.coefficients.setflags(write=True)

    def test_elementwise_elastic_net_has_analytic_soft_threshold_and_refuses_nonconvergence(self):
        x = np.asarray([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
        slopes = np.asarray([[3., -.1], [-2., .7]])
        y = x @ slopes + np.asarray([5., 9.])
        recipe = RegressionRecipe("elastic_net", penalty=.4, l1_ratio=.5,
                                  max_iterations=20, tolerance=1e-12)
        fitted = fit_regression(x, y, row_weights=np.ones(4), recipe=recipe, check=check)
        # Weighted X'X is identity. Each output/coordinate independently has
        # soft-threshold .2 and ridge denominator1.2, not a grouped L1 penalty.
        expected = np.asarray([[2.8 / 1.2, 0.], [-1.8 / 1.2, .5 / 1.2]])
        np.testing.assert_allclose(fitted.coefficients, expected, rtol=0, atol=1e-12)
        prediction = predict_regression(fitted, x, check=check)
        sse = math.fsum((float(prediction[i, j]) - float(y[i, j]))**2
                       for i in range(4) for j in range(2)) / 8
        penalty = .2 * math.fsum(abs(float(v)) for v in expected.flat)
        penalty += .1 * math.fsum(float(v)**2 for v in expected.flat)
        self.assertAlmostEqual(fitted.objective, sse + penalty, places=12)
        self.assertTrue(fitted.converged)
        correlated = np.asarray([[1., 1.], [1., 2.], [-1., -1.], [-1., -2.]])
        short = RegressionRecipe("elastic_net", penalty=.1, l1_ratio=.5,
                                 max_iterations=1, tolerance=1e-12)
        with self.assertRaises(RegressionNotConverged) as failed:
            fit_regression(correlated, correlated.sum(axis=1)[:, None],
                           row_weights=np.ones(4), recipe=short, check=check)
        self.assertEqual(failed.exception.iterations, 1)
        self.assertGreater(failed.exception.kkt_error, short.tolerance)
        self.assertGreater(failed.exception.gram_kkt_error, short.tolerance)
        self.assertEqual(failed.exception.tolerance, short.tolerance)
        self.assertIn("iterations=1, residual_kkt_error=", str(failed.exception))
        self.assertIn("gram_kkt_error=", str(failed.exception))
        self.assertIn("tolerance=", str(failed.exception))

    def test_correlated_unequal_scale_elastic_net_matches_scalar_solution_and_residual_kkt(self):
        # Two highly correlated columns with different units exercise cyclic
        # convergence beyond the orthogonal one-sweep oracle above. These
        # tiny deterministic arrays do not simulate EP07 biological outcomes.
        u = np.asarray([-2., -1., 0., 1., 2., -2., -1., 3.])
        v = np.asarray([1., -1., 1., -1., 1., -1., 1., -1.])
        x = np.column_stack((100 * u, u + .1 * v))
        y = x @ np.asarray([[.01, -.02], [8., -6.]]) + np.asarray([5., -7.])
        weights = np.asarray([1., 2., 3., 4., 4., 3., 2., 1.])
        recipe = RegressionRecipe("elastic_net", penalty=.02, l1_ratio=.25,
                                  max_iterations=5000, tolerance=1e-9)
        fitted = fit_regression(x, y, row_weights=weights, recipe=recipe, check=check)
        w = [float(value) / 20 for value in weights]
        xm = [math.fsum(w[i] * float(x[i, j]) for i in range(8)) for j in range(2)]
        ym = [math.fsum(w[i] * float(y[i, j]) for i in range(8)) for j in range(2)]
        xx = [[float(x[i, j]) - xm[j] for j in range(2)] for i in range(8)]
        yy = [[float(y[i, j]) - ym[j] for j in range(2)] for i in range(8)]
        g00 = math.fsum(w[i] * xx[i][0]**2 for i in range(8))
        g01 = math.fsum(w[i] * xx[i][0] * xx[i][1] for i in range(8))
        g11 = math.fsum(w[i] * xx[i][1]**2 for i in range(8))
        l1, l2 = .005, .015
        a, d = g00 + l2, g11 + l2
        determinant = a * d - g01**2
        expected = np.empty((2, 2))
        for output, sign in enumerate((1., -1.)):
            c0 = math.fsum(w[i] * xx[i][0] * yy[i][output] for i in range(8)) - l1 * sign
            c1 = math.fsum(w[i] * xx[i][1] * yy[i][output] for i in range(8)) - l1 * sign
            expected[:, output] = ((d * c0 - g01 * c1) / determinant,
                                   (a * c1 - g01 * c0) / determinant)
            self.assertTrue(all(sign * value > 0 for value in expected[:, output]))
        np.testing.assert_allclose(fitted.coefficients, expected, rtol=0, atol=1e-7)
        # An independent fresh-residual scalar derivative tests the declared
        # absolute coefficient-space KKT, including unpenalized intercepts.
        errors = []
        for output in range(2):
            residual = [math.fsum(xx[i][j] * float(fitted.coefficients[j, output])
                                  for j in range(2)) - yy[i][output] for i in range(8)]
            self.assertAlmostEqual(math.fsum(w[i] * residual[i] for i in range(8)), 0., places=12)
            for feature in range(2):
                coefficient = float(fitted.coefficients[feature, output])
                derivative = math.fsum(w[i] * xx[i][feature] * residual[i] for i in range(8))
                errors.append(abs(derivative + l2 * coefficient + l1 * math.copysign(1., coefficient)))
        self.assertLessEqual(max(errors), recipe.tolerance + 1e-11)
        self.assertGreater(fitted.iterations, 1)
        self.assertTrue(fitted.converged)

    def test_reduced_rank_ridge_uses_the_penalized_augmented_output_subspace(self):
        x, y = np.diag([10., 1.]), np.diag([1., 2.])
        fitted = fit_regression(x, y, row_weights=np.ones(2),
            recipe=RegressionRecipe("reduced_rank_ridge", penalty=1., rank=1,
                                    fit_intercept=False), check=check)
        # Penalized explained energies are25/51 and2/3, selecting output2.
        # Truncating unaugmented fitted values would instead select output1:
        # those energies are1250/2601 versus2/9.
        self.assertGreater(2 / 3, 25 / 51)
        self.assertGreater(1250 / 2601, 2 / 9)
        expected = np.asarray([[0., 0.], [0., 2 / 3]])
        np.testing.assert_allclose(fitted.coefficients, expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(fitted.input_mean, np.zeros(2))
        np.testing.assert_array_equal(fitted.output_mean, np.zeros(2))
        self.assertAlmostEqual(fitted.objective, 11 / 12, places=12)
        full = fit_regression(x, y, row_weights=np.ones(2),
            recipe=RegressionRecipe("ridge", penalty=1., fit_intercept=False), check=check)
        np.testing.assert_allclose(full.coefficients, np.diag([5 / 51, 2 / 3]), rtol=0, atol=1e-12)

    def test_network_seed_locality_fixed_stop_and_independent_forward_arithmetic(self):
        import torch

        x = np.asarray([[-1., -.5], [-.5, .25], [0., 1.], [.5, .25], [1., -.5]])
        y = np.column_stack((x[:, 0]**2 + 2, x[:, 0] - x[:, 1]))
        weights = np.asarray([1., 2., 3., 2., 1.])
        torch_state = torch.random.get_rng_state().clone()
        numpy_state = np.random.get_state()
        for activation, intercept in (("tanh", True), ("relu", False)):
            recipe = RegressionRecipe("shallow_network", penalty=.01, fit_intercept=intercept,
                width=3, activation=activation, seed=91, learning_rate=.02, epochs=4)
            fitted = fit_regression(x, y, row_weights=weights, recipe=recipe, check=check)
            repeated = fit_regression(x, y, row_weights=weights, recipe=recipe, check=check)
            parameters = fitted.network
            np.testing.assert_array_equal(parameters.hidden_kernel, repeated.network.hidden_kernel)
            np.testing.assert_array_equal(parameters.output_kernel, repeated.network.output_kernel)
            self.assertEqual(fitted.iterations, 4)
            self.assertIsNone(fitted.converged)
            self.assertEqual(fitted.stopping_reason, "fixed_epochs_no_convergence_claim")
            expected = np.empty_like(y)
            for row in range(len(x)):
                hidden = []
                for unit in range(3):
                    value = math.fsum((float(x[row, feature]) - float(fitted.input_mean[feature]))
                                     * float(parameters.hidden_kernel[feature, unit]) for feature in range(2))
                    value += float(parameters.hidden_bias[unit])
                    hidden.append(math.tanh(value) if activation == "tanh" else max(value, 0))
                for output in range(2):
                    expected[row, output] = (math.fsum(hidden[unit] * float(parameters.output_kernel[unit, output])
                                                       for unit in range(3))
                        + float(parameters.output_bias[output]) + float(fitted.output_mean[output]))
            prediction = predict_regression(fitted, x, check=check)
            np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-12)
            objective = .5 * math.fsum(float(weights[row] / 9) * (float(expected[row, output]) - float(y[row, output]))**2
                                       for row in range(5) for output in range(2))
            objective += .005 * math.fsum(float(value)**2 for array in
                (parameters.hidden_kernel, parameters.output_kernel) for value in array.flat)
            self.assertAlmostEqual(fitted.objective, objective, places=11)
            if not intercept:
                np.testing.assert_array_equal(parameters.hidden_bias, np.zeros(3))
                np.testing.assert_array_equal(parameters.output_bias, np.zeros(2))
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        final_numpy_state = np.random.get_state()
        self.assertEqual(numpy_state[0], final_numpy_state[0])
        np.testing.assert_array_equal(numpy_state[1], final_numpy_state[1])
        self.assertEqual(numpy_state[2:], final_numpy_state[2:])

    def test_invalid_inputs_recipes_and_cancellation_return_no_fitted_candidate(self):
        x, y = np.eye(2), np.ones((2, 1))
        ridge = RegressionRecipe("ridge", penalty=1.)
        for weights in ([0., 0.], [-1., 2.], [1., np.inf], [1.]):
            with self.assertRaises(ValueError):
                fit_regression(x, y, row_weights=weights, recipe=ridge, check=check)
        with self.assertRaises(ValueError):
            fit_regression(x, np.asarray([[0.], [np.nan]]), row_weights=[1, 1], recipe=ridge, check=check)
        for width in (True, 65):
            with self.assertRaises(ValueError):
                fit_regression(x, y, row_weights=[1, 1], check=check,
                    recipe=RegressionRecipe("shallow_network", penalty=0, width=width,
                        activation="tanh", seed=1, learning_rate=.01, epochs=1))
        with self.assertRaises(ValueError):
            fit_regression(x, y, row_weights=[1, 1], check=check,
                           recipe=RegressionRecipe("ridge", penalty=1., rank=1))
        calls = []
        def cancel():
            calls.append(True)
            if len(calls) == 4:
                raise RuntimeError("fixture cancellation")
        with self.assertRaisesRegex(RuntimeError, "fixture cancellation"):
            fit_regression(x, y, row_weights=[1, 1], check=cancel,
                recipe=RegressionRecipe("elastic_net", penalty=.1, l1_ratio=.5,
                                        max_iterations=5, tolerance=1e-8))


if __name__ == "__main__":
    unittest.main()
