#!/usr/bin/env python3
"""Direction, conditioning and C-ABI regressions for the existing solver."""

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import fit_v3_m1 as m1
import v3_m1_triangular_solve as triangular


def original_factored_solve(factor, rhs):
    return np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))


class TriangularSolveTests(unittest.TestCase):
    def test_known_solution_and_unchanged_inputs_for_strided_buffers(self):
        lower = np.array([[2., 0., 0.], [3., 4., 0.], [-1., 2., 5.]])
        expected = np.array([0.25, -2., 3.])
        rhs = (lower @ lower.T) @ expected
        for layout in ("C", "F", "sliced"):
            with self.subTest(layout=layout):
                matrix = np.array(lower, order="F" if layout == "F" else "C")
                vector = rhs.copy()
                if layout == "sliced":
                    storage = np.zeros((6, 6)); storage[::2, ::2] = lower
                    matrix = storage[::2, ::2]
                    vector_storage = np.zeros(6); vector_storage[::2] = rhs
                    vector = vector_storage[::2]
                matrix_before, vector_before = matrix.copy(), vector.copy()
                result = triangular.solve_cholesky(matrix, vector)
                np.testing.assert_allclose(result, expected, rtol=0., atol=2e-15)
                np.testing.assert_array_equal(matrix, matrix_before)
                np.testing.assert_array_equal(vector, vector_before)

    def test_only_lower_triangle_is_used(self):
        lower = np.array([[2., 0.], [1., 3.]])
        rhs = np.array([5., -2.])
        expected = np.linalg.solve(lower @ lower.T, rhs)
        lower[0, 1] = 1000.
        np.testing.assert_allclose(triangular.solve_cholesky(lower, rhs), expected,
                                   rtol=1e-15, atol=1e-15)

    def test_rejects_shape_nonfinite_and_singular_factor(self):
        with self.assertRaises(ValueError):
            triangular.solve_cholesky(np.eye(2), np.ones((2, 1)))
        with self.assertRaises(ValueError):
            triangular.solve_cholesky(np.zeros((2, 3)), np.ones(2))
        with self.assertRaises(np.linalg.LinAlgError):
            triangular.solve_cholesky(np.diag([1., 0.]), np.ones(2))
        with self.assertRaises(np.linalg.LinAlgError):
            triangular.solve_cholesky(np.eye(2), np.array([1., np.nan]))

    def test_directions_match_old_solve_and_independent_system_solution(self):
        rng = np.random.default_rng(731105)
        q, _ = np.linalg.qr(rng.standard_normal((m1.PARAMETER_DIMENSION,) * 2))
        planted = np.sin(np.arange(m1.PARAMETER_DIMENSION) + 0.4)
        # Cover strong and weak ridge conditioning without weakening any
        # production gate; forward tolerance scales with this test's condition.
        for condition in (1., 1e4, 1e8):
            with self.subTest(condition=condition):
                eigenvalues = np.geomspace(1. / condition, 1., m1.PARAMETER_DIMENSION)
                hessian = (q * eigenvalues) @ q.T
                hessian = (hessian + hessian.T) * 0.5
                gradient = -(hessian @ planted)
                ridge = float(eigenvalues[0])
                repaired, telemetry = m1._newton_direction(hessian, gradient, ridge)
                with mock.patch.object(m1, "_cholesky_factored_solve", original_factored_solve):
                    original, original_telemetry = m1._newton_direction(hessian, gradient, ridge)
                independent = np.linalg.solve(hessian, -gradient)
                tolerance = max(2e-13, 20 * np.finfo(float).eps * condition)
                for reference in (planted, original, independent):
                    np.testing.assert_allclose(repaired, reference, rtol=0., atol=tolerance)
                self.assertLessEqual(telemetry["scaled_backward_error"], 5e-11)
                self.assertLess(telemetry["directional_derivative"], 0.)
                self.assertGreater(telemetry["quadratic_predicted_decrease"], 0.)
                self.assertEqual(telemetry["diagonal_shift"], original_telemetry["diagonal_shift"])
                repeated, repeated_telemetry = m1._newton_direction(hessian, gradient, ridge)
                np.testing.assert_array_equal(repaired, repeated)
                self.assertEqual(telemetry, repeated_telemetry)

    def test_exactly_two_refinements_and_no_general_lu_solve(self):
        hessian = np.diag(np.linspace(1., 2., m1.PARAMETER_DIMENSION))
        gradient = np.ones(m1.PARAMETER_DIMENSION)
        with mock.patch.object(m1, "_cholesky_factored_solve", wraps=triangular.solve_cholesky) as solve:
            with mock.patch.object(np.linalg, "solve", side_effect=AssertionError("general LU used")):
                m1._newton_direction(hessian, gradient, 1.)
        self.assertEqual(solve.call_count, 3)  # direction, refinement1, refinement2

    def test_existing_newton_failure_gates_remain(self):
        hessian = np.eye(m1.PARAMETER_DIMENSION)
        gradient = np.ones(m1.PARAMETER_DIMENSION)
        asymmetric = hessian.copy(); asymmetric[0, 1] = 1e-3
        with self.assertRaises(m1.M1Error):
            m1._newton_direction(asymmetric, gradient, 1.)
        with self.assertRaises(m1.M1Error):
            m1._newton_direction(-hessian, gradient, 1.)
        with self.assertRaises(m1.M1Error):
            m1._newton_direction(hessian, np.full_like(gradient, np.nan), 1.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
