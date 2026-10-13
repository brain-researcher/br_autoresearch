"""Known-answer algebra checks, not fitted-model or scientific qualification."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import spatial_deformation_operators as operators


# Fixture roundoff tolerance only; no EP15 scientific/operator acceptance
# tolerance is selected or written by this test module.
ROUNDING = 5e-13


class SpatialDeformationOperatorTests(unittest.TestCase):
    def setUp(self):
        self.B = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])/np.sqrt(2.)
        self.O = np.array([[0., -1.], [1., 0.]])
        self.C = np.diag([np.log(2.), -np.log(2.)])
        self.X = np.array([[2., -1., 4., -5.], [-3., 3., 1., -1.], [4., 4., 4., 4.]])

    def test_m3_preserves_gram_constants_complement_and_inverse(self):
        output = operators.apply_m3(self.X, self.B, self.O)
        np.testing.assert_allclose(output@output.T, self.X@self.X.T, atol=ROUNDING, rtol=0)
        np.testing.assert_allclose(output.mean(axis=1), self.X.mean(axis=1), atol=ROUNDING, rtol=0)
        np.testing.assert_allclose(operators.apply_m3(self.B.T, self.B, self.O),
                                   np.vstack((-self.B[:, 1], self.B[:, 0])), atol=ROUNDING, rtol=0)
        complement = np.array([[1., 1., -1., -1.], [1., 1., 1., 1.]])
        np.testing.assert_array_equal(operators.apply_m3(complement, self.B, self.O), complement)
        np.testing.assert_allclose(operators.inverse_m3(output, self.B, self.O), self.X, atol=ROUNDING, rtol=0)
        reflection = np.diag([-1., 1.])
        np.testing.assert_allclose(operators.apply_m3(self.B.T, self.B, reflection),
                                   np.vstack((-self.B[:, 0], self.B[:, 1])), atol=ROUNDING, rtol=0)
        residuals = operators.raw_residuals(self.B, self.O)
        self.assertLess(residuals["Q_transpose_Q_minus_I_max_abs"], ROUNDING)
        self.assertLess(residuals["Q_Q_transpose_minus_I_max_abs"], ROUNDING)
        self.assertLess(residuals["Q_constant_minus_constant_max_abs"], ROUNDING)

    def test_noncommuting_m4_has_registered_order_and_known_geometry(self):
        # Orthogonal quarter-turn followed by stretch2 and compression1/2.
        expected_modes = np.vstack((-.5*self.B[:, 1], 2*self.B[:, 0]))
        actual = operators.apply_m4(self.B.T, self.B, self.O, self.C)
        np.testing.assert_allclose(actual, expected_modes, atol=ROUNDING, rtol=0)
        np.testing.assert_allclose(actual@actual.T, np.diag([.25, 4.]), atol=ROUNDING, rtol=0)
        wrong_order = np.vstack((-2*self.B[:, 1], .5*self.B[:, 0]))
        self.assertGreater(np.linalg.norm(actual-wrong_order), 1.)
        identity = np.eye(4)
        Q = identity+self.B@(self.O-np.eye(2))@self.B.T
        E = identity+self.B@np.diag([1., -.5])@self.B.T
        np.testing.assert_allclose(operators.apply_m4(self.X, self.B, self.O, self.C),
                                   self.X@(Q@E), atol=ROUNDING, rtol=0)
        np.testing.assert_allclose(operators.apply_m4(np.ones((1, 4)), self.B, self.O, self.C),
                                   np.ones((1, 4)), atol=ROUNDING, rtol=0)

    def test_m4_inverse_is_not_its_transpose(self):
        transformed = operators.apply_m4(self.X, self.B, self.O, self.C)
        L = np.eye(4)+self.B@(self.O@np.diag([2., .5])-np.eye(2))@self.B.T
        inverse = operators.inverse_m4(transformed, self.B, self.O, self.C)
        np.testing.assert_allclose(inverse, self.X, atol=ROUNDING, rtol=0)
        np.testing.assert_allclose(inverse, np.linalg.solve(L.T, transformed.T).T, atol=ROUNDING, rtol=0)
        self.assertGreater(np.linalg.norm(transformed@L.T-self.X), 1.)

    def test_zero_stretch_reuses_m3_exactly_and_empty_subspace_is_identity(self):
        expected = operators.apply_m3(self.X, self.B, self.O)
        with patch.object(operators, "apply_m3", wraps=operators.apply_m3) as shared, \
                patch.object(operators, "expm", side_effect=AssertionError("zero stretch needs no exponential")):
            actual = operators.apply_m4(self.X, self.B, self.O, np.zeros((2, 2)))
        shared.assert_called_once()
        np.testing.assert_array_equal(actual, expected)
        empty_basis, empty_active = np.empty((4, 0)), np.empty((0, 0))
        np.testing.assert_array_equal(operators.apply_m4(self.X, empty_basis, empty_active, empty_active), self.X)
        np.testing.assert_array_equal(operators.inverse_m4(self.X, empty_basis, empty_active, empty_active), self.X)
        residuals = operators.raw_residuals(empty_basis, empty_active, empty_active)
        self.assertEqual(residuals["Q_transpose_Q_minus_I_max_abs"], 0.)
        self.assertEqual(residuals["low_rank_vs_defining_L_max_abs"], 0.)

    def test_unqualified_inputs_are_diagnosed_without_silent_repairs(self):
        bad_basis = self.B.copy()
        bad_basis[:, 0] *= 2
        bad_O = np.diag([1., 2.])
        bad_C = np.array([[.1, .2], [0., 0.]])
        originals = [array.copy() for array in (bad_basis, bad_O, bad_C, self.X)]
        residuals = operators.raw_residuals(bad_basis, bad_O, bad_C)
        self.assertAlmostEqual(residuals["basis_orthonormality_max_abs"], 3.)
        self.assertEqual(residuals["O_transpose_O_minus_I_max_abs"], 3.)
        self.assertEqual(residuals["C_symmetry_max_abs"], .2)
        self.assertEqual(residuals["C_trace"], .1)
        self.assertGreater(residuals["low_rank_vs_defining_L_max_abs"], .01)
        # Diagnostic-only failures do not mutate or re-fit supplied parameters.
        for current, original in zip((bad_basis, bad_O, bad_C, self.X), originals):
            np.testing.assert_array_equal(current, original)
        valid_before = [array.copy() for array in (self.X, self.B, self.O, self.C)]
        operators.apply_m4(self.X, self.B, self.O, self.C)
        for current, original in zip((self.X, self.B, self.O, self.C), valid_before):
            np.testing.assert_array_equal(current, original)

    def test_nonfinite_nonreal_and_misaligned_axes_refuse(self):
        for operation in (operators.apply_m4, operators.inverse_m4):
            with self.assertRaises(ValueError):
                operation(self.X, self.B, self.O, None)
        cases = [
            (self.X[:, :3], self.B, self.O, self.C),
            (self.X, self.B, np.eye(3), self.C),
            (self.X, self.B, self.O, np.eye(1)),
            (self.X.astype(complex)+1j, self.B, self.O, self.C),
            (self.X, np.full_like(self.B, np.nan), self.O, self.C),
            (self.X, self.B, self.O, np.full_like(self.C, np.inf)),
        ]
        for values in cases:
            with self.subTest(shapes=[value.shape for value in values]), self.assertRaises(ValueError):
                operators.apply_m4(*values)


if __name__ == "__main__":
    unittest.main()
