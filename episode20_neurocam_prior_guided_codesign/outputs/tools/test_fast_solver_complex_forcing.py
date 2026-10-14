"""Regression for complex particular-solution forcing; run in Slurm only."""
import os
import socket
import unittest

import numpy as np

from fast_solver_engine import UpstreamModel


class ComplexForcingTests(unittest.TestCase):
    def setUp(self):
        # Only the coordinate transform is under test; no circuit assembly.
        self.model = UpstreamModel.__new__(UpstreamModel)
        self.model.P, self.model.S, self.model.K = 2, 2, 2

    def test_pure_imaginary_contact_current_reaches_branch_and_pad(self):
        current = np.zeros((8, 1), dtype=complex)
        current[5, 0] = 3j  # pad 0, contact branch 1
        pads, sources = self.model.currents_to_relative(current)
        expected = np.zeros((2, 3, 1), dtype=complex)
        expected[0, 0, 0] = expected[0, 2, 0] = 3j
        np.testing.assert_array_equal(pads, expected)
        np.testing.assert_array_equal(sources, np.zeros((2, 1)))
        self.assertEqual(pads.dtype, np.dtype(np.complex128))

    def test_pure_imaginary_pad_current(self):
        current = np.zeros((8, 1), dtype=complex)
        current[1, 0] = -2j
        pads, sources = self.model.currents_to_relative(current)
        expected = np.zeros((2, 3, 1), dtype=complex)
        expected[1, 0, 0] = -2j
        np.testing.assert_array_equal(pads, expected)
        np.testing.assert_array_equal(sources, np.zeros((2, 1)))

    def test_pure_imaginary_source_current(self):
        current = np.zeros((8, 1), dtype=complex)
        current[3, 0] = 7j
        pads, sources = self.model.currents_to_relative(current)
        np.testing.assert_array_equal(pads, np.zeros((2, 3, 1)))
        np.testing.assert_array_equal(sources, np.array([[0j], [7j]]))

    def test_mixed_complex_forcing_with_trailing_dimensions_is_power_dual(self):
        # x_original = transform @ x_relative, hence F_relative = transform.T @ F_original.
        transform = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0],
        ], dtype=float)
        values = np.arange(48).reshape(8, 2, 3)
        current = (values - 17) + 1j * (31 - 2 * values)
        expected = (transform.T @ current.reshape(8, -1)).reshape(8, 2, 3)
        pads, sources = self.model.currents_to_relative(current)
        np.testing.assert_array_equal(pads, expected[:6].reshape(2, 3, 2, 3))
        np.testing.assert_array_equal(sources, expected[6:])

    def test_real_transform_preserves_legacy_float64_output_exactly(self):
        for dtype in (np.float32, np.float64, np.longdouble):
            with self.subTest(dtype=dtype):
                current = np.array([
                    [1e-12, -0.3], [0.7, -2e-8], [1.5, 0.0], [-4.0, 0.5],
                    [1e-7, -2.5], [-3e-6, 0.125], [0.45, -0.75], [-1e-8, 2.0],
                ], dtype=dtype)
                # Retain the previous operation order and intermediate dtype.
                legacy = np.empty((2, 3, 2))
                branches = current[4:].reshape(2, 2, 2)
                legacy[:, 1:] = branches
                legacy[:, 0] = current[:2] + branches.sum(axis=1)
                pads, sources = self.model.currents_to_relative(current)
                self.assertEqual(pads.dtype, np.dtype(np.float64))
                self.assertEqual(pads.tobytes(), legacy.tobytes())
                self.assertEqual(sources.dtype, current.dtype)
                self.assertEqual(sources.tobytes(), current[2:4].tobytes())


if __name__ == "__main__":
    if "-ln" in socket.gethostname() or not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("compute allocation required; root owns the test session")
    unittest.main()
