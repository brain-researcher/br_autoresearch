from __future__ import annotations

import math
import unittest

import numpy as np

from ep12_executor.simultaneous_inference import (
    SimultaneousInferenceError,
    bilateral_with_aggregate,
    paired_studentized_max_t,
)


class SimultaneousInferenceTests(unittest.TestCase):
    def values(self) -> np.ndarray:
        return np.asarray(
            [
                [0.004, 0.020],
                [0.013, 0.008],
                [0.027, 0.015],
                [0.011, 0.031],
                [0.046, 0.018],
                [0.019, 0.043],
                [0.008, 0.012],
                [0.034, 0.026],
            ]
        )

    def test_bilateral_aggregate_and_direction_swap(self) -> None:
        values = self.values()
        augmented = bilateral_with_aggregate(values)
        self.assertTrue(np.allclose(augmented[:, 2], values.mean(axis=1)))
        swapped = paired_studentized_max_t(
            bilateral_with_aggregate(values[:, ::-1]),
            replicates=199,
            seed=41,
        )
        original = paired_studentized_max_t(
            augmented,
            replicates=199,
            seed=41,
        )
        self.assertTrue(
            np.allclose(swapped["estimate"], np.asarray(original["estimate"])[[1, 0, 2]])
        )
        self.assertEqual(swapped["critical_max_t"], original["critical_max_t"])

    def test_bootstrap_statistic_recomputes_resample_standard_error(self) -> None:
        values = bilateral_with_aggregate(self.values())
        result = paired_studentized_max_t(values, replicates=199, seed=7)
        estimates = values.mean(axis=0)
        rng = np.random.Generator(np.random.PCG64(7))
        indices = rng.integers(0, len(values), size=(199, len(values)))
        sampled = values[indices]
        means = sampled.mean(axis=1)
        resample_se = sampled.std(axis=1, ddof=1) / math.sqrt(len(values))
        roots = np.max(np.abs((means - estimates) / resample_se), axis=1)
        expected = float(np.quantile(roots, 0.95, method="higher"))
        self.assertEqual(result["critical_max_t"], expected)

    def test_replay_shift_scale_and_constant_coordinate(self) -> None:
        values = bilateral_with_aggregate(self.values())
        constant = np.column_stack((values, np.ones(len(values)) * 3.0))
        first = paired_studentized_max_t(constant, replicates=199, seed=99)
        second = paired_studentized_max_t(constant, replicates=199, seed=99)
        self.assertEqual(first, second)
        self.assertEqual(first["lower"][-1], 3.0)
        self.assertEqual(first["upper"][-1], 3.0)
        transformed = paired_studentized_max_t(
            constant * 2.5 + 4.0, replicates=199, seed=99
        )
        self.assertAlmostEqual(first["critical_max_t"], transformed["critical_max_t"], places=12)
        self.assertTrue(
            np.allclose(transformed["estimate"], np.asarray(first["estimate"]) * 2.5 + 4.0)
        )

    def test_invalid_shape_and_nonfinite_values_fail_closed(self) -> None:
        with self.assertRaises(SimultaneousInferenceError):
            bilateral_with_aggregate(np.ones((8, 1)))
        values = self.values()
        values[0, 0] = np.nan
        with self.assertRaises(SimultaneousInferenceError):
            paired_studentized_max_t(values, replicates=99, seed=1)


if __name__ == "__main__":
    unittest.main()
