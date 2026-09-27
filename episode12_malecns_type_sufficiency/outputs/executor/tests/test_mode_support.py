from __future__ import annotations

import unittest

import numpy as np

from ep12_executor.mode_support import (
    ModeSupportError,
    effective_component_memberships,
    match_component_contrasts,
    mixture_responsibilities,
    pairwise_symmetric_gaussian_kl,
)


class ModeSupportTests(unittest.TestCase):
    def test_responsibilities_and_effective_membership(self) -> None:
        residuals = np.asarray([[-2.0, 0.0], [-1.8, 0.1], [2.0, 0.0], [1.9, -0.1]])
        means = np.asarray([[-2.0, 0.0], [2.0, 0.0]])
        covariances = np.repeat((np.eye(2) * 0.2)[None, :, :], 2, axis=0)
        responsibilities = mixture_responsibilities(
            residuals, means, covariances, np.asarray([0.5, 0.5])
        )
        np.testing.assert_allclose(responsibilities.sum(axis=1), 1.0)
        effective = effective_component_memberships(responsibilities)
        self.assertTrue(np.all(effective > 1.9))

    def test_symmetric_kl_is_symmetric_and_positive(self) -> None:
        means = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
        covariances = np.repeat(np.eye(2)[None, :, :], 3, axis=0)
        values = pairwise_symmetric_gaussian_kl(means, covariances)
        np.testing.assert_allclose(values, values.T)
        np.testing.assert_allclose(np.diag(values), 0.0)
        self.assertTrue(np.all(values[np.triu_indices(3, 1)] > 0.0))

    def test_hungarian_match_is_label_invariant(self) -> None:
        left = np.asarray([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]])
        right = left[[1, 0]]
        result = match_component_contrasts(left, right)
        self.assertEqual(result["right_component_indices"], [1, 0])
        np.testing.assert_allclose(result["matched_cosines"], [1.0, 1.0])
        self.assertFalse(result["scientific_gate_passed"])
        self.assertTrue(result["point_estimate_only"])

    def test_invalid_covariance_fails_closed(self) -> None:
        with self.assertRaises(ModeSupportError):
            mixture_responsibilities(
                np.zeros((2, 2)),
                np.zeros((2, 2)),
                np.zeros((2, 2, 2)),
                np.asarray([0.5, 0.5]),
            )


if __name__ == "__main__":
    unittest.main()
