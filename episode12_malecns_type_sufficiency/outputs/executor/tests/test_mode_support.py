from __future__ import annotations

import unittest

import numpy as np

from ep12_executor.mode_support import (
    ModeSupportError,
    effective_component_memberships,
    match_component_contrasts,
    mixture_responsibilities,
    pairwise_symmetric_gaussian_kl,
    point_mode_diagnostics,
)
from ep12_executor.scientific_models import FittedPredictiveModel, helmert_ilr_basis


class ModeSupportTests(unittest.TestCase):
    @staticmethod
    def _singular_model(means: np.ndarray) -> FittedPredictiveModel:
        covariance = np.diag([0.2, 0.0])
        return FittedPredictiveModel(
            model_id="M2",
            basis=helmert_ilr_basis(3),
            nuisance_coefficients=np.zeros((1, 2)),
            latent_offsets=np.asarray(means, dtype=float),
            log_weights=np.log(np.asarray([0.5, 0.5])),
            prior_means=np.asarray(means, dtype=float),
            prior_covariances=np.repeat(covariance[None, :, :], 2, axis=0),
            prior_weights=np.asarray([0.5, 0.5]),
            integration_draws=2,
            integration_seed=1,
            parameter_count=1,
            diagnostics={},
            pseudocount=0.5,
        )

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

    def test_singular_shared_support_is_used_without_ambient_floor(self) -> None:
        model = self._singular_model(np.asarray([[-1.0, 0.0], [1.0, 0.0]]))
        residuals = np.asarray([[-1.1, 0.0], [-0.9, 0.0], [0.9, 0.0], [1.1, 0.0]])
        diagnostic = point_mode_diagnostics(
            model,
            residuals,
            fitted_component_parameter_rank=1,
        )
        self.assertTrue(diagnostic["diagnostic_supported"])
        self.assertEqual(
            diagnostic["covariance_support_geometry"][
                "common_covariance_support_rank"
            ],
            1,
        )
        self.assertEqual(
            diagnostic["covariance_support_geometry"]["ambient_dimension"], 2
        )
        self.assertEqual(
            diagnostic["covariance_support_geometry"][
                "mutually_singular_component_pairs"
            ],
            [],
        )

    def test_offsets_outside_shared_support_are_explicitly_mutually_singular(self) -> None:
        model = self._singular_model(np.asarray([[0.0, -1.0], [0.0, 1.0]]))
        values = pairwise_symmetric_gaussian_kl(
            model.prior_means, model.prior_covariances
        )
        self.assertTrue(np.isinf(values[0, 1]))
        diagnostic = point_mode_diagnostics(
            model,
            np.asarray([[0.0, -1.0], [0.0, 1.0]]),
            fitted_component_parameter_rank=1,
        )
        self.assertFalse(diagnostic["diagnostic_supported"])
        self.assertEqual(
            diagnostic["unsupported_reason"],
            "mutually_singular_component_affine_supports",
        )
        self.assertEqual(
            diagnostic["covariance_support_geometry"][
                "mutually_singular_component_pairs"
            ],
            [[0, 1]],
        )

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
