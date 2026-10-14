"""Changed-backend checks only; run in the coordinator's Slurm allocation."""
import unittest
from unittest.mock import patch
import numpy as np

import continuous_fold_runner as folds
from core_reference_direct_ridge import DirectCoreRidge
from test_core_reference_scalar_checkpoint import inputs
from test_core_reference_paired_inner import person_fixture
from test_production_core_reference import recipe
from test_continuous_tuning import dense_fit
import core_reference_paired_inner as paired


class DirectCoreTests(unittest.TestCase):
    def test_overlapping_sequence_solver_matches_augmented_least_squares_and_reuses_statistics(self):
        design, target, weights, penalties, limits = inputs()
        dense = design @ np.eye(design.shape[1])
        backend = DirectCoreRidge(check=lambda: None, chunk_rows=3)
        for multiplier in (1e-6, 1., 1e6):
            penalty = penalties * multiplier
            # Independent augmented least-squares oracle, not normal equations.
            augmented = np.vstack((np.sqrt(weights)[:, None] * dense, np.diag(np.sqrt(penalty))))
            rhs = np.vstack((np.sqrt(weights)[:, None] * target, np.zeros((len(penalty), 61))))
            expected = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
            result = backend.fit(design, target, weights, penalty, limits, lambda: None)
            np.testing.assert_allclose(dense @ result.coefficients, dense @ expected, rtol=2e-7, atol=2e-8)
            with backend.selected():
                np.testing.assert_allclose(design @ result.coefficients, dense @ result.coefficients,
                                           rtol=1e-12, atol=1e-12)
        self.assertEqual(backend.stats["solves"], 3)

    def test_full_registered_inner_and_outer_panel_matches_independent_dense_oracle(self):
        outcomes = []
        for direct in (False, True):
            backend = DirectCoreRidge(check=lambda: None, chunk_rows=13)
            context = backend.selected() if direct else patch.object(folds, "_fit", side_effect=dense_fit)
            inner, outer, progress = [], [], {}
            with context:
                result = paired.fit_person(person_fixture([]), recipe(), check=lambda: None,
                    retain_inner=inner.append, retain_outer=outer.append, progress=progress)
            outcomes.append((result, inner, outer))
        first, second = outcomes
        self.assertEqual((len(second[1]), len(second[2])), (30, 6))
        for old, new in zip(first[1], second[1]):
            self.assertEqual(old.split, new.split)
            np.testing.assert_allclose(old.normalized_image_losses, new.normalized_image_losses,
                                       rtol=1e-9, atol=1e-11)
        for old, new in zip(first[2], second[2]):
            self.assertEqual(old.result.image_index, new.result.image_index)
            for name in old.result.normalized_losses:
                self.assertAlmostEqual(old.result.normalized_losses[name], new.result.normalized_losses[name], places=10)


if __name__ == "__main__":
    unittest.main()
