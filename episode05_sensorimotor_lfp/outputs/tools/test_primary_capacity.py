"""Independent capacity equations, held-direction boundary and exact constants."""
import copy
import unittest

import numpy as np

from primary_capacity import (
    LANDMARK_JITTER, RECIPES, coordinates, fit_banks, fit_nystrom, landmark_rows,
    prepare_split, rank_one_coefficients, raw_nystrom, rbf_kernel, summarize_banks,
)
from primary_reference import LAMBDAS, SESSIONS


class CapacityTest(unittest.TestCase):
    def test_rank_one_matches_augmented_design_best_rank_one_fit(self):
        rng = np.random.default_rng(50501)
        z, y, penalty = rng.normal(size=(29, 5)), rng.normal(size=(29, 2)), .1
        augmented_x = np.vstack((z, np.sqrt(len(z)*penalty)*np.eye(5)))
        augmented_y = np.vstack((y, np.zeros((5, 2))))
        unconstrained = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
        u, s, vt = np.linalg.svd(augmented_x@unconstrained, full_matrices=False)
        best_fit = s[0]*np.outer(u[:, 0], vt[0])
        expected = np.linalg.lstsq(augmented_x, best_fit, rcond=None)[0]
        actual = rank_one_coefficients(z, y, penalty)
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-13)
        self.assertLessEqual(np.linalg.matrix_rank(actual), 1)
        np.testing.assert_array_equal(rank_one_coefficients(z, np.zeros_like(y), penalty), np.zeros((5, 2)))

    def test_nystrom_orientation_matches_regularized_kernel_with_duplicate_landmarks(self):
        z = np.array([[0., 1.], [0., 1.], [2., -1.], [1., 3.]])
        rows = np.array([0, 1, 2])
        phi, state = fit_nystrom(z, rows, 2)
        raw = raw_nystrom(z, state)
        cross = rbf_kernel(z, z[rows], 2)
        inner = rbf_kernel(z[rows], z[rows], 2)+LANDMARK_JITTER*np.eye(3)
        expected = cross@np.linalg.solve(inner, cross.T)
        np.testing.assert_allclose(raw@raw.T, expected, atol=2e-8, rtol=2e-8)
        np.testing.assert_allclose(phi.mean(axis=0), 0, atol=1e-15)
        np.testing.assert_array_equal(state.landmarks[0], state.landmarks[1])

    def test_held_direction_cannot_change_landmarks_bandwidth_or_centering(self):
        rng = np.random.default_rng(50502)
        labels = np.repeat(np.arange(8), 2)
        ids = np.array([f"trial{i:04d}" for i in range(len(labels))])
        x, y = rng.normal(size=(16, 15, 4)), rng.normal(size=(16, 15, 2))
        first = prepare_split(x, y, labels, ids, "synthetic", 3)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[labels == 3], changed_y[labels == 3] = 1e8, -1e8
        other = prepare_split(changed_x, changed_y, labels, ids, "synthetic", 3)
        for key in ("feature_mean", "feature_sd", "z", "residual", "phi"):
            np.testing.assert_array_equal(first[key], other[key])
        self.assertEqual(first["landmark_trial_ids"], other["landmark_trial_ids"])
        self.assertEqual(len(first["landmark_trial_ids"]), 7)
        self.assertFalse(set(first["landmark_trial_ids"]) & set(ids[labels == 3]))
        for field in ("landmarks", "feature_mean", "cholesky"):
            np.testing.assert_array_equal(getattr(first["kernel_state"], field), getattr(other["kernel_state"], field))
        self.assertEqual(first["kernel_state"].sigma_squared, other["kernel_state"].sigma_squared)

    def test_training_constant_coordinate_cannot_change_evaluation_RBF_distance(self):
        fitting = np.array([[1., 7.], [2., 7.], [3., 7.]])
        mean, sd = fitting.mean(axis=0), fitting.std(axis=0)
        z = coordinates(fitting, mean, sd, rbf=True)
        _, state = fit_nystrom(z, np.array([0, 1, 2]), 1)
        first = coordinates(np.array([[2., 7.]]), mean, sd, rbf=True)
        changed = coordinates(np.array([[2., 1e9]]), mean, sd, rbf=True)
        np.testing.assert_array_equal(raw_nystrom(first, state), raw_nystrom(changed, state))
        self.assertNotEqual(coordinates(np.array([[2., 1e9]]), mean, sd, rbf=False)[0, 1], 0)

    def test_all_constant_features_predict_exact_common_baseline(self):
        labels = np.repeat(np.arange(8), 2)
        angles, t = labels*np.pi/4, np.linspace(0, 1, 15)
        y = np.stack([np.cos(angles)[:, None]*(1+t), np.sin(angles)[:, None]*(2-t)], axis=-1)
        x = np.ones((16, 15, 3))
        banks, base, predictions = fit_banks(x, y, labels, np.array([f"t{i}" for i in range(16)]), "constant")
        for recipe in RECIPES:
            for prediction, row in zip(predictions[recipe], banks[recipe]["lambda_rows"]):
                np.testing.assert_array_equal(prediction, base)
                self.assertEqual(row["delta_R2"], 0.)

    def test_capacity_outer_lambda_excludes_evaluated_session(self):
        for recipe in RECIPES:
            banks = {s: {"recipe": recipe, "lambdas": list(LAMBDAS), "baseline_R2": .5,
                "lambda_rows": [{"lambda": p, "delta_R2": float(i), "identity_increment": .01}
                                for i, p in enumerate(LAMBDAS)]} for s in SESSIONS}
            before = summarize_banks(banks, recipe)
            for held in SESSIONS:
                changed = copy.deepcopy(banks)
                for i, row in enumerate(changed[held]["lambda_rows"]):
                    row["delta_R2"] = 1e8*(len(LAMBDAS)-i)
                after = summarize_banks(changed, recipe)
                first = next(r for r in before["sessions"] if r["session_id"] == held)
                second = next(r for r in after["sessions"] if r["session_id"] == held)
                self.assertEqual(first["lambda_index"], second["lambda_index"])
                self.assertEqual(first["inner_animal_balanced_scores"], second["inner_animal_balanced_scores"])


if __name__ == "__main__":
    unittest.main()
