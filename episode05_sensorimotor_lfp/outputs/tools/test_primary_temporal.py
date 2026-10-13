"""Focused tests of the adopted temporal/control scientific comparison."""
import copy
import unittest

import numpy as np

from primary_capacity import score_bank
from primary_reference import LAMBDAS, SESSIONS
from primary_temporal import (
    RECIPES, REPLICATES, average_control_banks, fit_banks, fit_tensor, prepare_split,
    shared_ridge_predictions, summarize_banks, time_basis, training_derangement,
    transform_tensor,
)


class TemporalTest(unittest.TestCase):
    def test_discrete_basis_centering_and_stationary_embedding(self):
        rng = np.random.default_rng(50601)
        x = rng.normal(size=(6, 15, 4))+np.arange(15)[None, :, None]
        evaluation = rng.normal(size=(2, 15, 4))
        tensor, state = fit_tensor(x)
        q = time_basis()
        np.testing.assert_allclose(q.T@q/15, np.eye(3), atol=3e-16)
        np.testing.assert_allclose(tensor.mean(axis=(0, 1)), 0, atol=5e-16)
        z = (x-state.channel_mean)/state.channel_sd
        np.testing.assert_allclose(tensor[:, :, :4], z, atol=5e-16)
        beta = rng.normal(size=(4, 2))
        embedded = np.vstack((beta, np.zeros((8, 2))))
        expected = ((evaluation-state.channel_mean)/state.channel_sd)@beta
        np.testing.assert_allclose(transform_tensor(evaluation, state)@embedded, expected, atol=1e-15)
        # Products have genuinely nonzero uncentered time-weighted means here.
        self.assertGreater(np.max(np.abs(state.product_mean[4:])), .1)

    def test_eigendecomposition_matches_independent_refits_for_each_arm(self):
        rng = np.random.default_rng(50602)
        fitting = rng.normal(size=(3, 15, 5))
        fitting -= fitting.mean(axis=(0, 1))
        evaluation, residual = rng.normal(size=(2, 15, 5)), rng.normal(size=(45, 2))
        permutations = [np.array([1, 2, 0]), np.array([2, 0, 1])]
        actual = shared_ridge_predictions(fitting, evaluation, residual, permutations)
        for arm, perm in enumerate([np.arange(3)]+permutations):
            design = fitting[perm].reshape(45, 5)
            for index, penalty in enumerate(LAMBDAS):
                coefficient = np.linalg.solve(design.T@design+45*penalty*np.eye(5), design.T@residual)
                expected = evaluation.reshape(30, 5)@coefficient
                np.testing.assert_allclose(actual[arm, index], expected, atol=3e-15, rtol=1e-12)

    def test_complete_trial_derangements_preserve_direction_time_and_gram(self):
        rng = np.random.default_rng(50603)
        held = 3
        labels = np.repeat([d for d in range(8) if d != held], 4)
        ids = np.array([f"trial{i:04d}" for i in range(len(labels))])
        fitting = rng.normal(size=(len(labels), 15, 6))
        flat = fitting.reshape(-1, 6)
        for seed in REPLICATES:
            perm = training_derangement(ids, labels, "synthetic", held, seed)
            np.testing.assert_array_equal(perm, training_derangement(ids, labels, "synthetic", held, seed))
            self.assertEqual(set(perm.tolist()), set(range(len(labels))))
            self.assertTrue(np.all(perm != np.arange(len(labels))))
            np.testing.assert_array_equal(labels[perm], labels)
            shuffled = fitting[perm]
            for target, source in enumerate(perm):
                np.testing.assert_array_equal(shuffled[target], fitting[source])
            np.testing.assert_allclose(shuffled.reshape(-1, 6).T@shuffled.reshape(-1, 6), flat.T@flat,
                                       atol=4e-13, rtol=2e-15)

    def test_held_direction_cannot_change_fitting_state_or_trial_maps(self):
        rng = np.random.default_rng(50604)
        labels = np.repeat(np.arange(8), 3)
        ids = np.array([f"trial{i:04d}" for i in range(len(labels))])
        x, y = rng.normal(size=(24, 15, 3)), rng.normal(size=(24, 15, 2))
        first = prepare_split(x, y, labels, ids, "synthetic", 3)
        changed_x, changed_y = x.copy(), y.copy()
        changed_x[labels == 3], changed_y[labels == 3] = 1e9, -1e9
        other = prepare_split(changed_x, changed_y, labels, ids, "synthetic", 3)
        for key in ("features", "residual", "intercept"):
            np.testing.assert_array_equal(first[key], other[key])
        for key in ("channel_mean", "channel_sd", "product_mean"):
            np.testing.assert_array_equal(getattr(first["tensor_state"], key), getattr(other["tensor_state"], key))
        for before, after in zip(first["permutations"], other["permutations"]):
            np.testing.assert_array_equal(before, after)

    def test_seed_aggregation_is_mean_score_not_ensemble_score(self):
        rng = np.random.default_rng(50605)
        labels = np.repeat(np.arange(8), 2)
        y = rng.normal(size=(16, 15, 2))
        baseline = np.zeros_like(y)
        noise = rng.normal(size=y.shape)
        seed_predictions = []
        banks = []
        for seed in REPLICATES:
            prediction = y+(1 if seed % 2 else -1)*noise
            predictions = np.repeat(prediction[None], len(LAMBDAS), axis=0)
            seed_predictions.append(prediction)
            banks.append(score_bank(y, labels, baseline, predictions, RECIPES[1]))
        actual = average_control_banks(banks)
        expected = float(np.square(noise).sum())
        self.assertAlmostEqual(actual["lambda_rows"][0]["candidate_SSE"], expected)
        ensemble_sse = float(np.square(y-np.mean(seed_predictions, axis=0)).sum())
        self.assertLess(ensemble_sse, 1e-25)
        self.assertGreater(actual["lambda_rows"][0]["candidate_SSE"], ensemble_sse)

    def test_constant_features_keep_exact_baseline_and_input_targets(self):
        rng = np.random.default_rng(50606)
        labels = np.repeat(np.arange(8), 2)
        ids = np.array([f"trial{i:04d}" for i in range(len(labels))])
        x, y = np.ones((16, 15, 3)), rng.normal(size=(16, 15, 2))
        original_x, original_y, original_labels, original_ids = x.copy(), y.copy(), labels.copy(), ids.copy()
        banks, baseline, predictions = fit_banks(x, y, labels, ids, "constant")
        for prediction in predictions[RECIPES[0]]:
            np.testing.assert_array_equal(prediction, baseline)
        for seed_predictions in predictions[RECIPES[1]]:
            for prediction in seed_predictions:
                np.testing.assert_array_equal(prediction, baseline)
        for actual, original in ((x, original_x), (y, original_y), (labels, original_labels), (ids, original_ids)):
            np.testing.assert_array_equal(actual, original)
        for metadata in banks[RECIPES[0]]["direction_fit_metadata"]:
            targets = metadata["target_trial_ids_in_fitting_order"]
            self.assertEqual(len(metadata["training_feature_source_trial_ids_by_replicate"]), 8)
            for mapping in metadata["training_feature_source_trial_ids_by_replicate"]:
                self.assertEqual(set(targets), set(mapping["source_trial_ids"]))
                self.assertTrue(all(a != b for a, b in zip(targets, mapping["source_trial_ids"])))

    def test_outer_lambda_exclusion_and_same_lambda_for_control(self):
        def bank(recipe, increasing):
            rows = []
            for index, penalty in enumerate(LAMBDAS):
                delta = index*.01 if increasing else (4-index)*.01
                rows.append({"lambda": penalty, "delta_R2": delta, "identity_increment": .01,
                             "candidate_R2": .5+delta, "candidate_SSE": 5-10*delta,
                             "wrong_pair_SSE": 5-10*delta+.1})
            return {"recipe": recipe, "lambdas": list(LAMBDAS), "baseline_R2": .5,
                    "baseline_SSE": 5., "denominator": 10., "direction_counts": [2]*8,
                    "lambda_rows": rows}
        sessions = {s: {RECIPES[0]: bank(RECIPES[0], True),
                        RECIPES[1]: average_control_banks([bank(RECIPES[1], False) for _ in REPLICATES])}
                    for s in SESSIONS}
        before, _ = summarize_banks(sessions)
        for held in SESSIONS:
            changed = copy.deepcopy(sessions)
            for i, row in enumerate(changed[held][RECIPES[0]]["lambda_rows"]):
                row["delta_R2"] = 1e8*(len(LAMBDAS)-i)
            after, _ = summarize_banks(changed)
            for original_panel, changed_panel in zip(before, after):
                first = next(r for r in original_panel["sessions"] if r["session_id"] == held)
                second = next(r for r in changed_panel["sessions"] if r["session_id"] == held)
                self.assertEqual(first["lambda_index"], 4)
                self.assertEqual(first["lambda_index"], second["lambda_index"])
                self.assertEqual(first["paired_inner_animal_balanced_scores"], second["paired_inner_animal_balanced_scores"])


if __name__ == "__main__":
    unittest.main()
