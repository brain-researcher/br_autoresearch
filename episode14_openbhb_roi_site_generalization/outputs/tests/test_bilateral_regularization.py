"""Focused arithmetic checks for the newly introduced bilateral penalty."""
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_bilateral_regularization as worker


class BilateralArithmetic(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(23)
        self.fit = self.rng.normal(size=(32, 8))
        self.held = self.rng.normal(size=(9, 8)) + 4
        self.age = self.rng.normal(size=32)
        self.pairs = np.array([[0, 4], [1, 5], [2, 6], [3, 7]])

    def test_orthogonal_rotation_preserves_ridge(self):
        direct, _ = worker.base.ridge_grid(self.fit, self.age, self.held, worker.ALPHAS)
        rotated, _ = worker.base.ridge_grid(worker.rotate(self.fit, self.pairs), self.age,
                                           worker.rotate(self.held, self.pairs), worker.ALPHAS)
        np.testing.assert_allclose(direct, rotated, atol=1e-10, rtol=1e-10)

    def test_scaled_features_match_explicit_block_penalty(self):
        alpha, ratio = 10., 100.
        z = worker.rotate(self.fit, self.pairs)
        held = worker.rotate(self.held, self.pairs)
        mean, ymean = z.mean(axis=0), self.age.mean()
        centered = z - mean
        penalty = alpha * np.diag([1.] * 4 + [ratio] * 4)
        weights = np.linalg.solve(centered.T @ centered + penalty, centered.T @ (self.age - ymean))
        expected = (held - mean) @ weights + ymean
        actual, _ = worker.base.ridge_grid(worker.rotate(self.fit, self.pairs, ratio), self.age,
                                          worker.rotate(self.held, self.pairs, ratio), np.array([alpha]))
        np.testing.assert_allclose(actual[:, 0], expected, atol=1e-10, rtol=1e-10)

    def test_fit_scale_has_no_evaluation_state(self):
        scale = worker.base.Scale().fit(self.fit)
        mean, std = scale.mean.copy(), scale.std.copy()
        worker.rotate(scale.transform(self.held), self.pairs, 10.)
        worker.rotate(scale.transform(self.held * 100), self.pairs, 10.)
        np.testing.assert_array_equal(scale.mean, mean)
        np.testing.assert_array_equal(scale.std, std)
        self.assertEqual(list(inspect.signature(worker.rotate).parameters), ["z", "pairs", "ratio"])

    def test_full_provider_suffixes_keep_measurement_families_separate(self):
        names = [f"{hemi}-region{region}_{family}" for hemi in ("lh", "rh")
                 for family in worker.PROVIDER_FAMILIES for region in range(34)]
        mappings, counts = worker.authenticate_pairs(names, worker.PROVIDER_FAMILIES)
        self.assertEqual(set(counts.values()), {34})
        for pairs in mappings.values():
            for left, right in pairs:
                left_family = [f for f in worker.PROVIDER_FAMILIES if names[left].endswith("_" + f)]
                right_family = [f for f in worker.PROVIDER_FAMILIES if names[right].endswith("_" + f)]
                self.assertEqual(left_family, right_family)
        bad = names.copy()
        bad[0] = bad[0].replace("surface_area_mm^2", "unknown_mm^2")
        with self.assertRaises(ValueError):
            worker.authenticate_pairs(bad, worker.PROVIDER_FAMILIES)

    def test_participant_loss_aggregation_and_ties(self):
        records = [{"appearances": [
            {"id": "a", "domain": "d1", "age": 20, "role": "external", "absolute_error": 1},
            {"id": "b", "domain": "d2", "age": 30, "role": "external", "absolute_error": 7}]},
            {"appearances": [
            {"id": "a", "domain": "d1", "age": 20, "role": "external", "absolute_error": 5}]}]
        collapsed = worker.base.collapse(records, "external", "absolute_error")
        self.assertEqual(collapsed["a"]["loss"], 3.)
        self.assertEqual(worker.base.summarize(collapsed)["domain_macro"], 5.)
        self.assertEqual(worker.choose(np.ones((5, 7))), (2, 6))


if __name__ == "__main__":
    unittest.main()
