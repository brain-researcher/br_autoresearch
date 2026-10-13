"""Focused full-span, band-identity, fair-baseline and outer-exclusion checks."""
import copy
import unittest

import numpy as np

from primary_multichannel import BLOCKS, fit_bank, guide_columns, ridge_solve, summarize_banks
from primary_reference import LAMBDAS, SESSIONS


class MultichannelTest(unittest.TestCase):
    def test_rank_deficient_full_orthogonal_span_equals_direct_ridge(self):
        rng = np.random.default_rng(20261003)
        a = rng.normal(size=(45, 3))
        x = np.column_stack((a, a[:, 0]+a[:, 1], np.ones(45)))
        scale = x.std(axis=0)
        z = (x-x.mean(axis=0))/np.where(scale > 0, scale, 1.)
        y, evaluation = rng.normal(size=(45, 2)), rng.normal(size=(7, 5))
        y -= y.mean(axis=0)
        _, _, vt = np.linalg.svd(z, full_matrices=True)
        rotation = vt.T
        self.assertEqual(rotation.shape, (5, 5))
        self.assertLess(np.linalg.matrix_rank(z), 5)
        for penalty in LAMBDAS:
            direct, rotated = ridge_solve(z, y, penalty), ridge_solve(z@rotation, y, penalty)
            np.testing.assert_allclose(evaluation@direct, (evaluation@rotation)@rotated, atol=2e-12, rtol=2e-12)

    def test_exact_guide_identity_and_electrode_major_order(self):
        guide = np.array([[9, 4, 8], [2, .5, 4], [9, .5, 4], [2, 4, 8], [2, 0, 0]])
        self.assertEqual(guide_columns(guide, [2, 9], ((.5, 4), (4, 8))), [1, 3, 2, 0])
        with self.assertRaises(ValueError):
            guide_columns(guide, [2, 9], ((0, 4),))
        with self.assertRaises(ValueError):
            guide_columns(np.vstack((guide, guide[1])), [2], ((.5, 4),))

    def test_constant_multichannel_features_add_no_intercept_gain(self):
        labels = np.repeat(np.arange(8), np.arange(2, 10))
        a, t = labels*np.pi/4, np.linspace(0, 1, 15)
        y = np.stack([np.cos(a)[:, None]*(1+t)+np.cos(2*a)[:, None],
                      np.sin(a)[:, None]*(2-t)+np.sin(2*a)[:, None]], axis=-1)
        x = np.broadcast_to(np.array([1., 2., -3.]), (len(y), 15, 3)).copy()
        bank, base, predictions = fit_bank(x, y, labels, next(iter(BLOCKS)))
        self.assertFalse(bank["numerical_PCA_computed"])
        for prediction, row in zip(predictions, bank["lambda_rows"]):
            np.testing.assert_allclose(prediction, base, atol=1e-14, rtol=0)
            self.assertEqual(row["delta_R2"], 0.)

    def test_each_recipe_outer_selection_excludes_its_own_row(self):
        for recipe in BLOCKS:
            banks = {s: {"recipe": recipe, "lambdas": list(LAMBDAS), "baseline_R2": .5,
                "lambda_rows": [{"lambda": p, "delta_R2": float(i), "identity_increment": .02}
                                for i, p in enumerate(LAMBDAS)]} for s in SESSIONS}
            before = summarize_banks(banks, recipe)
            for held in SESSIONS:
                changed = copy.deepcopy(banks)
                for i, row in enumerate(changed[held]["lambda_rows"]):
                    row["delta_R2"] = 1e9*(len(LAMBDAS)-i)
                result = summarize_banks(changed, recipe)
                a = next(r for r in before["sessions"] if r["session_id"] == held)
                b = next(r for r in result["sessions"] if r["session_id"] == held)
                self.assertEqual(a["lambda_index"], b["lambda_index"])
                self.assertEqual(a["inner_animal_balanced_scores"], b["inner_animal_balanced_scores"])


if __name__ == "__main__":
    unittest.main()
