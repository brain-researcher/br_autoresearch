"""Focused checks of new selection algebra and the nested fitting boundary."""
import unittest

import numpy as np

from ridge_training import fit_candidate, new_progress, scaling
from selector_training import (SELECTORS, build_order_bank, feature_columns,
                               fit_selection_panels, fit_with_orders, selector_order)


def progress():
    result = new_progress()
    result.update(selector_fits=0, selection_seconds=0.)
    return result


class SelectionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8202603)
        self.x = rng.normal(size=(32, 12))
        self.y = self.x[:, [1, 6, 10]] + rng.normal(size=(32, 3)) * .4
        self.directions = np.repeat(np.arange(8), 4)
        self.folds = [(np.flatnonzero(np.arange(32) % 4 != f),
                       np.flatnonzero(np.arange(32) % 4 == f)) for f in range(4)]

    def test_rank_three_updates_equal_direct_objective_refits(self):
        xm, xs, _ = scaling(self.x)
        ym, ys, _ = scaling(self.y)
        x, y = (self.x-xm)/xs, (self.y-ym)/ys
        for method in ("diversity", "conditional_target"):
            selected = []
            for _ in range(3):
                candidates = []
                for e in range(4):
                    if e in selected:
                        continue
                    cols = feature_columns(selected + [e])
                    kernel = np.eye(len(x)) + x[:, cols] @ x[:, cols].T / len(x)
                    if method == "diversity":
                        value = np.linalg.slogdet(kernel)[1]
                    else:
                        # Direct minimized ridge objective; no residual/block update.
                        value = -float(np.sum(y * np.linalg.solve(kernel, y))) / (len(y)*y.shape[1])
                    candidates.append((value, -e, e))
                selected.append(max(candidates)[2])
            actual = selector_order(self.x, self.y, self.directions, method, 3)
            np.testing.assert_array_equal(actual, selected)

    def test_inner_selector_cannot_read_its_validation_rows(self):
        first, _ = build_order_bank(self.x, self.y, self.directions, self.folds,
                                   "conditional_target", 3, "fixture", progress())
        changed_x, changed_y = self.x.copy(), self.y.copy()
        validation = self.folds[0][1]
        changed_x[validation] = np.arange(len(validation))[:, None] * 1000
        changed_y[validation] = np.arange(len(validation))[:, None] * -2000
        altered, _ = build_order_bank(changed_x, changed_y, self.directions, self.folds,
                                     "conditional_target", 3, "fixture", progress())
        np.testing.assert_array_equal(first[0], altered[0])
        for method in ("direction_repeatability", "diversity"):
            np.testing.assert_array_equal(
                selector_order(self.x, self.y, self.directions, method, 3),
                selector_order(self.x, changed_y, self.directions, method, 3))

    def test_fixed_selection_reduces_to_unchanged_ridge_and_full_model_is_shared(self):
        order = np.array([2, 0, 3, 1])
        columns = feature_columns(order[:2])
        actual, used = fit_with_orders(self.x, self.y, self.folds,
                                      [order]*4, order, 2, progress())
        expected = fit_candidate(self.x[:, columns], self.y, self.folds, new_progress())
        np.testing.assert_array_equal(used, columns)
        self.assertEqual(actual.alpha, expected.alpha)
        np.testing.assert_allclose(actual.coefficients, expected.coefficients, rtol=2e-10, atol=2e-12)
        panels = fit_selection_panels(self.x, self.y, self.directions, self.folds,
                                     "fixture", progress(), counts=(1, 2, 4), control_replicates=2)
        full = [model for model in panels if model.electrodes == 4]
        self.assertEqual(len(full), len(SELECTORS))
        self.assertTrue(all(model.fitted is full[0].fitted for model in full))
        original = fit_candidate(self.x, self.y, self.folds, new_progress())
        np.testing.assert_array_equal(full[0].fitted.coefficients, original.coefficients)


if __name__ == "__main__":
    unittest.main()
