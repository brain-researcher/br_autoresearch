"""Focused primary-reference nesting, shared-intercept and pairing checks."""
import copy
import unittest

import numpy as np

from primary_reference import LAMBDAS, RECIPE, SESSIONS, choose_outer_lambdas, fit_bank, wrong_pair_sse


class PrimaryReferenceTest(unittest.TestCase):
    def test_outer_session_score_row_cannot_select_its_lambda(self):
        banks = {s: {"recipe": RECIPE, "lambdas": list(LAMBDAS),
                     "lambda_rows": [{"lambda": penalty, "delta_R2": float(index)}
                                     for index, penalty in enumerate(LAMBDAS)]} for s in SESSIONS}
        original = choose_outer_lambdas(banks)
        for held in SESSIONS:
            changed = copy.deepcopy(banks)
            for index, row in enumerate(changed[held]["lambda_rows"]):
                row["delta_R2"] = 1e8*(len(LAMBDAS)-index)
            selected = choose_outer_lambdas(changed)[held]
            self.assertEqual(selected, original[held])
            self.assertNotIn(held, selected["outer_training_sessions"])
        with self.assertRaises(ValueError):
            choose_outer_lambdas({k: v for k, v in banks.items() if k != SESSIONS[0]})

    def test_constant_lmp_has_no_gain_from_extra_intercept(self):
        # Unequal direction counts and a second harmonic produce a nonzero
        # pooled residual mean after the first-harmonic equal-direction fit.
        labels = np.repeat(np.arange(8), np.arange(2, 10))
        angle = labels*np.pi/4
        t = np.linspace(0, 1, 15)
        y = np.stack([np.cos(angle)[:, None]*(1+t)+np.cos(2*angle)[:, None],
                      np.sin(angle)[:, None]*(2-t)+np.sin(2*angle)[:, None]], axis=-1)
        bank, baseline, predictions = fit_bank(np.ones((len(y), 15, 1)), y, labels)
        for prediction, row in zip(predictions, bank["lambda_rows"]):
            np.testing.assert_allclose(prediction, baseline, atol=1e-14, rtol=0)
            self.assertEqual(row["delta_R2"], 0.0)
        self.assertEqual(bank["baseline_fits"], 8)
        self.assertEqual(bank["ridge_fits"], 40)

    def test_wrong_pair_mean_keeps_whole_trajectories_within_direction(self):
        labels = np.repeat(np.arange(8), 3)
        y = np.arange(24*15*2, dtype=float).reshape(24, 15, 2)/30
        prediction = y*.8+np.arange(24)[:, None, None]*.03
        expected = 0.0
        for direction in range(8):
            rows = np.flatnonzero(labels == direction)
            expected += sum(np.square(y[i]-prediction[j]).sum()
                            for i in rows for j in rows if i != j)/2
        self.assertAlmostEqual(wrong_pair_sse(y, prediction, labels), expected, places=10)


if __name__ == "__main__":
    unittest.main()
