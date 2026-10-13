"""Scientific fixtures for the LOSO learned electrode-value selector."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from learned_electrode_value import donor_view, fit_ranker, mean_control_cells, rank_electrodes, view_features
from prepare_development_trial_roles import DEVELOPMENT


def records(view="outer"):
    result = []
    for index, name in enumerate(DEVELOPMENT):
        a = float(index+1)
        x = np.array([[a, 2-a, 0.], [a+2, 4-a, 0.]])
        y = np.array([a*.3-1., a*.8+.2])
        result.append({"filename": name, "animal": name.split("_")[0], "view": view,
            "channel_ids": [2, 5], "predictors": x.tolist(), "value_labels": y.tolist(),
            "fitting_rows": 32 if view == "outer" else 24, "label_denominator": 64.})
    return result


def donor_arrays(repeats):
    values = [-1., 1., -1., 1.] if repeats == 4 else [-1., 0., 1.]
    signal = np.tile(values, 8)
    labels = np.repeat(np.arange(8), repeats)
    x = np.zeros((len(signal), 6))
    x[:, 0], x[:, 3] = signal, signal
    y = (2*signal)[:, None]
    later_signal = np.tile([-1., 1.]*4, 8)
    later_x = np.zeros((64, 6))
    later_x[:, 0], later_x[:, 3] = later_signal, -later_signal
    return x, y, labels, later_x, (2*later_signal)[:, None], np.repeat(np.arange(8), 8)


class LearnedElectrodeValueTest(unittest.TestCase):
    def test_target_and_other_views_are_excluded_before_numerical_use(self):
        # Risk: target future labels or the wrong initializer view could enter
        # donor means/SDs even if excluded later from the regression itself.
        target = DEVELOPMENT[0]
        clean = records("0")
        expected = fit_ranker(clean, target, "0")
        changed = copy.deepcopy(clean)
        changed[0]["predictors"], changed[0]["value_labels"] = object(), object()
        for old in records("outer"):
            old["predictors"], old["value_labels"] = object(), object()
            changed.append(old)
        actual = fit_ranker(changed, target, "0")
        self.assertEqual(set(actual["donor_sessions"]), set(DEVELOPMENT)-{target})
        for key in ("predictor_mean", "predictor_sd", "coefficients", "intercept"):
            np.testing.assert_array_equal(actual[key], expected[key])

    def test_equal_animal_session_electrode_weights_and_donor_only_ridge(self):
        # Risk: the3-vs4 donor-session imbalance or target predictor scaling
        # could bias a supposedly animal-balanced cross-fitted ranker.
        target = DEVELOPMENT[0]
        data = records()
        selected = [r for r in data if r["filename"] != target]
        x = np.concatenate([r["predictors"] for r in selected])
        y = np.concatenate([r["value_labels"] for r in selected])
        w = np.concatenate([np.full(2, .5/(3 if r["animal"] == "Mihili" else 4)/2) for r in selected])
        self.assertAlmostEqual(float(w.sum()), 1.)
        mean = np.sum(w[:, None]*x, axis=0)
        sd = np.sqrt(np.sum(w[:, None]*(x-mean)**2, axis=0))
        sd = np.where(sd == 0, 1., sd)
        z = (x-mean)/sd
        intercept = float(w@y)
        beta = np.linalg.solve(z.T@(w[:, None]*z)+np.eye(3), z.T@(w*(y-intercept)))
        actual = fit_ranker(data, target, "outer")
        np.testing.assert_allclose(actual["predictor_mean"], mean, atol=1e-14)
        np.testing.assert_allclose(actual["predictor_sd"], sd, atol=1e-14)
        np.testing.assert_allclose(actual["coefficients"], beta, atol=2e-14)
        self.assertAlmostEqual(actual["intercept"], intercept, places=14)
        self.assertEqual(actual["predictor_sd"][2], 1.)

    def test_24_and_32_row_donor_labels_use_mean_loss_alpha_one_and_keep_negatives(self):
        # Risk:sum-loss ridge, target leakage into scaling, or clipping negative
        # generalization values would train a materially different ranking rule.
        for repeats, view in ((4, "outer"), (3, "0")):
            x, y, labels, later_x, later_y, later_labels = donor_arrays(repeats)
            actual = donor_view(x, y, labels, later_x, later_y, later_labels,
                                [2, 5], DEVELOPMENT[1], "Mihili", view)
            # Each physical block has one unit-variance predictive coordinate.
            # Alpha1 shrinks its coefficient by1/2. Later aligned and reversed
            # coordinates yield scaled squared errors1/4 and9/4 of the null.
            np.testing.assert_allclose(actual["value_labels"], [.75, -1.25], atol=3e-14)
            self.assertEqual(actual["fitting_rows"], 8*repeats)
            self.assertAlmostEqual(actual["label_denominator"], 64. if repeats == 4 else 96., places=12)
            self.assertEqual(actual["view"], view)
            self.assertEqual(actual["channel_ids"], [2, 5])

    def test_initializer_predictors_use_only_supplied_view_and_preserve_zero_coordinates(self):
        # Risk: raw direction signal could masquerade as residual association,
        # and redundancy could mistakenly include the electrode's own block.
        x, y, labels, _, _, _ = donor_arrays(4)
        actual = view_features(x, y, labels)
        # One shared variable coordinate, two constant coordinates perblock:
        # residual marginal gain1/2, no between-direction variance, corr²/9.
        np.testing.assert_allclose(actual, [[.5, 0., 1/9], [.5, 0., 1/9]], atol=2e-14)
        shifted_x, shifted_y = x.copy(), y.copy()
        for d in range(8):
            shifted_x[labels == d] += d*10
            shifted_y[labels == d] += d*5
        changed = view_features(shifted_x, shifted_y, labels)
        # Raw target SD is intentional: strong direction structure reduces
        # the residual association normalized by raw target variation.
        self.assertTrue(np.all(changed[:, 0] < actual[:, 0]))
        np.testing.assert_allclose(changed[:, 2], actual[:, 2], atol=2e-14)
        self.assertTrue(np.all(changed[:, 1] > actual[:, 1]))

    def test_ordinary_donor_label_permutations_allow_identity_and_record_exact_seeds(self):
        # Risk: rejection/reseeding or a target-dependent draw changes the
        # matched donor-label control and leaks selection across LOSO folds.
        calls = []
        class IdentityDraw:
            def seed(self, literal, version):
                calls.append((literal, version))
            def shuffle(self, values):
                pass
        data, target = records("0"), DEVELOPMENT[0]
        expected = fit_ranker(data, target, "0")
        with patch("learned_electrode_value.random.Random", side_effect=lambda: IdentityDraw()):
            actual = fit_ranker(data, target, "0", replicate=2)
        self.assertEqual(len(calls), 7)
        expected_literals = {f"ep08-learned-electrode-label-v1|20261003|{name}|0|2" for name in DEVELOPMENT if name != target}
        self.assertEqual({literal for literal, _ in calls}, expected_literals)
        self.assertTrue(all(version == 2 for _, version in calls))
        self.assertEqual(len(actual["label_permutations"]), 7)
        for row in actual["label_permutations"]:
            self.assertEqual(row["recipient_channel_ids"], [2, 5])
            self.assertEqual(row["donor_channel_ids"], [2, 5])
            self.assertEqual(row["replicate"], 2)
        np.testing.assert_array_equal(actual["coefficients"], expected["coefficients"])
        self.assertEqual(actual["intercept"], expected["intercept"])

    def test_physical_electrode_ids_break_exact_score_ties(self):
        # Risk: array order or a favorable tie choice could change a nested
        # acquisition set without changing learned value estimates.
        state = {"predictor_mean": np.zeros(3), "predictor_sd": np.ones(3),
                 "coefficients": np.zeros(3), "intercept": -.2}
        order, scores = rank_electrodes(np.array([[1., 2., 3.], [9., 8., 7.]]), [9, 3], state)
        np.testing.assert_array_equal(order, [1, 0])
        np.testing.assert_array_equal(scores, [-.2, -.2])

    def test_mean_control_cell_errors_preserve_common_denominator(self):
        # Risk: averaging predictions, choosing a seed or mismatching
        # normalizers changes the control estimand under equal target queues.
        panels = []
        for index in range(3):
            correct, wrong, native = (4., 9., 16.)[index], (12., 18., 24.)[index], (2., 5., 7.)[index]
            rows = []
            for n in (32, 48, 64, 80, 96):
                for e in (8, 16, 32, 96):
                    rows.append({"training_trials": n, "electrodes": e,
                        "standardized_SSE": correct, "standardized_wrong_SSE": wrong,
                        "null_SSE": 40., "native_target_SSE": native,
                        "Q": 1-correct/40, "Q_wrong": 1-wrong/40,
                        "pairing_increment": (wrong-correct)/40})
            panels.append({"cells": rows})
        actual = mean_control_cells(panels)
        self.assertEqual(len(actual["cells"]), 20)
        for row in actual["cells"]:
            self.assertAlmostEqual(row["standardized_SSE"], 29/3)
            self.assertEqual(row["standardized_wrong_SSE"], 18.)
            self.assertAlmostEqual(row["native_target_SSE"], 14/3)
            self.assertEqual(row["null_SSE"], 40.)
        self.assertAlmostEqual(actual["mean_Q"], 1-29/120)
        self.assertAlmostEqual(actual["mean_Q_wrong"], .55)
        self.assertAlmostEqual(actual["mean_pairing_increment"], (18-29/3)/40)
        altered = copy.deepcopy(panels)
        altered[1]["cells"][0]["null_SSE"] = 41.
        with self.assertRaises((ValueError, ArithmeticError)):
            mean_control_cells(altered)


if __name__ == "__main__":
    unittest.main()
