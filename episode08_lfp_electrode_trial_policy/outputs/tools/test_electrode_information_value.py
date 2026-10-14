"""Focused checks for leakage, paired arithmetic and ridge-duplication risks."""
import copy
import unittest

import numpy as np

from electrode_information_value import (contexts, fit_removals, removal_values,
    fit_value_predictor, predict_value, columns)


class InformationValueTests(unittest.TestCase):
    def test_session_and_animal_labels_cannot_enter_own_value_predictor(self):
        rng = np.random.default_rng(17)
        sessions = []
        for index in range(8):
            rows = [{"prefix_trials": 32, "base_features": rng.normal(size=9).tolist(),
                     "conditional_CV_gain": float(rng.normal()), "removal_value": float(rng.normal()),
                     "row_weight": 1/384} for _ in range(384)]
            sessions.append({"filename": f"s{index}", "animal": "Chewie" if index < 4 else "Mihili", "rows": rows})
        for opposite in (False, True):
            for augmented in (False, True):
                before = fit_value_predictor(sessions, "s0", 32, augmented, opposite)
                changed = copy.deepcopy(sessions)
                for session in changed:
                    if session["filename"] == "s0" or (opposite and session["animal"] == "Chewie"):
                        for row in session["rows"]:
                            row["removal_value"] = 1e12
                            row["base_features"] = [-1e8]*9
                            row["conditional_CV_gain"] = 1e9
                after = fit_value_predictor(changed, "s0", 32, augmented, opposite)
                for key in ("mean", "sd", "beta", "intercept"):
                    np.testing.assert_array_equal(before[key], after[key])
                self.assertNotIn("s0", before["donor_sessions"])

    def test_redundant_added_feature_cannot_change_baseline_shrinkage(self):
        rng = np.random.default_rng(85)
        sessions = []
        for index in range(8):
            rows = []
            for _ in range(384):
                x = rng.normal(size=9)
                rows.append({"prefix_trials": 32, "base_features": x.tolist(),
                             "conditional_CV_gain": float(x[0]),
                             "removal_value": float(3*x[0]+rng.normal()), "row_weight": 1/384})
            sessions.append({"filename": f"s{index}", "animal": "Chewie" if index < 4 else "Mihili", "rows": rows})
        baseline = fit_value_predictor(sessions, "s0", 32, False)
        augmented = fit_value_predictor(sessions, "s0", 32, True)
        np.testing.assert_allclose(predict_value(baseline, sessions[0]["rows"], False),
                                   predict_value(augmented, sessions[0]["rows"], True), rtol=1e-12, atol=1e-12)

    def test_removal_pair_uses_one_denominator_and_refits_fixed_objective(self):
        rng = np.random.default_rng(31)
        x, y = rng.normal(size=(96, 288)), rng.normal(size=(96, 3))
        directions = np.repeat(np.arange(8), 12)
        ex, ey = rng.normal(size=(40, 288)), rng.normal(size=(40, 3))
        ed = np.repeat(np.arange(8), 5)
        channels = list(range(96)); menu = contexts("fixture", channels)
        frozen = fit_removals(x, y, directions, menu, lambda: None, {"ridge_fits": 0})
        values, metrics = removal_values(frozen, ex, ey, ed, channels)
        s = frozen[0]
        tx = (x-s.direction_input_mean[directions]-s.input_mean)/s.input_sd
        ty = (y-s.direction_target_mean[directions])/s.target_sd-s.target_residual_mean
        vx = (ex-s.direction_input_mean[ed]-s.input_mean)/s.input_sd
        vy = (ey-s.direction_target_mean[ed])/s.target_sd-s.target_residual_mean
        sst = float(np.sum(((ey-y.mean(0))/np.where(y.std(0) == 0, 1., y.std(0)))**2))
        errors = []
        for cols in (columns(range(96)), columns(range(1, 96))):
            # Independent primal solution versus the production dual ridge.
            beta = np.linalg.solve(tx[:, cols].T@tx[:, cols]+96*np.eye(len(cols)), tx[:, cols].T@ty)
            errors.append(float(np.sum((vy-vx[:, cols]@beta)**2)))
        self.assertAlmostEqual(values[("full96", 0)], (errors[1]-errors[0])/sst, places=12)
        self.assertEqual(len(values), 384)
        for metric in metrics:
            self.assertAlmostEqual(metric["denominator"], sst, places=12)
        self.assertAlmostEqual(sum(c["row_weight"]*len(c["indices"]) for c in menu), 1.)


if __name__ == "__main__":
    unittest.main()
