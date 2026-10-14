"""Focused checks for the real-data bridge; execute only in Slurm."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import exploratory_fmri_design as design


def people_fixture():
    rows = []
    categories = design.RESPONSE_CATEGORIES
    for run in range(1, 5):
        for trial in range(64):
            category = categories[(trial+run) % 4]
            miss = trial == 7
            rows.append({"participant": "sub-fixture", "task_version": "ER", "run": str(run),
                         "trial_order": str(trial+1), "onset": str(2+trial*6),
                         "gain": str(5+trial//4), "loss": str(5+(trial*7+run)%16),
                         "choice": "-1" if miss else str(int("accept" in category)),
                         "strength": "-1" if miss else str(int(category.startswith("strong"))),
                         "response_category": "no_response" if miss else category,
                         "response_time": "" if miss else str(0.5+(trial%11)*0.13),
                         "no_response": str(int(miss)), "sequence_template": "fixture"})
    return {"sub-fixture": {"version": "ER", "rows": rows}}


class ExploratoryDesignTests(unittest.TestCase):
    def test_opposite_half_leakage_and_single_class_regularization(self):
        people = people_fixture()
        baseline, metadata = design.behavioral_covariates(people)
        changed = deepcopy(people)
        for row in changed["sub-fixture"]["rows"]:
            if int(row["run"]) in (2, 3) and int(row["choice"]) != -1:
                row["choice"] = "1"
                row["response_time"] = "3.9"
        alternative, alternate_metadata = design.behavioral_covariates(changed)
        for left, right in zip(baseline["sub-fixture"]["rows"], alternative["sub-fixture"]["rows"]):
            if int(left["run"]) in (2, 3):
                self.assertEqual(left["p_accept"], right["p_accept"])
                self.assertEqual(left["entropy_z"], right["entropy_z"])
        self.assertNotIn("p_accept", people["sub-fixture"]["rows"][0])
        fits = alternate_metadata["participants"]["sub-fixture"]["directions"]
        self.assertEqual(fits[1]["fit"]["training_classes"], [1])
        self.assertTrue(np.isfinite(fits[1]["coefficients"]).all())
        self.assertEqual(metadata["participants"]["sub-fixture"]["scoring_responded"], 252)
        # Reproduce an objective-based premature optimizer stop, then require
        # the unchanged gradient criterion on the original convex objective.
        train_rows = [row for row in people["sub-fixture"]["rows"]
                      if int(row["run"]) in (1, 4) and int(row["choice"]) != -1]
        x = design._features(train_rows)
        y = np.asarray([int(row["choice"]) for row in train_rows], dtype=float)
        premature = SimpleNamespace(x=np.zeros(3), success=True, nit=1,
                                    message="relative objective convergence")
        with patch.object(design, "minimize", return_value=premature):
            beta, polished = design._ridge_logistic(x, y, design.PENALTY, lambda: None)
        gradient = x.T @ (design.expit(x @ beta)-y)+design.PENALTY*beta
        self.assertGreater(polished["newton_polish_steps"], 0)
        self.assertLessEqual(polished["newton_polish_steps"], 25)
        self.assertLessEqual(np.max(np.abs(gradient)), 1e-6)

    def test_misses_durations_response_support_and_zero_entropy(self):
        rows = deepcopy(people_fixture()["sub-fixture"]["rows"][:64])
        for row in rows:
            row["entropy_z"] = 0.0
            if row["response_category"] == "strong_reject":
                row["response_category"] = "weak_reject"
                row["strength"] = "0"
        times = np.arange(205)*2.0
        original = design.compute_regressor
        calls = {}
        def record(events, **kwargs):
            calls[kwargs["con_id"]] = np.array(events)
            return original(events, **kwargs)
        with patch.object(design, "compute_regressor", side_effect=record):
            matrix, report = design.build_design(rows, times, pd.DataFrame(index=times), "S11")
        self.assertEqual(calls["offer"].shape[1], 64)
        self.assertEqual(calls["miss"].shape[1], 1)
        self.assertEqual(calls["decision_epoch"][1, 7], 4.0)
        self.assertEqual(sum(value.shape[1] for key, value in calls.items() if key.startswith("response_")), 63)
        self.assertTrue((matrix["entropy"] == 0).all())
        self.assertIn("response_strong_reject", report["zero_columns"])
        self.assertEqual(report["trial_count"], 64)
        self.assertIn("gain", matrix)
        self.assertIn("loss", matrix)

    def test_factorial_common_columns_timing_and_focal_estimability(self):
        augmented, _ = design.behavioral_covariates(people_fixture())
        rows = augmented["sub-fixture"]["rows"][:64]
        times = np.arange(205)*2.0
        nuisance = pd.DataFrame({"motion": np.sin(np.arange(len(times))*0.73)}, index=times)
        matrices = {arm: design.build_design(rows, times, nuisance, arm)[0] for arm in design.ARMS}
        for arm in design.ARMS:
            for name in ("offer", "miss", "motion", "constant", "drift_1"):
                np.testing.assert_array_equal(matrices[arm][name], matrices["S00"][name])
        for arm in ("S10", "S01", "S11"):
            for name in ("gain", "loss"):
                np.testing.assert_array_equal(matrices[arm][name], matrices["S00"][name])
        self.assertGreater(np.linalg.norm(matrices["R11"]["gain"]-matrices["S11"]["gain"]), 0)
        simple = pd.DataFrame({"gain": [1., 0., -1., 0.], "loss": [0., 1., 0., -1.],
                               "miss": [0., 0., 0., 0.], "constant": np.ones(4)})
        report = design.design_diagnostics(simple, ["gain", "loss", "miss"])
        self.assertTrue(report["estimable_gain"])
        self.assertAlmostEqual(report["focal"]["gain"]["variance"], 0.5)
        self.assertAlmostEqual(report["focal"]["gain"]["vif"], 1.0)
        self.assertAlmostEqual(report["task_condition_number"], 1.0)
        simple["duplicate_gain_confound"] = simple["gain"]
        report = design.design_diagnostics(simple, ["gain", "loss", "miss"])
        self.assertFalse(report["estimable_gain"])
        self.assertTrue(report["estimable_loss"])
        self.assertEqual(report["nuisance_absorbed_nonzero_task_columns"], ["gain"])
        self.assertFalse(report["task_condition_pass"])


if __name__ == "__main__":
    unittest.main()
