"""Two analytic metadata-control checks; execute only in coordinated Slurm."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import metadata_classifier as core


def check():
    pass


def fixture():
    records = []
    for _, filename, total in core.SESSIONS:
        training = total // 2
        calibration = (total - training) // 2
        records.append({"filename": filename, "status": "complete",
            "trial_counts": {"training": training, "calibration": calibration,
                             "evaluation": total - training - calibration},
            "regions": [{"region": region, "native_electrodes": 96,
                         "native_units": units, "training_finite_electrodes": 64,
                         "selected_electrode_ids": list(range(1, 16))}
                        for region, units in (("M1", 3), ("PMd", 1))]})
    return {"kind": "EP06_seven_session_development_views", "status": "completed",
        "stage": "preparation_only_no_scientific_scores", "reserved_sessions_opened": False,
        "original_trials": sum(row[2] for row in core.SESSIONS), "sessions": records}


class MetadataClassifierTests(unittest.TestCase):
    def test_analytic_count_pattern_and_frozen_source_scaling_under_target_shift(self):
        report = fixture()
        original = deepcopy(report)
        result = core.evaluate_metadata_classifier(report, check=check)
        self.assertEqual(report, original)
        # log1p(3)=log4 and log1p(1)=log2 standardize to +1/-1.
        # The paired session count has identical values for both labels;
        # its coefficient is zero. Unit coefficient b solves b=1/(1+exp(b)).
        lower, upper = 0., 1.
        for _ in range(80):
            middle = .5 * (lower + upper)
            if middle < 1. / (1. + math.exp(middle)):
                lower = middle
            else:
                upper = middle
        b = .5 * (lower + upper)
        for direction, rows in zip(result["directions"], ((6, 8), (8, 6))):
            self.assertEqual((direction["source_rows"], direction["target_rows"]), rows)
            np.testing.assert_allclose(direction["coefficients"], (b, 0., 0.), rtol=0, atol=3e-9)
            self.assertAlmostEqual(direction["intercept"], 0., places=10)
            self.assertAlmostEqual(direction["source_mean"][0], 1.5 * math.log(2), places=14)
            self.assertAlmostEqual(direction["source_sd_ddof0"][0], .5 * math.log(2), places=14)
            self.assertEqual(direction["source_constant_columns"], [False, True, False])
            self.assertAlmostEqual(direction["objective"], math.log1p(math.exp(-b)) + .5 * b * b, places=12)
            self.assertLessEqual(direction["maximum_absolute_gradient"], 1e-9)
            np.testing.assert_allclose([row["logit"] for row in direction["predictions"]],
                                       [b, -b] * (rows[1] // 2), rtol=0, atol=3e-9)
            self.assertEqual(direction["balanced_accuracy"], 1.)
        self.assertEqual(result["equal_direction_mean_balanced_accuracy"], 1.)

        shifted = deepcopy(report)
        for record in shifted["sessions"][3:]:
            for regional, units, usable in zip(record["regions"], (15, 7), (80, 48)):
                regional["native_units"] = units
                regional["training_finite_electrodes"] = usable
                regional["selected_electrode_ids"] = list(range(81, 96))
        changed = core.evaluate_metadata_classifier(shifted, check=check)["directions"][0]
        initial = result["directions"][0]
        for key in ("source_mean", "source_sd_ddof0", "source_constant_columns", "coefficients", "intercept"):
            self.assertEqual(changed[key], initial[key])
        # Target log16/log8 produce +5/+3 source SDs around 1.5*log2.
        # Target recentering would spuriously restore perfect classification.
        np.testing.assert_allclose([row["logit"] for row in changed["predictions"]],
                                   [5 * b, 3 * b] * 4, rtol=0, atol=2e-8)
        self.assertEqual([row["predicted_label"] for row in changed["predictions"]], [1] * 8)
        self.assertEqual(changed["balanced_accuracy"], .5)
        self.assertEqual([row["accuracy"] for row in changed["paired_session_accuracy"]], [.5] * 4)

    def test_constant_support_ties_and_invalid_missing_or_misbound_metadata_are_whole_unavailable(self):
        constant = fixture()
        for record in constant["sessions"]:
            for regional in record["regions"]:
                regional["native_units"] = 8
        result = core.evaluate_metadata_classifier(constant, check=check)
        for direction in result["directions"]:
            self.assertEqual(direction["source_constant_columns"], [True, True, False])
            self.assertEqual(direction["coefficients"], [0., 0., 0.])
            self.assertEqual(direction["intercept"], 0.)
            self.assertEqual(direction["tied_rows"], direction["target_rows"])
            self.assertTrue(all(row["logit"] == 0 and row["predicted_label"] is None
                                and row["credit"] == .5 for row in direction["predictions"]))
        self.assertEqual(result["equal_direction_mean_balanced_accuracy"], .5)
        cases = []
        bad = fixture(); bad["sessions"] = bad["sessions"][:-1]; cases.append(bad)
        bad = fixture(); bad["sessions"][-1]["regions"].pop(); cases.append(bad)
        bad = fixture(); bad["sessions"][-1]["regions"][0].pop("native_units"); cases.append(bad)
        bad = fixture(); bad["sessions"][-1]["regions"][0]["training_finite_electrodes"] = 14; cases.append(bad)
        bad = fixture(); bad["sessions"][-1]["regions"][0]["native_units"] = True; cases.append(bad)
        bad = fixture(); bad["sessions"][-1]["regions"][0]["native_units"] = float("nan"); cases.append(bad)
        bad = fixture(); bad["sessions"][-1]["regions"].reverse(); cases.append(bad)
        bad = fixture(); bad["sessions"][0]["filename"] = "Mihili_CO_20140303.mat"; cases.append(bad)
        bad = fixture(); bad["sessions"][0]["trial_counts"]["training"] = 0; cases.append(bad)
        bad = fixture(); bad["sessions"][0]["trial_counts"]["training"] += 1; cases.append(bad)
        bad = fixture(); bad["sessions"][0]["trial_counts"]["fit"] = bad["sessions"][0]["trial_counts"].pop("training"); cases.append(bad)
        bad = fixture(); bad["status"] = "preparing"; cases.append(bad)
        bad = fixture(); bad["reserved_sessions_opened"] = True; cases.append(bad)
        # Even an invalid final target row must fail before either model fit.
        with patch.object(core, "_fit_logistic", side_effect=AssertionError("must not fit incomplete metadata")):
            for invalid in cases:
                with self.subTest(invalid=invalid):
                    with self.assertRaises(core.MetadataUnavailable):
                        core.evaluate_metadata_classifier(invalid, check=check)
        with patch.object(core, "_fit_logistic", side_effect=core.ClassificationUnavailable("numerical failure")):
            with self.assertRaisesRegex(core.MetadataUnavailable, "source metadata fit unavailable"):
                core.evaluate_metadata_classifier(fixture(), check=check)


if __name__ == "__main__":
    unittest.main()
