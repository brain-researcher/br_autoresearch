"""Observed-loss arithmetic and complete private publication; invented maps only.

Reuses the existing complete control fixture; its cheap fitting callbacks test
composition, not optimizer qualification. Execute in the coordinator's Slurm
allocation. Temporary scientific fixtures stay on node-local scratch.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import observed_beta_summary as summary
import run_development_point_panel as runner
from calibration_falsifiers import VARIANTS
from development_lopo import (
    DEVELOPMENT_IDS, M2_VARIANTS, M3_VARIANTS, M0aPrediction, MatchedWarpPrediction,
    RandomBasisPrediction, fit_development_lopo,
)
import test_calibration_falsifiers as existing


def complete_fixture():
    fixture = existing.CalibrationFalsifierTests()
    fixture.setUp()

    def anatomical(training, context):
        return M0aPrediction(np.mean([row.task_b_partitions for row in training], axis=(0, 1)),
                             np.eye(18), context.training_ids)

    def basis(source, task_a, numerical, context):
        base = numerical["gain"] * (source.G - source.G.mean(axis=-1, keepdims=True))
        return tuple(RandomBasisPrediction(draw,
            base + (-1 if draw % 2 else 1) * (draw // 2 + 1) * .03 * fixture.u,
            source.K0, {}) for draw in range(8))

    def warp(source, predictions, gain, context):
        return tuple(MatchedWarpPrediction(prediction.name, draw,
            prediction.maps + (-1 if draw % 2 else 1) * (draw // 2 + 1) * .04 * fixture.u,
            prediction.geometry, {}) for prediction in predictions for draw in range(8))

    ordinary = fit_development_lopo(fixture.records, fit_shared=fixture.shared,
        **fixture.callbacks(), fit_m0a=anatomical, predict_random_bases=basis,
        predict_matched_warps=warp, include_participant_deletions=False)
    controls = fixture.fit(tuple(fold.calibration_reference for fold in ordinary.full.folds))
    return fixture, ordinary, controls


class ObservedBetaSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture, cls.ordinary, cls.controls = complete_fixture()
        cls.result = summary.observed_beta_summary(cls.ordinary.full, cls.controls, check=lambda: None)

    def test_full_contributions_equal_direct_observed_error_and_keep_all_weights(self):
        self.assertEqual(len(self.fixture.source_calls), 24)
        self.assertEqual(len(self.fixture.calibration_calls), 48)
        self.assertEqual(self.ordinary.deletions, ())
        result = self.result
        self.assertEqual(tuple(result["ordinary"]["losses"]), ("M0a", "M0b", *VARIANTS))
        weights = np.array([1 / (7 * size) for size in (1, 3, 3, 3, 3, 2, 3) for _ in range(size)])
        np.testing.assert_array_equal(result["condition_weights"], weights)
        for name in ("M0b", VARIANTS[-1]):
            expected = np.empty((12, 4, 18))
            for person, fold in enumerate(self.ordinary.full.folds):
                maps = fold.G if name == "M0b" else fold.m4_predictions[-1].maps
                for partition in range(4):
                    for condition in range(18):
                        observed = fold.observed_partitions[partition, condition]
                        prediction = maps[condition]
                        expected[person, partition, condition] = np.mean(
                            ((observed - observed.mean()) - (prediction - prediction.mean())) ** 2)
            actual = result["ordinary"]["losses"][name]
            np.testing.assert_allclose(actual["partition_condition_observed_mse"], expected, atol=1e-14)
            target = (expected * weights[None, None]).sum(axis=2).mean(axis=1)
            np.testing.assert_allclose(actual["participant_mean_observed_mse"], target, atol=1e-14)
            self.assertAlmostEqual(actual["mean_observed_mse"], float(target.mean()), places=13)
        baseline = result["ordinary"]["losses"]["M0b"]["partition_condition_observed_mse"]
        for name in VARIANTS:
            value = result["ordinary"]["losses"][name]["partition_condition_observed_mse"]
            difference = result["ordinary"]["model_minus_baseline"][name]["model_minus_M0b"]
            np.testing.assert_allclose(difference["partition_condition_mse_difference"], value - baseline)
        # Spatial offsets carry no information, but no participant demeaning is applied.
        shifted = replace(self.ordinary.full, folds=tuple(replace(fold,
            observed_partitions=fold.observed_partitions + np.arange(72).reshape(4, 18, 1))
            for fold in self.ordinary.full.folds))
        shifted_result = summary.observed_beta_summary(shifted, self.controls, check=lambda: None)
        np.testing.assert_allclose(shifted_result["ordinary"]["losses"]["M0b"]
            ["partition_condition_observed_mse"], baseline, rtol=1e-12, atol=1e-13)

    def test_control_draws_are_squared_before_mean_and_identity_keeps_paired_true(self):
        ordinary = self.result["ordinary"]["losses"]
        for controls, true_name in ((self.result["random_local_basis"], M3_VARIANTS[0]),
                *((self.result["matched_random_warp"][name], name) for name in M2_VARIANTS)):
            self.assertEqual(len(controls["draw_losses"]), 8)
            expected = np.mean([draw["partition_condition_observed_mse"]
                                for draw in controls["draw_losses"]], axis=0)
            np.testing.assert_array_equal(controls["mean_draw_loss"]["partition_condition_observed_mse"], expected)
            # The invented symmetric draws average to the true prediction.
            # Their average error must be larger than the ensemble's error.
            self.assertTrue(np.all(expected > ordinary[true_name]["partition_condition_observed_mse"]))
        for label, folds in (("condition", self.controls.condition_folds), ("identity", self.controls.identity_folds)):
            actual = self.result["calibration_controls"][label]
            self.assertEqual(tuple(map(len, actual["source_training_ids"])), (11 if label == "condition" else 10,) * 12)
            for name in VARIANTS:
                entry = actual["variants"][name]
                np.testing.assert_array_equal(entry["control_minus_true"]["partition_condition_mse_difference"],
                    entry["control"]["partition_condition_observed_mse"] - entry["true"]["partition_condition_observed_mse"])
            for person, fold in enumerate(folds):
                raw = self.ordinary.full.folds[person].observed_partitions
                maps = fold.true_panel.m1_predictions[0].maps
                expected = np.mean(((raw - raw.mean(axis=-1, keepdims=True))
                    - (maps - maps.mean(axis=-1, keepdims=True))[None]) ** 2, axis=-1)
                np.testing.assert_allclose(actual["variants"][VARIANTS[0]]["true"]
                    ["partition_condition_observed_mse"][person], expected)
        identity_true = self.result["calibration_controls"]["identity"]["variants"][VARIANTS[0]]["true"]
        self.assertFalse(np.allclose(identity_true["partition_condition_observed_mse"],
                                    ordinary[VARIANTS[0]]["partition_condition_observed_mse"]))

    def test_wrong_complete_roster_draws_and_paired_sources_refuse_before_scoring(self):
        original, controls = self.ordinary.full, self.controls
        poison = replace(original, folds=(replace(original.folds[0], observed_partitions=existing.Poison()),
                                         *original.folds[1:]))
        last = poison.folds[-1]
        bad_runs = (replace(poison, folds=poison.folds[:-1]),
            replace(poison, folds=(*poison.folds[:-1], replace(last, random_basis_predictions=last.random_basis_predictions[:-1]))),
            replace(poison, folds=(*poison.folds[:-1], replace(last, matched_warp_predictions=last.matched_warp_predictions[::-1]))))
        for value in bad_runs:
            with self.subTest(run=True), self.assertRaises(ValueError):
                summary.observed_beta_summary(value, controls, check=lambda: None)
        identity = controls.identity_folds[-1]
        ordinary_reference = original.folds[-1].calibration_reference
        for change in (replace(identity, source=ordinary_reference.source),
                       replace(identity, true_panel=ordinary_reference.panel),
                       replace(identity, control_packet_id=identity.true_packet_id)):
            with self.subTest(control=True), self.assertRaises(ValueError):
                summary.observed_beta_summary(poison, replace(controls,
                    identity_folds=(*controls.identity_folds[:-1], change)), check=lambda: None)
        nonfinite = original.folds[0].observed_partitions.copy()
        nonfinite[0, -1, -1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            summary.observed_beta_summary(replace(original,
                folds=(replace(original.folds[0], observed_partitions=nonfinite), *original.folds[1:])),
                controls, check=lambda: None)


class ObservedBetaPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, ordinary, controls = complete_fixture()
        cls.panel = SimpleNamespace(ordinary_fit=ordinary, calibration_falsifier_fit=controls,
            ordinary_points={"V_map": -2., "V_geometry": -.25},
            calibration_falsifier_points={"M3_projected_geometry_contrast_algebraically_zero": True},
            m2_alignment_points={"signed_descriptive_value": -.125})

    def setUp(self):
        scratch = os.environ.get("L_SCRATCH_JOB") or os.environ.get("L_SCRATCH")
        if not scratch or not Path(scratch).is_dir():
            raise RuntimeError("an existing node-local Slurm scratch directory is required")
        self.temporary = tempfile.TemporaryDirectory(prefix="ep15-observed-loss-", dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.outputs = Path(self.temporary.name) / "outputs"
        (self.outputs / "runs").mkdir(parents=True)

    def execute(self, destination):
        return runner.execute(manifest_path=self.outputs / "invented_manifest.json", output_dir=destination,
            deadline_epoch=time.time() + 120, rss_limit_kib=4 * 1024**2,
            durable_byte_cap=32 * 1024**2, analysis_scope="conditional_observed_beta",
            publication_mode="withheld_cost_measurement")

    def test_cost_run_keeps_complete_reusable_science_private_and_operational_record_outcome_free(self):
        destination = self.outputs / "runs" / "cost001"
        with patch.object(runner, "OUTPUTS", self.outputs), patch.object(runner, "run_released_development",
                return_value=(self.panel, {"participants": 12, "support_locations": 208})) as fitting:
            self.assertEqual(self.execute(destination), 0)
            fitting.assert_called_once()
            with self.assertRaises(FileExistsError):
                self.execute(destination)
            fitting.assert_called_once()
        report = json.loads((destination / "operational.json").read_text())
        self.assertEqual(report["status"], "completed_observed_beta_cost_panel")
        self.assertTrue(report["private_summary_complete"])
        self.assertTrue(report["scientific_summary_withheld"])
        self.assertFalse(report["point_summaries_published"])
        self.assertEqual(report["complete_panel"]["source_fits"], 24)
        self.assertEqual(report["complete_panel"]["calibration_panels"], 48)
        self.assertNotIn("mean_observed_mse", (destination / "operational.json").read_text())
        self.assertNotIn("V_map", (destination / "operational.json").read_text())
        private = destination / "point_summaries.withheld.json"
        self.assertEqual(stat.S_IMODE(private.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o700)
        self.assertFalse((destination / "point_summaries.json").exists())
        values = json.loads(private.read_text())
        self.assertEqual(len(values["observed_beta"]["ordinary"]["losses"]), 18)
        self.assertEqual(len(values["observed_beta"]["matched_random_warp"]), 4)
        self.assertEqual(values["assumption_conditional_cross_partition_statistics"]["ordinary"]["V_map"], -2.)
        self.assertFalse(values["recovery_ratios_or_class_decisions"])

    def test_incomplete_fit_never_publishes_partial_scientific_panel(self):
        destination = self.outputs / "runs" / "failed001"
        with patch.object(runner, "OUTPUTS", self.outputs), patch.object(runner,
                "run_released_development", side_effect=ValueError("invented incomplete fit")):
            self.assertEqual(self.execute(destination), 1)
        report = json.loads((destination / "operational.json").read_text())
        self.assertEqual(report["status"], "stopped")
        self.assertFalse(report["private_summary_complete"])
        self.assertFalse(report["point_summaries_published"])
        self.assertEqual(set(path.name for path in destination.iterdir()), {"settings.json", "operational.json"})


if __name__ == "__main__":
    unittest.main()
