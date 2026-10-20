"""Focused arithmetic and closed-access checks; no empirical audit input."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest import mock

import numpy as np
from scipy.stats import t

from calibration_falsifiers import TaskBObservation
from local_functional_source import TASK_A_CONDITIONS
from observed_map_audit_context import (
    ADOPTION_KEY, AUDIT_IDS, CONDITIONS, DEVELOPMENT_IDS, PARTITION_RUN_IDS,
    ROUTE_ID, SPECIFICATION, VARIANTS, audit_identity_pairings,
)
import report_observed_map_audit as reporter


def fixtures():
    # This local test dictionary does not alter or adopt the actual policy.
    policy = {ADOPTION_KEY: {"status": "adopted", "specification": SPECIFICATION,
                            "authority": "unit-test fixture only", "adopted_at_utc": "fixture"}}
    pattern = np.tile([-1., 1.], 94)
    observed = np.broadcast_to(pattern, (4, 18, 188)).copy()
    observations = tuple(TaskBObservation(person, CONDITIONS, tuple(range(188)), observed.copy())
                         for person in AUDIT_IDS)
    predictions = np.zeros((12, 16, 18, 188))
    predictions[:, 0] = pattern
    predictions[:, 1] = 3 * pattern
    basis = np.zeros((12, 8, 18, 188))
    basis[:, 1::2] = 2 * pattern
    warps = np.zeros((12, 4, 8, 18, 188))
    warps[:, :, 1::2] = 2 * pattern
    mapping = list(range(29))
    nonrest = [index for index, name in enumerate(TASK_A_CONDITIONS) if name != "rest"]
    for index, source in zip(nonrest, nonrest[1:] + nonrest[:1]):
        mapping[index] = source
    bundle = {
        "status": "complete_observed_map_audit_predictions", "route_id": ROUTE_ID,
        "development_ids": DEVELOPMENT_IDS, "audit_ids": AUDIT_IDS,
        "source_training_ids": DEVELOPMENT_IDS, "condition_ids": CONDITIONS,
        "variant_ids": VARIANTS, "partition_run_ids": PARTITION_RUN_IDS,
        "voxel_ids": tuple(range(188)), "donor_ids": tuple(donor for _, donor in audit_identity_pairings()),
        "condition_mapping": tuple(mapping), "M0a": np.zeros((18, 188)), "M0b": np.zeros((18, 188)),
        "predictions": predictions, "condition_control_predictions": np.zeros_like(predictions),
        "donor_control_predictions": np.zeros_like(predictions), "random_basis_predictions": basis,
        "matched_warp_predictions": warps, "numerical_certificates_complete": True,
        "audit_Task_B_accessed": False, "sampling_native_world": np.zeros((12, 188, 3)),
        "sampling_mask_headers": tuple({} for _ in AUDIT_IDS),
    }
    return policy, bundle, observations


class ObservedMapAuditReportTests(unittest.TestCase):
    def test_absent_adoption_cannot_touch_inputs_or_create_outputs(self):
        class Closed:
            def __iter__(self):
                raise AssertionError("closed input inspected")
            def __array__(self, *args, **kwargs):
                raise AssertionError("closed input converted")
        with TemporaryDirectory() as temporary:
            out, figure = Path(temporary) / "out", Path(temporary) / "figure"
            with self.assertRaises(PermissionError), mock.patch.object(reporter, "validate_prediction_bundle") as validate:
                reporter.report_audit_predictions(Closed(), Closed(), output_root=out,
                    figure_root=figure, policy={}, check=lambda: None)
            validate.assert_not_called()
            self.assertFalse(out.exists())
            self.assertFalse(figure.exists())

    def test_gain_sign_and_simultaneous_family_use_twelve_people_not_draws(self):
        deviations = np.arange(12, dtype=float) - 5.5
        gains = np.tile(deviations[:, None], (1, 16))
        gains[:, 0] += 20
        gains[:, 1] -= 20
        gains[:, 3] = 0
        primary = reporter._primary_family(gains)
        self.assertEqual(primary["participant_count"], 12)
        self.assertEqual(primary["degrees_of_freedom"], 11)
        self.assertEqual(primary["family_size"], 16)
        critical = t.isf(.05 / 32, 11)
        self.assertAlmostEqual(primary["critical_t"], critical)
        expected_width = critical * np.sqrt(13 / 12)  # sample variance of 0,...,11 is 13.
        self.assertTrue(np.allclose(primary["variants"][VARIANTS[0]]["simultaneous_95_percent_interval"],
                                    [20 - expected_width, 20 + expected_width]))
        decisions = [primary["variants"][name]["decision"] for name in VARIANTS[:4]]
        self.assertEqual(decisions, ["positive_heldout_observed_map_prediction_benefit",
                                    "negative_mean_gain", "inconclusive", "inconclusive"])
        with self.assertRaises(ValueError):
            reporter._primary_family(gains[:-1])

    def test_complete_report_retains_draw_losses_axes_and_exact_three_lines(self):
        policy, bundle, observations = fixtures()
        originals = bundle["predictions"].copy()
        with TemporaryDirectory() as temporary, mock.patch.object(reporter, "_figure") as figure:
            root, figures = Path(temporary) / "report", Path(temporary) / "figures"
            record = reporter.report_audit_predictions(bundle, observations, output_root=root,
                figure_root=figures, policy=policy, check=lambda: None)
            figure.assert_called_once()
            primary = record["primary"]["variants"]
            self.assertEqual(tuple(primary), VARIANTS)
            self.assertAlmostEqual(primary[VARIANTS[0]]["mean_gain"], 1.)
            self.assertAlmostEqual(primary[VARIANTS[1]]["mean_gain"], -3.)
            self.assertEqual(primary[VARIANTS[2]]["mean_gain"], 0.)
            controls = record["controls"]
            basis = controls["random_local_basis"]
            self.assertEqual(len(basis["draw_losses"]), 8)
            # Draw predictions 0 and 2*y average to y: averaging predictions
            # would falsely give zero loss, while each fixed draw has loss 1.
            self.assertAlmostEqual(basis["mean_draw_loss"]["mean_observed_mse"], 1.)
            self.assertEqual(basis["mean_draw_loss"]["partition_condition_observed_mse"].shape, (12, 4, 18))
            self.assertEqual(len(controls["matched_random_warp"]), 4)
            for variant in controls["matched_random_warp"].values():
                self.assertEqual(len(variant["draw_losses"]), 8)
                self.assertAlmostEqual(variant["mean_draw_loss"]["mean_observed_mse"], 1.)
            identity = controls["calibration"]["identity_derangement"]
            self.assertTrue(identity["identity_cycle_couples_recipients"])
            self.assertEqual(identity["inference"], "descriptive_only_no_intervals")
            self.assertEqual(tuple(identity["variants"]), VARIANTS)
            self.assertNotIn("interval", json.dumps(reporter._jsonable(controls)).replace("no_intervals", ""))
            self.assertEqual(len((root / "3lines.md").read_text().splitlines()), 3)
            self.assertEqual((root / "3lines.md").read_text(), (figures / "3lines.md").read_text())
            saved = json.loads((root / "observed_map_audit.complete.json").read_text())
            self.assertEqual(saved["participant_ids"], list(AUDIT_IDS))
            self.assertEqual(saved["support_size"], 188)
            self.assertFalse(saved["biological_mechanism_or_reliable_signal_recovery_established"])
            with self.assertRaisesRegex(ValueError, "fresh distinct"):
                reporter.report_audit_predictions(bundle, observations, output_root=root,
                    figure_root=figures, policy=policy, check=lambda: None)
        np.testing.assert_array_equal(originals, bundle["predictions"])

    def test_axis_join_and_nonfinite_values_cannot_shrink_the_audit(self):
        _, bundle, observations = fixtures()
        class Closed:
            def __array__(self, *args, **kwargs):
                raise AssertionError("observation converted before complete identity join")
        invalid = tuple(TaskBObservation(item.participant_id, CONDITIONS, item.voxel_ids, Closed())
                        for item in observations[:-1])
        with self.assertRaisesRegex(ValueError, "all original twelve"):
            reporter._score(bundle, invalid, check=lambda: None)
        changed = list(observations)
        changed[-1] = TaskBObservation(AUDIT_IDS[-1], CONDITIONS[:-1], bundle["voxel_ids"], Closed())
        with self.assertRaisesRegex(ValueError, "eighteen conditions"):
            reporter._score(bundle, changed, check=lambda: None)
        bad = observations[0].task_b_partitions.copy()
        bad[0, 0, 0] = np.nan
        changed = (TaskBObservation(AUDIT_IDS[0], CONDITIONS, bundle["voxel_ids"], bad),) + observations[1:]
        with self.assertRaisesRegex(ValueError, "finite"):
            reporter._score(bundle, changed, check=lambda: None)


if __name__ == "__main__":
    unittest.main()
