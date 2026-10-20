"""Focused fixtures for the INACTIVE common-support followup; no real data."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np

from calibration_falsifiers import TaskBObservation
from observed_map_audit_context import AUDIT_IDS, CONDITIONS, VARIANTS
import observed_map_common_support as support
import report_observed_map_common_support as reporter
import run_observed_map_common_support as runner
from test_report_observed_map_audit import fixtures


def inputs():
    policy, bundle, observations = fixtures()
    policy[support.ADOPTION_KEY] = {"status": "adopted", "specification": support.SPECIFICATION,
        "route_id": support.ROUTE_ID, "authority": "test fixture only; does not amend episode policy",
        "adopted_at_utc": "2026-10-20T00:00:00+00:00"}
    bundle["frozen_at_utc"] = "2026-10-19T10:00:00+00:00"
    return policy, bundle, observations


def plan():
    return {person: tuple({"participant_id": person, "run": run, "reg_id": condition,
                         "cond_name": name, "beta_member": f"{person}/{run}/{condition}"}
                        for run in range(1, 17) for condition, name in enumerate(CONDITIONS))
            for person in AUDIT_IDS}


def record(keep, bundle):
    return {"status": "complete_strict_availability_intersection", "route_id": support.ROUTE_ID,
            "participant_ids": AUDIT_IDS, "condition_ids": CONDITIONS,
            "original_voxel_ids": bundle["voxel_ids"], "keep": keep.tolist(),
            "retained_voxel_ids": np.asarray(bundle["voxel_ids"])[keep].tolist(),
            "retained_count": int(keep.sum()), "selected_member_counts": [288] * 12,
            "all_selected_members_scanned": True, "predictive_losses_computed": False,
            "frozen_at_utc": "2026-10-20T01:00:00+00:00"}


class NeverInspect:
    def __iter__(self):
        raise AssertionError("unadopted audit input inspected")
    def __array__(self, *args, **kwargs):
        raise AssertionError("unadopted audit input converted")


class CommonSupportTests(unittest.TestCase):
    def test_original_adoption_does_not_authorize_new_route(self):
        policy, _, _ = fixtures()  # Original188 adoption ONLY.
        with TemporaryDirectory() as temporary:
            with mock.patch.object(runner.original, "load_frozen_predictions") as load:
                with self.assertRaises(PermissionError):
                    runner.evaluate(evaluation_manifest=NeverInspect(), policy=policy,
                        check=lambda: None, durable_byte_cap=10000000)
                load.assert_not_called()
            with mock.patch.object(support, "validate_prediction_bundle") as validate:
                with self.assertRaises(PermissionError):
                    support.scan_audit_support(policy=policy, frozen_predictions=NeverInspect(),
                        release_bindings=NeverInspect(), member_rows=NeverInspect(),
                        condition_multiplicity_by_session=NeverInspect(), check=lambda: None)
                validate.assert_not_called()
            output = Path(temporary) / "report"
            with self.assertRaises(PermissionError):
                reporter.report_common_support(NeverInspect(), NeverInspect(), support_record=NeverInspect(),
                    policy=policy, output_root=output, figure_root=Path(temporary) / "figure", check=lambda: None)
            self.assertFalse(output.exists())

    def test_sampler_retains_strict_positive_weight_corner_rule_not_positive_beta(self):
        volume = np.full((2, 2, 2), -2., dtype=float)
        volume[1, 0, 0] = np.nan
        image = {"values": volume, "affine": np.eye(4), "slope": 1., "intercept": 0.}
        world = np.zeros((188, 3))
        world[0] = [.5, 0, 0]  # The NaN corner has positive weight.
        with mock.patch.object(support, "validate_beta_reference"), \
             mock.patch.object(support, "read_image", return_value=image):
            values, valid = support.sample_availability(None, "fixture", {}, {}, world)
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(valid[1:].all())  # NaN has exactly zero weight here.
        np.testing.assert_array_equal(values[1:], -2.)

    def test_complete_intersection_visits_last_person_and_member_and_averages_all_runs(self):
        policy, bundle, _ = inputs()
        rows = plan()
        calls = []
        def sample(person, row):
            calls.append((person, row["run"], row["reg_id"]))
            values = np.full(188, row["run"] + row["reg_id"], dtype=float)
            valid = np.ones(188, dtype=bool)
            if len(calls) == 1:
                valid[0] = False
            if person == AUDIT_IDS[-1] and row["run"] == 16 and row["reg_id"] == 17:
                values[1] = np.nan
                valid[2] = False
            return values, valid
        result = support.scan_compiled_plan(rows, original_voxel_ids=bundle["voxel_ids"],
            sample=sample, policy=policy, check=lambda: None)
        self.assertEqual(len(calls), 12 * 16 * 18)
        self.assertEqual(result.selected_member_counts, (288,) * 12)
        np.testing.assert_array_equal(np.flatnonzero(~result.keep), [0, 1, 2])
        observed = result.observations[0].task_b_partitions
        np.testing.assert_allclose(observed[:, 0, 3], [2.5, 6.5, 10.5, 14.5])
        bad_plan = deepcopy(rows)
        bad_plan[AUDIT_IDS[-1]] = bad_plan[AUDIT_IDS[-1]][:-1]
        with self.assertRaisesRegex(ValueError, "every_run_condition"):
            support.scan_compiled_plan(bad_plan, original_voxel_ids=bundle["voxel_ids"],
                sample=sample, policy=policy, check=lambda: None)

    def test_empty_intersection_does_not_skip_remaining_maps(self):
        policy, bundle, _ = inputs()
        calls = []
        def sample(person, row):
            calls.append(1)
            return np.zeros(188), np.zeros(188, dtype=bool) if len(calls) == 1 else np.ones(188, dtype=bool)
        result = support.scan_compiled_plan(plan(), original_voxel_ids=bundle["voxel_ids"],
            sample=sample, policy=policy, check=lambda: None)
        self.assertEqual(len(calls), 3456)
        self.assertFalse(result.keep.any())

    def test_restrict_then_recenter_every_panel_member_and_preserve_all_draws(self):
        policy, bundle, observations = inputs()
        keep = np.zeros(188, dtype=bool)
        keep[:2] = True
        # Different excluded-site offsets would contaminate means if any
        # observation, baseline or control were centered before restriction.
        changed = []
        for index, item in enumerate(observations):
            values = item.task_b_partitions.copy()
            values[..., 2:] = 100 + index
            values[..., :2] += 10 * index
            changed.append(TaskBObservation(item.participant_id, CONDITIONS, item.voxel_ids, values))
        for key in reporter.PREDICTION_KEYS:
            bundle[key][..., 2:] = -900
            bundle[key][..., :2] += 37
        original = {key: bundle[key].copy() for key in reporter.PREDICTION_KEYS}
        scored = reporter.score_common_support(bundle, tuple(changed), policy=policy,
            support_record=record(keep, bundle), check=lambda: None)
        gains = scored["primary"]["variants"]
        self.assertAlmostEqual(gains[VARIANTS[0]]["mean_gain"], 1.)
        self.assertAlmostEqual(gains[VARIANTS[1]]["mean_gain"], -3.)
        self.assertEqual(len(gains), 16)
        self.assertEqual(scored["primary"]["degrees_of_freedom"], 11)
        self.assertNotIn("benefit_threshold", scored["primary"])
        for item in gains.values():
            self.assertNotIn("decision", item)
            self.assertNotIn("simultaneous_95_percent_interval", item)
            self.assertEqual(len(item["participant_gain"]), 12)
        controls = scored["controls"]
        self.assertEqual(len(controls["random_local_basis"]["draw_losses"]), 8)
        self.assertAlmostEqual(controls["random_local_basis"]["mean_draw_loss"]["mean_observed_mse"], 1.)
        self.assertEqual(len(controls["matched_random_warp"]), 4)
        for item in controls["matched_random_warp"].values():
            self.assertEqual(len(item["draw_losses"]), 8)
            self.assertAlmostEqual(item["mean_draw_loss"]["mean_observed_mse"], 1.)
        for item in controls["calibration"].values():
            self.assertEqual(len(item["variants"]), 16)
            for variant in item["variants"].values():
                self.assertAlmostEqual(variant["control_loss"]["mean_observed_mse"], 1.)
        for key, values in original.items():
            np.testing.assert_array_equal(bundle[key], values)

    def test_unfrozen_incomplete_or_one_location_support_cannot_be_scored(self):
        policy, bundle, observations = inputs()
        keep = np.ones(188, dtype=bool)
        bad = record(keep, bundle)
        bad["all_selected_members_scanned"] = False
        with self.assertRaisesRegex(ValueError, "complete_common_support"):
            reporter.score_common_support(bundle, observations, policy=policy, support_record=bad, check=lambda: None)
        keep[1:] = False
        with self.assertRaisesRegex(ValueError, "fewer_than_two"):
            reporter.score_common_support(bundle, observations, policy=policy,
                support_record=record(keep, bundle), check=lambda: None)

    def test_frozen_support_precedes_scores_and_failed_report_resumes_without_B(self):
        policy, bundle, observations = inputs()
        keep = np.ones(188, dtype=bool)
        keep[-4:] = False
        scan = support.SupportScan(keep, observations, (288,) * 12, bundle["voxel_ids"])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            followup = root / "common_support001"
            original_execution = root / "audit_execution.json"
            manifest = root / "B-only.json"
            original_state = {"status": "failed", "first_B_opening_utc": "2026-10-19T10:00:04+00:00",
                "binding": {"prediction_dir": str(root / "predictions"), "evaluation_manifest": str(manifest)},
                "attempts": [{"status": "failed"}]}
            original_execution.write_text(json.dumps(original_state))
            preserved = original_execution.read_bytes()
            paths = {"FOLLOWUP_ROOT": followup, "SUPPORT_DIR": followup / "frozen_support",
                     "RESULT_DIR": followup / "evaluation", "EXECUTION_PATH": followup / "execution.json"}
            calls = []
            def fail_after_freeze(*args, **kwargs):
                frozen = paths["SUPPORT_DIR"] / "support.json"
                self.assertTrue(frozen.is_file())
                self.assertFalse(json.loads(frozen.read_text())["predictive_losses_computed"])
                calls.append(1)
                raise RuntimeError("fixture report interruption")
            with mock.patch.multiple(runner, **paths), \
                 mock.patch.multiple(runner.original, EXECUTION_PATH=original_execution, PREDICTION_DIR=root / "predictions"), \
                 mock.patch.dict(runner.os.environ, {"SLURM_JOB_ID": "fixture"}), \
                 mock.patch.object(runner.original, "load_frozen_predictions", return_value=bundle), \
                 mock.patch.object(runner.original, "load_audit_inputs", return_value={}) as metadata, \
                 mock.patch.object(runner, "scan_audit_support", return_value=scan) as opening, \
                 mock.patch.object(runner, "report_common_support", side_effect=fail_after_freeze):
                with self.assertRaisesRegex(RuntimeError, "fixture report interruption"):
                    runner.evaluate(evaluation_manifest=manifest, policy=policy, check=lambda: None, durable_byte_cap=10000000)
                frozen_bytes = (paths["SUPPORT_DIR"] / "support.json").read_bytes()
                with self.assertRaisesRegex(ValueError, "explicit_engineering_resume"):
                    runner.evaluate(evaluation_manifest=manifest, policy=policy, check=lambda: None, durable_byte_cap=10000000)
                with self.assertRaisesRegex(RuntimeError, "fixture report interruption"):
                    runner.evaluate(evaluation_manifest=manifest, policy=policy, check=lambda: None, durable_byte_cap=10000000,
                        engineering_resume_reason="fixture report-only repair")
                self.assertEqual(opening.call_count, 1)
                self.assertEqual(metadata.call_count, 1)
                self.assertEqual(len(calls), 2)
                self.assertEqual(frozen_bytes, (paths["SUPPORT_DIR"] / "support.json").read_bytes())
                self.assertEqual(preserved, original_execution.read_bytes())
                state = json.loads(paths["EXECUTION_PATH"].read_text())
                self.assertEqual(len(state["renewed_B_openings"]), 1)
                self.assertEqual(len(state["attempts"]), 2)

    def test_report_has_figure_and_three_exploratory_lines_or_scientific_stop(self):
        policy, bundle, observations = inputs()
        for retained in (184, 1):
            keep = np.zeros(188, dtype=bool)
            keep[:retained] = True
            with self.subTest(retained=retained), TemporaryDirectory() as temporary:
                root = Path(temporary)
                result = reporter.report_common_support(bundle, observations, support_record=record(keep, bundle),
                    policy=policy, output_root=root / "report", figure_root=root / "figures", check=lambda: None)
                self.assertFalse(result["original188_audit_completed"])
                self.assertFalse(result["uncertainty_calibrated_after_selection"])
                self.assertEqual("primary" in result, retained >= 2)
                self.assertTrue((root / "figures/observed_map_common_support.png").is_file())
                self.assertTrue((root / "figures/observed_map_common_support.pdf").is_file())
                summary = (root / "report/3lines.md").read_text()
                self.assertEqual(len(summary.splitlines()), 3)
                self.assertEqual(summary, (root / "figures/3lines.md").read_text())
                self.assertIn("No original 188-location audit pass", summary)


if __name__ == "__main__":
    unittest.main()
