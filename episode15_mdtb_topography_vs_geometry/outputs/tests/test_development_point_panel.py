"""Three integration seams, reusing existing source-free fixtures.

Opaque stage markers check orchestration only; they are not fake scientific
scores or substitutes for the already tested numerical components.
"""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import development_point_panel as panel
from development_handoff_reader import DevelopmentArrays
from development_lopo import DEVELOPMENT_IDS
from registered_endpoint_metrics import CONDITIONS
import test_calibration_falsifiers as control_fixture
import test_matched_random_basis as basis_fixture
from test_development_handoff_reader import release_fixture


class DevelopmentPointPanelTests(unittest.TestCase):
    def setUp(self):
        fixture = control_fixture.CalibrationFalsifierTests()
        fixture.setUp()
        bindings, self.support, _ = release_fixture([])
        self.header = bindings[0].anatomy_headers["space-SUIT_xfm.nii"]
        indices = self.support["source_transform_voxel_ijk"]
        ids = tuple(map(int, np.ravel_multi_index(indices.T, tuple(self.header["shape"][:3]), order="C")))
        records = tuple(replace(value, voxel_ids=ids) for value in fixture.records)
        packets = tuple(replace(value, voxel_ids=ids) for value in fixture.packets)
        observed = tuple(replace(value, voxel_ids=ids) for value in fixture.observations())
        self.data = DevelopmentArrays(records, packets, observed, ids, {})

    def run_panel(self, data=None, check=lambda: None):
        return panel.fit_development_point_panel(self.data if data is None else data,
            common_support=self.support, forward_header=self.header, check=check)

    def test_every_join_and_genuine_half_is_checked_before_model_construction(self):
        data = self.data
        # A late metadata mismatch must precede conversion of the first person's
        # response, not just precede that late person's eventual fit.
        records = (replace(data.participants[0], task_a_maps=control_fixture.Poison()),
                   *data.participants[1:])
        observed = (*data.observations[:-1], replace(data.observations[-1],
                                                   task_b_condition_ids=CONDITIONS[::-1]))
        cases = [replace(data, participants=records, observations=observed),
            replace(data, participants=data.participants[:-1]),
            replace(data, voxel_ids=data.voxel_ids[::-1]),
            replace(data, participants=(replace(data.participants[0],
                task_a_maps=data.participants[0].task_a_maps + 1.), *data.participants[1:])),
            replace(data, observations=(replace(data.observations[0],
                task_b_partitions=data.observations[0].task_b_partitions + 1.), *data.observations[1:])),
            replace(data, task_a_packets=(replace(data.task_a_packets[0],
                task_a_halves=data.task_a_packets[0].task_a_halves + 1.), *data.task_a_packets[1:]))]
        with patch.object(panel, "_fixed_callbacks") as build:
            for index, invalid in enumerate(cases):
                with self.subTest(case=index), self.assertRaises(ValueError):
                    self.run_panel(invalid)
            self.support["world_xyz_mm"][0, 0] += .1
            with self.assertRaisesRegex(ValueError, "coordinates"):
                self.run_panel()
            build.assert_not_called()

    def test_fixed_constructors_share_the_reference_source_and_all_existing_controls(self):
        # Reuse the existing admitted numerical fixture, without a model sweep,
        # response fitting by this driver, or any actual anatomy/response read.
        fixture = basis_fixture.MatchedRandomBasisTests
        fixture.setUpClass()
        checks = []
        shared, calibration, controls = panel._fixed_callbacks(
            fixture.anatomy.coordinates, fixture.anatomy.voxel_ids, lambda: checks.append(None))
        source = shared.__self__
        self.assertIsInstance(source, panel.LocalFunctionalSource)
        for name in ("calibrate_m3", "certify_m3", "predict_m3_geometry"):
            self.assertIs(calibration[name].__self__, source)
        self.assertIsInstance(calibration["predict_m1"], panel.LocalParcelPrediction)
        self.assertEqual(set(calibration["predict_m1"].cells), {4, 8, 16})
        self.assertIs(calibration["predict_m1"].source_recipe, source)
        self.assertIsInstance(calibration["predict_m2"], panel.SmoothRelocationPrediction)
        self.assertEqual(len(calibration["predict_m2"].fields), 4)
        self.assertIs(calibration["predict_m2"].source_recipe, source)
        self.assertIsInstance(calibration["predict_m3_variants"], panel.LocalIsometryVariants)
        self.assertEqual(len(calibration["predict_m3_variants"].operators), 4)
        self.assertIs(calibration["predict_m3_variants"].source_recipe, source)
        self.assertIsInstance(calibration["predict_m4"], panel.PairedStretchVariants)
        self.assertEqual(len(calibration["predict_m4"].specifications), 4)
        self.assertIs(calibration["predict_m4"].reference, source)
        self.assertIsInstance(controls["fit_m0a"], panel.AnatomicalGroupBaseline)
        self.assertIsInstance(controls["predict_random_bases"], panel.MatchedRandomLocalBasis)
        self.assertEqual(len(controls["predict_random_bases"].operators), 8)
        self.assertIs(controls["predict_random_bases"].source_recipe, source)
        self.assertIsInstance(controls["predict_matched_warps"], panel.MatchedRandomWarpPrediction)
        self.assertIs(controls["predict_matched_warps"].predictor, calibration["predict_m2"])
        self.assertTrue(checks)

    def test_owned_views_phase_ordering_no_deletion_expansion_and_failure_propagation(self):
        calibration_names = ("calibrate_m3", "certify_m3", "predict_m3_geometry", "predict_m1",
                             "predict_m2", "predict_m3_variants", "predict_m4")
        shared = object()
        calibration = {name: object() for name in calibration_names}
        controls = {name: object() for name in ("fit_m0a", "predict_random_bases", "predict_matched_warps")}
        references = tuple(object() for _ in DEVELOPMENT_IDS)
        full = SimpleNamespace(folds=tuple(SimpleNamespace(calibration_reference=value) for value in references))
        ordinary, falsifiers = SimpleNamespace(full=full), object()
        points, control_points, alignment = object(), object(), object()
        original_a = self.data.participants[0].task_a_maps.copy()
        original_b = self.data.observations[0].task_b_partitions.copy()
        phases = []
        joined = {}

        def ordinary_stage(records, **kwargs):
            phases.append("ordinary")
            self.assertEqual(tuple(value.participant_id for value in records), DEVELOPMENT_IDS)
            self.assertIs(kwargs["include_participant_deletions"], False)
            for name, value in {"fit_shared": shared, **calibration, **controls}.items():
                self.assertIs(kwargs[name], value)
            self.assertTrue(callable(kwargs["check"]))
            joined["records"] = records
            self.data.participants[0].task_a_maps[:] += 200.
            self.data.observations[0].task_b_partitions[:] += 500.
            np.testing.assert_array_equal(records[0].task_a_maps, original_a)
            np.testing.assert_array_equal(records[0].task_b_partitions, original_b)
            self.assertFalse(records[0].task_a_maps.flags.writeable)
            self.assertFalse(records[0].task_b_partitions.flags.writeable)
            return ordinary

        def control_stage(packets, records, actual_references, **kwargs):
            phases.append("controls")
            self.assertEqual(phases, ["ordinary", "controls"])
            self.assertIs(records, joined["records"])
            self.assertEqual(actual_references, references)
            self.assertIsNone(kwargs["deleted_id"])
            self.assertEqual(tuple(packet.participant_id for packet in packets), DEVELOPMENT_IDS)
            np.testing.assert_array_equal(packets[0].task_a_maps, original_a)
            for name, value in {"fit_shared": shared, **calibration}.items():
                self.assertIs(kwargs[name], value)
            return falsifiers

        def score_control(actual, observations):
            self.assertIs(actual, falsifiers)
            self.assertEqual(phases, ["ordinary", "controls", "points"])
            phases.append("control_points")
            self.assertEqual(tuple(value.participant_id for value in observations), DEVELOPMENT_IDS)
            np.testing.assert_array_equal(observations[0].task_b_partitions, original_b)
            self.assertIs(observations[0].task_b_partitions, joined["records"][0].task_b_partitions)
            return control_points

        def score_ordinary(actual):
            self.assertIs(actual, full)
            self.assertEqual(phases, ["ordinary", "controls"])
            phases.append("points")
            return points

        def score_alignment(actual):
            self.assertIs(actual, full)
            self.assertEqual(phases, ["ordinary", "controls", "points", "control_points"])
            return alignment

        with patch.object(panel, "_fixed_callbacks", return_value=(shared, calibration, controls)), \
             patch.object(panel, "fit_development_lopo", side_effect=ordinary_stage), \
             patch.object(panel, "fit_calibration_falsifiers", side_effect=control_stage) as fit_controls, \
             patch.object(panel, "exploratory_points", side_effect=score_ordinary) as score, \
             patch.object(panel, "exploratory_falsifier_points", side_effect=score_control) as score_falsifiers, \
             patch.object(panel, "m2_alignment_points", side_effect=score_alignment) as score_m2:
            result = self.run_panel()
            self.assertIs(result.ordinary_fit, ordinary)
            self.assertIs(result.ordinary_points, points)
            self.assertIs(result.calibration_falsifier_fit, falsifiers)
            self.assertIs(result.calibration_falsifier_points, control_points)
            self.assertIs(result.m2_alignment_points, alignment)
            self.setUp()
            phases.clear()
            fit_controls.side_effect = TimeoutError("caller budget exhausted")
            for value in (score, score_falsifiers, score_m2):
                value.reset_mock()
            with self.assertRaisesRegex(TimeoutError, "caller budget"):
                self.run_panel()
            for value in (score, score_falsifiers, score_m2):
                value.assert_not_called()
        with patch.object(panel, "_fixed_callbacks") as build:
            def expired():
                raise TimeoutError("deadline before fitting")
            with self.assertRaisesRegex(TimeoutError, "before fitting"):
                self.run_panel(check=expired)
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
