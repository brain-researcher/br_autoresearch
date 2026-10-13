"""Focused routing/binding checks on the existing fabricated full roster.

No data readers, fits or new numerical qualification. Only crossed_bounds is
replaced to inspect the caller's eight registered family selections; endpoint
composition, participant effects, forecasts and interval decisions are real.
The actual 10,000-draw engine has its separate completed qualification.
Run in the coordinator's substantive Slurm allocation with one BLAS thread.
"""
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

import numpy as np

import endpoint_evaluator as endpoint
import run_development_endpoint_panel as panel
import test_development_endpoints as fixture_source
import uncertainty_core as uncertainty


class DevelopmentEndpointPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, cls.responses, supports, cls.block_order = fixture_source.fixture()
        group = ("sub-02", "right")
        supports[group] = replace(supports[group], beta_strata={
            component: endpoint.InapplicableGain("undefined_calibration_gain")
            for component in ("H_BC", "H_CD")})
        cls.family = endpoint.EndpointFamily(supports)

    @classmethod
    def tearDownClass(cls):
        for name in ("result", "responses", "block_order", "family"):
            delattr(cls, name)

    def test_all_pairs_edges_families_and_inapplicable_forecasts_are_preserved(self):
        called, progress, resources = [], [], []
        actual_points = {}

        def inspect_bounds(statistic, family_name):
            called.append((statistic.pair_id, family_name))
            point = statistic(np.ones(4, dtype=int), np.ones(24, dtype=int))
            actual_points[statistic.pair_id] = point
            indices = [index for index, name in enumerate(statistic.component_ids)
                       if family_name != "F_raw" or name != "D_VNC_normalized"]
            ids = tuple(statistic.component_ids[index] for index in indices)
            scales = tuple(statistic.score_scales[index] for index in indices)
            estimate = point[indices]
            # Invented uncertainty only; the actual point/forecast computation
            # remains untouched. These are not new bootstrap evidence.
            error = np.full(len(indices), .002)
            return uncertainty.SimultaneousBounds(ids, scales, estimate, error, 2.,
                estimate - 2. * error, estimate + 2. * error,
                np.array([uncertainty.MARGINS[scale] for scale in scales]),
                "development", tuple(name for name in statistic.primary_endpoint_ids if name in ids))

        with patch.object(endpoint.EndpointStatistic, "crossed_bounds", inspect_bounds):
            result = panel.evaluate_endpoint_panel(self.result, self.responses, self.family,
                block_order=self.block_order, resource_check=lambda event=None: resources.append(event),
                progress=lambda *event: progress.append(event))

        self.assertEqual(called, [(pair, family) for pair in endpoint.PAIR_SIGNS for family in panel.FAMILIES])
        self.assertEqual(progress[-1], ("pair_complete", 4, 8))
        self.assertGreater(len(resources), 30)
        self.assertTrue(result["complete_endpoint_panel"])
        self.assertFalse(result["initial_coverage_complete"])
        self.assertFalse(result["candidate_selection"])
        self.assertFalse(result["adaptive_search_complete"])
        self.assertFalse(result["audit_access"])
        self.assertFalse(result["secondary_correlation_or_CKA_computed"])
        self.assertEqual(result["frozen_block_order"], self.block_order)
        self.assertNotEqual(self.block_order, self.result.design.block_labels)
        self.assertEqual(result["bootstrap"]["draws"], 10_000)
        self.assertEqual(result["bootstrap"]["seed"], 1729)
        self.assertEqual(result["bootstrap"]["order_statistic_one_based"], 9501)
        self.assertEqual(result["endpoint_family"]["h_component_ids"], [])
        self.assertEqual(tuple(result["pairs"]), tuple(endpoint.PAIR_SIGNS))

        for pair, value in result["pairs"].items():
            self.assertEqual(value["model_order"], endpoint.PAIR_SIGNS[pair])
            self.assertEqual(value["promotion_eligible"], pair != "F1")
            self.assertEqual(tuple(value["edges"]), panel.EDGE_IDS)
            self.assertEqual(tuple(value["forecasts"]), panel.EDGE_IDS)
            np.testing.assert_allclose(value["participant_effects"].mean(axis=0), actual_points[pair])
            intervals = value["interval_families"]
            self.assertEqual(intervals["F_raw"]["component_ids"], endpoint.PRIMARY[:-1])
            self.assertEqual(intervals["F_raw_normalized"]["component_ids"], endpoint.PRIMARY)
            self.assertEqual(intervals["F_raw"]["classification_scope"], "raw_only")
            self.assertEqual(intervals["F_raw_normalized"]["classification_scope"],
                             "complete_eligible_primary_family")
            for index, edge_id in enumerate(panel.EDGE_IDS):
                edge = value["edges"][edge_id]
                self.assertEqual((edge["before"], edge["after"]), endpoint.EDGES[index])
                self.assertEqual(edge["interval_family"],
                    "F_raw_normalized" if index == 4 else "F_raw")
                before, after = (endpoint.PRIMARY.index(name) for name in endpoint.EDGES[index])
                self.assertAlmostEqual(edge["signed_endpoint_change"],
                                       float(actual_points[pair][after] - actual_points[pair][before]))
            self.assertEqual(value["edges"]["A_all_to_A_N"]["role"], "required_size_control")
            for edge_id in ("B_to_C", "C_to_D"):
                forecast = value["forecasts"][edge_id]
                self.assertFalse(forecast["eligible"])
                self.assertEqual(forecast["eligibility_reason"], "calibration_beta_gain_inapplicable")
                self.assertIsNone(forecast["value"])
            self.assertEqual(value["forecasts"]["A_all_to_A_N"]["eligibility_reason"],
                             "size_control_has_no_adopted_explanatory_forecast")
            for edge_id in ("A_N_to_R_N", "raw_to_ceiling_normalized"):
                forecast = value["forecasts"][edge_id]
                self.assertTrue(forecast["eligible"])
                self.assertEqual(forecast["target"], "aggregate_edge_change")
                self.assertAlmostEqual(forecast["value"], value["edges"][edge_id]["signed_endpoint_change"])
                self.assertAlmostEqual(forecast["value"], float(forecast["participant_forecasts"].mean()))
        # Complete scientific output is JSON-compatible without serializing
        # private estimator/support objects or exposing them operationally.
        encoded = json.dumps(panel._jsonable(result), allow_nan=False)
        self.assertLess(len(encoded.encode()), 1 << 20)

    def test_registered_family_record_requires_exact_widths_scales_and_inapplicability(self):
        operator = "/invented/private/calibration_operators/attempt001"
        record = dict(panel._family_description(self.family),
            status="qualified_calibration_endpoint_family", analysis_id="prediction_qualification",
            candidate_scores_read=False, raw_calibration_responses_read=False, audit_values_read=False,
            source_private_operator_attempt=operator)
        expected = panel._bind_family_record(self.family, record, operator)
        self.assertEqual(expected["calibration_inapplicability"], {
            "ceiling_empty_groups": [], "gain_reasons": [
                {"group": ["sub-02", "right"], "component": component,
                 "reason": "undefined_calibration_gain"} for component in ("H_BC", "H_CD")]})
        cases = []
        wrong = deepcopy(record)
        wrong["group_widths"]["sub-01_left"] += 1
        cases.append(wrong)
        wrong = deepcopy(record)
        wrong["score_scales"][-1] = "raw_R2"
        cases.append(wrong)
        wrong = deepcopy(record)
        wrong["calibration_inapplicability"]["gain_reasons"][0]["reason"] = "different_reason"
        cases.append(wrong)
        wrong = deepcopy(record)
        wrong["candidate_scores_read"] = True
        cases.append(wrong)
        for wrong in cases:
            with self.subTest(record=wrong), self.assertRaises(panel.EndpointRunRefusal):
                panel._bind_family_record(self.family, wrong, operator)


if __name__ == "__main__":
    unittest.main()
