"""Known-answer audit gate checks, with fabricated sums and supplied bounds.

These fixtures test adjudication logic, not bootstrap coverage. They reuse the
existing endpoint fixture and supply explicit interval results; no study data,
fit, research simulation, new 10,000-draw suite or actual freeze is performed.
"""
from dataclasses import replace
import unittest

import numpy as np

import audit_adjudicator as core
import endpoint_evaluator as endpoint
import uncertainty_core as uncertainty
from test_endpoint_evaluator import fixture


def statistic_with(change=None):
    supports, errors, _ = fixture("audit")
    if change is not None:
        for person_index, person in enumerate(endpoint.SUBJECTS):
            for roi_index, roi in enumerate(endpoint.ROIS):
                entry = errors[person, roi]
                for stage in endpoint.STAGES:
                    advantage = np.broadcast_to(change(person_index, roi_index, stage,
                                                       np.arange(12)[:, None], np.arange(512)[None, :]),
                                                entry.sst[stage].shape)
                    entry.sse_a[stage][:] = (2. - advantage) * entry.sst[stage]
    return endpoint.EndpointFamily(supports).bind("audit", errors, pair_id="P1")


def relation_for(statistic, kind="contract_robust_relation", edge=None, forecast=None, h_sign=1):
    ids = statistic.component_ids
    primary = statistic.primary_endpoint_ids
    if kind == "contract_robust_relation":
        required = primary
        signs = {key: 1 for key in required}
    else:
        start, end, h, limit = core.EXPLAINED_EDGES[edge]
        required = (start, end) + (() if h is None else (h,))
        signs = {start: 1, end: -1}
        if h is not None:
            signs[h] = h_sign
    return core.FrozenAuditRelation(statistic.pair_id, kind, statistic.family, "F_raw_normalized",
        ids, primary, required, signs,
        {key: uncertainty.MARGINS[scale] for key, scale in zip(ids, statistic.score_scales)},
        {key: core.WIDTHS[scale] for key, scale in zip(ids, statistic.score_scales)}, edge, forecast)


def supplied_bounds(statistic, radius=.001):
    estimate = statistic(np.ones(4, dtype=int), np.ones(12, dtype=int))
    standard_error = np.full(len(estimate), radius, dtype=float)
    margins = np.array([uncertainty.MARGINS[scale] for scale in statistic.score_scales])
    return uncertainty.SimultaneousBounds(statistic.component_ids, statistic.score_scales, estimate,
        standard_error, 1., estimate - standard_error, estimate + standard_error, margins, "audit",
        statistic.primary_endpoint_ids)


def support_reversal(person, roi, stage, block, column):
    if stage == "D":
        return np.where(column % 64 < 32, -.08, .30)
    return .15


class AuditAdjudicatorTests(unittest.TestCase):
    def test_positive_acceptance_does_not_require_narrow_width_or_unselected_H(self):
        statistic = statistic_with()
        result = core.adjudicate_primary_audit(relation_for(statistic), statistic, supplied_bounds(statistic, .03))
        self.assertTrue(result.primary_audit_gates_pass)
        self.assertEqual(result.status, "primary_audit_gates_pass")
        self.assertFalse(result.precision_adequate)
        self.assertEqual(result.required_components, endpoint.PRIMARY)
        self.assertEqual(len(statistic.component_ids), 9)
        self.assertEqual(result.common_passing_participants, endpoint.SUBJECTS)
        self.assertEqual(len(result.participant_component_pass[0]), 7)

    def test_same_three_people_must_pass_every_required_component(self):
        def changes(person, roi, stage, block, column):
            if stage == "B":
                return -.1 if person == 0 else .2
            if stage == "C":
                return -.1 if person == 1 else .2
            return .2
        statistic = statistic_with(changes)
        result = core.adjudicate_primary_audit(relation_for(statistic), statistic, supplied_bounds(statistic))
        per_component = np.array(result.participant_component_pass).sum(axis=0)
        self.assertTrue((per_component >= 3).all())
        self.assertEqual(result.common_passing_participants, ("sub-03", "sub-06"))
        self.assertFalse(result.common_participant_gate_pass)
        self.assertTrue(result.signed_bounds_pass)
        self.assertTrue(result.leave_one_participant_gate_pass)
        self.assertTrue(result.leave_one_block_gate_pass)
        self.assertEqual(result.status, "unresolved_scientific_classification")
        self.assertIn("failed_gate_is_not_an_operationally_defined_stable_heterogeneity_pattern",
                      result.unresolved_requirements)

    def test_leave_out_gates_require_direction_not_margin_and_recompute_blocks(self):
        statistic = statistic_with(lambda person, roi, stage, block, column: -.017 if person == 3 else .021)
        result = core.adjudicate_primary_audit(relation_for(statistic), statistic, supplied_bounds(statistic, .0001))
        self.assertTrue(result.primary_audit_gates_pass)
        self.assertGreater(result.leave_one_participant_effects[0, 0], 0.)
        self.assertLess(result.leave_one_participant_effects[0, 0], .01)

        def block_dependence(person, roi, stage, block, column):
            return np.where(block == 11, .8, -.005) if stage == "B" else .2
        statistic = statistic_with(block_dependence)
        result = core.adjudicate_primary_audit(relation_for(statistic), statistic, supplied_bounds(statistic))
        self.assertTrue(result.signed_bounds_pass)
        self.assertTrue(result.common_participant_gate_pass)
        self.assertFalse(result.leave_one_block_gate_pass)
        self.assertLess(result.leave_one_block_effects[11, 0], 0.)
        self.assertEqual(result.status, "unresolved_scientific_classification")

    def test_support_reversal_keeps_forecast_failure_separate_from_reversal(self):
        statistic = statistic_with(support_reversal)
        forecast = core.FrozenForecast(-.1425, -1, .02)
        relation = relation_for(statistic, "predicted_measurement_reversal", "A_N_to_R_N", forecast)
        bounds = supplied_bounds(statistic)
        passed = core.adjudicate_primary_audit(relation, statistic, bounds)
        self.assertTrue(passed.primary_audit_gates_pass)
        self.assertAlmostEqual(passed.forecast.observed, -.1425, places=12)
        failed = core.adjudicate_primary_audit(replace(relation, forecast=replace(forecast, value=-.18)), statistic, bounds)
        self.assertTrue(failed.reversal_endpoint_gates_pass)
        self.assertFalse(failed.forecast.error_pass)
        self.assertEqual(failed.status, "unexplained_measurement_reversal")
        wrong_sign = core.adjudicate_primary_audit(replace(relation, forecast=replace(forecast, direction=1)), statistic, bounds)
        self.assertTrue(wrong_sign.forecast.error_pass)
        self.assertFalse(wrong_sign.forecast.prediction_direction_pass)
        self.assertFalse(wrong_sign.forecast.observation_direction_pass)
        self.assertEqual(wrong_sign.status, "unexplained_measurement_reversal")

    def test_beta_target_is_block_mean_and_H_is_a_separate_explanation_gate(self):
        def beta(person, roi, stage, block, column):
            if stage == "B":
                return .08
            if stage == "C":
                return -.10 - .001 * block + .08 * (column / 512. - .5)
            return .2
        statistic = statistic_with(beta)
        known_block_mean = -.18 - .001 * 5.5 - .08 / 1024.
        relation = relation_for(statistic, "predicted_measurement_reversal", "B_to_C",
                                core.FrozenForecast(known_block_mean, -1, .02))
        bounds = supplied_bounds(statistic)
        result = core.adjudicate_primary_audit(relation, statistic, bounds)
        self.assertTrue(result.primary_audit_gates_pass)
        self.assertAlmostEqual(result.forecast.observed, known_block_mean, places=12)
        self.assertGreater(abs(result.forecast.observed - (bounds.estimate[1] - bounds.estimate[0])), .001)
        wrong_H = dict(relation.directions)
        wrong_H["H_BC"] = -1
        failed = core.adjudicate_primary_audit(replace(relation, directions=wrong_H), statistic, bounds)
        self.assertTrue(failed.reversal_endpoint_gates_pass)
        self.assertTrue(failed.forecast.passed)
        self.assertFalse(failed.signed_bounds_pass)
        self.assertEqual(failed.status, "unexplained_measurement_reversal")
        selected_H_se = bounds.standard_error.copy()
        selected_H_se[statistic.component_ids.index("H_BC")] = .04
        selected_H_wide = replace(bounds, standard_error=selected_H_se,
                                 lower=bounds.estimate - selected_H_se,
                                 upper=bounds.estimate + selected_H_se)
        wide = core.adjudicate_primary_audit(relation, statistic, selected_H_wide)
        self.assertFalse(wide.precision_adequate)
        self.assertTrue(wide.reversal_endpoint_gates_pass)
        self.assertFalse(wide.primary_audit_gates_pass)
        self.assertEqual(wide.status, "unexplained_measurement_reversal")

    def test_ceiling_forecast_uses_registered_distinct_error_scale(self):
        def ceiling(person, roi, stage, block, column):
            return np.where(column % 4 == 0, -.10, np.where(column % 4 == 2, .16, .05)) if stage == "D" else .2
        statistic = statistic_with(ceiling)
        relation = relation_for(statistic, "predicted_measurement_reversal", "raw_to_ceiling_normalized",
                                core.FrozenForecast(-.10, -1, .05))
        bounds = supplied_bounds(statistic)
        result = core.adjudicate_primary_audit(relation, statistic, bounds)
        self.assertTrue(result.primary_audit_gates_pass)
        self.assertAlmostEqual(result.forecast.observed, -.13, places=12)
        self.assertGreater(result.forecast.absolute_error, .02)
        self.assertLess(result.forecast.absolute_error, .05)
        with self.assertRaisesRegex(core.AdjudicationError, "adopted_forecast_error_limit_changed"):
            core.adjudicate_primary_audit(replace(relation, forecast=replace(relation.forecast, absolute_error_limit=.02)), statistic, bounds)

    def test_precise_negative_patterns_do_not_claim_unqualified_nonreplication(self):
        for effect, field in ((0., "equivalent_required_components"), (-.1, "opposite_required_components")):
            statistic = statistic_with(lambda person, roi, stage, block, column: effect)
            relation = relation_for(statistic)
            result = core.adjudicate_primary_audit(relation, statistic, supplied_bounds(statistic))
            self.assertEqual(result.status, "adequate_negative_pattern_requires_development_qualification")
            self.assertEqual(getattr(result, field), endpoint.PRIMARY)
            self.assertIn("precise_development_evidence_not_adjudicated", result.unresolved_requirements)
            bounds = supplied_bounds(statistic)
            unselected_H_se = bounds.standard_error.copy()
            for key in ("H_BC", "H_CD"):
                unselected_H_se[statistic.component_ids.index(key)] = .04
            unselected_H_wide = replace(bounds, standard_error=unselected_H_se,
                                       lower=bounds.estimate - unselected_H_se,
                                       upper=bounds.estimate + unselected_H_se)
            wide = core.adjudicate_primary_audit(relation, statistic, unselected_H_wide)
            self.assertTrue(wide.precision_adequate)
            self.assertFalse(all(wide.component_precision_adequate))
            self.assertEqual(wide.status, "adequate_negative_pattern_requires_development_qualification")
            self.assertEqual(getattr(wide, field), endpoint.PRIMARY)
            self.assertEqual(len(wide.component_precision_adequate), 9)
        statistic = statistic_with(lambda person, roi, stage, block, column: 0.)
        result = core.adjudicate_primary_audit(relation_for(statistic), statistic, supplied_bounds(statistic, .04))
        self.assertEqual(result.status, "audit_underidentified")
        self.assertFalse(result.precision_adequate)

    def test_changed_lock_thresholds_identity_and_unsupported_relation_refuse(self):
        statistic = statistic_with()
        relation, bounds = relation_for(statistic), supplied_bounds(statistic)
        bad_margin = dict(relation.margins)
        bad_margin["B_Aall_raw"] = .005
        bad_width = dict(relation.maximum_full_widths)
        bad_width["B_Aall_raw"] = .04
        for changed, code in ((replace(relation, pair_id="P2"), "frozen_model_pair_changed"),
                              (replace(relation, margins=bad_margin), "adopted_margins_changed"),
                              (replace(relation, maximum_full_widths=bad_width), "adopted_full_widths_changed"),
                              (replace(relation, inference_family="F_raw"), "frozen_inference_roster_or_scale_changed"),
                              (replace(relation, relation="predicted_measurement_reversal", edge_id="A_all_to_A_N"),
                               "explained_edge_unregistered_or_size_control_forecast_unspecified")):
            with self.assertRaisesRegex(core.AdjudicationError, code):
                core.adjudicate_primary_audit(changed, statistic, bounds)
        different_point = replace(bounds, estimate=bounds.estimate + .1,
                                  lower=bounds.estimate + .1 - bounds.standard_error,
                                  upper=bounds.estimate + .1 + bounds.standard_error)
        with self.assertRaisesRegex(core.AdjudicationError, "bounds_do_not_match_audit_statistic"):
            core.adjudicate_primary_audit(relation, statistic, different_point)
        with self.assertRaisesRegex(core.AdjudicationError, "audit_bounds_required"):
            core.adjudicate_primary_audit(relation, statistic, replace(bounds, phase="development"))


if __name__ == "__main__":
    unittest.main()
