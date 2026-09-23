#!/usr/bin/env python3
"""Mechanical checks for the EP02 adaptive-search policy."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

try:
    import jsonschema
except ModuleNotFoundError:  # Optional on the lightweight login-node Python.
    jsonschema = None


EPISODE = Path(__file__).resolve().parents[2]
ROOT = EPISODE.parent
POLICY_PATH = EPISODE / "SEARCH_POLICY.json"
SCHEMA_PATH = ROOT / "SEARCH_POLICY.schema.json"


class SearchPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = json.loads(POLICY_PATH.read_text())
        cls.schema = json.loads(SCHEMA_PATH.read_text())

    @unittest.skipIf(jsonschema is None, "jsonschema is not installed in this Python")
    def test_draft_2020_12_schema(self) -> None:
        jsonschema.Draft202012Validator.check_schema(self.schema)
        jsonschema.Draft202012Validator(self.schema).validate(self.policy)

    def test_schema_required_keys_and_constants_without_optional_dependency(self) -> None:
        for key in self.schema["required"]:
            self.assertIn(key, self.policy)
        for key in ("schema_version", "common_protocol_ref", "evidence_mode"):
            self.assertEqual(self.policy[key], self.schema["properties"][key]["const"])
        status_schema = self.schema["properties"]["status"]
        if "const" in status_schema:
            self.assertEqual(self.policy["status"], status_schema["const"])
        else:
            self.assertIn(self.policy["status"], status_schema["enum"])
        self.assertEqual(self.policy["status"], "planned_unregistered")
        self.assertEqual(self.policy["episode_id"], "ep02")

    def test_roles_are_disjoint_and_cover_all_source_records(self) -> None:
        roles = self.policy["dataset_roles"]
        groups = [
            roles["development_full"]["source_records_1_based"],
            roles["primary_audit_evaluator_only"]["arms"]["stimLick_minus"]["source_records_1_based"],
            roles["primary_audit_evaluator_only"]["arms"]["stimLick_plus"]["source_records_1_based"],
            roles["post_primary_boundary_evaluator_only"]["source_records_1_based"],
        ]
        flat = [record for group in groups for record in group]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(sorted(flat), list(range(1, 25)))
        self.assertEqual([len(group) for group in groups], [9, 6, 5, 4])

    def test_primary_conditional_label_count_is_not_design_space(self) -> None:
        primary = self.policy["dataset_roles"]["primary_audit_evaluator_only"]
        self.assertEqual(primary["assignment_count"], math.comb(11, 6))
        self.assertEqual(primary["assignment_count"], 462)
        self.assertFalse(
            self.policy["audit"]["inference"][
                "all_462_allocations_are_the_documented_randomization_space"
            ]
        )

    def test_endpoint_calibration_is_runtime_only_and_unbound(self) -> None:
        calibration = self.policy["small_n_calibration"]
        self.assertEqual(calibration["status"], "endpoint_faithful_binding_calibration_required")
        self.assertIsNone(calibration["tracked_result_artifact"])
        self.assertTrue((EPISODE / calibration["implementation_ref"]).is_file())
        self.assertTrue((EPISODE / calibration["test_ref"]).is_file())
        self.assertIsNone(calibration["binding_rule"])
        self.assertFalse(calibration["positive_audit_opening_authorized"])
        self.assertFalse(calibration["positive_mechanistic_terminal_authorized"])
        self.assertEqual(calibration["design"]["result_storage"], "runtime_only_not_tracked")

    def test_candidate_mounts_exclude_outcome_audit_and_mixed_source(self) -> None:
        roles = self.policy["dataset_roles"]
        self.assertEqual(
            roles["runtime_role_handoff"]["allowed_candidate_role_ids"],
            ["development_full", "audit_structural"],
        )
        self.assertIsNone(roles["primary_audit_evaluator_only"]["candidate_mount"])
        self.assertIsNone(roles["post_primary_boundary_evaluator_only"]["candidate_mount"])
        self.assertEqual(roles["trusted_mixed_source"]["candidate_access"], "prohibited")
        handoff = roles["runtime_role_handoff"]
        self.assertIsNone(handoff["tracked_handoff_artifact"])
        self.assertFalse(handoff["candidate_may_read_steward_or_evaluator_paths"])
        self.assertFalse(handoff["launch_authorized"])
        self.assertTrue((EPISODE / handoff["firewall_contract_ref"]).is_file())

    def test_firewall_contract_requires_external_provisioning(self) -> None:
        handoff = self.policy["dataset_roles"]["runtime_role_handoff"]
        contract = json.loads((EPISODE / handoff["firewall_contract_ref"]).read_text())
        self.assertEqual(contract["current_environment_status"], "BLOCKED_PENDING_EXTERNAL_PROVISIONING")
        self.assertFalse(contract["same_uid_controls_are_sufficient"])
        self.assertIn("atomic_no_partial_output", contract["required_checks"])

    def test_mouse_is_only_inferential_and_resampling_unit(self) -> None:
        scope = self.policy["scientific_scope"]
        audit = self.policy["audit"]
        self.assertEqual(scope["primary_inferential_unit"], "mouse")
        self.assertFalse(scope["trial_level_pseudoreplication_allowed"])
        self.assertTrue(audit["inference"]["trials_sessions_and_seeds_are_not_resampling_units"])

    def test_primary_score_has_no_derived_fallback_or_interpolation(self) -> None:
        contract = self.policy["coordinate_and_outcome_contract"]
        score = contract["primary_mouse_score"]
        self.assertEqual(score["provider_derived_feature_fallback"], "prohibited")
        self.assertEqual(contract["missingness"]["primary_interpolation"], "prohibited")
        self.assertEqual(score["early_epoch"], "session_1")
        self.assertEqual(score["late_epoch"], "session_8")
        self.assertIn("all_eleven_primary", score["prelock_structural_support_gate"])
        self.assertIn("never_a_postlock_mouse_exclusion", score["missing_epoch_rule"])
        exposure = contract["superseded_phase0_v1_prelaunch_exposure"]
        self.assertTrue(exposure["values_treated_as_exposed_regardless_of_whether_viewed"])
        self.assertFalse(exposure["current_v2_packet_emits_these_fields"])
        self.assertEqual(
            set(exposure["permanent_exclusions"]),
            {"candidate_scoring", "model_selection", "audit_terminal", "boundary_interpretation"},
        )
        self.assertIn(
            "permanently_excluded",
            contract["permanently_excluded_diagnostics"]["cumulative_stimulated_trial_count"],
        )

    def test_search_depth_and_stage_arithmetic(self) -> None:
        budgets = self.policy["budgets"]
        ranges = list(budgets["stage_valid_trial_ranges"].values())
        self.assertEqual(sum(bounds[0] for bounds in ranges), budgets["minimum_valid_trials"])
        self.assertEqual(sum(bounds[1] for bounds in ranges), budgets["maximum_valid_trials"])
        self.assertGreaterEqual(self.policy["promotion"]["post_coverage_falsifier_fraction_min"], 0.4)
        self.assertGreaterEqual(self.policy["promotion"]["minimum_adaptive_successor_cycles"], 2)
        self.assertGreaterEqual(self.policy["promotion"]["minimum_incumbent_challenger_decisions"], 2)

    def test_primary_audit_is_one_shot_and_nonadaptive(self) -> None:
        audit = self.policy["audit"]
        self.assertTrue(audit["configuration_lock_required"])
        self.assertTrue(audit["lock_hash_required"])
        self.assertEqual(audit["maximum_open_count"], 1)
        self.assertTrue(audit["logical_opening_includes_optional_post_primary_boundary_stage"])
        self.assertEqual(audit["maximum_evaluator_payload_roles_within_the_one_opening"], 2)
        self.assertEqual(self.policy["budgets"]["audit_openings_max"], 1)
        self.assertTrue(
            self.policy["budgets"]["post_primary_boundary_is_an_internal_stage_not_a_second_opening"]
        )
        self.assertFalse(audit["audit_updates_search"])
        self.assertTrue(audit["all_eleven_mice_open_and_score_atomically"])
        self.assertFalse(audit["current_opening_authorized"])
        self.assertFalse(audit["current_positive_mechanistic_terminal_authorized"])
        self.assertNotIn("terminal_success_requires", audit)

    def test_permutation_inference_is_not_overclaimed(self) -> None:
        inference = self.policy["audit"]["inference"]
        self.assertTrue(inference["randomization_mechanism_receipt_required"])
        self.assertFalse(inference["all_462_allocations_are_the_documented_randomization_space"])
        self.assertFalse(inference["unblocked_462_sensitivity_is_intention_to_treat"])
        self.assertIn("2_to_4_mouse_cohorts", inference["current_documentary_support"])
        self.assertIn("for_every_mouse", inference["sharp_zero_null"])
        self.assertFalse(inference["weak_average_effect_null_claimed_exactly_tested"])
        self.assertFalse(inference["population_average_lower_bound_claim_allowed"])
        self.assertFalse(inference["arm_guards_establish_bidirectional_causal_effects"])
        contrast = self.policy["coordinate_and_outcome_contract"]["primary_contrast"]
        self.assertEqual(contrast["estimand_status"], "not_identified_from_the_current_release")
        self.assertIn("not_an_intention_to_treat", contrast["current_release_only_contrast"])
        outputs = self.policy["audit"]["output_packet"]
        self.assertIn("positive_direction_p_zero_and_p_margin", outputs)
        self.assertIn("sign_reflected_negative_direction_p_zero_and_p_margin", outputs)

    def test_scale_and_endpoint_calibration_blockers_are_explicit(self) -> None:
        score = self.policy["coordinate_and_outcome_contract"]["primary_mouse_score"]
        self.assertIsNone(score["development_scale_minimum_raw_probability_gain"])
        self.assertIsNone(score["minimum_absolute_raw_between_arm_gain"])
        recalibration = self.policy["small_n_calibration"]["endpoint_faithful_recalibration_required_after_score_receipts"]
        self.assertEqual(recalibration["status"], "launch_blocking")
        self.assertFalse(recalibration["generic_gaussian_rule_alone_may_authorize_audit"])

    def test_boundary_never_rescues_primary(self) -> None:
        boundary = self.policy["boundary_diagnostic"]
        self.assertEqual(boundary["separate_opening_count_max"], 0)
        self.assertEqual(boundary["payload_access_count_max_within_primary_logical_opening"], 1)
        self.assertTrue(boundary["same_permission_separated_evaluator_transaction_as_primary"])
        self.assertTrue(boundary["primary_subpacket_committed_before_boundary_payload_access"])
        self.assertFalse(boundary["primary_feedback_released_before_boundary_payload_access"])
        self.assertIn(
            "primary_audit_terminal_subpacket_committed_inside_the_same_evaluator_transaction",
            boundary["opening_prerequisites"],
        )
        self.assertFalse(boundary["may_change_primary_terminal_mapping"])
        self.assertFalse(boundary["may_rescue_failed_or_unresolved_primary"])
        self.assertIn(
            "optional_post_primary_boundary_diagnostic_state_bound_to_the_primary_subpacket_hash",
            self.policy["audit"]["output_packet"],
        )

    def test_every_terminal_class_is_mapped(self) -> None:
        declared = set(self.policy["stop"]["valid_terminal_classes"])
        mapped = set(self.policy["terminal_mapping"])
        self.assertEqual(declared, mapped)

    def test_launch_remains_blocked_for_physical_and_canonical_reasons(self) -> None:
        binding = self.policy["canonical_binding"]
        self.assertTrue(binding["launch_blocked"])
        self.assertIsNone(binding["canonical_program_id"])
        blockers = " ".join(binding["unresolved_launch_blockers"])
        self.assertIn("same_uid", blockers)
        self.assertIn("canonical", blockers)
        self.assertIn("scientist_signoff", blockers)


if __name__ == "__main__":
    unittest.main()
