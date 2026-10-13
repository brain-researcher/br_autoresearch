from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import observed_tail_plan as tail


EPISODE_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = EPISODE_ROOT / "outputs" / "executor" / "OBSERVED_POST36_TAIL_PLAN.json"
POLICY_PATH = EPISODE_ROOT / "SEARCH_POLICY.yaml"


class ObservedTailPlanTests(unittest.TestCase):
    def test_static_artifact_is_the_frozen_generated_plan(self) -> None:
        payload = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        receipt = tail.validate_plan(payload, policy_path=POLICY_PATH)
        self.assertEqual(receipt["slot_count"], 60)
        self.assertEqual(receipt["initial_tail_slot_count"], 16)
        self.assertEqual(receipt["reserve_tail_slot_count"], 44)
        self.assertEqual(receipt["plan_sha256"], tail.EXPECTED_PLAN_SHA256)
        self.assertFalse(receipt["final_connectivity_accessed"])

    def test_exact_policy_arithmetic(self) -> None:
        self.assertEqual(tail.required_falsifier_count(18), 8)
        self.assertEqual(tail.required_falsifier_count(30), 12)
        self.assertEqual(tail.required_falsifier_count(34), 14)
        self.assertEqual(tail.required_falsifier_count(78), 32)
        self.assertEqual(
            tail.minimum_added_falsifiers(
                prior_post_coverage_valid=18,
                prior_counted_falsifiers=0,
            ),
            12,
        )
        arithmetic = tail.build_plan()["policy_arithmetic"]
        self.assertEqual(arithmetic["earliest_fraction_gate_trial_under_initial_tail"], 48)
        self.assertEqual(arithmetic["earliest_all_countable_ids_trial_under_initial_tail"], 45)
        self.assertEqual(
            arithmetic["earliest_qualified_patience_trial_if_no_eligible_improvement"],
            52,
        )

    def test_initial_tail_has_controls_then_two_promotable_challengers(self) -> None:
        plan = tail.build_plan()
        slots = plan["all_slots"][:16]
        self.assertEqual([slot["valid_trial_ordinal"] for slot in slots], list(range(37, 53)))
        self.assertEqual(
            [slot["valid_trial_ordinal"] for slot in slots if slot["promotion_eligible"]],
            [49, 50],
        )
        self.assertEqual(
            [
                slot["valid_trial_ordinal"]
                for slot in slots
                if slot["slot_kind"] == "mandatory_falsifier"
            ],
            list(range(37, 49)) + [51, 52],
        )
        self.assertTrue(
            set(tail.COUNTABLE_OBSERVED_FALSIFIERS).issubset(
                {slot["falsifier_id"] for slot in slots}
            )
        )
        fraction_pass_ordinals = [
            slot["valid_trial_ordinal"]
            for slot in slots
            if slot["cumulative_if_every_prior_slot_is_valid"][
                "falsifier_fraction_gate_passes"
            ]
        ]
        patience_check_ordinals = [
            slot["valid_trial_ordinal"]
            for slot in slots
            if slot["cumulative_if_every_prior_slot_is_valid"][
                "patience_may_be_evaluated"
            ]
        ]
        self.assertEqual(fraction_pass_ordinals[0], 48)
        self.assertEqual(patience_check_ordinals, [52])

    def test_reserves_preserve_fraction_at_every_prefix_through_96(self) -> None:
        plan = tail.build_plan()
        reserve = plan["all_slots"][16:]
        self.assertEqual([slot["valid_trial_ordinal"] for slot in reserve], list(range(53, 97)))
        self.assertEqual(sum(slot["promotion_eligible"] for slot in reserve), 26)
        falsifier_ordinals = [
            slot["valid_trial_ordinal"]
            for slot in reserve
            if slot["counts_toward_post_coverage_falsifier_numerator"]
        ]
        self.assertEqual(
            falsifier_ordinals,
            [54, 56, 59, 61, 64, 66, 69, 71, 74, 76, 79, 81, 84, 86, 89, 91, 94, 96],
        )
        self.assertTrue(
            all(
                slot["cumulative_if_every_prior_slot_is_valid"][
                    "falsifier_fraction_gate_passes"
                ]
                for slot in reserve
            )
        )
        final = reserve[-1]["cumulative_if_every_prior_slot_is_valid"]
        self.assertEqual(final["counted_falsifier_trial_count"], 32)
        self.assertEqual(final["post_coverage_valid_trial_count"], 78)

    def test_no_improvement_stops_at_52_after_all_gates_pass(self) -> None:
        all_ids = set(tail.COUNTABLE_OBSERVED_FALSIFIERS)
        qualified = tail.assess_patience(
            valid_trial_ordinal=52,
            counted_post_coverage_falsifiers=14,
            completed_countable_falsifier_ids=all_ids,
            last_promotion_eligible_improvement_ordinal=36,
        )
        self.assertTrue(qualified["patience_actionable"])

        below_fraction = tail.assess_patience(
            valid_trial_ordinal=52,
            counted_post_coverage_falsifiers=13,
            completed_countable_falsifier_ids=all_ids,
            last_promotion_eligible_improvement_ordinal=36,
        )
        self.assertFalse(below_fraction["patience_actionable"])
        self.assertIn("post_coverage_falsifier_fraction", below_fraction["failed_gates"])

        missing_id = tail.assess_patience(
            valid_trial_ordinal=52,
            counted_post_coverage_falsifiers=14,
            completed_countable_falsifier_ids=set(list(all_ids)[:-1]),
            last_promotion_eligible_improvement_ordinal=36,
        )
        self.assertFalse(missing_id["patience_actionable"])
        self.assertIn("all_countable_falsifier_ids", missing_id["failed_gates"])

        reset = tail.assess_patience(
            valid_trial_ordinal=52,
            counted_post_coverage_falsifiers=14,
            completed_countable_falsifier_ids=all_ids,
            last_promotion_eligible_improvement_ordinal=40,
        )
        self.assertFalse(reset["patience_actionable"])
        self.assertIn(
            "trailing_valid_trials_without_eligible_improvement", reset["failed_gates"]
        )

        early_even_with_old_improvement = tail.assess_patience(
            valid_trial_ordinal=48,
            counted_post_coverage_falsifiers=12,
            completed_countable_falsifier_ids=all_ids,
            last_promotion_eligible_improvement_ordinal=0,
        )
        self.assertFalse(early_even_with_old_improvement["patience_actionable"])
        self.assertFalse(
            early_even_with_old_improvement["valid_trials_since_trial_36_gate"]
        )
        self.assertIn(
            "valid_trials_since_trial_36",
            early_even_with_old_improvement["failed_gates"],
        )

    def test_improvement_at_either_promotable_slot_continues_past_52(self) -> None:
        all_ids = set(tail.COUNTABLE_OBSERVED_FALSIFIERS)
        for improvement_ordinal in (49, 50):
            with self.subTest(improvement_ordinal=improvement_ordinal):
                assessment = tail.assess_patience(
                    valid_trial_ordinal=52,
                    counted_post_coverage_falsifiers=14,
                    completed_countable_falsifier_ids=all_ids,
                    last_promotion_eligible_improvement_ordinal=improvement_ordinal,
                )
                self.assertFalse(assessment["patience_actionable"])
                self.assertIn(
                    "trailing_valid_trials_without_eligible_improvement",
                    assessment["failed_gates"],
                )

    def test_nontrial_policy_gates_cannot_pad_observed_fraction(self) -> None:
        nontrial_ids = {
            gate["falsifier_id"] for gate in tail.NON_TRIAL_REQUIRED_GATES
        }
        self.assertEqual(
            nontrial_ids,
            {
                "fitted_T_U_null_full_revised_search_rerun",
                "forbidden_group_instance_post_result_novelty_audit",
            },
        )
        self.assertTrue(
            all(
                not gate["counts_as_observed_valid_trial"]
                and not gate["counts_toward_post_coverage_falsifier_fraction"]
                for gate in tail.NON_TRIAL_REQUIRED_GATES
            )
        )

    def test_hash_or_flag_mutation_fails_closed(self) -> None:
        payload = tail.build_plan()
        mutated = copy.deepcopy(payload)
        mutated["all_slots"][16]["promotion_eligible"] = False
        with self.assertRaises(tail.TailPlanError):
            tail.validate_plan(mutated, policy_path=None)


if __name__ == "__main__":
    unittest.main()
