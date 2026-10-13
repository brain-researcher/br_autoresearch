"""Hand-computable operator checks only; no provider records, fitting or recovery."""
from pathlib import Path
import math
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from rival_model_core import (
    BasisDefinition, ModelError, behavioral_readout, score_function_credit,
    performance_update, td_update, direct_reinforcement_update,
)


def definition(k=1, endpoint=1):
    return BasisDefinition(tuple(f"supplied_feature_{i}" for i in range(k)),
                           ("supplied_action_a", "supplied_action_b"), endpoint)


def policy(family, **changes):
    arguments = dict(performance_error=-1, baseline=0.5, baseline_weight=0.25,
                     alpha=0.1, beta_da=2, triggered=True,
                     stimulation_multiplier=3, positive_bias=2,
                     amplitude=1, amplitude_threshold=2)
    arguments.update(changes)
    return performance_update(np.zeros((2, 2)), [[1, -2], [3, 4]],
                              family=family, **arguments)


def td(family, **changes):
    arguments = dict(definition=definition(), family=family, action=0,
                     next_action=0 if family == "on_policy_TD_RPE" else None,
                     terminal=False, reward=1, gamma=0.5, alpha=0.1,
                     triggered=False, positive_rpe_bias=3, inverse_temperature=1)
    arguments.update(changes)
    return td_update([[2], [4]], [1], [2], **arguments)


class RivalModelTests(unittest.TestCase):
    def assertArray(self, actual, expected):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_shared_readout_and_action_relabeling_preserve_endpoint_probability(self):
        weights = np.array([[2.0, -1.0], [-1.0, 1.0]])
        original = behavioral_readout(weights, [1, 2], definition=definition(2),
                                      inverse_temperature=2)
        renamed = behavioral_readout(weights[::-1], [1, 2], definition=definition(2, 0),
                                     inverse_temperature=2)
        self.assertArray(original.action_values, [0, 1])
        self.assertEqual(original.endpoint_logit, 2)
        self.assertAlmostEqual(original.action_probabilities[1], 1 / (1 + math.exp(-2)))
        self.assertEqual(renamed.endpoint_logit, original.endpoint_logit)
        self.assertArray(renamed.action_probabilities, original.action_probabilities[::-1])
        self.assertEqual(definition(2).stored_coefficient_count, 4)

    def test_score_function_credit_is_action_specific_and_zero_sum(self):
        credit = score_function_credit(np.zeros((2, 2)), [1, -2],
                                       definition=definition(2), action=0,
                                       inverse_temperature=2)
        self.assertArray(credit, [[1, -2], [-1, 2]])
        self.assertArray(credit.sum(axis=0), [0, 0])

    def test_unsigned_gate_preserves_signed_error_while_bias_can_reverse_it(self):
        rate = policy("adaptive_rate_gate_ACTR")
        constant = policy("constant_rate_direct_policy")
        bias = policy("signed_PE_bias_ACTR")
        credit = np.array([[1, -2], [3, 4]])
        self.assertEqual(rate.error_before_intervention, -1.5)
        self.assertEqual(rate.error_after_intervention, -1.5)
        self.assertEqual(rate.rate_multiplier, 6)
        self.assertArray(rate.weights, -0.9 * credit)
        self.assertArray(constant.weights, -0.15 * credit)
        self.assertEqual(bias.error_after_intervention, 0.5)
        self.assertArray(bias.weights, 0.05 * credit)
        for result in (rate, constant, bias):
            self.assertEqual(result.baseline_after, 0.125)  # No stimulation in baseline.

    def test_hybrid_is_rate_only_below_boundary_and_adds_bias_at_explicit_boundary(self):
        ordinary = policy("amplitude_threshold_hybrid", amplitude=1.999)
        boundary = policy("amplitude_threshold_hybrid", amplitude=2)
        above = policy("amplitude_threshold_hybrid", amplitude=3)
        self.assertArray(ordinary.weights, policy("adaptive_rate_gate_ACTR").weights)
        self.assertArray(boundary.weights, 0.3 * np.array([[1, -2], [3, 4]]))
        self.assertArray(above.weights, boundary.weights)
        no_event = policy("amplitude_threshold_hybrid", amplitude=10, triggered=False)
        self.assertEqual(no_event.error_after_intervention, -1.5)
        self.assertEqual(no_event.rate_multiplier, 2)

    def test_constant_rate_is_nested_no_dopamine_effect(self):
        constant = policy("constant_rate_direct_policy")
        changed = policy("constant_rate_direct_policy", beta_da=9, positive_bias=200,
                         stimulation_multiplier=20, amplitude=30)
        neutral_gate = policy("adaptive_rate_gate_ACTR", beta_da=1,
                              stimulation_multiplier=1)
        self.assertArray(changed.weights, constant.weights)
        self.assertArray(neutral_gate.weights, constant.weights)

    def test_sarsa_uses_actual_next_action_and_max_q_uses_best_preupdate_value(self):
        on_policy = td("on_policy_TD_RPE")
        off_policy = td("off_policy_TD_RPE")
        self.assertEqual(on_policy.bootstrap_value, 4)
        self.assertEqual(off_policy.bootstrap_value, 8)
        self.assertEqual(on_policy.target, 3)
        self.assertEqual(off_policy.target, 5)
        self.assertEqual(on_policy.unperturbed_error, 1)  # Pinned critic would give 2.
        self.assertEqual(off_policy.unperturbed_error, 3)
        self.assertArray(on_policy.weights, [[2.1], [4]])
        self.assertArray(off_policy.weights, [[2.3], [4]])
        self.assertArray(on_policy.readout_before.action_values, [2, 4])
        self.assertArray(off_policy.readout_before.action_values, [2, 4])

    def test_td_event_bias_is_distinct_from_reward_target_and_unsigned_gate(self):
        result = td("on_policy_TD_RPE", triggered=True)
        self.assertEqual(result.target, 3)
        self.assertEqual(result.unperturbed_error, 1)
        self.assertEqual(result.effective_error, 4)
        self.assertArray(result.weights, [[2.4], [4]])

    def test_explicit_terminal_suppresses_bootstrap_and_refuses_ambiguous_next_action(self):
        result = td_update([[2], [4]], [1], None, definition=definition(),
                           family="on_policy_TD_RPE", action=0, next_action=None,
                           terminal=True, reward=1, gamma=0.5, alpha=0.1,
                           triggered=False, positive_rpe_bias=3, inverse_temperature=1)
        self.assertEqual(result.bootstrap_value, 0)
        self.assertEqual(result.target, 1)
        self.assertArray(result.weights, [[1.9], [4]])
        for family, changes in (("on_policy_TD_RPE", {"next_action": None}),
                                ("off_policy_TD_RPE", {"next_action": 0}),
                                ("on_policy_TD_RPE", {"terminal": True, "next_action": None})):
            with self.subTest(family=family, changes=changes), self.assertRaises(ModelError):
                td(family, **changes)

    def test_direct_reinforcement_targets_detected_action_even_when_endpoint_is_other_action(self):
        result = direct_reinforcement_update(
            np.zeros((2, 2)), [1, -2], definition=definition(2), detected_action=0,
            natural_reinforcement=0, triggered=True, stimulation_reinforcement=3,
            alpha=0.5, inverse_temperature=1)
        self.assertArray(result.preferences, [[1.5, -3], [0, 0]])
        self.assertEqual(result.reinforced_action, 0)
        self.assertEqual(result.total_reinforcement, 3)
        after = behavioral_readout(result.preferences, [1, -2], definition=definition(2),
                                   inverse_temperature=1)
        self.assertEqual(result.readout_before.endpoint_logit, 0)
        self.assertEqual(after.endpoint_logit, -7.5)
        self.assertLess(after.action_probabilities[1], 0.5)

    def test_direct_control_can_learn_from_natural_reinforcer_without_stimulation(self):
        initial = np.zeros((2, 1))
        result = direct_reinforcement_update(
            initial, [2], definition=definition(), detected_action=1,
            natural_reinforcement=1, triggered=False, stimulation_reinforcement=100,
            alpha=0.5, inverse_temperature=1)
        self.assertArray(result.preferences, [[0], [1]])
        self.assertEqual(result.total_reinforcement, 1)
        self.assertArray(initial, [[0], [0]])
        self.assertFalse(np.shares_memory(initial, result.preferences))

    def test_nonfinite_update_refuses_without_mutating_input(self):
        initial = np.array([[1.0], [2.0]])
        with self.assertRaises(ModelError):
            direct_reinforcement_update(
                initial, [2], definition=definition(), detected_action=0,
                natural_reinforcement=1e308, triggered=True, stimulation_reinforcement=1e308,
                alpha=1, inverse_temperature=1)
        self.assertArray(initial, [[1], [2]])
        with self.assertRaises(ModelError):
            policy("adaptive_rate_gate_ACTR", beta_da=-1)
        with self.assertRaises(ModelError):
            td("on_policy_TD_RPE", terminal=1)


if __name__ == "__main__":
    unittest.main()
