"""Small source-equation checks; no empirical source, fitting or simulation run."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from pinned_critic_core import CriticError, pinned_td_critic


def critic(weights, basis, rewards, **changes):
    parameters = dict(alpha=0.2, gamma=0.5, trace_decay=0.4, stim_cond=0)
    parameters.update(changes)
    return pinned_td_critic(weights, basis, rewards, **parameters)


class PinnedCriticTests(unittest.TestCase):
    def assertArray(self, actual, expected):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_nonunit_gamma_preserves_source_error_instead_of_canonical_td(self):
        # V=2, Vnext=4: source delta=3+.5*(4-2)=4, not canonical delta=3.
        result = critic([2], [[1, 2]], [3, 99], alpha=0.1)
        self.assertArray(result.td_errors, [4])
        self.assertArray(result.final_weights, [2.4])
        self.assertArray(result.values, [2, 4])
        self.assertArray(result.final_eligibility, [0.1])

    def test_sequential_weights_trace_alpha_and_source_value_overwrite(self):
        # Step0: delta=2,e=(.2,0),w=(1.4,2).
        # Step1: current=3.4,next=2,delta=-1.7,e=(.24,.2),w=(.992,1.66).
        result = critic([1, 2], [[1, 1, 0], [0, 1, 1]], [1, -1, 77])
        self.assertArray(result.td_errors, [2, -1.7])
        self.assertArray(result.final_weights, [0.992, 1.66])
        self.assertArray(result.final_eligibility, [0.24, 0.2])
        self.assertArray(result.values, [1, 3.4, 2])
        self.assertArray(result.next_values_before_update, [3, 2])
        self.assertNotEqual(result.values[1], result.next_values_before_update[0])

    def test_eligibility_resets_each_call_while_weights_can_carry_over(self):
        first = critic([0], [[1, 0]], [1, 0], alpha=0.5, gamma=1, trace_decay=1)
        second = critic(first.final_weights, [[1, 0]], [1, 0], alpha=0.5, gamma=1, trace_decay=1)
        self.assertArray(first.final_weights, [0.5])
        self.assertArray(second.final_weights, [0.75])
        self.assertArray(second.final_eligibility, [0.5])
        self.assertArray(first.final_weights, [0.5])

    def test_source_condition_threshold_multiplies_all_processed_rewards(self):
        # Alpha0 holds values at0, isolating the specified reward branch.
        ordinary = critic([0], [[1, 1, 1]], [1, 2, 90], alpha=0, stim_cond=19.5)
        boundary = critic([0], [[1, 1, 1]], [1, 2, 90], alpha=0, stim_cond=20)
        above = critic([0], [[1, 1, 1]], [1, 2, 90], alpha=0, stim_cond=21)
        self.assertArray(ordinary.td_errors, [1, 2])
        self.assertArray(boundary.td_errors, [10, 20])
        self.assertArray(above.td_errors, [10, 20])
        self.assertArray(boundary.final_weights, [0])

    def test_only_t_minus_one_transitions_no_final_reward_or_terminal_override(self):
        first = critic([2], [[1, 2]], [3, 0], alpha=0.1)
        changed_final_reward = critic([2], [[1, 2]], [3, 100000], alpha=0.1)
        self.assertEqual(first.td_errors.shape, (1,))
        self.assertEqual(first.values.shape, (2,))
        for field in ("final_weights", "values", "next_values_before_update", "td_errors", "final_eligibility"):
            np.testing.assert_array_equal(getattr(first, field), getattr(changed_final_reward, field))
        self.assertArray(first.td_errors, [4])  # A forced Vnext=0 would instead give2.

    def test_zero_gamma_and_zero_trace_decay_have_the_declared_source_meaning(self):
        zero_gamma = critic([2], [[1, 2]], [3, 0], alpha=0.1, gamma=0)
        self.assertArray(zero_gamma.td_errors, [3])
        self.assertArray(zero_gamma.final_weights, [2.3])
        no_trace = critic([1, 2], [[1, 1, 0], [0, 1, 1]], [1, -1, 0], trace_decay=0)
        self.assertArray(no_trace.final_eligibility, [0.2, 0.2])
        self.assertArray(no_trace.final_weights, [1.06, 1.66])

    def test_inputs_are_not_mutated_and_returned_weights_do_not_alias_them(self):
        weights = np.array([1.0, 2.0])
        basis = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        rewards = np.array([1.0, -1.0, 77.0])
        before = tuple(value.copy() for value in (weights, basis, rewards))
        result = critic(weights, basis, rewards)
        result.final_weights[0] = -999
        for actual, original in zip((weights, basis, rewards), before):
            np.testing.assert_array_equal(actual, original)

    def test_invalid_dimensions_scalars_and_nonfinite_inputs_refuse(self):
        invalid_inputs = (([], np.empty((0, 2)), [0, 0]),
                          ([0], [[1]], [0]),
                          ([0, 0], [[1, 0]], [0, 0]),
                          ([0], [[1, 0]], [0]),
                          ([np.nan], [[1, 0]], [0, 0]),
                          ([0], [[1, np.inf]], [0, 0]),
                          ([0], [[1, 0]], [0, np.nan]),
                          ([0j], [[1, 0]], [0, 0]))
        for values in invalid_inputs:
            with self.subTest(values=values), self.assertRaises(CriticError):
                critic(*values)
        for parameter in (dict(alpha=-0.1), dict(gamma=1.1), dict(trace_decay=-0.1),
                          dict(stim_cond=np.inf), dict(alpha=True)):
            with self.subTest(parameter=parameter), self.assertRaises(CriticError):
                critic([0], [[1, 0]], [0, 0], **parameter)

    def test_numerical_overflow_refuses_without_clipping_or_input_mutation(self):
        weights = np.array([1e308])
        with self.assertRaises(CriticError):
            critic(weights, [[2, 0]], [1, 0])
        np.testing.assert_array_equal(weights, [1e308])


if __name__ == "__main__":
    unittest.main()
