"""Synthetic equation and source-variant checks; no empirical inputs."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import actr_port as port


class ACTRPortTests(unittest.TestCase):
    def test_gaussian_source_width_and_normalization(self):
        values = port.gaussian_formula(5, 2, 10)
        self.assertAlmostEqual(values[6] / values[4], math.exp(-0.25), places=14)
        self.assertAlmostEqual(float(np.trapz(values)), 1.0, places=14)
        self.assertEqual(values[2], values[6])

    def test_two_tick_recurrent_and_eligibility_orientation(self):
        net = port.Network(np.array([[0., .3], [.2, 0.]]), np.array([[.5], [.1]]),
                           np.zeros(2), np.array([[False, True], [True, False]]))
        x1 = np.array([.22, .304])
        x2 = np.array([
            x1[0] + (-x1[0] + .3 * math.tanh(x1[1]) + .5) / 25 - .1,
            x1[1] + (-x1[1] + .2 * math.tanh(x1[0]) + .1) / 25 + .05,
        ])
        expected_e = np.array([[math.tanh(x1[i]) * (x2[j] - .67 * x1[j]) ** 3
                               for j in range(2)] for i in range(2)])
        result = port.recurrent_trial(net, np.ones((1, 2)), port.VARIANTS["article_25ms_times2"],
                                      np.array([[.2, -.1], [.3, .05]]))
        np.testing.assert_allclose(result.hidden, np.column_stack([x1, x2]), rtol=0, atol=1e-15)
        np.testing.assert_allclose(result.eligibility[1], expected_e, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(result.eligibility[0], np.zeros((2, 2)))

    def test_unperturbed_branch_does_not_accumulate_eligibility(self):
        net = port.seeded_network(42, units=3)
        result = port.recurrent_trial(net, np.ones((2, 4)), port.VARIANTS["pinned_code_30ms_set4"], None)
        np.testing.assert_array_equal(result.eligibility, np.zeros((4, 3, 3)))

    def test_article_uses_lick_trigger_multiplier_and_cap(self):
        variant = port.VARIANTS["article_25ms_times2"]
        self.assertEqual(port.stimulation(variant, "stimLick_minus", 3., -1., False, 100, 0, .9), (6., True))
        self.assertEqual(port.stimulation(variant, "stimLick_minus", 3., 1., True, 100, 0, .1), (3., False))
        self.assertEqual(port.stimulation(variant, "stimLick_plus", 3., 1., True, 100, 49, .9), (6., True))
        self.assertEqual(port.stimulation(variant, "stimLick_plus", 3., -1., True, 100, 50, .1), (3., False))

    def test_adopted_article_multiplier_uses_current_factor_without_pinned_fallback(self):
        for beta in (1.0, 3.0, np.float64(7.5)):
            self.assertEqual(port.article_stimulation_multiplier(base_beta=beta, triggered=True), 2 * beta)
            self.assertEqual(port.article_stimulation_multiplier(base_beta=beta, triggered=False), beta)
        self.assertEqual(port.article_stimulation_multiplier(base_beta=3.0, triggered=np.bool_(True)), 6.0)
        # Distinguishes multiplication from the pinned code's absolute value4.
        self.assertNotEqual(port.article_stimulation_multiplier(base_beta=3.0, triggered=True), 4.0)

    def test_adopted_article_multiplier_requires_explicit_boolean_trigger(self):
        with self.assertRaises(TypeError):
            port.article_stimulation_multiplier(base_beta=3.0)
        for trigger in (None, 0, 1, "stimLick_plus", np.array([True])):
            with self.subTest(trigger=trigger), self.assertRaises(ValueError):
                port.article_stimulation_multiplier(base_beta=3.0, triggered=trigger)

    def test_adopted_article_multiplier_refuses_invalid_or_overflowing_factors(self):
        for beta in (True, -1.0, 0.5, math.nan, math.inf, "3", 10 ** 400):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                port.article_stimulation_multiplier(base_beta=beta, triggered=True)
        with self.assertRaises(ValueError):
            port.article_stimulation_multiplier(base_beta=sys.float_info.max, triggered=True)
        self.assertEqual(port.article_stimulation_multiplier(base_beta=sys.float_info.max,
                                                            triggered=False), sys.float_info.max)

    def test_pinned_rule_preserves_error_proxy_assignment_and_subtraction(self):
        variant = port.VARIANTS["pinned_code_30ms_set4"]
        self.assertEqual(port.stimulation(variant, "stimLick_minus", 3., 1., True, 100, 100, .59), (4., True))
        self.assertEqual(port.stimulation(variant, "stimLick_minus", 3., 1., False, 100, 0, .6), (2.25, False))
        self.assertEqual(port.stimulation(variant, "stimLick_plus", 3., -1., False, 100, 100, .9), (4., True))
        self.assertEqual(port.stimulation(variant, "control", 3., -1., False, 100, 0, .1), (3., False))
        for arm in ("stimLick_minus", "stimLick_plus"):
            for error in (np.float64(-1), np.float64(1)):
                self.assertIsInstance(port.stimulation(variant, arm, 3., error, False, 100, 0, .9)[1], bool)

    def test_learning_update_has_source_transpose_support_and_sign(self):
        net = port.Network(np.array([[0., .1], [.2, 0.]]), np.zeros((2, 2)), np.zeros(2),
                           np.array([[False, True], [True, False]]))
        baseline = port.learning_step(net, np.array([[1., 2.], [3., 4.]]), .2, .1, 2.)
        np.testing.assert_allclose(net.recurrent, [[0., .0997], [.1998, 0.]], rtol=0, atol=1e-15)
        np.testing.assert_allclose(net.inputs[0], [.01632, .03264], rtol=0, atol=1e-15)
        self.assertAlmostEqual(baseline, .125, places=15)
        with self.assertRaisesRegex(ValueError, "cannot set update sign"):
            port.learning_step(net, np.zeros((2, 2)), .2, .1, -1.)

    def test_plant_bout_offset_and_one_based_lick_times(self):
        licks, state = port.behavioral_plant(np.full(300, 50.), np.random.default_rng(0),
                                             reward=np.full(300, 100.),
                                             transition_uniforms=np.full(300, .5),
                                             bout_offsets=np.array([1]))
        np.testing.assert_array_equal(licks, [103, 253])
        self.assertEqual(state[0], 0)
        np.testing.assert_array_equal(state[1:], np.ones(299))

    def test_seeded_replacement_is_reproducible_and_explicit_variants_differ(self):
        first, second = port.seeded_network(7), port.seeded_network(7)
        np.testing.assert_array_equal(first.recurrent, second.recurrent)
        np.testing.assert_array_equal(first.inputs, second.inputs)
        self.assertEqual(first.recurrent.shape, (50, 50))
        self.assertEqual(port.VARIANTS["article_25ms_times2"].tau_ms, 25.)
        self.assertEqual(port.VARIANTS["pinned_code_30ms_set4"].tau_ms, 30.)


if __name__ == "__main__":
    unittest.main()
