"""Focused tests of the contract equations; no source signals or margins."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import forecast_components as component


class ForecastComponentTests(unittest.TestCase):
    def test_fir_future_mutation_chunking_warmup_and_gap_reset(self):
        stream = np.arange(40., dtype=float).reshape(20, 2)
        coefficients = [.5, .3, .2]
        def execute(data, chunks):
            filter_ = component.CausalFIR(coefficients, 2)
            outputs, index = [], 0
            for size in chunks:
                for row in data[index:index+size]:
                    outputs.append(filter_.update(row))
                index += size
            return outputs
        single = execute(stream, [1] * 20)
        chunked = execute(stream, [3, 1, 9, 7])
        self.assertIsNone(single[0])
        self.assertIsNone(single[1])
        np.testing.assert_allclose(single[2:], chunked[2:], atol=0, rtol=0)
        changed = stream.copy()
        changed[10:] = 1e6
        mutated = execute(changed, [20])
        np.testing.assert_array_equal(single[2:10], mutated[2:10])
        filter_ = component.CausalFIR(coefficients, 2)
        for row in stream[:4]:
            filter_.update(row)
        self.assertIsNone(filter_.update([np.nan, 1]))
        self.assertIsNone(filter_.update([1, 2]))
        self.assertIsNone(filter_.update([3, 4]))
        np.testing.assert_allclose(filter_.update([5, 6]), [.5*5+.3*3+.2, .5*6+.3*4+.2*2])

    def test_factorization_reconstructs_voltage_and_excludes_future(self):
        raw = np.arange(60.).reshape(20, 3) / 7
        mean, slow_mean, scale = np.array([1., 2., 3.]), np.array([.2, .3, .4]), np.array([2., 3., 4.])
        args = ([.5, .3, .2], 2, 10, 15, mean, slow_mean, scale)
        blocks = component.factorized_blocks(raw, *args)
        np.testing.assert_allclose(blocks.reconstruct(True, True), (raw[2:15] - mean) / scale, atol=1e-14)
        changed = raw.copy()
        changed[15:] = 1e9
        future = component.factorized_blocks(changed, *args)
        np.testing.assert_array_equal(blocks.reconstruct(True, True), future.reconstruct(True, True))
        with self.assertRaises(ValueError):
            component.factorized_blocks(raw, [.5, .3, .2], 1, 10, 15, mean, slow_mean, scale)

    def test_both_forbidden_block_mutations_at_fixed_slots(self):
        raw = np.arange(60.).reshape(20, 3) / 7
        args = ([.5, .3, .2], 2, 10, 15, np.zeros(3), np.zeros(3), np.ones(3))
        blocks = component.factorized_blocks(raw, *args)
        changed = raw.copy()
        changed[10:15] = -1000
        changed_blocks = component.factorized_blocks(changed, *args)
        for older in (False, True):
            np.testing.assert_array_equal(blocks.reconstruct(False, older), changed_blocks.reconstruct(False, older))
        replaced_slow = component.FactorizedBlocks(blocks.complement, np.full_like(blocks.older_slow, 1e8), blocks.recent)
        for recent in (False, True):
            np.testing.assert_array_equal(blocks.reconstruct(recent, False), replaced_slow.reconstruct(recent, False))
        self.assertTrue(np.any(blocks.reconstruct(False, False) != 0))

    def test_first_bin_event_is_distinct_from_zero_bin_censoring(self):
        logp = component.calibrated_log_probabilities(np.zeros((4, 19)), [1, 0, 0, 0])
        loss, gradient = component.censoring_loss_and_logit_gradient(logp, [0, -1, -1, 18], [0, 0, 6, 18])
        np.testing.assert_allclose(loss, [np.log(19), 0, np.log(19/13), np.log(19)], atol=1e-14)
        np.testing.assert_array_equal(gradient[1], np.zeros(19))
        self.assertLess(gradient[0, 0], 0)
        np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=1e-14)

    def test_four_parameter_gradient_matches_independent_central_difference(self):
        rng = np.random.Generator(np.random.PCG64(20261003))
        logits = rng.normal(size=(8, 19))
        categories = np.array([0, 6, 17, 18, -1, -1, -1, 4])
        complete = np.array([0, 0, 0, 18, 0, 6, 11, 0])
        weights = np.array([1., 2., 1., 1., 1., 3., 1., 1.])
        parameters = np.array([.8, .2, -.3, .1])
        _, analytic = component.calibration_objective(parameters, logits, categories, complete, weights)
        numerical = []
        for column in range(4):
            direction = np.zeros(4)
            direction[column] = 1e-6
            plus = component.calibration_objective(parameters + direction, logits, categories, complete, weights)[0]
            minus = component.calibration_objective(parameters - direction, logits, categories, complete, weights)[0]
            numerical.append((plus - minus) / 2e-6)
        np.testing.assert_allclose(analytic, numerical, atol=2e-8, rtol=2e-8)

    def test_calibration_probabilities_and_fixed_coarse_offsets(self):
        logp = component.calibrated_log_probabilities(np.zeros((1, 19)), [1.3, .2, -.4, .7])
        probabilities = np.exp(logp[0])
        self.assertAlmostEqual(probabilities.sum(), 1., places=14)
        self.assertAlmostEqual(probabilities[0] / probabilities[18], np.exp(.2), places=14)
        self.assertAlmostEqual(probabilities[6] / probabilities[18], np.exp(-.4), places=14)
        self.assertAlmostEqual(probabilities[12] / probabilities[18], np.exp(.7), places=14)
        self.assertEqual(len(probabilities), 19)

    def test_calibration_optimizes_existing_recipe_without_mutating_logits(self):
        rng = np.random.Generator(np.random.PCG64(83))
        logits = rng.normal(size=(76, 19))
        original = logits.copy()
        categories = np.tile(np.arange(19), 4)
        result = component.fit_four_parameter_calibration(logits, categories, np.full(76, 18))
        self.assertTrue(result["accepted"])
        self.assertLessEqual(result["selected_objective"], result["identity_objective"] + 1e-10)
        self.assertFalse(result["recipe_frozen_for_empirical_use"])
        np.testing.assert_array_equal(logits, original)

    def test_failed_optimizer_retains_identity_without_dropping_packet(self):
        failure = SimpleNamespace(success=False, x=np.array([2., 1., 1., 1.]), fun=0., message="synthetic failure", nit=200)
        with patch.object(component, "minimize", return_value=failure):
            result = component.fit_four_parameter_calibration(np.zeros((2, 19)), [0, -1], [0, 6])
        self.assertTrue(result["identity_fallback"])
        self.assertEqual(result["parameters"], [1., 0., 0., 0.])


if __name__ == "__main__":
    unittest.main()
