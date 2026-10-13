"""Two independent primary-profile checks; run only in coordinated Slurm.

Fabricated bandwise fixtures exercise the selected arithmetic/identity binding,
not source qualification, transfer accuracy or permission to open neural data.
No imports or tests were executed while authoring these files.
"""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bandwise_prediction as prediction
import primary_profile as core
from test_bandwise_prediction import fixture, recipe, ForbiddenArray


def check():
    pass


def case(region="M1", shift=7):
    lfp, observed, _, _ = fixture()
    fitted = prediction.fit_bandwise(replace(lfp, region=region), observed,
                                     recipe=recipe(), check=check)
    evaluation, target, signal, _ = fixture(role="evaluation", id_start=101,
                                           signal_multiplier=3, observation_shift=shift)
    # Three additional distinct whole trials repeat one first-direction signal.
    # This makes uniform-trial and direction-balanced scores distinguishable.
    indices = np.asarray(tuple(range(16)) + (0, 0, 0))
    design = replace(evaluation.design,
        trial_ids=tuple(f"trial_{201 + index:06d}" for index in range(len(indices))),
        trial_directions=tuple(evaluation.design.trial_directions[index] for index in indices))
    evaluation = replace(evaluation, region=region, design=design, values=evaluation.values[indices])
    target = replace(target, design=design, values=target.values[indices])
    predicted = prediction.predict_bandwise(fitted, evaluation, check=check)
    return fitted, target, predicted, signal[indices]


def oracle(signal, shift):
    # Independent known [1,2]/sqrt(5) population axis and k/(k+.5) ridge gains.
    truth = [math.sqrt(5) * (3 * float(value) + shift) for value in signal.flat]
    energy = math.fsum(value * value for value in truth)
    errors = []
    for k in (1, 2, 0, 8):
        predicted = [math.sqrt(5) * 3 * float(value) * k / (k + .5) for value in signal.flat]
        errors.append(math.fsum((actual - estimate)**2 for actual, estimate in zip(truth, predicted)))
    return energy, tuple(errors), tuple(1 - error / energy for error in errors)


class PrimaryProfileTests(unittest.TestCase):
    def test_pooled_energy_preserves_mean_shift_and_uniform_rows_with_explicit_unavailable_results(self):
        fitted, observed, predictions, signal = case()
        actual = core.score_band_profile(fitted, observed, predictions, check=check)
        energy, errors, scores = oracle(signal, 7)
        self.assertTrue(actual.available)
        self.assertEqual(actual.design, predictions.design)
        self.assertEqual(tuple(value.block for value in actual.blocks), predictions.blocks)
        np.testing.assert_allclose(actual.residual_energy, energy, rtol=3e-14, atol=1e-9)
        np.testing.assert_allclose([value.squared_error for value in actual.blocks], errors,
                                   rtol=3e-13, atol=1e-8)
        np.testing.assert_allclose([value.score for value in actual.blocks], scores, rtol=0, atol=2e-12)
        truth = math.sqrt(5) * (3 * signal + 7)
        centered_energy = math.fsum((float(value) - float(np.mean(truth)))**2 for value in truth.flat)
        self.assertGreater(abs(scores[0] - (1 - errors[0] / centered_energy)), .01)
        directions = np.asarray(observed.design.trial_directions)
        error = truth - math.sqrt(5) * 3 * signal / 1.5
        balanced_energy = math.fsum(float(np.mean(truth[directions == direction]**2))
                                   for direction in range(8)) / 8
        balanced_error = math.fsum(float(np.mean(error[directions == direction]**2))
                                  for direction in range(8)) / 8
        self.assertGreater(abs(scores[0] - (1 - balanced_error / balanced_energy)), 1e-4)
        # Negative values stay visible; no clipping or coordinatewise recovery.
        wrong = replace(predictions.predictions[0], values=np.full(predictions.design.shape, 1000.))
        negative = core.score_band_profile(fitted, observed,
            replace(predictions, predictions=(wrong,) + predictions.predictions[1:]), check=check)
        negative_error = math.fsum((float(value) - 1000.)**2 for value in truth.flat)
        self.assertLess(negative.blocks[0].score, 0.)
        self.assertAlmostEqual(negative.blocks[0].score, 1 - negative_error / energy, places=10)
        # Zero residual energy is undefined even when nonzero predictions exist.
        mean_counts = np.stack([fitted.training_template[direction]
                                for direction in observed.design.trial_directions])
        zero = core.score_band_profile(fitted, replace(observed, values=mean_counts), predictions, check=check)
        self.assertEqual(zero.residual_energy, 0.)
        self.assertFalse(zero.available)
        self.assertTrue(all(value.score is None and value.unavailable_reason == "zero_residual_energy"
                            for value in zero.blocks))
        huge = core.score_band_profile(fitted,
            replace(observed, values=np.full(observed.design.shape, 1e200)), predictions, check=check)
        self.assertIsNone(huge.residual_energy)
        self.assertTrue(all(value.unavailable_reason == "nonfinite_residual_energy" for value in huge.blocks))
        bad = replace(predictions.predictions[0], values=np.full(predictions.design.shape, np.nan))
        missing = core.score_band_profile(fitted, observed,
            replace(predictions, predictions=(bad,) + predictions.predictions[1:]), check=check)
        self.assertEqual(len(missing.blocks), len(predictions.blocks))
        self.assertFalse(missing.available)
        self.assertIsNone(missing.blocks[0].score)
        self.assertTrue(all(value.available for value in missing.blocks[1:]))

    def test_exact_paired_support_rank_and_physical_block_identities_before_values(self):
        fitted, observed, predictions, signal = case()
        m1 = core.score_band_profile(fitted, observed, predictions, check=check)
        pmd_fit, pmd_y, pmd_predictions, _ = case("PMd", shift=11)
        pmd = core.score_band_profile(pmd_fit, pmd_y, pmd_predictions, check=check)
        actual = core.paired_profile(m1, pmd, check=check)
        self.assertNotEqual(m1.design.axis_ids, pmd.design.axis_ids)
        self.assertTrue(actual.available)
        self.assertEqual(actual.blocks, predictions.blocks)
        expected = tuple(a - b for a, b in zip(oracle(signal, 7)[2], oracle(signal, 11)[2]))
        np.testing.assert_allclose(actual.m1_minus_pmd, expected, rtol=0, atol=3e-12)
        wrong_block = replace(pmd.blocks[0], block=prediction.FeatureBlock("lmp", prediction.BANDS[1:2]))
        for invalid in (
            replace(pmd, design=replace(pmd.design, session_id="other_session")),
            replace(pmd, design=replace(pmd.design, trial_ids=pmd.design.trial_ids[::-1])),
            replace(pmd, design=replace(pmd.design, trial_directions=pmd.design.trial_directions[::-1])),
            replace(pmd, design=replace(pmd.design, directions=pmd.design.directions[::-1])),
            replace(pmd, design=replace(pmd.design, time_ids=pmd.design.time_ids[::-1])),
            replace(pmd, design=replace(pmd.design, axis_ids=pmd.design.axis_ids + ("extra_pc",))),
            replace(pmd, blocks=pmd.blocks[::-1]),
            replace(pmd, blocks=(wrong_block,) + pmd.blocks[1:]),
            replace(pmd, blocks=pmd.blocks[:-1]),
        ):
            with self.assertRaises(ValueError):
                core.paired_profile(m1, invalid, check=check)
        with self.assertRaises(ValueError):
            core.paired_profile(pmd, m1, check=check)
        with self.assertRaises(PermissionError):
            core.paired_profile(m1, replace(pmd, design=replace(pmd.design, role="calibration")), check=check)
        missing = replace(pmd, blocks=(replace(pmd.blocks[0], score=None,
            unavailable_reason="nonfinite_prediction_error_or_score"),) + pmd.blocks[1:])
        paired_missing = core.paired_profile(m1, missing, check=check)
        self.assertFalse(paired_missing.available)
        self.assertIsNone(paired_missing.m1_minus_pmd[0])
        self.assertEqual(len(paired_missing.m1_minus_pmd), len(predictions.blocks))
        np.testing.assert_array_equal(paired_missing.m1_minus_pmd[1:], actual.m1_minus_pmd[1:])
        # Even a mismatch in the final block binds before any supplied values.
        unread = replace(observed, values=ForbiddenArray())
        guarded = tuple(replace(value, values=ForbiddenArray()) for value in predictions.predictions)
        wrong_last = replace(guarded[-1], design=replace(predictions.design, axis_ids=("wrong_pc",)))
        with self.assertRaises(ValueError):
            core.score_band_profile(fitted, unread, replace(predictions,
                predictions=guarded[:-1] + (wrong_last,)), check=check)
        with self.assertRaises(PermissionError):
            core.score_band_profile(fitted, replace(unread,
                design=replace(unread.design, role="calibration")), predictions, check=check)


if __name__ == "__main__":
    unittest.main()
