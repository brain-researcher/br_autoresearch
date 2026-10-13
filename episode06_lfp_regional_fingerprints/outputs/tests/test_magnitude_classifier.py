"""Three analytical array-only checks; execute only in coordinated Slurm.

Tiny supplied score fixtures check algebra and label refitting, not neural
models, empirical support, or a synthetic research validation program.
"""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from bandwise_prediction import BANDS, FeatureBlock
from component_scoring import FrozenDesign
from primary_profile import BlockScore, ProfileScore, paired_profile
import profile_classifier as classifiers
import magnitude_classifier as core


BLOCKS = tuple(FeatureBlock(f"band_{i}", (band,)) for i, band in enumerate(BANDS[:3]))
MIHILI = ("M_a", "M_b", "M_c")
CHEWIE = ("C_a", "C_b", "C_c", "C_d")
RECIPE = classifiers.ClassifierRecipe("l2_logistic", 1.)


def check():
    pass


def fixture(animal, session, m1=(-3., -4., -5.), pmd=(-7., -6., -5.)):
    blocks = BLOCKS[:len(m1)]
    design = FrozenDesign(session, "evaluation", tuple(range(8)), tuple(range(8)),
                          tuple(range(8)), (5, 6), ("M1_pc0", "M1_pc1"))
    left = ProfileScore("M1", design, 100., tuple(BlockScore(block, None, value, None)
                                                for block, value in zip(blocks, m1)))
    right = ProfileScore("PMd", replace(design, axis_ids=("PMd_pc0", "PMd_pc1")), 100.,
                          tuple(BlockScore(block, None, value, None) for block, value in zip(blocks, pmd)))
    return classifiers.PairedSessionProfiles(animal, paired_profile(left, right, check=check))


def labels(animal, sessions, swap=False):
    return tuple(classifiers.PairedLabels(animal, session, int(not swap), int(swap)) for session in sessions)


def fit(profiles, swap=False):
    return core.fit_magnitude_classifier(profiles, source_animal="Mihili", expected_sessions=MIHILI,
        labels=labels("Mihili", MIHILI, swap), recipe=RECIPE, check=check)


def predict(fitted, profiles):
    return core.predict_magnitude_classifier(fitted, profiles, target_animal="Chewie-L",
                                             expected_sessions=CHEWIE, check=check)


def reference(mihili, chewie):
    return core.evaluate_magnitude_reference(mihili, chewie, mihili_sessions=MIHILI,
        chewie_sessions=CHEWIE, recipe=RECIPE, check=check)


class MagnitudeClassifierTests(unittest.TestCase):
    def test_analytic_shape_plus_magnitude_logistic_and_frozen_target_shift(self):
        source = tuple(fixture("Mihili", session) for session in MIHILI)
        target = tuple(fixture("Chewie-L", session) for session in CHEWIE)
        fitted = fit(source)
        self.assertEqual(fitted.magnitude_center, -5.)
        self.assertEqual(fitted.magnitude_scale, 1.4826)
        self.assertEqual(fitted.representation, "centered_unit_l2_shape_plus_signed_magnitude")
        self.assertFalse(fitted.coefficients.flags.writeable)
        # Rows are exactly +/-w. At lambda1 the optimum is beta=t*w/||w||,
        # with t=||w||/(1+exp(t*||w||)). This independently detects accidental
        # final unit normalization or unequal shape/magnitude block weights.
        w = np.array((1 / math.sqrt(2), 0., -1 / math.sqrt(2), 1 / 1.4826))
        length = math.sqrt(1 + (1 / 1.4826) ** 2)
        low, high = 0., length
        for _ in range(100):
            middle = (low + high) / 2
            if middle < length / (1 + math.exp(middle * length)):
                low = middle
            else:
                high = middle
        t = (low + high) / 2
        np.testing.assert_allclose(fitted.coefficients, t * w / length, rtol=0, atol=2e-9)
        self.assertAlmostEqual(fitted.intercept, 0., places=9)
        self.assertAlmostEqual(fitted.objective, math.log1p(math.exp(-t * length)) + .5 * t * t, places=10)
        ordinary = predict(fitted, target)
        np.testing.assert_allclose([row.margin for row in ordinary.rows],
                                   (t * length, -t * length) * 4, rtol=0, atol=3e-9)
        # Change every target's mean, then just one target: neither may alter
        # source c/q nor another unchanged target's prediction.
        shifted = tuple(fixture("Chewie-L", session, (-2., -3., -4.), (-6., -5., -4.))
                        for session in CHEWIE)
        offset_prediction = predict(fitted, shifted)
        np.testing.assert_allclose(np.array([row.margin for row in offset_prediction.rows])
                                   - np.array([row.margin for row in ordinary.rows]),
                                   fitted.coefficients[-1] / 1.4826, rtol=0, atol=2e-15)
        mixed = predict(fitted, (shifted[0],) + target[1:])
        self.assertEqual(mixed.rows[2:], ordinary.rows[2:])
        self.assertEqual((fitted.magnitude_center, fitted.magnitude_scale), (-5., 1.4826))
        reversed_fit = fit(source, swap=True)
        np.testing.assert_allclose(reversed_fit.coefficients, -fitted.coefficients, rtol=0, atol=2e-9)
        self.assertEqual((reversed_fit.magnitude_center, reversed_fit.magnitude_scale), (-5., 1.4826))

    def test_singleton_is_explicit_old_shape_stays_unavailable_and_zero_MAD_stops(self):
        source = tuple(fixture("Mihili", session, (1.,), (-1.,)) for session in MIHILI)
        target = tuple(fixture("Chewie-L", session, (0.,), (0.,)) for session in CHEWIE)
        with patch.object(classifiers, "_shape_matrix", side_effect=AssertionError("singleton shape called")):
            fitted = fit(source)
            predictions = predict(fitted, target)
        self.assertEqual(fitted.representation, "singleton_signed_magnitude")
        self.assertEqual(fitted.coefficients.shape, (1,))
        self.assertGreater(fitted.coefficients[0], 0.)
        # Target MAD is zero, but the positive source scale remains frozen.
        # A numerical source intercept need not be bitwise zero; exact ties
        # are instead exercised by the balanced-label source fits below.
        np.testing.assert_allclose([row.margin for row in predictions.rows], 0., rtol=0, atol=1e-9)
        self.assertEqual((fitted.magnitude_center, fitted.magnitude_scale), (0., 1.4826))
        with self.assertRaisesRegex(classifiers.ClassificationUnavailable, "centered_profile_shape"):
            classifiers.fit_profile_classifier(source, source_animal="Mihili", expected_sessions=MIHILI,
                labels=labels("Mihili", MIHILI), recipe=RECIPE, check=check)
        zero_MAD = tuple(fixture("Mihili", session, (.75, .5, .25), (.25, .5, .75))
                         for session in MIHILI)
        with patch.object(classifiers, "_fit_logistic", side_effect=AssertionError("zero-MAD fit reached")):
            with self.assertRaisesRegex(classifiers.ClassificationUnavailable, "source_magnitude_MAD"):
                fit(zero_MAD)
            other = tuple(fixture("Chewie-L", session) for session in CHEWIE)
            unavailable = reference(zero_MAD, other)
        self.assertFalse(unavailable.complete)
        self.assertEqual(unavailable.source_fit_attempts, (1, 0))
        self.assertEqual(unavailable.issues[0].stage, "observed_classifier")
        for direction in unavailable.directions:
            self.assertIsNone(direction.inclusive_upper_tail_fraction)
            self.assertTrue(all(value is None for value in direction.assignment_balanced_accuracies))

    def test_128_label_reference_refits24_source_classifiers_and_preserves_exact_ties(self):
        source = tuple(fixture("Mihili", session, (1.,), (-1.,)) for session in MIHILI)
        target = tuple(fixture("Chewie-L", session, (1.,), (-1.,)) for session in CHEWIE)
        original_fit = core.fit_magnitude_classifier
        with patch.object(core, "fit_magnitude_classifier", wraps=original_fit) as fitted:
            result = reference(source, target)
        self.assertTrue(result.complete)
        self.assertEqual(result.issues, ())
        self.assertEqual(result.source_fit_attempts, (8, 16))
        self.assertEqual(fitted.call_count, 24)
        for animal, count in (("Mihili", 8), ("Chewie-L", 16)):
            assignments = [tuple(label.m1_label for label in call.kwargs["labels"])
                           for call in fitted.call_args_list if call.kwargs["source_animal"] == animal]
            self.assertEqual(len(set(assignments)), count)
        expected = [[], []]
        for number in range(128):
            # With identical +/-magnitudes, prediction follows the source's
            # majority paired label; 2/2 source splits give exact half ties.
            m = sum(bool(number & (1 << bit)) for bit in range(3))
            c = sum(bool(number & (1 << bit)) for bit in range(3, 7))
            expected[0].append((4 - c) / 4 if m < 2 else c / 4)
            expected[1].append(.5 if c == 2 else (3 - m) / 3 if c < 2 else m / 3)
        for direction, wanted, tail in zip(result.directions, expected, (8, 10)):
            self.assertEqual(direction.observed_balanced_accuracy, 1.)
            np.testing.assert_array_equal(direction.assignment_balanced_accuracies, wanted)
            self.assertEqual(direction.inclusive_upper_tail_count, tail)
            self.assertEqual(direction.inclusive_upper_tail_fraction, tail / 128)


if __name__ == "__main__":
    unittest.main()
