"""Two development-reference arithmetic/access checks, coordinated Slurm only.

Tiny already-computed profile fixtures verify the128-label combinatorics and
signed means. They contain no neural arrays or empirical outcome access and
do not qualify an experimental candidate or run a simulation research program.
"""
from collections import Counter
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
import profile_references as core


BLOCKS = tuple(FeatureBlock(f"band_{i}", (band,)) for i, band in enumerate(BANDS[:3]))
MIHILI = ("M_a", "M_b", "M_c")
CHEWIE = ("C_a", "C_b", "C_c", "C_d")
RECIPE = classifiers.ClassifierRecipe("l2_logistic", .5)


def check():
    pass


def fixture(animal, session, m1=(.75, .5, .25), pmd=(.25, .5, .75)):
    design = FrozenDesign(session, "evaluation", tuple(range(8)), tuple(range(8)),
                          tuple(range(8)), (5, 6), ("M1_pc0", "M1_pc1"))
    left = ProfileScore("M1", design, 100., tuple(BlockScore(block, None, value, None)
                                                for block, value in zip(BLOCKS, m1)))
    right = ProfileScore("PMd", replace(design, axis_ids=("PMd_pc0", "PMd_pc1")), 100.,
                          tuple(BlockScore(block, None, value, None) for block, value in zip(BLOCKS, pmd)))
    return classifiers.PairedSessionProfiles(animal, paired_profile(left, right, check=check))


def panel():
    return (tuple(fixture("Mihili", session) for session in MIHILI),
            tuple(fixture("Chewie-L", session) for session in CHEWIE))


def with_difference(animal, session, difference):
    baseline = (-4., -5., -6.)
    return fixture(animal, session, tuple(value + delta for value, delta in zip(baseline, difference)), baseline)


def reference(mihili, chewie, **kwargs):
    return core.evaluate_development_reference(mihili, chewie, mihili_sessions=MIHILI,
        chewie_sessions=CHEWIE, recipe=RECIPE, check=kwargs.get("check", check))


def signed(mihili, chewie):
    return core.signed_profile_agreement(mihili, chewie, mihili_sessions=MIHILI,
                                        chewie_sessions=CHEWIE, check=check)


class ProfileReferenceTests(unittest.TestCase):
    def test_exact128_assignment_accuracy_multiplicity_and24_source_fits_with_half_ties(self):
        mihili, chewie = panel()
        original_fit = classifiers.fit_profile_classifier
        with patch.object(classifiers, "fit_profile_classifier", wraps=original_fit) as fitted:
            result = reference(mihili, chewie)
        self.assertTrue(result.complete)
        self.assertEqual(result.issues, ())
        self.assertEqual(result.source_fit_attempts, (8, 16))
        self.assertEqual(fitted.call_count, 24)
        self.assertEqual(len(result.assignments), 128)
        self.assertEqual(len(set(result.assignments)), 128)
        self.assertEqual(result.assignments[0], (0,) * 7)
        self.assertEqual(result.assignments[-1], (1,) * 7)
        self.assertEqual(result.assignments[1], (1, 0, 0, 0, 0, 0, 0))
        # Independent count oracle: identical +/-shape rows make logistic
        # prediction follow the source majority. A balanced four-session
        # source has beta=intercept=0 and every target row is an exact tie.
        expected = [[], []]
        for number in range(128):
            m = sum(bool(number & (1 << bit)) for bit in range(3))
            c = sum(bool(number & (1 << bit)) for bit in range(3, 7))
            expected[0].append((4 - c) / 4 if m < 2 else c / 4)
            expected[1].append(.5 if c == 2 else (3 - m) / 3 if c < 2 else m / 3)
        self.assertEqual(Counter(expected[0]), Counter({0.: 8, .25: 32, .5: 48, .75: 32, 1.: 8}))
        self.assertEqual(Counter(expected[1]), Counter({0.: 10, 1/3: 30, .5: 48, 2/3: 30, 1.: 10}))
        for direction, wanted, tail in zip(result.directions, expected, (8, 10)):
            self.assertTrue(direction.available)
            self.assertIsNone(direction.unavailable_reason)
            self.assertEqual(direction.observed_balanced_accuracy, 1.)
            np.testing.assert_array_equal(direction.assignment_balanced_accuracies, wanted)
            self.assertEqual(direction.inclusive_upper_tail_count, tail)
            self.assertEqual(direction.inclusive_upper_tail_fraction, tail / 128)
            # Joint global label complement leaves the same accuracy, while
            # the actual attainable tail is larger than the2/128 lower bound.
            self.assertEqual(direction.assignment_balanced_accuracies,
                             direction.assignment_balanced_accuracies[::-1])

    def test_raw_signed_omissions_centered_shape_and_unavailable_reference_do_not_drop_or_impute(self):
        differences = ((1., 1., 0.), (2., 0., 0.), (0., 2., 0.))
        mihili = tuple(with_difference("Mihili", session, value) for session, value in zip(MIHILI, differences))
        chewie = tuple(with_difference("Chewie-L", session, (2., 1., 1.)) for session in CHEWIE)
        agreement = signed(mihili, chewie)
        self.assertTrue(agreement.raw_available)
        self.assertTrue(agreement.raw_all_positive)
        self.assertEqual(agreement.mihili_raw_mean, (1., 1., 0.))
        self.assertEqual(agreement.chewie_raw_mean, (2., 1., 1.))
        self.assertAlmostEqual(agreement.raw_cosine, math.sqrt(3) / 2, places=14)
        self.assertAlmostEqual(agreement.centered_shape_cosine, .5, places=14)
        self.assertEqual(tuple((row.omitted_animal, row.omitted_session) for row in agreement.omissions),
            tuple(("Mihili", session) for session in MIHILI) + tuple(("Chewie-L", session) for session in CHEWIE))
        expected = (math.sqrt(3)/2, 2.5/math.sqrt(15), 3.5/math.sqrt(15)) + (math.sqrt(3)/2,) * 4
        np.testing.assert_allclose([row.cosine for row in agreement.omissions], expected, rtol=0, atol=3e-16)
        influence = tuple(with_difference("Mihili", session, value)
                          for session, value in zip(MIHILI, ((4., 0., 0.), (-1., 0., 0.), (-1., 0., 0.))))
        aligned = tuple(with_difference("Chewie-L", session, (1., 0., 0.)) for session in CHEWIE)
        influential = signed(influence, aligned)
        self.assertGreater(influential.raw_cosine, 0)
        self.assertEqual(influential.omissions[0].cosine, -1.)
        self.assertTrue(influential.raw_available)
        self.assertFalse(influential.raw_all_positive)
        zero = tuple(with_difference("Mihili", session, (0., 0., 0.)) for session in MIHILI)
        unavailable = signed(zero, aligned)
        self.assertFalse(unavailable.raw_available)
        self.assertIsNone(unavailable.raw_all_positive)
        self.assertEqual(unavailable.raw_unavailable_reason, "zero_mean_difference_norm")
        # An overall offset can pass raw direction agreement while centered
        # shape is undefined; retain the distinction rather than imputing0.
        constant = tuple(with_difference("Mihili", session, (1., 1., 1.)) for session in MIHILI)
        separate = signed(constant, chewie)
        self.assertTrue(separate.raw_all_positive)
        self.assertIsNone(separate.centered_shape_cosine)
        self.assertEqual(separate.centered_shape_unavailable_reason, "zero_mean_difference_norm")
        extreme = tuple(fixture("Mihili", session, (1e308, 0., -1e308), (-1e308, 0., 1e308))
                        for session in MIHILI)
        nonfinite_difference = signed(extreme, chewie)
        self.assertFalse(nonfinite_difference.raw_available)
        self.assertIsNone(nonfinite_difference.mihili_raw_mean)
        self.assertIsNone(nonfinite_difference.raw_cosine)
        ordinary_m, ordinary_c = panel()
        missing_block = replace(ordinary_m[0].profile.m1.blocks[0], score=None)
        missing_region = replace(ordinary_m[0].profile.m1,
                                 blocks=(missing_block,) + ordinary_m[0].profile.m1.blocks[1:])
        # Keep the old cached signed difference deliberately; readers must
        # inspect the required regional block, not trust that stale difference.
        bad_first = replace(ordinary_m[0], profile=replace(ordinary_m[0].profile, m1=missing_region))
        with patch.object(classifiers, "fit_profile_classifier", side_effect=AssertionError("fit reached")):
            for missing_panel in (ordinary_m[:-1], (bad_first,) + ordinary_m[1:]):
                result = reference(missing_panel, ordinary_c)
                self.assertFalse(result.complete)
                self.assertEqual(result.source_fit_attempts, (0, 0))
                self.assertEqual(len(result.assignments), 128)
                self.assertTrue(all(value is None for direction in result.directions
                                    for value in direction.assignment_balanced_accuracies))
                self.assertFalse(signed(missing_panel, ordinary_c).raw_available)
            with self.assertRaises(ValueError):
                reference(ordinary_m[::-1], ordinary_c)
        original_fit = classifiers.fit_profile_classifier
        def unavailable_null(*args, **kwargs):
            if (kwargs["source_animal"] == "Mihili"
                    and tuple(value.m1_label for value in kwargs["labels"]) == (0, 1, 1)):
                raise classifiers.ClassificationUnavailable("fixture_null_fit_unavailable")
            return original_fit(*args, **kwargs)
        with patch.object(classifiers, "fit_profile_classifier", side_effect=unavailable_null) as fits:
            stopped = reference(ordinary_m, ordinary_c)
        self.assertFalse(stopped.complete)
        self.assertEqual(stopped.source_fit_attempts, (2, 1))
        self.assertEqual(fits.call_count, 3)
        self.assertEqual(stopped.issues[0].stage, "null_classifier")
        self.assertEqual(stopped.issues[0].source_swap_bits, (1, 0, 0))
        for direction in stopped.directions:
            self.assertEqual(direction.observed_balanced_accuracy, 1.)
            self.assertEqual(direction.assignment_balanced_accuracies[0], 1.)
            self.assertTrue(all(value is None for value in direction.assignment_balanced_accuracies[1:]))
            self.assertIsNone(direction.inclusive_upper_tail_count)
            self.assertIsNone(direction.inclusive_upper_tail_fraction)
            self.assertFalse(direction.available)
        def cancelled():
            raise RuntimeError("cancelled before reference")
        with self.assertRaisesRegex(RuntimeError, "cancelled before reference"):
            reference(ordinary_m, ordinary_c, check=cancelled)


if __name__ == "__main__":
    unittest.main()
