"""Two small independent classifier checks; execute only in coordinated Slurm."""
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
import profile_classifier as core


BLOCKS = tuple(FeatureBlock(f"band_{index}", (band,)) for index, band in enumerate(BANDS[:3]))


def check():
    pass


def fixture(animal, session, m1=(2., 1., 0.), pmd=(0., 1., 2.)):
    design = FrozenDesign(session, "evaluation", tuple(range(8)), tuple(range(8)),
                          tuple(range(8)), (5, 6), ("M1_pc0", "M1_pc1"))
    left = ProfileScore("M1", design, 100.,
                        tuple(BlockScore(block, None, float(value), None) for block, value in zip(BLOCKS, m1)))
    right = ProfileScore("PMd", replace(design, axis_ids=("PMd_pc0", "PMd_pc1")), 100.,
                         tuple(BlockScore(block, None, float(value), None) for block, value in zip(BLOCKS, pmd)))
    return core.PairedSessionProfiles(animal, paired_profile(left, right, check=check))


def labels(animal, sessions, swapped=False):
    return tuple(core.PairedLabels(animal, session, int(not swapped), int(swapped)) for session in sessions)


class ProfileClassifierTests(unittest.TestCase):
    def test_cosine_shapes_explicit_labels_exact_ties_and_complete_paired_support(self):
        source_sessions = ("M_a", "M_b")
        source = (fixture("Mihili", "M_a"),
                  fixture("Mihili", "M_b", (18., 14., 10.), (-5., -3., -1.)))
        fit = core.fit_profile_classifier(source, source_animal="Mihili", expected_sessions=source_sessions,
            labels=labels("Mihili", source_sessions), recipe=core.ClassifierRecipe("cosine_prototype"), check=check)
        v = np.asarray((1., 0., -1.)) / math.sqrt(2)
        np.testing.assert_allclose(fit.prototypes, (-v, v), rtol=0, atol=2e-16)
        target_sessions = ("C_a", "C_b")
        target = (fixture("Chewie-L", "C_a", (108., 104., 100.), (7., 9., 11.)),
                  fixture("Chewie-L", "C_b", (7., 4., 1.), (-3., 1., 5.)))
        predicted = core.predict_profile_classifier(fit, target, target_animal="Chewie-L",
                                                      expected_sessions=target_sessions, check=check)
        self.assertEqual(tuple(row.predicted_label for row in predicted.rows), (1, 0, 1, 0))
        scored = core.score_profile_predictions(predicted, labels=labels("Chewie-L", target_sessions), check=check)
        self.assertEqual((scored.balanced_accuracy, scored.tied_rows, scored.total_rows), (1., 0, 4))
        self.assertEqual(core.score_profile_predictions(predicted,
            labels=labels("Chewie-L", target_sessions, swapped=True), check=check).balanced_accuracy, 0.)
        flipped = core.fit_profile_classifier(source, source_animal="Mihili", expected_sessions=source_sessions,
            labels=labels("Mihili", source_sessions, swapped=True),
            recipe=core.ClassifierRecipe("cosine_prototype"), check=check)
        np.testing.assert_array_equal(flipped.prototypes, fit.prototypes[::-1])
        reverse = core.predict_profile_classifier(flipped, target, target_animal="Chewie-L",
                                                  expected_sessions=target_sessions, check=check)
        self.assertEqual(core.score_profile_predictions(reverse,
            labels=labels("Chewie-L", target_sessions), check=check).balanced_accuracy, 0.)
        # Identical learned prototypes give exact ties without relying on a
        # nominally orthogonal floating-point dot product being precisely zero.
        tied_source = tuple(fixture("Mihili", session, (2., 1., 0.), (2., 1., 0.))
                            for session in source_sessions)
        tied_fit = core.fit_profile_classifier(tied_source, source_animal="Mihili",
            expected_sessions=source_sessions, labels=labels("Mihili", source_sessions),
            recipe=core.ClassifierRecipe("cosine_prototype"), check=check)
        tied = core.predict_profile_classifier(tied_fit, target, target_animal="Chewie-L",
                                               expected_sessions=target_sessions, check=check)
        self.assertTrue(all(row.predicted_label is None and row.margin == 0 for row in tied.rows))
        tied_score = core.score_profile_predictions(tied, labels=labels("Chewie-L", target_sessions), check=check)
        self.assertEqual((tied_score.balanced_accuracy, tied_score.tied_rows, tied_score.total_rows), (.5, 4, 4))
        # Complete metadata binds before an unavailable numeric value can be inspected.
        unavailable = replace(target[0].profile.m1, blocks=(replace(target[0].profile.m1.blocks[0], score=None),)
                               + target[0].profile.m1.blocks[1:])
        bad_first = replace(target[0], profile=replace(target[0].profile, m1=unavailable))
        bad_last = replace(target[1], profile=replace(target[1].profile, blocks=BLOCKS[::-1]))
        with self.assertRaises(ValueError):
            core.predict_profile_classifier(fit, (bad_first, bad_last), target_animal="Chewie-L",
                                             expected_sessions=target_sessions, check=check)
        with self.assertRaises(core.ClassificationUnavailable):
            core.predict_profile_classifier(fit, (bad_first, target[1]), target_animal="Chewie-L",
                                             expected_sessions=target_sessions, check=check)
        for invalid in (target[:-1], target[::-1]):
            with self.assertRaises(ValueError):
                core.predict_profile_classifier(fit, invalid, target_animal="Chewie-L",
                                                 expected_sessions=target_sessions, check=check)
        with self.assertRaises(ValueError):
            core.predict_profile_classifier(fit, source, target_animal="Mihili",
                                             expected_sessions=source_sessions, check=check)
        with self.assertRaises(ValueError):
            core.score_profile_predictions(replace(predicted, rows=predicted.rows[:-1]),
                                           labels=labels("Chewie-L", target_sessions), check=check)

    def test_logistic_matches_independent_scalar_optimum_and_exposes_numerical_failure(self):
        source_sessions = ("M_a", "M_b")
        source = tuple(fixture("Mihili", session) for session in source_sessions)
        penalty = .3
        fit = core.fit_profile_classifier(source, source_animal="Mihili", expected_sessions=source_sessions,
            labels=labels("Mihili", source_sessions), recipe=core.ClassifierRecipe("l2_logistic", penalty), check=check)
        # For paired +/-v rows the exact optimum is beta=b*v, intercept=0,
        # where lambda*b = 1/(1+exp(b)). Bisection is independent of Newton.
        low, high = 0., 1. / penalty
        for _ in range(80):
            midpoint = .5 * (low + high)
            if penalty * midpoint < 1. / (1. + math.exp(midpoint)):
                low = midpoint
            else:
                high = midpoint
        b = .5 * (low + high)
        np.testing.assert_allclose(fit.coefficients, b * np.asarray((1., 0., -1.)) / math.sqrt(2),
                                   rtol=0, atol=3e-9)
        self.assertAlmostEqual(fit.intercept, 0., places=12)
        self.assertAlmostEqual(fit.objective, math.log1p(math.exp(-b)) + .5 * penalty * b * b, places=12)
        self.assertLessEqual(fit.maximum_absolute_gradient, 1e-9)
        self.assertLessEqual(fit.iterations, 100)
        target = (fixture("Chewie-L", "C_a", (100., 80., 60.), (3., 5., 7.)),)
        prediction = core.predict_profile_classifier(fit, target, target_animal="Chewie-L",
                                                     expected_sessions=("C_a",), check=check)
        np.testing.assert_allclose([row.margin for row in prediction.rows], (b, -b), rtol=0, atol=3e-9)
        self.assertEqual(core.score_profile_predictions(prediction,
            labels=labels("Chewie-L", ("C_a",)), check=check).balanced_accuracy, 1.)
        with self.assertRaises(core.ClassificationUnavailable):
            core.fit_profile_classifier((fixture("Mihili", "M_a", (3., 3., 3.)), source[1]),
                source_animal="Mihili", expected_sessions=source_sessions,
                labels=labels("Mihili", source_sessions), recipe=core.ClassifierRecipe("l2_logistic", penalty), check=check)
        with patch.object(core.np.linalg, "solve", side_effect=np.linalg.LinAlgError("bounded numerical fixture")):
            with self.assertRaisesRegex(core.ClassificationUnavailable, "singular_logistic_newton_system"):
                core.fit_profile_classifier(source, source_animal="Mihili", expected_sessions=source_sessions,
                    labels=labels("Mihili", source_sessions), recipe=core.ClassifierRecipe("l2_logistic", penalty), check=check)
        with self.assertRaises(ValueError):
            core.fit_profile_classifier(source, source_animal="Mihili", expected_sessions=source_sessions,
                labels=labels("Mihili", source_sessions), recipe=core.ClassifierRecipe("l2_logistic", 0.), check=check)


if __name__ == "__main__":
    unittest.main()
