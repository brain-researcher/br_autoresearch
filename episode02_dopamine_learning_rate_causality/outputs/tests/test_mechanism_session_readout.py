"""Analytic array fixtures only; coordinator runs these inside Slurm."""
from dataclasses import replace
import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from development_score_core import DEVELOPMENT_RECORDS
from logistic_normal_observation import IntegrationSettings, ObservationRecipe
from mechanism_session_readout import (
    ReadoutError, SessionCounts, SessionReadout, aggregate_trial_probabilities,
    finite_session_logits, fit_observation_from_readouts,
    predict_development_from_readout, refit_final_observation_from_readouts,
)


def recipe():
    return ObservationRecipe((-4.0, 4.0), (0.0,), 0.0,
                             IntegrationSettings((12, 24), 1e-9, 1e-9, 1e-10, 100, 100000))


def panel(held=1):
    ids = [m for m in DEVELOPMENT_RECORDS if m != held]
    # Different probability and count orders exercise identity-based joining.
    counts = [SessionCounts(m, ("late", "lost", "early"), (3, 0, 1), (4, 0, 4)) for m in ids]
    readouts = [SessionReadout(m, ("early", "late", "lost"), (.25, .75, .6), (100, 20, 10)) for m in ids]
    return counts, readouts


class MechanismSessionReadoutTests(unittest.TestCase):
    def test_expected_rate_averages_probabilities_not_logits(self):
        readout = aggregate_trial_probabilities(
            mouse_id=1, session_ids=("early", "late"),
            trial_session_ids=("late", "early", "early", "late", "early"),
            trial_probabilities=(.8, .1, .9, float("nan"), .2),
            eligible=(True, True, True, False, True))
        self.assertEqual(readout.model_trial_counts, (3, 1))
        np.testing.assert_allclose(readout.probabilities, (.4, .8), rtol=0, atol=1e-15)
        logits = finite_session_logits(readout)
        self.assertAlmostEqual(logits[0], math.log(.4 / .6), places=14)
        mean_trial_logit = sum(math.log(p / (1-p)) for p in (.1, .9, .2)) / 3
        self.assertNotAlmostEqual(logits[0], mean_trial_logit, places=3)

    def test_boundaries_are_preserved_then_refused_not_smoothed(self):
        for probability in (0.0, 1.0):
            readout = aggregate_trial_probabilities(mouse_id=1, session_ids=("s",),
                trial_session_ids=("s",), trial_probabilities=(probability,), eligible=(True,))
            self.assertEqual(readout.probabilities, (probability,))
            with self.assertRaisesRegex(ReadoutError, "boundary probability"):
                finite_session_logits(readout)
        interior = SessionReadout(1, ("s",), (1e-300,), (1,))
        self.assertAlmostEqual(finite_session_logits(interior)[0], math.log(1e-300), places=12)

    def test_missing_model_support_and_implicit_eligibility_fail(self):
        kwargs = dict(mouse_id=1, session_ids=("s", "t"), trial_session_ids=("s", "t"),
                      trial_probabilities=(.2, .3), eligible=(True, True))
        invalid = ({"eligible": (True, False)}, {"eligible": (1, True)},
                   {"trial_session_ids": ("s", "unknown")}, {"trial_probabilities": (.2,)},
                   {"trial_probabilities": (True, .3)}, {"trial_probabilities": (float("nan"), .3)},
                   {"session_ids": ("s", "s")}, {"trial_probabilities": (1.1, .3)})
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ReadoutError):
                aggregate_trial_probabilities(**(kwargs | change))

    def test_join_preserves_empirical_denominators_zero_sessions_and_known_fit(self):
        counts, readouts = panel()
        fit = fit_observation_from_readouts(training_counts=counts[::-1], training_readouts=readouts,
                                           held_mouse_id=1, recipe=recipe())
        # k/n equals the fixed mechanism probability for every informative row:
        # the shared-offset marginal MLE is exactly zero with zero mouse SD.
        self.assertAlmostEqual(fit.selected.population_offset, 0.0, places=9)
        self.assertEqual(fit.training_mouse_ids, tuple(m for m in DEVELOPMENT_RECORDS if m != 1))
        held = SessionReadout(1, ("late", "lost", "early"), (.75, .6, .25), (1, 50, 1000))
        prediction = predict_development_from_readout(fit, held)
        self.assertEqual(prediction.session_ids, held.session_ids)
        score = prediction.log_probability(successes=(3, 0, 1), trials=(4, 0, 4))
        expected = 2 * math.log(4 * .25 * .75**3)
        self.assertAlmostEqual(score, expected, places=9)

    def test_exact_training_and_session_rosters_prevent_silent_drops(self):
        counts, readouts = panel()
        for changed_counts, changed_readouts in (
            (counts[:-1], readouts), (counts, readouts[:-1]),
            ([replace(counts[0], mouse_id=1)] + counts[1:], readouts),
            (counts, [replace(readouts[0], mouse_id=3)] + readouts[1:]),
            (counts, [readouts[0]] + readouts[:-1]),
            (counts, [replace(readouts[0], session_ids=("early", "late", "other"))] + readouts[1:]),
        ):
            with self.assertRaises(ReadoutError):
                fit_observation_from_readouts(training_counts=changed_counts,
                    training_readouts=changed_readouts, held_mouse_id=1, recipe=recipe())

    def test_final_refit_stays_separate_and_callback_can_abort(self):
        counts, readouts = panel(held=None)
        fit = refit_final_observation_from_readouts(training_counts=counts, training_readouts=readouts,
                                                  recipe=recipe(), check=lambda: None)
        self.assertEqual(fit.training_mouse_ids, DEVELOPMENT_RECORDS)
        self.assertAlmostEqual(fit.selected.population_offset, 0.0, places=9)
        with self.assertRaises(ReadoutError):
            predict_development_from_readout(fit, readouts[0])
        def cancel():
            raise RuntimeError("resource stop")
        with self.assertRaisesRegex(RuntimeError, "resource stop"):
            refit_final_observation_from_readouts(training_counts=counts, training_readouts=readouts,
                                                 recipe=recipe(), check=cancel)


if __name__ == "__main__":
    unittest.main()
