"""Five focused label checks over invented timestamps, never source samples.

Run only in the coordinator's Slurm allocation. Existing loss/scorer integration
checks distinguish event labels from complete-horizon coverage; no fit runs.
"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from classical_forecast_backend import GroupIdentity
from forecast_components import censoring_loss_and_logit_gradient
from forecast_labels import AcceptedOnset, Followup, RiskDecision, label_for_decision
from forecast_scoring import horizon_log_scores


IDENTITY = GroupIdentity("WAY", 1, 1, "test-series-1-segment-1")


def label(*, duration, crossing=None, known=None, state="at_risk",
          unknown_from=None, label_stop=None):
    """Fixture times are integer milliseconds relative to the 1000-ms row."""
    decision = RiskDecision(IDENTITY, 1000, state)
    followup = Followup(IDENTITY, None if duration is None else 1000 + duration,
                        None if unknown_from is None else 1000 + unknown_from)
    onset = (None if crossing is None else AcceptedOnset(IDENTITY, 1000 + crossing,
                                                       1000 + (crossing if known is None else known)))
    return label_for_decision(decision, followup, onset, ticks_per_second=1000,
        grid_origin_tick=0, label_stop_tick=None if label_stop is None else 1000 + label_stop)


class ForecastLabelTests(unittest.TestCase):
    def test_exact_half_open_onset_edges_and_full_nononset_category(self):
        edges = ((0, 0), (49, 0), (50, 1), (299, 5), (300, 6),
                 (599, 11), (600, 12), (899, 17), (900, 18), (901, 18))
        for onset_ms, expected in edges:
            with self.subTest(onset_ms=onset_ms):
                result = label(duration=1000, crossing=onset_ms)
                self.assertEqual(result.observed_category, expected)
                self.assertEqual(result.complete_bins, expected)
                self.assertEqual(result.observed_through_bins, 18)
        # Clock resolution is caller-supplied, not an assumed source rate.
        for clock in (20, 500, 1000):
            decision = RiskDecision(IDENTITY, clock, "at_risk")
            result = label_for_decision(decision, Followup(IDENTITY, 2 * clock),
                AcceptedOnset(IDENTITY, clock + 6 * (clock // 20), clock + 6 * (clock // 20)),
                ticks_per_second=clock, grid_origin_tick=0)
            self.assertEqual((result.observed_category, result.complete_bins), (6, 6))
        for duration in (900, 1000):
            result = label(duration=duration)
            self.assertEqual((result.observed_category, result.complete_bins,
                              result.observed_through_bins), (18, 18, 18))

    def test_partial_censoring_and_first_bin_event_have_different_likelihoods(self):
        for duration, complete in ((0, 0), (49, 0), (50, 1), (299, 5),
                                   (300, 6), (599, 11), (600, 12), (899, 17)):
            result = label(duration=duration)
            self.assertEqual((result.observed_category, result.complete_bins,
                              result.observed_through_bins), (-1, complete, complete))
        unknown = label(duration=None)
        self.assertEqual((unknown.observed_category, unknown.complete_bins,
                          unknown.observed_through_bins), (-1, 0, 0))
        observations = (label(duration=30), label(duration=30, crossing=20, known=25),
                        label(duration=275), label(duration=900))
        probabilities = np.arange(1., 20.) / 190.
        losses, gradients = censoring_loss_and_logit_gradient(
            np.tile(np.log(probabilities), (len(observations), 1)),
            np.array([row.observed_category for row in observations]),
            np.array([row.complete_bins for row in observations]))
        np.testing.assert_allclose(losses, [0., -np.log(probabilities[0]),
            -np.log(probabilities[5:].sum()), -np.log(probabilities[18])], atol=1e-14)
        np.testing.assert_array_equal(gradients[0], np.zeros(19))
        self.assertGreater(abs(gradients[1, 0]), .9)
        self.assertEqual(observations[1].observed_through_bins, 0)

    def test_event_label_does_not_supply_complete_horizon_coverage(self):
        observations = (label(duration=30, crossing=20, known=25),
                        label(duration=550, crossing=205, known=245),
                        label(duration=650, crossing=305, known=345),
                        label(duration=950, crossing=605, known=645),
                        label(duration=400, crossing=300, known=395, unknown_from=390))
        self.assertEqual([row.complete_bins for row in observations], [0, 4, 6, 12, 6])
        self.assertEqual([row.observed_through_bins for row in observations], [0, 11, 13, 18, 7])
        categories = np.array([row.observed_category for row in observations])
        coverage = np.array([row.observed_through_bins for row in observations])
        logp = np.full((len(observations), 19), -np.log(19.))
        expected = {"near_event": [False, True, True, True, True],
                    "primary_strict": [False, False, True, True, False],
                    "longer_diagnostic": [False, False, False, True, False]}
        for horizon, eligibility in expected.items():
            result = horizon_log_scores(logp, categories, coverage, horizon=horizon)
            np.testing.assert_array_equal(result["eligible"], eligibility)
            self.assertTrue(np.isnan(result["binary_outcomes"][~result["eligible"]]).all())
            self.assertTrue(np.isnan(result["log_scores"][~result["eligible"]]).all())
        # Additional allowed follow-up changes eligibility, never crossing time.
        extended = label(duration=950, crossing=205, known=245)
        self.assertEqual(extended.observed_category, observations[1].observed_category)
        self.assertEqual(extended.complete_bins, observations[1].complete_bins)
        self.assertEqual(extended.observed_through_bins, 18)

    def test_late_confirmation_prefix_and_pending_rows_never_restore_evidence(self):
        for pending_start, bins in ((200, 4), (100, 2)):
            result = label(duration=1000, crossing=200, known=700,
                           unknown_from=pending_start, label_stop=500)
            self.assertEqual((result.observed_category, result.complete_bins,
                              result.observed_through_bins), (-1, bins, bins))
        # Known exactly at the exclusive cutoff still belongs outside it.
        boundary = label(duration=1000, crossing=200, known=500,
                         unknown_from=200, label_stop=500)
        self.assertEqual((boundary.observed_category, boundary.complete_bins), (-1, 4))
        with self.assertRaises(ValueError):
            label(duration=1000, crossing=200, known=700, label_stop=500)
        with self.assertRaises(ValueError):
            label(duration=1000, crossing=200, known=700,
                  unknown_from=250, label_stop=500)
        accepted = label(duration=1000, crossing=200, known=499, label_stop=500)
        self.assertEqual((accepted.observed_category, accepted.complete_bins,
                          accepted.observed_through_bins), (4, 4, 10))
        # Adjudication after a status stop may confirm an earlier crossing,
        # but an unresolved crossing at/before it still blocks that label.
        for unknown, bins in ((200, 4), (300, 6)):
            censored = label(duration=400, crossing=300, known=395, unknown_from=unknown)
            self.assertEqual((censored.observed_category, censored.complete_bins,
                              censored.observed_through_bins), (-1, bins, bins))
        # A post-prefix onset does not turn its earlier prefix into a positive.
        suffix = label(duration=1000, crossing=600, known=650, label_stop=500)
        self.assertEqual((suffix.observed_category, suffix.complete_bins,
                          suffix.observed_through_bins), (-1, 10, 10))
        for state in ("pending", "unarmed", "unknown"):
            for confirmation in (245, 700):
                result = label(duration=1000, crossing=205, known=confirmation, state=state)
                self.assertEqual(result.decision.state, state)
                self.assertEqual((result.observed_category, result.complete_bins,
                                  result.observed_through_bins), (None, None, None))

    def test_recording_reset_identity_and_integer_clock_bounds_are_explicit(self):
        decision = RiskDecision(IDENTITY, 1000, "at_risk")
        followup = Followup(IDENTITY, 2000)
        onset = AcceptedOnset(IDENTITY, 1300, 1340)
        for changed in (replace(IDENTITY, participant=2),
                        replace(IDENTITY, series_or_run=2),
                        replace(IDENTITY, boundary_id="test-series-1-after-gap")):
            with self.assertRaises(ValueError):
                label_for_decision(decision, replace(followup, identity=changed), onset,
                                   ticks_per_second=1000, grid_origin_tick=0)
            with self.assertRaises(ValueError):
                label_for_decision(decision, followup, replace(onset, identity=changed),
                                   ticks_per_second=1000, grid_origin_tick=0)
        for clock in (0, 25, 1000.0, True):
            with self.assertRaises(ValueError):
                label_for_decision(decision, followup, onset,
                                   ticks_per_second=clock, grid_origin_tick=0)
        with self.assertRaises(ValueError):
            label_for_decision(replace(decision, decision_tick=1025), followup, onset,
                               ticks_per_second=1000, grid_origin_tick=0)
        for cutoff in (999, 1000):
            with self.assertRaises(ValueError):
                label_for_decision(decision, followup, onset, ticks_per_second=1000,
                                   grid_origin_tick=0, label_stop_tick=cutoff)
        result = label_for_decision(decision, followup, onset,
                                   ticks_per_second=1000, grid_origin_tick=0)
        self.assertEqual(result.decision, decision)
        self.assertEqual((decision.decision_tick, followup.observed_stop_tick,
                          onset.crossing_tick, onset.known_tick), (1000, 2000, 1300, 1340))


if __name__ == "__main__":
    unittest.main()
