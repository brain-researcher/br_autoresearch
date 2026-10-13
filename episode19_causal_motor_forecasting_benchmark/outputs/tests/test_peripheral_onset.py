"""Only new detector behavior; the earlier eight forecast tests are reused."""
import unittest
from unittest.mock import patch

import numpy as np

from forecast_components import CausalFIR
from peripheral_onset import (PilotSettings, decode_fixed_half_level_cue, decode_two_level_cue, detect_risk,
                              earliest_accepted_crossing, fit_pilot_noise, rest_window_stationarity,
                              sensor_summaries, trailing_mean)


class PeripheralOnsetTests(unittest.TestCase):
    def test_support_only_preserves_counts_and_never_fits_noise_thresholds(self):
        cue = np.tile(np.r_[np.zeros(500), np.ones(500)], 8)
        hand = np.zeros((len(cue), 3, 3))
        emg = np.tile(np.array([-1., 1.]), len(cue) * 4)[:, None] * np.ones((1, 5))
        features = np.ones((len(cue), 8))
        with patch("peripheral_onset.np.median", side_effect=AssertionError("Noise fitting is forbidden")):
            thresholds, report = fit_pilot_noise(features, emg, hand, cue, support_only=True)
        self.assertIsNone(thresholds)
        self.assertEqual(report["status"], "stationary_support_sufficient_thresholds_not_fit")
        self.assertEqual(report["stationary_windows"], 8)
        self.assertEqual((report["fit_windows"], report["validation_windows"]), (4, 4))
        self.assertNotIn("high_threshold", report)

    def test_fixed_half_level_smooth_transition_missingness_and_future_invariance(self):
        led = np.array([-.0017, 0., .2, .4999, .5, .8, 1., 1.0009,
                        np.nan, np.inf, -np.inf, .8, .4, 0.])
        expected = np.array([0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, 1, 0, 0])
        states, report = decode_fixed_half_level_cue(led)
        np.testing.assert_array_equal(states, expected)
        # An enormous future value cannot change an earlier amplitude scale.
        changed = led.copy()
        changed[11:] = 1e8
        np.testing.assert_array_equal(decode_fixed_half_level_cue(changed)[0][:11], states[:11])
        self.assertEqual(report["unknown_samples"], 3)
        self.assertFalse(report["known_end_return_phase_authenticated"])
        # Missing-to-high is not a new observed low-to-high start edge.
        edges = np.flatnonzero((states[:-1] == 0) & (states[1:] == 1)) + 1
        np.testing.assert_array_equal(edges, [4])

    def test_vector_mean_matches_qualified_stream_and_gap_reset(self):
        x = np.arange(120, dtype=float).reshape(60, 2)/17
        x[23] = np.nan
        expected = []
        stream = CausalFIR(np.ones(7)/7, 2)
        # Arbitrary chunks preserve the same state; no padding is introduced.
        for chunk in np.array_split(x, [11, 24, 37]):
            for sample in chunk:
                out = stream.update(sample)
                expected.append([np.nan, np.nan] if out is None else out)
        np.testing.assert_allclose(trailing_mean(x, 7), expected, rtol=0, atol=1e-13, equal_nan=True)

    def test_future_and_constant_offset_invariance_of_sensor_paths(self):
        rng = np.random.default_rng(19)
        emg = rng.normal(size=(8001, 5))
        hand = np.cumsum(rng.normal(size=(1000, 3, 3)), axis=0)
        before = sensor_summaries(emg, hand)
        shifted = sensor_summaries(emg + np.arange(5)*17, hand + np.arange(9).reshape(3, 3)*23)
        np.testing.assert_allclose(before, shifted, rtol=1e-10, atol=1e-9, equal_nan=True)
        emg[5001:] = 1e4
        hand[626:] = -1e4
        after = sensor_summaries(emg, hand)
        np.testing.assert_allclose(before[:626], after[:626], rtol=0, atol=0, equal_nan=True)

    def test_rejected_pending_never_restores_paused_risk_and_one_reach_per_cue(self):
        settings = PilotSettings(sample_rate_hz=100, entry_stillness_ms=50, confirmation_ms=40)
        x = np.zeros((100, 8))
        cue = np.zeros(100, dtype=int)
        cue[10:95] = 1
        x[20:22, 0] = 3  # Rejected before 40 ms.
        x[35:41, 1] = 3  # Accepted at 39, labelled at 35.
        x[70:80, 2] = 3  # Holding/return cannot rearm in this cue.
        report, risk = detect_risk(x, cue, np.ones(8), np.ones(8)*2, settings)
        self.assertFalse(np.any(risk[20:27]))
        self.assertTrue(np.any(risk[27:35]))
        self.assertEqual(report["candidate_sensor_onsets_with_confirmation"], 1)
        self.assertEqual(report["candidate_cue_segments"][0]["candidate_sensor_onset"]["crossing_sample"], 35)
        self.assertFalse(report["online_risk_contract_qualified"])
        self.assertFalse(np.any(risk[35:]))

    def test_earliest_accepted_crossing_not_first_confirmation(self):
        rows = [{"path_index": 1, "crossing_sample": 20, "confirmation_sample": 25},
                {"path_index": 0, "crossing_sample": 18, "confirmation_sample": 30}]
        self.assertEqual(earliest_accepted_crossing(rows)["crossing_sample"], 18)

    def test_nonstationary_motion_and_missing_primary_path_rejected(self):
        rng = np.random.default_rng(22)
        emg = rng.normal(size=(4000, 5))
        hand = np.zeros((500, 3, 3))
        hand[:, 0, 0] = np.arange(500)
        passed, reasons = rest_window_stationarity(emg, hand)
        self.assertFalse(passed)
        self.assertIn("index_finger_p2_quarter_motion_drift", reasons)
        hand[:] = 0
        emg[20, 4] = np.nan
        self.assertFalse(rest_window_stationarity(emg, hand)[0])

    def test_gap_does_not_manufacture_cue_or_negative_and_multilevel_unresolved(self):
        settings = PilotSettings(sample_rate_hz=100, entry_stillness_ms=50)
        x = np.zeros((100, 8))
        cue = np.zeros(100, dtype=int)
        cue[10:90] = 1
        x[30] = np.nan
        report, risk = detect_risk(x, cue, np.ones(8), np.ones(8)*2, settings)
        self.assertFalse(np.any(risk[30:]))
        self.assertEqual(report["candidate_cue_segments"][0]["close_reason"], "sensor_or_cue_gap_unknown")
        self.assertIsNone(report["candidate_cue_segments"][0]["candidate_sensor_onset"])
        states, report = decode_two_level_cue(np.array([0., 1., .5]))
        self.assertIsNone(states)
        self.assertEqual(report["status"], "unresolved_not_two_exact_levels")
        _, report = decode_two_level_cue(np.array([0., 1.]))
        self.assertFalse(report["numeric_polarity_authenticated"])
        self.assertFalse(report["known_end_return_phase_authenticated"])


if __name__ == "__main__":
    unittest.main()
