"""Focused diagnostic algebra/support checks; coordinated compute only."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import repeated_signal_structure as diagnostic
import repeated_image_sequences as reader
from test_repeated_image_sequences import fixture, bind


class TestRepeatedSignalStructure(unittest.TestCase):
    def test_dc_and_fixed_spectral_bands_preserve_variance_units(self):
        times = np.arange(1000)/250.
        values = np.zeros((1000, 61))
        values[:, 0] = (7. + 2*np.sin(2*np.pi*.5*times)
                       + 3*np.sin(2*np.pi*5*times) + 4*np.sin(2*np.pi*50*times))
        item = SimpleNamespace(sequence_id="sequence", ordinal=0,
            sample_times_seconds=times, retained_mask=np.ones(1000, dtype=bool))
        before = values.copy()
        result = diagnostic._sequence_metrics(values, item, lambda: None)
        self.assertAlmostEqual(result["mean_square_dc_V2"], 49., places=10)
        self.assertAlmostEqual(result["centered_variance_V2"], 14.5, places=10)
        self.assertAlmostEqual(result["total_mean_square_V2"], 63.5, places=10)
        for key, expected in zip(diagnostic.BANDS, (2., 4.5, 8.)):
            self.assertAlmostEqual(result["spectral_power_V2"][key], expected, places=10)
        self.assertAlmostEqual(result["spectral_parseval_difference_V2"], 0., places=10)
        np.testing.assert_array_equal(values, before)
        item.retained_mask[50] = False
        with self.assertRaisesRegex(ValueError, "contiguous"):
            diagnostic._sequence_metrics(values, item, lambda: None)

    def test_actual_lag_bins_include_endpoint_without_snapping_or_interpolation(self):
        ticks = np.arange(0, 4000, 4)
        onsets = 1000 + 10*np.arange(200)
        onsets[1] = 1001
        indices = np.roll(np.arange(200), 13)
        contrasts = np.zeros((len(ticks), 61))
        contrasts[:, 0] = ticks
        item = SimpleNamespace(sample_times_seconds=ticks/1000., event_onsets_seconds=onsets/1000.,
            image_indices=indices, retained_mask=np.ones(len(ticks), dtype=bool))
        result, minimum, maximum = diagnostic._event_templates(contrasts, item, lambda: None)
        self.assertEqual(result.shape, (200, 40, 61))
        self.assertEqual(result[indices[0], 39, 0], 1790.)  # [1780,1800], six samples.
        self.assertEqual(result[indices[1], 0, 0], 1012.)   # [1001,1021), not snapped to1000.
        self.assertEqual((minimum, maximum), (5, 6))
        item.retained_mask[np.flatnonzero(ticks == 1800)[0]] = False
        with self.assertRaisesRegex(ValueError, "original complete filter/scoring support"):
            diagnostic._event_templates(contrasts, item, lambda: None)

    def test_matched_and_fixed37_alignment_and_undefined_cosine(self):
        odd = np.zeros((200, 40, 61))
        odd[:, :, 0] = (np.arange(200)-99.5)[:, None]
        result = diagnostic._template_metrics(odd, odd.copy())
        self.assertTrue(result["cosine_defined"])
        self.assertAlmostEqual(result["matched_cosine"], 1., places=14)
        self.assertGreater(result["matched_minus_shift37_cosine"], 0.)
        self.assertEqual(len(result["by_lag"]["matched_cosine"]), 40)
        # Swapping half labels preserves both covariance comparisons because
        # this fixture has identical halves; scalar covariance averages lags.
        expected = float(np.mean(np.sum(odd*np.roll(odd, -37, axis=0), axis=2)))
        self.assertAlmostEqual(result["shift37_cross_covariance_V2"], expected, places=12)
        absent = diagnostic._template_metrics(np.zeros_like(odd), odd)
        self.assertFalse(absent["cosine_defined"])
        self.assertIsNone(absent["matched_cosine"])
        self.assertIsNone(absent["matched_minus_shift37_cosine"])

    def test_all_sequence_image_roles_retained_and_source_arrays_unchanged(self):
        bound = bind(fixture())
        def raw_values(bound_arg, span, check):
            count = span.stop_sample-span.start_sample
            ticks = np.arange(count, dtype=float)
            values = np.zeros((count, 62))
            values[:, 0] = np.sin(ticks/50.) + span.ordinal
            values[:, 1] = np.cos(ticks/40.)
            return values
        with patch.object(reader, "_read_interval", side_effect=raw_values):
            person = reader.load_bound_participant(bound, budget_check=lambda: None)
        saved = [row.sensor_samples.copy() for row in person.sequences]
        result = diagnostic.measure_structure(person, budget_check=lambda: None)
        self.assertEqual(result["status"], "complete_repeated_signal_structure")
        self.assertEqual(len(result["sequence_metrics"]), 12)
        self.assertEqual(result["image_identity_count"], 200)
        self.assertEqual(result["recorded_channel_count"], 62)
        self.assertEqual(result["sensor_contrast_count"], 61)
        self.assertEqual(np.asarray(result["image_order"]["cross_parity_same_position_fraction"]).shape, (6, 6))
        self.assertFalse(result["raw_EEG_detrending"])
        self.assertFalse(result["inference_performed"])
        self.assertFalse(result["primary_rescue"])
        for expected, row in zip(saved, person.sequences):
            np.testing.assert_array_equal(expected, row.sensor_samples)
        with self.assertRaisesRegex(ValueError, "all200_identities_and12"):
            diagnostic.measure_structure(replace(person, sequences=person.sequences[:-1]), budget_check=lambda: None)
        with self.assertRaisesRegex(ValueError, "canonical lexicographic"):
            diagnostic.measure_structure(replace(person, image_ids=tuple(reversed(person.image_ids))), budget_check=lambda: None)


if __name__ == "__main__":
    unittest.main()
