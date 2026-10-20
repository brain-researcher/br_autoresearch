"""Small projection/support fixtures only; execute in coordinated Slurm."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

import repeated_signal_detrend as diagnostic
from repeated_image_sequences import RepeatedParticipant
from test_repeated_image_control import sequence


def person_fixture():
    items = []
    for ordinal in range(12):
        item = sequence(ordinal)
        times = np.arange(1200) / 250.
        mask = np.ones(len(times), dtype=bool)
        mask[:10] = False
        onsets = item.event_onsets_seconds + .2
        coefficients = np.linspace(-2., 2., 62)
        slow = (3. + times - .3 * times**2 + .02 * times**3)[:, None] * coefficients[None, :]
        fast = np.sin((np.arange(len(times))[:, None] + 1) * (np.arange(62)[None, :] + 1) * .037 + ordinal)
        values = slow + .1 * fast
        items.append(replace(item, sample_times_seconds=times, retained_mask=mask,
            event_onsets_seconds=onsets, final_offset_seconds=float(onsets[-1] + .05), sensor_samples=values))
    return RepeatedParticipant("sub-fixture", tuple("i%03d" % i for i in range(200)), tuple(items), {})


class RepeatedSignalDetrendTests(unittest.TestCase):
    def test_exact_polynomial_removal_energy_conservation_and_nonretained_preservation(self):
        item = person_fixture().sequences[0]
        times = item.sample_times_seconds
        coefficients = np.arange(62, dtype=float) / 61.
        values = (1. + times - .2 * times**2 + .03 * times**3)[:, None] * coefficients
        item = replace(item, sensor_samples=values)
        before = values.copy()
        copied, record = diagnostic._project_sequence(item, lambda: None)
        np.testing.assert_array_equal(item.sensor_samples, before)
        np.testing.assert_array_equal(copied.sensor_samples[~item.retained_mask], before[~item.retained_mask])
        self.assertLess(float(np.max(np.abs(copied.sensor_samples[item.retained_mask]))), 1e-13)
        self.assertAlmostEqual(record["removed_energy_fraction"], 1., places=12)
        np.testing.assert_allclose(np.asarray(record["per_channel_polynomial_energy_V2"])
            + np.asarray(record["per_channel_residual_energy_V2"]), record["per_channel_total_energy_V2"],
            rtol=1e-12, atol=1e-25)
        self.assertEqual(record["per_channel_total_energy_V2"][0], 0.)
        self.assertEqual(record["per_channel_residual_energy_V2"][0], 0.)
        for field in ("sensor_samples", "sample_times_seconds", "retained_mask",
                      "event_onsets_seconds", "image_indices", "target_flags", "response_flags"):
            self.assertFalse(getattr(copied, field).flags.writeable)
            self.assertFalse(np.shares_memory(getattr(copied, field), getattr(item, field)))

    def test_full_structure_retains_all_images_actual_times_and_diagnostic_flags(self):
        person = person_fixture()
        original = [item.sensor_samples.copy() for item in person.sequences]
        result = diagnostic.run_participant(person, budget_check=lambda: None)
        self.assertEqual(result["status"], "complete_repeated_signal_detrend")
        self.assertEqual(result["projection_recipe"], diagnostic.PROJECTION_RECIPE)
        self.assertEqual(len(result["projection_accounting"]["sequence_rows"]), 12)
        structure = result["structure"]
        self.assertEqual((structure["physical_sequence_count"], structure["image_identity_count"],
                          structure["recorded_channel_count"]), (12, 200, 62))
        self.assertTrue(structure["raw_EEG_detrending"])
        self.assertFalse(result["original_R_refit_or_rescored"])
        self.assertFalse(result["primary_rescue"])
        self.assertFalse(result["out_of_sample_prediction"])
        self.assertFalse(structure["template_definition"]["event_snapping"])
        self.assertEqual(structure["template_definition"]["lag_bin_edges_ms"], list(range(0, 801, 20)))
        negative = result["negative_lag_structure"]
        self.assertEqual(negative["template_definition"]["lag_bin_edges_ms"], list(range(-800, 1, 20)))
        self.assertTrue(negative["template_definition"]["final_bin_includes_0ms"])
        self.assertFalse(negative["template_definition"]["final_bin_includes_800ms"])
        self.assertFalse(negative["registered_negative_lag_control_score"])
        self.assertEqual(negative["template_definition"]["diagnostic_onset_shift_ms"], -800)
        self.assertEqual(negative["image_order"], structure["image_order"])
        for before, item in zip(original, person.sequences):
            np.testing.assert_array_equal(before, item.sensor_samples)
        self.assertEqual(person.metadata_summary, {})

    def test_sequence_projection_cannot_use_other_sequences_and_incomplete_people_refuse(self):
        person = person_fixture()
        copied, first = diagnostic._project_person(person, budget_check=lambda: None)
        altered = replace(person, sequences=person.sequences[:1]
            + (replace(person.sequences[1], sensor_samples=person.sequences[1].sensor_samples * 40),)
            + person.sequences[2:])
        changed, second = diagnostic._project_person(altered, budget_check=lambda: None)
        np.testing.assert_array_equal(copied.sequences[0].sensor_samples, changed.sequences[0].sensor_samples)
        self.assertEqual(first["sequence_rows"][0], second["sequence_rows"][0])
        with patch.object(diagnostic, "measure_structure", side_effect=AssertionError("partial structure forbidden")):
            with self.assertRaisesRegex(ValueError, "all200_identities_and12"):
                diagnostic.run_participant(replace(person, sequences=person.sequences[:-1]), budget_check=lambda: None)

    def test_missing_negative_window_support_stops_without_dropping_images(self):
        person = person_fixture()
        first = person.sequences[0]
        mask = first.retained_mask.copy()
        # The positive window still fits; the prespecified first -800ms bin
        # does not. A negative-control failure cannot remove that image.
        mask[:60] = False
        altered = replace(person, sequences=(replace(first, retained_mask=mask),) + person.sequences[1:])
        with self.assertRaisesRegex(ValueError, "original complete filter/scoring support"):
            diagnostic.run_participant(altered, budget_check=lambda: None)


if __name__ == "__main__":
    unittest.main()
