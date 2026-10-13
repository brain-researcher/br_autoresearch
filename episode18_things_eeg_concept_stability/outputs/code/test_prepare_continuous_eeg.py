"""Three bounded bridge checks; run only in coordinator-owned Slurm compute.

These algebra/metadata fixtures do not select an empirical EEG filter and do
not constitute recovery or a synthetic research program.
"""
from dataclasses import replace
import unittest
from unittest.mock import Mock

import numpy as np

from continuous_fold_runner import _roster
from training_transforms import CommonSensorSpace
from prepare_continuous_eeg import (RecordingSegment, SequencePreparationRecipe,
                                    SequenceSpan, StimulusMarker, prepare_participant_sequences)


CHANNELS = tuple(f"channel_{index}" for index in range(62))
SOURCE_CHANNELS = tuple(reversed(CHANNELS)) + ("extra",)
RECIPE = SequencePreparationRecipe((.25, .5, .25), 1, 1, "none",
    "detrend_then_fir_then_decimate", 1000, 250, "V")


def fixture():
    sources, frozen, markers, spans = [], [], [], []
    for sequence in range(72):
        block, within = divmod(sequence, 6)
        start = sequence * 400 + 7
        span = SequenceSpan(f"physical_{sequence}", block, within, sequence,
            "segment_0" if sequence < 36 else "segment_1", start, start + 320, start, start + 320)
        spans.append(span)
        for presentation in range(309):
            zero = start + presentation + 5
            source = {"sample": str(zero + 1), "onset": format((zero + 1) / 1000, ".9f"),
                "time_stimon": format(zero / 1000 - (11 if sequence >= 36 else 0), ".9f"),
                "isteststim": "0", "blocksequencenumber": str(block),
                "sequencenumber": str(sequence), "withinsequencenumber": str(within),
                "presentationnumber": str(presentation), "objectnumber": str(presentation + 1),
                "stimnumber": str(block), "duration": "50", "stim": f"images/object/image_{presentation}.jpg"}
            row = {"participant": "fixture", "is_repeated": "0", "block": str(block),
                "physical_sequence": str(sequence), "presentation": str(presentation),
                "sample": source["sample"], "sample_zero_based": str(zero),
                "source_onset_seconds": source["onset"], "source_time_stimon_seconds": source["time_stimon"],
                "onset_recording_seconds": format(zero / 1000, ".9f"), "duration_seconds": "0.05",
                "concept_number": source["objectnumber"], "exemplar_number": source["stimnumber"],
                "canonical_image": f"object/image_{presentation}.jpg",
                "archive_member": f"exact/archive/object/image_{presentation}.jpg"}
            sources.append(source)
            frozen.append(row)
            markers.append(StimulusMarker(f"Mk{len(markers) + 1}", zero + 1))
    return dict(participant_id="fixture", main_source_events=sources, main_frozen_join=frozen,
        stimulus_markers=markers, recording_segments=(RecordingSegment("segment_0", 0, 14400),
        RecordingSegment("segment_1", 14400, 28800)), recording_sample_count=28800,
        sequence_spans=spans, original_channel_order=SOURCE_CHANNELS,
        frozen_channel_order=CHANNELS, recipe=RECIPE, budget_check=lambda: None)


def raw_reader(start, stop):
    sequence = (start - 7) // 400
    local = np.arange(stop - start, dtype=float)[:, None]
    values = sequence * 1000 + local ** 2 + 2 * np.arange(63)[None, :]
    values[:, -1] = np.nan  # Extra sensors are outside the fixed primary space.
    return values


class SequencePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arguments = fixture()

    def test_complete_roster_original_join_grid_and_projection_interface(self):
        reads = []

        def reader(start, stop):
            reads.append((start, stop))
            return raw_reader(start, stop)

        sources = list(self.arguments["main_source_events"])
        joins = list(self.arguments["main_frozen_join"])
        # Reuse the qualified onset rule: rounding source seconds at 1000 Hz
        # matches the one-based sample. Preserve the original string verbatim.
        onset = format(float(sources[0]["onset"]) + .0002, ".9f")
        sources[0] = {**sources[0], "onset": onset}
        joins[0] = {**joins[0], "source_onset_seconds": onset}
        sequences = prepare_participant_sequences(**{**self.arguments,
            "main_source_events": sources, "main_frozen_join": joins}, read_samples=reader)
        self.assertEqual(sequences[0].preparation_input.events[0].source_row["onset"], onset)
        self.assertEqual(len(sequences), 72)
        self.assertEqual(reads, [(s.start_sample, s.stop_sample) for s in self.arguments["sequence_spans"]])
        _roster(sequences, CommonSensorSpace(CHANNELS), lambda: None)
        for index, sequence in enumerate(sequences):
            self.assertEqual(sequence.channel_order, CHANNELS)
            self.assertEqual(sequence.sensor_samples.shape[1], 62)
            inputs = sequence.preparation_input
            self.assertEqual(len(inputs.events), 309)
            np.testing.assert_array_equal(inputs.onsets_seconds, (np.arange(309) + 5) / 1000)
            self.assertNotEqual(inputs.onsets_seconds[0] * 250, round(inputs.onsets_seconds[0] * 250))
            original = np.rint(sequence.sample_times_seconds * 1000).astype(int) + inputs.span.start_sample
            self.assertTrue(np.all(original % 4 == 1))
            np.testing.assert_allclose(np.diff(sequence.sample_times_seconds), .004, rtol=0, atol=1e-10)
            first_event = inputs.events[0]
            self.assertEqual(first_event.sample_zero_based, inputs.span.start_sample + 5)
            self.assertEqual(first_event.recording_segment_id, inputs.span.recording_segment_id)
            self.assertEqual(first_event.frozen_row["archive_member"], "exact/archive/object/image_0.jpg")
            self.assertEqual(first_event.source_row["time_stimon"], self.arguments["main_source_events"][index * 309]["time_stimon"])
            with self.assertRaises(TypeError):
                first_event.source_row["sample"] = "0"
            self.assertFalse(sequence.retained_mask.flags.writeable)
            self.assertFalse(sequence.sensor_samples.flags.writeable)
            self.assertFalse(inputs.onsets_seconds.flags.writeable)

    def test_finite_support_mask_sequence_local_filter_and_fixed_detrend(self):
        # An asymmetric five-tap filter has left support 3 and right support 1.
        recipe = replace(RECIPE, fir_coefficients=(1., 2., 3., 4., 5.), fir_origin_sample=1)
        arguments = {**self.arguments, "recipe": recipe}
        sequences = prepare_participant_sequences(**arguments, read_samples=raw_reader)
        for sequence in sequences:
            span = sequence.preparation_input.span
            local = np.rint(sequence.sample_times_seconds * 1000).astype(int)
            expected_mask = (local >= 3) & (local < 319)
            np.testing.assert_array_equal(sequence.retained_mask, expected_mask)
            self.assertFalse(sequence.retained_mask[0])
            for output_row in np.flatnonzero(expected_mask):
                n = local[output_row]
                # Independent direct sum, with source columns reversed by fixed channel labels.
                expected = sum(tap * (span.physical_sequence * 1000 + (n + 1 - k) ** 2
                                      + 2 * np.arange(61, -1, -1))
                               for k, tap in enumerate(recipe.fir_coefficients))
                np.testing.assert_allclose(sequence.sensor_samples[output_row], expected, atol=1e-8)
        # A predeclared linear time projection removes an exactly affine local signal.
        def affine_reader(start, stop):
            local = np.arange(stop - start)[:, None]
            return np.broadcast_to(17 + local * 2 + start, (stop - start, 63)).copy()

        projected = prepare_participant_sequences(**{**self.arguments,
            "recipe": replace(RECIPE, detrend="fixed_linear_projection")}, read_samples=affine_reader)
        for sequence in projected:
            np.testing.assert_allclose(sequence.sensor_samples, 0, rtol=0, atol=1e-9)

    def test_invalid_join_boundary_and_recipe_refuse_before_raw_reads(self):
        spans = self.arguments["sequence_spans"]
        malformed_join = list(self.arguments["main_frozen_join"])
        malformed_join[0] = {**malformed_join[0], "sample_zero_based": "0"}
        repeated = list(self.arguments["main_source_events"])
        repeated[0] = {**repeated[0], "isteststim": "1"}
        no_score = list(spans)
        no_score[-1] = replace(no_score[-1], score_start_sample=no_score[-1].start_sample,
                               score_stop_sample=no_score[-1].start_sample + 1)
        cases = [
            {"sequence_spans": spans[:-1]},
            {"sequence_spans": [spans[0], replace(spans[1], start_sample=spans[0].stop_sample - 1)] + spans[2:]},
            {"sequence_spans": [replace(spans[0], recording_segment_id="segment_1")] + spans[1:]},
            {"sequence_spans": no_score},
            {"main_frozen_join": malformed_join},
            {"main_source_events": repeated},
            {"stimulus_markers": self.arguments["stimulus_markers"][1:]},
            {"recipe": replace(RECIPE, fir_origin_sample=3)},
            {"recipe": replace(RECIPE, decimation_phase_mod4=4)},
            {"recipe": replace(RECIPE, input_sample_unit="uV")},
            {"recipe": replace(RECIPE, detrend="adaptive")},
            {"frozen_channel_order": CHANNELS[:-1]},
        ]
        for change in cases:
            with self.subTest(fields=tuple(change)):
                reader = Mock(side_effect=AssertionError("Invalid metadata reached raw EEG"))
                with self.assertRaises(ValueError):
                    prepare_participant_sequences(**{**self.arguments, **change}, read_samples=reader)
                reader.assert_not_called()
        reader = Mock(return_value=np.full((320, 63), np.nan))
        with self.assertRaisesRegex(ValueError, "Nonfinite primary-channel EEG"):
            prepare_participant_sequences(**self.arguments, read_samples=reader)
        self.assertEqual(reader.call_count, 1)


if __name__ == "__main__":
    unittest.main()
