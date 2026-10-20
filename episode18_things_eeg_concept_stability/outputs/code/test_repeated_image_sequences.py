"""Focused repeated-role adapter checks; run only in coordinated compute."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import repeated_image_sequences as repeated
from brainvision_sequence_reader import BrainVisionMetadata, DevelopmentRecordingPaths, RecordedMarker
from prepare_continuous_eeg import RecordingSegment, StimulusMarker


def fixture():
    """Compact algebra fixture, not an empirical recording or scientific study."""
    person = "sub-07"
    paths = DevelopmentRecordingPaths(person, Path("unused.vhdr"), Path("unused.vmrk"),
        Path("unused.eeg"), Path("unused_events.tsv"), Path("unused_join.tsv.gz"))
    channels = tuple(f"channel_{index:02d}" for index in range(62))
    source, frozen, cues, stimuli = [], [], [], []
    for sequence in range(12):
        # Vary E3 alignment so decimation must preserve the recording grid.
        start = sequence * 5003
        cues.append(RecordedMarker(f"cue{sequence}", "Event", "E  3", start, 1, 0))
        for position in range(200):
            image = (position + 7 * sequence) % 200
            zero = start + 1000 + 10 * position
            sample = zero + 1
            identity = f"object{image:03d}/image{image:03d}.jpg"
            row = dict(isteststim="1", sample=str(sample), onset=str(sample/1000.),
                time_stimon=str(zero/1000. + 20), duration="50", stim="images/"+identity,
                istarget=str(position % 7 == 0 and 1 or 0), response=str(position % 9 == 0 and 1 or 0),
                teststimnumber=str(image), sequencenumber="n/a", presentationnumber=str(position))
            source.append(row)
            frozen.append(dict(participant=person, is_repeated="1", sample=str(sample),
                sample_zero_based=str(zero), canonical_image=identity,
                source_onset_seconds=row["onset"], source_time_stimon_seconds=row["time_stimon"],
                onset_recording_seconds=str(zero/1000.), duration_seconds="0.05"))
            stimuli.append(StimulusMarker(f"stim{sequence}_{position}", sample))
    count = 12 * 5003
    metadata = BrainVisionMetadata(paths, channels, (1.,)*62, ("V",)*62,
        (1.,)*62, 1000., count, count*62*4, (RecordingSegment("segment0", 0, count),),
        tuple(stimuli), tuple(cues))
    taps = np.zeros(129)
    taps[0], taps[128] = .25, .75
    return metadata, source, frozen, channels, tuple(taps)


def bind(values):
    return repeated._bind_rows(*values, lambda: None)


class TestRepeatedImageSequences(unittest.TestCase):
    def test_complete_identity_clock_role_and_chronological_parity(self):
        values = fixture()
        bound = bind(values)
        self.assertEqual(bound.participant_id, "sub-07")
        self.assertEqual(len(bound.image_ids), 200)
        self.assertEqual([row.ordinal for row in bound.sequences], list(range(12)))
        self.assertEqual(len({row.sequence_id for row in bound.sequences}), 12)
        for index, row in enumerate(bound.sequences):
            self.assertEqual(row.start_sample, index*5003)
            self.assertEqual(row.score_start_sample, row.event_samples[0]-800)
            self.assertEqual(row.stop_sample, row.event_samples[-1]+801)
            self.assertEqual(sorted(row.image_indices), list(range(200)))
        self.assertFalse(bound.metadata_summary["EEG_amplitudes_read"])
        self.assertFalse(bound.metadata_summary["audit_access"])
        self.assertEqual(bound.metadata_summary["optional_source_fields"]["sequencenumber"]["missing_value_count"], 2400)

    def test_exact_join_coverage_task_flags_and_markers_reject_corruption(self):
        metadata, source, frozen, channels, taps = fixture()
        cases = []
        changed = copy.deepcopy(frozen); changed[0]["canonical_image"] = "wrong/image.jpg"
        cases.append((metadata, source, changed, channels, taps))
        changed = copy.deepcopy(frozen); changed[0]["sample_zero_based"] = str(int(changed[0]["sample_zero_based"])+1)
        cases.append((metadata, source, changed, channels, taps))
        changed = copy.deepcopy(source); changed[0]["response"] = "n/a"
        cases.append((metadata, changed, frozen, channels, taps))
        cases.append((metadata, source[:-1], frozen[:-1], channels, taps))
        changed = copy.deepcopy(source); changed[0], changed[1] = changed[1], changed[0]
        cases.append((metadata, changed, frozen, channels, taps))
        changed = replace(metadata, stimulus_markers=metadata.stimulus_markers[1:])
        cases.append((changed, source, frozen, channels, taps))
        # Preserve the overall 200-by-12 counts but break each-sequence inventory.
        changed_source, changed_frozen = copy.deepcopy(source), copy.deepcopy(frozen)
        for name in ("stim",):
            changed_source[0][name], changed_source[200][name] = changed_source[200][name], changed_source[0][name]
        changed_frozen[0]["canonical_image"], changed_frozen[200]["canonical_image"] = (
            changed_frozen[200]["canonical_image"], changed_frozen[0]["canonical_image"])
        cases.append((metadata, changed_source, changed_frozen, channels, taps))
        for index, values in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                bind(values)

    def test_sequence_schema_and_complete_filter_support_are_not_guessed(self):
        metadata, source, frozen, channels, taps = fixture()
        # An extra cue divides a repeated stream: report actual counts, not a
        # disguised 72-main-sequence representation or silent event deletion.
        cues = list(metadata.markers)
        cues.insert(1, RecordedMarker("extra", "Event", "E  3", 2000, 1, 0))
        with self.assertRaisesRegex(ValueError, "schema unresolved"):
            bind((replace(metadata, markers=tuple(cues)), source, frozen, channels, taps))
        segments = (RecordingSegment("first", 0, 3500),
                    RecordingSegment("second", 3500, metadata.recording_sample_count))
        with self.assertRaisesRegex(ValueError, "support must stay inside"):
            bind((replace(metadata, recording_segments=segments), source, frozen, channels, taps))
        cues = (replace(metadata.markers[0], sample_zero_based=73),)+metadata.markers[1:]
        with self.assertRaisesRegex(ValueError, "support must stay inside"):
            bind((replace(metadata, markers=cues), source, frozen, channels, taps))

    def test_filter_is_causal_sequence_local_with_exact_recording_phase(self):
        bound = bind(fixture())
        def values(bound_arg, span, check):
            count = span.stop_sample-span.start_sample
            result = np.zeros((count, 62))
            result[0, :] = span.ordinal+1
            result[200, :] = 2.
            result[-1, :] = 1000.  # Must not flow into the next physical stream.
            return result
        with patch.object(repeated, "_read_interval", side_effect=values):
            person = repeated.load_bound_participant(bound, budget_check=lambda: None)
        for sequence, span in zip(person.sequences, bound.sequences):
            local = np.rint(sequence.sample_times_seconds*1000).astype(int)
            self.assertTrue(np.all((local+span.start_sample) % 4 == 0))
            self.assertTrue(np.all(local[sequence.retained_mask] >= 128))
            self.assertTrue(np.all((local+span.start_sample)[sequence.retained_mask] >= span.score_start_sample))
            expected = np.convolve(values(bound, span, None)[:, 0], bound.fir_coefficients)[:span.stop_sample-span.start_sample]
            np.testing.assert_allclose(sequence.sensor_samples[:, 0], expected[local], rtol=0, atol=1e-12)
            self.assertEqual(sequence.cue_seconds, 0.)
            self.assertEqual(sequence.final_offset_seconds, sequence.event_onsets_seconds[-1]+.05)
            self.assertFalse(sequence.sensor_samples.flags.writeable)
            self.assertFalse(sequence.retained_mask.flags.writeable)
        self.assertTrue(person.metadata_summary["EEG_amplitudes_read"])

    def test_raw_reader_exact_offset_scaling_and_refusal(self):
        bound = bind(fixture())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"record.eeg"
            # Sparse file with one explicit bound interval; unselected bytes
            # are never read by the interval reader.
            with path.open("wb") as stream:
                stream.truncate(bound.recording.binary_bytes)
            span = bound.sequences[1]
            count = span.stop_sample-span.start_sample
            values = np.arange(count*62, dtype=np.float32).reshape(count, 62)
            with path.open("r+b") as stream:
                stream.seek(span.start_sample*62*4)
                values.astype("<f4").tofile(stream)
            scales = tuple((index+1)*1e-6 for index in range(62))
            metadata = replace(bound.recording, paths=replace(bound.recording.paths, binary=path),
                               volts_per_stored_unit=scales)
            bound = replace(bound, recording=metadata)
            actual = repeated._read_interval(bound, span, lambda: None)
            np.testing.assert_array_equal(actual, values.astype(float)*np.asarray(scales))
            with self.assertRaisesRegex(ValueError, "exactly bound"):
                repeated._read_interval(bound, replace(span, start_sample=span.start_sample+1), lambda: None)


if __name__ == "__main__":
    unittest.main()
