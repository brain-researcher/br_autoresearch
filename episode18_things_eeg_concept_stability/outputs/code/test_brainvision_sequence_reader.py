"""Bounded reader checks for coordinator-owned compute; no empirical EEG."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from brainvision_sequence_reader import (BrainVisionMetadata, DevelopmentRecordingPaths,
    RecordedMarker, RecordedSpanRecipe, SequenceVoltageReader, bind_development_paths,
    build_recorded_sequence_spans, read_recording_metadata)
from prepare_continuous_eeg import RecordingSegment, SequenceSpan, _events
from test_prepare_continuous_eeg import fixture as bridge_fixture


class BrainVisionSequenceReaderTests(unittest.TestCase):
    def test_header_scaled_float32_offsets_and_interval_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            participant = "sub-07"
            relative = f"{participant}/eeg/{participant}_task-rsvp_eeg"
            leaves = [relative + extension for extension in (".vhdr", ".vmrk", ".eeg")]
            manifest = root / "development_manifest.json"
            manifest.write_text(json.dumps({"role": "development", "participants": [participant],
                "EEG_relative_paths_for_controlled_mount": leaves,
                "frozen_event_join": "development_event_image_join.tsv.gz"}), encoding="utf-8")
            paths = bind_development_paths(development_manifest=manifest, bids_root=root,
                                           participant_id=participant)
            paths.header.parent.mkdir(parents=True)
            names = tuple(f"channel_{index}" for index in range(62))
            # Both resolution and explicit unit vary. The remaining channels
            # use the observed acquisition's documented resolution/unit pair.
            resolutions = (2., .5, 3.) + (.0488281,) * 59
            units = ("µV", "mV", "V") + ("µV",) * 59
            lines = ["Brain Vision Data Exchange Header File Version 1.0", "[Common Infos]",
                "Codepage=UTF-8", f"DataFile={paths.binary.name}", f"MarkerFile={paths.marker.name}",
                "DataFormat=BINARY", "DataOrientation=MULTIPLEXED", "NumberOfChannels=62",
                "SamplingInterval=1000", "[Binary Infos]", "BinaryFormat=IEEE_FLOAT_32", "[Channel Infos]"]
            lines.extend(f"Ch{index + 1}={name},,{resolution},{unit}"
                         for index, (name, resolution, unit) in enumerate(zip(names, resolutions, units)))
            paths.header.write_text("\n".join(lines) + "\n", encoding="utf-8")
            paths.marker.write_text("\n".join(("Brain Vision Data Exchange Marker File, Version 1.0",
                "[Common Infos]", "Codepage=UTF-8", f"DataFile={paths.binary.name}",
                "[Marker Infos]", "Mk1=New Segment,,1,1,0", "Mk2=Event,E  1,19,1,0")) + "\n", encoding="utf-8")
            values = np.arange(216 * 62, dtype=np.float32).reshape(216, 62) - 500
            values.astype("<f4").tofile(paths.binary)
            metadata = read_recording_metadata(paths, header_byte_limit=16384,
                marker_byte_limit=4096, budget_check=lambda: None)
            self.assertEqual(metadata.channel_order, names)
            self.assertEqual(metadata.resolutions, resolutions)
            self.assertEqual(metadata.source_units, units)
            self.assertEqual(metadata.recording_sample_count, 216)
            self.assertEqual(metadata.stimulus_markers[0].sample_one_based, 19)
            spans = tuple(SequenceSpan(f"physical_{index}", index // 6, index % 6, index,
                "Mk1", index * 3, index * 3 + 2, index * 3, index * 3 + 2) for index in range(72))
            reader = SequenceVoltageReader(metadata, spans, interval_byte_limit=2 * 62 * 12,
                                            budget_check=lambda: None)
            scales = np.array((2e-6, .5e-3, 3.) + (.0488281e-6,) * 59)
            expected = values[18:20].astype(float) * scales[None, :]
            np.testing.assert_allclose(reader(18, 20), expected, rtol=1e-15, atol=0)
            np.testing.assert_allclose(reader(0, 2), values[:2].astype(float) * scales, rtol=1e-15, atol=0)
            with self.assertRaisesRegex(ValueError, "prevalidated"):
                reader(19, 20)
            with self.assertRaises(MemoryError):
                SequenceVoltageReader(metadata, spans, interval_byte_limit=1,
                                      budget_check=lambda: None)(18, 20)
            # Missing units must not acquire an invented voltage conversion.
            paths.header.write_text("\n".join(lines).replace("Ch1=channel_0,,2.0,µV", "Ch1=channel_0,,2.0,") + "\n",
                                    encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "voltage unit"):
                read_recording_metadata(paths, header_byte_limit=16384,
                    marker_byte_limit=4096, budget_check=lambda: None)

    def test_all_72_recorded_cues_segments_and_explicit_span_refusals(self):
        existing = bridge_fixture()
        source = existing["main_source_events"]
        expected = existing["sequence_spans"]
        channels = existing["original_channel_order"]
        paths = DevelopmentRecordingPaths("fixture", Path("unused.vhdr"), Path("unused.vmrk"),
            Path("unused.eeg"), Path("unused_events.tsv"), Path("unused_join.tsv.gz"))
        cues = tuple(RecordedMarker(f"cue_{index}", "Event", "E  3", span.start_sample, 1, 0)
                     for index, span in enumerate(expected))
        # This reuses the existing 72x309 structural fixture; its short algebra
        # intervals are not a proposed empirical margin/filter recipe.
        metadata = BrainVisionMetadata(paths, channels, (1.,) * len(channels), ("V",) * len(channels),
            (1.,) * len(channels), 1000., existing["recording_sample_count"],
            existing["recording_sample_count"] * len(channels) * 4,
            existing["recording_segments"], tuple(existing["stimulus_markers"]), cues)
        selected = RecordedSpanRecipe(cue_anchor="preceding_recorded_E3", post_last_inclusive_samples=6,
                                     score_first_offset_samples=-5, score_last_inclusive_samples=6)
        spans = build_recorded_sequence_spans(metadata, source, recipe=selected)
        self.assertEqual(len(spans), 72)
        for actual, reference in zip(spans, expected):
            for field in ("physical_sequence", "block", "within_block", "recording_segment_id",
                          "start_sample", "stop_sample", "score_start_sample", "score_stop_sample"):
                self.assertEqual(getattr(actual, field), getattr(reference, field))
        with self.assertRaisesRegex(ValueError, "cue anchor explicitly"):
            build_recorded_sequence_spans(metadata, source, recipe=replace(selected, cue_anchor="nominal_delay"))
        with self.assertRaisesRegex(ValueError, "next recorded physical sequence"):
            build_recorded_sequence_spans(metadata, source, recipe=replace(selected, post_last_inclusive_samples=100))
        crossing_segment = replace(metadata, recording_segments=(RecordingSegment("short", 0, 100),
            RecordingSegment("remainder", 100, metadata.recording_sample_count)))
        with self.assertRaisesRegex(ValueError, "inside one physical sequence and segment"):
            build_recorded_sequence_spans(crossing_segment, source, recipe=selected)
        with self.assertRaisesRegex(ValueError, "309 ordered events"):
            build_recorded_sequence_spans(metadata, source[:-1], recipe=selected)

    def test_variable_recorded_cue_gaps_preserve_exact_frozen_join_and_negative_support(self):
        # Metadata-only regression for the four observed main-count cue gaps.
        # These are fixture rows, not acquired events or a selected EEG recipe.
        existing = bridge_fixture()
        gaps = (999, 1000, 1001, 1025)
        source, frozen, stimuli, cues = [], [], [], []
        for index, (old_source, old_frozen, old_marker) in enumerate(zip(
                existing["main_source_events"], existing["main_frozen_join"], existing["stimulus_markers"])):
            sequence, presentation = divmod(index, 309)
            cue = sequence * 40000 + 7
            zero = cue + gaps[sequence % len(gaps)] + presentation * 100
            onset = format((zero + 1) / 1000, ".9f")
            display = format(zero / 1000 - (11 if sequence >= 36 else 0), ".9f")
            source.append({**old_source, "sample": str(zero + 1), "onset": onset, "time_stimon": display})
            frozen.append({**old_frozen, "sample": str(zero + 1), "sample_zero_based": str(zero),
                "source_onset_seconds": onset, "source_time_stimon_seconds": display,
                "onset_recording_seconds": format(zero / 1000, ".9f")})
            stimuli.append(replace(old_marker, sample_one_based=zero + 1))
            if not presentation:
                cues.append(RecordedMarker(f"cue_{sequence}", "Event", "E  3", cue, 1, 0))
        channels = existing["original_channel_order"]
        paths = DevelopmentRecordingPaths("fixture", Path("unused.vhdr"), Path("unused.vmrk"),
            Path("unused.eeg"), Path("unused_events.tsv"), Path("unused_join.tsv.gz"))
        count = 72 * 40000
        segments = (RecordingSegment("segment_0", 0, count // 2),
                    RecordingSegment("segment_1", count // 2, count))
        metadata = BrainVisionMetadata(paths, channels, (1.,) * len(channels), ("V",) * len(channels),
            (1.,) * len(channels), 1000., count, count * len(channels) * 4,
            segments, tuple(stimuli), tuple(cues))
        selected = RecordedSpanRecipe(cue_anchor="preceding_recorded_E3", post_last_inclusive_samples=800,
                                     score_first_offset_samples=-800, score_last_inclusive_samples=800)
        spans = build_recorded_sequence_spans(metadata, source, recipe=selected)
        joined = _events("fixture", source, frozen, stimuli, spans, count)
        self.assertEqual(len(spans), 72)
        self.assertEqual(sum(map(len, joined.values())), 72 * 309)
        for sequence, span in enumerate(spans):
            events = joined[sequence]
            first, last = events[0].sample_zero_based, events[-1].sample_zero_based
            self.assertEqual(span.start_sample, cues[sequence].sample_zero_based)
            self.assertEqual(first - span.start_sample, gaps[sequence % len(gaps)])
            self.assertEqual(events[0].onset_sequence_seconds, gaps[sequence % len(gaps)] / 1000)
            self.assertEqual(span.score_start_sample, first - 800)
            self.assertEqual(span.score_stop_sample, last + 801)
            self.assertEqual(span.stop_sample, last + 801)
            self.assertGreaterEqual(span.score_start_sample - span.start_sample, 128)
            self.assertEqual(events[0].source_row["onset"], source[sequence * 309]["onset"])
            self.assertEqual(events[0].frozen_row["sample_zero_based"], frozen[sequence * 309]["sample_zero_based"])
        # E3 geometry never repairs a disagreeing E1 or frozen sample join.
        shifted = (replace(stimuli[0], sample_one_based=stimuli[0].sample_one_based + 1), *stimuli[1:])
        with self.assertRaisesRegex(ValueError, "exactly match its recorded E3/E1 interval"):
            build_recorded_sequence_spans(replace(metadata, stimulus_markers=shifted), source, recipe=selected)
        inserted = RecordedMarker("extra_cue", "Event", "E  3", stimuli[100].sample_one_based - 1, 1, 0)
        with self.assertRaisesRegex(ValueError, "exactly match its recorded E3/E1 interval"):
            build_recorded_sequence_spans(replace(metadata, markers=(cues[0], inserted, *cues[1:])),
                                          source, recipe=selected)
        mismatched = [{**frozen[0], "sample_zero_based": str(int(frozen[0]["sample_zero_based"]) + 1)}, *frozen[1:]]
        with self.assertRaisesRegex(ValueError, "frozen join"):
            _events("fixture", source, mismatched, stimuli, spans, count)


if __name__ == "__main__":
    unittest.main()
