"""Two cache/join regressions; run only in coordinator-owned Slurm compute.

Reuse the existing72-sequence bridge fixture. These check serialization and
refusal, not filtering qualification, mechanism recovery or a new simulation.
"""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from continuous_eeg_cache import (encode, load_prepared_participant,
                                   publish_prepared_participant)
from false_partition_search import BudgetStop
from prepare_continuous_eeg import prepare_participant_sequences
from test_prepare_continuous_eeg import fixture, raw_reader


class ContinuousEEGCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arguments = fixture()
        # Preserve the original text even when its seconds were rounded by
        # acquisition metadata; the bridge's qualified sample join still holds.
        cls.arguments["main_source_events"][0]["onset"] = "0.013200000"
        cls.arguments["main_frozen_join"][0]["source_onset_seconds"] = "0.013200000"
        cls.arguments["sequence_spans"] = [replace(span,
            score_start_sample=span.start_sample + 10, score_stop_sample=span.stop_sample - 3)
            for span in cls.arguments["sequence_spans"]]
        cls.sequences = prepare_participant_sequences(**cls.arguments, read_samples=raw_reader)

    def publish(self, root, name="person", check=lambda: None):
        return publish_prepared_participant(sequences=self.sequences,
            participant_id=self.arguments["participant_id"],
            recording_segments=self.arguments["recording_segments"],
            recording_sample_count=self.arguments["recording_sample_count"],
            recipe=self.arguments["recipe"], destination=root / name,
            local_stage=root / (name + "_stage"), remaining_durable_bytes=512 * 1024**2,
            check=check)

    def load(self, directory, participant="fixture"):
        return load_prepared_participant(directory, participant_id=participant,
            channel_order=self.arguments["frozen_channel_order"], recipe=self.arguments["recipe"],
            budget_check=lambda: None)

    def test_lossless_full_roster_clock_event_text_and_fold_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self.publish(root)
            restored = self.load(root / "person")
            self.assertEqual(report["sequences"], len(restored))
            self.assertEqual(len(restored), 72)
            self.assertGreater(report["full_grid_samples"], report["retained_samples"])
            for expected, actual in zip(self.sequences, restored):
                self.assertEqual((actual.sequence_id, actual.block, actual.within_block, actual.channel_order),
                    (expected.sequence_id, expected.block, expected.within_block, expected.channel_order))
                for name in ("sensor_samples", "sample_times_seconds", "retained_mask"):
                    np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
                    self.assertFalse(getattr(actual, name).flags.writeable)
                self.assertEqual(actual.preparation_input.span, expected.preparation_input.span)
                np.testing.assert_array_equal(actual.preparation_input.onsets_seconds,
                                              expected.preparation_input.onsets_seconds)
                np.testing.assert_array_equal(actual.preparation_input.source_sample_indices,
                                              expected.preparation_input.source_sample_indices)
                self.assertFalse(actual.preparation_input.onsets_seconds.flags.writeable)
                self.assertFalse(actual.preparation_input.source_sample_indices.flags.writeable)
                self.assertEqual(actual.preparation_input.events, expected.preparation_input.events)
            event = restored[0].preparation_input.events[0]
            self.assertEqual(event.source_row["onset"], "0.013200000")
            self.assertEqual(event.frozen_row["source_onset_seconds"], "0.013200000")
            self.assertEqual(event.sample_zero_based, 12)
            self.assertEqual(event.onset_sequence_seconds, .005)
            with self.assertRaises(TypeError):
                event.source_row["sample"] = "999"

    def test_wrong_identity_broken_join_roster_and_partial_publication_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.publish(root)
            with self.assertRaises(ValueError):
                self.load(root / "person", participant="different")
            metadata_path, report_path = root / "person/metadata.json", root / "person/complete.json"
            original = json.loads(metadata_path.read_text())
            report = json.loads(report_path.read_text())
            broken_join = copy.deepcopy(original)
            broken_join["sequences"][0]["events"][0]["frozen_row"]["sample_zero_based"] = "999"
            broken_roster = copy.deepcopy(original)
            broken_roster["sequences"].pop()
            for malformed in (broken_join, broken_roster):
                payload = encode(malformed)
                metadata_path.write_bytes(payload)
                changed_report = {**report, "files": {**report["files"], "metadata.json": len(payload)}}
                report_path.write_bytes(encode(changed_report))
                with self.assertRaises(ValueError):
                    self.load(root / "person")

            def stop_during_publication():
                if (root / "interrupted").exists():
                    raise BudgetStop("fixture publication stop")

            with self.assertRaises(BudgetStop):
                self.publish(root, name="interrupted", check=stop_during_publication)
            self.assertFalse((root / "interrupted/complete.json").exists())
            with self.assertRaises(FileNotFoundError):
                self.load(root / "interrupted")


if __name__ == "__main__":
    unittest.main()
