"""Bounded container fixtures only; no scientific world generation or reads."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from acquisition_readout import RawValue, PublishedValue
from array_acquisition import IntervalEvidence
from streaming_simulation_output import StreamingSimulationOutput


def sink(path, **overrides):
    settings = dict(settings_reference="explicit_fixture_only", max_payload_bytes=1_000_000,
                    max_records=10000, buffer_records=5, identifier_bytes=32,
                    max_dependencies_per_observation=2)
    settings.update(overrides)
    writer = StreamingSimulationOutput(path, **settings)
    writer.begin(state_ids=("pad", "hold"), initial_state_volts=np.zeros(2),
                 initial_field_volts=np.zeros(4096), metadata={"fixture": True})
    return writer


class StreamingSimulationOutputTests(unittest.TestCase):
    def test_long_stream_roundtrips_exact_records_with_bounded_shared_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream.h5"
            with sink(path) as writer:
                for index in range(100):
                    raw_id, valid = f"r:{index}", index % 2 == 0
                    writer.emit_raw(RawValue(raw_id, .25, 1, .5, valid, False))
                    writer.emit_observation(PublishedValue(f"o:{index}", "位点", "pixel",
                        .5 if valid else None, valid, False, float(index), float(index + 1),
                        ((raw_id, float(index), float(index + 1)),)))
                    writer.emit_interval(IntervalEvidence(index, float(index), float(index + 1), index % 2 == 0))
                result = writer.finish(status="complete", final_state_volts=np.ones(2),
                    final_field_volts=np.zeros(4096), metadata={"physical_complete": True})
                self.assertLessEqual(result["maximum_buffered_records"], 5)
                self.assertEqual(result["record_counts"], dict(raw_values=100, observations=100,
                                                             aperture_support=100, interval_evidence=100))
            with h5py.File(path, "r") as saved:
                self.assertTrue(saved.attrs["record_complete"])
                self.assertEqual(saved.attrs["simulation_status"], "complete")
                observations = saved["observations"][...]
                self.assertEqual(observations[0]["site_id"].decode(), "位点")
                self.assertEqual(observations[-1]["support_start"], 99)
                self.assertEqual(saved["aperture_support"][99]["conversion_id"], b"r:99")
                self.assertTrue(np.isnan(observations[1]["value_volts"]))
                self.assertFalse(observations[1]["has_value"])
                self.assertEqual(saved["raw_values"][99]["code_index"], 1)
                self.assertFalse(saved["interval_evidence"][99]["event_boundary_charge_applied"])
                metadata = json.loads(saved["terminal_metadata_json"][...].tobytes())
                self.assertFalse(metadata["full_history_materialized"])
                np.testing.assert_array_equal(saved["final_state_volts"][...], np.ones(2))

    def test_identifier_overflow_refuses_without_truncation_or_later_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overflow.h5"
            with sink(path, identifier_bytes=4) as writer:
                with self.assertRaisesRegex(ValueError, "no truncation"):
                    writer.emit_raw(RawValue("r:1234", .1, 0, 0., True, False))
                with self.assertRaisesRegex(RuntimeError, "no replay"):
                    writer.finish(status="complete", final_state_volts=np.zeros(2),
                                  final_field_volts=np.zeros(4096), metadata={})
            with h5py.File(path, "r") as saved:
                self.assertFalse(saved.attrs["record_complete"])
                self.assertEqual(len(saved["raw_values"]), 0)
            with self.assertRaises(FileExistsError):
                sink(path)

    def test_record_cap_preserves_flushed_prefix_and_failed_run_is_distinct_from_incomplete_file(self):
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "partial.h5"
            with sink(partial, max_records=3, buffer_records=3) as writer:
                for index in range(3):
                    writer.emit_raw(RawValue(f"r:{index}", .1, 0, 0., True, False))
                with self.assertRaisesRegex(ValueError, "record cap"):
                    writer.emit_raw(RawValue("r:3", .1, 0, 0., True, False))
            with h5py.File(partial, "r") as saved:
                self.assertEqual(len(saved["raw_values"]), 3)
                self.assertFalse(saved.attrs["record_complete"])
            failed = Path(directory) / "failed.h5"
            with sink(failed) as writer:
                writer.emit_raw(RawValue("r:0", .1, 0, 0., True, False))
                writer.finish(status="failed", final_state_volts=np.ones(2),
                    final_field_volts=np.zeros(4096), metadata={"failure": "solver stopped"})
            with h5py.File(failed, "r") as saved:
                self.assertTrue(saved.attrs["record_complete"])
                self.assertEqual(saved.attrs["simulation_status"], "failed")
                self.assertEqual(len(saved["raw_values"]), 1)


if __name__ == "__main__":
    unittest.main()
