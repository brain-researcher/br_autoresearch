"""Metadata reader fixtures with intentionally invalid waveform payloads."""
from pathlib import Path
import struct
import sys
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from read_way_cue_metadata import read_cues


def element(kind, payload):
    return struct.pack("<II", kind, len(payload)) + payload + b"\0" * (-len(payload) % 8)


def header(cls, count, name=""):
    return element(6, struct.pack("<II", cls, 0)) + element(5, struct.pack("<ii", 1, count)) + element(1, name.encode())


def scalar(value):
    return element(14, header(6, 1) + element(9, struct.pack("<d", value)))


def structure(name, entries):
    names = list(entries[0])
    width = max(map(len, names)) + 1
    table = b"".join(field.encode().ljust(width, b"\0") for field in names)
    payload = header(2, len(entries), name) + struct.pack("<HHI", 5, 4, width) + element(1, table)
    payload += b"".join(entry[field] for entry in entries for field in names)
    return element(14, payload)


def fixture(compressed=False, missing=False):
    # Waveform elements are deliberately not valid matrices internally. A
    # metadata reader that enters one cannot pass this fixture.
    poison = element(14, b"DO NOT PARSE ANY WAVEFORM VALUES")
    entries = [{"trial_start_time": scalar(start), "LEDon": scalar(2),
                "LEDoff": scalar(5.5), "eeg": poison, "kin": poison,
                "tHandStart": poison} for start in (0, 8)]
    if missing:
        for entry in entries:
            del entry["LEDoff"]
    ws = structure("ws", [{"win": structure("", entries), "name": poison}])
    prefix = b"MATLAB 5.0 MAT-file".ljust(124, b" ") + b"\0\1IM"
    if compressed:
        compressed_bytes = zlib.compress(ws)
        # miCOMPRESSED top-level payloads have no8-byte padding in MATLAB.
        return prefix + struct.pack("<II", 15, len(compressed_bytes)) + compressed_bytes
    return prefix + ws


class CueMetadataTests(unittest.TestCase):
    def test_only_three_scalar_fields_from_each_window(self):
        trials, report = read_cues(fixture())
        self.assertEqual(trials, [{"trial_start_time": 0., "LEDon": 2., "LEDoff": 5.5},
                                  {"trial_start_time": 8., "LEDon": 2., "LEDoff": 5.5}])
        self.assertEqual(report["LED_scalar_fields_decoded"], 6)
        self.assertGreater(report["unrequested_matrix_payload_bytes_skipped"], 0)
        self.assertFalse(report["waveform_numeric_arrays_decoded"])
        self.assertFalse(report["provider_movement_event_values_decoded"])

    def test_compressed_field_only_read(self):
        self.assertEqual(read_cues(fixture(True))[0], read_cues(fixture())[0])
        self.assertTrue(read_cues(fixture(True))[1]["compressed_matrix_bytes_materialized"])

    def test_missing_metadata_and_truncated_payload_fail_without_fallback(self):
        with self.assertRaisesRegex(ValueError, "Required official LED"):
            read_cues(fixture(missing=True))
        with self.assertRaises(ValueError):
            read_cues(fixture()[:-12])


if __name__ == "__main__":
    unittest.main()
