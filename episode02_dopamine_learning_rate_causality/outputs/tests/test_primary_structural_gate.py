"""Synthetic known-answer checks for the steward's primary structural gate.

The fixture is a scipy-written MAT v5 file with the frozen 20-field, 1 x 24
struct layout and invented values; it is not an observation. It checks that
the streaming parser matches `loadmat` on the decoded fields, returns only a
finiteness mask when asked, never inflates past record 20, and that the
per-mouse checks and arm aggregates give the expected answers.
Run only inside a Slurm allocation.
"""
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib

import numpy as np
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import materialize_roles as trusted  # noqa: E402
import primary_structural_gate as gate  # noqa: E402


def synthetic_records(rng):
    records = np.empty((1, gate.COLUMNS), dtype=[(field, object) for field in trusted.EXPECTED_FIELDS])
    for column in range(gate.COLUMNS):
        rows = 3 if column == 1 else int(rng.integers(20, 60))  # three rows exercise small data elements
        state = rng.integers(0, 2, (1, rows)).astype(np.uint8)
        if column % 3 == 0:  # float storage with lost rows, as in the release
            state = state.astype(np.float64)
            state[0, : rows // 4] = np.nan
        element = records[0, column]
        for field in trusted.EXPECTED_FIELDS:
            element[field] = rng.normal(size=(rows, 7)).astype(np.float32)
        element["trialID"] = rng.integers(0, 3, (1, rows)).astype(np.int8)
        element["seshID"] = np.sort(rng.integers(1, 9, rows)).astype(np.uint8).reshape(1, rows)
        element["lickState"] = state
        element["stimState"] = rng.integers(0, 2, (1, rows)).astype(np.uint8)
    return records


def same_array(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


class StreamingParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name) / "fixture.mat"
        savemat(cls.path, {gate.VARIABLE: synthetic_records(np.random.default_rng(20261016))},
                do_compression=True, format="5")
        cls.loaded = loadmat(cls.path, struct_as_record=True, squeeze_me=False)[gate.VARIABLE]
        with open(cls.path, "rb") as stream:
            stream.read(128)
            _, nbytes = struct.unpack("<II", stream.read(8))
            cls.inflated_size = len(zlib.decompress(stream.read(nbytes)))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_decoded_fields_match_loadmat_and_inflation_stops_at_record_20(self):
        plan = {record: {"trialID": "values", "seshID": "values",
                         "lickState": "values" if record % 2 else "finite"} for record in range(1, 21)}
        with open(self.path, "rb") as stream:
            source = gate.open_variable(stream)
            decoded = gate.decode_records(source, trusted.EXPECTED_FIELDS, plan, gate.LAST_DECODED_RECORD)
        self.assertEqual(source.produced, source.position)
        self.assertLess(source.position, self.inflated_size)
        self.assertEqual(sorted(decoded), list(range(1, 21)))
        for record in range(1, 21):
            expected = self.loaded[0, record - 1]
            self.assertEqual(set(decoded[record]), {"trialID", "seshID", "lickState"})
            for field in ("trialID", "seshID"):
                self.assertTrue(same_array(decoded[record][field], expected[field]))
            if record % 2:
                self.assertTrue(same_array(decoded[record]["lickState"], expected["lickState"]))
            else:
                mask = decoded[record]["lickState"]
                self.assertEqual(mask.dtype, bool)
                np.testing.assert_array_equal(mask, np.isfinite(expected["lickState"]).reshape(-1))

    def test_skipping_everything_still_walks_to_record_20(self):
        with open(self.path, "rb") as stream:
            source = gate.open_variable(stream)
            self.assertEqual(gate.decode_records(source, trusted.EXPECTED_FIELDS, {}, gate.LAST_DECODED_RECORD), {})
        self.assertEqual(source.produced, source.position)
        self.assertLess(source.position, self.inflated_size)

    def test_schema_mismatch_is_refused(self):
        with open(self.path, "rb") as stream, self.assertRaises(ValueError):
            gate.open_variable(stream, fields=tuple(reversed(trusted.EXPECTED_FIELDS)))


class SupportTests(unittest.TestCase):
    days = np.repeat(np.arange(1, 9), 40)
    codes = np.where(np.arange(320) % 10 == 0, 1, 0)
    finite = np.ones(320, dtype=bool)

    def test_known_answers(self):
        checks, counts = gate.mouse_support(self.codes, self.days, self.finite)
        self.assertTrue(all(checks.values()))
        self.assertEqual((counts["eligible_trials_session_1"], counts["eligible_trials_session_8"]), (36, 36))
        early = self.days < 8
        checks, _ = gate.mouse_support(self.codes[early], self.days[early], self.finite[early])
        self.assertFalse(checks["sessions_1_and_8_present"])
        self.assertFalse(checks["at_least_30_eligible_in_sessions_1_and_8"])
        bad_codes = self.codes.copy()
        bad_codes[5] = 3
        self.assertFalse(gate.mouse_support(bad_codes, self.days, self.finite)[0]["trialID_codes_within_0_1_2"])
        lost = self.finite.copy()
        lost[:10] = False  # nine code-0 rows lost in session 1 leave 27 eligible
        checks, counts = gate.mouse_support(self.codes, self.days, lost)
        self.assertFalse(checks["at_least_30_eligible_in_sessions_1_and_8"])
        self.assertEqual(counts["eligible_trials_session_1"], 27)
        self.assertEqual(counts["nonfinite_lickState_rows_in_sessions_1_and_8"], 10)
        self.assertEqual(counts["sessions_with_nonfinite_lickState_rows"], 1)
        reordered = self.days.copy()
        reordered[[0, -1]] = reordered[[-1, 0]]
        self.assertFalse(gate.mouse_support(self.codes, reordered, self.finite)[0]["seshID_contiguous_1_to_K_in_row_order"])
        self.assertFalse(any(gate.mouse_support(self.codes[:-1], self.days, self.finite)[0].values()))

    def test_arm_summary_is_aggregate_counts_only(self):
        lost = self.finite.copy()
        lost[:10] = False
        summary = gate.arm_summary([gate.mouse_support(self.codes, self.days, self.finite),
                                    gate.mouse_support(self.codes, self.days, lost)])
        self.assertTrue(all(type(value) is int for value in summary.values()))
        self.assertEqual((summary["mice"], summary["mice_meeting_all_criteria"]), (2, 1))
        self.assertEqual(summary["eligible_trials_session_1_total"], 36 + 27)


if __name__ == "__main__":
    unittest.main()
