"""Focused tests of the source's structural access boundary; run on compute."""
import io
import struct
import unittest

import numpy as np
from scipy.io import savemat

from mat5_structural import StructuralReader, element, read_structural


def fixture(compressed, target_shape=(20, 2)):
    target = np.arange(np.prod(target_shape), dtype=float).reshape(target_shape)
    fields = {
        # Deliberately precedes trials: the reader must resolve onsets first.
        "target_position": target,
        "neural": {"sbp": np.ones((20, 2), dtype=complex)},
        "cursor_position": np.ones((20, 2), dtype=complex),
        "participant_id": "T10", "post_implant_day": 104, "bin_size": .01,
        "electrodes": {"electrode_id": np.array([0, 1]),
                       "array": np.array(["dPCG", "dPCG"], dtype=object),
                       "x": np.array([1, 2]), "y": np.array([1, 1]),
                       "impedance": np.ones((2, 1), dtype=complex)},
        "blocks": {"block_id": [1, 2], "sample_start_index": [0, 10],
                   "sample_end_index": [10, 20]},
        "trials": {"trial_start_index": [1, 4, 12], "trial_end_index": [3, 8, 18],
                   "block_id": [1, 1, 2], "success": np.ones(3, dtype=complex)},
    }
    output = io.BytesIO()
    savemat(output, fields, do_compression=compressed)
    return output, target


class StructuralAccessTests(unittest.TestCase):
    def test_compressed_and_uncompressed_field_whitelist(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                source, target = fixture(compressed)
                result, trace = read_structural(source)
                np.testing.assert_array_equal(result["trial_targets"], target[[1, 4, 12]])
                self.assertEqual(result["participant_id"], "T10")
                self.assertEqual(result["electrodes"]["array"], ["dPCG", "dPCG"])
                self.assertEqual(trace["target_rows_decoded"], 3)
                self.assertFalse(trace["neural_cursor_impedance_values_decoded"])
                for forbidden in ("neural", "cursor_position", "electrodes.impedance", "trials.success"):
                    self.assertIn(forbidden, trace["skipped_paths"])
                    self.assertNotIn(forbidden, trace["decoded_paths"])
                self.assertNotIn("impedance", result["electrodes"])
                self.assertNotIn("success", result["trials"])

    def test_selected_complex_field_fails_closed(self):
        source, _ = fixture(True)
        with self.assertRaisesRegex(ValueError, "Complex selected field"):
            StructuralReader(source).read({"cursor_position": None})

    def test_wrong_target_shape_fails_closed(self):
        source, _ = fixture(True, (20, 3))
        with self.assertRaisesRegex(ValueError, "n_time_bins x 2"):
            read_structural(source)

    def test_regular_and_small_tags_in_both_byte_orders(self):
        for endian in ("<", ">"):
            regular = io.BytesIO(struct.pack(endian+"II", 5, 4)+b"abcd"+b"\0"*4)
            item = element(regular, endian)
            self.assertEqual(item.kind, 5)
            self.assertEqual(item.data.read(4), b"abcd")
            item.finish()
            self.assertEqual(regular.tell(), 16)
            small = io.BytesIO(struct.pack(endian+"HH", 1, 3)+b"abc\0")
            item = element(small, endian)
            self.assertEqual(item.kind, 1)
            self.assertEqual(item.data.read(3), b"abc")


if __name__ == "__main__":
    unittest.main()
