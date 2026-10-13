"""Generated NIfTI source-boundary fixtures; no empirical data or scores."""
import gzip
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from prepare_development_neural_pilot import (
    decode_selected, mask_membership, open_header, parse_header, read_mask,
    stream_selected, validate_session_rows, SequentialNpyRows,
)


def header_bytes(shape, order='<'):
    header = bytearray(348)
    struct.pack_into(order + 'i', header, 0, 348)
    struct.pack_into(order + '8h', header, 40, len(shape), *shape, *([1] * (7 - len(shape))))
    struct.pack_into(order + '2h', header, 70, 16, 32)
    struct.pack_into(order + '8f', header, 76, *([1.] * 8))
    struct.pack_into(order + '3f', header, 108, 352., 1., 0.)
    header[344:348] = b'n+1\0'
    return bytes(header)


def progress():
    return {key: 0 for key in (
        'source_volumes_consumed', 'uncompressed_beta_payload_bytes_consumed',
        'selected_volumes_decoded', 'excluded_volumes_discarded_as_opaque_bytes',
        'selected_roi_nonfinite_values', 'selected_roi_zero_values')}


class NeuralHandoffTest(unittest.TestCase):
    def test_only_selected_session_rows_decode_and_spatial_identity_is_preserved(self):
        # Rectangular spatial axes distinguish F storage from C voxel identity.
        shape = (2, 3, 2, 4)
        first = np.arange(12, dtype=np.float32).reshape(shape[:3])
        last = first + 100
        first[0, 0, 0] = 0
        last[1, 2, 1] = np.nan
        selected = np.array([0, 3, 5, 8, 11])
        rows = [{'session_beta_axis_index': 3, 'beta_index': 0},
                {'session_beta_axis_index': 0, 'beta_index': 0}]
        for order in ('<', '>'):
            with self.subTest(endian=order):
                dtype = np.dtype(order + 'f4')
                allowed = [first.astype(dtype).tobytes(order='F'), last.astype(dtype).tobytes(order='F')]
                opaque = b'x' * 48
                source = io.BytesIO(allowed[0] + opaque + opaque + allowed[1])
                decoded = []
                def guarded(raw, source_dtype, indices):
                    self.assertEqual(raw, allowed[len(decoded)])
                    self.assertNotEqual(raw, opaque)
                    decoded.append(raw)
                    return decode_selected(raw, source_dtype, indices)
                destination = np.empty((2, 5), dtype=np.float32)
                state = progress()
                stream_selected(source, shape, dtype, rows, selected, destination, state, decoder=guarded)
                np.testing.assert_array_equal(destination[0], first.ravel(order='C')[selected])
                np.testing.assert_array_equal(destination[1], last.ravel(order='C')[selected])
                self.assertEqual(len(decoded), 2)
                self.assertEqual(state['source_volumes_consumed'], 4)
                self.assertEqual(state['excluded_volumes_discarded_as_opaque_bytes'], 2)
                self.assertEqual(state['selected_roi_nonfinite_values'], 1)
                self.assertEqual(state['selected_roi_zero_values'], 1)
                # The bounded-memory disk writer uses the same frozen row order.
                output = io.BytesIO()
                disk_rows = SequentialNpyRows(output, destination.shape)
                disk_rows[0] = destination[0]
                disk_rows[1] = destination[1]
                output.seek(0)
                np.testing.assert_array_equal(np.load(output, allow_pickle=False), destination)

    def test_changed_header_fails_before_payload_is_read(self):
        raw = header_bytes((2, 3, 2, 4))
        expected = parse_header(raw)
        expected['shape'] = [2, 3, 2, 5]
        source = io.BytesIO(raw + b'\0' * 4 + b'closed-payload')
        with self.assertRaisesRegex(ValueError, 'frozen structural'):
            open_header(source, expected)
        self.assertEqual(source.tell(), 348)

    def test_mask_has_same_spatial_order_and_preserves_nan_zero_semantics(self):
        shape = (2, 3, 2)
        values = np.zeros(shape, dtype=np.float32)
        values[0, 2, 1], values[1, 0, 0], values[1, 2, 1] = 1., -2., np.nan
        header = header_bytes(shape)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'generated.nii.gz'
            with gzip.open(path, 'wb') as target:
                target.write(header + b'\0' * 4 + values.tobytes(order='F'))
            membership, nan_count = read_mask(path, parse_header(header))
        expected = np.zeros(12, dtype=bool)
        expected[[5, 6]] = True
        np.testing.assert_array_equal(membership, expected)
        self.assertEqual(nan_count, 1)
        with self.assertRaisesRegex(ValueError, 'Infinite'):
            mask_membership(np.array([0., np.inf]))

    def test_truncation_retains_partial_accounting_and_cannot_complete(self):
        source = io.BytesIO(np.arange(12, dtype=np.float32).tobytes())
        state = progress()
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            stream_selected(source, (2, 3, 2, 2), np.dtype('<f4'),
                            [{'session_beta_axis_index': 1}], np.array([0]),
                            np.empty((1, 1), dtype=np.float32), state)
        self.assertEqual(state['source_volumes_consumed'], 1)
        self.assertEqual(state['selected_volumes_decoded'], 0)
        self.assertEqual(state['excluded_volumes_discarded_as_opaque_bytes'], 1)

    def test_duplicate_or_out_of_axis_rows_fail_before_streaming(self):
        for indices in ([0, 0], [-1], [4], [1.0]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                validate_session_rows([{'session_beta_axis_index': x} for x in indices], 4)


if __name__ == '__main__':
    unittest.main()
