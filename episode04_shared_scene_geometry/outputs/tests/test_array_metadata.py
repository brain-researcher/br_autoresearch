"""Header byte boundary and metadata-to-volume axis checks; synthetic only."""
import gzip
import io
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from inspect_array_metadata import header_bytes, parse_header, require_trial_axis, small_attributes


def fixture(version, order='<'):
    size = 348 if version == 1 else 540
    raw = bytearray(size)
    struct.pack_into(order+'i', raw, 0, size)
    if version == 1:
        struct.pack_into(order+'8h', raw, 40, 4, 7, 8, 9, 1044, 1, 1, 1)
        struct.pack_into(order+'2h', raw, 70, 16, 32)
        struct.pack_into(order+'8f', raw, 76, 1, 2, 2, 2, 1, 0, 0, 0)
        struct.pack_into(order+'f', raw, 108, 352)
        raw[344:348] = b'n+1\0'
    else:
        raw[4:12] = b'n+2\0\r\n\x1a\n'
        struct.pack_into(order+'2h', raw, 12, 16, 32)
        struct.pack_into(order+'8q', raw, 16, 4, 7, 8, 9, 1044, 1, 1, 1)
        struct.pack_into(order+'8d', raw, 104, 1, 2, 2, 2, 1, 0, 0, 0)
        struct.pack_into(order+'q', raw, 168, 544)
    return bytes(raw)


class ArrayMetadataTests(unittest.TestCase):
    def test_actual_h5py_attribute_ids_are_closed_without_context_manager(self):
        import h5py
        with tempfile.TemporaryDirectory() as temporary:
            with h5py.File(Path(temporary)/'synthetic_metadata.h5', 'w') as source:
                source.attrs['model_name'] = 'synthetic-only-model'
                source.attrs['image_size'] = [224, 224]
                source.attrs['not_whitelisted'] = 'not returned'
                result = small_attributes(source)
                self.assertEqual(result, {'model_name': 'synthetic-only-model',
                                          'image_size': [224, 224]})

    def test_both_header_formats_stop_before_payload_for_both_endians(self):
        for version in (1, 2):
            for order in ('<', '>'):
                with self.subTest(version=version, order=order):
                    header = fixture(version, order)
                    stream = io.BytesIO(gzip.compress(header + b'FORBIDDEN_VOXEL_BODY'*10000))
                    raw, _ = header_bytes(stream)
                    self.assertEqual(raw, header)
                    parsed = parse_header(raw)
                    self.assertEqual(parsed['header_bytes_decompressed'], len(header))
                    self.assertEqual(parsed['shape'], [7, 8, 9, 1044])
                    self.assertFalse(parsed['voxel_values_decoded'])

    def test_session_axis_matches_full_tsv_rows_not_run_local_maximum(self):
        header = parse_header(fixture(1))
        require_trial_axis(header, 1044)
        with self.assertRaisesRegex(ValueError, 'fourth axis'):
            require_trial_axis(header, 87)
        with self.assertRaises(ValueError):
            require_trial_axis(header | {'shape': [7, 8, 9]}, 1044)


if __name__ == '__main__':
    unittest.main()
