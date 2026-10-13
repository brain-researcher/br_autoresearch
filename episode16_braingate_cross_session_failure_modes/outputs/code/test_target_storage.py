"""Synthetic header-only access checks; run inside coordinator-owned compute."""
import io
import struct
import unittest
from unittest.mock import patch
import zlib

import inspect_target_storage as reader


def field(kind, body, endian, pad=True):
    return struct.pack(endian + "II", kind, len(body)) + body + (b"\0" * (-len(body) % 8) if pad else b"")


def matrix(name, endian, single=False):
    dtype_tag, array_class, itemsize = (7, 7, 4) if single else (9, 6, 8)
    metadata = field(6, struct.pack(endian + "II", array_class, 0), endian)
    metadata += field(5, struct.pack(endian + "ii", 1000, 2), endian)
    metadata += field(1, name.encode(), endian)
    body = b"\xa5" * (2000 * itemsize)  # Opaque synthetic bytes, never interpreted.
    return field(14, metadata + field(dtype_tag, body, endian), endian), len(metadata), len(body)


def file_header(endian):
    return (b"MATLAB 5.0 MAT-file synthetic EP16 metadata fixture".ljust(116, b" ")
            + b"\0" * 8 + struct.pack(endian + "H", 256) + (b"IM" if endian == "<" else b"MI"))


class GuardedBytes(io.BytesIO):
    def __init__(self, raw, forbidden):
        super().__init__(raw)
        self.forbidden = forbidden

    def read(self, count=-1):
        start = self.tell()
        stop = len(self.getbuffer()) if count < 0 else start + count
        for left, right in self.forbidden:
            if start < right and stop > left:
                raise AssertionError("Attempted read of a numeric body")
        return super().read(count)


class TargetStorageTests(unittest.TestCase):
    def test_uncompressed_bodies_never_read_for_both_precisions_and_byte_orders(self):
        for endian in ("<", ">"):
            for single in (False, True):
                with self.subTest(endian=endian, single=single):
                    neural, nmeta, nbody = matrix("neural", endian)
                    target, tmeta, tbody = matrix("target_position", endian, single)
                    first_body = 128 + 8 + nmeta + 8
                    second_body = 128 + len(neural) + 8 + tmeta + 8
                    source = GuardedBytes(file_header(endian) + neural + target,
                        [(first_body, first_body + nbody), (second_body, second_body + tbody)])
                    actual = reader.inspect_session(source)
                    self.assertEqual(actual["shape"], [1000, 2])
                    self.assertEqual(actual["storage_dtype"], endian + ("f4" if single else "f8"))
                    self.assertEqual(actual["numeric_values_decoded"], 0)
                    self.assertEqual(actual["declared_numeric_body_bytes"], tbody)

    def test_compressed_matrix_inflation_stops_at_metadata(self):
        endian = "<"
        neural, nmeta, _ = matrix("neural", endian)
        target, tmeta, _ = matrix("target_position", endian, single=True)
        source = io.BytesIO(file_header(endian)
            + field(15, zlib.compress(neural), endian, pad=False)
            + field(15, zlib.compress(target), endian, pad=False))
        observed = []
        base = reader.Inflated

        class CountedInflated(base):
            def __init__(self, compressed):
                super().__init__(compressed)
                self.total = 0
                observed.append(self)

            def read(self, count):
                raw = super().read(count)
                self.total += len(raw)
                return raw

        with patch.object(reader, "Inflated", CountedInflated):
            actual = reader.inspect_session(source)
        self.assertEqual(actual["storage_dtype"], "<f4")
        self.assertEqual([stream.total for stream in observed], [8 + nmeta, 8 + tmeta + 8])


if __name__ == "__main__":
    unittest.main()
