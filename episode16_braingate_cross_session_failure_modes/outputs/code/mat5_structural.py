"""Field-selective Level-5 MAT reader for EP16 structural support.

Format source: https://www.mathworks.com/help/pdf_doc/matlab/matfile_format.pdf
This deliberately supports the dense numeric/character/cell/struct encodings
needed by the documented release. Unsupported selected encodings fail closed.
Unselected compressed top-level variables are skipped after their array name;
their value stream is not decompressed. Nested unselected fields are skipped
without decoding values. Target matrices support selected-row reads only.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import struct
import zlib

import numpy as np


MISSING = object()
DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4",
          7: "f4", 9: "f8", 12: "i8", 13: "u8", 16: "u1", 17: "u2", 18: "u4"}


def exact(stream, count):
    value = stream.read(count)
    if len(value) != count:
        raise ValueError("Truncated MAT element")
    return value


def skip(stream, count):
    if count < 0:
        raise ValueError("Negative element remainder")
    if hasattr(stream, "skip"):
        stream.skip(count)
    elif stream.seekable():
        stream.seek(count, 1)
    else:
        while count:
            chunk = exact(stream, min(count, 65536))
            count -= len(chunk)


class Limited:
    def __init__(self, stream, length):
        self.stream, self.remaining = stream, length

    def read(self, count):
        count = min(count, self.remaining)
        data = self.stream.read(count)
        self.remaining -= len(data)
        return data

    def skip(self, count):
        if count > self.remaining:
            raise ValueError("Element overrun")
        skip(self.stream, count)
        self.remaining -= count


class Inflated:
    def __init__(self, compressed):
        self.compressed = compressed
        self.decoder = zlib.decompressobj()
        self.pending = b""

    def read(self, count):
        output = bytearray()
        while len(output) < count:
            if not self.pending:
                self.pending = self.compressed.read(min(65536, self.compressed.remaining))
                if not self.pending:
                    break
            decoded = self.decoder.decompress(self.pending, count-len(output))
            output.extend(decoded)
            self.pending = self.decoder.unconsumed_tail
            if self.decoder.eof:
                break
        return bytes(output)

    def skip(self, count):
        while count:
            chunk = exact(self, min(count, 65536))
            count -= len(chunk)


@dataclass
class Element:
    kind: int
    data: object
    parent: object
    padding: int

    def finish(self):
        skip(self.data, self.data.remaining)
        skip(self.parent, self.padding)


def element(stream, endian, allow_eof=False):
    raw = stream.read(8)
    if not raw and allow_eof:
        return None
    if len(raw) != 8:
        raise ValueError("Truncated MAT tag")
    first, second = struct.unpack(endian+"II", raw)
    short_type, short_length = struct.unpack(endian+"HH", raw[:4])
    if first > 0xFFFF:
        if short_length > 4:
            raise ValueError("Invalid small-data MAT tag")
        return Element(short_type, Limited(io.BytesIO(raw[4:4+short_length]), short_length), stream, 0)
    # miCOMPRESSED is not padded by the SciPy/MATLAB Level-5 writers.
    return Element(first, Limited(stream, second), stream, 0 if first == 15 else (-second) % 8)


def raw_subelement(stream, endian):
    item = element(stream, endian)
    value = exact(item.data, item.data.remaining)
    kind = item.kind
    item.finish()
    return kind, value


@dataclass
class SelectedRows:
    rows: np.ndarray


class StructuralReader:
    def __init__(self, source):
        self.source = source
        self.decoded_paths = set()
        self.skipped_paths = set()

    def read(self, selection):
        self.source.seek(0)
        header = exact(self.source, 128)
        if not header.startswith(b"MATLAB 5.0 MAT-file"):
            raise ValueError("Only the observed Level-5 format is supported")
        self.endian = {b"IM": "<", b"MI": ">"}.get(header[126:128])
        if self.endian is None:
            raise ValueError("Unknown MAT byte order")
        values = {}
        while True:
            outer = element(self.source, self.endian, allow_eof=True)
            if outer is None:
                break
            if outer.kind == 15:
                reader = Inflated(outer.data)
                matrix = element(reader, self.endian)
                if matrix.kind != 14:
                    raise ValueError("Compressed top-level element is not a matrix")
                name, value = self.matrix(matrix.data, selection, "", top=True)
                # Do not finish an unselected inflated matrix: skip the outer
                # compressed bytes directly, so neural/cursor values stay opaque.
                if value is not MISSING:
                    matrix.finish()
            elif outer.kind == 14:
                name, value = self.matrix(outer.data, selection, "", top=True)
            else:
                raise ValueError("Unexpected top-level MAT element")
            outer.finish()
            if value is not MISSING:
                if name in values:
                    raise ValueError("Duplicate selected top-level variable")
                values[name] = value
        return values

    def matrix(self, stream, selection, path, top=False):
        _, flags_raw = raw_subelement(stream, self.endian)
        flags = struct.unpack(self.endian+"I", flags_raw[:4])[0]
        array_class = flags & 255
        _, shape_raw = raw_subelement(stream, self.endian)
        shape = np.frombuffer(shape_raw, dtype=self.endian+"i4").astype(int)
        _, name_raw = raw_subelement(stream, self.endian)
        name = name_raw.decode("utf-8")
        if top:
            path = name
            if name not in selection:
                self.skipped_paths.add(path)
                return name, MISSING
            selection = selection[name]
        if flags & 0x800:
            raise ValueError(f"Complex selected field is unsupported: {path}")
        size = int(np.prod(shape))
        if array_class == 2:
            if not isinstance(selection, dict):
                raise ValueError(f"Struct requires an explicit field whitelist: {path}")
            _, length_raw = raw_subelement(stream, self.endian)
            width = struct.unpack(self.endian+"i", length_raw)[0]
            _, names_raw = raw_subelement(stream, self.endian)
            if width <= 0 or len(names_raw) % width:
                raise ValueError("Invalid struct field names")
            fields = [names_raw[i:i+width].split(b"\0", 1)[0].decode("utf-8")
                      for i in range(0, len(names_raw), width)]
            objects = []
            for index in range(size):
                obj = {}
                for field in fields:
                    child = element(stream, self.endian)
                    child_path = path+"."+field
                    if field not in selection:
                        self.skipped_paths.add(child_path)
                    else:
                        if child.kind != 14:
                            raise ValueError("Selected struct value is not a matrix")
                        _, obj[field] = self.matrix(child.data, selection[field], child_path)
                    child.finish()
                objects.append(obj)
            return name, objects[0] if size == 1 else objects
        if array_class == 1:
            if selection is not None:
                raise ValueError("Cell selection must be explicitly permitted")
            cells = []
            for index in range(size):
                child = element(stream, self.endian)
                if child.kind != 14:
                    raise ValueError("Unexpected selected cell encoding")
                _, value = self.matrix(child.data, None, path+"[]")
                child.finish()
                cells.append(value)
            self.decoded_paths.add(path)
            return name, cells
        item = element(stream, self.endian)
        if item.kind not in DTYPES:
            raise ValueError(f"Unsupported selected primitive encoding at {path}")
        dtype = np.dtype(self.endian+DTYPES[item.kind])
        if isinstance(selection, SelectedRows):
            if list(shape[1:]) != [2] or len(shape) != 2:
                raise ValueError("Target coordinates must have the documented n_time_bins x 2 shape")
            indices = np.asarray(selection.rows, dtype=np.int64)
            if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= shape[0]):
                raise ValueError("Trial onset is outside target-coordinate support")
            unique, inverse = np.unique(indices, return_inverse=True)
            selected = np.empty((len(unique), 2), dtype=float)
            previous = 0
            for column in range(2):
                for index, row in enumerate(unique):
                    byte_offset = (column*int(shape[0])+int(row))*dtype.itemsize
                    skip(item.data, byte_offset-previous)
                    selected[index, column] = np.frombuffer(exact(item.data, dtype.itemsize), dtype=dtype)[0]
                    previous = byte_offset+dtype.itemsize
            value = selected[inverse]
        else:
            if selection is not None:
                raise ValueError("Numeric field has an invalid selection")
            raw = exact(item.data, item.data.remaining)
            if array_class == 4:
                encoding = {16: "utf-8", 17: "utf-16-le" if self.endian == "<" else "utf-16-be",
                            18: "utf-32-le" if self.endian == "<" else "utf-32-be",
                            4: "utf-16-le" if self.endian == "<" else "utf-16-be",
                            1: "ascii", 2: "ascii"}.get(item.kind)
                if encoding is None:
                    raise ValueError("Unrecognized selected character encoding")
                value = raw.decode(encoding).rstrip("\0")
            elif array_class in range(6, 16):
                if len(raw) != size*dtype.itemsize:
                    raise ValueError("Numeric dimensions differ from element length")
                value = np.frombuffer(raw, dtype=dtype).reshape(tuple(shape), order="F").copy()
            else:
                raise ValueError(f"Unsupported selected array class {array_class} at {path}")
        item.finish()
        self.decoded_paths.add(path)
        return name, value


STRUCTURAL_FIELDS = {
    "participant_id": None, "post_implant_day": None, "bin_size": None,
    "electrodes": {"electrode_id": None, "array": None, "x": None, "y": None},
    "blocks": {"block_id": None, "sample_start_index": None, "sample_end_index": None},
    "trials": {"trial_start_index": None, "trial_end_index": None, "block_id": None}}


def field_vector(value, field):
    """Accept both a scalar struct of vectors and a vector of scalar structs."""
    records = value if isinstance(value, list) else [value]
    return np.concatenate([np.asarray(record[field]).reshape(-1, order="F")
                           for record in records])


def read_structural(source):
    reader = StructuralReader(source)
    values = reader.read(STRUCTURAL_FIELDS)
    missing = set(STRUCTURAL_FIELDS)-set(values)
    if missing:
        raise ValueError(f"Missing documented structural variables: {sorted(missing)}")
    starts = field_vector(values["trials"], "trial_start_index")
    if not np.all(np.isfinite(starts)) or np.any(starts != np.floor(starts)):
        raise ValueError("Trial starts must be finite integer source indices")
    values["trial_targets"] = reader.read({"target_position": SelectedRows(starts.astype(np.int64))})["target_position"]
    forbidden = {"neural", "cursor_position", "electrodes.impedance"}
    if any(path == bad or path.startswith(bad+".") for path in reader.decoded_paths for bad in forbidden):
        raise AssertionError("Forbidden decoded field in structural reader")
    return values, {"decoded_paths": sorted(reader.decoded_paths),
                    "skipped_paths": sorted(reader.skipped_paths),
                    "target_rows_decoded": len(starts),
                    "neural_cursor_impedance_values_decoded": False}
