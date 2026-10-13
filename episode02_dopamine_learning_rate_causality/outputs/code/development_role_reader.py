"""Dormant reader for the existing EP02 development-only NPY role pack.

The caller must first provision the externally enforced development access
boundary and satisfy the episode's source-semantics/margin requirements.
``admit_development_root(root)`` must raise unless that exact root is admitted;
it runs before any filesystem access here. A callback, manifest, same-user
mode bits, or this reader cannot establish isolation or create authorization.

Only the nine fixed control identities and four original fields are exposed.
No codes, eligibility, lick predicate, event zero, sample times or missingness
are inferred. In particular a 701-column lick array is not a time vector.
No mixed MAT, audit data, provider summaries or network fallback is supported.

Use both reader and mouse as context managers. One mouse may be open at once;
the explicit byte cap covers its four complete NPY files. Arrays retain their
dtype, axes and values in read-only maps. Borrowed arrays must not be used
after mouse/reader closure. There is no empirical command-line entrypoint.
"""
from dataclasses import dataclass
import json
import math
from numbers import Integral
import os
from pathlib import Path
import stat
from types import MappingProxyType

import numpy as np


DEVELOPMENT_RECORDS = (1, 2, 4, 6, 9, 11, 15, 16, 19)
FIELDS = ("trialID", "seshID", "lickState", "lick")
MANIFEST_LIMIT_BYTES = 256 * 1024
# Existing source identity from COHORT_MAP.json; no new digest/receipt chain.
SOURCE_SHA256 = "1039b9555ef50b80e01cc923b372a7960daade774fdb6a236c34cafcedf75866"


def _integer(value, label, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _check(callback):
    if not callable(callback):
        raise TypeError("caller-owned resource/cancellation callback required")
    callback()


def _open_regular(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("role-pack field must be a regular file")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


@dataclass(frozen=True)
class _Field:
    relative_path: str
    dtype: np.dtype
    shape: tuple
    npy_bytes: int


def _manifest(root):
    with _open_regular(root / "MANIFEST.json") as handle:
        if os.fstat(handle.fileno()).st_size > MANIFEST_LIMIT_BYTES:
            raise ValueError("development manifest exceeds the fixed metadata bound")
        raw = handle.read(MANIFEST_LIMIT_BYTES + 1)
    if len(raw) > MANIFEST_LIMIT_BYTES:
        raise ValueError("development manifest grew beyond the metadata bound")
    manifest = json.loads(raw)
    if (manifest.get("schema_version") != "ep02.dudman.role_materialization.v1"
            or manifest.get("pack_format") != "deterministic_npy_tree_v1"
            or manifest.get("role") != "development_full"
            or manifest.get("visibility") != "candidate_readable"
            or manifest.get("source_payload_sha256") != SOURCE_SHA256
            or manifest.get("round_trip_verified") is not True
            or manifest.get("outcome_summaries_emitted") is not False):
        raise PermissionError("only the existing source-bound development role pack is admitted")
    if _integer(manifest.get("record_count"), "record count") != len(DEVELOPMENT_RECORDS):
        raise ValueError("development pack requires exactly nine control records")
    entries = manifest.get("record_entries")
    if not isinstance(entries, list) or len(entries) != len(DEVELOPMENT_RECORDS):
        raise ValueError("complete development record metadata required")
    records = {}
    # Validate the entire role/identity roster before opening any NPY file.
    for expected, entry in zip(DEVELOPMENT_RECORDS, entries):
        index = _integer(entry.get("source_index_1_based"), "source index", 1)
        name = f"source_record_{expected:02d}"
        if (index != expected or entry.get("opaque_animal_id") != name
                or entry.get("cohort_id") != "development_control"):
            raise PermissionError("ordered development identities differ from the fixed nine controls")
        fields = entry.get("fields")
        if not isinstance(fields, dict) or any(field not in fields for field in FIELDS):
            raise ValueError("all four permitted raw-field definitions are required")
        selected = {}
        for field in FIELDS:
            metadata = fields[field]
            relative = f"{name}/{field}.npy"
            if metadata.get("relative_path") != relative:
                raise PermissionError("field path must retain the exact development record layout")
            shape = metadata.get("shape")
            if not isinstance(shape, list):
                raise ValueError("original array shape must be declared")
            shape = tuple(_integer(size, "axis length") for size in shape)
            dtype = np.dtype(metadata["dtype"])
            size = _integer(metadata.get("npy_bytes"), "NPY byte count", 1)
            if dtype.hasobject or dtype.itemsize <= 0 or math.prod(shape) * dtype.itemsize > size:
                raise ValueError("original non-object dtype/shape must fit the declared NPY bytes")
            selected[field] = _Field(relative, dtype, shape, size)
        records[index] = MappingProxyType(selected)
    return MappingProxyType(records)


class DevelopmentMouse:
    """One borrowed raw control record; no endpoint or source-semantic claims."""

    def __init__(self, reader, source_index, arrays):
        self.source_index_1_based = source_index
        self._reader = reader
        self._arrays = MappingProxyType(arrays)
        self._closed = False

    @property
    def arrays(self):
        if self._closed:
            raise RuntimeError("development mouse is closed")
        return self._arrays

    def close(self):
        if not self._closed:
            for array in self._arrays.values():
                mapping = getattr(array, "_mmap", None)
                if mapping is not None:
                    mapping.close()
            self._closed = True
            if self._reader._active is self:
                self._reader._active = None

    def __enter__(self):
        if self._closed:
            raise RuntimeError("development mouse is closed")
        return self

    def __exit__(self, *_):
        self.close()


class DevelopmentReader:
    """A reader, not an admission grant or role-isolation mechanism."""

    def __init__(self, root, *, admit_development_root, maximum_mapped_bytes, check):
        self._closed, self._active = True, None
        root = Path(root)
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError("explicit absolute development root without parent traversal required")
        self._maximum = _integer(maximum_mapped_bytes, "mapped-byte bound", 1)
        if not callable(admit_development_root):
            raise TypeError("external development-root admission callback required")
        _check(check)
        # External runtime decision only; accepting a Boolean is not admission.
        if admit_development_root(root) is not None:
            raise TypeError("external admission must return None on success or raise on refusal")
        self._check = check
        _check(check)
        if root.resolve(strict=True) != root or not root.is_dir():
            raise PermissionError("admitted development root must be a canonical nonsymlink directory")
        self.root = root
        self._records = _manifest(root)
        _check(check)
        self._closed = False

    def open_mouse(self, source_index_1_based):
        if self._closed:
            raise RuntimeError("development reader is closed")
        _check(self._check)
        index = _integer(source_index_1_based, "source index", 1)
        if index not in self._records:
            raise PermissionError("only the nine fixed development controls may be opened")
        if self._active is not None:
            raise RuntimeError("close the active development mouse before opening another")
        fields = self._records[index]
        if sum(field.npy_bytes for field in fields.values()) > self._maximum:
            raise ValueError("one complete mouse exceeds the caller's mapped-byte bound")
        folder = self.root / f"source_record_{index:02d}"
        if folder.resolve(strict=True) != folder or not folder.is_dir():
            raise PermissionError("development record directory must retain its admitted path")
        arrays = {}
        try:
            for field in FIELDS:
                _check(self._check)
                metadata = fields[field]
                with _open_regular(self.root / metadata.relative_path) as handle:
                    if os.fstat(handle.fileno()).st_size != metadata.npy_bytes:
                        raise ValueError("NPY file size differs from the development manifest")
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                    else:
                        raise ValueError("expected the existing materializer's NPY v1/v2 format")
                    count = math.prod(shape)
                    if (dtype.hasobject or dtype != metadata.dtype or shape != metadata.shape
                            or handle.tell() + count * dtype.itemsize != metadata.npy_bytes):
                        raise ValueError("NPY header differs from its original dtype, axes or byte count")
                    if count:
                        value = np.memmap(handle, mode="r", dtype=dtype, shape=shape,
                            offset=handle.tell(), order="F" if fortran else "C")
                    else:
                        value = np.empty(shape, dtype=dtype, order="F" if fortran else "C")
                    value.flags.writeable = False
                    arrays[field] = value
                _check(self._check)
            result = DevelopmentMouse(self, index, arrays)
            self._active = result
            return result
        except BaseException:
            for value in arrays.values():
                mapping = getattr(value, "_mmap", None)
                if mapping is not None:
                    mapping.close()
            raise

    def close(self):
        if self._active is not None:
            self._active.close()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("development reader is closed")
        return self

    def __exit__(self, *_):
        self.close()


def open_development_pack(root, *, admit_development_root, maximum_mapped_bytes, check):
    return DevelopmentReader(root, admit_development_root=admit_development_root,
        maximum_mapped_bytes=maximum_mapped_bytes, check=check)
