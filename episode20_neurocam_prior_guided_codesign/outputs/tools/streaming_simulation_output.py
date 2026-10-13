"""Bounded append sink for one conditional EP20 simulation.

Completed conversions, published observations and interval evidence are written
as compound records, not a complete Python history. Observation support uses a
separate ragged table. Fixed UTF-8 identifier widths are caller limits and never
truncate. The payload cap excludes HDF5 metadata/allocated-chunk overhead; the
job's storage reservation must cover those too. This is not a restart facility.
"""
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path

import numpy as np

from acquisition_readout import RawValue, PublishedValue
from array_acquisition import IntervalEvidence


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be an explicit positive integer")
    return value


def _finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("record numeric value must be finite")
    return result


def _json(value):
    def default(item):
        if is_dataclass(item):
            return asdict(item)
        raise TypeError(f"unsupported metadata item {type(item).__name__}")
    return json.dumps(value, default=default, sort_keys=True, allow_nan=False).encode("utf-8")


def _terminal_write(method):
    def write(self, *args, **kwargs):
        self._require_open()
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self._failed = True
            raise
    return write


class StreamingSimulationOutput:
    """Exclusive single-writer result; callback failure terminates the attempt.

    The caller owns causal access, physical execution and terminal status. Only
    completed raw conversions and actually published observations belong here.
    A file is publishable only after ``finish`` and ``record_complete=True``;
    ``simulation_status`` independently distinguishes complete and failed runs.
    Closing without finish preserves an explicitly incomplete container.
    """
    def __init__(self, path, *, settings_reference, max_payload_bytes,
                 max_records, buffer_records, identifier_bytes,
                 max_dependencies_per_observation):
        import h5py
        for name, value in (("max_payload_bytes", max_payload_bytes),
                            ("max_records", max_records), ("buffer_records", buffer_records),
                            ("identifier_bytes", identifier_bytes),
                            ("max_dependencies_per_observation", max_dependencies_per_observation)):
            _positive(value, name)
        if max_dependencies_per_observation + 1 > buffer_records:
            raise ValueError("one observation and all its support must fit the explicit buffer")
        if not isinstance(settings_reference, str) or not settings_reference.strip():
            raise ValueError("an explicit retained settings reference is required")
        self.path = Path(path)
        self.max_payload_bytes, self.max_records = max_payload_bytes, max_records
        self.buffer_records, self.identifier_bytes = buffer_records, identifier_bytes
        self.max_dependencies = max_dependencies_per_observation
        self.payload_bytes = self.record_count = self.buffered_records = 0
        self.maximum_buffered_records = 0
        self._closed = self._finished = self._failed = self._started = False
        width = f"S{identifier_bytes}"
        self.dtypes = {
            "raw_values": np.dtype([("conversion_id", width), ("aperture_mean_volts", "<f8"),
                ("code_index", "<i8"), ("decoded_volts", "<f8"), ("valid", "?"), ("clipped", "?")]),
            "observations": np.dtype([("observation_id", width), ("site_id", width), ("role", width),
                ("value_volts", "<f8"), ("has_value", "?"), ("valid", "?"), ("clipped", "?"),
                ("assigned_timestamp", "<f8"), ("available_at", "<f8"),
                ("support_start", "<u8"), ("support_count", "<u8")]),
            "aperture_support": np.dtype([("conversion_id", width), ("start", "<f8"), ("end", "<f8")]),
            "interval_evidence": np.dtype([("physical_event_index", "<u8"), ("start", "<f8"),
                ("end", "<f8"), ("event_boundary_charge_applied", "?")]),
        }
        self._buffers = {name: [] for name in self.dtypes}
        self.counts = dict.fromkeys(self.dtypes, 0)
        reference = _json({"settings_reference": settings_reference,
                           "scope": "conditional_simulation_internal_record",
                           "payload_cap_excludes_hdf5_overhead": True})
        if len(reference) > max_payload_bytes:
            raise ValueError("settings reference exceeds payload cap")
        self._file = h5py.File(self.path, "x")
        try:
            self._file.attrs["record_complete"] = False
            self._file.attrs["simulation_status"] = "incomplete"
            for name, dtype in self.dtypes.items():
                self._file.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype,
                                          chunks=(min(buffer_records, 4096),))
            self._bytes("settings_json", reference)
        except BaseException:
            self._failed = True
            self._file.close()
            self._closed = True
            raise

    def _require_open(self):
        if self._closed or self._finished or self._failed:
            raise RuntimeError("output attempt is closed, finished or failed; no replay")

    def _id(self, value):
        if not isinstance(value, str) or not value or "\0" in value:
            raise ValueError("record identifiers must be nonempty UTF-8 strings without NUL")
        encoded = value.encode("utf-8")
        if len(encoded) > self.identifier_bytes:
            raise ValueError("record identifier exceeds explicit byte width; no truncation")
        return encoded

    def _reserve(self, payload_bytes, records=0):
        if (self.payload_bytes + payload_bytes > self.max_payload_bytes
                or self.record_count + records > self.max_records):
            raise ValueError("streamed result exceeds explicit payload or record cap")
        self.payload_bytes += payload_bytes
        self.record_count += records

    def _bytes(self, name, data):
        self._reserve(len(data))
        self._file.create_dataset(name, data=np.frombuffer(data, dtype=np.uint8))

    def begin(self, *, state_ids, initial_state_volts, initial_field_volts, metadata):
        self._require_open()
        if self._started:
            raise RuntimeError("initial state is already recorded")
        try:
            ids = tuple(self._id(value) for value in state_ids)
            state = np.asarray(initial_state_volts, dtype=np.float64)
            field = np.asarray(initial_field_volts, dtype=np.float64)
            if (state.ndim != 1 or len(state) != len(ids) or field.shape not in ((4096,), (64, 64))
                    or not np.isfinite(state).all() or not np.isfinite(field).all()):
                raise ValueError("complete finite physical initial state is required")
            self._state_shape = state.shape
            self._field_shape = field.shape
            for name, array in (("initial_state_volts", state), ("initial_field_volts", field),
                                ("state_ids", np.asarray(ids, dtype=f"S{self.identifier_bytes}"))):
                self._reserve(array.nbytes)
                self._file.create_dataset(name, data=array)
            self._bytes("initial_metadata_json", _json(metadata))
            self._started = True
            self._file.flush()
        except BaseException:
            self._failed = True
            raise

    def _append(self, rows):
        self._require_open()
        if not self._started:
            raise RuntimeError("record the initial physical state before streaming")
        count = sum(len(values) for values in rows.values())
        size = sum(self.dtypes[name].itemsize * len(values) for name, values in rows.items())
        try:
            if count > self.buffer_records:
                raise ValueError("one emitted item exceeds the live buffer cap")
            if self.buffered_records + count > self.buffer_records:
                self.flush()
            self._reserve(size, count)
            for name, values in rows.items():
                self._buffers[name].extend(values)
                self.counts[name] += len(values)
            self.buffered_records += count
            self.maximum_buffered_records = max(self.maximum_buffered_records, self.buffered_records)
            if self.buffered_records == self.buffer_records:
                self.flush()
        except BaseException:
            self._failed = True
            raise

    @_terminal_write
    def emit_raw(self, value):
        if not isinstance(value, RawValue):
            raise TypeError("a completed RawValue is required")
        if type(value.code_index) is not int or not 0 <= value.code_index < 2**63:
            raise ValueError("raw code index cannot be represented exactly")
        self._append({"raw_values": [(self._id(value.conversion_id), _finite(value.aperture_mean_volts),
            value.code_index, _finite(value.decoded_volts), value.valid, value.clipped)]})

    @_terminal_write
    def emit_observation(self, value):
        if not isinstance(value, PublishedValue):
            raise TypeError("an actually published observation is required")
        if len(value.contributing_apertures) > self.max_dependencies:
            raise ValueError("observation dependency support exceeds its explicit cap")
        support = []
        for raw_id, start, end in value.contributing_apertures:
            start, end = _finite(start), _finite(end)
            if start >= end:
                raise ValueError("observation aperture support must have positive duration")
            support.append((self._id(raw_id), start, end))
        has_value = value.value_volts is not None
        if has_value != bool(value.valid):
            raise ValueError("validity and missing published value disagree")
        numeric = _finite(value.value_volts) if has_value else math.nan
        row = (self._id(value.observation_id), self._id(value.site_id), self._id(value.role),
               numeric, has_value, value.valid, value.clipped, _finite(value.assigned_timestamp),
               _finite(value.available_at), self.counts["aperture_support"], len(support))
        self._append({"aperture_support": support, "observations": [row]})

    @_terminal_write
    def emit_interval(self, value):
        if not isinstance(value, IntervalEvidence):
            raise TypeError("returned physical interval evidence is required")
        index = value.physical_event_index
        if type(index) is not int or not 0 <= index < 2**64:
            raise ValueError("physical interval index cannot be represented exactly")
        self._append({"interval_evidence": [(index, _finite(value.start), _finite(value.end),
                                            value.event_boundary_charge_applied)]})

    def flush(self):
        self._require_open()
        try:
            for name, values in self._buffers.items():
                if not values:
                    continue
                dataset = self._file[name]
                start = len(dataset)
                dataset.resize((start + len(values),))
                dataset[start:] = np.asarray(values, dtype=self.dtypes[name])
                values.clear()
            self.buffered_records = 0
            self._file.attrs["serialized_payload_bytes"] = self.payload_bytes
            self._file.flush()
        except BaseException:
            self._failed = True
            raise

    def finish(self, *, status, final_state_volts, final_field_volts, metadata):
        self._require_open()
        if not self._started or status not in ("complete", "failed"):
            raise ValueError("started attempt and explicit terminal simulation status required")
        try:
            self.flush()
            state = np.asarray(final_state_volts, dtype=np.float64)
            field = np.asarray(final_field_volts, dtype=np.float64)
            if (state.shape != self._state_shape or field.shape != self._field_shape
                    or not np.isfinite(state).all() or not np.isfinite(field).all()):
                raise ValueError("complete finite final physical state is required")
            for name, array in (("final_state_volts", state), ("final_field_volts", field)):
                self._reserve(array.nbytes)
                self._file.create_dataset(name, data=array)
            terminal = dict(metadata, simulation_status=status, record_counts=dict(self.counts),
                            maximum_buffered_records=self.maximum_buffered_records,
                            full_history_materialized=False)
            self._bytes("terminal_metadata_json", _json(terminal))
            self._file.attrs["serialized_payload_bytes"] = self.payload_bytes
            self._file.attrs["simulation_status"] = status
            self._file.flush()
            self._file.attrs["record_complete"] = True
            self._file.flush()
            self._finished = True
            return dict(path=str(self.path), status=status, record_counts=dict(self.counts),
                        serialized_payload_bytes=self.payload_bytes,
                        maximum_buffered_records=self.maximum_buffered_records)
        except BaseException:
            self._failed = True
            raise

    def close(self):
        if not self._closed:
            # Closing an unfinished/failed sink cannot promote its partial data.
            self._file.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
