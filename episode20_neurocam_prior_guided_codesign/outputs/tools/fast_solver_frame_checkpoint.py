"""Frame-boundary restart data for the continuous EP20 solver.

This module is not connected to the running prefix driver. A checkpoint is
after a complete frame's output and before ANY next-frame boundary charge.
The state is stored in original physical coordinates, including all contact,
source, buffer, filter and hold voltages. Restore uses the existing solver's
coordinate transforms; its numerical seam must pass the split-run comparison.

No RNG cursor, clipping latch or unfinished aperture is carried here: this
path evaluates the saved phasors at absolute time, finds clipping anew on
each polynomial, and the validated planner closes all holds/apertures within
each frame. Diagnostics are retained for reporting, not used as state.
"""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from fast_io_readout import CompactRecordWriter, ReadoutError, WORK_FIELDS


BOUNDARY_PHASE = "completed_frame_before_next_boundary_charge"
BINDING_FIELDS = ("template_digest", "field_descriptor_digest", "noise_descriptor_digest")


def numerical_controls(fast):
    """The existing solver controls that a resumed job must retain."""
    base = fast.base
    return dict(degree=base.degree, wr_tolerance=base.wr_tolerance,
                wr_max_sweeps=base.wr_max_sweeps, event_tolerance=base.event_tolerance,
                residual_limit=base.residual_limit, segment_levels=fast.segment_levels,
                dwell_seconds=fast.dwell)


def plan_bindings(plan):
    """Canonical planner bindings understood by the existing compact reader."""
    return dict(template_digest=plan.template.digest,
                field_descriptor_digest=plan.field_descriptor_digest,
                noise_descriptor_digest=plan.noise_descriptor_digest)


@dataclass(frozen=True)
class FrameCheckpoint:
    next_frame: int
    time: float
    last_row: int
    physical_state: np.ndarray
    physical_energy: float
    cumulative_energy: dict
    plan_digests: dict
    controls: dict
    diagnostics: dict
    committed_output_rows: int
    boundary_phase: str = BOUNDARY_PHASE

    def validate(self, state_count):
        if (type(self.next_frame) is not int or not 1 <= self.next_frame <= 4000
                or self.committed_output_rows != self.next_frame
                or type(self.committed_output_rows) is not int
                or type(self.last_row) is not int or self.last_row < 0
                or self.boundary_phase != BOUNDARY_PHASE):
            raise ValueError("checkpoint must identify a completed frame before its next boundary")
        if not (math.isfinite(self.time) and self.time > 0 and math.isfinite(self.physical_energy)):
            raise ValueError("checkpoint time and physical energy must be finite")
        state = np.asarray(self.physical_state)
        if state.shape != (state_count,) or state.dtype != np.dtype("float64") or not np.all(np.isfinite(state)):
            raise ValueError("checkpoint requires every finite float64 physical state")
        if set(self.cumulative_energy) != set(WORK_FIELDS) or not all(
                math.isfinite(self.cumulative_energy[name]) for name in WORK_FIELDS):
            raise ValueError("checkpoint requires all ten finite cumulative energy fields")
        if set(self.plan_digests) != set(BINDING_FIELDS):
            raise ValueError("checkpoint requires the existing plan/input bindings")
        # Reject nonfinite reporting metadata rather than producing invalid JSON.
        json.dumps((self.plan_digests, self.controls, self.diagnostics), allow_nan=False)


def capture_frame(fast, plan, *, next_frame, time, physical_state, physical_energy,
                  cumulative_energy, committed_output_rows, diagnostics):
    """Capture the already-computed frame endpoint without advancing the solver."""
    if fast.row is None or fast.base.row != fast.row:
        raise ValueError("checkpoint requires the solver's active row at frame end")
    checkpoint = FrameCheckpoint(
        next_frame=int(next_frame), time=float(time), last_row=int(fast.row),
        physical_state=np.array(physical_state, copy=True),
        physical_energy=float(physical_energy), cumulative_energy=dict(cumulative_energy),
        plan_digests=plan_bindings(plan), controls=numerical_controls(fast),
        diagnostics=dict(diagnostics), committed_output_rows=int(committed_output_rows))
    checkpoint.validate(fast.base.N)
    return checkpoint


def _atomic_save(path, checkpoint):
    path = Path(path)
    metadata = dict(format_version=1, next_frame=checkpoint.next_frame,
                    last_row=checkpoint.last_row, committed_output_rows=checkpoint.committed_output_rows,
                    boundary_phase=checkpoint.boundary_phase, plan_digests=checkpoint.plan_digests,
                    controls=checkpoint.controls, diagnostics=checkpoint.diagnostics)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, physical_state=checkpoint.physical_state,
                     time=np.array(checkpoint.time), physical_energy=np.array(checkpoint.physical_energy),
                     work_fields=np.array(WORK_FIELDS),
                     cumulative_energy=np.array([checkpoint.cumulative_energy[name] for name in WORK_FIELDS]),
                     metadata_json=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def commit_frame(path, checkpoint, writer):
    """Flush matching HDF5 output first, then atomically replace the checkpoint.

    If interrupted between these operations, the preceding checkpoint remains
    authoritative; rows beyond its committed count must be overwritten on
    resume. A checkpoint never advances ahead of its durable output rows.
    This ordering does not repair HDF5 consistency flags after an unclean
    process death; a refused reopen must be diagnosed separately.
    """
    checkpoint.validate(len(checkpoint.physical_state))
    n = checkpoint.committed_output_rows
    if writer._closed or writer._finished or writer._next != n:
        raise ValueError("writer must be open at exactly the checkpoint frame boundary")
    f = writer._file
    if (json.loads(f.attrs["plan_digests_json"]) != checkpoint.plan_digests
            or tuple(json.loads(f.attrs["work_fields_json"])) != WORK_FIELDS
            or not np.all(f["windows_written"][:n])):
        raise ValueError("checkpoint and committed output prefix differ")
    expected = np.array([checkpoint.cumulative_energy[name] for name in WORK_FIELDS])
    if not np.array_equal(f["energy/cumulative_totals"][n - 1], expected):
        raise ValueError("checkpoint cumulative energy must match the committed output row")
    f.flush()
    os.fsync(f.id.get_vfd_handle())  # CompactRecordWriter uses the default file driver.
    _atomic_save(path, checkpoint)


class FrameRecordWriter(CompactRecordWriter):
    """Original row writer with a validated restart and a bounded-stop status.

    No dataset is resized and no historic short-key record is migrated. New
    production records reserve all 4000 rows before their first chunk.
    """

    @classmethod
    def resume(cls, path, *, template, plan_digests, windows, accounting_columns,
               checkpoint, rolling_checkpoint_path):
        import h5py
        checkpoint.validate(len(checkpoint.physical_state))
        n, windows = checkpoint.committed_output_rows, int(windows)
        if not n <= windows or checkpoint.plan_digests != plan_digests:
            raise ReadoutError("checkpoint does not belong to this record target")
        writer = cls.__new__(cls)
        writer.template, writer.windows = template, windows
        writer.accounting_columns = tuple(accounting_columns)
        writer.retain_aperture_means = True
        writer._closed = writer._finished = False
        writer._file = h5py.File(path, "r+")
        try:
            f = writer._file
            if (json.loads(f.attrs["plan_digests_json"]) != plan_digests
                    or tuple(json.loads(f.attrs["work_fields_json"])) != WORK_FIELDS
                    or tuple(json.loads(f.attrs["accounting_columns_json"])) != writer.accounting_columns
                    or not bool(f.attrs["retain_aperture_means"])):
                raise ReadoutError("resume requires the original output bindings and columns")
            r, o = template.conversions, template.observations
            required = {
                "raw/aperture_mean": ((windows, r), "<f8"),
                "raw/code": ((windows, r), "<u4"),
                "raw/valid_bits": ((windows, (r + 7) // 8), "u1"),
                "raw/clipped_bits": ((windows, (r + 7) // 8), "u1"),
                "observations/value": ((windows, o), "<f8"),
                "observations/flags": ((windows, o), "u1"),
                "energy/window_totals": ((windows, len(WORK_FIELDS)), "<f8"),
                "energy/cumulative_totals": ((windows, len(WORK_FIELDS)), "<f8"),
                "windows_written": ((windows,), "?")}
            if writer.accounting_columns:
                required["accounting/window"] = ((windows, len(writer.accounting_columns)), "<f8")
            if any(name not in f or f[name].shape != shape or f[name].dtype != np.dtype(dtype)
                   for name, (shape, dtype) in required.items()):
                raise ReadoutError("resume record dimensions or numeric storage differ")
            expected = np.array([checkpoint.cumulative_energy[name] for name in WORK_FIELDS])
            if (not np.all(f["windows_written"][:n])
                    or not np.array_equal(f["energy/cumulative_totals"][n - 1], expected)):
                raise ReadoutError("checkpoint and durable output prefix differ")
            # The driver has already validated/restored the full physical state,
            # exact time, active row and controls. All output validation above
            # must also succeed before replacing the canonical checkpoint.
            # Publish an explicitly chosen older boundary BEFORE invalidating
            # newer rows; an interruption must never leave that checkpoint ahead
            # of the surviving output prefix.
            _atomic_save(rolling_checkpoint_path, checkpoint)
            # A row written after the authoritative checkpoint is replayed.
            # Validation is complete before any existing row is invalidated.
            f["windows_written"][n:] = False
            f.attrs["record_complete"] = False
            f.attrs["simulation_status"] = "incomplete"
            f.attrs["windows_completed"] = n
            f.attrs["record_energy_totals_json"] = json.dumps(checkpoint.cumulative_energy, sort_keys=True)
            f.attrs["terminal_metadata_json"] = json.dumps(dict(resumed_at_frame=n,
                physical_time=checkpoint.time, boundary_phase=checkpoint.boundary_phase))
            writer._next = n
            f.flush()
            return writer
        except BaseException:
            writer.close()
            raise

    def finish(self, *, status, metadata):
        if status != "checkpointed":
            return super().finish(status=status, metadata=metadata)
        if self._closed or self._finished or not 0 < self._next < self.windows:
            raise ReadoutError("checkpointed status requires an incomplete open record")
        f = self._file
        f.attrs["terminal_metadata_json"] = json.dumps(metadata, sort_keys=True, default=repr)
        f.attrs["windows_completed"] = self._next
        f.attrs["record_energy_totals_json"] = json.dumps(dict(zip(
            WORK_FIELDS, f["energy/cumulative_totals"][self._next - 1].tolist())))
        f.attrs["simulation_status"] = "checkpointed"
        f.attrs["record_complete"] = False
        f.flush()
        self._finished = True


def load_frame(path, *, state_count):
    """Read a committed checkpoint; legacy terminal-only state files are not restart data."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.pop("format_version") != 1 or tuple(archive["work_fields"].tolist()) != WORK_FIELDS:
            raise ValueError("unsupported frame checkpoint or energy-field order")
        checkpoint = FrameCheckpoint(
            physical_state=archive["physical_state"].copy(), time=float(archive["time"].item()),
            physical_energy=float(archive["physical_energy"].item()),
            cumulative_energy=dict(zip(WORK_FIELDS, archive["cumulative_energy"].tolist())), **metadata)
    checkpoint.validate(state_count)
    return checkpoint


def restore_frame(checkpoint, fast, plan):
    """Restore at the saved absolute time; apply no source or DAQ charge here.

    The caller resumes plan.window(checkpoint.next_frame), whose existing
    first-action logic applies the new frame's charges once. Reporting
    diagnostics remain available on the checkpoint for driver aggregation.
    """
    checkpoint.validate(fast.base.N)
    if checkpoint.plan_digests != plan_bindings(plan) or checkpoint.controls != numerical_controls(fast):
        raise ValueError("resume must retain the original inputs, schedule and numerical controls")
    previous = plan.window(checkpoint.next_frame - 1)
    if (float(previous.end).hex() != float(checkpoint.time).hex()
            or checkpoint.last_row != int(plan.template.action_row[-1])):
        raise ValueError("checkpoint time and row must be the exact preceding frame endpoint")
    fast.set_full_state(checkpoint.physical_state, checkpoint.time, checkpoint.last_row)
    return checkpoint.next_frame
