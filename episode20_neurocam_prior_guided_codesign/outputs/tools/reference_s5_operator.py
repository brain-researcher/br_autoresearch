"""Adopted prospective S5 operator: complete transient record, exact DAQ clock.

This is a new dry-air experiment. Its record is not a ten-column energy record
from the PBS production solver. Physical states and all raw/published voltages
are retained; no fictitious PBS work quantities are supplied.
"""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from conditional_acquisition_recipe import build_conditional_frontend, build_conditional_schedule
from fast_io_exact import shift
from fast_io_planner import template_from_schedule
from fast_io_readout import VectorizedReadout
from reference_s5_family import AirFamily, OMEGA, PROBE_SITE, specification

FRAMES = 4000
FRAME_SECONDS = .004


def operator_specification():
    return dict(record_seconds=16., frames=FRAMES, frame_seconds=FRAME_SECONDS,
        probe_site=list(PROBE_SITE), scan="full64_rows_62.5us_each",
        daq="conditional24_ADC_actual_apertures_third_to_fifth_decoded_words",
        amplitude="OLS_sine_cosine_intercept_at_actual_assigned_pixel_timestamps",
        ratio="amplitude_of_site_divided_by_amplitude_of_driven_output_site",
        nearest_source_sites=[[30, 31], [32, 31]], nearest_gate_sites=[[31, 30], [31, 32]],
        distant_mask="Manhattan_distance_at_least_two", aggregation="maximum_amplitude_ratio",
        initial_history="zero_all_states; first_action_and_every_later_seam_charge_retained",
        noise="zero_stochastic; deterministic_charge_marker_quantizer_retained",
        exact_zero="zero_numerator_is_negative_infinity_dB; zero_denominator_is_undefined",
        uncertainty="explored_parameter_range_not_device_confidence_interval")


class S5Operator:
    def __init__(self, family):
        if (family.rows, family.cols, family.adc_count, family.probe_site) != (64, 64, 24, PROBE_SITE):
            raise ValueError("production S5 operator requires full64x64,24ADC and the selected probe")
        self.family = family
        frontend = build_conditional_frontend()
        schedule = build_conditional_schedule(frontend, state_ids=family.state_ids, check=family.check)
        self.template = t = template_from_schedule(schedule, frontend)
        self.readout = VectorizedReadout(t)
        self.raw_groups = []
        self.charge_deltas = []
        action_index = np.searchsorted(t.action_end, t.conv_aperture_start, side="right")
        for action in range(t.actions):
            rows = np.flatnonzero(action_index == action)
            if not len(rows) or not (np.all(t.conv_aperture_start[rows] == t.conv_aperture_start[rows[0]])
                    and np.all(t.conv_aperture_end[rows] == t.conv_aperture_end[rows[0]])):
                raise ValueError("selected recipe must have one common aperture per action")
            if not (t.action_start[action] <= t.conv_aperture_start[rows[0]]
                    < t.conv_aperture_end[rows[0]] <= t.action_end[action]):
                raise ValueError("aperture crosses a circuit action")
            self.raw_groups.append(rows)
            q = np.zeros(family.full_size)
            lo, hi = t.action_charge_ptr[action:action + 2]
            np.add.at(q, t.action_charge_state[lo:hi], t.action_charge_value[lo:hi])
            self.charge_deltas.append(family.charge_delta(q))
        role = t.observation_roles.index("pixel")
        self.pixel_rows = np.flatnonzero(t.obs_role == role)
        sites = [t.site_labels[t.obs_site[o]] for o in self.pixel_rows]
        if sites != [f"site{p}" for p in range(4096)]:
            raise ValueError("physical pixel order differs from row-major64x64")

    def window(self, frame):
        if not 0 <= frame < FRAMES:
            raise ValueError("frame outside the complete16-second record")
        t = self.template
        return SimpleNamespace(index=frame,
            conversion_id=frame * t.conversions + np.arange(t.conversions),
            returned_id=frame * t.returned + np.arange(t.returned),
            observation_id=frame * t.observations + np.arange(t.observations),
            conv_aperture_start=shift(frame, t.conv_aperture_start),
            conv_aperture_end=shift(frame, t.conv_aperture_end),
            obs_assigned=shift(frame, t.obs_assigned), obs_available=shift(frame, t.obs_available))

    def _step(self, state, row, slot, marker, start, end):
        self.family.check()
        if end == start:
            return state, np.zeros(24)
        if self.family.structure == "R1":
            endpoint, integral = self.family.dense_step(row, slot, marker, end - start)
            augmented = np.concatenate((state, [math.sin(OMEGA * start), math.cos(OMEGA * start), 1.]))
            return (endpoint @ augmented)[:self.family.size], integral @ augmented
        return self.family.sparse_step(state, row, slot, marker, start, end)

    def frame(self, state, frame):
        """Carry every state; apply each start charge exactly once before dynamics."""
        t, w = self.template, self.window(frame)
        starts, ends = shift(frame, t.action_start), shift(frame, t.action_end)
        state = np.asarray(state, dtype=float).copy()
        if state.shape != (self.family.size,) or not np.all(np.isfinite(state)):
            raise ValueError("invalid complete circuit state")
        mean = np.empty(t.conversions)
        boundary_energy = 0.
        for a, raw in enumerate(self.raw_groups):
            start, end = float(starts[a]), float(ends[a])
            row, slot = int(t.action_row[a]), a % 4
            marker = float(t.action_external_volts[a, 0])
            before = self.family.physical_energy(state, start)
            state += self.charge_deltas[a]
            boundary_energy += self.family.physical_energy(state, start) - before
            p, q = float(w.conv_aperture_start[raw[0]]), float(w.conv_aperture_end[raw[0]])
            state, _ = self._step(state, row, slot, marker, start, p)
            state, integral = self._step(state, row, slot, marker, p, q)
            mean[raw] = integral[self.readout.conv_adc[raw]] / (q - p)
            state, _ = self._step(state, row, slot, marker, q, end)
        if not np.all(np.isfinite(state)):
            raise ArithmeticError("nonfinite frame endpoint")
        code, decoded, clipped = self.readout.quantize(mean)
        result = self.readout.assemble(w, mean, code, decoded, t.conv_valid.copy(), clipped)
        if not np.all(result.valid[self.pixel_rows]):
            raise ArithmeticError("invalid pixel in prescribed complete scan")
        return state, result, boundary_energy


class SineSummary:
    def __init__(self, count=4096):
        self.xtx = np.zeros((count, 3, 3))
        self.xty = np.zeros((count, 3))
        self.count = 0

    def add(self, times, values):
        times, values = np.asarray(times), np.asarray(values)
        if times.shape != (len(self.xty),) or values.shape != times.shape or not np.all(np.isfinite(values)):
            raise ValueError("one finite observation and actual timestamp per physical pad required")
        x = np.column_stack((np.sin(OMEGA * times), np.cos(OMEGA * times), np.ones(len(times))))
        self.xtx += x[:, :, None] * x[:, None, :]
        self.xty += x * values[:, None]
        self.count += 1

    def amplitudes(self, sites):
        if self.count != FRAMES:
            raise ValueError("no calibration or prediction estimate before all4000 frames")
        sites = np.asarray(sites, dtype=int)
        coefficients = np.linalg.solve(self.xtx[sites], self.xty[sites])
        return np.hypot(coefficients[:, 0], coefficients[:, 1])


def ratio_db(amplitude, denominator):
    """Return explicit JSON-safe status; never replace zero with a fitted floor."""
    if not math.isfinite(amplitude) or amplitude < 0 or not math.isfinite(denominator) or denominator < 0:
        raise ValueError("amplitudes must be finite and nonnegative")
    if denominator == 0:
        return dict(db=None, status="undefined_zero_denominator")
    if amplitude == 0:
        return dict(db=None, status="negative_infinity_zero_numerator")
    return dict(db=20 * math.log10(amplitude / denominator), status="finite")


def nearest_summary(summary):
    sites = [31 * 64 + 31, 30 * 64 + 31, 32 * 64 + 31, 31 * 64 + 30, 31 * 64 + 32]
    amplitudes = summary.amplitudes(sites)
    return dict(driven_amplitude_volts=float(amplitudes[0]),
        nearest_source=ratio_db(float(max(amplitudes[1:3])), float(amplitudes[0])),
        nearest_gate=ratio_db(float(max(amplitudes[3:5])), float(amplitudes[0])))


def atomic_npz(path, **values):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as out:
        np.savez(out, **values); out.flush(); os.fsync(out.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def durable_hdf(file):
    file.flush()
    descriptor = file.id.get_vfd_handle()
    if isinstance(descriptor, tuple):
        descriptor = descriptor[0]
    os.fsync(descriptor)


class AirRecord:
    """Clean chunk restart. Only canonical latest checkpoints are accepted.

    Atomic NPZ publication follows durable HDF output. A process killed inside
    HDF may leave a consistency flag; an HDF reopen refusal needs diagnosis.
    """
    def __init__(self, directory, operator, manifest, *, resume=False):
        self.directory, self.operator = Path(directory), operator
        self.checkpoint = self.directory / "frame_checkpoint.npz"
        family, t = operator.family, operator.template
        self.binding = json.dumps(dict(manifest=manifest, structure=family.structure,
            parameters=asdict(family.parameters), family=specification(), operator=operator_specification(),
            template_digest=t.digest), sort_keys=True, allow_nan=False)
        self.summary = SineSummary()
        self.next_frame, self.state, self.boundary_energy, self.clipped = 0, np.zeros(family.size), 0., 0
        if resume:
            with np.load(self.checkpoint, allow_pickle=False) as saved:
                if str(saved["binding"]) != self.binding:
                    raise ValueError("checkpoint differs from frozen family/operator/parameters/plan")
                n = int(saved["next_frame"])
                state, xtx, xty = saved["state"], saved["xtx"], saved["xty"]
                if (not 0 <= n <= FRAMES or float(saved["time"]) != n * FRAME_SECONDS
                        or int(saved["last_row"]) != (63 if n else -1)
                        or str(saved["charge_phase"]) != "before_next_frame_start_charge"
                        or state.shape != self.state.shape or xtx.shape != self.summary.xtx.shape
                        or xty.shape != self.summary.xty.shape
                        or not all(np.all(np.isfinite(v)) for v in (state, xtx, xty))):
                    raise ValueError("invalid checkpoint state/time/phase/summary")
                expected = family.reconstruct(state, n * FRAME_SECONDS)
                if not np.array_equal(expected, saved["physical_state"]):
                    raise ValueError("checkpoint physical-state reconstruction disagrees")
                self.next_frame, self.state = n, state.copy()
                self.summary.xtx, self.summary.xty, self.summary.count = xtx.copy(), xty.copy(), n
                self.boundary_energy, self.clipped = float(saved["boundary_energy"]), int(saved["clipped"])
            self.file = h5py.File(self.directory / "record.h5", "r+")
            if self.file.attrs["binding"] != self.binding or not np.all(self.file["committed"][:n]):
                self.file.close(); raise ValueError("record does not match checkpoint committed prefix")
            self.file["committed"][n:] = False
            durable_hdf(self.file)
        else:
            self.directory.mkdir(parents=True, exist_ok=True)
            if any(self.directory.iterdir()):
                raise ValueError("new record directory must be empty")
            self.file = h5py.File(self.directory / "record.h5", "w")
            self.file.attrs["binding"] = self.binding
            for key, width, dtype in (("aperture_mean", t.conversions, "f8"),
                    ("code", t.conversions, "i4"), ("raw_clipped", t.conversions, "?"),
                    ("pixel", 4096, "f8"), ("pixel_clipped", 4096, "?")):
                self.file.create_dataset(key, shape=(FRAMES, width), chunks=(1, width), dtype=dtype)
            self.file.create_dataset("committed", shape=(FRAMES,), dtype="?")
            self.file.create_dataset("pixel_relative_timestamp", data=t.obs_assigned[operator.pixel_rows])
            self.file.create_dataset("raw_relative_aperture_start", data=t.conv_aperture_start)
            self.file.create_dataset("raw_relative_aperture_end", data=t.conv_aperture_end)
            durable_hdf(self.file); self._save()

    def _save(self):
        time = self.next_frame * FRAME_SECONDS
        atomic_npz(self.checkpoint, binding=np.asarray(self.binding), next_frame=self.next_frame,
            time=time, last_row=63 if self.next_frame else -1,
            charge_phase=np.asarray("before_next_frame_start_charge"), state=self.state,
            physical_state=self.operator.family.reconstruct(self.state, time),
            xtx=self.summary.xtx, xty=self.summary.xty,
            boundary_energy=self.boundary_energy, clipped=self.clipped)

    def commit(self, state, result, boundary_energy):
        n, rows = self.next_frame, self.operator.pixel_rows
        if result.index != n or n >= FRAMES:
            raise ValueError("record commit is not the next prescribed frame")
        self.file["aperture_mean"][n] = result.aperture_mean
        self.file["code"][n] = result.code
        self.file["raw_clipped"][n] = result.raw_clipped
        self.file["pixel"][n] = result.value[rows]
        self.file["pixel_clipped"][n] = result.clipped[rows]
        self.file["committed"][n] = True
        durable_hdf(self.file)
        self.summary.add(result.assigned[rows], result.value[rows])
        self.state, self.next_frame = state.copy(), n + 1
        self.boundary_energy += boundary_energy
        self.clipped += int(np.count_nonzero(result.raw_clipped))
        self._save()

    def close(self):
        self.file.close()
