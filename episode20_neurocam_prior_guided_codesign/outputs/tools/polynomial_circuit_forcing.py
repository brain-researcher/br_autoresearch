"""Bind one original-port cubic to the joint conditioned circuit solver.

The linear Schur projection is performed once per coefficient row. Solver
midpoints evaluate this fixed polynomial; Newton iterations do not redraw or
reproject noise. Original-port work is evaluated at the actual midpoint state,
including access/field/external-port corrections. It is never mean noise times
mean voltage. No spectrum, temperature or physical parameter is inferred.
"""
import math
from numbers import Real

import numpy as np

from analog_conditioning import ControlledVoltageDrives
from polynomial_noise import NoisePolynomial


class PolynomialCircuitForcing:
    def __init__(self, polynomial, *, interval_start_seconds, duration_seconds,
                 base_controlled_drives, field_port_volts, external_branch_drives,
                 max_stored_numeric_bytes, check):
        if not isinstance(polynomial, NoisePolynomial):
            raise ValueError("forcing requires one original physical-port polynomial")
        if not isinstance(base_controlled_drives, ControlledVoltageDrives) or not callable(check):
            raise ValueError("polynomial forcing requires the joint controlled circuit and a budget check")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
               or not math.isfinite(value) for value in (interval_start_seconds, duration_seconds)):
            raise ValueError("polynomial interval must be finite and explicit")
        start, duration = float(interval_start_seconds), float(duration_seconds)
        end = start + duration
        if (duration <= 0 or not math.isfinite(end) or not start < end
                or start < polynomial.support_start_seconds or end > polynomial.support_end_seconds):
            raise ValueError("the complete physical interval must lie inside polynomial support")
        count = len(polynomial.state_ids)
        # Includes the original polynomial even if its caller also accounts it;
        # Python object overhead and solver/transient workspace are separate.
        numeric_bytes = polynomial.sample_numeric_bytes + 8 * (4 * count + 4 * 64 + 4096)
        if (isinstance(max_stored_numeric_bytes, bool) or not isinstance(max_stored_numeric_bytes, int)
                or numeric_bytes > max_stored_numeric_bytes):
            raise ValueError("polynomial circuit retained numeric-storage limit exceeded")
        field = np.asarray(field_port_volts)
        if field.shape != (4096,) or field.dtype.kind not in "fiu" or not np.all(np.isfinite(field)):
            raise ValueError("field port voltages must be a finite real 4096-vector")
        self._field = np.frombuffer(np.asarray(field, dtype="<f8").tobytes(), dtype="<f8")
        self._external = tuple(external_branch_drives)
        self._polynomial = polynomial
        self._base = base_controlled_drives
        self._start, self._duration = start, duration
        self._check = check
        projected = np.empty((4, count))
        amplifier = np.zeros((4, 64))
        for index, row in enumerate(polynomial.numeric_array):
            check()
            sample = polynomial.layout.sample_from_values(row)
            projected[index] = polynomial.layout.project_current(
                sample, external_branch_drives=self._external)
            values = polynomial.layout.amplifier_input_volts(sample)
            if values is not None:
                amplifier[index] = values
        self._projected = np.frombuffer(projected.astype("<f8", copy=False).tobytes(), dtype="<f8").reshape(4, count)
        self._amplifier = np.frombuffer(amplifier.astype("<f8", copy=False).tobytes(), dtype="<f8").reshape(4, 64)
        self.numeric_storage_bytes = numeric_bytes
        check()

    def _time(self, local_time_seconds):
        if (isinstance(local_time_seconds, (bool, np.bool_)) or not isinstance(local_time_seconds, Real)
                or not math.isfinite(local_time_seconds) or not 0 <= local_time_seconds < self._duration):
            raise ValueError("forcing queries must lie inside the physical half-open interval")
        absolute = self._start + float(local_time_seconds)
        if not self._polynomial.support_start_seconds <= absolute < self._polynomial.support_end_seconds:
            raise ValueError("rounded forcing time lies outside the original polynomial cell")
        return absolute

    @staticmethod
    def _at(rows, delta):
        try:
            with np.errstate(over="raise", invalid="raise"):
                values = ((rows[3] * delta + rows[2]) * delta + rows[1]) * delta + rows[0]
        except FloatingPointError as error:
            raise ValueError("projected cubic forcing must remain finite") from error
        if not np.all(np.isfinite(values)):
            raise ValueError("projected cubic forcing must remain finite")
        return values

    def evaluate(self, local_time_seconds):
        self._check()
        delta = self._time(local_time_seconds) - self._polynomial.origin_seconds
        return (self._at(self._projected, delta),
                self._base.with_input_noise_volts(self._at(self._amplifier, delta)))

    def work_rates(self, local_time_seconds, midpoint_state):
        self._check()
        sample = self._polynomial.sample_at(self._time(local_time_seconds))
        work = self._polynomial.layout.work(sample, mean_state_volts=midpoint_state,
            field_port_volts=self._field, duration_seconds=1., external_branch_drives=self._external)
        return (work.noise_current_source_interval_work_joule,
                work.noise_access_voltage_source_interval_work_joule,
                work.field_source_interval_work_correction_joule,
                work.daq_external_source_interval_work_correction_joule)
