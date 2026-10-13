"""Explicit left limits of one already bound physical-interval cubic.

Radau's final stage uses the current interval's right endpoint. That stage
must retain this interval's polynomial, field and external-branch context;
it must not acquire the next cell, switch the DAQ or apply a boundary charge.
The ordinary evaluate/work_rates methods retain their half-open contract.
"""
import math
from numbers import Real

import numpy as np

from polynomial_circuit_forcing import PolynomialCircuitForcing


class RadauPolynomialCircuitForcing(PolynomialCircuitForcing):
    """Add explicit endpoint queries without broadening ordinary queries."""

    def _time_left(self, local_time_seconds):
        if (isinstance(local_time_seconds, (bool, np.bool_))
                or not isinstance(local_time_seconds, Real)
                or not math.isfinite(local_time_seconds)
                or not 0 <= local_time_seconds <= self._duration):
            raise ValueError("left-limit queries must lie in the current closed physical interval")
        if local_time_seconds < self._duration:
            # Preserve the inherited interior rounding/domain checks as well.
            return self._time(local_time_seconds)
        absolute = self._start + self._duration
        if (not math.isfinite(absolute)
                or not self._polynomial.support_start_seconds <= absolute <= self._polynomial.support_end_seconds):
            raise ValueError("left-limit endpoint lies outside the original polynomial cell")
        return absolute

    def evaluate_left(self, local_time_seconds):
        """Evaluate this same interval at an interior or exact final stage."""
        self._check()
        delta = self._time_left(local_time_seconds) - self._polynomial.origin_seconds
        return (self._at(self._projected, delta),
                self._base.with_input_noise_volts(self._at(self._amplifier, delta)))

    def work_rates_left(self, local_time_seconds, state):
        """Original-port work at the stage state, with unchanged branch roles."""
        self._check()
        delta = self._time_left(local_time_seconds) - self._polynomial.origin_seconds
        # sample_at intentionally remains half-open. Construct an ordinary
        # validated original-port sample from the SAME polynomial at its left
        # limit; do not clamp time or select a neighbouring polynomial.
        layout = self._polynomial.layout
        sample = layout.sample_from_values(self._at(self._polynomial.numeric_array, delta))
        work = layout.work(sample, mean_state_volts=state, field_port_volts=self._field,
            duration_seconds=1., external_branch_drives=self._external)
        return (work.noise_current_source_interval_work_joule,
                work.noise_access_voltage_source_interval_work_joule,
                work.field_source_interval_work_correction_joule,
                work.daq_external_source_interval_work_correction_joule)
