"""One immutable cubic in original physical-noise port coordinates.

Rows contain ordinary power coefficients, not unscaled derivatives:
x(t)=c0+c1*(t-origin)+c2*(t-origin)**2+c3*(t-origin)**3.
Row j therefore has the declared port unit / seconds**j. This finite numerical
representation neither selects a spectrum nor redraws a stochastic path.
The supplying spectral adapter owns its analytic truncation bound; coefficient
and Horner arithmetic errors are separate, as for the existing held waveform.
"""
from dataclasses import dataclass, field
import math
from numbers import Real

import numpy as np

from physical_noise_ports import PhysicalNoiseLayout, NoiseSample


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and real")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    return value


@dataclass(frozen=True, init=False, eq=False)
class NoisePolynomial:
    """Owned LE-float64 coefficients on one fixed half-open original cell.

    Coefficients and support survive numerical, aperture and storage splits
    unchanged. The 32*P+24 numeric byte count excludes Python object overhead;
    the caller bounds live objects and evaluation work. No history is retained.
    """
    layout: object
    _numeric_bytes: bytes = field(repr=False)
    origin_seconds: float
    support_start_seconds: float
    support_end_seconds: float

    def __init__(self, layout, coefficients, origin_seconds,
                 support_start_seconds, support_end_seconds):
        if not isinstance(layout, PhysicalNoiseLayout):
            raise ValueError("a bound physical-noise layout is required")
        origin = _real(origin_seconds, "polynomial origin")
        start = _real(support_start_seconds, "polynomial support start")
        end = _real(support_end_seconds, "polynomial support end")
        if not start < origin < end or not math.isfinite(end - start):
            raise ValueError("finite support must strictly contain its fixed origin")
        count = layout.numeric_value_count
        if isinstance(coefficients, np.ndarray):
            if coefficients.shape != (4, count) or coefficients.dtype.kind not in "fiu":
                raise ValueError("four real coefficient rows must cover the original physical ports")
            values = coefficients
        else:
            if not hasattr(coefficients, "__len__") or len(coefficients) != 4:
                raise ValueError("exactly four finite coefficient rows are required")
            rows = []
            for row in coefficients:
                if len(rows) >= 4 or not hasattr(row, "__len__") or len(row) != count:
                    raise ValueError("coefficient rows must cover all original physical ports")
                owned = []
                for value in row:
                    if len(owned) >= count:
                        raise ValueError("coefficient row exceeds the bound physical port count")
                    owned.append(_real(value, "noise coefficient"))
                if len(owned) != count:
                    raise ValueError("coefficient row must cover every original physical port")
                rows.append(tuple(owned))
            if len(rows) != 4:
                raise ValueError("exactly four finite coefficient rows are required")
            values = rows
        try:
            with np.errstate(over="raise", invalid="raise"):
                packed = np.asarray(values, dtype="<f8").tobytes(order="C")
        except (FloatingPointError, OverflowError) as error:
            raise ValueError("noise coefficients must be representable finite float64") from error
        frozen = np.frombuffer(packed, dtype="<f8").reshape(4, count)
        if not np.all(np.isfinite(frozen)):
            raise ValueError("noise coefficients must be finite")
        # A physically absent/zero-conductance source must vanish as a whole
        # polynomial, not merely cancel at one queried time. Interval-specific
        # external-mux zero paths are additionally checked by the assembler.
        for row in frozen:
            layout._validate_values(row)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "_numeric_bytes", packed)
        object.__setattr__(self, "origin_seconds", origin)
        object.__setattr__(self, "support_start_seconds", start)
        object.__setattr__(self, "support_end_seconds", end)

    @property
    def numeric_array(self):
        return np.frombuffer(self._numeric_bytes, dtype="<f8").reshape(4, self.layout.numeric_value_count)

    @property
    def sample_numeric_bytes(self):
        return len(self._numeric_bytes) + 24

    @property
    def state_ids(self):
        return self.layout.state_ids

    @property
    def voltage_reference_id(self):
        return self.layout.voltage_reference_id

    def sample_at(self, absolute_seconds):
        time = _real(absolute_seconds, "polynomial query time")
        if not self.support_start_seconds <= time < self.support_end_seconds:
            raise ValueError("polynomial query lies outside its original half-open cell")
        delta = time - self.origin_seconds
        rows = self.numeric_array
        try:
            with np.errstate(over="raise", invalid="raise"):
                values = ((rows[3] * delta + rows[2]) * delta + rows[1]) * delta + rows[0]
        except FloatingPointError as error:
            raise ValueError("polynomial noise evaluation must remain representable") from error
        return NoiseSample(self.layout, values)
