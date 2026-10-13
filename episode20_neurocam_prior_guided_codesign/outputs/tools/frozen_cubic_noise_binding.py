"""Owned spectral/clock replay without expanding every preflight polynomial.

Only this exact binding has the descriptor path. Ordinary providers retain
exhaustive original-coefficient validation/hash replay. The proof concerns
finite arithmetic and structurally absent physical coordinates, not accuracy
of trigonometric arithmetic, Taylor approximation or circuit output. Every
actual use still constructs and validates the original four coefficient rows.
No draw, physical parameter, clock, tolerance or polynomial origin changes.
"""
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import math
from numbers import Real
import struct
from types import MappingProxyType

import numpy as np

from circuit_noise_provider import SpectralCubicCircuitNoise
from physical_noise_ports import PhysicalNoiseLayout
from polynomial_noise import NoisePolynomial
from spectral_noise import SpectralNoise, _upper_add, _upper_multiply


@dataclass(frozen=True, slots=True)
class FrozenOriginalCellClock:
    """The existing integer-tick cycle with the exact original _shift order."""
    cycle_count: int
    cycle_ticks: int
    slot_ticks: int
    ticks_per_second: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in
               (self.cycle_count, self.cycle_ticks, self.slot_ticks, self.ticks_per_second)):
            raise ValueError("finite positive integer original clock counts required")
        if self.cycle_ticks % self.slot_ticks:
            raise ValueError("original slots must exactly partition each cycle")
        try:
            period = self.cycle_ticks / self.ticks_per_second
            end = self.cycle_count * period
            slot = self.slot_ticks / self.ticks_per_second
        except OverflowError as error:
            raise ValueError("original clock must be representable") from error
        if not (math.isfinite(end) and 0. < slot <= period <= end):
            raise ValueError("original clock must be finite and ordered")

    @property
    def period(self):
        return self.cycle_ticks / self.ticks_per_second

    def _shift(self, cycle, relative):
        if relative == 0.:
            return cycle * self.period
        if relative == self.period:
            return (cycle + 1) * self.period
        return cycle * self.period + relative

    def __call__(self, key):
        if (isinstance(key, bool) or not isinstance(key, Real) or not math.isfinite(key)
                or not 0. < key < self.cycle_count * self.period):
            raise ValueError("original clock query must be strictly inside the record")
        cycle_guess = int(key / self.period)
        slots = self.cycle_ticks // self.slot_ticks
        slot_seconds = self.slot_ticks / self.ticks_per_second
        for cycle in range(max(0, cycle_guess - 1), min(self.cycle_count, cycle_guess + 2)):
            slot_guess = int((key - cycle * self.period) / slot_seconds)
            for slot in range(max(0, slot_guess - 1), min(slots, slot_guess + 2)):
                start = self._shift(cycle, (slot * self.slot_ticks) / self.ticks_per_second)
                end = self._shift(cycle, ((slot + 1) * self.slot_ticks) / self.ticks_per_second)
                if start < key < end:
                    return start, end, start + (end - start) / 2.
        raise ValueError("noise key lies on an original clock boundary")


_PACKED = ("_omega_array", "_frequency_indices", "_port_indices", "_amplitude_real", "_amplitude_imag")
_COUNTERS = ("_evaluations", "_term_evaluations", "_cubic_evaluations")


class _EvaluationView:
    """Only accounting is writable while reusing the unchanged numeric kernel."""
    _time = SpectralNoise._time

    def __init__(self, owner):
        self._owner = owner
        for name in _COUNTERS:
            setattr(self, name, getattr(owner, name))

    def __getattr__(self, name):
        return getattr(self._owner, name)


class _OwnedSpectrum(SpectralNoise):
    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("the spectral numerical binding is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise AttributeError("the spectral numerical binding is immutable")

    def __init__(self, original, check):
        super().__init__(frequencies_hz=original.frequencies_hz,
            quadrature_widths_hz=original.quadrature_widths_hz,
            port_ids=original.port_ids, port_units=original.port_units,
            latent_ids=original.latent_ids, sparse_terms=original.sparse_terms,
            quadratures=original.quadratures, time_window_seconds=original.time_window_seconds,
            limits=original._limits, check=check)
        # A caller mutation before binding must not be silently replaced by
        # reconstruction from stale metadata. Compare the actual old kernels.
        for name in _PACKED:
            check()
            current, previous = getattr(self, name), getattr(original, name)
            if (current.dtype != previous.dtype or current.shape != previous.shape
                    or current.tobytes() != previous.tobytes()):
                raise ValueError("spectral evaluation arrays disagree with their original basis")
            dtype = "<i8" if current.dtype.kind == "i" else "<f8"
            owned = np.frombuffer(current.astype(dtype, copy=False).tobytes(), dtype=dtype).reshape(current.shape)
            object.__setattr__(self, name, owned)
        for name in ("_amplitudes", "_absolute_bounds", "_derivative_bounds", "_omega"):
            if getattr(self, name) != getattr(original, name):
                raise ValueError("spectral analytic metadata changed before binding")
        for name in _COUNTERS:
            value = getattr(original, name)
            if type(value) is not int or value < 0:
                raise ValueError("original spectral accounting is invalid")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_sealed", True)

    def _evaluate_owned(self, method, time):
        view = _EvaluationView(self)
        try:
            return method(view, time)
        finally:
            # Failed numerical attempts consume exactly the old counters too.
            for name in _COUNTERS:
                object.__setattr__(self, name, getattr(view, name))

    def cubic_coefficients(self, origin_seconds):
        return self._evaluate_owned(SpectralNoise.cubic_coefficients, origin_seconds)

    def evaluate(self, time_seconds):
        return self._evaluate_owned(SpectralNoise.evaluate, time_seconds)


def _proof(spectrum, layout, check):
    """Sufficient, deliberately conservative; failure keeps exhaustive replay."""
    count = len(spectrum._amplitudes)
    # Sequential binary64 reductions have gamma_n < 1/3 here. The factor
    # eight covers the real/imag products, reduction and gradual underflow;
    # these are finiteness envelopes, not claims of correctly rounded trig.
    if count * np.finfo(np.float64).eps >= .25:
        return False
    try:
        omega = max(1., max(spectrum.angular_frequencies_rad_per_second))
        smallest = float(np.nextafter(0., 1.))
        envelope = _upper_add(max(spectrum._absolute_bounds), count * smallest)
        envelope = _upper_multiply(8., envelope)
        bounds = [envelope]
        for _ in range(3):
            bounds.append(_upper_multiply(bounds[-1], omega))
        # The existing cubic kernel's final /6 is after all three omega
        # products; omitting factorial divisions also encloses intermediates.
        radius = max(1., _upper_add(abs(spectrum.time_window_seconds[0]),
                                   abs(spectrum.time_window_seconds[1])))
        horner = bounds[-1]
        for value in reversed(bounds[:-1]):
            horner = _upper_add(value, _upper_multiply(radius, horner))
        if horner >= np.finfo(np.float64).max / 8.:
            return False
    except ValueError:
        return False
    nonzero = set()
    for _, port, amplitude in spectrum._amplitudes:
        check()
        if amplitude != 0.:
            nonzero.add(port)
    if any(layout._nodal_count + i in nonzero for i in layout._zero_h):
        return False
    if any(layout._nodal_count + layout._term_count + i in nonzero for i in layout._zero_r):
        return False
    branches = {branch.branch_id: branch for branch in layout.external_daq_branches}
    start = layout.numeric_value_count - len(layout.external_daq_current_ports)
    for index, port in enumerate(layout.external_daq_current_ports):
        check()
        branch = branches[port.branch_id]
        if ((branch.track_conductance_siemens == 0. or branch.off_conductance_siemens == 0.)
                and start + index in nonzero):
            # A nonzero waveform on a potentially open switch needs the
            # ordinary per-phase coefficient checks, not this global proof.
            return False
    return True


def _hash_value(digest, value, check):
    """Stream only the already bounded owned numerical/port basis once."""
    check()
    if value is None:
        digest.update(b"n")
    elif type(value) is str:
        raw = value.encode("utf-8")
        digest.update(b"s" + struct.pack("<Q", len(raw)) + raw)
    elif type(value) is int:
        _hash_value(digest, str(value), check)
    elif type(value) is float:
        digest.update(b"f" + struct.pack("<d", value))
    elif type(value) is complex:
        digest.update(b"c" + struct.pack("<dd", value.real, value.imag))
    elif isinstance(value, tuple):
        digest.update(b"t" + struct.pack("<Q", len(value)))
        for item in value:
            _hash_value(digest, item, check)
    elif is_dataclass(value):
        _hash_value(digest, type(value).__name__, check)
        _hash_value(digest, tuple((f.name, getattr(value, f.name)) for f in fields(value)), check)
    else:
        raise ValueError("unexpected mutable value in the owned physical/spectral basis")


@dataclass(frozen=True, slots=True)
class _Descriptor:
    owner: object
    cell: tuple

    @property
    def sample_numeric_bytes(self):
        return self.owner.sample_numeric_bytes

    @property
    def layout(self):
        return self.owner.layout


@dataclass(frozen=True, init=False)
class FrozenCubicNoiseBinding(SpectralCubicCircuitNoise):
    """Exact typed optimization; instantiate through bind_frozen_cubic_noise."""
    def __init__(self, original, spectrum, clock, check, peak_numeric_bound):
        if (type(spectrum) is not _OwnedSpectrum or type(clock) is not FrozenOriginalCellClock
                or type(original.layout) is not PhysicalNoiseLayout
                or not _proof(spectrum, original.layout, check)):
            raise ValueError("descriptor binding requires its owned finite/zero-coordinate proof")
        super().__init__(layout=original.layout, spectrum=spectrum,
            port_absolute_tolerances=original.port_absolute_tolerances,
            cell_for_key=clock, check=check)
        if (self._normalized_derivative_bound != original._normalized_derivative_bound
                or self._normalized_fourth_derivative_bound != original._normalized_fourth_derivative_bound):
            raise ValueError("provider remainder metadata disagrees with its owned spectral basis")
        zero = self.layout.sample_from_values(np.zeros(self.layout.numeric_value_count))
        object.__setattr__(self, "_zero_sample", zero)
        object.__setattr__(self, "_binding_peak_numeric_bound", peak_numeric_bound)
        digest = hashlib.sha256(b"owned_original_cell_cubic_v1\0")
        for name in ("state_ids", "voltage_reference_id", "port_ids", "port_units",
                     "nodal_current_ports", "contact_stars", "contact_ownership", "contact_noise_role",
                     "amplifier_input_noise_role", "excluded_retained_contact_components",
                     "external_daq_branches", "external_daq_current_ports"):
            _hash_value(digest, (name, getattr(self.layout, name)), check)
        for name in ("frequencies_hz", "quadrature_widths_hz", "latent_ids", "sparse_terms",
                     "quadratures", "time_window_seconds", "_absolute_bounds", "_derivative_bounds"):
            _hash_value(digest, (name, getattr(spectrum, name)), check)
        for name in _PACKED:
            _hash_value(digest, name, check)
            digest.update(getattr(spectrum, name).tobytes())
        _hash_value(digest, (clock, self.port_absolute_tolerances,
            self._normalized_fourth_derivative_bound,
            "unchanged_outward_M1_omega3_radius4_over24;original_half_open_support;float_errors_excluded"), check)
        object.__setattr__(self, "_basis_digest", digest.digest())

    def interval_descriptor(self, array, key, start, end, daq_selected_sources):
        self._check()
        if (array.physical_noise_layout is not self.layout
                or array.analog_conditioning is None):
            raise ValueError("owned cubic binding needs its original conditioned array/layout")
        cell = self._cell(key)
        self._cell_bound(cell)
        if not cell[0] <= start < end <= cell[1]:
            raise ValueError("the complete interval must stay inside its original polynomial support")
        # This is only a metadata/actual mux-phase check. Structural proof
        # already established that every possibly zero-G original coordinate
        # is identically zero. No zero waveform is used for physical advance.
        array.validate_noise_interval(self._zero_sample, daq_selected_sources=daq_selected_sources)
        if array.physical_noise_layout is not self.layout:
            raise ValueError("physical layout changed during interval validation")
        return _Descriptor(self, cell)

    def joint_digest(self, field, sample):
        self._check()
        if type(sample) is _Descriptor and sample.owner is self:
            cell = sample.cell
        elif type(sample) is NoisePolynomial and sample is self._cached_polynomial and sample.layout is self.layout:
            cell = (sample.support_start_seconds, sample.support_end_seconds, sample.origin_seconds)
        else:
            raise ValueError("the owned binding cannot certify an arbitrary polynomial")
        digest = hashlib.sha256(b"field_and_owned_original_cubic_v1\0")
        digest.update(field.astype("<f8", copy=False).tobytes(order="C"))
        digest.update(self._basis_digest)
        digest.update(struct.pack("<ddd", *cell))
        return digest.digest()

    @property
    def metadata(self):
        result = dict(super().metadata)
        result.update(complete_preflight="immutable_basis_original_clock_descriptor",
            use_time_original_coefficients_still_materialized_and_validated=True,
            binding_numeric_peak_admission_bytes=self._binding_peak_numeric_bound,
            binding_retained_numeric_bytes=self.spectrum.numeric_storage_bytes + self.layout.sample_numeric_bytes + 32,
            descriptor_reserves_full_original_polynomial_bytes=self.sample_numeric_bytes,
            arbitrary_callbacks_retain_exhaustive_replay=True)
        return MappingProxyType(result)


def bind_frozen_cubic_noise(provider, *, clock, max_numeric_bytes, check):
    """Return an owned fast binding, or unchanged provider if proof is absent.

    A mismatch between source metadata and its actual evaluation arrays is a
    concrete integrity error and refuses. Unsupported clocks/types, insufficient
    finite/zero proof or binding workspace instead keep the exhaustive route.
    The peak cap conservatively reserves three source numerical payloads plus
    two port vectors; it is not an RSS claim and excludes Python object bytes.
    """
    if type(max_numeric_bytes) is not int or max_numeric_bytes <= 0 or not callable(check):
        raise ValueError("explicit binding numeric cap and callback required")
    check()
    if (type(provider) is not SpectralCubicCircuitNoise or type(clock) is not FrozenOriginalCellClock
            or provider._cell_for_key is not clock or type(provider.spectrum) is not SpectralNoise
            or type(provider.layout) is not PhysicalNoiseLayout
            or provider._cached_polynomial is not None):
        # Bind before the first polynomial. A prior cache could describe an
        # older mutable spectrum; preserving that provider is safer than
        # silently replacing an already promised cell with reconstructed rows.
        return provider
    peak = 3 * provider.spectrum.numeric_storage_bytes + 16 * provider.layout.numeric_value_count + 32
    if peak > max_numeric_bytes:
        return provider
    spectrum = _OwnedSpectrum(provider.spectrum, check)
    if not _proof(spectrum, provider.layout, check):
        return provider
    result = FrozenCubicNoiseBinding(provider, spectrum, clock, check, peak)
    check()
    return result
