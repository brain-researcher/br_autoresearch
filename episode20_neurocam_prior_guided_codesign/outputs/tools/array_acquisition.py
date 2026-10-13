"""Finite prescribed array-to-readout execution with no inferred DAQ recipe.

The runner exclusively owns advancement of its supplied array for this attempt
and constructs a fresh readout. Every physical event, binding and limit is
validated before the first array mutation. Aperture endpoints split physical
intervals exactly; field discretization remains caller-owned. Jointly computed
hold means are used directly, never a filter driven by source interval means.
Means inherit the sparse solver's local consistency estimate and the caller's
field discretization; the worst local estimate is not a global error certificate.

This is a fixed open-loop mechanism executor, not a policy, calibrated reference,
hardware-legality certificate or candidate comparison. No source/noise/load is
inferred. Direct source bindings declare a memoryless readout boundary; hold
bindings use only explicitly assembled physical groups, including their declared
external marker/reference ports. Active conditioning and missing physical costs
are not inferred.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
import struct
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np

from acquisition_readout import AcquisitionReadout, Observation, RawConversion, ReturnedSample
from cpe_array import NeurocamCPEArray
from daq_mux_hold import DAQGroupDrive
from polynomial_noise import NoisePolynomial
from sparse_array_rc import SolverControls, _array


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    return float(value)


def _bounded(values, cls, maximum, name):
    result = []
    for value in values:
        if not isinstance(value, cls) or len(result) >= maximum:
            raise ValueError(f"{name} has an unsupported item or exceeds its finite limit")
        result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class PhysicalInterval:
    start: float
    end: float
    gate_row: int
    field_volts: object
    voltage_reference_id: str
    daq_selected_sources: object
    injected_charge_coulomb: object


@dataclass(frozen=True, slots=True)
class ProviderInterval:
    """Compact physical event with a caller-defined field key and sparse charge.

    ``field_key`` is an immutable integer or finite real scalar, such as an
    explicitly selected evaluation time. It does not choose a sampling/hold
    convention. ``injected_charge_coulomb`` contains distinct ``(state, charge)``
    pairs; an empty tuple explicitly means no boundary charge. Selections and
    charge entries are snapshotted, including when their caller supplied lists.
    """
    start: float
    end: float
    gate_row: int
    field_key: object
    voltage_reference_id: str
    daq_selected_sources: object
    injected_charge_coulomb: object

    def __post_init__(self):
        key = self.field_key
        if isinstance(key, bool) or not isinstance(key, Real):
            raise ValueError("A provider field key must be an immutable integer or finite real scalar")
        key = int(key) if isinstance(key, Integral) else _finite(key, "provider field key")
        object.__setattr__(self, "field_key", key)
        selections = self.daq_selected_sources
        if selections is not None:
            if not isinstance(selections, Mapping):
                raise ValueError("A provider event needs explicit DAQ selections or None")
            object.__setattr__(self, "daq_selected_sources", MappingProxyType(dict(selections)))
        charges, seen = [], set()
        for pair in self.injected_charge_coulomb:
            pair = tuple(pair)
            if len(pair) != 2:
                raise ValueError("Sparse boundary charge entries must contain a state index and charge")
            index, value = pair
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in seen:
                raise ValueError("Sparse boundary charges need distinct nonnegative integer state indices")
            charges.append((index, _finite(value, "boundary charge")))
            seen.add(index)
        object.__setattr__(self, "injected_charge_coulomb", tuple(charges))


def _field_digest(field):
    # The comparison concerns the already validated physical float64 values,
    # independent of the provider's input dtype or C/Fortran storage order.
    return hashlib.sha256(field.astype("<f8", copy=False).tobytes(order="C")).digest()


def _bind_noise_provider(array, provider):
    """Bind the array's already fixed physical port basis, without drawing noise."""
    layout = array.physical_noise_layout
    if provider is None:
        if layout is not None:
            raise ValueError("A configured physical noise layout requires an explicit noise provider")
        return None, 0
    if not callable(provider) or layout is None:
        raise ValueError("A noise provider requires the array's explicit fixed physical noise layout")
    numeric_bytes = getattr(provider, "sample_numeric_bytes", layout.sample_numeric_bytes)
    if (type(numeric_bytes) is not int or numeric_bytes < layout.sample_numeric_bytes):
        raise ValueError("A noise provider must declare its complete live sample numeric bytes")
    return layout, numeric_bytes


def _provider_noise(array, provider, layout, key, daq_selected_sources, *, interval_start, interval_end,
                    admitted_numeric_bytes, verify_only=False):
    if provider is None:
        if array.physical_noise_layout is not None:
            raise ValueError("Physical noise layout changed after attempt binding")
        return None
    if array.physical_noise_layout is not layout:
        raise ValueError("Physical noise layout changed after attempt binding")
    if verify_only:
        from frozen_cubic_noise_binding import FrozenCubicNoiseBinding
        if type(provider) is FrozenCubicNoiseBinding:
            if provider.sample_numeric_bytes != admitted_numeric_bytes:
                raise ValueError("Noise sample exceeds the provider's declared numeric storage")
            return provider.interval_descriptor(array, key, interval_start, interval_end,
                                                daq_selected_sources)
    sample = provider(key)
    declared_bytes = getattr(provider, "sample_numeric_bytes", layout.sample_numeric_bytes)
    if (type(declared_bytes) is not int or declared_bytes != admitted_numeric_bytes
            or sample.sample_numeric_bytes > admitted_numeric_bytes):
        raise ValueError("Noise sample exceeds the provider's declared numeric storage")
    if isinstance(sample, NoisePolynomial):
        if not sample.support_start_seconds <= interval_start < interval_end <= sample.support_end_seconds:
            raise ValueError("The complete physical interval must lie within its fixed noise polynomial support")
    # Validate against this actual mux phase during the complete preflight,
    # not only after earlier intervals have already advanced the circuit.
    array.validate_noise_interval(sample, daq_selected_sources=daq_selected_sources)
    if sample.layout is not layout or array.physical_noise_layout is not layout:
        raise ValueError("Noise sample must retain the exact bound physical port layout")
    return sample


def _joint_input_digest(field, noise_sample, *, noise_provider=None):
    """Bind original physical amplitudes, including ports that cancel on nodes."""
    if noise_sample is None:
        return _field_digest(field)
    if noise_provider is not None:
        from frozen_cubic_noise_binding import FrozenCubicNoiseBinding
        if type(noise_provider) is FrozenCubicNoiseBinding:
            return noise_provider.joint_digest(field, noise_sample)
    polynomial = isinstance(noise_sample, NoisePolynomial)
    digest = hashlib.sha256(b"field_and_physical_noise_polynomial_v1\0" if polynomial
                            else b"field_and_physical_noise_v1\0")
    digest.update(field.astype("<f8", copy=False).tobytes(order="C"))
    if polynomial:
        digest.update(struct.pack("<ddd", noise_sample.origin_seconds,
                                  noise_sample.support_start_seconds, noise_sample.support_end_seconds))
    digest.update(noise_sample.numeric_array.tobytes(order="C"))
    return digest.digest()


def _daq_external_numeric_bytes(selections):
    """Retained float64 external voltages; Python objects/IDs are excluded.

    Count each physical event's values, even if immutable drives are shared
    between events. Frontend admission separately validates the complete ports.
    """
    if selections is None:
        return 0
    if not isinstance(selections, Mapping):
        raise ValueError("Explicit DAQ selections must be a mapping or None")
    return 8 * sum(len(value.external_volts) for value in selections.values()
                   if isinstance(value, DAQGroupDrive))


@dataclass(frozen=True)
class ArrayInputBinding:
    input_id: str
    adc_id: str
    kind: str
    source_index: int | None
    hold_adc_id: str | None

    def __post_init__(self):
        if any(not isinstance(value, str) or not value for value in (self.input_id, self.adc_id)):
            raise ValueError("Every binding needs explicit input and physical ADC identities")
        if self.kind == "source":
            if (isinstance(self.source_index, bool) or not isinstance(self.source_index, int)
                    or not 0 <= self.source_index < 64 or self.hold_adc_id is not None):
                raise ValueError("A direct source binding needs source_index 0..63 and no hold ADC")
        elif self.kind == "hold":
            if self.source_index is not None or self.hold_adc_id != self.adc_id:
                raise ValueError("A hold binding must name its own physical ADC and no direct source")
        else:
            raise ValueError("Only explicit source or hold bindings are implemented")


@dataclass(frozen=True)
class AcquisitionLimits:
    max_physical_intervals: int
    max_split_intervals: int
    max_conversions: int
    max_returned_samples: int
    max_observations: int
    max_stored_plan_numeric_bytes: int

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an explicit positive integer")


@dataclass(frozen=True)
class IntervalEvidence:
    physical_event_index: int
    start: float
    end: float
    event_boundary_charge_applied: bool


@dataclass(frozen=True)
class ExecutionFailure:
    """Confirmed progress only; a non-returning array call may have advanced.

    Operator counts cover calls that returned their diagnostics. Work within
    an interrupted/failed solver call is unknown, not asserted to be zero.
    """
    phase: str
    attempted_split_index: int
    completed_split_intervals: int
    physical_time: float
    array_call_returned: bool
    propagation_operator_products: int
    maximum_local_scaled_partition_error: float
    exception_type: str
    message: str


class AcquisitionExecutionError(RuntimeError):
    pass


class ArrayAcquisition:
    """One finite attempt; a failed attempt can never resume or replay.

    During propagation, query deadlines cannot exceed reached physical time.
    After all physical intervals, explicit finalize() advances only the digital
    publication horizon. It does not advance analog state through that delay.
    Raw diagnostics are returned only by successful finalization.

    With ProviderInterval, supply a deterministic ``field_provider(field_key)``
    returning a real finite 64x64 voltage field. The caller must freeze the
    provider's parameters for the attempt. All fields are traversed and checked
    before any electrical mutation, then regenerated once per physical event.
    A packed 32-byte digest per provider event detects changed float64 values.
    One copied field is held across that event's aperture splits, never the
    complete generated field plan. Provider failures/divergence close the
    attempt permanently before the current event's electrical mutation.

    An optional deterministic noise_provider uses the SAME field key and the
    array's prebound physical noise layout. It requires ProviderInterval for
    every event, never infers a time from an eager field record. Original port
    amplitudes join the field digest. One validated sample remains constant
    across all aperture/solver splits; there is no per-split random draw.

    Retained plan-byte accounting covers copied NumPy field/charge arrays,
    the shared zero-charge vector, packed field digests and 16 bytes per sparse
    charge pair (index/value payload), plus eight bytes per retained external
    input voltage per physical event. Interval/conversion/output caps bound
    object counts; Python object overhead, provider-owned parameters, the
    supplied array and transient field/charge/solver workspace are outside that
    byte counter. It is not a peak-process-memory bound.
    When noise is enabled, the counter additionally reserves one live physical
    amplitude vector, not one per event. Physical port/Python objects and
    temporary projection/digest buffers remain outside this numeric bound.
    Do not advance or mutate the supplied array outside this runner mid-attempt.
    """

    def __init__(self, *, array, physical_intervals, input_bindings, conversions,
                 returned_samples, observations, acquisition_window,
                 final_publication_horizon, controls, limits, field_provider=None, check=None,
                 noise_provider=None):
        if (not isinstance(array, NeurocamCPEArray) or not isinstance(controls, SolverControls)
                or not isinstance(limits, AcquisitionLimits)):
            raise ValueError("Supply the explicit array, numerical controls and finite limits")
        if ((array.analog_conditioning is not None and not callable(check))
                or (check is not None and not callable(check))):
            raise ValueError("Active conditioning requires an explicit resource callback")
        self._check = check
        intervals = _bounded(physical_intervals, (PhysicalInterval, ProviderInterval),
                             limits.max_physical_intervals, "physical plan")
        raw = _bounded(conversions, RawConversion, limits.max_conversions, "conversion plan")
        returned = _bounded(returned_samples, ReturnedSample, limits.max_returned_samples, "returned plan")
        published = _bounded(observations, Observation, limits.max_observations, "observation plan")
        bindings = _bounded(input_bindings, ArrayInputBinding, limits.max_conversions, "input bindings")
        if not intervals or not raw:
            raise ValueError("Supply at least one physical interval and one actual conversion")
        provider_count = sum(isinstance(event, ProviderInterval) for event in intervals)
        if provider_count and not callable(field_provider):
            raise ValueError("Provider intervals require an explicit deterministic field provider")
        if not provider_count and field_provider is not None:
            raise ValueError("A field provider requires at least one ProviderInterval")
        noise_layout, noise_bytes = _bind_noise_provider(array, noise_provider)
        if noise_provider is not None and provider_count != len(intervals):
            raise ValueError("Physical noise requires ProviderInterval keys for the complete physical plan")
        bound = {binding.input_id: binding for binding in bindings}
        if len(bound) != len(bindings) or set(bound) != {conversion.input_id for conversion in raw}:
            raise ValueError("Bindings must exactly and uniquely cover all raw input IDs")
        frontend = array.daq_frontend
        hold_ids = set() if frontend is None else {group.adc_id for group in frontend.groups}
        for binding in bindings:
            if (binding.kind == "hold" and binding.hold_adc_id not in hold_ids
                    or binding.kind == "source" and binding.adc_id in hold_ids):
                raise ValueError("A hold binding needs its assembled group; a direct binding cannot bypass that group")
        if any(bound[conversion.input_id].adc_id != conversion.adc_id for conversion in raw):
            raise ValueError("Every raw conversion must match its binding's physical ADC identity")
        size = len(array.state_ids)
        numeric_bytes = (8 * ((len(intervals) - provider_count) * (4096 + size) + size)
                         + 32 * provider_count + noise_bytes
                         + 16 * sum(len(event.injected_charge_coulomb) for event in intervals
                                    if isinstance(event, ProviderInterval)))
        if numeric_bytes > limits.max_stored_plan_numeric_bytes:
            raise ValueError("Field/charge plan exceeds its retained numeric-byte limit")
        copied = []
        for event in intervals:
            start, end = _finite(event.start, "physical start"), _finite(event.end, "physical end")
            if start >= end or not math.isfinite(end - start) or copied and copied[-1].end != start:
                raise ValueError("Physical intervals must be positive, finite and exactly contiguous")
            if (isinstance(event.gate_row, bool) or not isinstance(event.gate_row, int)
                    or not 0 <= event.gate_row < 64 or event.voltage_reference_id != array.voltage_reference_id):
                raise ValueError("Every event needs a legal row and the array's explicit voltage reference")
            selections = event.daq_selected_sources
            if frontend is None:
                if selections is not None:
                    raise ValueError("A physical DAQ selection requires an assembled frontend")
            else:
                if not isinstance(selections, Mapping):
                    raise ValueError("Supply every physical group's selection in each event")
                selections = MappingProxyType(dict(selections))
                frontend.resistor_edges(selected_sources=selections, source_node_offset=4096,
                                         first_hold_state=size - frontend.hold_state_count)
                frontend.external_drives(selected_sources=selections,
                    first_hold_state=size - frontend.hold_state_count,
                    voltage_reference_id=event.voltage_reference_id)
            numeric_bytes += _daq_external_numeric_bytes(selections)
            if numeric_bytes > limits.max_stored_plan_numeric_bytes:
                raise ValueError("Field/charge/external-voltage plan exceeds its retained numeric-byte limit")
            if isinstance(event, ProviderInterval):
                if any(index >= size for index, _ in event.injected_charge_coulomb):
                    raise ValueError("Sparse boundary charge state lies outside the supplied array")
                copied.append(ProviderInterval(start, end, event.gate_row, event.field_key,
                                               event.voltage_reference_id, selections,
                                               event.injected_charge_coulomb))
            else:
                field = _array("event field_volts", event.field_volts, (64, 64))
                charge = _array("event injected_charge_coulomb", event.injected_charge_coulomb, (size,))
                field.flags.writeable = charge.flags.writeable = False
                copied.append(PhysicalInterval(start, end, event.gate_row, field, event.voltage_reference_id,
                                               selections, charge))
        window = tuple(acquisition_window)
        if len(window) != 2:
            raise ValueError("Supply an explicit acquisition start/end window")
        acquisition_start, acquisition_end = (_finite(window[0], "acquisition start"),
                                             _finite(window[1], "acquisition end"))
        horizon = _finite(final_publication_horizon, "final publication horizon")
        if (not copied[0].start <= acquisition_start < acquisition_end <= copied[-1].end
                or not math.isfinite(acquisition_end - acquisition_start)
                or any(c.aperture_start < acquisition_start or c.completed_at > acquisition_end for c in raw)
                or horizon < max([copied[-1].end] + [s.available_at for s in returned]
                                 + [o.available_at for o in published])):
            raise ValueError("Physical plan, complete acquisition window and explicit final publication horizon disagree")
        # All readout types, IDs, digital arithmetic and publication relations
        # validate now. No existing readout/cached aperture state can be reused.
        readout = AcquisitionReadout(conversions=raw, returned_samples=returned, observations=published,
                                     max_conversions=limits.max_conversions,
                                     max_returned_samples=limits.max_returned_samples,
                                     max_observations=limits.max_observations,
                                     max_interval_updates=limits.max_split_intervals)
        boundaries = sorted(set(readout.required_aperture_boundaries)
                            | {event.start for event in copied} | {copied[-1].end})
        if len(boundaries) - 1 > limits.max_split_intervals:
            raise ValueError("The exact union of physical/aperture boundaries exceeds the split-interval limit")
        splits, event_index = [], 0
        for start, end in zip(boundaries, boundaries[1:]):
            while start >= copied[event_index].end:
                event_index += 1
            splits.append((event_index, start, end, start == copied[event_index].start))
        starts, ends = {}, {}
        for conversion in raw:
            starts.setdefault(conversion.aperture_start, []).append(conversion.input_id)
            ends.setdefault(conversion.aperture_end, []).append(conversion.input_id)
        # Validate every generated field before the first electrical mutation,
        # without retaining those arrays. The packed digest is the narrow
        # replacement for the eager route's frozen full-field snapshots.
        digests, provider_offsets = bytearray(), []
        for event in copied:
            if isinstance(event, ProviderInterval):
                provider_offsets.append(len(digests))
                field = _array("provider field_volts", field_provider(event.field_key), (64, 64))
                noise = _provider_noise(array, noise_provider, noise_layout, event.field_key,
                                        event.daq_selected_sources, interval_start=event.start, interval_end=event.end,
                                        admitted_numeric_bytes=noise_bytes, verify_only=True)
                digests.extend(_joint_input_digest(field, noise, noise_provider=noise_provider))
                del field, noise
            else:
                provider_offsets.append(None)
        self._array, self._controls, self._readout = array, controls, readout
        self._events, self._splits, self._bindings = tuple(copied), tuple(splits), bound
        self._field_provider, self._field_digests = field_provider, bytes(digests)
        self._noise_provider, self._noise_layout, self._noise_bytes = noise_provider, noise_layout, noise_bytes
        self._provider_offsets = tuple(provider_offsets)
        self._materialized_event, self._materialized_field = None, None
        self._materialized_noise = None
        self._starts, self._ends, self._active = starts, ends, Counter()
        self._zero_charge = np.zeros(size)
        self._zero_charge.flags.writeable = False
        self._acquisition_window = (acquisition_start, acquisition_end)
        self._horizon, self._numeric_bytes = horizon, numeric_bytes
        self._physical_time, self._publication_frontier = copied[0].start, copied[0].start
        self._next, self._status = 0, "ready"
        self._evidence = []
        self._operator_products, self._worst_local_error = 0, 0.0
        self._failure, self._report = None, None

    @property
    def status(self):
        return self._status

    @property
    def physical_time(self):
        return self._physical_time

    @property
    def evidence(self):
        return tuple(self._evidence)

    @property
    def failure(self):
        return self._failure

    def _require_not_failed(self):
        if self._status == "failed":
            raise AcquisitionExecutionError("This attempt failed permanently; do not replay from its advanced array")

    def _abort(self, exception, phase, array_call_returned, attempted_split_index=None):
        self._status = "failed"
        self._materialized_event, self._materialized_field = None, None
        self._materialized_noise = None
        index = self._next if attempted_split_index is None else attempted_split_index
        self._failure = ExecutionFailure(phase, index, len(self._evidence), self._physical_time,
                                         array_call_returned, self._operator_products, self._worst_local_error,
                                         type(exception).__name__, str(exception)[:1000])
        if not isinstance(exception, Exception):
            raise exception
        raise AcquisitionExecutionError(f"Acquisition attempt failed during {phase}; no continuation is permitted") from exception

    def advance_next(self):
        """Advance one exact split interval and return only newly published values."""
        self._require_not_failed()
        if self._status in ("physical_complete", "complete"):
            raise ValueError("The physical plan is finished; only final publication remains")
        attempted_index = self._next
        event_index, start, end, first = self._splits[attempted_index]
        event = self._events[event_index]
        phase, returned = "aperture scheduling", False
        try:
            if isinstance(event, ProviderInterval):
                phase = "field materialization"
                if first:
                    field = _array("provider field_volts", self._field_provider(event.field_key), (64, 64))
                    noise = _provider_noise(self._array, self._noise_provider, self._noise_layout, event.field_key,
                                            event.daq_selected_sources, interval_start=event.start, interval_end=event.end,
                                            admitted_numeric_bytes=self._noise_bytes)
                    offset = self._provider_offsets[event_index]
                    if _joint_input_digest(field, noise, noise_provider=self._noise_provider) != self._field_digests[offset:offset + 32]:
                        raise ValueError("The deterministic provider changed a prevalidated field or physical noise sample")
                    field.flags.writeable = False
                    self._materialized_event, self._materialized_field = event_index, field
                    self._materialized_noise = noise
                if self._materialized_event != event_index:
                    raise ValueError("The physical event has no validated materialized field")
                field, charge = self._materialized_field, self._zero_charge
                if first and event.injected_charge_coulomb:
                    charge = np.zeros(len(self._zero_charge))
                    for index, value in event.injected_charge_coulomb:
                        charge[index] = value
                    charge.flags.writeable = False
            else:
                field = event.field_volts
                charge = event.injected_charge_coulomb if first else self._zero_charge
            phase = "aperture scheduling"
            for key in self._ends.get(start, ()):
                self._active[key] -= 1
                if self._active[key] == 0:
                    del self._active[key]
            for key in self._starts.get(start, ()):
                self._active[key] += 1
            sources = tuple(sorted({self._bindings[key].source_index for key in self._active
                                    if self._bindings[key].kind == "source"}))
            # The array requires a nonempty diagnostic observation. This fallback
            # is internal only, is not digitized, and adds no acquisition count.
            sources = sources or (0,)
            phase = "array propagation"
            result = self._array.advance_row(gate_row=event.gate_row, duration_seconds=end - start,
                                              field_volts=field,
                                              voltage_reference_id=event.voltage_reference_id,
                                              observed_sources=sources,
                                              injected_charge_coulomb=charge,
                                              controls=self._controls, daq_selected_sources=event.daq_selected_sources,
                                              **({"check": self._check} if self._array.analog_conditioning is not None else {}),
                                              **({"noise_sample": self._materialized_noise} if self._noise_provider is not None else {}),
                                              **({"interval_start_seconds": start}
                                                 if isinstance(self._materialized_noise, NoisePolynomial) else {}))
            returned = True
            self._physical_time = end
            self._publication_frontier = end
            self._operator_products += result.kernel_interval.operator_products
            self._worst_local_error = max(self._worst_local_error, result.kernel_interval.maximum_scaled_partition_error)
            phase = "readout accumulation"
            if self._active:
                source_means = dict(zip(sources, result.kernel_interval.observation.output_mean_volts))
                hold_means = {value.adc_id: value.mean_volts for value in result.daq_hold_observations}
                means = {key: (source_means[self._bindings[key].source_index]
                               if self._bindings[key].kind == "source" else hold_means[self._bindings[key].hold_adc_id])
                         for key in self._active}
                self._readout.add_interval(start=start, end=end, mean_volts_by_input=means)
            phase = "causal publication"
            fresh = self._readout.new_observations_available_by(end)
            phase = "interval bookkeeping"
            self._evidence.append(IntervalEvidence(event_index, start, end, first))
            self._next += 1
            self._status = "physical_complete" if self._next == len(self._splits) else "running"
            if self._next == len(self._splits) or self._splits[self._next][0] != event_index:
                self._materialized_event, self._materialized_field = None, None
                self._materialized_noise = None
        except BaseException as exception:
            self._abort(exception, phase, returned, attempted_index)
        return fresh

    def observations_available_by(self, deadline):
        self._require_not_failed()
        deadline = _finite(deadline, "publication deadline")
        if deadline > self._publication_frontier:
            raise ValueError("The query cannot advance the reached publication frontier")
        return self._readout.observations_available_by(deadline)

    def observations_for_history(self, *, target_time, history_seconds, publication_deadline):
        self._require_not_failed()
        deadline = _finite(publication_deadline, "publication deadline")
        if deadline > self._publication_frontier:
            raise ValueError("The query cannot advance the reached publication frontier")
        return self._readout.observations_for_history(target_time=target_time, history_seconds=history_seconds,
                                                       publication_deadline=deadline)

    def finalize(self):
        """Finish declared digital publication only; leave physical_time unchanged."""
        self._require_not_failed()
        if self._status == "complete":
            return self._report
        if self._status != "physical_complete":
            raise ValueError("Every physical interval, including unsampled gaps, must finish first")
        try:
            report = self._readout.finalize(finished_at=self._horizon,
                                            acquisition_start=self._acquisition_window[0],
                                            acquisition_end=self._acquisition_window[1])
            report.update(physical_end_time=self._physical_time, final_publication_horizon=self._horizon,
                          split_intervals_executed=len(self._evidence), physical_intervals=len(self._events),
                          stored_plan_numeric_bytes=self._numeric_bytes, interval_evidence=self.evidence,
                          provider_field_digest_bytes=len(self._field_digests),
                          propagation_operator_products=self._operator_products,
                          maximum_local_scaled_partition_error=self._worst_local_error)
            if self._noise_provider is not None:
                report.update(physical_noise_applied=True, live_physical_noise_numeric_bytes=self._noise_bytes,
                              provider_digest_binds_physical_noise=True)
            self._report, self._publication_frontier, self._status = report, self._horizon, "complete"
        except BaseException as exception:
            self._abort(exception, "final publication", False)
        return report

    def run(self):
        """Execute only this fixed finite plan, then its declared publication horizon."""
        self._require_not_failed()
        while self._status not in ("physical_complete", "complete"):
            self.advance_next()
        return self.finalize()
