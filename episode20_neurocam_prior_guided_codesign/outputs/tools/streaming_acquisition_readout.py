"""Bounded, persistent readout for an explicitly declared finite acquisition.

This is a streaming alternative to AcquisitionReadout, not a DAQ recipe. One
instance survives every physical window. Uniform apertures, quantization,
signed digital arithmetic, validity and clipping use the eager implementation.
The caller performs a complete metadata-only pass before electrical mutation,
then verifies each replayed window before advancing its one physical array.
This module does not replace that whole-plan/replay integrity requirement.

Records have sequential r:0 / s:0 / o:0 identities in their respective streams.
Inclusive last-use windows close raw/sample dependencies explicitly, allowing
eviction without a full-record seen-ID set or value cache. Delayed outputs must
keep every dependency live. No failed operation may be retried on this instance.
Callbacks are trusted simulator/storage consumers, not inverse-reader release.
"""

from collections import Counter
from dataclasses import dataclass, fields, replace
import heapq
import math

from acquisition_readout import (AcquisitionReadout, Observation, QuantizerTransfer,
                                 RawConversion, ReturnedSample, _finite, _ids)
from observation_nuisance import ObservationNuisanceBinding, SavedObservationNuisance
from owned_quantizer_snapshot import snapshot_quantizer


@dataclass(frozen=True)
class ReadoutWindow:
    index: int
    start: float
    end: float
    conversions: tuple
    returned_samples: tuple
    observations: tuple
    raw_last_use_windows: tuple
    sample_last_use_windows: tuple
    nuisance_bindings: tuple = ()

    def __post_init__(self):
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("Window index must be a nonnegative integer")
        if _finite(self.start, "window start") >= _finite(self.end, "window end"):
            raise ValueError("Window must have positive duration")
        # Snapshot the small immutable plans, including quantizers shared by many
        # conversions. This does not retain the caller's mutable input sequences.
        quantizers = {}
        def quantizer(value):
            if value is None:
                return None
            if not isinstance(value, QuantizerTransfer):
                raise ValueError("Explicit quantizer required")
            key = id(value)
            if key not in quantizers:
                quantizers[key] = snapshot_quantizer(value)
            return quantizers[key]
        for name, cls in (("conversions", RawConversion), ("returned_samples", ReturnedSample),
                          ("observations", Observation)):
            result = []
            for plan in getattr(self, name):
                if not isinstance(plan, cls):
                    raise ValueError("Wrong streaming record type")
                if cls is RawConversion:
                    plan = replace(plan, quantizer=quantizer(plan.quantizer))
                elif cls is ReturnedSample:
                    plan = replace(plan, output_quantizer=quantizer(plan.output_quantizer))
                else:
                    plan = replace(plan)
                result.append(plan)
            object.__setattr__(self, name, tuple(result))
        for name in ("raw_last_use_windows", "sample_last_use_windows"):
            object.__setattr__(self, name, tuple(tuple(item) for item in getattr(self, name)))
        bindings = []
        for binding in self.nuisance_bindings:
            if not isinstance(binding, ObservationNuisanceBinding):
                raise ValueError("Explicit immutable observation nuisance bindings required")
            bindings.append(replace(binding))
        object.__setattr__(self, "nuisance_bindings", tuple(bindings))


@dataclass(frozen=True)
class StreamingReadoutLimits:
    max_conversions: int
    max_returned_samples: int
    max_observations: int
    max_interval_updates: int
    max_windows: int
    max_live_conversions: int
    max_live_returned_samples: int
    max_live_observations: int
    max_stored_plan_numeric_bytes: int
    max_accounting_categories: int

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("All streaming resource limits must be explicit positive integers")


def _terminal_on_error(method):
    def guarded(self, *args, **kwargs):
        if self.failed or self._finalized:
            raise ValueError("Streaming readout is terminal; no retry or replay")
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            self.failed = True
            raise
    return guarded


class StreamingAcquisitionReadout:
    """One persistent readout with bounded live dependencies and output callbacks.

    Declare a raw aperture in its start window. Declare digital records before
    their availability (strictly after that window's start). All dependencies
    must already be declared; future references and reuse of retired IDs refuse.
    A raw dependency lasts at least as long as each returned sample using it.
    Every contributor counts, including a zero-weight contributor. Readout
    windows partition the explicit acquisition window exactly; they never reset
    integrals or digital state. The final horizon only drains acquired values.

    validate_only uses identical coverage/dependency/eviction logic without any
    invented means, quantization, or output values. It is not an executed result.
    Numeric storage accounting bounds retained plans, not Python peak memory.
    Caller-controlled record/string sizes and the enclosing job still matter.
    """

    def __init__(self, *, input_ids, acquisition_start, acquisition_end,
                 final_publication_horizon, limits, on_raw, on_observation,
                 validate_only=False, observation_nuisance=None):
        if not isinstance(limits, StreamingReadoutLimits) or not isinstance(validate_only, bool):
            raise ValueError("Explicit streaming limits and validation mode required")
        self.input_ids = _ids(input_ids, "input IDs")
        self.acquisition_start = _finite(acquisition_start, "acquisition start")
        self.acquisition_end = _finite(acquisition_end, "acquisition end")
        self.final_publication_horizon = _finite(final_publication_horizon, "final horizon")
        if not self.acquisition_start < self.acquisition_end <= self.final_publication_horizon:
            raise ValueError("Explicit acquisition and publication horizons are inconsistent")
        if not callable(on_raw) or not callable(on_observation):
            raise ValueError("Bounded output callbacks are required, even in validation mode")
        if observation_nuisance is not None and not isinstance(observation_nuisance, SavedObservationNuisance):
            raise ValueError("An explicitly bound saved-world nuisance adapter is required")
        self.limits, self.validate_only = limits, validate_only
        self._observation_nuisance = observation_nuisance
        self._nuisance_bindings, self._nuisance_applied = {}, 0
        self._on_raw, self._on_observation = on_raw, on_observation
        self._raw, self._samples, self._observations = {}, {}, {}
        self._raw_last, self._sample_last = {}, {}
        self._raw_values, self._sample_values = {}, {}
        self._integral, self._covered_to = {}, {}
        self._by_input = {key: [] for key in self.input_ids}
        self._cursor = dict.fromkeys(self.input_ids, 0)
        self._active = dict.fromkeys(self.input_ids, ())
        self._last_end = dict.fromkeys(self.input_ids, self.acquisition_start)
        self._raw_due, self._sample_due, self._observation_due = [], [], []
        self._used_raw = set()
        self._counts = dict(conversions=0, returned_samples=0, observations=0)
        self._completed_raw = self._completed_samples = self._published = 0
        self._raw_delivered = self._observations_delivered = 0
        self._uncertain_callback = None
        self._pixels = self._valid_pixels = self._unused = 0
        self._by_role, self._by_adc = Counter(), Counter()
        self._known_bits, self._unknown_bits = 0, False
        self._updates, self._max_updates = 0, limits.max_interval_updates
        self._window = None
        self._physical_complete = self._finalized = self.failed = False
        self.windows_closed = 0
        self.physical_time = self.acquisition_start
        self._publication_time = self.acquisition_start
        self._peak = dict(conversions=0, returned_samples=0, observations=0,
                          stored_plan_numeric_bytes=0)

    @property
    def live_counts(self):
        return dict(conversions=len(self._raw), returned_samples=len(self._samples),
                    observations=len(self._observations),
                    stored_plan_numeric_bytes=self._numeric_bytes())

    def accounting(self):
        """Read-only prefix evidence, available after success or any failure.

        Computation counters commit before their output callback. A callback
        that raises may already have buffered or written the current record;
        it is neither declared undelivered nor counted as a successful return.
        Successful callback return is not durable-storage confirmation. The
        sink separately owns buffered/persisted counts and its failure state.
        """
        return dict(metadata_only=self.validate_only, failed=self.failed,
            complete=self._finalized, physical_time=self.physical_time,
            publication_frontier=self._publication_time, windows_closed=self.windows_closed,
            admitted_records=dict(self._counts), completed_conversions=self._completed_raw,
            completed_returned_samples=self._completed_samples,
            computed_observations=self._published,
            nuisance_observations_processed=None if self.validate_only else self._nuisance_applied,
            raw_callbacks_returned=self._raw_delivered,
            observation_callbacks_returned=self._observations_delivered,
            callback_with_uncertain_effect=self._uncertain_callback,
            callback_return_confirms_persistence=False,
            retired_conversions_without_returned_sample=self._unused,
            admitted_conversions_by_role=dict(self._by_role),
            admitted_conversions_by_adc=dict(self._by_adc),
            admitted_known_output_bit_subtotal=self._known_bits,
            admitted_output_bits_unknown=self._unknown_bits,
            interval_updates=self._updates, live_counts=self.live_counts,
            peak_live_counts=dict(self._peak))

    def _deliver(self, kind, value):
        callback = self._on_raw if kind == "raw" else self._on_observation
        try:
            callback(value)
        except BaseException:
            key = value.conversion_id if kind == "raw" else value.observation_id
            self._uncertain_callback = (kind, key)
            raise
        if kind == "raw":
            self._raw_delivered += 1
        else:
            self._observations_delivered += 1

    def _numeric_bytes(self):
        # Include pending/cached scalar state, digital input weights/indices,
        # explicit aperture support and each distinct live quantizer once.
        size = (112 * len(self._raw) + 64 * len(self._samples) + 64 * len(self._observations)
                + 24 * len(self._nuisance_bindings))
        quantizers = {}
        for plan in self._raw.values():
            quantizers[id(plan.quantizer)] = plan.quantizer
        for plan in self._samples.values():
            size += 16 * len(plan.weights)
            if plan.output_quantizer is not None:
                quantizers[id(plan.output_quantizer)] = plan.output_quantizer
        size += sum(16 * len(plan.weights) for plan in self._observations.values())
        return size + sum(16 + 8 * (len(q.thresholds_volts) + len(q.code_volts))
                          for q in quantizers.values())

    @staticmethod
    def _last_use(entries, plans, key, index, maximum):
        result = {}
        for entry in entries:
            if len(entry) != 2:
                raise ValueError("Each last-use declaration needs an ID and inclusive window")
            name, last = entry
            if (name in result or isinstance(last, bool) or not isinstance(last, int)
                    or not index <= last < maximum):
                raise ValueError("Duplicate or invalid last-use window")
            result[name] = last
        if set(result) != {getattr(plan, key) for plan in plans}:
            raise ValueError("Last-use declarations must cover exactly the newly declared inputs")
        return result

    @_terminal_on_error
    def accept_window(self, window):
        if (not isinstance(window, ReadoutWindow) or self._window is not None
                or self._physical_complete or window.index != self.windows_closed
                or window.index >= self.limits.max_windows or window.start != self.physical_time
                or window.end > self.acquisition_end):
            raise ValueError("Windows must be contiguous, ordered and inside the acquisition")
        # Reconstruct even an already frozen dataclass so runtime state owns its
        # plans; the executor independently binds the whole replayed window.
        window = replace(window)
        groups = (("conversions", "conversion_id", "r", window.conversions, self._raw),
                  ("returned_samples", "sample_id", "s", window.returned_samples, self._samples),
                  ("observations", "observation_id", "o", window.observations, self._observations))
        for name, key, prefix, plans, live in groups:
            if (self._counts[name] + len(plans) > getattr(self.limits, "max_" + name)
                    or len(live) + len(plans) > getattr(self.limits, "max_live_" + name)):
                raise ValueError("Total or live streaming record limit exceeded")
            for offset, plan in enumerate(plans):
                if getattr(plan, key) != f"{prefix}:{self._counts[name] + offset}":
                    raise ValueError("Streaming record IDs must be unique sequential canonical IDs")
        raw_last = self._last_use(window.raw_last_use_windows, window.conversions,
                                 "conversion_id", window.index, self.limits.max_windows)
        sample_last = self._last_use(window.sample_last_use_windows, window.returned_samples,
                                    "sample_id", window.index, self.limits.max_windows)
        raw = dict(self._raw)
        samples = dict(self._samples)
        for plan in window.conversions:
            if (plan.input_id not in self._by_input or not window.start <= plan.aperture_start < window.end
                    or plan.completed_at > self.acquisition_end):
                raise ValueError("Raw conversion must be declared in its aperture-start window")
            raw[plan.conversion_id] = plan
        all_raw_last = dict(self._raw_last, **raw_last)
        for plan in window.returned_samples:
            if (any(key not in raw for key in plan.conversion_ids)
                    or not window.start < plan.available_at <= self.final_publication_horizon
                    or plan.available_at < max(raw[key].completed_at for key in plan.conversion_ids)
                    or any(all_raw_last[key] < sample_last[plan.sample_id] for key in plan.conversion_ids)):
                raise ValueError("Returned sample has unknown/retired inputs, late declaration or short dependency lifetime")
            samples[plan.sample_id] = plan
        for plan in window.observations:
            if (any(key not in samples for key in plan.sample_ids)
                    or not window.start < plan.available_at <= self.final_publication_horizon
                    or plan.available_at < max(samples[key].available_at for key in plan.sample_ids)):
                raise ValueError("Observation has unknown/retired inputs or invalid publication/declaration time")
        nuisance_bindings = {binding.observation_id: binding for binding in window.nuisance_bindings}
        if len(nuisance_bindings) != len(window.nuisance_bindings):
            raise ValueError("Duplicate observation nuisance binding")
        if self._observation_nuisance is None:
            if nuisance_bindings:
                raise ValueError("Nuisance bindings require an explicitly enabled saved-world adapter")
        else:
            pixel_ids = {plan.observation_id for plan in window.observations if plan.role == "pixel"}
            if set(nuisance_bindings) != pixel_ids:
                raise ValueError("Nuisance bindings must cover exactly every newly declared pixel observation")
            for plan in window.observations:
                if plan.role == "pixel":
                    raw_ids = {key for sample in plan.sample_ids for key in samples[sample].conversion_ids}
                    support = tuple((key, raw[key].aperture_start, raw[key].aperture_end)
                                    for key in sorted(raw_ids))
                    # Metadata-only in both passes: never inspect nuisance values
                    # to admit a plan or to decide whether an observation exists.
                    self._observation_nuisance.validate_binding(
                        nuisance_bindings[plan.observation_id], plan, support)
        roles = set(self._by_role).union(plan.role for plan in window.conversions)
        adcs = set(self._by_adc).union(plan.adc_id for plan in window.conversions)
        if len(roles) + len(adcs) > self.limits.max_accounting_categories:
            raise ValueError("Cumulative role/ADC accounting exceeds its explicit category bound")
        self._raw, self._samples = raw, samples
        self._observations.update((p.observation_id, p) for p in window.observations)
        self._nuisance_bindings.update(nuisance_bindings)
        self._raw_last.update(raw_last)
        self._sample_last.update(sample_last)
        if self._numeric_bytes() > self.limits.max_stored_plan_numeric_bytes:
            raise ValueError("Live numeric plan/dependency storage exceeds its explicit bound")
        self._window = window.index, window.start, window.end
        for name, _, _, plans, _ in groups:
            self._counts[name] += len(plans)
        for plan in window.conversions:
            key = plan.conversion_id
            self._by_input[plan.input_id].append(key)
            self._covered_to[key], self._integral[key] = plan.aperture_start, 0.0
            heapq.heappush(self._raw_due, (plan.completed_at, key))
            self._by_role[plan.role] += 1
            self._by_adc[plan.adc_id] += 1
        for key in self._by_input:
            self._by_input[key].sort(key=lambda name: self._raw[name].aperture_start)
        for plan in window.returned_samples:
            self._used_raw.update(plan.conversion_ids)
            heapq.heappush(self._sample_due, (plan.available_at, plan.sample_id))
        for plan in window.observations:
            heapq.heappush(self._observation_due, (plan.available_at, plan.observation_id))
            self._pixels += plan.role == "pixel"
            for bits in (plan.payload_bits, plan.overhead_bits):
                if bits is None:
                    self._unknown_bits = True
                else:
                    self._known_bits += bits
        for key, value in self.live_counts.items():
            self._peak[key] = max(self._peak[key], value)

    @property
    def required_aperture_boundaries(self):
        return tuple(sorted({value for plan in self._raw.values()
                             for value in (plan.aperture_start, plan.aperture_end)
                             if value >= self.physical_time}))

    def active_input_ids(self, start, end):
        start, end = _finite(start, "interval start"), _finite(end, "interval end")
        if self._window is None or start != self.physical_time or not start < end <= self._window[2]:
            raise ValueError("Interval must extend the current physical frontier within its window")
        return tuple(key for key, ids in self._by_input.items()
                     if self._active[key] or (self._cursor[key] < len(ids)
                        and self._raw[ids[self._cursor[key]]].aperture_start < end))

    def _coverage(self, start, end, input_ids, means):
        expected = self.active_input_ids(start, end)
        supplied = tuple(input_ids)
        if len(set(supplied)) != len(supplied) or set(supplied) != set(expected):
            raise ValueError("Supply exactly every active aperture input, without omissions or extras")
        if self._updates >= self._max_updates:
            raise ValueError("Cumulative interval update limit exceeded")
        changes, pending = [], []
        for key in supplied:
            cursor, ids = self._cursor[key], self._by_input[key]
            active = list(self._active[key])
            while cursor < len(ids) and self._raw[ids[cursor]].aperture_start < end:
                active.append(ids[cursor])
                cursor += 1
            retained = []
            for name in active:
                plan = self._raw[name]
                if start < plan.aperture_start or end > plan.aperture_end or start != self._covered_to[name]:
                    raise ValueError("Aperture coverage gap/overlap or unsplit aperture boundary")
                integral = self._integral[name]
                if means is not None:
                    integral += _finite(means[key], "analog interval mean") * (end - start)
                    if not math.isfinite(integral):
                        raise ValueError("Aperture integral is not representable")
                changes.append((name, integral))
                if end < plan.aperture_end:
                    retained.append(name)
            pending.append((key, cursor, tuple(retained)))
        for name, integral in changes:
            self._integral[name], self._covered_to[name] = integral, end
        for key, cursor, active in pending:
            self._cursor[key], self._active[key], self._last_end[key] = cursor, active, end
        self.physical_time = end
        self._updates += 1

    @_terminal_on_error
    def add_interval(self, *, start, end, mean_volts_by_input):
        if self.validate_only:
            raise ValueError("Metadata-only validation cannot accept invented analog values")
        self._coverage(start, end, mean_volts_by_input, mean_volts_by_input)

    @_terminal_on_error
    def validate_interval(self, *, start, end, input_ids):
        if not self.validate_only:
            raise ValueError("Executed readout requires actual analog means")
        self._coverage(start, end, input_ids, None)

    # Reuse the already checked exact arithmetic, including zero-weight input
    # validity, signed combinations, requantization and aperture provenance.
    _conversion = AcquisitionReadout._conversion
    _returned = AcquisitionReadout._returned
    _apertures = AcquisitionReadout._apertures
    _publish = AcquisitionReadout._publish

    @_terminal_on_error
    def advance_publication(self, deadline):
        deadline = _finite(deadline, "publication deadline")
        maximum = self.final_publication_horizon if self._physical_complete else self.physical_time
        if not self._publication_time <= deadline <= maximum:
            raise ValueError("Publication cannot move backward or expose unacquired future values")
        while self._raw_due and self._raw_due[0][0] <= deadline:
            _, key = self._raw_due[0]
            if self._covered_to[key] != self._raw[key].aperture_end:
                raise ValueError("Due conversion aperture is incomplete")
            if self.validate_only:
                self._raw_values[key] = None
            else:
                value = self._conversion(key)
            heapq.heappop(self._raw_due)
            self._completed_raw += 1
            if not self.validate_only:
                self._deliver("raw", value)
        while self._sample_due and self._sample_due[0][0] <= deadline:
            _, key = self._sample_due[0]
            if any(name not in self._raw_values for name in self._samples[key].conversion_ids):
                raise ValueError("Due returned sample has incomplete conversions")
            if self.validate_only:
                self._sample_values[key] = None
            else:
                self._returned(key)
            heapq.heappop(self._sample_due)
            self._completed_samples += 1
        while self._observation_due and self._observation_due[0][0] <= deadline:
            _, key = self._observation_due[0]
            plan = self._observations[key]
            if any(name not in self._sample_values for name in plan.sample_ids):
                raise ValueError("Due observation has incomplete returned samples")
            if not self.validate_only:
                value, = self._publish((plan,))
                if plan.role == "pixel" and self._observation_nuisance is not None:
                    value = self._observation_nuisance.apply(value, self._nuisance_bindings[key])
                    self._nuisance_applied += 1
                self._valid_pixels += value.role == "pixel" and value.valid
            heapq.heappop(self._observation_due)
            del self._observations[key]
            self._nuisance_bindings.pop(key, None)
            self._published += 1
            if not self.validate_only:
                self._deliver("observation", value)
        self._publication_time = deadline

    def _evict(self, index):
        raw = {key for key, last in self._raw_last.items() if last <= index}
        samples = {key for key, last in self._sample_last.items() if last <= index}
        if (any(key not in self._raw_values for key in raw)
                or any(key not in self._sample_values for key in samples)
                or any(set(plan.sample_ids) & samples for plan in self._observations.values())
                or any(set(plan.conversion_ids) & raw for key, plan in self._samples.items() if key not in samples)):
            raise ValueError("Last-use closure would evict incomplete or still-needed dependencies")
        for key in samples:
            del self._samples[key], self._sample_values[key], self._sample_last[key]
        for key in raw:
            self._unused += key not in self._used_raw
            self._used_raw.discard(key)
            del self._raw[key], self._raw_values[key], self._raw_last[key]
            del self._integral[key], self._covered_to[key]
        # Completed queue prefixes are not retained across windows. Active
        # apertures survive separately, including ones crossing this seam.
        for key, ids in self._by_input.items():
            self._by_input[key] = ids[self._cursor[key]:]
            self._cursor[key] = 0

    @_terminal_on_error
    def close_window(self, index):
        if (self._window is None or index != self._window[0]
                or self.physical_time != self._window[2]):
            raise ValueError("Only the fully covered current window may close")
        self.advance_publication(self.physical_time)
        self._physical_complete = self.physical_time == self.acquisition_end
        if not self._physical_complete:
            self._evict(index)
        self._window = None
        self.windows_closed += 1

    @_terminal_on_error
    def finalize(self):
        if self._window is not None or not self._physical_complete or not self._counts["conversions"]:
            raise ValueError("Finalization requires the complete nonempty acquisition")
        if any(last >= self.windows_closed for last in (*self._raw_last.values(), *self._sample_last.values())):
            raise ValueError("Last-use declarations extend beyond the complete finite plan")
        self.advance_publication(self.final_publication_horizon)
        self._evict(self.windows_closed - 1)
        if self._raw or self._samples or self._observations:
            raise ValueError("Unresolved dependencies remain at final publication")
        duration = self.acquisition_end - self.acquisition_start
        bits = None if self._unknown_bits else self._known_bits
        result = dict(actual_internal_conversions=self._completed_raw,
            hardware_returned_samples=self._completed_samples, published_observations=self._published,
            retained_pixel_observations=self._pixels,
            valid_retained_pixel_observations=None if self.validate_only else self._valid_pixels,
            conversions_by_role=dict(self._by_role), conversions_by_adc=dict(self._by_adc),
            conversions_without_returned_sample=self._unused,
            raw_conversion_rate_hz=self._completed_raw / duration,
            retained_pixel_rate_hz=self._pixels / duration,
            hardware_returned_sample_rate_hz=self._completed_samples / duration,
            output_bits=bits, known_output_bit_subtotal=self._known_bits,
            output_bitrate_bps=None if bits is None else bits / duration,
            hardware_transport_bits=None, modeled_power_watt=None,
            modeled_conversion_energy_joule=None, actual_DAQ_configuration_qualified=False,
            metadata_only=self.validate_only, windows_closed=self.windows_closed,
            interval_updates=self._updates, peak_live_counts=dict(self._peak),
            acquisition_start=self.acquisition_start, acquisition_end=self.acquisition_end,
            final_publication_horizon=self.final_publication_horizon)
        result["nuisance_observations_processed"] = None if self.validate_only else self._nuisance_applied
        result["observation_nuisance"] = (None if self._observation_nuisance is None
                                         else dict(self._observation_nuisance.metadata))
        self._finalized = True
        return result
