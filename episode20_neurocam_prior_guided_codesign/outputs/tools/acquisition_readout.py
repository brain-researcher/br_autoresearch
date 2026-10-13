"""Explicit finite acquisition plan; no inferred NeuroCam/NI configuration.

The caller advances its persistent analog model on the union of physical row,
input and aperture boundaries and supplies interval means at the ADC inputs.
This component integrates complete uniform rectangular apertures, quantizes
each physical raw conversion, then applies explicit returned-sample and
observation arithmetic. The rectangular aperture is this component's declared
model, not a measured DAQ default. A nonuniform physical aperture or decimation
filter needs an explicit upstream model or a separately specified readout.
Only published observations are exposed by the causal query interface.

It supplies no analog conditioning, multiplexing ghosting, sample-and-hold
loading, noise, actual clock/ADC map, missing calibration, or hardware legality
certificate. Those must be represented before these input means are supplied.
No primary-band prefilter, square-root noise reduction or bit-cost default is
assumed. This finite plan is not a full-record throughput/memory benchmark.
"""

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
import math
from numbers import Real


def _name(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _ids(values, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an explicit sequence")
    values = tuple(values)
    for value in values:
        _name(value, name)
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be nonempty and unique; use weights for repeated terms")
    return values


def _combination(instance, id_field):
    ids = _ids(getattr(instance, id_field), id_field)
    weights = tuple(_finite(value, "digital weight") for value in instance.weights)
    if len(ids) != len(weights) or not any(value != 0 for value in weights):
        raise ValueError("Supply one explicit weight per input, with nonzero support")
    if _finite(instance.divisor, "digital divisor") <= 0:
        raise ValueError("Digital divisor must be positive; no normalization is inferred")
    object.__setattr__(instance, id_field, ids)
    object.__setattr__(instance, "weights", weights)
    for name in ("assigned_timestamp", "available_at"):
        _finite(getattr(instance, name), name)
    if instance.assigned_timestamp > instance.available_at:
        raise ValueError("A sample timestamp cannot be later than its publication")


@dataclass(frozen=True)
class QuantizerTransfer:
    """Explicit monotone ideal/calibrated transfer, not a chosen ADC model.

    Threshold ties select the upper code. Outside the supplied input range,
    saturate to the end code and report clipping. Code voltages are explicit
    decoded representatives; gain/range/precision are never filled from defaults.
    """
    input_min_volts: float
    input_max_volts: float
    thresholds_volts: tuple
    code_volts: tuple

    def __post_init__(self):
        low, high = (_finite(self.input_min_volts, "input minimum"),
                     _finite(self.input_max_volts, "input maximum"))
        thresholds = tuple(_finite(v, "quantizer threshold") for v in self.thresholds_volts)
        levels = tuple(_finite(v, "decoded code voltage") for v in self.code_volts)
        if (low >= high or len(levels) < 2 or len(thresholds) != len(levels) - 1
                or any(a >= b for a, b in zip(thresholds, thresholds[1:]))
                or any(a >= b for a, b in zip(levels, levels[1:]))
                or not low < thresholds[0] <= thresholds[-1] < high
                or levels[0] < low or levels[-1] > high):
            raise ValueError("Supply ordered thresholds/codes within an explicit increasing input range")
        object.__setattr__(self, "thresholds_volts", thresholds)
        object.__setattr__(self, "code_volts", levels)

    def convert(self, volts):
        value = _finite(volts, "conversion input")
        index = bisect_right(self.thresholds_volts, value)
        return index, self.code_volts[index], value < self.input_min_volts or value > self.input_max_volts


@dataclass(frozen=True)
class RawConversion:
    """One declared conversion with a uniform rectangular input aperture."""

    conversion_id: str
    adc_id: str
    input_id: str
    role: str
    aperture_start: float
    aperture_end: float
    completed_at: float
    assigned_timestamp: float
    quantizer: QuantizerTransfer
    valid: bool

    def __post_init__(self):
        for name in ("conversion_id", "adc_id", "input_id", "role"):
            _name(getattr(self, name), name)
        for name in ("aperture_start", "aperture_end", "completed_at", "assigned_timestamp"):
            _finite(getattr(self, name), name)
        if (self.aperture_start >= self.aperture_end or self.completed_at < self.aperture_end
                or self.assigned_timestamp > self.completed_at):
            raise ValueError("Explicit finite aperture must finish before conversion completion")
        if not isinstance(self.quantizer, QuantizerTransfer) or not isinstance(self.valid, bool):
            raise ValueError("Every conversion needs an explicit quantizer and validity flag")


@dataclass(frozen=True)
class ReturnedSample:
    sample_id: str
    conversion_ids: tuple
    weights: tuple
    divisor: float
    assigned_timestamp: float
    available_at: float
    output_quantizer: QuantizerTransfer | None

    def __post_init__(self):
        _name(self.sample_id, "sample_id")
        _combination(self, "conversion_ids")
        if self.output_quantizer is not None and not isinstance(self.output_quantizer, QuantizerTransfer):
            raise ValueError("Specify a returned-word quantizer or explicit None for no requantization")


@dataclass(frozen=True)
class Observation:
    observation_id: str
    site_id: str
    role: str
    sample_ids: tuple
    weights: tuple
    divisor: float
    assigned_timestamp: float
    available_at: float
    payload_bits: int | None
    overhead_bits: int | None

    def __post_init__(self):
        for name in ("observation_id", "site_id", "role"):
            _name(getattr(self, name), name)
        _combination(self, "sample_ids")
        for name in ("payload_bits", "overhead_bits"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError("Bit costs must be explicit nonnegative integers or unknown None")


@dataclass(frozen=True)
class RawValue:
    conversion_id: str
    aperture_mean_volts: float
    code_index: int
    decoded_volts: float
    valid: bool
    clipped: bool


@dataclass(frozen=True)
class PublishedValue:
    observation_id: str
    site_id: str
    role: str
    value_volts: float | None
    valid: bool
    clipped: bool
    assigned_timestamp: float
    available_at: float
    contributing_apertures: tuple


class AcquisitionReadout:
    """Bounded plan with independently counted conversions, returns and pixels.

    A returned sample is an explicit linear combination divided by its supplied
    divisor, after raw quantization, then optionally requantized. Observation
    arithmetic occurs after that step. Weights may be signed for an explicitly
    declared digital filter; this does not authorize analog mixing or a new
    hardware recipe. Every listed input contributes to timing/validity support.
    An invalid contributor makes the published observation value absent (None);
    its timing, clipping and work still count. No inputs are dropped/reweighted.

    Interval boundaries must exactly use the caller's shared timeline; this
    component never snaps clocks or prorates a mean over only part of an interval.
    Analog state advancement during unsampled/discarded time remains the caller's
    responsibility. Split a long acquisition into justified bounded plans while
    retaining the underlying analog/conditioning state; this object never resets it.
    """

    def __init__(self, *, conversions, returned_samples, observations,
                 max_conversions, max_returned_samples, max_observations, max_interval_updates):
        for value in (max_conversions, max_returned_samples, max_observations, max_interval_updates):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("All finite plan/work limits must be explicit positive integers")
        def inventory(values, cls, key, maximum):
            result = {}
            for value in values:
                if not isinstance(value, cls) or getattr(value, key) in result or len(result) >= maximum:
                    raise ValueError("Wrong, duplicate or excessive planned acquisition item")
                result[getattr(value, key)] = value
            return result
        self._raw = inventory(conversions, RawConversion, "conversion_id", max_conversions)
        self._samples = inventory(returned_samples, ReturnedSample, "sample_id", max_returned_samples)
        self._observations = inventory(observations, Observation, "observation_id", max_observations)
        if not self._raw:
            raise ValueError("At least one actual conversion is required")
        for sample in self._samples.values():
            if (any(i not in self._raw for i in sample.conversion_ids)
                    or sample.available_at < max(self._raw[i].completed_at for i in sample.conversion_ids)):
                raise ValueError("Returned sample requires known completed conversions before availability")
        for observation in self._observations.values():
            if (any(i not in self._samples for i in observation.sample_ids)
                    or observation.available_at < max(self._samples[i].available_at for i in observation.sample_ids)):
                raise ValueError("Observation cannot publish before every contributing returned sample")
        self._by_input = {}
        for conversion in self._raw.values():
            self._by_input.setdefault(conversion.input_id, []).append(conversion.conversion_id)
        for ids in self._by_input.values():
            ids.sort(key=lambda i: self._raw[i].aperture_start)
        self._cursor = {name: 0 for name in self._by_input}
        self._active = {name: () for name in self._by_input}
        self._last_end = {name: -math.inf for name in self._by_input}
        self._covered_to = {key: value.aperture_start for key, value in self._raw.items()}
        self._integral = dict.fromkeys(self._raw, 0.0)
        self._raw_values = {}
        self._sample_values = {}
        # One delivery cursor over references to the existing immutable plans.
        # Do not retain another PublishedValue cache or scan the complete
        # observation prefix at every analog interval. Cumulative/history
        # queries remain independent and do not consume incremental delivery.
        self._publication_order = tuple(sorted(self._observations.values(),
                                               key=lambda plan: (plan.available_at, plan.observation_id)))
        self._publication_cursor = 0
        self._incremental_deadline = -math.inf
        self._updates = 0
        self._max_updates = max_interval_updates

    @property
    def required_aperture_boundaries(self):
        """Exact supplied endpoints to merge with the caller's physical timeline."""
        return tuple(sorted({t for c in self._raw.values() for t in (c.aperture_start, c.aperture_end)}))

    def add_interval(self, *, start, end, mean_volts_by_input):
        """Accept exact analog interval means; validate all contributions first."""
        start, end = _finite(start, "interval start"), _finite(end, "interval end")
        if start >= end or not mean_volts_by_input or self._updates >= self._max_updates:
            raise ValueError("Supply a positive interval within the finite update limit")
        pending, changes = [], []
        for name, mean in mean_volts_by_input.items():
            value = _finite(mean, "analog interval mean")
            if name not in self._by_input or start < self._last_end[name]:
                raise ValueError("Unknown input or overlapping/out-of-order analog interval")
            ids, cursor = self._by_input[name], self._cursor[name]
            active = list(self._active[name])
            while cursor < len(ids) and self._raw[ids[cursor]].aperture_start < end:
                active.append(ids[cursor])
                cursor += 1
            retained = []
            for key in active:
                conversion = self._raw[key]
                if (start < conversion.aperture_start or end > conversion.aperture_end
                        or start != self._covered_to[key]):
                    raise ValueError("Aperture coverage gap/overlap or partial-interval mean; split the physical timeline")
                integral = self._integral[key] + value * (end - start)
                if not math.isfinite(integral):
                    raise ValueError("Aperture integral is not representable")
                changes.append((key, integral))
                if end < conversion.aperture_end:
                    retained.append(key)
            pending.append((name, cursor, tuple(retained)))
        for key, integral in changes:
            self._integral[key], self._covered_to[key] = integral, end
        for name, cursor, active in pending:
            self._cursor[name], self._active[name], self._last_end[name] = cursor, active, end
        self._updates += 1

    def _conversion(self, key):
        if key not in self._raw_values:
            conversion = self._raw[key]
            if self._covered_to[key] != conversion.aperture_end:
                raise ValueError("Conversion aperture is incomplete")
            mean = self._integral[key] / (conversion.aperture_end - conversion.aperture_start)
            code, volts, clipped = conversion.quantizer.convert(mean)
            self._raw_values[key] = RawValue(key, mean, code, volts, conversion.valid, clipped)
        return self._raw_values[key]

    def _returned(self, key):
        if key not in self._sample_values:
            plan = self._samples[key]
            inputs = [self._conversion(i) for i in plan.conversion_ids]
            value = math.fsum(w * item.decoded_volts for w, item in zip(plan.weights, inputs)) / plan.divisor
            _finite(value, "returned sample")
            clipped = any(item.clipped for item in inputs)
            if plan.output_quantizer is not None:
                _, value, extra_clip = plan.output_quantizer.convert(value)
                clipped = clipped or extra_clip
            self._sample_values[key] = (value, all(item.valid for item in inputs), clipped)
        return self._sample_values[key]

    def _apertures(self, plan):
        raw_ids = {i for sample in plan.sample_ids for i in self._samples[sample].conversion_ids}
        return tuple((i, self._raw[i].aperture_start, self._raw[i].aperture_end)
                     for i in sorted(raw_ids))

    def _publish(self, plans):
        result = []
        for plan in plans:
            inputs = [self._returned(i) for i in plan.sample_ids]
            valid = all(item[1] for item in inputs)
            value = None
            if valid:
                value = math.fsum(w * item[0] for w, item in zip(plan.weights, inputs)) / plan.divisor
                _finite(value, "published observation")
            result.append(PublishedValue(plan.observation_id, plan.site_id, plan.role, value,
                                          valid, any(item[2] for item in inputs),
                                          plan.assigned_timestamp, plan.available_at, self._apertures(plan)))
        return tuple(sorted(result, key=lambda item: (item.available_at, item.observation_id)))

    def observations_available_by(self, deadline):
        """Only publication time grants visibility; an earlier timestamp does not."""
        deadline = _finite(deadline, "publication deadline")
        return self._publish(plan for plan in self._observations.values() if plan.available_at <= deadline)

    def new_observations_available_by(self, deadline):
        """Deliver only newly due values to one monotone incremental consumer.

        Repeated deadlines return no previously delivered values; decreasing
        deadlines refuse. Cumulative/history/final queries do not move this
        cursor. No cursor or deadline commits until the complete due batch
        publishes, so an incomplete aperture cannot silently consume a prefix.
        Empty intervals inspect only the next due plan. New batches use a
        binary search and materialize only their own values, never the earlier
        prefix. Raw/sample storage and final cumulative reports remain eager;
        this method is not a full-record streaming-memory implementation.
        """
        deadline = _finite(deadline, "incremental publication deadline")
        if deadline < self._incremental_deadline:
            raise ValueError("Incremental publication deadlines must be nondecreasing")
        cursor, order = self._publication_cursor, self._publication_order
        if cursor == len(order) or order[cursor].available_at > deadline:
            self._incremental_deadline = deadline
            return ()
        stop = bisect_right(order, deadline, lo=cursor, key=lambda plan: plan.available_at)
        published = self._publish(order[cursor:stop])
        self._publication_cursor, self._incremental_deadline = stop, deadline
        return published

    def observations_for_history(self, *, target_time, history_seconds, publication_deadline):
        """Require whole apertures inside [target-H,target], plus publication.

        A delayed reporting deadline cannot admit an aperture after the target.
        This filter does not implement inverse fitting, state replay, capacity
        limits or isolation from the analog world's hidden parameters.
        """
        target = _finite(target_time, "target_time")
        history = _finite(history_seconds, "history_seconds")
        deadline = _finite(publication_deadline, "publication deadline")
        if history <= 0:
            raise ValueError("History must be explicitly positive")
        # Inspect the plan's public timing only. Out-of-window raw values must
        # never be materialized, even if the reporting deadline is delayed.
        selected = (plan for plan in self._observations.values()
                    if plan.available_at <= deadline
                    and all(start >= target - history and end <= target
                            for _, start, end in self._apertures(plan)))
        return self._publish(selected)

    def finalize(self, *, finished_at, acquisition_start, acquisition_end):
        """Final complete evidence only after every conversion and publication.

        The explicit acquisition window sets count/rate denominators; delayed
        publication may occur later. Unknown bit/power costs remain unknown.
        Raw codes are exposed here only after the full plan's publication horizon.
        """
        finished_at = _finite(finished_at, "finished_at")
        start, end = _finite(acquisition_start, "acquisition_start"), _finite(acquisition_end, "acquisition_end")
        if (start >= end or any(c.aperture_start < start or c.completed_at > end for c in self._raw.values())
                or finished_at < max([end] + [s.available_at for s in self._samples.values()]
                                     + [o.available_at for o in self._observations.values()])):
            raise ValueError("Complete acquisition/publication must fit the explicit final horizon")
        raw = tuple(self._conversion(key) for key in self._raw)
        for key in self._samples:
            self._returned(key)
        published = self.observations_available_by(finished_at)
        costs = [value for o in self._observations.values() for value in (o.payload_bits, o.overhead_bits)]
        bits = None if any(value is None for value in costs) else sum(costs)
        duration = end - start
        used_raw = {key for sample in self._samples.values() for key in sample.conversion_ids}
        pixel_outputs = sum(o.role == "pixel" for o in self._observations.values())
        return dict(raw_values=raw, observations=published,
                    actual_internal_conversions=len(raw), hardware_returned_samples=len(self._samples),
                    retained_pixel_observations=pixel_outputs,
                    valid_retained_pixel_observations=sum(v.role == "pixel" and v.valid for v in published),
                    conversions_by_role=dict(Counter(c.role for c in self._raw.values())),
                    conversions_by_adc=dict(Counter(c.adc_id for c in self._raw.values())),
                    conversions_without_returned_sample=len(set(self._raw) - used_raw),
                    raw_conversion_rate_hz=len(raw) / duration, retained_pixel_rate_hz=pixel_outputs / duration,
                    hardware_returned_sample_rate_hz=len(self._samples) / duration,
                    output_bits=bits, known_output_bit_subtotal=sum(v for v in costs if v is not None),
                    output_bitrate_bps=None if bits is None else bits / duration,
                    hardware_transport_bits=None,
                    modeled_power_watt=None, modeled_conversion_energy_joule=None,
                    actual_DAQ_configuration_qualified=False)
