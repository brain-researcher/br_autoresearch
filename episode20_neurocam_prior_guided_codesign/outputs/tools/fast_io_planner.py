"""O1: template-level planning for EP20's repeating 16-second acquisition.

The legacy RepeatingAcquisitionSchedule supplies one validated 4-ms template.
This module stores it once as typed arrays and generates every cycle as an
exact vectorized time shift (repeating_acquisition_schedule._shift). Midpoint
holds use the legacy bisection predicate in closed, vectorized form on the
absolute shifted slot times (raw_input_hold / streaming_midpoint_hold). The
field and noise inputs are bound by descriptor digests computed once.

Per-window replay digests are replaced, as approved on 2026-10-12, by:
(1) the legacy per-window validators run once on template windows 0-2 and on
the final window; (2) a vectorized structural proof over all 4,000 windows that
every cycle is an order-preserving exact shift with the same hold/split
pattern; and (3) spot checks against the legacy scalar bounds. Planning only:
no electrical state is advanced. Schedules, thresholds, tolerances, holds and
outputs are those of the legacy recipe; nothing here selects a new value.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import struct
from types import SimpleNamespace

import numpy as np

from fast_io_exact import (CYCLE_COUNT, CYCLE_SECONDS, bitwise_equal, float_bits, last_window,
                           midpoint, shift, upper_div, upper_mul, upper_result)


RECORD_SECONDS = 16.0
SOURCE_NODE_OFFSET = 4096


class PlanningError(ValueError):
    """The supplied schedule leaves the representation validated here."""


def _floats(values, name):
    values = tuple(values)
    if any(type(value) is not float for value in values):
        raise PlanningError(f"{name} must be builtin floats in the legacy template")
    array = np.array(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise PlanningError(f"{name} must be finite")
    return array


def _local(identifier, prefix):
    if not isinstance(identifier, str) or not identifier.startswith(prefix + ":"):
        raise PlanningError("template records need local canonical IDs")
    suffix = identifier[len(prefix) + 1:]
    if not suffix.isdecimal() or str(int(suffix)) != suffix:
        raise PlanningError("template records need local canonical IDs")
    return int(suffix)


def _csr(groups, dtype):
    pointer = np.zeros(len(groups) + 1, dtype=np.int64)
    pointer[1:] = np.cumsum([len(group) for group in groups])
    flat = [item for group in groups for item in group]
    return pointer, np.array(flat, dtype=dtype)


def _frozen(array):
    array = np.ascontiguousarray(array)
    array.setflags(write=False)
    return array


def _digest_arrays(prefix, items):
    digest = hashlib.sha256(prefix)
    for name, value in items:
        digest.update(name.encode("utf-8") + b"\0")
        if isinstance(value, np.ndarray):
            digest.update(str(value.dtype.str).encode() + str(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=repr).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TemplateArrays:
    """One 4-ms template, cycle-relative times, local integer IDs."""
    reference_id: str
    adc_ids: tuple
    input_ids: tuple
    input_adc: np.ndarray
    external_ports: tuple
    # physical actions
    action_start: np.ndarray
    action_end: np.ndarray
    action_row: np.ndarray
    action_selected: np.ndarray
    action_external_volts: np.ndarray
    action_charge_ptr: np.ndarray
    action_charge_state: np.ndarray
    action_charge_value: np.ndarray
    action_selection_objects: tuple
    action_charge_objects: tuple
    # raw conversions
    conversion_roles: tuple
    conv_adc: np.ndarray
    conv_input: np.ndarray
    conv_role: np.ndarray
    conv_aperture_start: np.ndarray
    conv_aperture_end: np.ndarray
    conv_completed: np.ndarray
    conv_assigned: np.ndarray
    conv_valid: np.ndarray
    conv_quantizer: np.ndarray
    quantizers: tuple
    # returned words
    ret_conv_ptr: np.ndarray
    ret_conv: np.ndarray
    ret_weights: np.ndarray
    ret_divisor: np.ndarray
    ret_assigned: np.ndarray
    ret_available: np.ndarray
    ret_output_quantizer: np.ndarray
    # observations
    site_labels: tuple
    observation_roles: tuple
    obs_site: np.ndarray
    obs_role: np.ndarray
    obs_sample_ptr: np.ndarray
    obs_sample: np.ndarray
    obs_weights: np.ndarray
    obs_divisor: np.ndarray
    obs_assigned: np.ndarray
    obs_available: np.ndarray
    obs_payload_bits: np.ndarray
    obs_overhead_bits: np.ndarray
    # dependency lifetimes (relative publication deadlines)
    raw_lifetime: np.ndarray
    sample_lifetime: np.ndarray
    final_publication_horizon: float
    acquisition_window: tuple
    digest: str

    @property
    def actions(self):
        return len(self.action_start)

    @property
    def conversions(self):
        return len(self.conv_adc)

    @property
    def returned(self):
        return len(self.ret_divisor)

    @property
    def observations(self):
        return len(self.obs_site)


def template_from_schedule(schedule, frontend):
    """Typed arrays for the legacy schedule's validated 4-ms template.

    The legacy builder already ran its complete template checks. This copies
    every record field and refuses representations that this planner does
    not reproduce exactly (observation nuisances, non-float timestamps or
    divisors, seam-crossing apertures, non-local dependencies).
    """
    from acquisition_readout import Observation, QuantizerTransfer, RawConversion, ReturnedSample
    from daq_mux_hold import DAQGroupDrive, PassiveDAQFrontend
    from raw_input_hold import PhysicalAction
    from repeating_acquisition_schedule import RepeatingAcquisitionSchedule

    if type(schedule) is not RepeatingAcquisitionSchedule or not isinstance(frontend, PassiveDAQFrontend):
        raise PlanningError("the legacy repeating schedule and its frontend are required")
    actions, readout = schedule._actions, schedule._readout
    if readout.nuisance_bindings:
        raise PlanningError("observation-nuisance templates are not represented by this planner")
    adc_ids = tuple(group.adc_id for group in frontend.groups)
    bindings = tuple(schedule.input_bindings)
    input_ids = tuple(binding.input_id for binding in bindings)
    if (len(set(input_ids)) != len(input_ids) or any(binding.kind != "hold" for binding in bindings)
            or set(binding.adc_id for binding in bindings) != set(adc_ids)):
        raise PlanningError("one hold input binding per physical ADC is required")
    input_adc = np.array([adc_ids.index(binding.hold_adc_id) for binding in bindings], dtype=np.int64)
    external_ports = tuple((group.adc_id, port.input_id) for group in frontend.groups
                           for port in group.external_inputs)

    references = {action.voltage_reference_id for action in actions}
    if len(references) != 1 or not all(type(action) is PhysicalAction for action in actions):
        raise PlanningError("one common voltage reference and PhysicalAction records are required")
    reference_id = references.pop()
    count = len(actions)
    selected = np.full((count, len(adc_ids)), -1, dtype=np.int64)
    external = np.full((count, len(external_ports)), np.nan)
    for index, action in enumerate(actions):
        mapping = action.daq_selected_sources
        if mapping is None or set(mapping) != set(adc_ids):
            raise PlanningError("every template action needs one explicit selection per ADC")
        for column, adc in enumerate(adc_ids):
            value = mapping[adc]
            member = value.selected_input if isinstance(value, DAQGroupDrive) else value
            if isinstance(value, DAQGroupDrive):
                if value.voltage_reference_id != reference_id:
                    raise PlanningError("external drives must use the action reference")
                volts = dict(value.external_volts)
                for port, (port_adc, port_input) in enumerate(external_ports):
                    if port_adc == adc:
                        if port_input not in volts or type(volts[port_input]) is not float:
                            raise PlanningError("every external port needs its explicit voltage")
                        external[index, port] = volts[port_input]
            if member is None:
                continue
            if type(member) is int and 0 <= member < 64:
                selected[index, column] = member
            elif isinstance(member, str) and (adc, member) in external_ports:
                selected[index, column] = 64 + external_ports.index((adc, member))
            else:
                raise PlanningError("unsupported DAQ selection member")
    charge_ptr, charge_state = _csr([[pair[0] for pair in action.injected_charge_coulomb]
                                     for action in actions], np.int64)
    charge_value = _floats((pair[1] for action in actions for pair in action.injected_charge_coulomb),
                           "boundary charges")

    raw = readout.conversions
    roles = tuple(sorted({plan.role for plan in raw}))
    quantizers, quantizer_index = [], {}
    for plan in raw:
        if type(plan) is not RawConversion or type(plan.quantizer) is not QuantizerTransfer:
            raise PlanningError("template conversions need exact legacy record types")
        if id(plan.quantizer) not in quantizer_index:
            quantizer_index[id(plan.quantizer)] = len(quantizers)
            quantizers.append(plan.quantizer)
    for position, plan in enumerate(raw):
        if _local(plan.conversion_id, "r") != position or type(plan.valid) is not bool:
            raise PlanningError("template conversion IDs must be sequential")
    samples = readout.returned_samples
    for position, plan in enumerate(samples):
        if type(plan) is not ReturnedSample or _local(plan.sample_id, "s") != position:
            raise PlanningError("template returned IDs must be sequential")
        if type(plan.divisor) is not float:
            raise PlanningError("returned divisors must be builtin floats")
        if plan.output_quantizer is not None:
            if type(plan.output_quantizer) is not QuantizerTransfer:
                raise PlanningError("returned output quantizer must be an exact legacy transfer")
            if id(plan.output_quantizer) not in quantizer_index:
                quantizer_index[id(plan.output_quantizer)] = len(quantizers)
                quantizers.append(plan.output_quantizer)
    observations = readout.observations
    site_labels, site_index = [], {}
    observation_roles = tuple(sorted({plan.role for plan in observations}))
    for position, plan in enumerate(observations):
        if type(plan) is not Observation or _local(plan.observation_id, "o") != position:
            raise PlanningError("template observation IDs must be sequential")
        if type(plan.divisor) is not float:
            raise PlanningError("observation divisors must be builtin floats")
        for bits in (plan.payload_bits, plan.overhead_bits):
            if bits is not None and type(bits) is not int:
                raise PlanningError("bit costs must be builtin integers or None")
        if plan.site_id not in site_index:
            site_index[plan.site_id] = len(site_labels)
            site_labels.append(plan.site_id)

    ret_conv_ptr, ret_conv = _csr([[_local(key, "r") for key in plan.conversion_ids] for plan in samples], np.int64)
    obs_sample_ptr, obs_sample = _csr([[_local(key, "s") for key in plan.sample_ids]
                                       for plan in observations], np.int64)
    aperture_start = _floats((plan.aperture_start for plan in raw), "aperture starts")
    aperture_end = _floats((plan.aperture_end for plan in raw), "aperture ends")
    completed = _floats((plan.completed_at for plan in raw), "completion times")
    if np.any(aperture_start < 0) or np.any(aperture_end > CYCLE_SECONDS) or np.any(completed > CYCLE_SECONDS):
        raise PlanningError("seam-crossing apertures are not represented by this planner")
    arrays = dict(
        reference_id=reference_id, adc_ids=adc_ids, input_ids=input_ids, input_adc=_frozen(input_adc),
        external_ports=external_ports,
        action_start=_frozen(_floats((a.start for a in actions), "action starts")),
        action_end=_frozen(_floats((a.end for a in actions), "action ends")),
        action_row=_frozen(np.array([a.gate_row for a in actions], dtype=np.int64)),
        action_selected=_frozen(selected), action_external_volts=_frozen(external),
        action_charge_ptr=_frozen(charge_ptr), action_charge_state=_frozen(charge_state),
        action_charge_value=_frozen(charge_value),
        action_selection_objects=tuple(a.daq_selected_sources for a in actions),
        action_charge_objects=tuple(a.injected_charge_coulomb for a in actions),
        conversion_roles=roles,
        conv_adc=_frozen(np.array([adc_ids.index(p.adc_id) for p in raw], dtype=np.int64)),
        conv_input=_frozen(np.array([input_ids.index(p.input_id) for p in raw], dtype=np.int64)),
        conv_role=_frozen(np.array([roles.index(p.role) for p in raw], dtype=np.int64)),
        conv_aperture_start=_frozen(aperture_start), conv_aperture_end=_frozen(aperture_end),
        conv_completed=_frozen(completed),
        conv_assigned=_frozen(_floats((p.assigned_timestamp for p in raw), "raw timestamps")),
        conv_valid=_frozen(np.array([p.valid for p in raw], dtype=bool)),
        conv_quantizer=_frozen(np.array([quantizer_index[id(p.quantizer)] for p in raw], dtype=np.int64)),
        quantizers=tuple(quantizers),
        ret_conv_ptr=_frozen(ret_conv_ptr), ret_conv=_frozen(ret_conv),
        ret_weights=_frozen(_floats((w for p in samples for w in p.weights), "returned weights")),
        ret_divisor=_frozen(_floats((p.divisor for p in samples), "returned divisors")),
        ret_assigned=_frozen(_floats((p.assigned_timestamp for p in samples), "returned timestamps")),
        ret_available=_frozen(_floats((p.available_at for p in samples), "returned availability")),
        ret_output_quantizer=_frozen(np.array([-1 if p.output_quantizer is None
                                               else quantizer_index[id(p.output_quantizer)]
                                               for p in samples], dtype=np.int64)),
        site_labels=tuple(site_labels), observation_roles=observation_roles,
        obs_site=_frozen(np.array([site_index[p.site_id] for p in observations], dtype=np.int64)),
        obs_role=_frozen(np.array([observation_roles.index(p.role) for p in observations], dtype=np.int64)),
        obs_sample_ptr=_frozen(obs_sample_ptr), obs_sample=_frozen(obs_sample),
        obs_weights=_frozen(_floats((w for p in observations for w in p.weights), "observation weights")),
        obs_divisor=_frozen(_floats((p.divisor for p in observations), "observation divisors")),
        obs_assigned=_frozen(_floats((p.assigned_timestamp for p in observations), "observation timestamps")),
        obs_available=_frozen(_floats((p.available_at for p in observations), "observation availability")),
        obs_payload_bits=_frozen(np.array([-1 if p.payload_bits is None else p.payload_bits
                                           for p in observations], dtype=np.int64)),
        obs_overhead_bits=_frozen(np.array([-1 if p.overhead_bits is None else p.overhead_bits
                                            for p in observations], dtype=np.int64)),
        raw_lifetime=_frozen(_floats(schedule._raw_lifetimes, "raw lifetimes")),
        sample_lifetime=_frozen(_floats(schedule._sample_lifetimes, "sample lifetimes")),
        final_publication_horizon=float(schedule.final_publication_horizon),
        acquisition_window=tuple(map(float, schedule.acquisition_window)))
    numeric = [(name, value) for name, value in sorted(arrays.items()) if isinstance(value, np.ndarray)]
    labels = [(name, arrays[name]) for name in ("reference_id", "adc_ids", "input_ids", "external_ports",
                                                "conversion_roles", "site_labels", "observation_roles",
                                                "final_publication_horizon", "acquisition_window")]
    quantizer_tables = [(f"quantizer_{i}", np.array((q.input_min_volts, q.input_max_volts,
                                                     *q.thresholds_volts, *q.code_volts)))
                        for i, q in enumerate(quantizers)]
    selections = [("selections", [[repr(a.daq_selected_sources[key]) for key in adc_ids] for a in actions])]
    arrays["digest"] = _digest_arrays(b"fast_io_template_v1\0", numeric + labels + quantizer_tables + selections)
    return TemplateArrays(**arrays)


@dataclass(frozen=True)
class CubicCellNoiseBound:
    """Closed form of SpectralCubicCircuitNoise._cell_bound on original slots.

    The frozen original clock maps every query strictly inside a physical
    slot to that slot; the bound is the outward M4 * radius**4 / 24 with the
    binding's cached normalized fourth-derivative bound. It refuses above one,
    as the legacy provider does; it never subdivides or recentres a cell.
    """
    normalized_fourth_derivative_bound: float

    def cells(self, slot_start, slot_end):
        origin = slot_start + (slot_end - slot_start) / 2.
        radius = np.maximum(upper_result(origin - slot_start), upper_result(slot_end - origin))
        bound = np.full(np.shape(radius), self.normalized_fourth_derivative_bound)
        for _ in range(4):
            bound = upper_mul(bound, radius)
        bound = upper_div(bound, 24.)
        if np.any(bound > 1.):
            raise PlanningError("original cubic noise cell exceeds its pointwise A/V tolerance")
        return origin, bound

    def __call__(self, start, end, middle, slot_start, slot_end):
        if not np.all((slot_start <= start) & (start < middle) & (middle < end) & (end <= slot_end)):
            raise PlanningError("a numerical noise interval cannot cross an original physical cell")
        return self.cells(slot_start, slot_end)[1]


@dataclass(frozen=True)
class WindowPlan:
    """All planning records of one 4-ms window as typed arrays (global IDs)."""
    index: int
    start: float
    end: float
    action_start: np.ndarray
    action_end: np.ndarray
    cell_origin: np.ndarray
    hold_start: np.ndarray
    hold_end: np.ndarray
    hold_key: np.ndarray
    hold_action: np.ndarray
    hold_first: np.ndarray
    hold_event: np.ndarray
    hold_field_bound: np.ndarray
    hold_noise_bound: np.ndarray
    split_bounds: np.ndarray
    split_hold: np.ndarray
    split_new_event: np.ndarray
    split_active: np.ndarray
    conversion_id: np.ndarray
    conv_aperture_start: np.ndarray
    conv_aperture_end: np.ndarray
    conv_completed: np.ndarray
    conv_assigned: np.ndarray
    conv_split_first: np.ndarray
    conv_split_count: np.ndarray
    returned_id: np.ndarray
    ret_assigned: np.ndarray
    ret_available: np.ndarray
    observation_id: np.ndarray
    obs_assigned: np.ndarray
    obs_available: np.ndarray
    raw_last: np.ndarray
    sample_last: np.ndarray
    candidates: int

    @property
    def split_event(self):
        return self.hold_event[self.split_hold]

    @property
    def split_charge(self):
        return self.split_new_event & self.hold_first[self.split_hold]


def bisect_holds(slot_start, slot_end, *, lipschitz, epsilon, noise_bound, max_depth=60):
    """Vectorized legacy midpoint-hold bisection over independent atomic slots.

    Mirrors streaming_midpoint_hold.compile_streaming_midpoint_hold for one raw
    segment: a candidate [a, b] is accepted when upper(radius*L) <= epsilon and
    the scaled noise bound <= 1; otherwise it is split at its exact midpoint.
    Returned holds are chronological (the legacy LIFO right-then-left order).
    Returns starts, ends, keys, owning slot, field bounds, noise bounds and the
    number of candidate bound evaluations.
    """
    slot_start = np.asarray(slot_start, dtype=np.float64)
    slot_end = np.asarray(slot_end, dtype=np.float64)
    a, b = slot_start, slot_end
    owner = np.arange(len(a), dtype=np.int64)
    parts = []
    candidates = 0
    for _ in range(max_depth):
        if not len(a):
            break
        middle, radius = midpoint(a, b)
        error = upper_mul(radius, lipschitz)
        scaled = np.asarray(noise_bound(a, b, middle, slot_start[owner], slot_end[owner]), dtype=np.float64)
        if scaled.shape != a.shape or not np.all(np.isfinite(scaled)) or np.any(scaled < 0):
            raise PlanningError("scaled physical-noise hold bound must be finite and nonnegative")
        candidates += len(a)
        accept = (error <= epsilon) & (scaled <= 1.)
        parts.append((a[accept], b[accept], middle[accept], owner[accept], error[accept], scaled[accept]))
        refine = ~accept
        a, b, owner = (np.concatenate((a[refine], middle[refine])),
                       np.concatenate((middle[refine], b[refine])),
                       np.concatenate((owner[refine], owner[refine])))
    else:
        if len(a):
            raise PlanningError("hold bisection exceeded its refinement depth")
    start, end, key, slot, field, noise = (np.concatenate(values) for values in zip(*parts))
    order = np.argsort(start, kind="stable")
    start, end, key, slot, field, noise = (value[order] for value in (start, end, key, slot, field, noise))
    if np.any(slot[1:] < slot[:-1]) or not np.array_equal(start[1:], end[:-1]):
        raise PlanningError("compiled holds must exactly and chronologically cover their slots")
    return start, end, key, slot, field, noise, candidates


class FullRecordPlan:
    """Template plus closed-form holds; windows are generated on demand.

    ``validate()`` performs the complete template-level validation and
    returns exact full-record totals. ``window(k)`` returns the typed planning
    records of window k; ``legacy_window(k)`` materializes the same records
    as legacy dataclasses (interoperability and byte-equality testing).
    """

    def __init__(self, *, template, raw_time, contact_factors, noise, epsilon_input_volts,
                 max_intervals, max_work, check, noise_bound=None, world_id=None):
        if type(template) is not TemplateArrays or not callable(check):
            raise PlanningError("typed template and explicit check are required")
        if template.acquisition_window != (0., RECORD_SECONDS):
            raise PlanningError("the complete original 16-second record is required")
        self.template, self._check = template, check
        self.epsilon = float(epsilon_input_volts)
        self.max_intervals, self.max_work = int(max_intervals), int(max_work)
        breakpoints = [float(t) for t in raw_time.breakpoints() if 0. < t < RECORD_SECONDS]
        if breakpoints:
            raise PlanningError("interior raw breakpoints (event/M4 worlds) are not yet represented")
        self.raw_edges = (0., RECORD_SECONDS)
        self.bound_contact = raw_time.bind_contact_factors(contact_factors)
        from raw_input_hold import _MidpointProvider
        self.field_provider = _MidpointProvider(self.bound_contact)
        self.lipschitz = float(self.bound_contact.derivative_bound_volts_per_second(*self.raw_edges))
        self.noise = noise
        if noise_bound is None:
            noise_bound = CubicCellNoiseBound(float(noise._normalized_fourth_derivative_bound))
        self.noise_bound = noise_bound
        self.world_id = raw_time.metadata["world_id"] if world_id is None else world_id
        self.field_descriptor_digest = self._field_digest(raw_time, contact_factors)
        self.noise_descriptor_digest = self._noise_digest(noise)
        self._window_hold_offsets = None
        self._template_pattern = None
        self.totals = None

    # -- descriptors digested once -------------------------------------------------
    def _field_digest(self, raw_time, contacts):
        modes = np.ascontiguousarray(contacts.contact_modes, dtype="<f8")
        drivers = np.ascontiguousarray(contacts.raw_drivers, dtype="<f8")
        if not np.array_equal(self.bound_contact._modes, contacts.contact_modes):
            raise PlanningError("bound field provider must own the supplied contact modes")
        details = dict(raw_time.metadata, contact_metadata=contacts.metadata,
                       carrier_hz=float(raw_time._frequency) if hasattr(raw_time, "_frequency") else None,
                       applied_scale=float(raw_time._scale) if hasattr(raw_time, "_scale") else None,
                       lipschitz_volts_per_second=float.hex(self.lipschitz))
        digest = hashlib.sha256(b"fast_io_field_descriptor_v1\0")
        digest.update(json.dumps(details, sort_keys=True, default=repr).encode("utf-8"))
        digest.update(modes.tobytes())
        digest.update(drivers.tobytes())
        return digest.hexdigest()

    def _noise_digest(self, noise):
        if noise is None:
            return None
        basis = getattr(noise, "_basis_digest", None)
        if not isinstance(basis, bytes):
            raise PlanningError("the owned frozen cubic binding (basis digest) is required")
        digest = hashlib.sha256(b"fast_io_noise_descriptor_v1\0" + basis)
        digest.update(struct.pack("<d", float(noise._normalized_fourth_derivative_bound)))
        digest.update(json.dumps([float.hex(float(v)) for v in noise.port_absolute_tolerances[:1]]
                                 + [len(noise.port_absolute_tolerances)]).encode())
        return digest.hexdigest()

    def verify_descriptors(self, raw_time, contacts):
        """Recompute both descriptor digests (immutability check at use time)."""
        return (self._field_digest(raw_time, contacts) == self.field_descriptor_digest
                and self._noise_digest(self.noise) == self.noise_descriptor_digest)

    # -- holds and windows ------------------------------------------------------------
    def _holds(self, action_start, action_end):
        return bisect_holds(action_start, action_end, lipschitz=self.lipschitz, epsilon=self.epsilon,
                            noise_bound=self.noise_bound)

    def _cells(self, action_start, action_end):
        if isinstance(self.noise_bound, CubicCellNoiseBound):
            return self.noise_bound.cells(action_start, action_end)[0]
        return action_start + (action_end - action_start) / 2.

    def window(self, k):
        """Typed planning records of window k (exact shifts, closed-form holds)."""
        t = self.template
        if type(k) is not int or not 0 <= k < CYCLE_COUNT:
            raise PlanningError("window index must lie in the 16-second record")
        if self._window_hold_offsets is None:
            raise PlanningError("validate() must complete before windows are generated")
        start, end = float(shift(k, 0.0)), float(shift(k, CYCLE_SECONDS))
        action_start, action_end = shift(k, t.action_start), shift(k, t.action_end)
        hold_start, hold_end, hold_key, hold_action, field, noise, candidates = self._holds(action_start, action_end)
        hold_first = hold_start == action_start[hold_action]
        first_event = int(self._window_hold_offsets[k])
        hold_event = first_event + np.arange(len(hold_start), dtype=np.int64)
        ap_start, ap_end = shift(k, t.conv_aperture_start), shift(k, t.conv_aperture_end)
        inside = np.concatenate((ap_start, ap_end))
        inside = inside[(inside > start) & (inside < end)]
        bounds = np.unique(np.concatenate(([start, end], hold_start, inside)))
        if not np.all(np.isfinite(bounds)) or np.any(float_bits(bounds) == float_bits(-0.0)):
            raise PlanningError("split boundaries must be finite and nonnegative")
        split_hold = np.searchsorted(hold_end, bounds[:-1], side="right").astype(np.int64)
        # No hold crosses a window seam, so every window opens a new event.
        split_new_event = np.ones(len(split_hold), dtype=bool)
        split_new_event[1:] = split_hold[1:] != split_hold[:-1]
        first = np.searchsorted(bounds, ap_start, side="left")
        last = np.searchsorted(bounds, ap_end, side="left")
        if (np.any(first >= len(bounds)) or np.any(last >= len(bounds))
                or not np.array_equal(bounds[first], ap_start) or not np.array_equal(bounds[last], ap_end)
                or np.any(last <= first)):
            raise PlanningError("every aperture boundary must be an exact split boundary")
        delta = np.zeros((len(bounds), len(t.input_ids)), dtype=np.int64)
        np.add.at(delta, (first, t.conv_input), 1)
        np.add.at(delta, (last, t.conv_input), -1)
        active = np.cumsum(delta, axis=0)[:-1] > 0
        base_r, base_s, base_o = k * t.conversions, k * t.returned, k * t.observations
        return WindowPlan(
            index=k, start=start, end=end, action_start=action_start, action_end=action_end,
            cell_origin=self._cells(action_start, action_end),
            hold_start=hold_start, hold_end=hold_end, hold_key=hold_key, hold_action=hold_action,
            hold_first=hold_first, hold_event=hold_event, hold_field_bound=field, hold_noise_bound=noise,
            split_bounds=bounds, split_hold=split_hold, split_new_event=split_new_event, split_active=active,
            conversion_id=base_r + np.arange(t.conversions, dtype=np.int64),
            conv_aperture_start=ap_start, conv_aperture_end=ap_end,
            conv_completed=shift(k, t.conv_completed), conv_assigned=shift(k, t.conv_assigned),
            conv_split_first=first.astype(np.int64), conv_split_count=(last - first).astype(np.int64),
            returned_id=base_s + np.arange(t.returned, dtype=np.int64),
            ret_assigned=shift(k, t.ret_assigned), ret_available=shift(k, t.ret_available),
            observation_id=base_o + np.arange(t.observations, dtype=np.int64),
            obs_assigned=shift(k, t.obs_assigned), obs_available=shift(k, t.obs_available),
            raw_last=last_window(shift(k, t.raw_lifetime)), sample_last=last_window(shift(k, t.sample_lifetime)),
            candidates=int(candidates))

    def _template_times(self):
        t = self.template
        values = np.concatenate(([0.0, CYCLE_SECONDS], t.action_start, t.action_end, t.conv_aperture_start,
                                 t.conv_aperture_end, t.conv_completed, t.conv_assigned, t.ret_assigned,
                                 t.ret_available, t.obs_assigned, t.obs_available, t.raw_lifetime,
                                 t.sample_lifetime))
        if np.any(values < 0) or np.any(values > CYCLE_SECONDS):
            raise PlanningError("template times must lie inside one closed cycle")
        return np.unique(values)

    def validate(self, *, chunk_windows=250):
        """Structural proof over all windows plus exact full-record totals.

        Every template time is shifted for every window and must stay strictly
        increasing, so all equalities and orderings of the 4-ms template (and
        therefore all legacy dependency, coverage, publication and last-use
        relations) hold unchanged in every window. Holds are computed for all
        6.4 M slots; each window must reproduce window 0's hold/split pattern.
        """
        t, check = self.template, self._check
        times = self._template_times()
        if times[0] != 0.0 or times[-1] != CYCLE_SECONDS:
            raise PlanningError("template times must span one closed cycle")
        if not (np.array_equal(t.action_start[1:], t.action_end[:-1]) and t.action_start[0] == 0.0
                and t.action_end[-1] == CYCLE_SECONDS and np.all(t.action_end > t.action_start)):
            raise PlanningError("template actions must exactly cover one cycle")
        # Legacy split boundaries: window bounds, hold starts and aperture
        # boundaries. Their template part is shifted; interior hold starts
        # (bisection points) are recomputed in every window.
        split_times = np.unique(np.concatenate(([0.0, CYCLE_SECONDS], t.action_start, t.action_end,
                                                t.conv_aperture_start, t.conv_aperture_end)))
        # Window 0 sets the reference pattern; every later window must match it.
        zero_start, zero_end = shift(0, t.action_start), shift(0, t.action_end)
        h0 = self._holds(zero_start, zero_end)
        holds_per_slot = np.bincount(h0[3], minlength=t.actions)
        holds_per_window = int(len(h0[0]))
        interior = h0[0] != zero_start[h0[3]]
        self._window_hold_offsets = np.arange(CYCLE_COUNT + 1, dtype=np.int64) * holds_per_window
        reference = self.window(0)
        merged = np.concatenate((shift(0, split_times), reference.hold_start[interior]))
        order = np.argsort(merged, kind="stable")
        if np.any(np.diff(merged[order]) <= 0):
            raise PlanningError("interior hold boundaries must not coincide with template split times")
        if not np.array_equal(merged[order], reference.split_bounds):
            raise PlanningError("template and generic split constructions disagree")
        maximum_field = maximum_noise = 0.
        candidates = compiled = 0
        for first in range(0, CYCLE_COUNT, chunk_windows):
            check()
            cycles = np.arange(first, min(CYCLE_COUNT, first + chunk_windows), dtype=np.int64)
            shifted = shift(cycles[:, None], times[None, :])
            if np.any(np.diff(shifted, axis=1) <= 0):
                raise PlanningError("an exact shift collapsed or reordered template times")
            starts = shift(cycles[:, None], t.action_start[None, :]).ravel()
            ends = shift(cycles[:, None], t.action_end[None, :]).ravel()
            hs, he, hk, slot, field, noise, count = self._holds(starts, ends)
            check()
            per_window = np.bincount(slot // t.actions, minlength=len(cycles))
            per_slot = np.bincount(slot, minlength=len(starts)).reshape(len(cycles), t.actions)
            if np.any(per_window != holds_per_window) or np.any(per_slot != holds_per_slot[None, :]):
                raise PlanningError("every window must reproduce the template hold pattern")
            interior_starts = hs.reshape(len(cycles), holds_per_window)[:, interior]
            union = np.concatenate((shift(cycles[:, None], split_times[None, :]), interior_starts), axis=1)[:, order]
            if np.any(np.diff(union, axis=1) <= 0):
                raise PlanningError("every window must reproduce the template split pattern")
            candidates += int(count)
            compiled += len(hs)
            maximum_field = max(maximum_field, float(field.max()))
            maximum_noise = max(maximum_noise, float(noise.max()))
        horizon = float(shift(CYCLE_COUNT - 1, max(CYCLE_SECONDS, float(t.ret_available.max()),
                                                   float(t.obs_available.max()))))
        if horizon != t.final_publication_horizon or float(shift(CYCLE_COUNT - 1, CYCLE_SECONDS)) != RECORD_SECONDS:
            raise PlanningError("final publication horizon or record end differs from the legacy schedule")
        if compiled > self.max_intervals or 1 + 2 * candidates > self.max_work:
            raise PlanningError("closed-form holds exceed the unchanged legacy planning limits")
        self._template_pattern = dict(holds_per_window=holds_per_window, splits_per_window=len(reference.split_hold),
                                      template_split_times=len(split_times),
                                      interior_hold_boundaries=int(np.sum(interior)),
                                      template_times=len(times))
        self.totals = self._totals(reference, compiled, candidates, maximum_field, maximum_noise)
        return dict(self.totals)

    def _totals(self, reference, compiled, candidates, maximum_field, maximum_noise):
        t = self.template
        windows = CYCLE_COUNT
        hold_report = dict(
            world_id=self.world_id, scope="internal_simulator_only",
            hold_convention="midpoint_constant_input_on_each_half_open_interval",
            support_start=0.0, support_end=RECORD_SECONDS, voltage_reference_id=t.reference_id,
            epsilon_input_volts=self.epsilon, maximum_analytic_hold_bound_volts=maximum_field,
            original_physical_actions=windows * t.actions, required_atomic_intervals=windows * t.actions,
            compiled_intervals=compiled, max_intervals=self.max_intervals,
            work_units=1 + 2 * candidates, max_work=self.max_work, work_limit_scope="each_complete_planning_pass",
            candidate_bound_evaluations=candidates, raw_segment_derivative_enclosures=1,
            retained_raw_boundary_count=2, retained_raw_boundary_numeric_bytes=16,
            retained_numeric_bytes_exclude_python_objects_and_contact_factors=True,
            bound_covers="analytic_midpoint_hold_approximation_only",
            outward_arithmetic="each_nonnegative_binary64_multiply_add_divide_and_radius_rounded_up",
            waveform_evaluation_arithmetic_error_included=False, circuit_output_error_included=False,
            nuisances_applied=False, acquisition_policy_receives_future_fields_or_splits=False,
            actual_DAQ_or_electrical_parameters_selected=False, electrical_state_advanced=False,
            dense_voltage_plan_materialized=False, full_physical_action_or_hold_tuple_retained=False,
            physical_noise_hold_bound_applied=True, maximum_scaled_physical_noise_hold_bound=maximum_noise,
            noise_bound_uses_same_midpoint_key=True)
        readout = self.readout_totals(windows)
        return dict(hold=hold_report, readout=readout, windows=windows,
                    physical_input_events=compiled, physical_fragments=compiled,
                    split_intervals=windows * len(reference.split_hold),
                    legacy_equivalent_window_digest_bytes=32 * windows,
                    raw_conversions=windows * t.conversions, returned_words=windows * t.returned,
                    observations=windows * t.observations,
                    aperture_support_records=windows * int(sum(
                        len({c for s in t.obs_sample[t.obs_sample_ptr[o]:t.obs_sample_ptr[o + 1]]
                             for c in t.ret_conv[t.ret_conv_ptr[s]:t.ret_conv_ptr[s + 1]]})
                        for o in range(t.observations))),
                    template_digest=t.digest, field_descriptor_digest=self.field_descriptor_digest,
                    noise_descriptor_digest=self.noise_descriptor_digest,
                    field_lipschitz_volts_per_second=self.lipschitz,
                    final_publication_horizon=t.final_publication_horizon,
                    pattern=dict(self._template_pattern))

    def readout_totals(self, windows):
        """Legacy StreamingAcquisitionReadout.finalize (metadata-only) for a W-window prefix."""
        t = self.template
        start, end = 0.0, float(shift(windows - 1, CYCLE_SECONDS))
        duration = end - start
        roles = {}
        for index, role in enumerate(t.conversion_roles):
            roles[role] = windows * int(np.sum(t.conv_role == index))
        adcs = {}
        for index, adc in enumerate(t.adc_ids):
            count = int(np.sum(t.conv_adc == index))
            if count:
                adcs[adc] = windows * count
        used = np.zeros(t.conversions, dtype=bool)
        used[t.ret_conv] = True
        pixel_role = t.observation_roles.index("pixel") if "pixel" in t.observation_roles else -1
        pixels = windows * int(np.sum(t.obs_role == pixel_role))
        unknown = bool(np.any(t.obs_payload_bits < 0) or np.any(t.obs_overhead_bits < 0))
        known = windows * int(np.sum(np.where(t.obs_payload_bits < 0, 0, t.obs_payload_bits))
                              + np.sum(np.where(t.obs_overhead_bits < 0, 0, t.obs_overhead_bits)))
        bits = None if unknown else known
        completed_raw, completed_samples = windows * t.conversions, windows * t.returned
        reference = self.window(0)
        return dict(actual_internal_conversions=completed_raw, hardware_returned_samples=completed_samples,
            published_observations=windows * t.observations, retained_pixel_observations=pixels,
            valid_retained_pixel_observations=None, conversions_by_role=roles, conversions_by_adc=adcs,
            conversions_without_returned_sample=windows * int(np.sum(~used)),
            raw_conversion_rate_hz=completed_raw / duration, retained_pixel_rate_hz=pixels / duration,
            hardware_returned_sample_rate_hz=completed_samples / duration,
            output_bits=bits, known_output_bit_subtotal=known,
            output_bitrate_bps=None if bits is None else bits / duration,
            hardware_transport_bits=None, modeled_power_watt=None, modeled_conversion_energy_joule=None,
            actual_DAQ_configuration_qualified=False, metadata_only=True, windows_closed=windows,
            interval_updates=windows * len(reference.split_hold), peak_live_counts=self.peak_live_counts(),
            acquisition_start=start, acquisition_end=end, final_publication_horizon=(
                t.final_publication_horizon if windows == CYCLE_COUNT else end),
            nuisance_observations_processed=None, observation_nuisance=None)

    def peak_live_counts(self):
        """Legacy StreamingAcquisitionReadout live counts after one window's admission."""
        t = self.template
        size = 112 * t.conversions + 64 * t.returned + 64 * t.observations
        size += 16 * len(t.ret_weights) + 16 * len(t.obs_weights)
        used = sorted({int(q) for q in t.conv_quantizer} | {int(q) for q in t.ret_output_quantizer if q >= 0})
        size += sum(16 + 8 * (len(t.quantizers[q].thresholds_volts) + len(t.quantizers[q].code_volts))
                    for q in used)
        return dict(conversions=t.conversions, returned_samples=t.returned, observations=t.observations,
                    stored_plan_numeric_bytes=size)

    # -- legacy materialization (interoperability and byte-equality tests) -----------
    def legacy_readout_window(self, k, window=None):
        from acquisition_readout import Observation, RawConversion, ReturnedSample
        from streaming_acquisition_readout import ReadoutWindow
        t = self.template
        w = self.window(k) if window is None else window
        roles, adcs, inputs = t.conversion_roles, t.adc_ids, t.input_ids
        ap_start, ap_end = w.conv_aperture_start.tolist(), w.conv_aperture_end.tolist()
        completed, assigned = w.conv_completed.tolist(), w.conv_assigned.tolist()
        raw_ids = [f"r:{value}" for value in w.conversion_id.tolist()]
        raw = tuple(RawConversion(raw_ids[i], adcs[t.conv_adc[i]], inputs[t.conv_input[i]], roles[t.conv_role[i]],
                                  ap_start[i], ap_end[i], completed[i], assigned[i],
                                  t.quantizers[t.conv_quantizer[i]], bool(t.conv_valid[i]))
                    for i in range(t.conversions))
        sample_ids = [f"s:{value}" for value in w.returned_id.tolist()]
        base_r = k * t.conversions
        ret_assigned, ret_available = w.ret_assigned.tolist(), w.ret_available.tolist()
        weights, divisors = t.ret_weights.tolist(), t.ret_divisor.tolist()
        pointer, members = t.ret_conv_ptr.tolist(), t.ret_conv.tolist()
        samples = tuple(ReturnedSample(sample_ids[i], tuple(f"r:{base_r + c}" for c in members[pointer[i]:pointer[i + 1]]),
                                       tuple(weights[pointer[i]:pointer[i + 1]]), divisors[i],
                                       ret_assigned[i], ret_available[i],
                                       None if t.ret_output_quantizer[i] < 0
                                       else t.quantizers[t.ret_output_quantizer[i]])
                        for i in range(t.returned))
        base_s = k * t.returned
        obs_assigned, obs_available = w.obs_assigned.tolist(), w.obs_available.tolist()
        weights, divisors = t.obs_weights.tolist(), t.obs_divisor.tolist()
        pointer, members = t.obs_sample_ptr.tolist(), t.obs_sample.tolist()
        payload, overhead = t.obs_payload_bits.tolist(), t.obs_overhead_bits.tolist()
        observations = tuple(Observation(f"o:{w.observation_id[i]}", t.site_labels[t.obs_site[i]],
                                         t.observation_roles[t.obs_role[i]],
                                         tuple(f"s:{base_s + s}" for s in members[pointer[i]:pointer[i + 1]]),
                                         tuple(weights[pointer[i]:pointer[i + 1]]), divisors[i],
                                         obs_assigned[i], obs_available[i],
                                         None if payload[i] < 0 else payload[i],
                                         None if overhead[i] < 0 else overhead[i])
                             for i in range(t.observations))
        raw_last = tuple(zip(raw_ids, w.raw_last.tolist()))
        sample_last = tuple(zip(sample_ids, w.sample_last.tolist()))
        return ReadoutWindow(k, w.start, w.end, raw, samples, observations, raw_last, sample_last)

    def legacy_window(self, k, window=None, readout_window=None):
        from array_acquisition import ProviderInterval
        from streaming_array_acquisition import ArrayWindow, StreamPhysicalInterval
        t = self.template
        w = self.window(k) if window is None else window
        readout = self.legacy_readout_window(k, w) if readout_window is None else readout_window
        starts, ends, keys = w.hold_start.tolist(), w.hold_end.tolist(), w.hold_key.tolist()
        events, owners, firsts = w.hold_event.tolist(), w.hold_action.tolist(), w.hold_first.tolist()
        fragments = tuple(StreamPhysicalInterval(events[i], ProviderInterval(
            starts[i], ends[i], int(t.action_row[owners[i]]), keys[i], t.reference_id,
            t.action_selection_objects[owners[i]],
            t.action_charge_objects[owners[i]] if firsts[i] else ()))
            for i in range(len(starts)))
        return ArrayWindow(k, w.start, w.end, fragments, readout)


def legacy_window_validation(plan, *, array, noise, limits, input_bindings, windows=(0, 1, 2),
                             final_window=True):
    """Run the unchanged legacy per-window validators once on template windows.

    StreamingArrayAcquisition._prepare validates fragments, rows, DAQ paths,
    charges, field values, noise descriptors, event IDs and numeric-byte caps;
    the metadata-only StreamingAcquisitionReadout validates dependencies,
    coverage, publication and eviction over contiguous windows 0..2 (two seams).
    The final window is checked by _prepare with its exact prior cursor.
    Returns the legacy window digests and the legacy numeric-byte maximum.
    """
    from array_acquisition import _bind_noise_provider
    from streaming_acquisition_readout import StreamingAcquisitionReadout, StreamingReadoutLimits
    from streaming_array_acquisition import StreamingArrayAcquisition, _PlanCursor, _window_digest
    layout, noise_bytes = _bind_noise_provider(array, noise)
    zero = np.zeros(len(array.state_ids))
    zero.flags.writeable = False
    fake = SimpleNamespace(_check=plan._check, _end=RECORD_SECONDS, _limits=limits, _zero_charge=zero,
                           _noise_bytes=noise_bytes, _array=array, _provider=plan.field_provider,
                           _noise_provider=noise, _noise_layout=layout, _max_window_numeric_bytes=0,
                           _bindings={binding.input_id: binding for binding in input_bindings})
    used_inputs = set()
    contiguous = tuple(windows)
    if contiguous != tuple(range(len(contiguous))):
        raise PlanningError("legacy dry-readout validation needs contiguous windows from 0")
    count = len(contiguous)
    readout_limits = StreamingReadoutLimits(max_conversions=104000000, max_returned_samples=104000000,
        max_observations=17984000, max_interval_updates=40000000, max_windows=4000,
        max_live_conversions=52000, max_live_returned_samples=52000, max_live_observations=9000,
        max_stored_plan_numeric_bytes=64 * 1024**2, max_accounting_categories=32)
    dry = StreamingAcquisitionReadout(input_ids=plan.template.input_ids, acquisition_start=0.0,
        acquisition_end=float(shift(count - 1, CYCLE_SECONDS)),
        final_publication_horizon=float(shift(count - 1, CYCLE_SECONDS)), limits=readout_limits,
        on_raw=lambda _: None, on_observation=lambda _: None, validate_only=True)
    splitter = SimpleNamespace(_limits=limits)
    cursor = _PlanCursor(0.0)
    digests = {}
    for k in contiguous:
        legacy = plan.legacy_window(k)
        normalized, fields, following = StreamingArrayAcquisition._prepare(fake, legacy, cursor)
        digests[k] = _window_digest(normalized, fields).hex()
        StreamingArrayAcquisition._validate_bindings(fake, normalized, used_inputs)
        dry.accept_window(normalized.readout_window)
        for _, start, end in StreamingArrayAcquisition._splits(splitter, normalized, dry):
            plan._check()
            following.splits += 1
            dry.validate_interval(start=start, end=end, input_ids=dry.active_input_ids(start, end))
            dry.advance_publication(end)
        dry.close_window(k)
        cursor = following
    if used_inputs != set(fake._bindings):
        raise PlanningError("template conversions must use every explicit input binding")
    report = dry.finalize()
    if final_window:
        k = CYCLE_COUNT - 1
        w = plan.window(k)
        prior_events = int(w.hold_event[0])
        final_cursor = _PlanCursor(float(shift(k, 0.0)), windows=k, last_event=prior_events - 1,
                                   events=prior_events, fragments=prior_events)
        normalized, fields, following = StreamingArrayAcquisition._prepare(fake, plan.legacy_window(k, w),
                                                                           final_cursor)
        if following.time != RECORD_SECONDS or following.windows != CYCLE_COUNT:
            raise PlanningError("final window must close the complete record")
        digests[k] = _window_digest(normalized, fields).hex()
    return dict(window_digests=digests, maximum_window_numeric_bytes=fake._max_window_numeric_bytes,
                dry_readout_report=report, cursor_after_prefix=dict(windows=cursor.windows,
                events=cursor.events, fragments=cursor.fragments, splits=cursor.splits))


def scalar_bound_spot_check(plan, windows):
    """Compare closed-form hold bounds with the legacy scalar predicates.

    For every candidate interval of the listed windows (parents and
    children), the legacy _upper_mul field bound and the binding's
    hold_bound(a, b, midpoint) must equal the vectorized values bitwise.
    """
    from raw_world_time import _upper_mul
    from raw_input_hold import _midpoint
    checked = mismatches = 0
    for k in windows:
        w = plan.window(k)
        starts, ends = w.action_start.tolist(), w.action_end.tolist()
        noise_cells, noise_bounds = plan.noise_bound.cells(w.action_start, w.action_end)
        hold_index = 0
        for slot, (a, b) in enumerate(zip(starts, ends)):
            pending = [(a, b)]
            while pending:
                left, right = pending.pop()
                middle, radius = _midpoint(left, right)
                error = _upper_mul(radius, plan.lipschitz)
                scaled = plan.noise.hold_bound(left, right, middle)
                checked += 1
                if scaled != float(noise_bounds[slot]) or float_bits(scaled) != float_bits(noise_bounds[slot]):
                    mismatches += 1
                if error <= plan.epsilon and scaled <= 1.:
                    if (w.hold_start[hold_index] != left or w.hold_end[hold_index] != right
                            or float_bits(w.hold_key[hold_index]) != float_bits(middle)
                            or float_bits(w.hold_field_bound[hold_index]) != float_bits(error)
                            or float_bits(w.hold_noise_bound[hold_index]) != float_bits(scaled)):
                        mismatches += 1
                    hold_index += 1
                else:
                    pending.extend(((middle, right), (left, middle)))
        if hold_index != len(w.hold_start):
            mismatches += 1
        clock_cells = [plan.noise._cell_for_key(float(key)) for key in w.hold_key[:: max(1, len(w.hold_key) // 64)]]
        for (cs, ce, co), key in zip(clock_cells, w.hold_key[:: max(1, len(w.hold_key) // 64)]):
            slot = int(np.searchsorted(w.action_end, key, side="right"))
            checked += 1
            if (cs, ce) != (starts[slot], ends[slot]) or float_bits(co) != float_bits(noise_cells[slot]):
                mismatches += 1
    return dict(checked=checked, mismatches=mismatches, windows=list(windows))
