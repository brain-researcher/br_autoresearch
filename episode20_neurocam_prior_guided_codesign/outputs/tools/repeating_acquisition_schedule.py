"""Repeat one explicit P-full acquisition cycle without whole-record plans.

The caller supplies the actual conditional recipe: all mux/hold actions,
external voltages, charge, raw apertures/conversions, returned-word mapping,
quantizers, observation arithmetic, bits and latency. The published source
stream proxy is checked separately from declared physical conversions. This
builder neither reconstructs NI clocks nor authenticates source labels/wiring.

One 4-ms template contains the complete 64-row scan. Its physical state is NOT
periodic or reset: only the prescribed controls/readout metadata repeat across
the full 16-s record. Conversion and digital dependencies must be contained in
the template; delayed publication can cross cycle boundaries. Cross-cycle
filter references are explicitly unsupported, never truncated or substituted.
"""

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, replace
import math
from types import MappingProxyType

from acquisition_readout import QuantizerTransfer
from array_acquisition import ArrayInputBinding, _daq_external_numeric_bytes
from daq_mux_hold import PassiveDAQFrontend
from observation_nuisance import ObservationNuisanceBinding
from raw_input_hold import PhysicalAction
from streaming_acquisition_readout import ReadoutWindow


CYCLE_SECONDS = 0.004
ROW_DWELL_SECONDS = 0.0000625
RECORD_SECONDS = 16.0
CYCLE_COUNT = 4000
SOURCE_WORDS_PER_CYCLE = 400
_ROW_EDGES = tuple(row / 16000 for row in range(65))


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    return float(value)


def _local_id(value, prefix, count):
    if not isinstance(value, str) or not value.startswith(prefix + ":"):
        raise ValueError("Template dependencies require local canonical record IDs")
    suffix = value[len(prefix) + 1:]
    if not suffix.isdecimal() or str(int(suffix)) != suffix or not 0 <= int(suffix) < count:
        raise ValueError("Cross-cycle, noncanonical or missing template dependency is unsupported")
    return int(suffix)


def _shift(cycle, relative):
    # Every shared cycle boundary has exactly one binary64 representation.
    # No acquisition or publication timestamp is clamped to the record end.
    if relative == 0.0:
        return cycle * CYCLE_SECONDS
    if relative == CYCLE_SECONDS:
        return (cycle + 1) * CYCLE_SECONDS
    return cycle * CYCLE_SECONDS + relative


def _last_window(available_at):
    # First physical window whose end reaches this absolute deadline. The
    # last-use INDEX can stay at the last physical window for a later digital
    # drain; the timestamp itself is preserved in full.
    left, right = 0, CYCLE_COUNT - 1
    while left < right:
        middle = (left + right) // 2
        if (middle + 1) * CYCLE_SECONDS < available_at:
            left = middle + 1
        else:
            right = middle
    return left


@dataclass(frozen=True, slots=True)
class ReturnedInputBinding:
    """Caller-labelled source/external input and its acquisition point.

    The label is not inferred from instantaneous mux selection: a held voltage
    may be converted while all switches are off. Membership and raw dependency
    ADCs are checked; the supplied label is not authenticated wiring evidence.
    """
    sample_id: str
    input_label: int | str
    acquisition_time_s: float

    def __post_init__(self):
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("A returned binding requires its explicit sample ID")
        label = self.input_label
        if not ((type(label) is int and 0 <= label < 64)
                or isinstance(label, str) and bool(label.strip())):
            raise ValueError("A returned input is a neural source0..63 or named external port")
        point = _finite(self.acquisition_time_s, "acquisition point")
        if not 0 <= point < CYCLE_SECONDS:
            raise ValueError("The acquisition point must lie inside the template's half-open cycle")
        object.__setattr__(self, "acquisition_time_s", point)


@dataclass(frozen=True, slots=True)
class RepeatingAcquisitionSchedule:
    """Owned finite template plus lazy factories; no electrical execution."""
    _actions: tuple
    _readout: ReadoutWindow
    _sample_lifetimes: tuple
    _raw_lifetimes: tuple
    input_bindings: tuple
    returned_input_bindings: tuple
    metadata: object
    final_publication_horizon: float
    _check: object
    acquisition_window: tuple = (0.0, RECORD_SECONDS)

    def _cycle(self, index):
        if type(index) is not int or not 0 <= index < CYCLE_COUNT:
            raise ValueError("A cycle index must lie in the full16s record")
        return index

    def physical_actions_for_cycle(self, index):
        index = self._cycle(index)
        for action in self._actions:
            self._check()
            yield replace(action, start=_shift(index, action.start), end=_shift(index, action.end))

    def physical_action_factory(self):
        for index in range(CYCLE_COUNT):
            yield from self.physical_actions_for_cycle(index)

    def readout_window(self, index):
        index = self._cycle(index)
        self._check()
        template = self._readout
        nr, ns, no = len(template.conversions), len(template.returned_samples), len(template.observations)
        def identity(value, prefix, count):
            return f"{prefix}:{index * count + _local_id(value, prefix, count)}"
        raw, samples, observations, nuisance = [], [], [], []
        for plan in template.conversions:
            self._check()
            raw.append(replace(plan, conversion_id=identity(plan.conversion_id, "r", nr),
                aperture_start=_shift(index, plan.aperture_start), aperture_end=_shift(index, plan.aperture_end),
                completed_at=_shift(index, plan.completed_at), assigned_timestamp=_shift(index, plan.assigned_timestamp)))
        for plan in template.returned_samples:
            self._check()
            samples.append(replace(plan, sample_id=identity(plan.sample_id, "s", ns),
                conversion_ids=tuple(identity(key, "r", nr) for key in plan.conversion_ids),
                assigned_timestamp=_shift(index, plan.assigned_timestamp), available_at=_shift(index, plan.available_at)))
        for plan in template.observations:
            self._check()
            observations.append(replace(plan, observation_id=identity(plan.observation_id, "o", no),
                sample_ids=tuple(identity(key, "s", ns) for key in plan.sample_ids),
                assigned_timestamp=_shift(index, plan.assigned_timestamp), available_at=_shift(index, plan.available_at)))
        for binding in template.nuisance_bindings:
            nuisance.append(ObservationNuisanceBinding(identity(binding.observation_id, "o", no),
                binding.physical_site_index, _shift(index, binding.acquisition_time_s)))
        raw_last = tuple((plan.conversion_id, _last_window(_shift(index, deadline)))
                         for plan, deadline in zip(raw, self._raw_lifetimes))
        sample_last = tuple((plan.sample_id, _last_window(_shift(index, deadline)))
                            for plan, deadline in zip(samples, self._sample_lifetimes))
        return ReadoutWindow(index, index * CYCLE_SECONDS, (index + 1) * CYCLE_SECONDS,
                             tuple(raw), tuple(samples), tuple(observations), raw_last, sample_last,
                             tuple(nuisance))

    def readout_window_factory(self):
        for index in range(CYCLE_COUNT):
            yield self.readout_window(index)


def build_full_record_schedule(*, frontend, cycle_actions, cycle_readout, input_bindings,
        returned_input_bindings, marker_input_id, voltage_reference_id,
        max_template_actions, max_template_records, max_template_numeric_bytes, check):
    """Bind a complete explicit4ms P-full template and repeat it4000 times.

    No defaults select an electrical or DAQ parameter. One pixel observation
    per physical site/cycle and400 caller-labelled source words/source/cycle
    preserve the full-row/source-stream scope. Observation weights, assigned
    times and delays are copied verbatim. The caller must encode its selected
    third-through-fifth aggregation and any supported digital recipe explicitly.

    Template dependencies are local: a returned sample uses this cycle's raw
    records and an observation uses this cycle's returned records. A dependency
    on a prior/future cycle refuses before any factory is returned. Only
    publication/lifetime may extend beyond the cycle or acquisition end.

    Numeric limits cover retained schedule payload, not Python object overhead,
    frontend parameters, the live emitted window or solver/output memory.
    The parent streaming runner retains its own live-window/process bounds.
    """
    for name, value in (("max_template_actions", max_template_actions),
                        ("max_template_records", max_template_records),
                        ("max_template_numeric_bytes", max_template_numeric_bytes)):
        _positive_integer(value, name)
    if not callable(check):
        raise ValueError("An explicit planning/deadline check is required")
    if not isinstance(frontend, PassiveDAQFrontend) or len(frontend.groups) != 24:
        raise ValueError("The full-record template requires exactly24 supplied physical ADC groups")
    if not isinstance(marker_input_id, str) or not marker_input_id.strip():
        raise ValueError("The G1 marker needs an explicit named external input")
    if not isinstance(voltage_reference_id, str) or not voltage_reference_id:
        raise ValueError("An explicit common voltage reference is required")
    frontend.validate_source_count(64)
    source_groups, external_groups = {}, {}
    for group in frontend.groups:
        if len(group.source_indices) + len(group.external_inputs) > 4:
            raise ValueError("A supplied group exceeds the fixed four-input ADC group family")
        for source in group.source_indices:
            if source in source_groups:
                raise ValueError("Each of the64 source streams needs one explicit physical ADC assignment")
            source_groups[source] = group.adc_id
        for port in group.external_inputs:
            external_groups.setdefault(port.input_id, set()).add(group.adc_id)
    if set(source_groups) != set(range(64)) or marker_input_id not in external_groups:
        raise ValueError("The supplied frontend must cover all64 neural sources and the G1 marker port")

    actions, numeric = [], 0
    for supplied in cycle_actions:
        check()
        if not isinstance(supplied, PhysicalAction) or len(actions) >= max_template_actions:
            raise ValueError("The finite cycle action template exceeds its bound or has an invalid record")
        action = replace(supplied)
        if (action.start != (actions[-1].end if actions else 0.0) or action.end > CYCLE_SECONDS
                or action.voltage_reference_id != voltage_reference_id):
            raise ValueError("Template actions must exactly cover one4ms cycle using the common reference")
        row = bisect_right(_ROW_EDGES, action.start) - 1
        if not 0 <= row < 64 or action.gate_row != row or action.end > _ROW_EDGES[row + 1]:
            raise ValueError("P-full actions must retain all64 ascending62.5us row dwells")
        frontend.resistor_edges(selected_sources=action.daq_selected_sources, source_node_offset=4096,
                                 first_hold_state=4160)
        frontend.external_drives(selected_sources=action.daq_selected_sources, first_hold_state=4160,
                                 voltage_reference_id=voltage_reference_id)
        numeric += 24 + 16 * len(action.injected_charge_coulomb) + _daq_external_numeric_bytes(action.daq_selected_sources)
        if numeric > max_template_numeric_bytes:
            raise ValueError("Retained template numeric-byte bound exceeded")
        actions.append(action)
    if not actions or actions[-1].end != CYCLE_SECONDS:
        raise ValueError("The P-full physical template must finish its complete4ms cycle")
    if (not isinstance(cycle_readout, ReadoutWindow) or cycle_readout.index != 0
            or cycle_readout.start != 0.0 or cycle_readout.end != CYCLE_SECONDS):
        raise ValueError("Supply one explicit cycle-zero4ms readout template")
    counts = tuple(len(getattr(cycle_readout, name)) for name in
                   ("conversions", "returned_samples", "observations"))
    if not all(counts) or sum(counts) > max_template_records:
        raise ValueError("Finite template record bound exceeded or a required record category is empty")
    template = replace(cycle_readout)
    raw, samples, observations = template.conversions, template.returned_samples, template.observations
    nr, ns, no = counts
    for plans, key, prefix in ((raw, "conversion_id", "r"), (samples, "sample_id", "s"),
                               (observations, "observation_id", "o")):
        if any(getattr(plan, key) != f"{prefix}:{index}" for index, plan in enumerate(plans)):
            raise ValueError("Each template record category needs sequential local canonical IDs")
    raw_input_ids = {plan.input_id for plan in raw}
    bindings = []
    for binding in input_bindings:
        check()
        if len(bindings) >= len(raw_input_ids):
            raise ValueError("Input bindings exceed the finite set of raw input IDs")
        bindings.append(binding)
    bindings = tuple(bindings)
    if (any(not isinstance(binding, ArrayInputBinding) or binding.kind != "hold" for binding in bindings)
            or len({binding.input_id for binding in bindings}) != len(bindings)
            or {binding.input_id for binding in bindings} != raw_input_ids):
        raise ValueError("Raw inputs need exact unique bindings to their existing physical ADC hold nodes")
    binding_map = {binding.input_id: binding for binding in bindings}
    adc_ids = {group.adc_id for group in frontend.groups}
    for plan in raw:
        check()
        if (binding_map[plan.input_id].adc_id != plan.adc_id or plan.adc_id not in adc_ids
                or not 0 <= plan.aperture_start < plan.aperture_end <= plan.completed_at <= CYCLE_SECONDS):
            raise ValueError("Raw conversions must use their supplied physical ADC and remain inside the cycle")

    origins = []
    for origin in returned_input_bindings:
        check()
        if not isinstance(origin, ReturnedInputBinding) or len(origins) >= ns:
            raise ValueError("Supply one finite explicit input binding per returned word")
        origins.append(replace(origin))
    if len(origins) != ns or {origin.sample_id for origin in origins} != {plan.sample_id for plan in samples}:
        raise ValueError("Returned input bindings must exactly and uniquely cover all returned words")
    origin_map = {origin.sample_id: origin for origin in origins}
    origins = tuple(origin_map[plan.sample_id] for plan in samples)
    source_counts, external_counts = Counter(), Counter()
    sample_raw_indices = []
    for plan, origin in zip(samples, origins):
        check()
        dependencies = tuple(_local_id(key, "r", nr) for key in plan.conversion_ids)
        sample_raw_indices.append(dependencies)
        allowed_adcs = ({source_groups[origin.input_label]} if type(origin.input_label) is int
                        else external_groups.get(origin.input_label, set()))
        if (any(raw[index].adc_id not in allowed_adcs for index in dependencies)
                or not any(raw[index].aperture_start <= origin.acquisition_time_s <= raw[index].aperture_end
                           for index in dependencies)
                or plan.available_at <= 0 or plan.available_at < max(raw[index].completed_at for index in dependencies)):
            raise ValueError("Returned source label, physical ADC, aperture support or availability is inconsistent")
        if origin.input_label == marker_input_id and any(raw[index].role != "marker" for index in dependencies):
            raise ValueError("G1 returned words must retain marker-role physical conversion accounting")
        (source_counts if type(origin.input_label) is int else external_counts)[origin.input_label] += 1
    if source_counts != Counter({source: SOURCE_WORDS_PER_CYCLE for source in range(64)}):
        raise ValueError("The100kS/s/source proxy requires400 explicit returned words for each of64 sources per4ms")
    if external_counts[marker_input_id] < 1:
        raise ValueError("The supplied G1 marker needs explicit returned words in addition to source-stream work")

    sample_lifetimes = [plan.available_at for plan in samples]
    pixel_sites, seen_sites, site_labels = {}, set(), set()
    for plan in observations:
        check()
        dependencies = tuple(_local_id(key, "s", ns) for key in plan.sample_ids)
        if plan.available_at <= 0 or plan.available_at < max(samples[index].available_at for index in dependencies):
            raise ValueError("Observation publication must follow all of its explicitly declared returned words")
        for index in dependencies:
            sample_lifetimes[index] = max(sample_lifetimes[index], plan.available_at)
        if plan.role == "pixel":
            labels = {origins[index].input_label for index in dependencies}
            rows = {bisect_right(_ROW_EDGES, origins[index].acquisition_time_s) - 1 for index in dependencies}
            if len(labels) != 1 or type(next(iter(labels))) is not int or len(rows) != 1:
                raise ValueError("A full-row pixel observation must retain one explicit source and acquired row")
            site = next(iter(rows)) * 64 + next(iter(labels))
            if site in seen_sites or plan.site_id in site_labels:
                raise ValueError("The full-row cycle must publish each physical pixel exactly once")
            pixel_sites[plan.observation_id] = site
            seen_sites.add(site)
            site_labels.add(plan.site_id)
    if seen_sites != set(range(4096)):
        raise ValueError("The cycle must preserve all4096 physical pixel observations")
    nuisance_ids = set()
    for binding in template.nuisance_bindings:
        if (binding.observation_id in nuisance_ids or binding.observation_id not in pixel_sites
                or binding.physical_site_index != pixel_sites[binding.observation_id]
                or not 0 <= binding.acquisition_time_s < CYCLE_SECONDS):
            raise ValueError("Cycle nuisance bindings must retain the explicit physical pixel and acquisition point")
        nuisance_ids.add(binding.observation_id)
    if nuisance_ids and nuisance_ids != set(pixel_sites):
        raise ValueError("Enabled nuisance bindings must cover exactly all cycle pixel observations")

    raw_lifetimes = [plan.completed_at for plan in raw]
    for deadline, dependencies in zip(sample_lifetimes, sample_raw_indices):
        for index in dependencies:
            raw_lifetimes[index] = max(raw_lifetimes[index], deadline)
    quantizers = {id(plan.quantizer): plan.quantizer for plan in raw}
    quantizers.update((id(plan.output_quantizer), plan.output_quantizer) for plan in samples
                      if plan.output_quantizer is not None)
    if any(not isinstance(value, QuantizerTransfer) for value in quantizers.values()):
        raise ValueError("Every transfer must be supplied explicitly")
    numeric += (112 * nr + 64 * ns + 64 * no + 32 * ns + 8 * (nr + ns)
                + 24 * len(template.nuisance_bindings)
                + sum(16 * len(plan.weights) for plan in (*samples, *observations))
                + sum(16 + 8 * (len(q.thresholds_volts) + len(q.code_volts)) for q in quantizers.values()))
    if numeric > max_template_numeric_bytes:
        raise ValueError("Retained template numeric-byte bound exceeded")
    latest = max(CYCLE_SECONDS, *(plan.available_at for plan in samples), *(plan.available_at for plan in observations))
    horizon = _shift(CYCLE_COUNT - 1, latest)
    metadata = MappingProxyType(dict(
        scope="caller_supplied_conditional_P_full_recipe_not_authenticated_device_wiring",
        physical_ADC_groups=24, neural_source_lines=64, physical_pixels=4096,
        row_dwell_seconds=ROW_DWELL_SECONDS, cycle_seconds=CYCLE_SECONDS, cycles=CYCLE_COUNT,
        acquisition_seconds=RECORD_SECONDS, retained_template_actions=len(actions),
        full_physical_actions=CYCLE_COUNT * len(actions), retained_template_numeric_bytes=numeric,
        retained_numeric_scope="template_payload_excludes_Python_frontend_live_window_and_solver_memory",
        physical_conversions_per_cycle=nr, full_physical_conversions=CYCLE_COUNT * nr,
        returned_source_words_per_cycle=sum(source_counts.values()),
        full_returned_source_words=CYCLE_COUNT * sum(source_counts.values()),
        source_stream_proxy_samples_per_second=64 * 100000,
        source_stream_proxy_is_physical_ADC_clock=False,
        full_returned_words=CYCLE_COUNT * ns, full_pixel_observations=CYCLE_COUNT * 4096,
        full_G1_returned_words=CYCLE_COUNT * external_counts[marker_input_id],
        raw_conversions_by_role_per_cycle=tuple(sorted(Counter(plan.role for plan in raw).items())),
        raw_conversions_by_ADC_per_cycle=tuple(sorted(Counter(plan.adc_id for plan in raw).items())),
        source_labels="explicit_caller_recipe_not_inferred_from_instantaneous_mux_selection",
        electrical_state_reset_at_cycle_seams=False, cross_cycle_filter_dependencies_supported=False,
        delayed_publication_supported=True, whole_record_conversion_or_action_tuple_materialized=False,
        actual_DAQ_configuration_qualified=False, calibrated_reference_model=False, candidate_scored=False))
    return RepeatingAcquisitionSchedule(tuple(actions), template, tuple(sample_lifetimes), tuple(raw_lifetimes),
        bindings, origins, metadata, horizon, check)
