"""Compact midpoint-input approximation for EP20's internal simulator only.

The caller supplies actual physical actions, aperture/DAQ boundaries, epsilon
and finite work limits. No acquisition recipe, physical parameter, new world,
policy feature or tolerated error is inferred. The analytic hold bound excludes
waveform evaluation arithmetic and electrical-output error. No voltage plan is
materialized: ArrayAcquisition will prevalidate and digest each provider field
before advancing its one persistent electrical/readout state.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType

import numpy as np

from array_acquisition import ProviderInterval
from daq_mux_hold import DAQGroupDrive
from raw_world_time import RawWorldTime, _upper_mul, _upper_result


class HoldResolutionError(ValueError):
    """Selected tolerance cannot be met under explicit finite planning limits."""


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    return float(value)


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be an explicit positive integer")
    return value


@dataclass(frozen=True, slots=True)
class PhysicalAction:
    """Caller-owned physical timing/selection, with no field sampling time yet."""
    start: float
    end: float
    gate_row: int
    voltage_reference_id: str
    daq_selected_sources: object
    injected_charge_coulomb: object = ()

    def __post_init__(self):
        start, end = _finite(self.start, "physical start"), _finite(self.end, "physical end")
        if not 0 <= start < end <= 16.:
            raise ValueError("physical action must lie in the selected[0,16]window")
        if type(self.gate_row) is not int or not 0 <= self.gate_row < 64:
            raise ValueError("explicit gate row0..63 required")
        if not isinstance(self.voltage_reference_id, str) or not self.voltage_reference_id:
            raise ValueError("explicit voltage reference identity required")
        # Reuse the executor's immutable mapping/sparse-charge validation; the
        # temporary key0 is discarded and selects no waveform evaluation time.
        normalized = ProviderInterval(start, end, self.gate_row, 0, self.voltage_reference_id,
                                      self.daq_selected_sources, self.injected_charge_coulomb)
        if normalized.daq_selected_sources is not None:
            for key, value in normalized.daq_selected_sources.items():
                if not isinstance(key, str) or not key:
                    raise ValueError("DAQ mappings need named physical groups")
                if isinstance(value, DAQGroupDrive):
                    if value.voltage_reference_id != self.voltage_reference_id:
                        raise ValueError("External DAQ voltages must use the action's explicit reference")
                    selected = value.selected_input
                    if type(selected) is int and not 0 <= selected < 64:
                        raise ValueError("Neural DAQ selections remain source0..63")
                elif value is not None and (type(value) is not int or not 0 <= value < 64):
                    raise ValueError("DAQ selections need source0..63, None or an explicit DAQGroupDrive")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "daq_selected_sources", normalized.daq_selected_sources)
        object.__setattr__(self, "injected_charge_coulomb", normalized.injected_charge_coulomb)


class _MidpointProvider:
    def __init__(self, bound_contact):
        self._bound = bound_contact

    def __call__(self, key):
        midpoint = _finite(key, "midpoint field key")
        return self._bound.volts(np.array([midpoint]))[:, 0].reshape(64, 64)


@dataclass(frozen=True, slots=True)
class MidpointHoldPlan:
    """Internal simulator plan; never release its fields/splits to a policy."""
    intervals: tuple
    field_provider: object
    analytic_error_bounds_volts: tuple
    report: object


def _midpoint(start, end):
    midpoint = start + (end - start) / 2
    if not start < midpoint < end:
        raise HoldResolutionError("no representable interior midpoint; tolerance is not relaxed")
    radius = max(_upper_result(midpoint - start), _upper_result(end - midpoint))
    return midpoint, radius


def compile_midpoint_hold(raw_time, contact_factors, physical_actions, *, epsilon_input_volts,
                          max_intervals, max_work, additional_boundaries_s=(), check=None,
                          noise_hold_bound=None):
    """Create a finite, immutable constant-input provider plan without voltages.

The derivative bound is cached per raw support/taper/M4 segment; changing rows
or aperture splits does not recompute it. A work unit is one contact derivative
enclosure or one candidate-interval bound evaluation. Caps bound this planner,
not downstream electrical solves or total process memory. Future raw changes
and numerical refinement are internal simulator information, never additional
conversions or scheduler observations. Nuisances remain unapplied.
An optional physical-noise bound returns the maximum per-port analytic hold
error divided by that port's explicit A/V tolerance. It must describe a frozen
trajectory at this same midpoint, not redraw noise at numerical substeps.
"""
    if not isinstance(raw_time, RawWorldTime):
        raise ValueError("explicit validated RawWorldTime required")
    epsilon = _finite(epsilon_input_volts, "epsilon_input_volts")
    if epsilon <= 0:
        raise ValueError("epsilon_input_volts must be positive")
    maximum = _positive_integer(max_intervals, "max_intervals")
    work_limit = _positive_integer(max_work, "max_work")
    if check is None:
        check = lambda: None
    if not callable(check):
        raise ValueError("check must be callable")
    if noise_hold_bound is not None and not callable(noise_hold_bound):
        raise ValueError("noise_hold_bound must be an explicit callable or None")
    actions = []
    for action in physical_actions:
        check()
        if not isinstance(action, PhysicalAction):
            raise ValueError("supply compact PhysicalAction records, not eager voltage arrays")
        if len(actions) >= maximum:
            raise HoldResolutionError("original physical actions exceed max_intervals")
        if actions and (actions[-1].end != action.start
                        or actions[-1].voltage_reference_id != action.voltage_reference_id):
            raise ValueError("physical actions must be exactly contiguous with one reference")
        actions.append(action)
    if not actions:
        raise ValueError("a nonempty finite physical schedule is required")
    actions = tuple(actions)
    start, end = actions[0].start, actions[-1].end
    raw_edges = [start] + [float(t) for t in raw_time.breakpoints() if start < t < end] + [end]
    boundaries = set(raw_edges)
    boundaries.update(action.start for action in actions)
    if len(boundaries) - 1 > maximum:
        raise HoldResolutionError("required raw/physical boundaries exceed max_intervals")
    additional_count = 0
    for value in additional_boundaries_s:
        check()
        additional_count += 1
        if additional_count > maximum + 1:
            raise HoldResolutionError("supplied boundary entries exceed finite planning limit")
        value = _finite(value, "additional physical/aperture boundary")
        if not start <= value <= end:
            raise ValueError("additional boundaries must lie inside the supplied physical window")
        boundaries.add(value)
        if len(boundaries) - 1 > maximum:
            raise HoldResolutionError("required boundary union exceeds max_intervals")
    boundaries = sorted(boundaries)
    check()
    bound_contact = raw_time.bind_contact_factors(contact_factors)
    cache, intervals, errors = {}, [], []
    work, candidates, derivative_enclosures = 0, 0, 0
    maximum_noise_error = 0.

    def spend():
        nonlocal work
        check()
        if work >= work_limit:
            raise HoldResolutionError("max_work reached before tolerance-qualified plan completed")
        work += 1

    action_index = 0
    for left, right in zip(boundaries, boundaries[1:]):
        while left >= actions[action_index].end:
            action_index += 1
        action = actions[action_index]
        if right > action.end:
            raise ValueError("required physical action boundary was lost")
        raw_index = bisect_right(raw_edges, left) - 1
        if raw_index not in cache:
            spend()
            cache[raw_index] = bound_contact.derivative_bound_volts_per_second(raw_edges[raw_index], raw_edges[raw_index + 1])
            derivative_enclosures += 1
        lipschitz = cache[raw_index]
        pending = [(left, right)]
        while pending:
            spend()
            candidates += 1
            a, b = pending.pop()
            midpoint, radius = _midpoint(a, b)
            error = _upper_mul(radius, lipschitz)
            noise_error = 0.
            if noise_hold_bound is not None:
                spend()
                noise_error = _finite(noise_hold_bound(a, b, midpoint), "scaled physical-noise hold bound")
                if noise_error < 0:
                    raise ValueError("scaled physical-noise hold bound must be nonnegative")
            if error <= epsilon and noise_error <= 1.:
                if len(intervals) >= maximum:
                    raise HoldResolutionError("max_intervals reached before tolerance-qualified plan completed")
                # The physical boundary's explicit charge belongs only to the
                # first numerical child, never repeated at later refinements.
                charge = action.injected_charge_coulomb if a == action.start else ()
                intervals.append(ProviderInterval(a, b, action.gate_row, midpoint,
                    action.voltage_reference_id, action.daq_selected_sources, charge))
                errors.append(error)
                maximum_noise_error = max(maximum_noise_error, noise_error)
            else:
                if len(intervals) + len(pending) + 2 > maximum:
                    raise HoldResolutionError("required bisection exceeds max_intervals")
                # LIFO right-then-left gives deterministic chronological output.
                pending.extend(((midpoint, b), (a, midpoint)))
    check()
    report = MappingProxyType({
        "world_id": raw_time.metadata["world_id"], "scope": "internal_simulator_only",
        "hold_convention": "midpoint_constant_input_on_each_half_open_interval",
        "epsilon_input_volts": epsilon, "maximum_analytic_hold_bound_volts": max(errors),
        "original_physical_actions": len(actions), "required_atomic_intervals": len(boundaries) - 1,
        "compiled_intervals": len(intervals), "max_intervals": maximum,
        "work_units": work, "max_work": work_limit, "candidate_bound_evaluations": candidates,
        "cached_raw_segment_derivative_enclosures": derivative_enclosures,
        "bound_covers": "analytic_midpoint_hold_approximation_only",
        "outward_arithmetic": "each_nonnegative_binary64_multiply_add_divide_and_radius_rounded_up",
        "waveform_evaluation_arithmetic_error_included": False, "circuit_output_error_included": False,
        "nuisances_applied": False, "acquisition_policy_receives_future_fields_or_splits": False,
        "actual_DAQ_or_electrical_parameters_selected": False, "electrical_state_advanced": False,
        "dense_voltage_plan_materialized": False,
        **({"physical_noise_hold_bound_applied": True,
            "maximum_scaled_physical_noise_hold_bound": maximum_noise_error,
            "noise_bound_uses_same_midpoint_key": True}
           if noise_hold_bound is not None else {})})
    return MidpointHoldPlan(tuple(intervals), _MidpointProvider(bound_contact), tuple(errors), report)
