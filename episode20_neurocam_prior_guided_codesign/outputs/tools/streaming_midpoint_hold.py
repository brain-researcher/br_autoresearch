"""Replay midpoint holds without retaining a full physical-action/hold tuple.

Only raw breakpoints, one physical action and a refinement stack are retained.
StreamingArrayAcquisition still owns exact window/field preflight and replay
checks. No device recipe, numerical tolerance or storage-dependent input is
inferred; the complete declared support must be covered on every pass.
"""
from dataclasses import dataclass
from types import MappingProxyType

from array_acquisition import ProviderInterval
from raw_input_hold import (
    HoldResolutionError, PhysicalAction, _MidpointProvider, _finite,
    _midpoint, _positive_integer,
)
from raw_world_time import RawWorldTime, _upper_mul


@dataclass(frozen=True, slots=True)
class StreamingMidpointHoldPlan:
    start: float
    end: float
    interval_factory: object
    field_provider: object
    report: object


def compile_streaming_midpoint_hold(raw_time, contact_factors, physical_action_factory, *,
        support, epsilon_input_volts, max_intervals, max_work, check, noise_hold_bound=None):
    """Exhaust one bounded preparation pass, then replay identical hold rules.

    Interval/work limits apply to EACH planning pass. The caller's resource
    callback runs in every pass as well. Preparation retains no intervals or
    voltages. Later passes must match its support and bound/count summary;
    exact physical content is separately bound by the acquisition executor's
    existing preflight and per-window replay digests.
    """
    if not isinstance(raw_time, RawWorldTime):
        raise ValueError("explicit validated RawWorldTime required")
    if not callable(physical_action_factory) or not callable(check):
        raise ValueError("replayable physical factory and resource check required")
    if noise_hold_bound is not None and not callable(noise_hold_bound):
        raise ValueError("noise_hold_bound must be an explicit callable or None")
    if not isinstance(support, (tuple, list)) or len(support) != 2:
        raise ValueError("explicit two-endpoint support required")
    start, end = (_finite(value, "physical support") for value in support)
    if not 0 <= start < end <= 16.:
        raise ValueError("physical support must lie inside [0,16]")
    epsilon = _finite(epsilon_input_volts, "epsilon_input_volts")
    if epsilon <= 0:
        raise ValueError("epsilon_input_volts must be positive")
    maximum = _positive_integer(max_intervals, "max_intervals")
    work_limit = _positive_integer(max_work, "max_work")
    raw_edges = [start]
    for value in raw_time.breakpoints():
        check()
        edge = _finite(value, "raw breakpoint")
        if start < edge < end:
            if edge <= raw_edges[-1]:
                raise ValueError("raw breakpoints must be strictly increasing")
            if len(raw_edges) >= maximum:
                raise HoldResolutionError("raw support segments exceed max_intervals")
            raw_edges.append(edge)
    raw_edges = tuple(raw_edges + [end])
    bound_contact = raw_time.bind_contact_factors(contact_factors)
    first_report = {}

    def scan(expected):
        work = candidates = enclosures = action_count = atomic_count = interval_count = 0
        maximum_error = 0.
        maximum_noise_error = 0.
        expected_start, reference = start, None
        raw_index, bounded_raw_index, lipschitz = 0, -1, None

        def spend():
            nonlocal work
            check()
            if work >= work_limit:
                raise HoldResolutionError("max_work reached before streaming hold pass completed")
            work += 1

        for action in physical_action_factory():
            check()
            if not isinstance(action, PhysicalAction):
                raise ValueError("physical factory must yield PhysicalAction records")
            action_count += 1
            if action_count > maximum:
                raise HoldResolutionError("physical actions exceed max_intervals")
            if action.start != expected_start or action.end > end:
                raise ValueError("physical factory must exactly cover declared support")
            if reference is None:
                reference = action.voltage_reference_id
            elif action.voltage_reference_id != reference:
                raise ValueError("one physical voltage reference is required")
            left = action.start
            while left < action.end:
                check()
                while left >= raw_edges[raw_index + 1]:
                    raw_index += 1
                right = min(action.end, raw_edges[raw_index + 1])
                atomic_count += 1
                if atomic_count > maximum:
                    raise HoldResolutionError("raw/physical boundary union exceeds max_intervals")
                if bounded_raw_index != raw_index:
                    spend()
                    lipschitz = bound_contact.derivative_bound_volts_per_second(
                        raw_edges[raw_index], raw_edges[raw_index + 1])
                    bounded_raw_index = raw_index
                    enclosures += 1
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
                        if interval_count >= maximum:
                            raise HoldResolutionError("streaming hold exceeds max_intervals")
                        interval_count += 1
                        maximum_error = max(maximum_error, error)
                        maximum_noise_error = max(maximum_noise_error, noise_error)
                        charge = action.injected_charge_coulomb if a == action.start else ()
                        yield ProviderInterval(a, b, action.gate_row, midpoint,
                            action.voltage_reference_id, action.daq_selected_sources, charge)
                    else:
                        if interval_count + len(pending) + 2 > maximum:
                            raise HoldResolutionError("streaming refinement exceeds max_intervals")
                        pending.extend(((midpoint, b), (a, midpoint)))
                left = right
            expected_start = action.end
        check()
        if not action_count or expected_start != end:
            raise ValueError("physical factory ended before the complete declared support")
        summary = dict(
            world_id=raw_time.metadata["world_id"], scope="internal_simulator_only",
            hold_convention="midpoint_constant_input_on_each_half_open_interval",
            support_start=start, support_end=end, voltage_reference_id=reference,
            epsilon_input_volts=epsilon, maximum_analytic_hold_bound_volts=maximum_error,
            original_physical_actions=action_count, required_atomic_intervals=atomic_count,
            compiled_intervals=interval_count, max_intervals=maximum,
            work_units=work, max_work=work_limit, work_limit_scope="each_complete_planning_pass",
            candidate_bound_evaluations=candidates, raw_segment_derivative_enclosures=enclosures,
            retained_raw_boundary_count=len(raw_edges),
            retained_raw_boundary_numeric_bytes=8 * len(raw_edges),
            retained_numeric_bytes_exclude_python_objects_and_contact_factors=True,
            bound_covers="analytic_midpoint_hold_approximation_only",
            outward_arithmetic="each_nonnegative_binary64_multiply_add_divide_and_radius_rounded_up",
            waveform_evaluation_arithmetic_error_included=False, circuit_output_error_included=False,
            nuisances_applied=False, acquisition_policy_receives_future_fields_or_splits=False,
            actual_DAQ_or_electrical_parameters_selected=False, electrical_state_advanced=False,
            dense_voltage_plan_materialized=False, full_physical_action_or_hold_tuple_retained=False)
        if noise_hold_bound is not None:
            summary.update(physical_noise_hold_bound_applied=True,
                maximum_scaled_physical_noise_hold_bound=maximum_noise_error,
                noise_bound_uses_same_midpoint_key=True)
        if expected is None:
            first_report.update(summary)
        elif summary != expected:
            raise ValueError("replayed physical hold support or bound/count summary changed")

    # Late malformed actions refuse here, before the caller constructs an array.
    for _ in scan(None):
        pass
    report = MappingProxyType(first_report)
    return StreamingMidpointHoldPlan(start, end, lambda: scan(report),
                                     _MidpointProvider(bound_contact), report)
