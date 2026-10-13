"""Array-only implementation of component_details.md section 1.1.

This checks return support at one caller-authenticated WAY start. It does not
read source data, convert units, derive motion, fit calibration, detect cues,
arm a risk state, or produce onset labels. Caller declarations describe facts
already established by that start; this module cannot authenticate them.
Passing the predicate does not qualify the online-risk detector or timing bound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NATIVE_RATE_HZ = 500
RETURN_TICKS = 250
DIGIT_CLEARANCE_M = 0.15


@dataclass(frozen=True)
class ReturnAdmissions:
    """Caller-established facts, with absent facts explicitly unknown.

    ``physical_units_admitted`` includes the source conversions/calibration for
    position in metres, force in newtons, and the declared object-motion unit;
    unit labels alone are insufficient. ``backward_motion_admitted`` says the
    supplied motion has only backward causal support. The primary summaries
    retain the existing five-EMG/three-hand-path order and threshold units.
    ``calibration_admitted`` says every reference, tolerance and low threshold
    was fixed from permitted development calibration, or from already elapsed
    admitted calibration, and was available by the evaluated start.
    """

    position_unit: str | None = None
    force_unit: str | None = None
    object_motion_unit: str | None = None
    physical_units_admitted: bool = False
    backward_motion_admitted: bool = False
    primary_summaries_causal: bool = False
    calibration_admitted: bool = False


@dataclass(frozen=True)
class FrozenReturnReferences:
    """Supplied coordinate references/tolerances and all eight low thresholds.

    Object fields have shape (3,), force fields (2, 3), and ``primary_low``
    (8,). Motion references and tolerances use ``object_motion_unit`` exactly;
    this module does not choose a derivative, filter, or motion scale. Arrays
    are copied and made read-only. None/nonfinite values leave support unknown;
    malformed shapes and negative tolerances are programming errors.
    """

    object_position_m: object
    object_position_tolerance_m: object
    object_motion: object
    object_motion_tolerance: object
    object_motion_unit: str | None
    plate_force_n: object
    plate_force_tolerance_n: object
    primary_low: object

    def __post_init__(self):
        for name in (
            "object_position_m", "object_position_tolerance_m",
            "object_motion", "object_motion_tolerance", "plate_force_n",
            "plate_force_tolerance_n", "primary_low",
        ):
            value = getattr(self, name)
            if value is not None:
                frozen = np.array(value, copy=True)
                frozen.flags.writeable = False
                object.__setattr__(self, name, frozen)


@dataclass(frozen=True)
class StartContext:
    """Known context at the start, in native ticks of one series.

    ``phase`` is 'initial_series', 'after_reach', or unknown (None). Initial
    entry means a known absence of prior reaches in this series, not a default
    assigned after a gap. ``phase_history_known`` must be established by the
    caller after resets. ``continuous_history_start_tick`` is the first tick
    after the latest gap/reset; it cannot precede ``series_start_tick``.
    After a reach, an authenticated end must follow that reach, precede the
    entire return window, and belong to the current uninterrupted history.
    """

    start_tick: int | None
    start_authenticated: bool = False
    phase: str | None = None
    phase_history_known: bool = False
    series_start_tick: int | None = None
    continuous_history_start_tick: int | None = None
    prior_reach_tick: int | None = None
    end_tick: int | None = None
    end_authenticated: bool = False


@dataclass(frozen=True)
class ReturnSupport:
    """Predicate evidence only; 'unknown' must never become a negative label."""

    status: str
    reasons: tuple[str, ...]
    interval_start_tick: int | None
    interval_stop_tick: int | None


def _tick(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer native tick")
    return int(value)


def _true(value):
    return isinstance(value, (bool, np.bool_)) and bool(value)


def evaluate_completed_return(
    *, evaluation_tick, first_tick, positions_m, object_motion, plate_forces_n,
    primary_summaries, primary_history_valid, object_motion_history_valid,
    sample_valid, admissions, references, context, sample_rate_hz=500,
):
    """Check exactly [start_tick - 250, start_tick) on the native 500-Hz grid.

    Each supplied array starts at ``first_tick``. Shapes are (N, 4, 3) for P1,
    P2, P3, P4 XYZ positions, (N, 3) for supplied backward object motion,
    (N, 2, 3) for separate plate forces, and (N, 8) for primary summaries.
    Boolean history flags have shapes (N, 8), (N,), and (N,) respectively;
    ``sample_valid`` includes gaps, saturation and unavailable sensor samples.
    All upstream summary histories must be complete at every selected tick.
    Arrays must expose a shape and support slicing. Irrelevant prefix/current/
    future values are never converted or checked; different trailing lengths
    are allowed when every array covers the same required interval.

    Off-start calls are outside this predicate's domain and return without
    reading sensor arrays: 'unknown' before the start, 'not_supported' after
    it. At the start, missing/nonfinite prerequisites take 'unknown' priority
    over an observed failure. There is no later accumulation, retroactive
    restoration, or persistent arming state here.
    """
    evaluation_tick = _tick(evaluation_tick, "evaluation_tick")
    first_tick = _tick(first_tick, "first_tick")
    unknown, failed = [], []
    if context is None or context.start_tick is None:
        return ReturnSupport("unknown", ("missing_start_context",), None, None)
    stop = _tick(context.start_tick, "start_tick")
    begin = stop - RETURN_TICKS
    left, right = begin - first_tick, stop - first_tick
    if evaluation_tick < stop:
        # The claimed start is not yet available; do not inspect its future
        # support interval even if the caller supplied a longer array.
        return ReturnSupport("unknown", ("start_not_yet_available",), begin, stop)
    if evaluation_tick > stop:
        return ReturnSupport("not_supported", ("evaluation_not_at_start",), begin, stop)

    if sample_rate_hz != NATIVE_RATE_HZ:
        unknown.append("native_500_hz_grid_unavailable")
    if not _true(context.start_authenticated):
        unknown.append("start_cue_not_authenticated")
    if not _true(context.phase_history_known) or context.phase not in ("initial_series", "after_reach"):
        unknown.append("phase_history_unknown")

    series_begin = history_begin = None
    if context.series_start_tick is None or context.continuous_history_start_tick is None:
        unknown.append("series_or_gap_history_unknown")
    else:
        series_begin = _tick(context.series_start_tick, "series_start_tick")
        history_begin = _tick(context.continuous_history_start_tick, "continuous_history_start_tick")
        if history_begin < series_begin or begin < history_begin:
            unknown.append("return_window_crosses_series_or_gap_boundary")
    if context.phase == "initial_series" and context.prior_reach_tick is not None:
        unknown.append("initial_series_conflicts_with_prior_reach")
    if context.phase == "after_reach":
        if context.prior_reach_tick is None or context.end_tick is None or not _true(context.end_authenticated):
            unknown.append("prior_reach_or_authenticated_end_unknown")
        else:
            reach = _tick(context.prior_reach_tick, "prior_reach_tick")
            end = _tick(context.end_tick, "end_tick")
            if series_begin is not None and reach < series_begin:
                unknown.append("prior_reach_outside_current_series")
            if history_begin is not None and end < history_begin:
                unknown.append("end_precedes_current_uninterrupted_history")
            if not reach < end < begin:
                failed.append("end_must_follow_reach_and_precede_entire_window")

    if admissions is None:
        unknown.append("caller_facts_missing")
    else:
        if admissions.position_unit != "m" or admissions.force_unit != "N" or not _true(admissions.physical_units_admitted):
            unknown.append("physical_units_not_admitted")
        if not _true(admissions.backward_motion_admitted):
            unknown.append("backward_object_motion_not_admitted")
        if not _true(admissions.primary_summaries_causal):
            unknown.append("primary_summary_causality_unknown")
        if not _true(admissions.calibration_admitted):
            unknown.append("reference_calibration_not_admitted_by_start")
        if not isinstance(admissions.object_motion_unit, str) or not admissions.object_motion_unit.strip():
            unknown.append("object_motion_unit_missing")

    def reference(name, shape, tolerance=False):
        value = getattr(references, name)
        if value is None:
            unknown.append(name + "_missing")
            return None
        if np.shape(value) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        try:
            value = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError):
            unknown.append(name + "_unavailable")
            return None
        if not np.all(np.isfinite(value)):
            unknown.append(name + "_nonfinite")
            return None
        if tolerance and np.any(value < 0):
            raise ValueError(f"{name} must be nonnegative")
        return value

    prepared_refs = {}
    if references is None:
        unknown.append("return_references_missing")
    else:
        if (not isinstance(references.object_motion_unit, str)
                or not references.object_motion_unit.strip()
                or admissions is None
                or references.object_motion_unit != admissions.object_motion_unit):
            unknown.append("object_motion_reference_unit_unresolved")
        for name, shape, tolerance in (
            ("object_position_m", (3,), False),
            ("object_position_tolerance_m", (3,), True),
            ("object_motion", (3,), False),
            ("object_motion_tolerance", (3,), True),
            ("plate_force_n", (2, 3), False),
            ("plate_force_tolerance_n", (2, 3), True),
            ("primary_low", (8,), False),
        ):
            prepared_refs[name] = reference(name, shape, tolerance)

    def window(values, tail_shape, name, *, flags=False):
        if values is None:
            unknown.append(name + "_missing")
            return None
        shape = getattr(values, "shape", None)
        if shape is None or len(shape) != 1 + len(tail_shape) or tuple(shape[1:]) != tail_shape:
            raise ValueError(f"{name} must have shape (N, {tail_shape})")
        if left < 0 or right > shape[0]:
            unknown.append(name + "_incomplete_return_window")
            return None
        # Slice BEFORE conversion: even nonnumeric future values are irrelevant.
        selected = values[left:right]
        if flags:
            selected = np.asarray(selected)
            if selected.dtype.kind != "b" or not np.all(selected):
                unknown.append(name + "_incomplete_or_unknown")
                return None
        else:
            try:
                selected = np.asarray(selected, dtype=np.float64)
            except (TypeError, ValueError):
                unknown.append(name + "_unavailable")
                return None
            if not np.all(np.isfinite(selected)):
                unknown.append(name + "_nonfinite")
                return None
        return selected

    positions = window(positions_m, (4, 3), "positions_m")
    motion = window(object_motion, (3,), "object_motion")
    forces = window(plate_forces_n, (2, 3), "plate_forces_n")
    summaries = window(primary_summaries, (8,), "primary_summaries")
    window(primary_history_valid, (8,), "primary_history_valid", flags=True)
    window(object_motion_history_valid, (), "object_motion_history_valid", flags=True)
    window(sample_valid, (), "sample_valid", flags=True)

    if unknown:
        return ReturnSupport("unknown", tuple(unknown), begin, stop)
    ref = prepared_refs
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.hypot.reduce(positions[:, 1:3] - positions[:, 0:1], axis=2)
        position_departure = np.abs(positions[:, 0] - ref["object_position_m"])
        motion_departure = np.abs(motion - ref["object_motion"])
        force_departure = np.abs(forces - ref["plate_force_n"])
    if any(not np.all(np.isfinite(values)) for values in (
        distances, position_departure, motion_departure, force_departure,
    )):
        return ReturnSupport("unknown", ("nonfinite_computed_return_condition",), begin, stop)
    for digit, name in enumerate(("index", "thumb")):
        if np.any(distances[:, digit] < DIGIT_CLEARANCE_M):
            failed.append(name + "_object_clearance_below_0_15_m")
    if np.any(summaries > ref["primary_low"]):
        failed.append("primary_path_exceeds_low_threshold")
    if np.any(position_departure > ref["object_position_tolerance_m"]):
        failed.append("object_position_outside_supported_reference")
    if np.any(motion_departure > ref["object_motion_tolerance"]):
        failed.append("object_motion_outside_supported_reference")
    for plate in range(2):
        if np.any(force_departure[:, plate] > ref["plate_force_tolerance_n"][plate]):
            failed.append(f"plate_{plate + 1}_force_outside_released_reference")
    return ReturnSupport("not_supported" if failed else "supported", tuple(failed), begin, stop)
