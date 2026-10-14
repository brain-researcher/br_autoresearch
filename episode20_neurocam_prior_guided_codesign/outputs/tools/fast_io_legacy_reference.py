"""Legacy reference harness for Stage 1 O1/O2 equivalence (development tests).

Every reference value comes from unmodified legacy code: the streaming
midpoint-hold planner, bind_hold_windows, RepeatingAcquisitionSchedule
readout windows, StreamingArrayAcquisition._prepare/_splits/_window_digest
(called on a plain attribute namespace), StreamingAcquisitionReadout and the
eager AcquisitionReadout whose arithmetic the streaming readout reuses.
Nothing here advances electrical state; hold means are supplied explicitly.
"""
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np

from fast_io_exact import CYCLE_SECONDS, float_bits, shift


READOUT_LIMITS = dict(max_conversions=104000000, max_returned_samples=104000000,
    max_observations=17984000, max_interval_updates=40000000, max_windows=4000,
    max_live_conversions=52000, max_live_returned_samples=52000, max_live_observations=9000,
    max_stored_plan_numeric_bytes=64 * 1024**2, max_accounting_categories=32)
WINDOW_SEQUENCES = ("conversions", "returned_samples", "observations", "raw_last_use_windows",
                    "sample_last_use_windows", "nuisance_bindings")


def canonical_window_bytes(window):
    """The record stream that _window_digest hashes, before its field digests."""
    from streaming_array_acquisition import _canonical
    encoder = json.JSONEncoder(sort_keys=True, allow_nan=False, separators=(",", ":"))
    quantizers, chunks = {}, []

    def record(value):
        for part in encoder.iterencode(_canonical(value, quantizers)):
            chunks.append(part.encode("utf-8"))
        chunks.append(b"\n")
    record((window.index, window.start, window.end))
    record(("physical_intervals", len(window.physical_intervals)))
    for fragment in window.physical_intervals:
        record(fragment)
    readout = window.readout_window
    record((readout.index, readout.start, readout.end))
    for name in WINDOW_SEQUENCES:
        sequence = getattr(readout, name)
        record((name, len(sequence)))
        for item in sequence:
            record(item)
    return b"".join(chunks)


def validator_context(*, field_provider, array, noise, limits, input_bindings, check):
    """Attribute namespace on which legacy StreamingArrayAcquisition methods run."""
    from array_acquisition import _bind_noise_provider
    layout, noise_bytes = _bind_noise_provider(array, noise)
    zero = np.zeros(len(array.state_ids))
    zero.flags.writeable = False
    return SimpleNamespace(_check=check, _end=16.0, _limits=limits, _zero_charge=zero,
                           _noise_bytes=noise_bytes, _array=array, _provider=field_provider,
                           _noise_provider=noise, _noise_layout=layout, _max_window_numeric_bytes=0,
                           _bindings={binding.input_id: binding for binding in input_bindings})


def legacy_window_digest(context, window, cursor):
    """Legacy _prepare normalization, field/noise joint digests and window digest."""
    from streaming_array_acquisition import StreamingArrayAcquisition, _window_digest
    normalized, fields, following = StreamingArrayAcquisition._prepare(context, window, cursor)
    digest = _window_digest(normalized, fields)
    stream = canonical_window_bytes(normalized)
    mirror = hashlib.sha256(stream)
    mirror.update(fields)
    if mirror.digest() != digest:
        raise AssertionError("canonical stream mirror differs from legacy _window_digest")
    return dict(digest=digest, fields=fields, canonical=stream, normalized=normalized, following=following)


def cursor_for(window_index, events_before, last_event):
    from streaming_array_acquisition import _PlanCursor
    return _PlanCursor(float(shift(window_index, 0.0)), windows=window_index, last_event=last_event,
                       events=events_before, fragments=events_before)


def legacy_fragment_check(window, plan_window, template):
    """Bitwise comparison of one legacy ArrayWindow's fragments with the plan."""
    fragments = window.physical_intervals
    n = len(fragments)
    if n != len(plan_window.hold_start):
        return dict(equal=False, reason="fragment count", legacy=n, plan=len(plan_window.hold_start))
    start = np.fromiter((f.interval.start for f in fragments), dtype=np.float64, count=n)
    end = np.fromiter((f.interval.end for f in fragments), dtype=np.float64, count=n)
    key = np.fromiter((f.interval.field_key for f in fragments), dtype=np.float64, count=n)
    row = np.fromiter((f.interval.gate_row for f in fragments), dtype=np.int64, count=n)
    event = np.fromiter((f.event_index for f in fragments), dtype=np.int64, count=n)
    differing = [name for name, legacy, new in (
        ("start", start, plan_window.hold_start), ("end", end, plan_window.hold_end),
        ("field_key", key, plan_window.hold_key), ("gate_row", row, template.action_row[plan_window.hold_action]),
        ("event_index", event, plan_window.hold_event))
        if not np.array_equal(float_bits(legacy) if legacy.dtype.kind == "f" else legacy,
                              float_bits(new) if np.asarray(new).dtype.kind == "f" else new)]
    owners, firsts = plan_window.hold_action.tolist(), plan_window.hold_first.tolist()
    selections, charges = template.action_selection_objects, template.action_charge_objects
    for i, fragment in enumerate(fragments):
        interval = fragment.interval
        expected_charge = charges[owners[i]] if firsts[i] else ()
        # Float == is bitwise for nonzero values; zeros additionally compare signs.
        if (interval.voltage_reference_id != template.reference_id
                or interval.daq_selected_sources != selections[owners[i]]
                or interval.injected_charge_coulomb != expected_charge
                or any(a[1] == 0.0 and math.copysign(1.0, a[1]) != math.copysign(1.0, b[1])
                       for a, b in zip(interval.injected_charge_coulomb, expected_charge))):
            differing.append(f"payload@{i}")
            break
    return dict(equal=not differing, differing=differing, fragments=n)


def run_streaming_readout(windows, *, input_ids, input_adc, hold_means, splitter, validate_only, check):
    """Legacy StreamingAcquisitionReadout over contiguous windows 0..W-1."""
    from streaming_acquisition_readout import StreamingAcquisitionReadout, StreamingReadoutLimits
    from streaming_array_acquisition import StreamingArrayAcquisition
    raw_out, obs_out = [], []
    end = windows[-1].end
    readout = StreamingAcquisitionReadout(input_ids=input_ids, acquisition_start=0.0, acquisition_end=end,
        final_publication_horizon=end, limits=StreamingReadoutLimits(**READOUT_LIMITS),
        on_raw=raw_out.append, on_observation=obs_out.append, validate_only=validate_only)
    per_window = []
    column = {key: int(index) for key, index in zip(input_ids, input_adc)}
    for window in windows:
        check()
        raw_before, obs_before = len(raw_out), len(obs_out)
        readout.accept_window(window.readout_window)
        bounds, active_sets = [], []
        means = None if validate_only else np.asarray(hold_means(window.index), dtype=np.float64)
        for j, (_, start, end_time) in enumerate(StreamingArrayAcquisition._splits(splitter, window, readout)):
            active = readout.active_input_ids(start, end_time)
            bounds.append((start, end_time))
            active_sets.append(active)
            if validate_only:
                readout.validate_interval(start=start, end=end_time, input_ids=active)
            else:
                readout.add_interval(start=start, end=end_time,
                                     mean_volts_by_input={key: float(means[j, column[key]]) for key in active})
            readout.advance_publication(end_time)
        samples = None if validate_only else dict(readout._sample_values)
        readout.close_window(window.index)
        per_window.append(dict(index=window.index, raw=raw_out[raw_before:], observations=obs_out[obs_before:],
                               samples=samples, bounds=bounds, active=active_sets))
    report = readout.finalize()
    return per_window, report


def run_eager_readout(window, *, input_ids, input_adc, hold_means, splitter):
    """Legacy eager AcquisitionReadout (the arithmetic the streaming readout reuses)."""
    from acquisition_readout import AcquisitionReadout
    from streaming_array_acquisition import StreamingArrayAcquisition
    plan = window.readout_window
    eager = AcquisitionReadout(conversions=plan.conversions, returned_samples=plan.returned_samples,
        observations=plan.observations, max_conversions=len(plan.conversions),
        max_returned_samples=len(plan.returned_samples), max_observations=len(plan.observations),
        max_interval_updates=10**6)
    means = np.asarray(hold_means, dtype=np.float64)
    bounds = []
    for j, (_, start, end) in enumerate(StreamingArrayAcquisition._splits(splitter, window, eager)):
        bounds.append((start, end))
        eager.add_interval(start=start, end=end, mean_volts_by_input={
            key: float(means[j, int(index)]) for key, index in zip(input_ids, input_adc)})
    result = eager.finalize(finished_at=window.end, acquisition_start=window.start, acquisition_end=window.end)
    return dict(raw=result["raw_values"], observations=result["observations"],
                samples=dict(eager._sample_values), bounds=bounds)


def compare_readout(legacy, readout, template, completed=None, eager_order=False):
    """Bitwise comparison of legacy RawValue/PublishedValue/sample values with arrays.

    ``completed`` (streaming runs) also checks the legacy emission order:
    heap order (completed_at, 'r:<id>') and (available_at, 'o:<id>').
    ``eager_order`` checks the eager readout's order: plan order for raw values
    and (available_at, 'o:<id>') for published observations.
    """
    problems = []
    raw = {value.conversion_id: value for value in legacy["raw"]}
    ids = [f"r:{value}" for value in readout.raw_id.tolist()]
    if set(raw) != set(ids) or len(raw) != len(legacy["raw"]):
        problems.append("raw identity set")
    else:
        rows = [raw[key] for key in ids]
        for name, legacy_values, new in (
                ("aperture_mean", [v.aperture_mean_volts for v in rows], readout.aperture_mean),
                ("decoded", [v.decoded_volts for v in rows], readout.decoded)):
            if not np.array_equal(float_bits(np.array(legacy_values)), float_bits(new)):
                problems.append(name)
        if [v.code_index for v in rows] != readout.code.tolist():
            problems.append("code")
        if [v.valid for v in rows] != readout.raw_valid.tolist() or [v.clipped for v in rows] != readout.raw_clipped.tolist():
            problems.append("raw flags")
        if any(type(v.code_index) is not int for v in rows):
            problems.append("code type")
    published = {value.observation_id: value for value in legacy["observations"]}
    keys = [f"o:{value}" for value in readout.observation_id.tolist()]
    if set(published) != set(keys) or len(published) != len(legacy["observations"]):
        problems.append("observation identity set")
    else:
        rows = [published[key] for key in keys]
        has = [v.value_volts is not None for v in rows]
        values = np.array([v.value_volts if v.value_volts is not None else np.nan for v in rows])
        if has != readout.has_value.tolist():
            problems.append("has_value")
        if not np.array_equal(float_bits(values[readout.has_value]), float_bits(readout.value[readout.has_value])):
            problems.append("observation value")
        if [v.valid for v in rows] != readout.valid.tolist() or [v.clipped for v in rows] != readout.clipped.tolist():
            problems.append("observation flags")
        for name, legacy_values, new in (("assigned", [v.assigned_timestamp for v in rows], readout.assigned),
                                         ("available", [v.available_at for v in rows], readout.available)):
            if not np.array_equal(float_bits(np.array(legacy_values)), float_bits(new)):
                problems.append(name)
        if [v.site_id for v in rows] != [template.site_labels[s] for s in readout.site.tolist()]:
            problems.append("site")
        if [v.role for v in rows] != [template.observation_roles[r] for r in readout.role.tolist()]:
            problems.append("role")
        pointer = readout.support_ptr.tolist()
        support = list(zip(readout.support_raw.tolist(), readout.support_start.tolist(), readout.support_end.tolist()))
        for o, value in enumerate(rows):
            expected = tuple((f"r:{raw_id}", start, end) for raw_id, start, end in support[pointer[o]:pointer[o + 1]])
            if (tuple(item[0] for item in value.contributing_apertures) != tuple(item[0] for item in expected)
                    or any(float_bits(a[1]) != float_bits(b[1]) or float_bits(a[2]) != float_bits(b[2])
                           for a, b in zip(value.contributing_apertures, expected))):
                problems.append("contributing apertures")
                break
    if legacy.get("samples") is not None:
        samples = legacy["samples"]
        sample_ids = [f"s:{value}" for value in readout.returned_id.tolist()]
        if set(samples) != set(sample_ids):
            problems.append("returned identity set")
        else:
            rows = [samples[key] for key in sample_ids]
            if not np.array_equal(float_bits(np.array([r[0] for r in rows])), float_bits(readout.returned_value)):
                problems.append("returned value")
            if ([bool(r[1]) for r in rows] != readout.returned_valid.tolist()
                    or [bool(r[2]) for r in rows] != readout.returned_clipped.tolist()):
                problems.append("returned flags")
    if completed is not None:
        order = readout.raw_publication_order(completed)
        if [value.conversion_id for value in legacy["raw"]] != [ids[i] for i in order]:
            problems.append("raw publication order")
    if completed is not None or eager_order:
        order = readout.observation_publication_order()
        if [value.observation_id for value in legacy["observations"]] != [keys[i] for i in order]:
            problems.append("observation publication order")
    if eager_order and [value.conversion_id for value in legacy["raw"]] != ids:
        problems.append("eager raw order")
    return problems


def compare_splits(legacy_bounds, legacy_active, plan_window, input_ids):
    """Legacy split grid and active inputs versus the plan's typed arrays."""
    starts = np.array([a for a, _ in legacy_bounds])
    ends = np.array([b for _, b in legacy_bounds])
    grid = plan_window.split_bounds
    problems = []
    if (len(starts) != len(grid) - 1 or not np.array_equal(float_bits(starts), float_bits(grid[:-1]))
            or not np.array_equal(float_bits(ends), float_bits(grid[1:]))):
        problems.append("split grid")
    if legacy_active is not None:
        active = [set(row) for row in legacy_active]
        new = [{input_ids[i] for i in np.flatnonzero(row)} for row in plan_window.split_active]
        if active != new:
            problems.append("active inputs")
    return problems


def synthetic_hold_means(plan_window, *, thresholds, levels, input_min, input_max, n_adc, seed, retained=None):
    """Deterministic adversarial hold means for one window, [split, ADC].

    Per (slot, ADC) the category is typical (N(0, 30 uV)), wide (U(+-12.5 mV),
    reaching clipping), near-threshold (both aperture intervals within 3 ulps of
    one threshold, so the code depends on exact integration arithmetic) or
    special (signed zeros, range ends and their neighbours, exact thresholds and
    levels, subnormals). ``retained`` replaces the leading rows with actual
    legacy-solver hold means.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    splits = len(plan_window.split_hold)
    slot = plan_window.hold_action[plan_window.split_hold]
    slots = int(slot.max()) + 1
    category = rng.choice(4, p=(.35, .15, .35, .15), size=(slots, n_adc))[slot]
    typical = rng.normal(0., 30e-6, (splits, n_adc))
    wide = rng.uniform(-0.0125, 0.0125, (splits, n_adc))
    near = thresholds[rng.integers(0, len(thresholds), (slots, n_adc))][slot]
    steps = rng.integers(-3, 4, (splits, n_adc))
    for step in range(3):
        near = np.where(steps > step, np.nextafter(near, np.inf), near)
        near = np.where(steps < -step, np.nextafter(near, -np.inf), near)
    specials = np.array([0.0, -0.0, input_min, input_max, np.nextafter(input_max, np.inf),
                         np.nextafter(input_min, -np.inf), 5e-324, -5e-324, thresholds[len(thresholds) // 2],
                         levels[len(levels) // 2], thresholds[0], thresholds[-1], levels[0], levels[-1]])
    special = specials[rng.integers(0, len(specials), (splits, n_adc))]
    means = np.choose(category, (typical, wide, near, special))
    if retained is not None:
        retained = np.asarray(retained, dtype=np.float64)
        means[:len(retained)] = retained
    return means


def load_retained_hold_means(path):
    """Actual legacy-solver hold means for the first 8 split intervals (attempt004)."""
    with open(path, encoding="utf-8") as stream:
        report = json.load(stream)
    intervals = report["diagnostics"]["component_intervals"]
    bounds = [(float(item["start_seconds"]), float(item["end_seconds"])) for item in intervals]
    means = [[float(v) for v in item["returned_kernel"]["hold_mean_volts"]] for item in intervals]
    return bounds, np.array(means)


def window_ends_check(windows):
    """Every legacy readout window must use index*CYCLE_SECONDS bounds."""
    return all(w.start == float(shift(w.index, 0.0)) and w.end == float(shift(w.index, CYCLE_SECONDS))
               for w in windows)
