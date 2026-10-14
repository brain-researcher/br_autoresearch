"""Compare Route C fast-path complete windows with the unchanged reference.

Use the original state/hold/code/pixel/work tolerances. Fast-path interval
work is partial and is NEVER summed as complete physical work: only the
producer's named window_work rows are compared. Missing/nonfinite required
data cannot pass, and all three registered windows are always reported.
The original comparator and its active reference job remain unchanged.
"""
import argparse
import json
import math
import os
from pathlib import Path
import socket

import numpy as np

from fast_solver_compare import (STATE_ATOL, STATE_RTOL, WORK_ATOL, WORK_RTOL,
                                 WINDOWS, _quantizer, conversion_means, scaled)
from fast_solver_reference import WORK_FIELDS

STATE_COUNT = 69848
ROOT = Path(__file__).resolve().parents[2] / "outputs/runs/fast_forward_stage1/solver"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def finite_array(value, shape, reason):
    value = np.asarray(value)
    require(value.shape == shape and np.issubdtype(value.dtype, np.number)
            and np.isfinite(value).all(), reason)
    return value


def voltage_comparison(reference, actual):
    reference, actual = np.asarray(reference), np.asarray(actual)
    require(reference.shape == actual.shape and reference.size > 0
            and np.isfinite(reference).all() and np.isfinite(actual).all(), "invalid_voltage_values")
    error = scaled(actual, reference)
    return dict(available=True, count=int(reference.size), max_scaled=float(error.max()),
                max_abs=float(np.abs(actual-reference).max()),
                violations=int(np.sum(error > 1)), passed=bool(error.max() <= 1))


def physical_work_comparison(reference, actual):
    reference = finite_array(reference, (len(WORK_FIELDS),), "invalid_reference_window_work")
    actual = finite_array(actual, reference.shape, "invalid_complete_fastpath_window_work")
    result = {}
    for name, expected, observed in zip(WORK_FIELDS, reference, actual):
        tolerance = WORK_ATOL + WORK_RTOL*abs(expected)
        difference = abs(observed-expected)
        result[name] = dict(reference=float(expected), solver=float(observed),
            abs_diff=float(difference), scaled=float(difference/tolerance), passed=bool(difference <= tolerance))
    return result


def complete_window_work(intervals, name):
    require(tuple(str(x) for x in intervals["work_fields"]) == WORK_FIELDS,
            "fastpath_work_fields_differ")
    names = np.asarray(intervals["window_names"])
    require(names.ndim == 1 and len(set(map(str, names))) == len(names), "duplicate_window_work_names")
    rows = np.flatnonzero(names == name)
    require(len(rows) == 1, "complete_window_work_unavailable")
    values = np.asarray(intervals["window_work"])
    require(values.shape == (len(names), len(WORK_FIELDS)), "window_work_shape_differs")
    return finite_array(values[rows[0]], (len(WORK_FIELDS),), "complete_window_work_nonfinite")


def interval_prefix(intervals, end):
    start, stop = np.asarray(intervals["t_start"]), np.asarray(intervals["t_end"])
    require(start.ndim == stop.ndim == 1 and start.shape == stop.shape, "invalid_interval_time_shape")
    rows = np.flatnonzero(np.isfinite(stop) & (stop <= end))
    require(len(rows) > 0 and np.array_equal(rows, np.arange(len(rows))), "incomplete_interval_prefix")
    a, b = start[rows], stop[rows]
    require(np.isfinite(a).all() and a[0] == 0. and b[-1] == end
            and np.all(b > a) and np.array_equal(a[1:], b[:-1]), "interval_gap_overlap_or_wrong_endpoint")
    holds = np.asarray(intervals["hold_mean"])
    require(holds.shape == (len(start), 24), "hold_mean_shape_differs")
    holds = finite_array(holds[rows], (len(rows), 24), "nonfinite_hold_mean")
    return rows, dict(t_start=a, t_end=b, hold_mean=holds)


def code_pixel_comparison(reference_means, actual_means, thresholds, levels, pixel_rows):
    """Same threshold tie convention, admitted one-code differences and pixel limit."""
    reference_means, actual_means = np.asarray(reference_means), np.asarray(actual_means)
    require(reference_means.ndim == 1 and reference_means.shape == actual_means.shape
            and len(reference_means) > 0 and np.isfinite(reference_means).all()
            and np.isfinite(actual_means).all(), "missing_or_nonfinite_conversion_means")
    thresholds, levels = np.asarray(thresholds), np.asarray(levels)
    require(thresholds.ndim == levels.ndim == 1 and len(thresholds) > 0
            and len(levels) == len(thresholds)+1 and np.isfinite(thresholds).all()
            and np.isfinite(levels).all() and np.all(np.diff(thresholds) > 0)
            and np.all(np.diff(levels) > 0), "invalid_quantizer")
    cr = np.searchsorted(thresholds, reference_means, side="right")
    cn = np.searchsorted(thresholds, actual_means, side="right")
    tolerance = STATE_ATOL + STATE_RTOL*np.abs(reference_means)
    nearest = np.minimum(np.abs(reference_means-thresholds[np.clip(cr-1, 0, len(thresholds)-1)]),
                         np.abs(reference_means-thresholds[np.clip(cr, 0, len(thresholds)-1)]))
    near = nearest <= tolerance
    differ = cn != cr
    bad = differ & ~(near & (np.abs(cn-cr) == 1))
    errors = scaled(actual_means, reference_means)
    codes = dict(available=True, conversions=int(len(cr)), identical=int(np.sum(~differ)),
        differ=int(differ.sum()), differ_allowed=int(np.sum(differ & ~bad)), violations=int(bad.sum()),
        reference_near_threshold=int(near.sum()), conversion_mean_max_scaled=float(errors.max()),
        conversion_mean_max_abs=float(np.abs(actual_means-reference_means).max()), passed=bool(not bad.any()))
    rows = np.asarray(pixel_rows)
    require(rows.ndim == 2 and rows.shape[1] == 3 and np.issubdtype(rows.dtype, np.integer)
            and np.all((rows >= 0) & (rows < len(cr))), "invalid_pixel_conversion_rows")
    if not len(rows):
        return codes, dict(available=True, count=0, passed=True, reason="no_complete_pixel_in_this_window")
    reference_pixels = levels[cr[rows]].sum(axis=1)/3.0
    actual_pixels = levels[cn[rows]].sum(axis=1)/3.0
    difference = np.abs(actual_pixels-reference_pixels)
    lsb = float(levels[1]-levels[0])
    limit = lsb/3.0*(1+1e-9)  # Identical floating-point boundary allowance to the original comparator.
    pixels = dict(available=True, count=int(len(rows)), max_abs=float(difference.max()),
        max_lsb=float(difference.max()/lsb), violations=int(np.sum(difference > limit)),
        passed=bool(np.all(difference <= limit)))
    return codes, pixels


def readout_comparison(reference, ref_prefix, new_prefix, end, thresholds, levels):
    start, stop, adc = (np.asarray(reference[name]) for name in ("conv_start", "conv_end", "conv_adc"))
    require(start.ndim == stop.ndim == adc.ndim == 1 and start.shape == stop.shape == adc.shape
            and np.isfinite(start).all() and np.isfinite(stop).all() and np.all(stop > start)
            and np.all(start >= 0) and np.issubdtype(adc.dtype, np.integer)
            and np.all((adc >= 0) & (adc < 24)), "invalid_reference_conversion_plan")
    inside = stop <= end
    mr = conversion_means(ref_prefix, start[inside], stop[inside], adc[inside])
    mn = conversion_means(new_prefix, start[inside], stop[inside], adc[inside])
    pixels = np.asarray(reference["pixel_conv"])
    require(pixels.ndim == 2 and pixels.shape[1] == 3 and np.issubdtype(pixels.dtype, np.integer)
            and np.all((pixels >= 0) & (pixels < len(start))), "invalid_reference_pixel_plan")
    lookup = np.full(len(start), -1, dtype=np.int64)
    lookup[np.flatnonzero(inside)] = np.arange(int(inside.sum()))
    complete = np.all(lookup[pixels] >= 0, axis=1)
    return code_pixel_comparison(mr, mn, thresholds, levels, lookup[pixels[complete]])


def optional_voltage_diagnostic(reference, actual, ref_rows, new_rows, key, columns):
    try:
        a = finite_array(np.asarray(reference[key])[ref_rows], (len(ref_rows), columns), "unavailable_reference_diagnostic")
        b = finite_array(np.asarray(actual[key])[new_rows], (len(new_rows), columns), "unavailable_fastpath_diagnostic")
        result = voltage_comparison(a, b)
        result.pop("passed")  # Descriptive only, as in the original comparator.
        return result
    except (KeyError, ValueError, IndexError):
        return dict(available=False, reason="not_recorded_as_finite_complete_interval_values")


def window_energy_audit(report, name):
    value = report.get("window_energy_audit", {}).get(name)
    fields = ("stored_start", "stored_end", "input_work", "heat", "supplied_charge_work", "residual", "relative_residual")
    if not isinstance(value, dict) or value.get("available") is not True:
        return dict(available=False, reason="complete_window_energy_audit_not_available")
    if not all(isinstance(value.get(key), (int, float)) and math.isfinite(value[key]) for key in fields):
        return dict(available=False, reason="complete_window_energy_audit_nonfinite_or_incomplete")
    # Require the independent integration diagnostics, rather than treating a
    # placeholder zero residual or missing quadrature record as complete.
    if (not isinstance(value.get("method"), str) or not value["method"]
            or value.get("upstream_quadrature_orders") != [16, 32]
            or value.get("downstream_quadrature_order") != 16):
        return dict(available=False, reason="independent_energy_audit_method_incomplete")
    for key in ("quadrature_signed_difference", "quadrature_absolute_difference_sum"):
        diagnostic = value.get(key)
        if not isinstance(diagnostic, dict) or not all(
                isinstance(diagnostic.get(field), (int, float)) and math.isfinite(diagnostic[field])
                for field in ("input_work", "heat")):
            return dict(available=False, reason="invalid_energy_audit_quadrature_diagnostics")
    # The audit is descriptive: this comparator invents no new residual margin.
    result = {key: value[key] for key in fields}
    result.update(available=True, role="descriptive_no_new_residual_threshold")
    for key in ("method", "upstream_quadrature_orders", "downstream_quadrature_order",
                "quadrature_signed_difference", "quadrature_absolute_difference_sum",
                "downstream_order_doubling", "residual_acceptance"):
        if key in value:
            try:
                json.dumps(value[key], allow_nan=False)
            except (ValueError, TypeError):
                return dict(available=False, reason="invalid_energy_audit_diagnostics")
            result[key] = value[key]
    return result


def window_solver_diagnostics(report, name):
    value = report.get("window_solver_diagnostics", {}).get(name)
    fields = ("segments", "sweeps", "max_sweeps", "max_wr_change", "max_rep_pad", "max_rep_source", "clip_events")
    if not isinstance(value, dict) or value.get("available") is not True or not all(
            isinstance(value.get(key), (int, float)) and math.isfinite(value[key]) for key in fields):
        return dict(available=False, reason="complete_window_solver_diagnostics_not_available")
    return dict(available=True, scope=value.get("scope", "completed segments and downstream intervals"),
                **{key: value[key] for key in fields})


def compare_arrays(ref, new, ref_ck, new_ck, *, thresholds, levels, solver_report, state_ids=None):
    """Array interface for saved evidence and small focused tests; no archive substitution."""
    results = {}
    for name, slots in WINDOWS:
        window = dict(available=False, passed=False, status="invalid_input")
        try:
            ri = np.flatnonzero(np.asarray(ref_ck["slots"]) == slots)
            ni = np.flatnonzero(np.asarray(new_ck["slots"]) == slots)
            if not len(ri) or not len(ni):
                window.update(status="missing_window", missing_reference=not len(ri), missing_solver=not len(ni),
                              reason="required_checkpoint_unavailable")
                results[name] = window
                continue
            require(len(ri) == len(ni) == 1, "required_checkpoint_duplicated")
            end = float(ref_ck["times"][ri[0]])
            require(math.isfinite(end) and float(new_ck["times"][ni[0]]) == end, "checkpoint_times_differ_or_nonfinite")
            xr = finite_array(ref_ck["states"][ri[0]], (STATE_COUNT,), "invalid_reference_full_state")
            xn = finite_array(new_ck["states"][ni[0]], (STATE_COUNT,), "invalid_fastpath_full_state")
            state = voltage_comparison(xr, xn)
            worst = int(np.argmax(scaled(xn, xr)))
            require(state_ids is None or len(state_ids) == STATE_COUNT, "state_id_count_differs")
            state.update(worst_index=worst, worst_id=None if state_ids is None else state_ids[worst],
                         worst_ref=float(xr[worst]), worst_new=float(xn[worst]))
            ref_rows, ref_prefix = interval_prefix(ref, end)
            new_rows, new_prefix = interval_prefix(new, end)
            require(np.array_equal(ref_prefix["t_start"], new_prefix["t_start"])
                    and np.array_equal(ref_prefix["t_end"], new_prefix["t_end"]), "interval_grids_differ")
            holds = voltage_comparison(ref_prefix["hold_mean"], new_prefix["hold_mean"])
            require(tuple(str(x) for x in ref["work_fields"]) == WORK_FIELDS, "reference_work_fields_differ")
            ref_work = finite_array(np.asarray(ref["work"])[ref_rows],
                                    (len(ref_rows), len(WORK_FIELDS)), "nonfinite_reference_interval_work")
            work = physical_work_comparison(ref_work.sum(axis=0), complete_window_work(new, name))
            codes, pixels = readout_comparison(ref, ref_prefix, new_prefix, end, thresholds, levels)
            work_passed = all(item["passed"] for item in work.values())
            window.update(available=True, state=state, hold_mean=holds, work=work,
                work_source="complete_named_window_work_not_partial_interval_work", work_passed=work_passed,
                codes=codes, pixels=pixels,
                hold_end=optional_voltage_diagnostic(ref, new, ref_rows, new_rows, "hold_end", 24),
                source_mean=optional_voltage_diagnostic(ref, new, ref_rows, new_rows, "source_mean", 64),
                solver=window_solver_diagnostics(solver_report, name),
                per_interval_solver=dict(available=False, reason="fastpath_has_no_complete_per_interval_solver_diagnostics"),
                energy_audit=window_energy_audit(solver_report, name),
                passed=bool(state["passed"] and holds["passed"] and work_passed and codes["passed"] and pixels["passed"]))
            window["status"] = "passed" if window["passed"] else "physical_tolerance_failure"
        except (KeyError, ValueError, IndexError, TypeError) as error:
            window["reason"] = str(error)
        results[name] = window
    physical_passed = all(value["available"] and value["passed"] for value in results.values())
    energy_available = all(value.get("energy_audit", {}).get("available", False) for value in results.values())
    return dict(windows=results, required_windows=[name for name, _ in WINDOWS],
        all_physical_windows_passed=physical_passed, complete_window_energy_audits_available=energy_available,
        missing_windows=[name for name, value in results.items() if value["status"] == "missing_window"],
        invalid_windows=[name for name, value in results.items() if value["status"] == "invalid_input"],
        available_window_failure=any(value["available"] and not value["passed"] for value in results.values()),
        energy_audit_incomplete_windows=[name for name, value in results.items()
                                        if not value.get("energy_audit", {}).get("available", False)],
        solver_status=solver_report.get("status"),
        qualified=bool(physical_passed and energy_available and solver_report.get("status") == "complete"),
        no_new_energy_residual_threshold=True)


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def compare(reference_dir, solver_dir, *, state_ids=None):
    thresholds, levels = _quantizer()
    return compare_arrays(load_archive(reference_dir / "reference_intervals.npz"),
        load_archive(solver_dir / "solver_intervals.npz"),
        load_archive(reference_dir / "reference_checkpoints.npz"),
        load_archive(solver_dir / "solver_checkpoints.npz"), thresholds=thresholds, levels=levels,
        solver_report=json.loads((solver_dir / "solver_report.json").read_text()), state_ids=state_ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require("-ln" not in socket.gethostname() and os.environ.get("SLURM_JOB_ID"), "compute_allocation_required")
    require(all(path.resolve().is_relative_to(ROOT) for path in (args.reference, args.solver, args.output)),
            "selected_stage1_solver_paths_only")
    from conditional_forward_recipe import expected_state_ids
    result = compare(args.reference, args.solver, state_ids=expected_state_ids())
    result.update(reference=str(args.reference), solver=str(args.solver), comparator="fastpath_complete_window_work")
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    with args.output.open("x") as stream:
        stream.write(encoded + "\n")
    print(encoded[:6000])
    return 0 if result["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
