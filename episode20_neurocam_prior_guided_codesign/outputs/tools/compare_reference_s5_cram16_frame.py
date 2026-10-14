#!/usr/bin/env python3
"""Independent full-frame CRAM16 comparison and carried second-frame cost.

Original and CRAM16 families each construct their own S5Operator, including
every charge solve. Original interval endpoints are retained only in RAM;
original inputs are reconstructed from the preceding endpoint and actual
charge. No record, amplitude estimator or scientific outcome is accessed.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import time


def compare_structure(structure, deadline, record):
    import numpy as np
    from fast_solver_continuous import make_clock
    from reference_s5_family import AirFamily, PROBE_SITE, initial_grid
    from reference_s5_cram16_prototype import Cram16PrototypeAirFamily, THETA
    from reference_s5_structured_solve import StructuredCapacitanceSolve
    from reference_s5_shifted_structured import StructuredShiftedInverse
    from reference_s5_operator import S5Operator

    started = time.monotonic()
    clock = make_clock(max(.001, deadline-started), 1536.)
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded full-frame comparison")
        clock.check()

    def error(actual, expected, atol):
        delta = np.abs(actual-expected)
        if not np.all(np.isfinite(delta)):
            raise ArithmeticError("nonfinite comparison error")
        return float(np.max(delta)), float(np.max(delta/(atol+2e-8*np.abs(expected))))

    original = AirFamily(structure, initial_grid()[1], check=check)
    candidate = Cram16PrototypeAirFamily(structure, initial_grid()[1], check=check)
    operators = {"original": S5Operator(original), "cram16": S5Operator(candidate)}
    first, second = operators["original"], operators["cram16"]
    # Same constructor, independent evaluation of all solver-dependent charges.
    # The digest identifies actual acquisition timings/topology, not source code.
    if first.template.digest != second.template.digest:
        raise AssertionError("independent acquisition templates differ")

    def qualify_max_coupling_inverses():
        # Retain the qualified full-size capacitance check. The new shifted
        # inverses additionally face every CRAM pole, every source/marker slot,
        # the driven row and both row edges at the largest coupling endpoint.
        # This checks matrices only; full trajectory comparison follows below.
        import scipy.sparse.linalg as sla
        family = AirFamily(structure, initial_grid()[-1], check=check)
        inverse = StructuredCapacitanceSolve(family)
        rng = np.random.default_rng(820)
        capacitance_checks = []
        for columns in (1, 3):
            rhs = rng.normal(size=(family.size, columns))*1e-12
            if columns == 1:
                rhs = rhs[:, 0]
            expected = family.cap_factor.solve(rhs)
            for trans in ("N", "T"):
                check()
                actual = inverse.solve(rhs, trans=trans)
                ratio = float(np.max(np.abs(actual-expected)/(2e-14+2e-12*np.abs(expected))))
                residual = float(np.max(np.abs(family.capacitance @ actual-rhs))/np.max(np.abs(rhs)))
                if not np.isfinite([ratio, residual]).all() or ratio > 1. or residual > 2e-12:
                    raise AssertionError("maximum-coupling capacitance inverse exceeds original tolerances")
                capacitance_checks.append(dict(rhs_columns=columns, transpose=trans,
                    tolerance_ratio=ratio, relative_residual=residual))
        record["max_coupling_inverse_checks"] = capacitance_checks
        shifted_checks = record["max_coupling_shifted_inverse_checks"] = []
        indices = np.arange(family.size, dtype=float)[:, None]
        columns = np.arange(1, 4, dtype=float)[None, :]
        trial = np.sin(indices*columns*.173)+1j*np.cos(indices*columns*.219)
        rhs_all = np.asarray(family.capacitance @ trial, dtype=np.complex128)
        window = first.window(1)
        for physical_row in (0, PROBE_SITE[0], 63):
            for slot in range(4):
                actions = np.flatnonzero((first.template.action_row == physical_row)
                    & (np.arange(first.template.actions) % 4 == slot))
                if not len(actions):
                    raise AssertionError("required original row/slot action is missing")
                raw = first.raw_groups[int(actions[0])]
                duration = float(window.conv_aperture_end[raw[0]]-window.conv_aperture_start[raw[0]])
                conductance, _ = family.matrices(physical_row, slot)
                for pole_index, pole in enumerate(THETA):
                    check()
                    matrix = (pole*family.capacitance+duration*conductance).tocsc()
                    matrix.sum_duplicates()
                    matrix.sort_indices()
                    direct = sla.splu(matrix)
                    shifted = StructuredShiftedInverse(family, pole, duration, physical_row, slot)
                    matrix_norm = float(np.max(np.asarray(abs(matrix).sum(axis=1)).ravel()))
                    for count in (1, 3):
                        rhs = rhs_all[:, 0] if count == 1 else rhs_all
                        expected, actual = direct.solve(rhs), shifted.solve(rhs)
                        ratio = float(np.max(np.abs(actual-expected)/(2e-14+2e-12*np.abs(expected))))
                        residual = matrix @ actual-rhs
                        if count == 1:
                            values, right, errors = (value[:, None] for value in (actual, rhs, residual))
                        else:
                            values, right, errors = actual, rhs, residual
                        errors_norm = np.max(np.abs(errors), axis=0)
                        rhs_norm = np.max(np.abs(right), axis=0)
                        relative = float(np.max(errors_norm/rhs_norm))
                        backward = float(np.max(errors_norm/(
                            matrix_norm*np.max(np.abs(values), axis=0)+rhs_norm)))
                        shifted_checks.append(dict(row=physical_row, slot=slot, pole_index=pole_index,
                            rhs_columns=count, duration_hex=duration.hex(), tolerance_ratio=ratio,
                            relative_residual=relative, normwise_backward_residual=backward))
                        if (not np.isfinite([ratio, relative, backward]).all()
                                or ratio > 1. or relative > 2e-12):
                            raise AssertionError("maximum-coupling CRAM shift exceeds original inverse tolerances")

    qualify_max_coupling_inverses()
    gc.collect()
    charge_max, charge_ratio = 0., 0.
    for actual, expected in zip(second.charge_deltas, first.charge_deltas):
        absolute, ratio = error(actual, expected, 2e-12)
        charge_max, charge_ratio = max(charge_max, absolute), max(charge_ratio, ratio)
    if len(first.charge_deltas) != len(second.charge_deltas) or charge_ratio > 1.:
        raise AssertionError("independent full-size switch-charge solves differ")
    maximum_intervals = 3*first.template.actions
    if first.template.actions != 1600 or maximum_intervals != 4800:
        raise AssertionError("original complete1600-action frame required")
    endpoints = np.empty((maximum_intervals, original.size), dtype=np.float64)
    integrals = np.empty((maximum_intervals, original.adc_count), dtype=np.float64)
    identities = []
    record.update(physical_coordinates=original.size, constructor_seconds=time.monotonic()-started,
        maximum_intervals=maximum_intervals, reference_ram_bytes=endpoints.nbytes+integrals.nbytes,
        independent_charge_max_abs_error=charge_max, independent_charge_max_tolerance_ratio=charge_ratio,
        passes=[])
    reference_frame = None
    expected_interval_count = None

    for name, family in (("original", original), ("cram16", candidate)):
        operator = operators[name]
        entry = dict(name=name, frame=0, completed_intervals=0,
            input_max_tolerance_ratio=0., state_max_abs_error=0., state_max_tolerance_ratio=0.,
            integral_max_abs_error=0., integral_max_tolerance_ratio=0.)
        record["passes"].append(entry)
        sparse = family.sparse_step
        before_products = family.stats["operator_products"]
        before_cram = candidate.cram_report() if name == "cram16" else None
        previous_action = -1
        energy_difference_bound = 0.
        previous_fast_endpoint = np.zeros(family.size)

        def energy_error_bound(actual, expected, at):
            # Exact quadratic identity bounded in absolute value; an added
            # dimension-scaled roundoff term covers the energy dot products.
            from reference_s5_family import AMPLITUDE_VOLTS, OMEGA
            u = AMPLITUDE_VOLTS*np.sin(OMEGA*at)
            delta = np.abs(actual-expected)
            c_abs = abs(original.capacitance)
            bound = .5*float(delta @ (c_abs @ (np.abs(actual)+np.abs(expected))))
            bound += abs(u)*float(delta @ np.abs(original.cap_probe))
            scale = .5*float(np.abs(actual) @ (c_abs @ np.abs(actual)))
            scale += .5*float(np.abs(expected) @ (c_abs @ np.abs(expected)))
            scale += abs(u)*float((np.abs(actual)+np.abs(expected)) @ np.abs(original.cap_probe))
            return bound + (original.size+8)*np.finfo(float).eps*scale

        def step(state, row, slot, marker, start, end, **kwargs):
            nonlocal previous_action, energy_difference_bound, previous_fast_endpoint
            index = entry["completed_intervals"]
            if index >= maximum_intervals:
                raise AssertionError("more than three intervals per physical action")
            action = int(np.searchsorted(first.template.action_end, start, side="right"))
            if (not 0 <= action < first.template.actions or row != first.template.action_row[action]
                    or slot != action % 4 or end > first.template.action_end[action]):
                raise AssertionError("interval is outside its declared physical action")
            new_action = action != previous_action
            if new_action and action != previous_action+1:
                raise AssertionError("physical action order changed")
            previous_reference = np.zeros(original.size) if index == 0 else endpoints[index-1]
            expected_input = previous_reference.copy()
            if new_action:
                expected_input += first.charge_deltas[action]
            identity = (int(row), int(slot), float(marker).hex(), float(start).hex(), float(end).hex())
            if name == "original":
                # This establishes input reconstruction from endpoint+charge
                # before relying on it to save approximately 158 MiB of RAM.
                if not np.array_equal(state, expected_input):
                    raise AssertionError("original input is not prior endpoint plus its scheduled charge")
                identities.append(identity)
            else:
                if index >= len(identities) or identity != identities[index]:
                    raise AssertionError("full-frame acquisition interval identity changed")
                _, ratio = error(state, expected_input, 2e-12)
                entry["input_max_tolerance_ratio"] = max(entry["input_max_tolerance_ratio"], ratio)
                if ratio > 1.:
                    raise AssertionError("independently carried input exceeds original state tolerance")
                if new_action:
                    energy_difference_bound += energy_error_bound(previous_fast_endpoint, previous_reference, start)
                    energy_difference_bound += energy_error_bound(state, expected_input, start)
            result = sparse(state, row, slot, marker, start, end, **kwargs)
            if name == "original":
                endpoints[index], integrals[index] = result
            else:
                absolute, ratio = error(result[0], endpoints[index], 2e-12)
                int_absolute, int_ratio = error(result[1], integrals[index], 2e-20)
                for key, value in (("state_max_abs_error", absolute), ("state_max_tolerance_ratio", ratio),
                        ("integral_max_abs_error", int_absolute), ("integral_max_tolerance_ratio", int_ratio)):
                    entry[key] = max(entry[key], value)
                if max(ratio, int_ratio) > 1.:
                    raise AssertionError("full-frame endpoint or hold integral exceeds original tolerances")
                previous_fast_endpoint = result[0].copy()
            previous_action = action
            entry["completed_intervals"] += 1
            return result

        family.sparse_step = step
        before = time.monotonic()
        try:
            frame_result = operator.frame(np.zeros(family.size), 0)
            if previous_action != first.template.actions-1 or entry["completed_intervals"] != 4800:
                raise AssertionError("full frame must retain all1600 actions and4800 intervals")
            entry["status"] = "complete"
        finally:
            family.sparse_step = sparse
            entry["elapsed_seconds"] = time.monotonic()-before
            entry["operator_products"] = family.stats["operator_products"]-before_products
            if name == "cram16":
                after_cram = candidate.cram_report()
                entry["cram_delta"] = {key: after_cram[key]-before_cram[key] for key in
                    ("hits", "misses", "evictions", "constructor_seconds", "action_seconds", "actions")}
                entry["cram_report"] = after_cram
        if name == "original":
            reference_frame = frame_result
            expected_interval_count = entry["completed_intervals"]
        else:
            if entry["completed_intervals"] != expected_interval_count:
                raise AssertionError("full-frame interval count differs")
            actual_readout, expected_readout = frame_result[1], reference_frame[1]
            if actual_readout.index != expected_readout.index:
                raise AssertionError("readout frame identity differs")
            exact_fields = []
            for field in expected_readout.__dataclass_fields__:
                if field == "index":
                    continue
                actual, expected = getattr(actual_readout, field), getattr(expected_readout, field)
                if field == "aperture_mean":
                    absolute, ratio = error(actual, expected, 2e-12)
                    entry["raw_aperture_max_abs_error"] = absolute
                    entry["raw_aperture_max_tolerance_ratio"] = ratio
                    if ratio > 1.:
                        raise AssertionError("raw aperture mean exceeds state tolerance")
                elif not np.array_equal(actual, expected, equal_nan=True):
                    raise AssertionError(f"complete readout field differs: {field}")
                else:
                    exact_fields.append(field)
            entry["identical_readout_fields"] = exact_fields
            energy_error = abs(frame_result[2]-reference_frame[2])
            # Also account for summing 1,600 before/after differences.
            energy_difference_bound += first.template.actions*np.finfo(float).eps*(
                abs(frame_result[2])+abs(reference_frame[2]))
            entry["boundary_energy_abs_error_joule"] = energy_error
            entry["boundary_energy_state_and_roundoff_bound_joule"] = energy_difference_bound
            if not np.isfinite(energy_error) or energy_error > energy_difference_bound:
                raise AssertionError("boundary-energy change exceeds the propagated state-error bound")
            carried_state = frame_result[0]
        print(json.dumps(dict(structure=structure, frame=0, solver=name,
            completed_intervals=entry["completed_intervals"], elapsed_seconds=entry["elapsed_seconds"])), flush=True)
    # Release stored reference states before the next-phase timing measurement.
    del endpoints, integrals, identities, reference_frame
    gc.collect()
    entry = dict(name="cram16_carried", frame=1, completed_intervals=0,
                 role="cost_measurement_only_no_original_frame1_comparison")
    record["passes"].append(entry)
    before_cram = candidate.cram_report()
    sparse = candidate.sparse_step
    def carried_step(*args, **kwargs):
        result = sparse(*args, **kwargs)
        entry["completed_intervals"] += 1
        return result
    candidate.sparse_step = carried_step
    before = time.monotonic()
    try:
        second.frame(carried_state, 1)
        if entry["completed_intervals"] != 4800:
            raise AssertionError("carried cost frame must retain all4800 intervals")
        entry["status"] = "complete"
    finally:
        candidate.sparse_step = sparse
        entry["elapsed_seconds"] = time.monotonic()-before
        after_cram = candidate.cram_report()
        entry["cram_delta"] = {key: after_cram[key]-before_cram[key] for key in
            ("hits", "misses", "evictions", "constructor_seconds", "action_seconds", "actions")}
        entry["cram_report"] = after_cram
    print(json.dumps(dict(structure=structure, frame=1, solver="cram16",
        role=entry["role"], completed_intervals=entry["completed_intervals"],
        elapsed_seconds=entry["elapsed_seconds"])), flush=True)
    record.update(status="passed", elapsed_seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-seconds", type=float, default=3300.)
    parser.add_argument("--structures", nargs="+", choices=("R2", "R3"), default=["R2", "R3"])
    args = parser.parse_args()
    if not math.isfinite(args.total_seconds) or not 0 < args.total_seconds <= 3300:
        parser.error("a positive finite total limit of at most3,300 seconds is required")
    if len(set(args.structures)) != len(args.structures):
        parser.error("each requested structure may appear only once")
    host = socket.gethostname()
    if (not os.environ.get("SLURM_JOB_ID", "").isdigit() or "-ln" in host
            or not host.startswith("sh") or os.environ.get("SLURM_CPUS_PER_TASK") != "1"):
        parser.error("coordinator-owned one-CPU compute step required")
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")):
        parser.error("all six numerical thread counts must be one")
    output = args.output.resolve()
    permitted = Path(__file__).resolve().parents[1] / "runs" / "reference_s5"
    if not output.is_relative_to(permitted) or output.exists() or not output.parent.is_dir():
        parser.error("fresh output in an existing reference_s5 run directory required")
    started = time.monotonic()
    report = dict(kind="isolated_cram16_full_frame_comparison", point=1, compared_frame=0,
        independent_charge_operators=True, carried_frame1="timing_only", heldaside_access=False,
        record_writer_created=False, total_seconds_limit=args.total_seconds, memory_mib_limit=1536.,
        coefficient_source="https://docs.openmc.org/en/stable/_modules/openmc/deplete/cram.html",
        qualification_scope="point01 frame0; frame1 timing; maximum-coupling inverse checks; no full-record accuracy claim",
        requested_structures=args.structures,
        tolerance=dict(state_atol=2e-12, integral_atol=2e-20, rtol=2e-8), structures=[])
    def stop(signum, frame):
        raise TimeoutError("bounded full-frame alarm")
    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, args.total_seconds)
    try:
        for structure in args.structures:
            row = dict(structure=structure, status="starting")
            report["structures"].append(row)
            try:
                compare_structure(structure, started+args.total_seconds, row)
            except Exception as exc:
                row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                if isinstance(exc, (TimeoutError, MemoryError)):
                    break
            gc.collect()
        report["status"] = "passed" if len(report["structures"]) == len(args.structures) and all(
            row["status"] == "passed" for row in report["structures"]) else "incomplete_or_failed"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous)
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
