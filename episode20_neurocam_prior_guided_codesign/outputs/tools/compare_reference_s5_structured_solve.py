#!/usr/bin/env python3
"""Full-size inverse and unchanged 24-interval S5 prefix comparison.

No record, calibration, response HDF or held-aside outcome is read/written.
This measures an exact solver replacement; prefix timing is not a claim that
an entire 4,000-frame run fits a given resource request.
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


class PrefixComplete(Exception):
    pass


def compare_structure(structure, deadline, record):
    import numpy as np
    from fast_solver_continuous import make_clock
    from reference_s5_family import AirFamily, initial_grid
    from reference_s5_structured_solve import StructuredCachedAirFamily
    from reference_s5_operator import S5Operator

    started = time.monotonic()
    clock = make_clock(max(.001, deadline-started), 768.)
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded structure comparison")
        clock.check()
    point = initial_grid()[1]
    original = AirFamily(structure, point, check=check)
    structured = StructuredCachedAirFamily(structure, point, check=check)
    operator = S5Operator(original)
    record.update(physical_coordinates=original.size, constructor_seconds=time.monotonic()-started,
                  inverse_comparison=[], passes=[])
    rng = np.random.default_rng(820)
    for columns in (1, 3):
        rhs = rng.normal(size=(original.size, columns)) * 1e-12
        if columns == 1:
            rhs = rhs[:, 0]
        expected = original.cap_factor.solve(rhs)
        actual = structured.cap_factor.solve(rhs)
        ratio = float(np.max(np.abs(actual-expected)/(2e-14+2e-12*np.abs(expected))))
        residual = float(np.max(np.abs(original.capacitance @ actual-rhs))/np.max(np.abs(rhs)))
        if ratio > 1. or residual > 2e-12:
            raise AssertionError("full-size inverse differs from original capacitance solve")
        row = dict(rhs_columns=columns, tolerance_ratio=ratio, relative_residual=residual, repeats=128)
        for name, family in (("original", original), ("structured", structured)):
            before = time.monotonic()
            for _ in range(128):
                check()
                family.cap_factor.solve(rhs)
            row[name+"_seconds"] = time.monotonic()-before
        record["inverse_comparison"].append(row)

    reference = []
    def errors(actual, expected, absolute):
        difference = np.abs(actual-expected)
        if not np.all(np.isfinite(difference)):
            raise ArithmeticError("nonfinite comparison")
        return float(np.max(difference)), float(np.max(difference/(absolute+2e-8*np.abs(expected))))

    for name, family in (("original", original), ("structured_cold", structured),
                         ("structured_warm", structured)):
        check()
        entry = dict(name=name, completed_intervals=0, state_max_abs_error=0.,
            state_max_tolerance_ratio=0., integral_max_abs_error=0.,
            integral_max_tolerance_ratio=0., input_max_tolerance_ratio=0.)
        record["passes"].append(entry)
        before_products = family.stats["operator_products"]
        before_plan = structured.plan_report() if family is structured else None
        sparse = family.sparse_step
        def step(state, row, slot, marker, start, end, **kwargs):
            index = entry["completed_intervals"]
            key = (int(row), int(slot), float(marker).hex(), float(start).hex(), float(end).hex())
            if name != "original":
                expected = reference[index]
                if key != expected[0]:
                    raise AssertionError("acquisition interval changed")
                _, ratio = errors(state, expected[1], 2e-12)
                entry["input_max_tolerance_ratio"] = max(entry["input_max_tolerance_ratio"], ratio)
                if ratio > 1.:
                    raise AssertionError("carried state exceeds original tolerance")
            result = sparse(state, row, slot, marker, start, end, **kwargs)
            if name == "original":
                reference.append((key, state.copy(), result[0].copy(), result[1].copy()))
            else:
                state_error, state_ratio = errors(result[0], expected[2], 2e-12)
                integral_error, integral_ratio = errors(result[1], expected[3], 2e-20)
                for key, value in (("state_max_abs_error", state_error),
                        ("state_max_tolerance_ratio", state_ratio),
                        ("integral_max_abs_error", integral_error),
                        ("integral_max_tolerance_ratio", integral_ratio)):
                    entry[key] = max(entry[key], value)
                if max(state_ratio, integral_ratio) > 1.:
                    raise AssertionError("structured propagation exceeds original tolerances")
            entry["completed_intervals"] += 1
            if entry["completed_intervals"] == 24:
                raise PrefixComplete()
            return result
        operator.family = family
        family.sparse_step = step
        begin = time.monotonic()
        try:
            operator.frame(np.zeros(family.size), 0)
            raise AssertionError("expected 24-interval boundary was not reached")
        except PrefixComplete:
            entry["status"] = "complete"
        finally:
            family.sparse_step = sparse
            entry["elapsed_seconds"] = time.monotonic()-begin
            entry["operator_products"] = family.stats["operator_products"]-before_products
            if family is structured:
                after_plan = structured.plan_report()
                entry["plan_delta"] = {key: after_plan[key]-before_plan[key] for key in
                    ("hits", "misses", "evictions", "setup_seconds", "core_seconds",
                     "setup_products", "core_products")}
    warm = record["passes"][-1]
    if warm["plan_delta"]["hits"] != 24 or warm["plan_delta"]["misses"] != 0:
        raise AssertionError("the identical warm prefix must reuse its 24 scalar plans")
    record.update(status="passed", elapsed_seconds=time.monotonic()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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
    report = dict(kind="exact_separable_capacitance_comparison", point=1,
        intervals_per_pass=24, heldaside_access=False, record_writer_created=False,
        total_seconds_limit=180., structure_seconds_limit=80., memory_mib_limit=768.,
        tolerance=dict(state_atol=2e-12, integral_atol=2e-20, rtol=2e-8), structures=[])
    def stop(signum, frame):
        raise TimeoutError("bounded comparison alarm")
    previous = signal.signal(signal.SIGALRM, stop)
    try:
        for structure in ("R2", "R3"):
            deadline = min(started+180., time.monotonic()+80.)
            signal.setitimer(signal.ITIMER_REAL, max(.001, deadline-time.monotonic()))
            row = dict(structure=structure, status="starting")
            report["structures"].append(row)
            try:
                compare_structure(structure, deadline, row)
            except Exception as error:
                row.update(status="failed", reason=f"{type(error).__name__}: {error}")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.)
            gc.collect()
        report["status"] = "passed" if all(r["status"] == "passed" for r in report["structures"]) else "failed"
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
