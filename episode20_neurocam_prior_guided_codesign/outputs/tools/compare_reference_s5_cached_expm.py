#!/usr/bin/env python3
"""Bounded original/cached full-size prefix comparison without record access.

Run unchanged S5Operator.frame for the first eight actions (24 interval
advances), first with the original family, then with cold and warm plan caches.
The original template, switch charges, timing and operator arithmetic are reused.
Compare every input/output physical state and every hold integral in RAM; write
only error aggregates, timings and counts. No HDF/record, amplitude or fit access.
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


class BoundedStop(Exception):
    pass


def compare_structure(structure, *, deadline, memory_mib, record):
    import numpy as np
    from fast_solver_continuous import make_clock
    from reference_s5_family import AirFamily, initial_grid
    from reference_s5_cached_expm import CachedAirFamily
    from reference_s5_operator import S5Operator

    started = time.monotonic()
    clock = make_clock(max(.001, deadline - started), memory_mib)
    def check():
        if time.monotonic() >= deadline:
            raise BoundedStop("structure_time_limit")
        clock.check()

    check()
    point = initial_grid()[1]
    original = AirFamily(structure, point, check=check)
    cached = CachedAirFamily(structure, point, check=check)
    operator = S5Operator(original)
    record.update(physical_coordinates=original.size,
                  constructor_seconds=time.monotonic() - started, passes=[])
    reference = []

    def errors(actual, expected, absolute):
        difference = np.abs(actual - expected)
        if not np.all(np.isfinite(difference)):
            raise ArithmeticError("nonfinite comparison difference")
        return (float(np.max(difference)),
                float(np.max(difference / (absolute + 2e-8 * np.abs(expected)))))

    for name, family in (("original", original), ("cached_cold", cached), ("cached_warm", cached)):
        check()
        entry = dict(name=name, completed_intervals=0, elapsed_seconds=0.,
            state_max_abs_error=0., state_max_tolerance_ratio=0.,
            integral_max_abs_error=0., integral_max_tolerance_ratio=0.,
            input_max_tolerance_ratio=0., cycles=[
                dict(cycle=i, intervals=0, seconds=0., operator_products=0) for i in range(2)])
        record["passes"].append(entry)
        before_products = family.stats["operator_products"]
        before_plan = cached.plan_report() if family is cached else None
        sparse = family.sparse_step

        def step(state, row, slot, marker, start, end, **kwargs):
            index = entry["completed_intervals"]
            cycle = entry["cycles"][index // 12]
            call_key = (int(row), int(slot), float(marker).hex(),
                        float(start).hex(), float(end).hex())
            if name != "original":
                expected = reference[index]
                if call_key != expected[0]:
                    raise AssertionError("original acquisition interval identity changed")
                _, input_ratio = errors(state, expected[1], 2e-12)
                entry["input_max_tolerance_ratio"] = max(entry["input_max_tolerance_ratio"], input_ratio)
                if input_ratio > 1:
                    raise AssertionError("carried physical input state outside original tolerance")
            interval_started = time.monotonic()
            products = family.stats["operator_products"]
            try:
                result = sparse(state, row, slot, marker, start, end, **kwargs)
            finally:
                cycle["seconds"] += time.monotonic() - interval_started
                cycle["operator_products"] += family.stats["operator_products"] - products
            if name == "original":
                reference.append((call_key, state.copy(), result[0].copy(), result[1].copy()))
            else:
                state_abs, state_ratio = errors(result[0], expected[2], 2e-12)
                integral_abs, integral_ratio = errors(result[1], expected[3], 2e-20)
                for key, value in (("state_max_abs_error", state_abs),
                        ("state_max_tolerance_ratio", state_ratio),
                        ("integral_max_abs_error", integral_abs),
                        ("integral_max_tolerance_ratio", integral_ratio)):
                    entry[key] = max(entry[key], value)
                if max(state_ratio, integral_ratio) > 1:
                    raise AssertionError("cached propagation outside original state/integral tolerance")
            entry["completed_intervals"] += 1
            cycle["intervals"] += 1
            if entry["completed_intervals"] == 24:
                raise PrefixComplete()
            return result

        # Reuse the original constructor's acquisition and charge arrays, with
        # the identical physical family. Only sparse propagation is substituted.
        operator.family = family
        family.sparse_step = step
        begin = time.monotonic()
        try:
            operator.frame(np.zeros(family.size), 0)
            raise AssertionError("prefix comparison unexpectedly completed a whole frame")
        except PrefixComplete:
            entry["status"] = "complete"
        finally:
            family.sparse_step = sparse
            entry["elapsed_seconds"] = time.monotonic() - begin
            entry["operator_products"] = family.stats["operator_products"] - before_products
            if family is cached:
                after_plan = cached.plan_report()
                entry["plan_delta"] = {key: after_plan[key] - before_plan[key] for key in
                    ("hits", "misses", "evictions", "setup_seconds", "core_seconds",
                     "setup_products", "core_products")}
                entry["plan_entries"] = after_plan["entries"]
    warm = record["passes"][-1]
    if warm["plan_delta"]["hits"] != 24 or warm["plan_delta"]["misses"] != 0:
        raise AssertionError("repeated exact prefix did not reuse all 24 scalar plans")
    record.update(status="all_prefix_comparisons_passed", plan_cache=cached.plan_report(),
                  elapsed_seconds=time.monotonic() - started)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-seconds", type=float, default=180.)
    parser.add_argument("--structure-seconds", type=float, default=80.)
    parser.add_argument("--memory-mib", type=float, default=768.)
    args = parser.parse_args()
    if (not math.isfinite(args.total_seconds) or not 0 < args.total_seconds <= 180
            or not math.isfinite(args.structure_seconds) or not 0 < args.structure_seconds <= 80
            or not math.isfinite(args.memory_mib) or not 0 < args.memory_mib <= 768):
        parser.error("positive limits required: at most 180 s total, 80 s/structure, 768 MiB")
    hostname = socket.gethostname()
    if (not os.environ.get("SLURM_JOB_ID", "").isdigit() or "-ln" in hostname
            or not hostname.startswith("sh") or os.environ.get("SLURM_CPUS_PER_TASK") != "1"):
        parser.error("the coordinator-owned one-CPU compute step is required")
    threads = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
    if any(os.environ.get(key) != "1" for key in threads):
        parser.error("all six numerical thread settings must be one")
    permitted = Path(__file__).resolve().parents[1] / "runs" / "reference_s5"
    output = args.output.resolve()
    if not output.is_relative_to(permitted) or output.exists() or not output.parent.is_dir():
        parser.error("output must be a fresh file in an existing reference_s5 run directory")
    report = dict(kind="isolated_cached_sparse_action_comparison", structures=[],
        point=1, intervals_per_pass=24, heldaside_access=False, record_writer_created=False,
        frame_operator="unchanged S5Operator.frame; zero-state first eight actions",
        tolerance=dict(state_atol=2e-12, integral_atol=2e-20, rtol=2e-8),
        limits=dict(total_seconds=args.total_seconds, structure_seconds=args.structure_seconds,
                    memory_mib=args.memory_mib), status="starting")
    total_deadline = started + args.total_seconds
    def stop(signum, frame):
        raise BoundedStop("bounded_alarm")
    prior_handler = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.001, total_deadline - time.monotonic()))
    try:
        for structure in ("R2", "R3"):
            if time.monotonic() >= total_deadline:
                raise BoundedStop("total_time_limit")
            row = dict(structure=structure, status="starting")
            report["structures"].append(row)
            deadline = min(total_deadline, time.monotonic() + args.structure_seconds)
            signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
            try:
                compare_structure(structure, deadline=deadline, memory_mib=args.memory_mib, record=row)
            except (BoundedStop, TimeoutError, MemoryError) as error:
                row.update(status="bounded_stop", reason=f"{type(error).__name__}: {error}")
            except Exception as error:
                row.update(status="failed", reason=f"{type(error).__name__}: {error}")
            finally:
                row["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                signal.setitimer(signal.ITIMER_REAL, max(.001, total_deadline - time.monotonic()))
            gc.collect()
        report["status"] = ("all_comparisons_passed" if all(
            row["status"] == "all_prefix_comparisons_passed" for row in report["structures"])
            else "incomplete_or_failed")
    except (BoundedStop, TimeoutError, MemoryError) as error:
        report.update(status="bounded_stop", reason=f"{type(error).__name__}: {error}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, prior_handler)
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if report["status"] == "all_comparisons_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
