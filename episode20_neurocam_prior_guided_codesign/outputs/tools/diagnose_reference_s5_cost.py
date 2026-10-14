#!/usr/bin/env python3
"""Bounded full-size S5 timing probe; no response records or fitted outcomes.

Invoke the unchanged frame operator and stop after its first 24 interval
advances (eight actions). Instrumentation exists only in this process. Neither
live source files nor worker state are modified. All times are inclusive:
nested section totals must not be added as if they were disjoint.
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


class ProbeStop(Exception):
    pass


class PrefixComplete(Exception):
    pass


def add_section(sections, name, elapsed, products=0):
    row = sections.setdefault(name, dict(calls=0, seconds=0., vector_products=0))
    row["calls"] += 1
    row["seconds"] += elapsed
    row["vector_products"] += products


def instrumented_prefix(structure, parameters, *, total_deadline, seconds, memory_mib, control):
    started = time.monotonic()
    deadline = min(total_deadline, started + seconds)
    import numpy as np
    import reference_s5_family as source
    from fast_solver_continuous import make_clock
    from reference_s5_operator import S5Operator

    clock = make_clock(max(.001, deadline - time.monotonic()), memory_mib)
    record = dict(structure=structure, point=1, target_completed_intervals=24,
        completed_intervals=0, attempted_intervals=0, physical_coordinates=None,
        status="starting", stage="family_construction", sections={},
        mux_cycles=[dict(cycle=0, sections={}, completed_intervals=0),
                    dict(cycle=1, sections={}, completed_intervals=0)],
        outcome_access=False, record_writer_created=False)
    # Publish the operational dictionary to the caller before any expensive
    # construction, so a signal or constructor refusal still yields timings.
    control["current_record"] = record
    family = operator = None
    original_forward = source._SineIntegralOperator._matmat
    original_adjoint = source._SineIntegralOperator._rmatmat

    def store(name, elapsed, products=0, cycle=None):
        add_section(record["sections"], name, elapsed, products)
        if cycle is not None:
            add_section(record["mux_cycles"][cycle]["sections"], name, elapsed, products)

    def current_cycle():
        if record["stage"] != "frame_prefix":
            return None
        return min(record["completed_intervals"] // 12, 1)

    def check():
        begin = time.monotonic()
        cycle = current_cycle()
        try:
            if begin >= total_deadline:
                raise ProbeStop("total_time_limit")
            if begin >= deadline:
                raise ProbeStop("structure_time_limit")
            clock.check()
        finally:
            store("check", time.monotonic() - begin, cycle=cycle)

    def timed(name, function):
        def call(*args, **kwargs):
            begin, cycle = time.monotonic(), current_cycle()
            try:
                return function(*args, **kwargs)
            finally:
                store(name, time.monotonic() - begin, cycle=cycle)
        return call

    def product_wrapper(name, function):
        def call(instance, values):
            begin, cycle = time.monotonic(), current_cycle()
            try:
                return function(instance, values)
            finally:
                store(name, time.monotonic() - begin, int(values.shape[1]), cycle)
        return call

    control["phase"] = structure + "_prefix"
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
    try:
        check()
        family = timed("family_constructor", source.AirFamily)(structure, parameters, check=check)
        record["physical_coordinates"] = family.size
        # These wrappers retain the original bound methods, parameters and
        # state arithmetic; only wall-time accounting is added around calls.
        family.matrices = timed("matrices", family.matrices)
        family.certify_controlled_rails = timed("controlled_rail_check", family.certify_controlled_rails)
        original_sparse = family.sparse_step

        def sparse(*args, **kwargs):
            record["attempted_intervals"] += 1
            cycle = current_cycle()
            begin = time.monotonic()
            try:
                result = original_sparse(*args, **kwargs)
            finally:
                store("sparse_step", time.monotonic() - begin, cycle=cycle)
            record["completed_intervals"] += 1
            record["mux_cycles"][cycle]["completed_intervals"] += 1
            if record["completed_intervals"] == 24:
                raise PrefixComplete()
            return result

        family.sparse_step = sparse
        source._SineIntegralOperator._matmat = product_wrapper("forward_operator", original_forward)
        source._SineIntegralOperator._rmatmat = product_wrapper("adjoint_operator", original_adjoint)
        record["stage"] = "acquisition_construction"
        operator = timed("acquisition_constructor", S5Operator)(family)
        record["stage"] = "frame_prefix"
        timed("frame_prefix", operator.frame)(np.zeros(family.size), 0)
        raise AssertionError("timing probe unexpectedly completed a whole frame")
    except PrefixComplete:
        record["status"] = "timed_original_first_eight_actions"
    except ProbeStop as error:
        record.update(status="bounded_stop", reason=str(error))
    except TimeoutError as error:
        record.update(status="bounded_stop", reason="original_clock_limit: " + str(error))
    except MemoryError as error:
        record.update(status="bounded_stop", reason="memory_limit: " + str(error))
    except Exception as error:
        record.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        source._SineIntegralOperator._matmat = original_forward
        source._SineIntegralOperator._rmatmat = original_adjoint
        record["elapsed_seconds"] = time.monotonic() - started
        record["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if family is not None:
            record["solver_counts"] = {key: family.stats[key] for key in
                ("operator_products", "dense_exponentials", "sparse_exponentials")}
            record["cache"] = family.cache.report()
        # Restore the total alarm while releasing construction memory. The
        # original methods remain unchanged in every other Python process.
        remaining = total_deadline - time.monotonic()
        if remaining > 0:
            control["phase"] = "between_structures"
            signal.setitimer(signal.ITIMER_REAL, remaining)
    return record


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--total-seconds", type=float, default=180.)
    parser.add_argument("--structure-seconds", type=float, default=80.)
    parser.add_argument("--memory-mib", type=float, default=768.)
    args = parser.parse_args()
    if (not math.isfinite(args.total_seconds) or not 0 < args.total_seconds <= 180
            or not math.isfinite(args.structure_seconds) or not 0 < args.structure_seconds <= 80
            or not math.isfinite(args.memory_mib) or not 0 < args.memory_mib <= 768):
        parser.error("limits must be positive and at most 180 s total, 80 s per structure, 768 MiB")
    hostname = socket.gethostname()
    if not os.environ.get("SLURM_JOB_ID") or "-ln" in hostname:
        parser.error("the coordinator-owned compute allocation is required")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        parser.error("one explicitly allocated task CPU is required")
    thread_keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
    if any(os.environ.get(key) != "1" for key in thread_keys):
        parser.error("all six numerical thread counts must be one")
    scope = Path(__file__).resolve().parents[1] / "runs" / "reference_s5"
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir() or scope not in output.parents:
        parser.error("a fresh JSON path under the existing reference_s5 run directory is required")
    total_deadline = started + args.total_seconds
    report = dict(hostname=hostname, job_id=os.environ["SLURM_JOB_ID"],
        status="started_timing_only", total_limit_seconds=args.total_seconds,
        per_structure_limit_seconds=args.structure_seconds, peak_RSS_limit_mib=args.memory_mib,
        records=[], measurements="inclusive_sections_do_not_sum_nested_timings",
        check_measurement="original production make_clock.check plus timing/total-deadline instrumentation",
        probe="original_full_array_point01_first8actions_first24intervals",
        outcome_access=False, scientific_estimate=False, modified_live_source=False)
    control = dict(phase="imports", current_record=None)

    def stop(signum, _frame):
        if signum == signal.SIGALRM:
            kind = "total_time_limit" if time.monotonic() >= total_deadline else "structure_time_limit"
        else:
            kind = f"signal_{signum}"
        raise ProbeStop(kind + "_at_" + control["phase"])

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, total_deadline - time.monotonic()))
        from reference_s5_family import initial_grid
        parameters = initial_grid()[1]
        for structure in ("R2", "R3"):
            if time.monotonic() >= total_deadline:
                raise ProbeStop("total_time_limit_before_" + structure)
            control["phase"] = structure + "_imports"
            signal.setitimer(signal.ITIMER_REAL,
                            min(args.structure_seconds, total_deadline - time.monotonic()))
            row = instrumented_prefix(structure, parameters, total_deadline=total_deadline,
                seconds=args.structure_seconds, memory_mib=args.memory_mib, control=control)
            report["records"].append(row)
            control["current_record"] = None
            print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
            gc.collect()
            if row.get("reason", "").startswith(("memory_limit", "signal_")):
                raise ProbeStop("stop_after_" + row["reason"])
        report["status"] = "timing_attempts_finished"
    except ProbeStop as error:
        report.update(status="bounded_stop", reason=str(error))
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        current = control["current_record"]
        if current is not None and not any(current is row for row in report["records"]):
            report["records"].append(current)
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
