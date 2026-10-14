#!/usr/bin/env python3
"""Bounded full-array, independently charged original-versus-CRAM16 prefix.

Frame0 actions0..7 are the cold prefix; actions8..15 continue their carried
states and original clock. Cache hits/misses are reported rather than assumed.
All physical interval states and24 hold integrals are compared. No complete
frame, scientific record, calibration or held-aside observation is produced.
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


def compare(structure, deadline, row):
    import numpy as np
    from fast_solver_continuous import make_clock
    from reference_s5_family import AirFamily, initial_grid
    from reference_s5_operator import S5Operator
    from reference_s5_cram16_prototype import Cram16PrototypeAirFamily

    clock = make_clock(max(.001, deadline-time.monotonic()), 768.)
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded CRAM16 prefix comparison")
        clock.check()
    def errors(actual, expected, absolute):
        difference = np.abs(actual-expected)
        if not np.all(np.isfinite(difference)):
            raise ArithmeticError("nonfinite prefix comparison")
        return float(np.max(difference)), float(np.max(difference/(absolute+2e-8*np.abs(expected))))
    row["phase"] = "construct_independent_original_and_candidate_operators"
    before = time.perf_counter()
    original = AirFamily(structure, initial_grid()[1], check=check)
    candidate = Cram16PrototypeAirFamily(structure, initial_grid()[1], check=check)
    original_operator = S5Operator(original)
    candidate_operator = S5Operator(candidate)
    row.update(constructor_seconds=time.perf_counter()-before, physical_states=original.size,
        passes=[], charge_tolerance_ratio=0., acquisition_actions=16)
    for name in ("action_row", "action_start", "action_end", "action_external_volts",
                 "conv_aperture_start", "conv_aperture_end"):
        if not np.array_equal(getattr(original_operator.template, name), getattr(candidate_operator.template, name)):
            raise AssertionError("original acquisition template differs")
    for index in range(16):
        if not np.array_equal(original_operator.raw_groups[index], candidate_operator.raw_groups[index]):
            raise AssertionError("original aperture membership differs")
        _, ratio = errors(candidate_operator.charge_deltas[index], original_operator.charge_deltas[index], 2e-12)
        row["charge_tolerance_ratio"] = max(row["charge_tolerance_ratio"], ratio)
        if ratio > 1:
            raise AssertionError("independent original charge solve differs")
    reference = []
    row["phase"] = "full_state_prefix_comparison"
    for name, family, operator in (("original", original, original_operator),
                                   ("cram16", candidate, candidate_operator)):
        entry = dict(name=name, completed_intervals=0, blocks=[], state_max_abs_error=0.,
            state_max_tolerance_ratio=0., integral_max_abs_error=0.,
            integral_max_tolerance_ratio=0., input_max_tolerance_ratio=0.)
        row["passes"].append(entry)
        sparse = family.sparse_step
        pass_started = time.perf_counter()
        block_started = pass_started
        block_snapshot = candidate.cram_report() if name == "cram16" else None
        def step(state, physical_row, slot, marker, start, end, **kwargs):
            nonlocal block_started, block_snapshot
            index = entry["completed_intervals"]
            key = (int(physical_row), int(slot), float(marker).hex(), float(start).hex(), float(end).hex())
            if name != "original":
                expected = reference[index]
                if key != expected[0]:
                    raise AssertionError("original physical interval differs")
                _, ratio = errors(state, expected[1], 2e-12)
                entry["input_max_tolerance_ratio"] = max(entry["input_max_tolerance_ratio"], ratio)
                if ratio > 1:
                    raise AssertionError("carried CRAM16 input exceeds original tolerance")
            result = sparse(state, physical_row, slot, marker, start, end, **kwargs)
            if name == "original":
                reference.append((key, state.copy(), result[0].copy(), result[1].copy()))
            else:
                state_error, state_ratio = errors(result[0], expected[2], 2e-12)
                integral_error, integral_ratio = errors(result[1], expected[3], 2e-20)
                for field, value in (("state_max_abs_error", state_error),
                        ("state_max_tolerance_ratio", state_ratio),
                        ("integral_max_abs_error", integral_error),
                        ("integral_max_tolerance_ratio", integral_ratio)):
                    entry[field] = max(entry[field], value)
                if max(state_ratio, integral_ratio) > 1:
                    raise AssertionError("CRAM16 state or hold integral exceeds original tolerance")
            entry["completed_intervals"] += 1
            if entry["completed_intervals"] % 24 == 0:
                block = dict(actions=8, intervals=24,
                    kind="cold" if entry["completed_intervals"] == 24 else "carried_continuation",
                    elapsed_seconds=time.perf_counter()-block_started)
                if name == "cram16":
                    after = candidate.cram_report()
                    block["cache_delta"] = {field: after[field]-block_snapshot[field] for field in
                        ("hits", "misses", "evictions", "constructor_seconds", "action_seconds", "actions")}
                    block["numeric_cache_bytes"] = after["numeric_cache_bytes"]
                    block_snapshot = after
                entry["blocks"].append(block)
                block_started = time.perf_counter()
            if entry["completed_intervals"] == 48:
                raise PrefixComplete()
            return result
        family.sparse_step = step
        try:
            operator.frame(np.zeros(family.size), 0)
            raise AssertionError("expected16-action prefix boundary was not reached")
        except PrefixComplete:
            entry["status"] = "complete"
        finally:
            family.sparse_step = sparse
            entry["elapsed_seconds"] = time.perf_counter()-pass_started
            if name == "cram16":
                entry["cram_report"] = candidate.cram_report()
    row.update(status="passed", phase="done", peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-seconds", type=float, default=110.)
    args = parser.parse_args()
    if not math.isfinite(args.total_seconds) or not 0 < args.total_seconds <= 110:
        parser.error("positive finite limit at most110 seconds required")
    host = socket.gethostname()
    if (not os.environ.get("SLURM_JOB_ID", "").isdigit() or "-ln" in host
            or not host.startswith("sh") or os.environ.get("SLURM_CPUS_PER_TASK") != "1"):
        parser.error("coordinator-owned one-CPU compute step required")
    if any(os.environ.get(key) != "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")):
        parser.error("all six numerical thread counts must be one")
    output = args.output.resolve()
    permitted = Path(__file__).resolve().parents[1]/"runs"/"reference_s5"
    if not output.is_relative_to(permitted) or output.exists() or not output.parent.is_dir():
        parser.error("fresh output in an existing reference_s5 run directory required")
    started = time.monotonic()
    report = dict(kind="isolated_full_augmentation_cram16_prefix", point=1,
        coefficient_source="https://docs.openmc.org/en/stable/_modules/openmc/deplete/cram.html",
        actions_per_structure=16, blocks="first8 cold; next8 carried continuation at actual frame0 clock",
        independent_charge_operators=True, full_frame_qualified=False,
        record_written=False, science_or_heldaside_access=False, total_seconds_limit=args.total_seconds,
        memory_mib_limit=768., tolerance=dict(state_atol=2e-12, integral_atol=2e-20, rtol=2e-8), structures=[])
    def stop(signum, frame):
        raise TimeoutError("bounded CRAM16 prefix comparison alarm")
    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, args.total_seconds)
    try:
        for structure in ("R2", "R3"):
            row = dict(structure=structure, status="starting")
            report["structures"].append(row)
            try:
                compare(structure, started+args.total_seconds, row)
            except Exception as error:
                row.update(status="failed", reason=f"{type(error).__name__}: {error}")
                if isinstance(error, (MemoryError, TimeoutError)):
                    break
            gc.collect()
        report["status"] = "passed" if len(report["structures"]) == 2 and all(
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
