#!/usr/bin/env python3
"""One actual-interval full-state shifted-inverse comparison; no trajectory.

The published OpenMC IPF CRAM48 pole index20 is used only to measure one
linear system. No CRAM action, scientific record, fit or held-aside access.
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


POLE = complex(19.01323489060250, 1.194282058271408)
REPEATS = 64


def compare(structure, deadline, row):
    import numpy as np
    import scipy.sparse.linalg as sla
    from fast_solver_continuous import make_clock
    from reference_s5_family import AirFamily, PROBE_SITE, initial_grid
    from reference_s5_operator import S5Operator
    from reference_s5_shifted_structured import StructuredShiftedInverse

    clock = make_clock(max(.001, deadline-time.monotonic()), 768.)
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded shifted-inverse comparison")
        clock.check()
    row["phase"] = "construct_original_family_and_operator"
    before = time.perf_counter()
    family = AirFamily(structure, initial_grid()[1], check=check)
    operator = S5Operator(family)
    row["original_constructor_seconds"] = time.perf_counter()-before
    slot = PROBE_SITE[1]//24
    actions = np.flatnonzero((operator.template.action_row == PROBE_SITE[0])
                            & (np.arange(operator.template.actions) % 4 == slot))
    if not len(actions):
        raise ValueError("selected original acquisition action is absent")
    action = int(actions[0])
    raw = operator.raw_groups[action]
    window = operator.window(1)
    start, end = float(window.conv_aperture_start[raw[0]]), float(window.conv_aperture_end[raw[0]])
    duration = end-start
    row["interval"] = dict(frame=1, action=action, physical_row=PROBE_SITE[0], slot=slot,
        start_hex=start.hex(), end_hex=end.hex(), duration_hex=duration.hex(), duration_seconds=duration)
    g, _ = family.matrices(PROBE_SITE[0], slot)
    matrix = (POLE*family.capacitance+duration*g).tocsc()
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix_norm = float(np.max(np.asarray(abs(matrix).sum(axis=1)).ravel()))
    row.update(physical_state_count=family.size, matrix_nnz=int(matrix.nnz), solves=[])
    row["phase"] = "factor_original_complex_sparse_matrix"
    before = time.perf_counter()
    original = sla.splu(matrix)
    row["original_factor_seconds"] = time.perf_counter()-before
    check()
    row["phase"] = "construct_exact_structured_inverse"
    before = time.perf_counter()
    structured = StructuredShiftedInverse(family, POLE, duration, PROBE_SITE[0], slot)
    row["structured_constructor_seconds"] = time.perf_counter()-before
    row["structured_retained_numeric_bytes"] = structured.retained_numeric_bytes()
    row["ground_inverse_diagonal"] = [structured.ground_diagonal.real, structured.ground_diagonal.imag]
    check()
    indices = np.arange(family.size, dtype=float)[:, None]
    columns = np.arange(1, 4, dtype=float)[None, :]
    trial = np.sin(indices*columns*.173)+1j*np.cos(indices*columns*.219)
    rhs_all = np.asarray(family.capacitance @ trial, dtype=np.complex128)
    row["phase"] = "compare_and_time_solves"
    for count in (1, 3):
        rhs = rhs_all[:, 0].copy() if count == 1 else rhs_all.copy()
        expected, actual = original.solve(rhs), structured.solve(rhs)
        difference = np.abs(actual-expected)
        tolerance_ratio = float(np.max(difference/(2e-14+2e-12*np.abs(expected))))
        residual = matrix @ actual-rhs
        if count == 1:
            values, right, errors = (value[:, None] for value in (actual, rhs, residual))
        else:
            values, right, errors = actual, rhs, residual
        residual_norm, rhs_norm = np.max(np.abs(errors), axis=0), np.max(np.abs(right), axis=0)
        backward = float(np.max(residual_norm/(matrix_norm*np.max(np.abs(values), axis=0)+rhs_norm)))
        relative = float(np.max(residual_norm/rhs_norm))
        entry = dict(rhs_columns=count, max_solution_difference=float(np.max(difference)),
            relative_max_solution_difference=float(np.max(difference)/np.max(np.abs(expected))),
            solution_tolerance_ratio=tolerance_ratio, normwise_complex_backward_residual=backward,
            residual_relative_to_rhs=relative, repeats=REPEATS)
        row["solves"].append(entry)
        if not all(math.isfinite(v) for v in (tolerance_ratio, backward, relative)) or tolerance_ratio > 1 or relative > 2e-12:
            raise AssertionError("exact structured inverse differs from original sparse solve")
        for name, solver in (("original", original), ("structured", structured)):
            before = time.perf_counter()
            for _ in range(REPEATS):
                check()
                solver.solve(rhs)
            entry[name+"_mean_solve_seconds"] = (time.perf_counter()-before)/REPEATS
    single = row["solves"][0]["structured_mean_solve_seconds"]
    row["rough_costs"] = dict(warm_24_pole_single_rhs_seconds=24*single,
        cold_24_pole_constructor_and_single_rhs_seconds=24*(row["structured_constructor_seconds"]+single),
        one_row_4slots_2nominal_lengths_24poles_numeric_bytes=192*row["structured_retained_numeric_bytes"],
        caveat="one pole/point/action only; exact duration variants, other poles and augmented action accuracy unqualified")
    row.update(status="passed", phase="done", peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-seconds", type=float, default=100.)
    args = parser.parse_args()
    if not math.isfinite(args.total_seconds) or not 0 < args.total_seconds <= 110:
        parser.error("positive finite bound at most110 seconds required")
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
        parser.error("fresh output in existing reference_s5 run directory required")
    started = time.monotonic()
    report = dict(kind="one_exact_structured_complex_shift_comparison", point=1,
        pole=dict(real=POLE.real, imag=POLE.imag, method="IPF CRAM48", zero_based_index=20,
            source="https://docs.openmc.org/en/stable/_modules/openmc/deplete/cram.html"),
        matrix="pole*C + actual_duration*G", total_seconds_limit=args.total_seconds,
        memory_mib_limit=768., science_or_record_access=False, heldaside_access=False,
        exponential_action_computed=False, tolerance=dict(atol=2e-14, rtol=2e-12, relative_residual=2e-12),
        structures=[])
    def stop(signum, frame):
        raise TimeoutError("bounded shifted-inverse comparison alarm")
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
