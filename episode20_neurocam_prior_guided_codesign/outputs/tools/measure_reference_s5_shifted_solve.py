#!/usr/bin/env python3
"""One bounded complex-shift cost measurement, without exponential integration.

The shift is one published CRAM48 pole from OpenMC's primary implementation:
https://docs.openmc.org/en/stable/_modules/openmc/deplete/cram.html
theta = 19.01323489060250 + 1.194282058271408j (21st pair, zero-based 20).
This measures (theta*C + h*G) factorization and solves at an actual prescribed
aperture. It does not implement CRAM, approximate a trajectory, fit anything,
read a record or establish exponential-action accuracy for nonnormal systems.
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
REPEATS = 16


def measure(structure, deadline, row):
    import numpy as np
    import scipy.sparse.linalg as sla
    from fast_solver_continuous import make_clock
    from reference_s5_family import AirFamily, PROBE_SITE, initial_grid
    from reference_s5_operator import S5Operator

    clock = make_clock(max(.001, deadline-time.monotonic()), 768.)
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded shifted-solve measurement")
        clock.check()
    row["phase"] = "construct_original_circuit_and_acquisition"
    before = time.perf_counter()
    family = AirFamily(structure, initial_grid()[1], check=check)
    operator = S5Operator(family)
    row["constructor_seconds"] = time.perf_counter()-before
    row["physical_state_count"] = family.size
    slot = PROBE_SITE[1]//24
    actions = np.flatnonzero((operator.template.action_row == PROBE_SITE[0])
                            & (np.arange(operator.template.actions) % 4 == slot))
    if not len(actions):
        raise ValueError("the selected probe-row/source-slot action is absent")
    action = int(actions[0])
    raw = operator.raw_groups[action]
    window = operator.window(1)
    start = float(window.conv_aperture_start[raw[0]])
    end = float(window.conv_aperture_end[raw[0]])
    duration = end-start
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid original physical aperture")
    row["interval"] = dict(frame=1, action=action, physical_row=PROBE_SITE[0], slot=slot,
        start_hex=start.hex(), end_hex=end.hex(), duration_hex=duration.hex(), duration_seconds=duration,
        source="unchanged S5Operator.window(1) conversion aperture")
    row["phase"] = "construct_complex_shift"
    before = time.perf_counter()
    g, _ = family.matrices(PROBE_SITE[0], slot)
    matrix = (POLE*family.capacitance + duration*g).tocsc()
    matrix.sum_duplicates()
    matrix.sort_indices()
    row.update(matrix_nnz=int(matrix.nnz), matrix_assembly_seconds=time.perf_counter()-before)
    # Opaque factor storage may exceed the exported sparse arrays below.
    # RSS is separately measured; no dense inverse or matrix exponential.
    check()
    row["phase"] = "complex_sparse_factorization"
    before = time.perf_counter()
    factor = sla.splu(matrix)
    row["factor_seconds"] = time.perf_counter()-before
    check()
    lower, upper = factor.L, factor.U
    storage = sum(array.nbytes for sparse in (lower, upper)
                  for array in (sparse.data, sparse.indices, sparse.indptr))
    storage += factor.perm_r.nbytes + factor.perm_c.nbytes
    row.update(factor_L_nnz=int(lower.nnz), factor_U_nnz=int(upper.nnz),
        factor_exported_storage_bytes=int(storage),
        storage_scope="exported CSC L/U arrays and permutations; excludes opaque metadata and temporaries")
    del lower, upper
    matrix_norm = float(np.max(np.asarray(abs(matrix).sum(axis=1)).ravel()))
    # Deterministic linear-algebra RHSs, not measured/simulated EEG or circuit
    # trajectories. C scaling represents the shifted state's natural units.
    indices = np.arange(family.size, dtype=float)[:, None]
    columns = np.arange(1, 4, dtype=float)[None, :]
    trial = np.sin(indices*columns*.173) + 1j*np.cos(indices*columns*.219)
    rhs_all = np.asarray(family.capacitance @ trial, dtype=np.complex128)
    row["solves"] = []
    row["phase"] = "reused_factor_solves"
    for count in (1, 3):
        rhs = rhs_all[:, 0].copy() if count == 1 else rhs_all.copy()
        solution = factor.solve(rhs)  # warm-up, also establishes shape
        before = time.perf_counter()
        for _ in range(REPEATS):
            check()
            solution = factor.solve(rhs)
        elapsed = time.perf_counter()-before
        residual = matrix @ solution-rhs
        if count == 1:
            solution, rhs, residual = (x[:, None] for x in (solution, rhs, residual))
        residual_norm = np.max(np.abs(residual), axis=0)
        rhs_norm = np.max(np.abs(rhs), axis=0)
        denominator = matrix_norm*np.max(np.abs(solution), axis=0)+rhs_norm
        backward = residual_norm/denominator
        rhs_relative = residual_norm/rhs_norm
        if not (np.all(np.isfinite(backward)) and np.all(np.isfinite(rhs_relative))):
            raise ArithmeticError("nonfinite complex shifted-solve residual")
        row["solves"].append(dict(rhs_columns=count, repeats=REPEATS,
            mean_solve_seconds=elapsed/REPEATS,
            normwise_complex_backward_residual=float(np.max(backward)),
            residual_relative_to_rhs=float(np.max(rhs_relative)),
            backward_residual_definition="max_j ||Mx_j-b_j||inf/(||M||inf||x_j||inf+||b_j||inf)"))
    single = row["solves"][0]["mean_solve_seconds"]
    triple = row["solves"][1]["mean_solve_seconds"]
    row["rough_costs"] = dict(
        warm_24_pole_solve_seconds=24*single,
        warm_24_pole_three_rhs_solve_seconds=24*triple,
        cold_24_pole_factor_plus_solve_seconds=24*(row["factor_seconds"]+single),
        cold_24_pole_factor_plus_three_rhs_solve_seconds=24*(row["factor_seconds"]+triple),
        all_rows_slots_two_nominal_durations_24_poles_factor_count=64*4*2*24,
        extrapolated_exported_factor_bytes=64*4*2*24*storage,
        one_row_exported_factor_bytes=4*2*24*storage,
        caveat="one pole/point/interval only; omits other-pole costs, forcing/integrals, refinement and exact-duration variants")
    row.update(status="measured", phase="done", peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-seconds", type=float, default=80.)
    args = parser.parse_args()
    if not math.isfinite(args.total_seconds) or not 0 < args.total_seconds <= 80:
        parser.error("a finite positive measurement limit of at most 80 seconds is required")
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
    report = dict(kind="one_complex_shift_factor_and_solve_cost", point=1,
        pole=dict(real=POLE.real, imag=POLE.imag, method="IPF CRAM48", zero_based_index=20,
            source="https://docs.openmc.org/en/stable/_modules/openmc/deplete/cram.html"),
        matrix="pole*C + actual_duration*G", exponential_action_computed=False,
        science_or_record_access=False, heldaside_access=False,
        total_seconds_limit=args.total_seconds, memory_mib_limit=768., structures=[])
    def stop(signum, frame):
        raise TimeoutError("bounded shifted-solve alarm")
    previous = signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, args.total_seconds)
    try:
        for structure in ("R2", "R3"):
            row = dict(structure=structure, status="starting")
            report["structures"].append(row)
            try:
                measure(structure, started+args.total_seconds, row)
            except Exception as exc:
                row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
                if isinstance(exc, (TimeoutError, MemoryError)):
                    break
            gc.collect()
        report["status"] = "measured" if len(report["structures"]) == 2 and all(
            row["status"] == "measured" for row in report["structures"]) else "incomplete_or_failed"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous)
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(report, allow_nan=False), flush=True)
    return 0 if report["status"] == "measured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
