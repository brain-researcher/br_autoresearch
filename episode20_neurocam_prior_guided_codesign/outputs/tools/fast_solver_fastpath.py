"""Run the fast path (O6 particular solutions + O7 multirate WR + O4) on cycle 0.

Exact inputs only (the analytic tones of the saved draw and the 11-Hz field).
Same world, circuit, schedule, noise draw and split intervals as the other
Stage 1 drivers; whole dwells are processed (the 5-us state is obtained from
an extra coarse cut in the first dwell). Development test only. Outputs in
--run-dir: settings.json, solver_intervals.npz (window work totals stored as
window_work), solver_checkpoints.npz, solver_report.json.
"""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time

from fast_solver_stage1 import WORK_FIELDS, build_reference_setting, build_splits, _write_json, EPISODE_ROOT

WINDOWS = (("5us", 5e-6), ("dwell", 62.5e-6), ("frame", 4e-3))


def prefix_work(contributions, end, fields=WORK_FIELDS):
    """Impulses belong to [0,end); integrated segments end in (0,end]."""
    selected = [values for t, boundary, values in contributions
                if (t < end if boundary else t <= end + 1e-15)]
    return {field: math.fsum(values.get(field, 0.0) for values in selected) for field in fields}


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--slots", type=int, required=True, help="multiple of 25 (whole dwells)")
    parser.add_argument("--deadline-seconds", type=float, required=True)
    parser.add_argument("--segment-levels", type=int, default=12)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--wr-tolerance", type=float, default=1e-10)
    parser.add_argument("--offsets-json", type=Path, default=None)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if (not run_dir.is_dir() or not run_dir.is_relative_to(EPISODE_ROOT / "outputs/runs/fast_forward_stage1/solver")
            or any((run_dir / n).exists() for n in ("settings.json", "solver_report.json", "solver_intervals.npz"))):
        raise SystemExit("supply a fresh existing directory under runs/fast_forward_stage1/solver")
    host, job = socket.gethostname(), os.environ.get("SLURM_JOB_ID", "")
    if "-ln" in host or not job:
        raise SystemExit("compute allocation required; refusing to run on a login node")
    if args.slots % 25:
        raise SystemExit("the fast path processes whole dwells")
    deadline = float(args.deadline_seconds)
    stop = {"signal": None}
    signal.signal(signal.SIGTERM, lambda s, f: stop.update(signal=s))
    signal.signal(signal.SIGINT, lambda s, f: stop.update(signal=s))

    def check():
        if stop["signal"] is not None:
            raise TimeoutError(f"signal {stop['signal']} received")
        if time.monotonic() - started >= deadline:
            raise TimeoutError("internal deadline reached")

    report = dict(host=host, slurm_job_id=job, slots_requested=args.slots, mode="fastpath_exact",
                  deadline_seconds=deadline, status="failed", segment_levels=args.segment_levels,
                  degree=args.degree, wr_tolerance=args.wr_tolerance,
                  completed_intervals=0, physical_frontier_seconds=0.0,
                  offsets_fixture=None if args.offsets_json is None else str(args.offsets_json))
    phase = "imports"
    records = None
    ck = dict(times=[], slots=[], states=[], energy=[])
    contributions = []          # (time, is_boundary_impulse, {field: value})
    upstream_audits, downstream_audits, segment_diagnostics = [], [], []
    fast = None
    timings = {}
    audit_seconds = 0.0
    try:
        import numpy as np
        from fast_solver_engine import FastArraySolver, spec_from_cpe_array
        from fast_solver_multirate import MultirateSolver, ToneSet
        from fast_solver_production import ExactInputs, ProductionWorkModel
        offsets = None if args.offsets_json is None else json.loads(args.offsets_json.read_text())
        _write_json(run_dir / "settings.json", dict(report, offsets_volts=offsets, reference_modules_modified=False))
        phase = "setting"
        clock = time.monotonic()
        setting = build_reference_setting(check, offsets)
        actions, readout, plan, splits = build_splits(setting, args.slots, check)
        array = setting["array"]
        timings["setting_seconds"] = time.monotonic() - clock
        phase = "fast_setup"
        clock = time.monotonic()
        spec = spec_from_cpe_array(array)
        work_model = ProductionWorkModel(array)
        base = FastArraySolver(spec, degree=args.degree, wr_tolerance=args.wr_tolerance, work_model=work_model,
                               energy_audit=True)
        exact = ExactInputs(array, setting["noise"], setting["raw_time"], setting["contacts"])
        tones = ToneSet(exact, array)
        fast = MultirateSolver(base, tones, work_model, segment_levels=args.segment_levels)
        timings["fast_setup_seconds"] = time.monotonic() - clock
        n = len(splits)
        records = dict(t_start=np.full(n, np.nan), t_end=np.full(n, np.nan), hold_mean=np.full((n, 24), np.nan),
                       hold_end=np.full((n, 24), np.nan), source_mean=np.full((n, 64), np.nan),
                       work=np.full((n, len(WORK_FIELDS)), np.nan), sweeps=np.full(n, np.nan),
                       wr_change=np.full(n, np.nan), rep_pad=np.full(n, np.nan), rep_source=np.full(n, np.nan),
                       clip_events=np.zeros(n, np.int64), elapsed=np.full(n, np.nan), audit=np.full((n, 2), np.nan),
                       downstream_audit=np.full((n, 2), np.nan))
        state_size = len(array.state_ids)
        P = spec.pads
        holds_local = np.asarray(spec.holds) - base.n_up
        report["diagnostic_availability"] = {
            "source_mean": {"available": False, "reason": "source means are not retained by the dwell driver"},
            "interval_solver": {"available": False, "reason": "WR sweeps and representation checks belong to coarse segments; see window_solver_diagnostics"},
            "interval_work": {"available": False, "reason": "interval work stores downstream terms only; use complete window_work"},
            "interval_energy_audit": {"available": False, "reason": "upstream audit uses coarse segments; use window_energy_audit"}}
        report["work_accounting"] = {"window_work_complete": False,
                                      "boundary_ownership": "impulses in [0,end), integrated segments ending in (0,end]"}

        def checkpoint(t, slot):
            state = fast.full_state(t)
            ck["times"].append(t)
            ck["slots"].append(slot)
            ck["states"].append(state)
            ck["energy"].append((float(array._physical_energy(state, np.zeros(4096))),
                                 float(0.5 * state @ (spec.capacitance @ state))))

        phase = "propagation"
        clock_prop = time.monotonic()
        first = plan.intervals[splits[0][1]]
        fast.set_full_state(np.asarray(array.volts), 0.0, first.gate_row)
        checkpoint(0.0, 0)
        for dwell in range(args.slots // 25):
            check()
            dwell_splits = [(i, s) for i, s in enumerate(splits) if s[0] // 25 == dwell]
            _, (_, hold_index, t_r, _) = dwell_splits[0]
            held = plan.intervals[hold_index]
            row = held.gate_row
            source_charge = np.zeros(state_size)
            for node, value in held.injected_charge_coulomb:
                if P <= node < P + 64:
                    source_charge[node] = value
            entry, w_src = fast.begin_dwell(row, t_r, source_charge)
            contributions.append((t_r, True, dict(supplied_charge_boundary_work_joule=w_src)))
            base_fractions = fast.fractions
            extra = [5e-6 / fast.dwell] if dwell == 0 else []
            fast.fractions = sorted(set(base_fractions) | set(extra))
            seg_bounds = [t_r + fast.dwell * f for f in fast.fractions]
            # snap coarse bounds to the exact schedule times they represent (rounding only)
            split_times = sorted({s[2] for _, s in dwell_splits} | {dwell_splits[-1][1][3]})
            seg_bounds[0], seg_bounds[-1] = t_r, dwell_splits[-1][1][3]
            for i in range(1, len(seg_bounds) - 1):
                near = min(split_times, key=lambda value: abs(value - seg_bounds[i]))
                if abs(near - seg_bounds[i]) <= 1e-15:
                    seg_bounds[i] = near
            seg_records = fast.coarse_dwell(entry, t_r)
            fast.fractions = base_fractions
            a_end_coarse = base.a_y.copy()
            base.a_y = seg_records[0]["ya0"].copy()
            for i, record in enumerate(seg_records):
                check()
                seg_t0, seg_t1 = seg_bounds[i], seg_bounds[i + 1]
                part = fast.particular_works(row, seg_t0, seg_t1)
                hom = fast.homogeneous_works(entry, [record], t_r)
                contributions.append((seg_t1, False, {f: part[f] + hom[f] for f in part}))
                audit_clock = time.monotonic()
                try:
                    audit16 = fast.upstream_energy_audit(entry, record, t_r, 16)
                    check()
                    audit32 = fast.upstream_energy_audit(entry, record, t_r, 32)
                finally:
                    audit_seconds += time.monotonic() - audit_clock
                upstream_audits.append((seg_t1, audit16, audit32))
                segment_diagnostics.append((seg_t1, {k: record[k] for k in
                                                     ("sweeps", "change", "rep_pad", "rep_source")}))
            current_action, down = None, None
            for split_index, (action_index, hold_index, start, end) in dwell_splits:
                check()
                clock = time.monotonic()
                held = plan.intervals[hold_index]
                works = dict.fromkeys(WORK_FIELDS, 0.0)
                audit_down = dict(input_work=0.0, heat=0.0)
                if action_index != current_action:
                    current_action = action_index
                    down = base._down_matrices(held.daq_selected_sources)
                    hold_charge = np.zeros(state_size)
                    for node, value in held.injected_charge_coulomb:
                        if node >= base.n_up:
                            hold_charge[node] = value
                    if np.any(hold_charge):
                        nodes = np.flatnonzero(hold_charge)
                        before = base.down_x[nodes - base.n_up].copy()
                        base._apply_charge(entry, hold_charge)
                        after = base.down_x[nodes - base.n_up]
                        charge_work = float(0.5 * ((before + after) @ hold_charge[nodes]))
                        contributions.append((start, True, dict(supplied_charge_boundary_work_joule=charge_work)))
                points = [start, *[b for b in seg_bounds if start < b < end], end]
                hold_mean = np.zeros(24)
                for u0, u1 in zip(points[:-1], points[1:]):
                    seg = max(i for i in range(len(seg_records)) if seg_bounds[i] <= u0)
                    record = seg_records[seg]
                    # fractions of the coarse segment, rounded so identical relative grids share weights
                    a = min(max(round((u0 - seg_bounds[seg]) / record["h"], 12), 0.0), 1.0)
                    b = min(max(round((u1 - seg_bounds[seg]) / record["h"], 12), 0.0), 1.0)
                    hom_poly, _ = fast.a_fine(entry, record, a, b)
                    res = fast.downstream_piece(down, u1 - u0, ("fine", record["key"], a, b), u0, hom_poly, row)
                    hold_mean += ((u1 - u0) / (end - start)) * res["hold_mean"]
                    for name, value in res["works"].items():
                        if name in works:
                            works[name] += value
                    audit_down["input_work"] += res["works"]["audit_down_input_joule"]
                    audit_down["heat"] += res["works"]["audit_down_heat_joule"]
                    records["clip_events"][split_index] += res["events"]
                contributions.append((end, False, works))
                downstream_audits.append((end, audit_down))
                records["t_start"][split_index], records["t_end"][split_index] = start, end
                records["hold_mean"][split_index] = hold_mean
                records["hold_end"][split_index] = base.down_x[holds_local]
                records["work"][split_index] = [works[f] for f in WORK_FIELDS]
                records["downstream_audit"][split_index] = (audit_down["input_work"], audit_down["heat"])
                records["elapsed"][split_index] = time.monotonic() - clock
                if dwell == 0 and end == 5e-6:
                    # 5-us state: units at the 5-us coarse cut, A and downstream from the fine pass
                    cut = next(i for i, bnd in enumerate(seg_bounds) if bnd == 5e-6)
                    saved = base.unit_y.copy()
                    for t, members, y0 in seg_records[cut]["unit_y0"]:
                        base.unit_y[members] = y0
                    checkpoint(end, 2)
                    base.unit_y[:] = saved
            mismatch = float(np.abs(base.a_y - a_end_coarse).max() / max(np.abs(a_end_coarse).max(), 1e-300))
            fast.stats["max_fine_coarse_mismatch"] = max(fast.stats["max_fine_coarse_mismatch"], mismatch)
            delta = entry["block"].phi @ (base.a_y - a_end_coarse)
            count, width = len(entry["pads"]), base.model.unit_size
            delta_pads = delta[:count * width].reshape(count, width).copy()
            delta_pads[:, 1:] += delta_pads[:, :1]
            mismatch_volts = max(float(np.abs(delta_pads).max()), float(np.abs(delta[count * width:]).max()))
            fast.stats["max_fine_coarse_mismatch_volts"] = max(
                fast.stats.get("max_fine_coarse_mismatch_volts", 0.0), mismatch_volts)
            checkpoint(t_r + fast.dwell, (dwell + 1) * 25)
            print(json.dumps(dict(event="progress", dwells=dwell + 1, elapsed=time.monotonic() - started,
                                  stats=fast.stats)), flush=True)
        timings["propagation_seconds"] = time.monotonic() - clock_prop
        timings["upstream_energy_audit_seconds"] = audit_seconds
        phase = "window_works"
        window_work, window_audit, window_diagnostics = {}, {}, {}
        for name, t_end in WINDOWS:
            if t_end > args.slots * 2.5e-6 + 1e-15:
                continue
            total = prefix_work(contributions, t_end)
            for f, v in fast.input_works(0.0, t_end).items():
                total[f] += v
            window_work[name] = total
            if not np.all(np.isfinite(list(total.values()))):
                raise ValueError("nonfinite complete per-window physical work")
            up = [(lo, hi) for t, lo, hi in upstream_audits if t <= t_end + 1e-15]
            down = [v for t, v in downstream_audits if t <= t_end + 1e-15]
            audit = {field: math.fsum(hi[field] for lo, hi in up) + math.fsum(v[field] for v in down)
                     for field in ("input_work", "heat")}
            lower = {field: math.fsum(lo[field] for lo, hi in up) + math.fsum(v[field] for v in down)
                     for field in ("input_work", "heat")}
            k = next(i for i, t in enumerate(ck["times"]) if abs(t - t_end) <= 1e-15)
            e0, e1 = ck["energy"][0][1], ck["energy"][k][1]
            supplied = total["supplied_charge_boundary_work_joule"]
            residual = e1 - e0 - supplied - audit["input_work"] + audit["heat"]
            scale = max(abs(e0), abs(e1), abs(supplied), abs(audit["input_work"]), abs(audit["heat"]))
            if not np.all(np.isfinite([*audit.values(), e0, e1, supplied, residual])):
                raise ValueError("nonfinite independently integrated per-window energy audit")
            window_audit[name] = dict(available=True, stored_start=e0, stored_end=e1, **audit,
                supplied_charge_work=supplied, residual=residual, relative_residual=residual / scale if scale else 0.0,
                method="independent Norton external forcing work and conductance quadratic heat; upstream particular+coarse homogeneous trajectory; actual downstream clipped trajectory",
                upstream_quadrature_orders=[16, 32], downstream_quadrature_order=16,
                quadrature_signed_difference={f: audit[f] - lower[f] for f in audit},
                quadrature_absolute_difference_sum={f: math.fsum(abs(hi[f] - lo[f]) for lo, hi in up) for f in audit},
                downstream_order_doubling={"available": False, "reason": "unchanged qualified downstream kernel uses16-point quadrature"},
                residual_acceptance={"available": False, "reason": "diagnostic only; no new residual tolerance is declared"})
            diagnostics = [v for t, v in segment_diagnostics if t <= t_end + 1e-15]
            mask = np.isfinite(records["t_end"]) & (records["t_end"] <= t_end + 1e-15)
            window_diagnostics[name] = dict(available=True, scope="completed coarse segments and downstream fine intervals in window",
                segments=len(diagnostics), sweeps=sum(v["sweeps"] for v in diagnostics),
                max_sweeps=max(v["sweeps"] for v in diagnostics), max_wr_change=max(v["change"] for v in diagnostics),
                max_rep_pad=max(v["rep_pad"] for v in diagnostics), max_rep_source=max(v["rep_source"] for v in diagnostics),
                clip_events=int(records["clip_events"][mask].sum()))
        report["window_work"] = window_work
        report["window_energy_audit"] = window_audit
        report["window_solver_diagnostics"] = window_diagnostics
        report["work_accounting"]["window_work_complete"] = True
        phase = "complete"
        report.update(status="complete")
    except BaseException as error:
        import traceback
        report["failure"] = dict(phase=phase, exception_type=type(error).__name__, message=str(error)[:2000])
        report["traceback"] = traceback.format_exc()[-4000:]
    finally:
        try:
            import numpy as np
            if records is not None:
                complete = np.isfinite(records["t_end"])
                report["completed_intervals"] = int(complete.sum())
                report["physical_frontier_seconds"] = float(records["t_end"][complete].max()) if complete.any() else 0.0
                window_work = report.get("window_work", {})
                np.savez(run_dir / "solver_intervals.npz", **records, work_fields=np.array(WORK_FIELDS),
                         window_names=np.array(list(window_work), dtype=str),
                         window_work=np.array([[window_work[w][f] for f in WORK_FIELDS] for w in window_work])
                         if window_work else np.empty((0, len(WORK_FIELDS))))
            if ck["states"]:
                np.savez(run_dir / "solver_checkpoints.npz", times=np.array(ck["times"]),
                         slots=np.array(ck["slots"]), states=np.vstack(ck["states"]), energy=np.array(ck["energy"]))
            if fast is not None:
                report["fast_stats"] = {k: float(v) for k, v in fast.stats.items()}
                report["particular_max_residual"] = fast.part.max_residual
                report["base_stats"] = {k: float(v) for k, v in fast.base.stats.items()}
        except BaseException as error:
            report["save_failure"] = f"{type(error).__name__}: {str(error)[:500]}"
        timings["upstream_energy_audit_seconds"] = audit_seconds
        report["timings"] = timings
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _write_json(run_dir / "solver_report.json", report)
        print(json.dumps(dict(event="terminal", status=report["status"], phase=phase,
                              elapsed=report["elapsed_seconds"])), flush=True)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
