"""Continuous B2/2001 prefixes for adopted Route C Stage 2, then the 16-s record.

No submissions or automatic successors. Start at the original initial condition
or a committed frame boundary, carry all states, and stream O2 output.
First-frame evidence uses the existing fastpath comparator's archive interface.
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
import traceback


EPISODE_ROOT = Path(__file__).resolve().parents[2]
FIRST_WINDOWS = (("5us", 5e-6, 2), ("dwell", 62.5e-6, 25), ("frame", 4e-3, 1600))
ACCOUNTING_COLUMNS = ("propagation_seconds", "upstream_energy_audit_seconds", "readout_write_seconds", "sweeps", "max_sweeps",
                      "max_wr_change", "max_rep_pad", "max_rep_source", "clip_events",
                      "energy_residual_joule", "energy_relative_residual", "peak_rss_kib",
                      "cache_retained_numeric_bytes")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False, default=float)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def make_clock(deadline_seconds, memory_mib):
    from fast_io_case import PhaseClock

    class BoundedClock(PhaseClock):
        interrupted = None

        def check(self):
            super().check()
            if self.interrupted is not None:
                raise TimeoutError(f"signal {self.interrupted} received")
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > memory_mib * 1024:
                raise MemoryError("explicit peak RSS limit reached")

    return BoundedClock(deadline_seconds)


def compute_context():
    """One numerical worker; record and charge the actual allocated CPU count."""
    from fast_io_case import THREAD_VARIABLES
    host, job = socket.gethostname(), os.environ.get("SLURM_JOB_ID", "")
    allocated = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if "-ln" in host or not host.startswith("sh") or not job.isdecimal() or allocated < 1:
        raise RuntimeError("an actual Sherlock compute shell and explicit CPU allocation are required")
    for variable in THREAD_VARIABLES:
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"{variable} must be 1 before numerical imports")
    return dict(hostname=host, job_id=job, allocated_cpus=allocated, numerical_workers=1,
                memory_mib=os.environ.get("SLURM_MEM_PER_NODE"))


def window_summary(fast, contributions, upstream, downstream, diagnostics, clips,
                   start, end, energy_start, energy_end):
    from fast_solver_stage1 import WORK_FIELDS
    from fast_solver_fastpath import prefix_work

    work = prefix_work(contributions, end)
    for field, value in fast.input_works(start, end).items():
        work[field] += value
    if not all(math.isfinite(work[field]) for field in WORK_FIELDS):
        raise ValueError("nonfinite complete physical work")
    up = [(lo, hi) for t, lo, hi in upstream if t <= end]
    down = [value for t, value in downstream if t <= end]
    fields = ("input_work", "heat")
    audit = {field: math.fsum(hi[field] for lo, hi in up) + math.fsum(v[field] for v in down)
             for field in fields}
    lower = {field: math.fsum(lo[field] for lo, hi in up) + math.fsum(v[field] for v in down)
             for field in fields}
    boundary = work["supplied_charge_boundary_work_joule"]
    # Direct contact capacitance is rejected by spec_from_cpe_array. Fixed C
    # makes .5*(before+after).q the physical energy increment of each impulse.
    work["physical_boundary_energy_change_joule"] = boundary
    work["physical_interval_energy_change_joule"] = energy_end - energy_start - boundary
    residual = energy_end - energy_start - boundary - audit["input_work"] + audit["heat"]
    scale = max(abs(energy_start), abs(energy_end), abs(boundary), *(abs(v) for v in audit.values()))
    energy = dict(available=True, stored_start=energy_start, stored_end=energy_end, **audit,
        supplied_charge_work=boundary, residual=residual, relative_residual=residual / scale if scale else 0.,
        method="independent Norton forcing work and conductance heat on coarse upstream and actual clipped downstream",
        upstream_quadrature_orders=[16, 32], downstream_quadrature_order=16,
        quadrature_signed_difference={f: audit[f] - lower[f] for f in fields},
        quadrature_absolute_difference_sum={f: math.fsum(abs(hi[f] - lo[f]) for lo, hi in up) for f in fields},
        residual_acceptance=dict(available=False, reason="diagnostic; no new tolerance declared"))
    selected = [v for t, v in diagnostics if t <= end]
    solver = dict(available=True, scope="coarse segments and downstream pieces in this window",
        segments=len(selected), sweeps=sum(v["sweeps"] for v in selected),
        max_sweeps=max(v["sweeps"] for v in selected), max_wr_change=max(v["change"] for v in selected),
        max_rep_pad=max(v["rep_pad"] for v in selected), max_rep_source=max(v["rep_source"] for v in selected),
        clip_events=sum(value for t, value in clips if t <= end))
    if not all(math.isfinite(v) for v in (*work.values(), energy_start, energy_end, residual,
                                          *audit.values(), *lower.values())):
        raise ValueError("nonfinite window energy accounting")
    return work, energy, solver


def run_frames(case, plan, fast, frames, run_dir, clock, report, *, record_dir=None,
               resume_checkpoint=None, stop_after_frame=None):
    import numpy as np
    from fast_io_readout import VectorizedReadout, WORK_FIELDS as ALL_WORK_FIELDS
    from fast_solver_stage1 import WORK_FIELDS
    from fast_solver_continuous_kernel import coarse_bounds, dwell_spans
    from fast_solver_frame_checkpoint import (FrameRecordWriter, capture_frame, commit_frame,
                                              load_frame, plan_bindings, restore_frame)

    base, spec, array, template = fast.base, fast.base.spec, case.array, plan.template
    state_size, pads = len(array.state_ids), spec.pads
    hold_local = np.asarray(spec.holds) - base.n_up
    readout = VectorizedReadout(template)
    record_dir = run_dir if record_dir is None else Path(record_dir)
    stop = frames if stop_after_frame is None else stop_after_frame
    first_dir = record_dir / "first_frame"
    checkpoint_path = record_dir / "frame_checkpoint.npz"
    saved = None if resume_checkpoint is None else load_frame(resume_checkpoint, state_count=state_size)
    start_frame = 0 if saved is None else saved.next_frame
    if not 0 <= start_frame < stop <= frames <= 4000:
        raise ValueError("require a nonempty chronological chunk within the record target")
    if saved is None:
        first_dir.mkdir()
        cumulative = dict.fromkeys(ALL_WORK_FIELDS, 0.)
        initial = np.asarray(array.volts)
        fast.set_full_state(initial, 0., int(template.action_row[0]))
        initial_energy = float(array._physical_energy(initial, np.zeros(pads)))
        checkpoints = [(0., 0, initial.copy(), initial_energy)]
        frame_start_energy, final_state, final_time = initial_energy, None, 0.
        record_diagnostics = dict(max_abs_energy_relative_residual=0., record_clip_events=0,
                                  record_sweeps=0, record_max_sweeps=0)
    else:
        restore_frame(saved, fast, plan)
        cumulative = dict(saved.cumulative_energy)
        checkpoints = []
        frame_start_energy, final_state, final_time = saved.physical_energy, saved.physical_state.copy(), saved.time
        record_diagnostics = dict(saved.diagnostics)
    report.update(record_diagnostics)
    report.update(frames_completed=start_frame, frames_this_run=0, start_frame=start_frame,
                  stop_after_frame=stop, record_target_frames=frames, physical_frontier_seconds=final_time,
                  record_directory=str(record_dir), checkpoint_path=str(checkpoint_path),
                  record_energy_totals=dict(cumulative), record_complete=False)
    per_frame_seconds, first_report = [], None
    components = dict(propagation_seconds=0., upstream_energy_audit_seconds=0., readout_write_seconds=0.,
                      first_frame_evidence_write_seconds=0., frame_checkpoint_write_seconds=0.)
    report["first_frame_evidence"] = str(first_dir)
    report["component_timings_seconds"] = components
    writer_args = dict(template=template, plan_digests=plan_bindings(plan), windows=frames,
                       accounting_columns=ACCOUNTING_COLUMNS)
    if saved is None:
        writer = FrameRecordWriter(record_dir / "record.h5", **writer_args, retain_aperture_means=True)
    else:
        writer = FrameRecordWriter.resume(record_dir / "record.h5", **writer_args, checkpoint=saved,
                                          rolling_checkpoint_path=checkpoint_path)
    terminal_status = "failed"
    try:
        for frame in range(start_frame, stop):
            frame_wall_start = time.monotonic()
            clock.check()
            window = plan.window(frame)
            if window.start != final_time:
                raise ValueError("continuous frame seam is not the preceding physical endpoint")
            count = len(window.split_hold)
            means = np.empty((count, 24))
            hold_ends = np.empty((count, 24)) if frame == 0 else None
            contributions, upstream, downstream, diagnostics, clips = [], [], [], [], []
            short_energy = {}
            audit_seconds = 0.
            propagation_start = time.monotonic()
            for first, last, lo, hi, row in dwell_spans(window, template):
                clock.check()
                t0, t1 = float(window.action_start[first]), float(window.action_end[last - 1])
                source_charge = np.zeros(state_size)
                for node, value in template.action_charge_objects[first]:
                    if pads <= node < pads + 64:
                        source_charge[node] = value
                entry, source_work = fast.begin_dwell(row, t0, source_charge)
                contributions.append((t0, True, dict(supplied_charge_boundary_work_joule=source_work)))
                bounds = coarse_bounds(t0, t1, fast.fractions, (5e-6,) if frame == 0 and first == 0 else ())
                records = []
                for a, b in zip(bounds[:-1], bounds[1:]):
                    clock.check()
                    record = fast.coarse_piece(entry, float(a), float(b))
                    records.append(record)
                    part = fast.particular_works(row, float(a), float(b))
                    hom = fast.homogeneous_works(entry, [record], float(a))
                    contributions.append((float(b), False, {key: part[key] + hom[key] for key in part}))
                    audit_start = time.monotonic()
                    try:
                        audit16 = fast.upstream_energy_audit(entry, record, float(a), 16)
                        clock.check()
                        audit32 = fast.upstream_energy_audit(entry, record, float(a), 32)
                    finally:
                        elapsed_audit = time.monotonic() - audit_start
                        audit_seconds += elapsed_audit
                        components["upstream_energy_audit_seconds"] += elapsed_audit
                    upstream.append((float(b), audit16, audit32))
                    diagnostics.append((float(b), {key: record[key] for key in
                                                   ("sweeps", "change", "rep_pad", "rep_source")}))
                coarse_endpoint = base.a_y.copy()
                base.a_y = records[0]["ya0"].copy()
                current_action, down, segment = None, None, 0
                for index in range(lo, hi):
                    clock.check()
                    action = int(window.hold_action[window.split_hold[index]])
                    start, end = map(float, window.split_bounds[index:index + 2])
                    if action != current_action:
                        if not window.split_charge[index]:
                            raise ValueError("first action split must own its boundary charge")
                        current_action = action
                        down = fast.down_matrices(template.action_selection_objects[action])
                        charge = np.zeros(state_size)
                        for node, value in template.action_charge_objects[action]:
                            if node >= base.n_up:
                                charge[node] = value
                        if np.any(charge):
                            nodes = np.flatnonzero(charge)
                            before = base.down_x[nodes - base.n_up].copy()
                            base._apply_charge(entry, charge)
                            after = base.down_x[nodes - base.n_up]
                            impulse = float(.5 * ((before + after) @ charge[nodes]))
                            contributions.append((start, True, dict(supplied_charge_boundary_work_joule=impulse)))
                    points = [start, *bounds[(bounds > start) & (bounds < end)], end]
                    mean = np.zeros(24)
                    works, audit_down, events = dict.fromkeys(WORK_FIELDS, 0.), dict(input_work=0., heat=0.), 0
                    for u0, u1 in zip(points[:-1], points[1:]):
                        while segment + 1 < len(records) and u0 >= records[segment]["t1"]:
                            segment += 1
                        homogeneous, h = fast.fine_piece(entry, records[segment], float(u0), float(u1))
                        result = fast.downstream_exact_piece(down, h, float(u0), homogeneous, row)
                        mean += (h / (end - start)) * result["hold_mean"]
                        for key in works:
                            works[key] += result["works"].get(key, 0.)
                        audit_down["input_work"] += result["works"]["audit_down_input_joule"]
                        audit_down["heat"] += result["works"]["audit_down_heat_joule"]
                        events += result["events"]
                    means[index] = mean
                    if hold_ends is not None:
                        hold_ends[index] = base.down_x[hold_local]
                    contributions.append((end, False, works))
                    downstream.append((end, audit_down))
                    clips.append((end, events))
                    if frame == 0 and first == 0 and end == 5e-6:
                        cut = next(i for i, value in enumerate(bounds) if value == end)
                        saved = base.unit_y.copy()
                        try:
                            for kind, members, values in records[cut]["unit_y0"]:
                                base.unit_y[members] = values
                            state = fast.full_state(end)
                            energy = float(array._physical_energy(state, np.zeros(pads)))
                            checkpoints.append((end, 2, state, energy))
                            short_energy["5us"] = energy
                        finally:
                            base.unit_y[:] = saved
                modal_difference = base.a_y - coarse_endpoint
                relative_mismatch = float(np.abs(modal_difference).max()
                                          / max(float(np.abs(coarse_endpoint).max()), 1e-300))
                fast.stats["max_fine_coarse_mismatch"] = max(
                    fast.stats["max_fine_coarse_mismatch"], relative_mismatch)
                delta = entry["block"].phi @ modal_difference
                width, active_count = base.model.unit_size, len(entry["pads"])
                delta_pads = delta[:active_count * width].reshape(active_count, width).copy()
                delta_pads[:, 1:] += delta_pads[:, :1]
                mismatch = max(float(np.abs(delta_pads).max()), float(np.abs(delta[active_count * width:]).max()))
                fast.stats["max_fine_coarse_mismatch_volts"] = max(
                    fast.stats.get("max_fine_coarse_mismatch_volts", 0.), mismatch)
                fast.stats["dwells"] += 1
                if frame == 0 and first == 0:
                    state = fast.full_state(t1)
                    energy = float(array._physical_energy(state, np.zeros(pads)))
                    checkpoints.append((t1, 25, state, energy))
                    short_energy["dwell"] = energy
                report["physical_frontier_seconds"] = t1
            frame_state = fast.full_state(window.end)
            energy_end = float(array._physical_energy(frame_state, np.zeros(pads)))
            work, audit, diagnostic = window_summary(fast, contributions, upstream, downstream, diagnostics, clips,
                                                     window.start, window.end, frame_start_energy, energy_end)
            for key in cumulative:
                cumulative[key] = math.fsum((cumulative[key], work[key]))
            propagation_seconds = time.monotonic() - propagation_start - audit_seconds
            components["propagation_seconds"] += propagation_seconds
            io_start = time.monotonic()
            observed = readout.read_window(window, means)
            accounting = dict(propagation_seconds=propagation_seconds, upstream_energy_audit_seconds=audit_seconds,
                readout_write_seconds=0.,
                **{key: diagnostic[key] for key in ("sweeps", "max_sweeps", "max_wr_change", "max_rep_pad",
                                                    "max_rep_source", "clip_events")},
                energy_residual_joule=audit["residual"], energy_relative_residual=audit["relative_residual"],
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cache_retained_numeric_bytes=fast.weight_cache.bytes)
            writer.write_window(observed, energy_window=work, energy_cumulative=cumulative, accounting=accounting)
            io_seconds = time.monotonic() - io_start
            components["readout_write_seconds"] += io_seconds
            # The measured interval includes readout and all preceding HDF5 row
            # writes; update its own accounting cell after that interval closes.
            writer._file["accounting/window"][frame, ACCOUNTING_COLUMNS.index("readout_write_seconds")] = io_seconds
            if frame == 0:
                evidence_start = time.monotonic()
                checkpoints.append((window.end, 1600, frame_state.copy(), energy_end))
                short_energy["frame"] = energy_end
                first_report = dict(status="complete", window_work={}, window_energy_audit={}, window_solver_diagnostics={})
                for name, end, slots in FIRST_WINDOWS:
                    short_work, short_audit, short_diag = window_summary(fast, contributions, upstream, downstream,
                        diagnostics, clips, 0., end, initial_energy, short_energy[name])
                    first_report["window_work"][name] = {f: short_work[f] for f in WORK_FIELDS}
                    first_report["window_energy_audit"][name] = short_audit
                    first_report["window_solver_diagnostics"][name] = short_diag
                np.savez(first_dir / "solver_intervals.npz", t_start=window.split_bounds[:-1],
                    t_end=window.split_bounds[1:], hold_mean=means, hold_end=hold_ends,
                    work_fields=np.array(WORK_FIELDS), window_names=np.array([v[0] for v in FIRST_WINDOWS]),
                    window_work=np.array([[first_report["window_work"][name][f] for f in WORK_FIELDS]
                                          for name, _, _ in FIRST_WINDOWS]))
                np.savez(first_dir / "solver_checkpoints.npz", times=np.array([v[0] for v in checkpoints]),
                    slots=np.array([v[1] for v in checkpoints]), states=np.vstack([v[2] for v in checkpoints]),
                    energy=np.array([[v[3], v[3]] for v in checkpoints]))
                write_json(first_dir / "solver_report.json", first_report)
                checkpoints.clear()
                evidence_seconds = time.monotonic() - evidence_start
                components["first_frame_evidence_write_seconds"] += evidence_seconds
                components["readout_write_seconds"] += evidence_seconds
            next_diagnostics = dict(
                max_abs_energy_relative_residual=max(record_diagnostics["max_abs_energy_relative_residual"],
                                                    abs(audit["relative_residual"])),
                record_clip_events=record_diagnostics["record_clip_events"] + diagnostic["clip_events"],
                record_sweeps=record_diagnostics["record_sweeps"] + diagnostic["sweeps"],
                record_max_sweeps=max(record_diagnostics["record_max_sweeps"], diagnostic["max_sweeps"]),
                last_window_energy_audit=audit)
            checkpoint_start = time.monotonic()
            checkpoint = capture_frame(fast, plan, next_frame=frame + 1, time=window.end,
                physical_state=frame_state, physical_energy=energy_end, cumulative_energy=cumulative,
                committed_output_rows=writer._next, diagnostics=next_diagnostics)
            commit_frame(checkpoint_path, checkpoint, writer)
            components["frame_checkpoint_write_seconds"] += time.monotonic() - checkpoint_start
            # These snapshots and counters describe committed output only. A
            # later partial frame may advance the separate physical frontier.
            final_state, final_time = frame_state, window.end
            record_diagnostics = next_diagnostics
            report.update(record_diagnostics)
            report.update(frames_completed=frame + 1, frames_this_run=frame + 1 - start_frame,
                          record_energy_totals=dict(cumulative))
            frame_start_energy = energy_end
            clock.check()
            per_frame_seconds.append(time.monotonic() - frame_wall_start)
            print(json.dumps(dict(event="frame", frame=frame + 1, physical_time=final_time,
                seconds=per_frame_seconds[-1], elapsed=clock.elapsed, cache=fast.weight_cache.report())), flush=True)
            per_frame_seconds[-1] = time.monotonic() - frame_wall_start
        report["per_frame_seconds"] = per_frame_seconds
        report["mean_frame_seconds"] = math.fsum(per_frame_seconds) / len(per_frame_seconds)
        report["projected_16_second_record_seconds"] = report["mean_frame_seconds"] * 4000
        report["first_frame_comparison_status"] = ("evidence_written_comparison_not_run" if start_frame == 0
                                                    else "existing_evidence_preserved_comparison_not_run")
        terminal_status = "complete" if stop == frames else "checkpointed"
        return dict(status=terminal_status, record_complete=stop == frames)
    finally:
        report["weight_cache"] = fast.weight_cache.report()
        report["per_frame_seconds"] = per_frame_seconds
        report["output_rows_written"] = writer._next
        try:
            if final_state is not None:
                report["final_state_snapshot_seconds"] = final_time
                report["final_state_snapshot_scope"] = "last committed frame endpoint; may precede failure frontier"
                np.savez(run_dir / "final_state.npz", time=np.array(final_time), state=final_state)
        finally:
            terminal = terminal_status if sys.exc_info()[0] is None else "failed"
            try:
                writer.finish(status=terminal, metadata=dict(frames_completed=report.get("frames_completed", 0),
                              committed_output_rows=report.get("frames_completed", 0),
                              output_rows_written=writer._next,
                              frames_this_run=report["frames_this_run"], checkpoint_path=str(checkpoint_path),
                              physical_frontier_seconds=report["physical_frontier_seconds"],
                              final_state_snapshot_seconds=report.get("final_state_snapshot_seconds")))
            finally:
                writer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--record-dir", type=Path, help="persistent output/checkpoint directory; defaults to run-dir")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--frames", type=int, choices=(1, 10, 100, 1000, 4000), required=True)
    parser.add_argument("--stop-after-frame", type=int, help="global completed-frame boundary for this finite chunk")
    parser.add_argument("--deadline-seconds", type=float, required=True)
    parser.add_argument("--memory-limit-mib", type=float, required=True)
    parser.add_argument("--cache-mib", type=float, default=1024.)
    parser.add_argument("--segment-levels", type=int, default=12)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--wr-tolerance", type=float, default=1e-10)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(EPISODE_ROOT / "outputs/runs") or not run_dir.is_dir() or any(run_dir.iterdir()):
        parser.error("supply a fresh empty existing directory under this episode's outputs/runs")
    record_dir = run_dir if args.record_dir is None else args.record_dir.resolve()
    if not record_dir.is_relative_to(EPISODE_ROOT / "outputs/runs") or not record_dir.is_dir():
        parser.error("record-dir must be an existing directory under this episode's outputs/runs")
    resume_checkpoint = None if args.resume_checkpoint is None else args.resume_checkpoint.resolve()
    if resume_checkpoint is None:
        if any(record_dir.iterdir()):
            parser.error("a new record requires a fresh empty record-dir")
    elif (not resume_checkpoint.is_relative_to(EPISODE_ROOT / "outputs/runs") or not resume_checkpoint.is_file()
          or record_dir == run_dir or not (record_dir / "record.h5").is_file()):
        parser.error("resume requires an existing checkpoint and record, and a separate fresh run-dir")
    stop = args.frames if args.stop_after_frame is None else args.stop_after_frame
    if not 1 <= stop <= args.frames:
        parser.error("stop-after-frame must lie within the requested record target")
    if (not all(math.isfinite(v) and v > 0 for v in
                (args.deadline_seconds, args.memory_limit_mib, args.cache_mib, args.wr_tolerance))
            or args.cache_mib >= args.memory_limit_mib or not 1 <= args.segment_levels <= 30
            or not 1 <= args.degree <= 10):
        parser.error("finite positive bounds, cache below memory limit, levels 1..30, degree 1..10 required")
    from fast_io_case import build_case, build_plan
    clock = make_clock(args.deadline_seconds, args.memory_limit_mib)
    report = dict(status="failed", frames_requested=args.frames, frames_completed=0, frames_this_run=0,
                  physical_frontier_seconds=0., scope="conditional_B2_2001_continuous_speed_measurement",
                  deadline_seconds=args.deadline_seconds, memory_limit_mib=args.memory_limit_mib,
                  scientific_scoring=False, inputs="original_saved_B2_2001_and_attempt002_noise",
                  per_interval_work_rows_retained=False, energy_fields_retained=10,
                  resumed_from=None if resume_checkpoint is None else str(resume_checkpoint),
                  record_directory=str(record_dir), record_complete=False,
                  solver_statistics_scope="current process; cumulative frame diagnostics are reported separately")
    previous = {}
    fast = None
    try:
        report["allocation"] = compute_context()
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, lambda signum, frame: setattr(clock, "interrupted", signum))
        settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        write_json(run_dir / "settings.json", settings | {"run_dir": str(run_dir),
                   "record_dir": str(record_dir), "stop_after_frame": stop, "mode": "continuous_exact"})
        case = build_case(clock)
        plan, totals = build_plan(case, clock)
        report["plan_totals"] = totals
        from fast_solver_engine import FastArraySolver, spec_from_cpe_array
        from fast_solver_production import ExactInputs, ProductionWorkModel
        from fast_solver_multirate import ToneSet
        from fast_solver_continuous_kernel import ContinuousMultirateSolver
        clock.set("continuous_solver_setup")
        spec = spec_from_cpe_array(case.array)
        work_model = ProductionWorkModel(case.array)
        base = FastArraySolver(spec, degree=args.degree, wr_tolerance=args.wr_tolerance,
                               work_model=work_model, energy_audit=True)
        exact = ExactInputs(case.array, case.noise, case.raw_time, case.contacts)
        fast = ContinuousMultirateSolver(base, ToneSet(exact, case.array), work_model,
            segment_levels=args.segment_levels, cache_bytes=int(args.cache_mib * 1024**2))
        report["precomputation_seconds"] = clock.elapsed
        clock.set("continuous_propagation_readout")
        report.update(run_frames(case, plan, fast, args.frames, run_dir, clock, report,
                                 record_dir=record_dir, resume_checkpoint=resume_checkpoint,
                                 stop_after_frame=stop))
    except BaseException as error:
        report["failure"] = dict(phase=clock.phase, exception_type=type(error).__name__,
                                 message=str(error)[:2000], traceback=traceback.format_exc()[-6000:])
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if fast is not None:
            report["fast_stats"] = {k: float(v) for k, v in fast.stats.items()}
            report["base_stats"] = {k: float(v) for k, v in fast.base.stats.items()}
            report["particular_max_residual"] = fast.part.max_residual
        report["elapsed_seconds"] = clock.elapsed
        if report["status"] in ("complete", "checkpointed"):
            frame_total = math.fsum(report["per_frame_seconds"])
            report["non_frame_seconds_including_setup_and_final_write"] = report["elapsed_seconds"] - frame_total
            report["projected_16_second_record_seconds"] = (
                report["non_frame_seconds_including_setup_and_final_write"] + report["mean_frame_seconds"] * 4000)
        report["allocated_cpu_hours"] = report["elapsed_seconds"] * report.get("allocation", {}).get("allocated_cpus", 0) / 3600.
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["timings_seconds"] = dict(clock.timings)
        report["timings_seconds"][clock.phase] = report["timings_seconds"].get(clock.phase, 0.) + time.monotonic() - clock._phase_started
        write_json(run_dir / "solver_report.json", report)
        print(json.dumps(dict(event="terminal", status=report["status"], frames=report["frames_completed"],
                              elapsed=report["elapsed_seconds"])), flush=True)
    return 0 if report["status"] in ("complete", "checkpointed") else 2


if __name__ == "__main__":
    sys.exit(main())
