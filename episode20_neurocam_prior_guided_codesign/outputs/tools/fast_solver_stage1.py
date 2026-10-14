"""Run the fast solver (O4+O7, optional O6 inputs) on a prefix of cycle 0.

Same world, circuit, schedule, noise draw, holds and split intervals as
`fast_solver_reference.py`; only the solver differs. Modes:
* ref_fed: the reference's midpoint field holds and per-slot noise cubics;
* exact:   O6 analytic field and noise tones of the same draw and ports.
Development test only (Stage 1). Outputs (exclusive, in --run-dir):
settings.json, solver_intervals.npz, solver_checkpoints.npz, solver_report.json.
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

EPISODE_ROOT = Path(__file__).resolve().parents[2]
WORLD_PATH = EPISODE_ROOT / "outputs/runs/development_worlds_job_47085635/panel/development_worlds.h5"
NOISE_PATH = (EPISODE_ROOT / "outputs/runs/model_specification/conditional_forward_cost_attempt002"
              / "conditional_noise_draw.npz")
WORK_FIELDS = (
    "field_source_boundary_work_joule", "supplied_charge_boundary_work_joule",
    "field_source_interval_work_joule", "daq_external_source_interval_work_joule",
    "controlled_voltage_source_interval_work_joule", "conditioning_drive_branch_dissipation_joule",
    "noise_current_source_interval_work_joule", "noise_access_voltage_source_interval_work_joule")


def _write_json(path, value):
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=float) + "\n"
    with open(path, "x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def build_reference_setting(check, offsets=None):
    """World, unmodified reference array, schedule, noise and holds (as the reference driver)."""
    import numpy as np
    from analog_conditioning import AnalogConditioning, SourceConditioning
    import cpe_array as array_module
    from conditional_acquisition_recipe import (build_conditional_frontend, build_conditional_schedule,
                                                conditional_noise_cell)
    from conditional_forward_recipe import (CONDITIONING, REFERENCE_ID, build_conditional_circuit_parameters,
                                            build_conditional_noise, expected_state_ids)
    from development_simulation import load_saved_development_world
    from raw_world_time import RawWorldTime
    from surface_to_contact import build_contact_factors, registered_pad_geometry
    world = load_saved_development_world(WORLD_PATH, condition_id="B2", replicate_seed=2001,
        max_elements=2000000, max_array_bytes=64 * 1024**2, max_metadata_bytes=1024**2, check=check)
    geometry = registered_pad_geometry("G0", short_axis="x", max_elements=2000000, check=check)
    contacts = build_contact_factors(world, geometry, max_elements=2000000, check=check)
    raw_time = RawWorldTime(world, max_elements=2000000, check=check)
    frontend = build_conditional_frontend()
    parameters = build_conditional_circuit_parameters(daq_frontend=frontend, check=check)
    if offsets is not None:
        parameters["analog_conditioning"] = AnalogConditioning(
            channels=tuple(SourceConditioning(**dict(CONDITIONING, offset_volts=float(v))) for v in offsets),
            source_load_role="additional_fixed_shunt_excludes_conditioning",
            source_capacitance_role="additional_fixed_capacitance_excludes_conditioning",
            voltage_reference_id=REFERENCE_ID)
    array = array_module.NeurocamCPEArray(**parameters)
    if array.state_ids != expected_state_ids():
        raise ValueError("state basis differs from the selected recipe")
    schedule = build_conditional_schedule(frontend, state_ids=array.state_ids, check=check)
    noise = build_conditional_noise(array=array, check=check, retained_draw_path=None,
        current_hold_tolerance_amp=1e-12, voltage_hold_tolerance_volts=1e-8,
        max_evaluations=200000000, max_term_evaluations=200000000000000,
        reuse_draw_path=NOISE_PATH, cubic_cell_for_key=conditional_noise_cell)
    return dict(world=world, contacts=contacts, raw_time=raw_time, frontend=frontend, array=array,
                schedule=schedule, noise=noise)


def build_splits(setting, slots, check):
    import numpy as np
    from raw_input_hold import compile_midpoint_hold
    schedule, noise = setting["schedule"], setting["noise"]
    iterator = schedule.physical_actions_for_cycle(0)
    actions = tuple(next(iterator) for _ in range(slots))
    readout = schedule.readout_window(0)
    plan = compile_midpoint_hold(setting["raw_time"], setting["contacts"], actions, epsilon_input_volts=1e-8,
        max_intervals=40000000, max_work=100000000, check=check, noise_hold_bound=noise.hold_bound)
    boundaries = sorted({point for conversion in readout.conversions
                         for point in (conversion.aperture_start, conversion.aperture_end)})
    boundary_array = np.asarray(boundaries)
    splits, position = [], 0
    for hold_index, held in enumerate(plan.intervals):
        while not actions[position].start <= held.start < actions[position].end:
            position += 1
        lo = np.searchsorted(boundary_array, held.start, side="right")
        hi = np.searchsorted(boundary_array, held.end, side="left")
        points = [held.start, *boundaries[lo:hi], held.end]
        splits.extend((position, hold_index, a, b) for a, b in zip(points, points[1:]))
    return actions, readout, plan, splits


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--mode", choices=("ref_fed", "exact"), default="ref_fed")
    parser.add_argument("--deadline-seconds", type=float, required=True)
    parser.add_argument("--offsets-json", type=Path, default=None)
    parser.add_argument("--checkpoint-slots", type=str, default="")
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--wr-tolerance", type=float, default=1e-10)
    parser.add_argument("--wr-max-sweeps", type=int, default=12)
    parser.add_argument("--grade-levels", type=int, default=8)
    parser.add_argument("--grade", choices=("dwell", "action", "hold", "all", "none"), default="dwell",
                        help="which pieces are geometrically graded toward their start")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if (not run_dir.is_dir() or not run_dir.is_relative_to(EPISODE_ROOT / "outputs/runs/fast_forward_stage1/solver")
            or any((run_dir / n).exists() for n in ("settings.json", "solver_report.json", "solver_intervals.npz"))):
        raise SystemExit("supply a fresh existing directory under runs/fast_forward_stage1/solver")
    host, job = socket.gethostname(), os.environ.get("SLURM_JOB_ID", "")
    if "-ln" in host or not job:
        raise SystemExit("compute allocation required; refusing to run on a login node")
    deadline = float(args.deadline_seconds)
    checkpoints = sorted({int(v) for v in args.checkpoint_slots.split(",") if v.strip()} | {args.slots})
    stop = {"signal": None}

    def on_signal(signum, frame):
        stop["signal"] = signum

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def check():
        if stop["signal"] is not None:
            raise TimeoutError(f"signal {stop['signal']} received")
        if time.monotonic() - started >= deadline:
            raise TimeoutError("internal deadline reached")

    report = dict(host=host, slurm_job_id=job, slots_requested=args.slots, mode=args.mode,
                  deadline_seconds=deadline, status="failed", degree=args.degree, wr_tolerance=args.wr_tolerance,
                  wr_max_sweeps=args.wr_max_sweeps, grade_levels=args.grade_levels, grade=args.grade,
                  offsets_fixture=None if args.offsets_json is None else str(args.offsets_json))
    phase = "imports"
    records = None
    ck_times, ck_states, ck_slots, ck_energy = [], [], [], []
    solver = None
    timings = {}
    try:
        import numpy as np
        from fast_solver_engine import FastArraySolver, spec_from_cpe_array
        from fast_solver_production import ExactInputs, ProductionWorkModel, ReferenceFedInputs
        offsets = None if args.offsets_json is None else json.loads(args.offsets_json.read_text())
        _write_json(run_dir / "settings.json", dict(report, checkpoint_slots=checkpoints, offsets_volts=offsets,
                    world_path=str(WORLD_PATH), noise_path=str(NOISE_PATH), reference_modules_modified=False))
        phase = "setting"
        clock = time.monotonic()
        setting = build_reference_setting(check, offsets)
        actions, readout, plan, splits = build_splits(setting, args.slots, check)
        array = setting["array"]
        timings["setting_seconds"] = time.monotonic() - clock
        phase = "solver_setup"
        clock = time.monotonic()
        spec = spec_from_cpe_array(array)
        work_model = ProductionWorkModel(array)
        solver = FastArraySolver(spec, degree=args.degree, wr_tolerance=args.wr_tolerance,
                                 wr_max_sweeps=args.wr_max_sweeps, grade_levels=args.grade_levels,
                                 work_model=work_model, energy_audit=True)
        solver.set_state(np.asarray(array.volts))
        if args.mode == "ref_fed":
            adapter = ReferenceFedInputs(array)
        else:
            adapter = ExactInputs(array, setting["noise"], setting["raw_time"], setting["contacts"])
        timings["solver_setup_seconds"] = time.monotonic() - clock
        capacitance = spec.capacitance
        n = len(splits)
        records = dict(
            t_start=np.full(n, np.nan), t_end=np.full(n, np.nan), action_index=np.full(n, -1, np.int64),
            hold_index=np.full(n, -1, np.int64), gate_row=np.full(n, -1, np.int64),
            hold_mean=np.full((n, 24), np.nan), hold_end=np.full((n, 24), np.nan),
            source_mean=np.full((n, 64), np.nan), source_end=np.full((n, 64), np.nan),
            buffer_end=np.full((n, 64), np.nan), filter_end=np.full((n, 64), np.nan),
            work=np.full((n, len(WORK_FIELDS)), np.nan), audit=np.full((n, 2), np.nan),
            sweeps=np.zeros(n, np.int64), wr_change=np.full(n, np.nan), rep_pad=np.full(n, np.nan),
            rep_source=np.full(n, np.nan), clip_events=np.zeros(n, np.int64), elapsed=np.full(n, np.nan),
            graded=np.zeros(n, bool))
        phase = "propagation"
        current_action = current_hold = None
        state0 = solver.state()
        ck_times.append(0.0)
        ck_states.append(state0)
        ck_slots.append(0)
        ck_energy.append((float(array._physical_energy(state0, np.zeros(4096))),
                          float(0.5 * state0 @ (capacitance @ state0))))
        checkpoint_set = set(checkpoints)
        state_size = len(array.state_ids)
        buffers = np.asarray(spec.buffers) - solver.n_up
        filters = np.asarray(spec.filters) - solver.n_up
        for index, (action_index, hold_index, start, end) in enumerate(splits):
            check()
            held = plan.intervals[hold_index]
            action = actions[action_index]
            if action_index != current_action:
                if args.mode == "ref_fed":
                    polynomial = setting["noise"](held.field_key)
                    if (polynomial.support_start_seconds != action.start
                            or polynomial.support_end_seconds != action.end):
                        raise ValueError("noise polynomial must cover exactly its original action")
                    adapter.bind_polynomial(polynomial)
                current_action = action_index
            if hold_index != current_hold:
                if args.mode == "ref_fed":
                    adapter.bind_field(plan.field_provider(held.field_key))
                current_hold = hold_index
            charge = None
            if start == held.start and held.injected_charge_coulomb:
                charge = np.zeros(state_size)
                for node, value in held.injected_charge_coulomb:
                    charge[node] = value
            first_of_dwell = start == action.start and (action_index == 0
                                                        or actions[action_index - 1].gate_row != action.gate_row)
            grade = {"dwell": first_of_dwell, "action": start == action.start, "hold": start == held.start,
                     "all": True,
                     "none": False}[args.grade]
            clock = time.monotonic()
            pieces = adapter.piece(start, end)
            result = solver.advance(gate_row=held.gate_row, t_start=start, t_end=end, inputs=pieces,
                                    charge=charge, selection=held.daq_selected_sources, grade=grade)
            records["elapsed"][index] = time.monotonic() - clock
            records["t_start"][index], records["t_end"][index] = start, end
            records["action_index"][index], records["hold_index"][index] = action_index, hold_index
            records["gate_row"][index] = held.gate_row
            records["hold_mean"][index] = result.hold_mean
            records["hold_end"][index] = result.hold_end
            records["source_mean"][index] = result.source_mean
            entry = solver._a_blocks[solver.row]
            records["source_end"][index] = entry["block"].phi[entry["source_pos"]] @ solver.a_y
            records["buffer_end"][index] = solver.down_x[buffers]
            records["filter_end"][index] = solver.down_x[filters]
            works = dict(result.works)
            works["field_source_boundary_work_joule"] = 0.0
            records["work"][index] = [works.get(name, 0.0) for name in WORK_FIELDS]
            records["audit"][index] = (works.get("audit_input_joule", np.nan), works.get("audit_heat_joule", np.nan))
            records["sweeps"][index] = result.sweeps
            records["wr_change"][index] = result.wr_change
            records["rep_pad"][index] = result.representation_error["pad"]
            records["rep_source"][index] = result.representation_error["source"]
            records["clip_events"][index] = result.clip_events
            records["graded"][index] = grade
            last_of_action = index + 1 == n or splits[index + 1][0] != action_index
            done = action_index + 1
            if last_of_action and done in checkpoint_set:
                state = solver.state()
                ck_times.append(end)
                ck_states.append(state)
                ck_slots.append(done)
                ck_energy.append((float(array._physical_energy(state, np.zeros(4096))),
                                  float(0.5 * state @ (capacitance @ state))))
            if last_of_action and done % 25 == 0:
                print(json.dumps(dict(event="progress", actions=done, intervals=index + 1, physical_time=end,
                                      elapsed=time.monotonic() - started, stats=solver.stats)), flush=True)
        phase = "complete"
        report.update(status="complete")
    except BaseException as error:
        report["failure"] = dict(phase=phase, exception_type=type(error).__name__, message=str(error)[:2000])
        import traceback
        report["traceback"] = traceback.format_exc()[-4000:]
    finally:
        try:
            import numpy as np
            if records is not None:
                done_mask = np.isfinite(records["t_end"])
                report["completed_intervals"] = int(done_mask.sum())
                report["physical_frontier_seconds"] = float(np.nanmax(records["t_end"])) if done_mask.any() else 0.0
                report["solver_seconds"] = float(np.nansum(records["elapsed"]))
                np.savez(run_dir / "solver_intervals.npz", **records, work_fields=np.array(WORK_FIELDS))
            if ck_states:
                np.savez(run_dir / "solver_checkpoints.npz", times=np.array(ck_times), slots=np.array(ck_slots),
                         states=np.vstack(ck_states), energy=np.array(ck_energy))
            if solver is not None:
                report["solver_stats"] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                          for k, v in solver.stats.items()}
                report["unit_types"] = len(solver.unit_blocks)
                report["unit_modal_residuals"] = [b.modal_residual for b in solver.unit_blocks]
                report["active_block_types"] = len({id(e["block"]) for e in solver._a_blocks.values()})
        except BaseException as error:
            report["save_failure"] = f"{type(error).__name__}: {str(error)[:500]}"
        report["timings"] = timings
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _write_json(run_dir / "solver_report.json", report)
        print(json.dumps(dict(event="terminal", status=report["status"], phase=phase,
                              elapsed=report["elapsed_seconds"])), flush=True)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
