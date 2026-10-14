"""Record the UNMODIFIED reference forward chain on a prefix of cycle 0.

Stage 1 equivalence evidence for the fast solver (development test, not a
formal run). Every physical quantity comes from the existing classes:
NeurocamCPEArray.advance_row with the attempt004 Radau/triangular controls,
the saved B2/2001 world, the retained attempt002 noise draw (cubic cells),
the original midpoint holds and the original hold/aperture splits. This
driver only chooses how many 2.5-us actions to run and what to record.

Optional fixtures (each recorded in settings.json):
* --tolerance tight: rtol 1e-8 and 1e-11 V (adjudication reference), with up
  to 12 partition refinements; charge and energy tolerances are unchanged.
* --offsets-json: synthetic per-channel amplifier offsets (the clipping
  fixture). Twelve partition refinements are allowed for that fixture.

Outputs (exclusive, in --run-dir): settings.json, reference_intervals.npz
(per split interval), reference_checkpoints.npz (full 69,848-state vectors)
and reference_report.json (terminal JSON written on every exit path).
"""
import argparse
from dataclasses import asdict, replace
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
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with open(path, "x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--slots", type=int, required=True,
                        help="number of 2.5-us actions from t=0 (2: 5 us, 25: dwell, 1600: frame)")
    parser.add_argument("--deadline-seconds", type=float, required=True)
    parser.add_argument("--tolerance", choices=("standard", "tight"), default="standard")
    parser.add_argument("--offsets-json", type=Path, default=None)
    parser.add_argument("--checkpoint-slots", type=str, default="",
                        help="comma-separated action counts after which the full state is saved")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if (not run_dir.is_dir() or not run_dir.is_relative_to(EPISODE_ROOT / "outputs/runs/fast_forward_stage1/solver")
            or any((run_dir / name).exists() for name in
                   ("settings.json", "reference_report.json", "reference_intervals.npz"))):
        raise SystemExit("supply a fresh existing directory under runs/fast_forward_stage1/solver")
    if not 1 <= args.slots <= 1600:
        raise SystemExit("slots must lie in 1..1600 (cycle 0 only)")
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

    phase = "imports"
    failure = None
    report = dict(host=host, slurm_job_id=job, slots_requested=args.slots, tolerance=args.tolerance,
                  offsets_fixture=None if args.offsets_json is None else str(args.offsets_json),
                  deadline_seconds=deadline, status="failed")
    records = None
    checkpoint_times, checkpoint_states, checkpoint_fields, checkpoint_slots = [], [], [], []
    conversions = None
    try:
        import numpy as np
        from analog_conditioning import AnalogConditioning, SourceConditioning
        import cpe_array as array_module
        from conditional_acquisition_recipe import (build_conditional_frontend, build_conditional_schedule,
                                                    conditional_noise_cell)
        from conditional_forward_recipe import (CONDITIONING, REFERENCE_ID, build_conditional_circuit_parameters,
                                                build_conditional_noise, expected_state_ids, metadata)
        from conditioned_array_solver import ConditionedSolverControls
        from development_simulation import load_saved_development_world
        from raw_input_hold import compile_midpoint_hold
        from raw_world_time import RawWorldTime
        from surface_to_contact import build_contact_factors, registered_pad_geometry

        tight = args.tolerance == "tight"
        offsets = None
        if args.offsets_json is not None:
            offsets = json.loads(args.offsets_json.read_text())
            if (not isinstance(offsets, list) or len(offsets) != 64
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in offsets)):
                raise ValueError("offset fixture must be a JSON list of 64 finite volts")
        controls = ConditionedSolverControls(
            relative_tolerance=1e-8 if tight else 1e-5, voltage_atol=1e-11 if tight else 1e-8,
            charge_atol_coulomb=1e-20, max_refinements=12 if (tight or offsets is not None) else 8,
            max_operator_products=200000, linear_relative_tolerance=1e-9, linear_charge_atol_coulomb=1e-22,
            energy_atol_joule=1e-18, max_newton_iterations=12, max_linear_iterations=10000,
            gmres_restart=40, max_workspace_bytes=512 * 1024**2, integration_method="radau_iia_2",
            radau_preconditioner="contact_block", radau_linear_solver="triangular_contact_schur")
        settings = dict(report, solver_controls=asdict(controls), electrical=metadata(),
                        world_path=str(WORLD_PATH), noise_path=str(NOISE_PATH),
                        field_hold_epsilon_volts=1e-8, noise_current_hold_tolerance_amp=1e-12,
                        noise_voltage_hold_tolerance_volts=1e-8, checkpoint_slots=checkpoints,
                        offsets_volts=offsets, observed_sources=[0],
                        splits="original_midpoint_holds_split_at_every_interior_aperture_boundary",
                        reference_modules_modified=False)
        _write_json(run_dir / "settings.json", settings)

        phase = "world"
        world = load_saved_development_world(WORLD_PATH, condition_id="B2", replicate_seed=2001,
            max_elements=2000000, max_array_bytes=64 * 1024**2, max_metadata_bytes=1024**2, check=check)
        if world.nuisance_arrays:
            raise ValueError("B2 is expected to have no nuisance arrays")
        geometry = registered_pad_geometry("G0", short_axis="x", max_elements=2000000, check=check)
        contacts = build_contact_factors(world, geometry, max_elements=2000000, check=check)
        raw_time = RawWorldTime(world, max_elements=2000000, check=check)

        phase = "circuit"
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
        iterator = schedule.physical_actions_for_cycle(0)
        actions = tuple(next(iterator) for _ in range(args.slots))
        readout = schedule.readout_window(0)

        phase = "noise"
        noise = build_conditional_noise(array=array, check=check, retained_draw_path=None,
            current_hold_tolerance_amp=1e-12, voltage_hold_tolerance_volts=1e-8,
            max_evaluations=200000000, max_term_evaluations=200000000000000,
            reuse_draw_path=NOISE_PATH, cubic_cell_for_key=conditional_noise_cell)

        phase = "holds"
        plan = compile_midpoint_hold(raw_time, contacts, actions, epsilon_input_volts=1e-8,
            max_intervals=40000000, max_work=100000000, check=check, noise_hold_bound=noise.hold_bound)
        boundaries = sorted({point for conversion in readout.conversions
                             for point in (conversion.aperture_start, conversion.aperture_end)})
        boundary_array = np.asarray(boundaries)
        splits = []
        action_of_hold = []
        position = 0
        for hold_index, held in enumerate(plan.intervals):
            while not actions[position].start <= held.start < actions[position].end:
                position += 1
            action_of_hold.append(position)
            lo = np.searchsorted(boundary_array, held.start, side="right")
            hi = np.searchsorted(boundary_array, held.end, side="left")
            points = [held.start, *boundaries[lo:hi], held.end]
            splits.extend((position, hold_index, a, b) for a, b in zip(points, points[1:]))
        n = len(splits)
        records = dict(
            t_start=np.full(n, np.nan), t_end=np.full(n, np.nan), action_index=np.full(n, -1, np.int64),
            hold_index=np.full(n, -1, np.int64), field_key=np.full(n, np.nan),
            gate_row=np.full(n, -1, np.int64), hold_mean=np.full((n, 24), np.nan),
            hold_end=np.full((n, 24), np.nan), source_mean=np.full((n, 64), np.nan),
            source_end=np.full((n, 64), np.nan), buffer_end=np.full((n, 64), np.nan),
            filter_end=np.full((n, 64), np.nan), buffer_mean=np.full((n, 64), np.nan),
            work=np.full((n, len(WORK_FIELDS)), np.nan), energy=np.full((n, 3), np.nan),
            subintervals=np.zeros(n, np.int64), partition_error=np.full(n, np.nan),
            endpoint_error=np.full(n, np.nan), mean_error=np.full(n, np.nan),
            energy_residual=np.full(n, np.nan), numerical_dissipation=np.full(n, np.nan),
            operator_products=np.zeros(n, np.int64), elapsed=np.full(n, np.nan))
        conv_start = np.array([c.aperture_start for c in readout.conversions])
        conv_end = np.array([c.aperture_end for c in readout.conversions])
        conv_adc = np.array([int(c.adc_id.rsplit("_", 1)[1]) for c in readout.conversions], np.int64)
        conv_marker = np.array([c.role == "marker" for c in readout.conversions])
        sample_to_raw = {s.sample_id: s.conversion_ids for s in readout.returned_samples}
        raw_index = {c.conversion_id: i for i, c in enumerate(readout.conversions)}
        pixel_conv = np.array([[raw_index[sample_to_raw[sid][0]] for sid in o.sample_ids]
                               for o in readout.observations if o.role == "pixel"], np.int64)
        pixel_site = np.array([int(o.site_id[4:]) for o in readout.observations if o.role == "pixel"], np.int64)
        conversions = dict(conv_start=conv_start, conv_end=conv_end, conv_adc=conv_adc,
                           conv_marker=conv_marker, pixel_conv=pixel_conv, pixel_site=pixel_site)
        report.update(split_intervals_planned=n, holds_planned=len(plan.intervals),
                      hold_report={k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                                   for k, v in dict(plan.report).items()})

        phase = "propagation"
        polynomial, current_action, field, current_hold = None, None, None, None
        state_size = len(array.state_ids)
        completed = 0
        checkpoint_set = set(checkpoints)
        for index, (action_index, hold_index, start, end) in enumerate(splits):
            check()
            held = plan.intervals[hold_index]
            if action_index != current_action:
                polynomial = noise(held.field_key)
                array.validate_noise_interval(polynomial, daq_selected_sources=held.daq_selected_sources)
                if (polynomial.support_start_seconds != actions[action_index].start
                        or polynomial.support_end_seconds != actions[action_index].end):
                    raise ValueError("noise polynomial must cover exactly its original action")
                current_action = action_index
            if hold_index != current_hold:
                field = plan.field_provider(held.field_key)
                current_hold = hold_index
            charge = np.zeros(state_size)
            if start == held.start:
                for node, value in held.injected_charge_coulomb:
                    charge[node] = value
            clock = time.monotonic()
            result = array.advance_row(gate_row=held.gate_row, duration_seconds=end - start, field_volts=field,
                voltage_reference_id=held.voltage_reference_id, observed_sources=(0,),
                injected_charge_coulomb=charge, controls=controls,
                daq_selected_sources=held.daq_selected_sources, check=check,
                noise_sample=polynomial, interval_start_seconds=start)
            elapsed = time.monotonic() - clock
            kernel = result.kernel_interval
            end_volts = np.asarray(kernel.observation.end_volts)
            mean_volts = np.asarray(kernel.observation.mean_volts)
            records["t_start"][index], records["t_end"][index] = start, end
            records["action_index"][index], records["hold_index"][index] = action_index, hold_index
            records["field_key"][index], records["gate_row"][index] = held.field_key, held.gate_row
            records["hold_mean"][index] = [item.mean_volts for item in result.daq_hold_observations]
            records["hold_end"][index] = [item.end_volts for item in result.daq_hold_observations]
            records["source_mean"][index] = mean_volts[4096:4160]
            records["source_end"][index] = end_volts[4096:4160]
            records["buffer_end"][index] = end_volts[69696:69760]
            records["buffer_mean"][index] = mean_volts[69696:69760]
            records["filter_end"][index] = end_volts[69760:69824]
            records["work"][index] = [getattr(result, name) for name in WORK_FIELDS]
            records["energy"][index] = (result.physical_energy_before_boundary_joule,
                                        result.physical_energy_after_boundary_joule,
                                        result.physical_energy_end_joule)
            records["subintervals"][index] = kernel.subintervals
            records["partition_error"][index] = kernel.maximum_scaled_partition_error
            records["endpoint_error"][index] = kernel.estimated_endpoint_error_volts
            records["mean_error"][index] = kernel.estimated_mean_error_volts
            records["energy_residual"][index] = kernel.energy_balance_residual_joule
            records["numerical_dissipation"][index] = kernel.numerical_dissipation_joule
            records["operator_products"][index] = kernel.operator_products
            records["elapsed"][index] = elapsed
            completed = index + 1
            done_actions = action_index + 1
            last_of_action = index + 1 == n or splits[index + 1][0] != action_index
            if last_of_action and done_actions in checkpoint_set:
                checkpoint_times.append(end)
                checkpoint_states.append(end_volts.copy())
                checkpoint_fields.append(np.asarray(array.previous_field_volts))
                checkpoint_slots.append(done_actions)
            if last_of_action and done_actions % 25 == 0:
                print(json.dumps(dict(event="progress", actions=done_actions, intervals=completed,
                                      physical_time=end, elapsed=time.monotonic() - started)), flush=True)
        phase = "complete"
        report.update(status="complete", completed_intervals=completed)
    except BaseException as error:  # terminal JSON on every exit path
        failure = dict(phase=phase, exception_type=type(error).__name__, message=str(error)[:2000])
        report["failure"] = failure
    finally:
        try:
            import numpy as np
            if records is not None:
                completed_mask = np.isfinite(records["t_end"])
                report["completed_intervals"] = int(completed_mask.sum())
                report["physical_frontier_seconds"] = (float(np.nanmax(records["t_end"]))
                                                       if completed_mask.any() else 0.0)
                report["solver_seconds"] = float(np.nansum(records["elapsed"]))
                report["max_subintervals"] = int(records["subintervals"].max()) if completed_mask.any() else 0
                np.savez(run_dir / "reference_intervals.npz", **records, **(conversions or {}),
                         work_fields=np.array(WORK_FIELDS))
            if checkpoint_states:
                np.savez(run_dir / "reference_checkpoints.npz", times=np.array(checkpoint_times),
                         slots=np.array(checkpoint_slots, np.int64), states=np.vstack(checkpoint_states),
                         fields=np.vstack(checkpoint_fields))
        except BaseException as error:
            report["save_failure"] = f"{type(error).__name__}: {str(error)[:500]}"
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _write_json(run_dir / "reference_report.json", report)
        print(json.dumps(dict(event="terminal", status=report["status"], phase=phase,
                              elapsed=report["elapsed_seconds"])), flush=True)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
