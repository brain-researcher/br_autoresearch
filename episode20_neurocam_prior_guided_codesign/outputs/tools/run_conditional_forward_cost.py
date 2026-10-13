"""One compute-only, full-record conditional EP20 resource-cost attempt.

Only the retained B2/2001 development group is opened. Setup, full preflight
and actual propagation share the same finite wall/RSS budget. No retry,
shortened record, tolerance relaxation, calibration or candidate score exists.
"""

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time


MAX_SECONDS = 6000.0
MAX_RSS_KIB = 13 * 1024**2 // 2
EPISODE_ROOT = Path(__file__).resolve().parents[2]
WORLD_PATH = EPISODE_ROOT / "outputs/runs/development_worlds_job_47085635/panel/development_worlds.h5"


def _compute_context():
    hostname = socket.gethostname()
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if (not job_id.isdecimal() or "-ln" in hostname or not hostname.startswith("sh")
            or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
            or os.environ.get("SLURM_MEM_PER_NODE") != "7168"):
        raise RuntimeError("requires an actual Sherlock compute shell with 1 CPU and 7 GiB")
    for key in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"):
        if os.environ.get(key, "").strip():
            raise RuntimeError("this CPU-only case does not use a GPU allocation")
    for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"):
        if os.environ.get(key, "") not in ("", "0"):
            raise RuntimeError("this CPU-only case does not use GPUs")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"{key} must be 1 before scientific imports")
    return dict(job_id=job_id, hostname=hostname, allocated_cpus=1, allocated_memory_mib=7168)


def _write_json(path, value):
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if len(encoded.encode("utf-8")) > 4 * 1024**2:
        raise RuntimeError("operational JSON exceeds its 4 MiB bound")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--noise-representation", choices=("constant", "cubic"), default="constant")
    parser.add_argument("--reuse-noise", type=Path)
    args = parser.parse_args(argv)
    allocation = _compute_context()  # No numerical import or payload access precedes this.
    if args.noise_representation == "cubic" and args.reuse_noise is None:
        raise RuntimeError("The cubic repair must reuse an explicitly retained original noise realization")
    reused_noise = None if args.reuse_noise is None else args.reuse_noise.resolve()
    if reused_noise is not None and reused_noise != (EPISODE_ROOT /
            "outputs/runs/model_specification/conditional_forward_cost_attempt002/conditional_noise_draw.npz"):
        raise RuntimeError("Only the selected attempt002 original noise archive is in scope")
    run_dir = args.run_dir.resolve()
    if (not run_dir.is_relative_to(EPISODE_ROOT / "outputs/runs") or not run_dir.is_dir()
            or any((run_dir / name).exists() for name in (
                "settings.json", "cost_report.json", "conditional_noise_draw.npz", "conditional_forward.h5"))):
        raise RuntimeError("supply a fresh task-owned run directory under this episode's outputs/runs")
    started = time.monotonic()
    phase = "scientific imports"
    next_rss_check = next_progress = 0.0
    last_rss = 0
    diagnostics = {}
    result = None
    failure = None

    def check():
        nonlocal next_rss_check, next_progress, last_rss
        now = time.monotonic()
        elapsed = now - started
        if elapsed >= MAX_SECONDS:
            raise TimeoutError("6000-second inclusive setup/preflight/propagation limit reached")
        if now >= next_rss_check:
            last_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            next_rss_check = now + 0.5
            if last_rss > MAX_RSS_KIB:
                raise MemoryError("6.5 GiB observed peak RSS bound exceeded")
        if now >= next_progress:
            print(json.dumps(dict(event="cost_progress", phase=phase, elapsed_seconds=elapsed,
                                  peak_rss_kib=last_rss), sort_keys=True), flush=True)
            next_progress = now + 60.0

    def set_phase(value):
        nonlocal phase
        phase = str(value)
        check()

    def interrupt(signum, frame):
        raise TimeoutError(f"finite attempt interrupted by signal {signum}")

    # The process-level timer includes setup/imports, not just wrapper time.
    previous_alarm = signal.signal(signal.SIGALRM, interrupt)
    previous_term = signal.signal(signal.SIGTERM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        check()
        from conditional_acquisition_recipe import (acquisition_recipe_metadata,
            build_conditional_frontend, build_conditional_schedule, conditional_noise_cell)
        from conditional_forward_recipe import (build_conditional_circuit_parameters,
            build_conditional_noise, expected_state_ids, metadata)
        from conditioned_array_solver import ConditionedSolverControls
        from development_simulation import load_saved_development_world
        from raw_world_time import _upper_div
        from streaming_acquisition_readout import StreamingReadoutLimits
        from streaming_array_acquisition import StreamingArrayLimits
        from streaming_development_simulation import run_streaming_development_simulation

        solver = ConditionedSolverControls(relative_tolerance=1e-5, voltage_atol=1e-8,
            charge_atol_coulomb=1e-20, max_refinements=8, max_operator_products=200000,
            linear_relative_tolerance=1e-9, linear_charge_atol_coulomb=1e-22,
            energy_atol_joule=1e-18, max_newton_iterations=12,
            max_linear_iterations=10000, gmres_restart=40, max_workspace_bytes=512 * 1024**2)
        array_limits = StreamingArrayLimits(max_windows=4000, max_input_bindings=24,
            max_physical_events=40000000, max_total_fragments=40000000,
            max_window_fragments=200000, max_split_intervals=40000000,
            max_window_split_intervals=200000, max_stored_window_numeric_bytes=128 * 1024**2,
            max_window_digest_bytes=4000 * 32, max_total_operator_products=1000000000)
        readout_limits = StreamingReadoutLimits(max_conversions=104000000,
            max_returned_samples=104000000, max_observations=17984000,
            max_interval_updates=40000000, max_windows=4000,
            max_live_conversions=52000, max_live_returned_samples=52000,
            max_live_observations=9000, max_stored_plan_numeric_bytes=64 * 1024**2,
            max_accounting_categories=32)
        output_limits = dict(max_payload_bytes=24 * 1024**3, max_records=212736000,
            buffer_records=8192, identifier_bytes=64, max_dependencies_per_observation=3)
        selected = dict(electrical=metadata(), acquisition=dict(acquisition_recipe_metadata()),
            allocation=allocation, full_duration_seconds=16., world_path=str(WORLD_PATH),
            world_selection=dict(condition_id="B2", replicate_seed=2001),
            solver_controls=asdict(solver), array_limits=asdict(array_limits),
            readout_limits=asdict(readout_limits), output_limits=output_limits,
            field_hold_absolute_tolerance_volts=1e-8,
            noise_current_hold_absolute_tolerance_amp=1e-12,
            noise_voltage_hold_absolute_tolerance_volts=1e-8,
            max_hold_intervals=40000000, max_hold_work=100000000,
            max_spectral_evaluations=200000000, max_spectral_term_evaluations=200000000000000,
            max_elements=2000000, world_numeric_bytes_limit=64 * 1024**2,
            world_metadata_bytes_limit=1024**2,
            maximum_inclusive_wall_seconds=MAX_SECONDS, maximum_peak_rss_kib=MAX_RSS_KIB,
            retained_noise_path=str(reused_noise if reused_noise is not None else run_dir / "conditional_noise_draw.npz"),
            noise_representation=args.noise_representation,
            original_named_quadratures_reused=reused_noise is not None,
            polynomial_origin="original_2.5us_mux_slot_midpoint" if args.noise_representation == "cubic" else None,
            result_path=str(run_dir / "conditional_forward.h5"),
            zero_initial_state_is_not_stationary_equilibrium=True,
            input_hold_bounds_are_not_output_accuracy_certificates=True,
            calibrated_reference=False, candidate_scored=False, autonomous_retry=False)
        set_phase("retaining prospective settings")
        _write_json(run_dir / "settings.json", selected)
        set_phase("loading exactly saved B2 seed 2001")
        world = load_saved_development_world(WORLD_PATH, condition_id="B2", replicate_seed=2001,
            max_elements=2000000, max_array_bytes=64 * 1024**2,
            max_metadata_bytes=1024**2, check=check)
        set_phase("constructing full circuit parameters and one acquisition cycle")
        frontend = build_conditional_frontend()
        parameters = build_conditional_circuit_parameters(daq_frontend=frontend, check=check)
        schedule = build_conditional_schedule(frontend, state_ids=expected_state_ids(), check=check)

        def noise_factory(*, array, check):
            # Wrapper invokes this once on its one persistent physical array.
            set_phase("full physical noise binding and retained draw")
            noise_started = time.monotonic()
            # The wrapper check already invokes execution_check, including
            # the throttled RSS/progress guard; do not call it twice per term.
            noise = build_conditional_noise(array=array, check=check,
                retained_draw_path=None if reused_noise is not None else run_dir / "conditional_noise_draw.npz",
                current_hold_tolerance_amp=1e-12, voltage_hold_tolerance_volts=1e-8,
                max_evaluations=200000000, max_term_evaluations=200000000000000,
                reuse_draw_path=reused_noise,
                cubic_cell_for_key=conditional_noise_cell if args.noise_representation == "cubic" else None)
            _, derivative = noise.spectrum.bounds(0., 16.)
            maximum = 0.0
            for index, (value, tolerance) in enumerate(zip(derivative, noise.port_absolute_tolerances)):
                if index % 4096 == 0:
                    check()
                maximum = max(maximum, _upper_div(value, tolerance))
            radius = 16.0 if maximum == 0 else math.nextafter(1. / maximum, 0.)
            diagnostics.update(noise_binding_elapsed_seconds=time.monotonic() - noise_started,
                noise_maximum_normalized_derivative_per_second=maximum,
                conservative_noise_midpoint_radius_limit_seconds=radius,
                ideal_symmetric_noise_hold_length_seconds=2 * radius,
                actual_midpoint_holds_still_require_the_outward_rounded_radius_check=True,
                spectral_accounting_after_binding=noise.spectrum.accounting())
            # The selected constant-hold predicate uses this SAME cached
            # global derivative bound on every interval. Its outward midpoint
            # radius is at least half the interval width. Exact rational
            # arithmetic on the stored float avoids rounding a lower bound up
            # spuriously: any complete 16s plan needs ceil(16*L/2) holds.
            required = Fraction(16) * Fraction.from_float(maximum) / 2
            minimum_holds = -(-required.numerator // required.denominator)
            diagnostics.update(minimum_constant_holds_required=minimum_holds,
                maximum_constant_holds=selected["max_hold_intervals"],
                lower_bound_scope="selected_global_derivative_constant_hold_predicate_only")
            if args.noise_representation == "cubic":
                first_start, first_end, first_origin = conditional_noise_cell(1.25e-6)
                diagnostics.update(selected_noise_representation="cubic_original_physical_slot",
                    first_original_cell_normalized_remainder_bound=noise.hold_bound(
                        first_start, first_end, first_origin),
                    constant_hold_lower_bound_applies_to_selected_representation=False,
                    original_cell_count=6400000,
                    full_preflight_and_solver_cost_not_established=True)
            print(json.dumps(dict(event="noise_bound_diagnostic", **diagnostics), sort_keys=True), flush=True)
            if args.noise_representation == "constant" and minimum_holds > selected["max_hold_intervals"]:
                raise ValueError(f"selected constant-noise hold rule requires at least {minimum_holds} "
                                 f"intervals, above cap {selected['max_hold_intervals']}; "
                                 "do not enumerate or relaunch this infeasible plan")
            return noise

        execution_check = check
        set_phase("streaming full-record wrapper")
        remaining = MAX_SECONDS - (time.monotonic() - started)
        result = run_streaming_development_simulation(world=world, geometry_id="G0", short_axis="x",
            circuit_parameters=parameters, physical_actions=schedule.physical_action_factory,
            readout_window_factory=schedule.readout_window_factory, input_bindings=schedule.input_bindings,
            acquisition_window=schedule.acquisition_window,
            final_publication_horizon=schedule.final_publication_horizon,
            solver_controls=solver, array_limits=array_limits, readout_limits=readout_limits,
            epsilon_input_volts=1e-8, max_hold_intervals=40000000, max_hold_work=100000000,
            max_elements=2000000, max_seconds=remaining,
            settings_reference=str(run_dir / "settings.json"),
            result_path=run_dir / "conditional_forward.h5", output_limits=output_limits,
            noise_factory=noise_factory, execution_check=execution_check, on_phase=set_phase)
    except BaseException as error:
        failure = dict(phase=phase, exception_type=type(error).__name__, message=str(error)[:1000])
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous_term)

    elapsed = time.monotonic() - started
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    status = "complete" if failure is None and result is not None and result.get("status") == "complete" else "failed"
    report = dict(status=status, phase=phase, failure=failure, allocation=allocation,
        elapsed_seconds=elapsed, peak_rss_kib=peak, diagnostics=diagnostics,
        wrapper=result, calibrated_reference=False, candidate_scored=False,
        full_record_completed=status == "complete", payload_scope="saved_development_B2_seed2001_only")
    _write_json(run_dir / "cost_report.json", report)
    print(json.dumps(dict(event="cost_terminal", status=status, elapsed_seconds=elapsed,
        peak_rss_kib=peak, phase=phase, report=str(run_dir / "cost_report.json"),
        wrapper_status=None if result is None else result.get("status")), sort_keys=True), flush=True)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
